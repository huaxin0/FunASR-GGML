#include "pipeline/offline_batching.hpp"
#include "compute/gpu_context.hpp"
#include "compute/gpu_profile.hpp"

#include <cstdio>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, msg) do { \
    if (condition) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
    } \
} while (0)

#define TEST_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        std::printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
                    name, (int)(expected), (int)(actual), __LINE__); \
    } \
} while (0)

void test_offline_batch_stats_average_active_batch() {
    std::printf("\n--- Test 1: OfflineBatchStats average active batch ---\n");

    funasr::OfflineBatchStats stats;
    TEST_EQ((int)stats.average_active_batch(), 0, "empty average");

    stats.decode_steps = 4;
    stats.active_batch_sum = 10;
    const double avg = stats.average_active_batch();
    TEST_ASSERT(avg > 2.49 && avg < 2.51, "average active batch should be 2.5");
}

void test_frontend_length_bucket_order() {
    std::printf("\n--- Frontend length bucket order ---\n");
    std::vector<funasr::AudioChunk> chunks(8);
    const size_t lengths[] = {10, 80, 20, 70, 30, 60, 40, 50};
    for (size_t i = 0; i < chunks.size(); ++i) {
        chunks[i].infer_start_sample = 0;
        chunks[i].infer_end_sample = lengths[i];
    }

    const auto order = funasr::plan_frontend_request_order(chunks, 2, 4);
    const std::vector<size_t> expected = {1, 3, 2, 0, 5, 7, 6, 4};
    TEST_ASSERT(order == expected,
                "lookahead windows are sorted by descending duration");

    const auto fifo = funasr::plan_frontend_request_order(chunks, 2, 2);
    TEST_ASSERT(fifo == std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7}),
                "bucket window equal to batch preserves FIFO");
}

void test_offline_auto_tune_memory_tiers() {
    std::printf("\n--- Offline auto tune memory tiers ---\n");
    funasr::OfflineAutoTuneInput input;
    input.gpu_total_bytes = 8ULL * 1024 * 1024 * 1024;
    input.gpu_free_bytes = input.gpu_total_bytes;
    input.model_file_bytes = 1200ULL * 1024 * 1024;
    input.kv_bytes_per_block = 14ULL * 1024 * 1024;
    const auto laptop = funasr::tune_offline_runtime(input);
    TEST_ASSERT(laptop.used_memory_probe, "valid probe is used");
    TEST_ASSERT(laptop.kv_num_blocks == 224,
                "free 8 GiB profile selects the measured 224-block pool");
    TEST_ASSERT(laptop.batch_size == 48,
                "free 8 GiB profile selects the measured batch 48");

    input.gpu_total_bytes = 16ULL * 1024 * 1024 * 1024;
    input.gpu_free_bytes = input.gpu_total_bytes;
    const auto workstation = funasr::tune_offline_runtime(input);
    TEST_ASSERT(workstation.kv_num_blocks >= 256,
                "larger GPU receives a larger KV budget");
    TEST_ASSERT(workstation.batch_size == 56,
                "larger GPU enables the batch-56 experiment tier");

    input.gpu_free_bytes = 0;
    const auto fallback = funasr::tune_offline_runtime(input);
    TEST_ASSERT(!fallback.used_memory_probe,
                "missing memory probe falls back safely");
    TEST_ASSERT(fallback.batch_size == 12 && fallback.kv_num_blocks == 192,
                "fallback keeps conservative defaults");
}

void test_paged_kv_pool_capacity_and_release() {
    std::printf("\n--- Test 2: PagedKVBlockPool capacity and release ---\n");

    funasr::PagedKVBlockPool pool(8, 16);
    TEST_EQ(pool.capacity(), 8, "capacity");
    TEST_EQ(pool.free_count(), 8, "initial free count");
    TEST_ASSERT(pool.has_capacity(48), "48 tokens need 3 free blocks");

    funasr::KVHandle a = pool.acquire(1, 48);
    TEST_ASSERT(a.valid(), "first handle valid");
    TEST_ASSERT(a.paged(), "first handle paged");
    TEST_EQ((int)a.block_table.size(), 3, "first blocks");
    TEST_EQ(pool.free_count(), 5, "free count after first acquire");

    funasr::KVHandle b = pool.acquire(2, 96);
    TEST_ASSERT(!b.valid(), "oversized second handle invalid");
    TEST_EQ((int)b.block_table.size(), 0, "oversized second blocks");
    TEST_EQ(pool.free_count(), 5, "failed oversized acquire leaves free count unchanged");

    pool.release(a);
    TEST_EQ(pool.free_count(), 8, "free count after release");
}

void test_fallback_reason_total() {
    std::printf("\n--- Test 3: OfflineDecodeFallbackStats total ---\n");

    funasr::OfflineDecodeFallbackStats stats;
    TEST_ASSERT(stats.total() == 0, "empty fallback stats total is zero");

    stats.single_request = 2;
    stats.token_id_fast_path_unavailable = 3;
    stats.host_embedding_batch_unavailable = 5;
    stats.serial_env_forced = 7;
    stats.invalid_paged_input = 11;

    TEST_ASSERT(stats.total() == 28, "fallback reason total sums all counters");
}

void test_offline_batch_stats_sync_fallback_total() {
    std::printf("\n--- Test 4: OfflineBatchStats fallback total sync ---\n");

    funasr::OfflineBatchStats stats;
    stats.decode_fallback_calls = 99;
    stats.fallback_reasons.single_request = 4;
    stats.fallback_reasons.token_id_fast_path_unavailable = 6;

    stats.sync_fallback_total();

    TEST_ASSERT(stats.decode_fallback_calls == 10,
                "sync_fallback_total replaces aggregate fallback count");
}

void test_offline_batch_stats_scheduler_profile_helpers() {
    std::printf("\n--- Test 5: OfflineBatchStats scheduler profile helpers ---\n");

    funasr::OfflineBatchStats stats;
    TEST_ASSERT(stats.average_prefill_wall_ms() == 0.0,
                "empty average prefill wall is zero");
    TEST_ASSERT(stats.average_decode_dispatch_ms() == 0.0,
                "empty average decode dispatch is zero");

    stats.admit_success = 4;
    stats.prefill_wall_ms = 20.0;
    stats.decode_steps = 5;
    stats.decode_dispatch_wall_ms = 12.5;

    const double avg_prefill = stats.average_prefill_wall_ms();
    const double avg_decode = stats.average_decode_dispatch_ms();
    TEST_ASSERT(avg_prefill > 4.99 && avg_prefill < 5.01,
                "average prefill wall should use admitted requests");
    TEST_ASSERT(avg_decode > 2.49 && avg_decode < 2.51,
                "average decode dispatch should use decode steps");
}

void test_offline_batch_stats_admit_round_helpers() {
    std::printf("\n--- Test 6: OfflineBatchStats admit round helpers ---\n");

    funasr::OfflineBatchStats stats;
    TEST_ASSERT(stats.average_admit_round_size() == 0.0,
                "empty average admit round size is zero");

    stats.admit_rounds = 3;
    stats.admit_round_requests = 12;
    stats.max_admit_round_size = 8;

    const double avg = stats.average_admit_round_size();
    TEST_ASSERT(avg > 3.99 && avg < 4.01,
                "average admit round size uses admitted requests per round");
    TEST_EQ(stats.max_admit_round_size, 8, "max admit round size");
}

void test_paged_decode_profile_cache_probe_rates() {
    std::printf("\n--- Test 7: PagedDecodeProfile cache probe rates ---\n");

    funasr::PagedDecodeProfile profile;
    TEST_ASSERT(profile.shape_cache_probe_hit_rate() == 0.0,
                "empty shape cache probe hit rate is zero");
    TEST_ASSERT(profile.full_graph_cache_probe_hit_rate() == 0.0,
                "empty full graph cache probe hit rate is zero");
    TEST_ASSERT(profile.param_cache_probe_hit_rate() == 0.0,
                "empty param cache probe hit rate is zero");

    profile.graph_cache_probe_calls = 10;
    profile.shape_cache_probe_hits = 7;
    profile.param_cache_probe_hits = 5;
    profile.full_graph_cache_probe_hits = 2;
    profile.graph_cache_hits = 6;
    profile.graph_cache_misses = 4;

    const double shape_rate = profile.shape_cache_probe_hit_rate();
    const double param_rate = profile.param_cache_probe_hit_rate();
    const double full_rate = profile.full_graph_cache_probe_hit_rate();
    const double cache_rate = profile.graph_cache_hit_rate();
    TEST_ASSERT(shape_rate > 69.9 && shape_rate < 70.1,
                "shape cache probe hit rate is percentage");
    TEST_ASSERT(param_rate > 49.9 && param_rate < 50.1,
                "param cache probe hit rate is percentage");
    TEST_ASSERT(full_rate > 19.9 && full_rate < 20.1,
                "full graph cache probe hit rate is percentage");
    TEST_ASSERT(cache_rate > 59.9 && cache_rate < 60.1,
                "actual graph cache hit rate is percentage");
}

void test_paged_decode_bucketed_max_n_kv() {
    std::printf("\n--- Test 8: paged decode bucketed max_n_kv helper ---\n");

    TEST_EQ(funasr::paged_decode_graph_max_n_kv(523, 9, 128, false),
            523, "bucket disabled uses exact max_n_kv");
    TEST_EQ(funasr::paged_decode_graph_max_n_kv(523, 9, 128, true),
            1152, "bucket enabled uses block-table capacity");
    TEST_EQ(funasr::paged_decode_graph_max_n_kv(523, 0, 128, true),
            523, "invalid max_blocks falls back to exact max_n_kv");
    TEST_EQ(funasr::paged_decode_graph_max_n_kv(523, 9, 0, true),
            523, "invalid block_size falls back to exact max_n_kv");
}

void test_selected_audio_duration_uses_selected_chunks() {
    std::printf("\n--- Test 9: selected audio duration uses selected chunks ---\n");

    std::vector<funasr::AudioChunk> chunks(2);
    chunks[0].start_sample = 0;
    chunks[0].end_sample = 16000;
    chunks[1].start_sample = 16000;
    chunks[1].end_sample = 32000;

    const float seconds = funasr::selected_audio_duration_seconds(chunks, 16000);
    TEST_ASSERT(seconds > 1.999f && seconds < 2.001f,
                "selected duration should cover only selected chunks");
}

void test_paged_kv_pool_reference_counts_and_duplicate_release() {
    std::printf("\n--- Test 10: paged KV reference counts ---\n");

    funasr::PagedKVBlockPool pool(4, 4);
    funasr::KVHandle handle = pool.acquire(1, 4);
    const int block = handle.block_table.front();

    TEST_EQ(pool.ref_count(block), 1, "acquired block ref count");
    TEST_ASSERT(pool.retain_block(block), "retain acquired block");
    TEST_EQ(pool.ref_count(block), 2, "retained block ref count");

    pool.release(handle);
    TEST_EQ(pool.ref_count(block), 1, "first owner release keeps block live");
    TEST_EQ(pool.free_count(), 3, "retained block not returned early");

    pool.release(handle);
    TEST_EQ(pool.ref_count(block), 0, "last owner release clears ref count");
    TEST_EQ(pool.free_count(), 4, "last owner returns block once");

    const int errors_before = pool.ownership_errors();
    pool.release(handle);
    TEST_EQ(pool.free_count(), 4, "duplicate release does not duplicate free entry");
    TEST_EQ(pool.ownership_errors(), errors_before + 1,
            "duplicate release records ownership error");
}

void test_paged_kv_pool_append_is_atomic() {
    std::printf("\n--- Test 11: paged KV append is atomic ---\n");

    funasr::PagedKVBlockPool pool(2, 4);
    funasr::KVHandle handle = pool.acquire(7, 4);

    TEST_ASSERT(pool.append_block(handle), "append available block");
    TEST_EQ((int)handle.block_table.size(), 2, "append grows block table");
    TEST_EQ(pool.free_count(), 0, "append consumes one block");

    TEST_ASSERT(!pool.append_block(handle), "append fails when pool exhausted");
    TEST_EQ((int)handle.block_table.size(), 2,
            "failed append leaves block table unchanged");
}

void test_paged_kv_pool_cow_replaces_one_reference() {
    std::printf("\n--- Test 12: paged KV COW ownership transfer ---\n");

    funasr::PagedKVBlockPool pool(3, 4);
    funasr::KVHandle cache = pool.acquire(100, 3);
    funasr::KVHandle request = cache;
    TEST_ASSERT(pool.retain(request), "request retains cached prefix block");

    const int shared = cache.block_table.front();
    const int replacement = pool.allocate_block();
    TEST_ASSERT(replacement >= 0, "allocate private COW replacement");
    TEST_ASSERT(pool.replace_block_after_cow(request, 0, replacement),
                "replace request partial block");

    TEST_EQ(request.block_table.front(), replacement,
            "request points at private replacement");
    TEST_EQ(pool.ref_count(shared), 1, "cache remains sole shared-block owner");
    TEST_EQ(pool.ref_count(replacement), 1, "request owns replacement once");

    pool.release(request);
    pool.release(cache);
    TEST_EQ(pool.free_count(), pool.capacity(), "COW states release all blocks");
}

void test_dynamic_paged_block_decisions() {
    std::printf("\n--- Test 13: dynamic paged block decisions ---\n");

    TEST_EQ(funasr::paged_blocks_for_tokens(0, 128), 0,
            "zero tokens need no blocks");
    TEST_EQ(funasr::paged_blocks_for_tokens(1, 128), 1,
            "first token needs one block");
    TEST_EQ(funasr::paged_blocks_for_tokens(128, 128), 1,
            "full block remains one block");
    TEST_EQ(funasr::paged_blocks_for_tokens(129, 128), 2,
            "boundary crossing needs another block");
    TEST_EQ(funasr::paged_prefix_partial_rows(17, 128), 17,
            "partial prefix reports valid rows");
    TEST_EQ(funasr::paged_prefix_partial_rows(128, 128), 0,
            "aligned prefix needs no COW rows");
    TEST_ASSERT(!funasr::paged_decode_requires_append(127, 1, 128),
                "last position in existing block needs no append");
    TEST_ASSERT(funasr::paged_decode_requires_append(128, 1, 128),
                "first position beyond block needs append");

    const int actual_prefill = 539;
    TEST_EQ(funasr::paged_blocks_for_tokens(actual_prefill, 128), 5,
            "dynamic allocation follows actual prefill only");
}

void test_paged_kv_capacity_resolution() {
    std::printf("\n--- Test 14: paged KV capacity resolution ---\n");

    funasr::OfflineBatchConfig cfg;
    cfg.batch_size = 12;
    cfg.ctx_size = 4096;
    cfg.kv_block_size = 128;
    cfg.kv_num_blocks = 160;

    TEST_EQ(funasr::resolve_paged_kv_num_blocks(cfg), 160,
            "explicit paged block count");
    TEST_EQ(funasr::resolve_paged_kv_physical_rows(cfg), 20480,
            "explicit paged physical rows");

    cfg.kv_num_blocks = 0;
    cfg.batch_size = 3;
    cfg.ctx_size = 4100;
    TEST_EQ(funasr::resolve_paged_kv_num_blocks(cfg), 99,
            "derived blocks round each context upward");
    TEST_EQ(funasr::resolve_paged_kv_physical_rows(cfg), 12672,
            "derived rows match resolved blocks");
}

void test_gpu_kv_physical_row_resolution() {
    std::printf("\n--- Test 15: GPU KV physical row resolution ---\n");

    TEST_EQ(funasr::GPUContext::resolve_physical_kv_rows(4096, 12, 0),
            49152, "default rows preserve slot allocation");
    TEST_EQ(funasr::GPUContext::resolve_physical_kv_rows(4096, 32, 20480),
            20480, "explicit rows decouple scheduler concurrency");
}

void test_unified_initial_block_admission() {
    std::printf("\n--- Unified initial block admission ---\n");

    TEST_EQ(funasr::paged_initial_admission_blocks(522, 18, 128), 1,
            "partial shared prefix reserves its COW block");
    TEST_EQ(funasr::paged_initial_admission_blocks(522, 128, 128), 1,
            "aligned shared prefix reserves the first append block");
    TEST_EQ(funasr::paged_initial_admission_blocks(522, 0, 128), 1,
            "uncached prompt reserves its first block");
    TEST_EQ(funasr::paged_initial_admission_blocks(128, 128, 128), 0,
            "fully cached prompt needs no prompt block");
}

void test_paged_prefill_admission_hints_skip_known_insufficient_capacity() {
    std::printf("\n--- Test 16: paged prefill admission capacity hints ---\n");

    funasr::PagedPrefillAdmissionHints hints;
    constexpr size_t full_window_samples = 480000;

    TEST_ASSERT(hints.can_prepare(full_window_samples, 0),
                "unknown chunk length must be prepared once to learn its demand");

    hints.observe(full_window_samples, 5);
    TEST_EQ(hints.required_blocks(full_window_samples), 5,
            "observed block demand is retained");
    TEST_ASSERT(!hints.can_prepare(full_window_samples, 4),
                "known insufficient capacity skips repeated frontend work");
    TEST_ASSERT(hints.can_prepare(full_window_samples, 5),
                "exact known capacity permits frontend work");

    hints.observe(full_window_samples, 3);
    TEST_EQ(hints.required_blocks(full_window_samples), 5,
            "smaller observations do not weaken a conservative hint");
    TEST_EQ(hints.required_blocks(full_window_samples / 2), 0,
            "different chunk lengths learn independent hints");
}

void test_plan_prefill_chunks_covers_prompt_without_gaps() {
    std::printf("\n--- Test 17: sequential prefill chunk planning ---\n");

    const std::vector<std::pair<int, int>> chunks =
        funasr::plan_prefill_chunks(0, 522, 128);
    TEST_EQ(static_cast<int>(chunks.size()), 5,
            "522 tokens split into five chunks");
    TEST_EQ(chunks[0].first, 0, "first chunk offset");
    TEST_EQ(chunks[0].second, 128, "first chunk count");
    TEST_EQ(chunks[1].first, 128, "second chunk offset");
    TEST_EQ(chunks[1].second, 128, "second chunk count");
    TEST_EQ(chunks[2].first, 256, "third chunk offset");
    TEST_EQ(chunks[2].second, 128, "third chunk count");
    TEST_EQ(chunks[3].first, 384, "fourth chunk offset");
    TEST_EQ(chunks[3].second, 128, "fourth chunk count");
    TEST_EQ(chunks[4].first, 512, "final chunk offset");
    TEST_EQ(chunks[4].second, 10, "final chunk count");

    const std::vector<std::pair<int, int>> cached =
        funasr::plan_prefill_chunks(11, 522, 256);
    TEST_EQ(static_cast<int>(cached.size()), 2,
            "cached prefix leaves two chunks");
    TEST_EQ(cached[0].first, 11, "cached first offset");
    TEST_EQ(cached[0].second, 256, "cached first count");
    TEST_EQ(cached[1].first, 267, "cached final offset");
    TEST_EQ(cached[1].second, 255, "cached final count");

    TEST_ASSERT(funasr::plan_prefill_chunks(-1, 522, 128).empty(),
                "negative first token is invalid");
    TEST_ASSERT(funasr::plan_prefill_chunks(0, 0, 128).empty(),
                "empty prompt is invalid");
    TEST_ASSERT(funasr::plan_prefill_chunks(0, 522, 0).empty(),
                "zero chunk size is invalid");
    TEST_ASSERT(funasr::plan_prefill_chunks(522, 522, 128).empty(),
                "fully cached prompt needs no chunks");
    TEST_ASSERT(funasr::plan_prefill_chunks(523, 522, 128).empty(),
                "first token beyond prompt is invalid");
}

void test_prefill_chunking_is_disabled_by_default() {
    std::printf("\n--- Test 18: prefill chunking default ---\n");

    const funasr::OfflineBatchConfig config;
    TEST_EQ(config.prefill_chunk_tokens, 0,
            "sequential chunked prefill is opt-in");
}

void test_offline_result_token_ids_copy_exact_request_tokens() {
    std::printf("\n--- Test 19: exact result token audit ---\n");

    funasr::OfflineRequest request;
    request.tokens = {151643, 42, 7, 151645};
    funasr::OfflineChunkResult result;

    funasr::copy_offline_result_tokens(request, result);

    TEST_EQ(static_cast<int>(result.token_ids.size()), 4,
            "all generated token ids are copied");
    TEST_EQ(result.token_ids[0], 151643, "first token id is exact");
    TEST_EQ(result.token_ids[1], 42, "second token id is exact");
    TEST_EQ(result.token_ids[2], 7, "third token id is exact");
    TEST_EQ(result.token_ids[3], 151645, "final token id is exact");
}

int main() {
    std::printf("========================================\n");
    std::printf("Offline Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_offline_batch_stats_average_active_batch();
    test_frontend_length_bucket_order();
    test_offline_auto_tune_memory_tiers();
    test_paged_kv_pool_capacity_and_release();
    test_fallback_reason_total();
    test_offline_batch_stats_sync_fallback_total();
    test_offline_batch_stats_scheduler_profile_helpers();
    test_offline_batch_stats_admit_round_helpers();
    test_paged_decode_profile_cache_probe_rates();
    test_paged_decode_bucketed_max_n_kv();
    test_selected_audio_duration_uses_selected_chunks();
    test_paged_kv_pool_reference_counts_and_duplicate_release();
    test_paged_kv_pool_append_is_atomic();
    test_paged_kv_pool_cow_replaces_one_reference();
    test_dynamic_paged_block_decisions();
    test_paged_kv_capacity_resolution();
    test_gpu_kv_physical_row_resolution();
    test_unified_initial_block_admission();
    test_paged_prefill_admission_hints_skip_known_insufficient_capacity();
    test_plan_prefill_chunks_covers_prompt_without_gaps();
    test_prefill_chunking_is_disabled_by_default();
    test_offline_result_token_ids_copy_exact_request_tokens();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
