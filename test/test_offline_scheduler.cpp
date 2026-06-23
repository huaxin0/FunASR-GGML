#include "pipeline/offline_batching.hpp"
#include "compute/gpu_profile.hpp"

#include <cstdio>

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

int main() {
    std::printf("========================================\n");
    std::printf("Offline Scheduler Unit Tests\n");
    std::printf("========================================\n");

    test_offline_batch_stats_average_active_batch();
    test_paged_kv_pool_capacity_and_release();
    test_fallback_reason_total();
    test_offline_batch_stats_sync_fallback_total();
    test_offline_batch_stats_scheduler_profile_helpers();
    test_offline_batch_stats_admit_round_helpers();
    test_paged_decode_profile_cache_probe_rates();
    test_paged_decode_bucketed_max_n_kv();

    std::printf("\n========================================\n");
    std::printf("Tests passed: %d\n", tests_passed);
    std::printf("Tests failed: %d\n", tests_failed);
    std::printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
