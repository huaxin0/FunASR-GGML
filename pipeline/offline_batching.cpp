#include "pipeline/offline_batching.hpp"
#include "pipeline/recognizer.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <limits>
#include <numeric>

namespace funasr {

namespace {

class PromptEmbeddingLease {
public:
    PromptEmbeddingLease(
        Recognizer& recognizer,
        const GPUEmbeddingHandle& handle)
        : recognizer_(&recognizer), handle_(handle) {}

    PromptEmbeddingLease(const PromptEmbeddingLease&) = delete;
    PromptEmbeddingLease& operator=(const PromptEmbeddingLease&) = delete;

    ~PromptEmbeddingLease() {
        if (recognizer_ && handle_.valid()) {
            recognizer_->release_prompt_gpu(handle_);
        }
    }

private:
    Recognizer* recognizer_ = nullptr;
    GPUEmbeddingHandle handle_;
};

}  // namespace

int paged_blocks_for_tokens(int tokens, int block_size) {
    if (tokens <= 0 || block_size <= 0) {
        return 0;
    }
    return (tokens + block_size - 1) / block_size;
}

int paged_prefix_partial_rows(int prefix_tokens, int block_size) {
    if (prefix_tokens <= 0 || block_size <= 0) {
        return 0;
    }
    return prefix_tokens % block_size;
}

bool paged_decode_requires_append(
    int n_past,
    size_t allocated_blocks,
    int block_size) {
    return paged_blocks_for_tokens(n_past + 1, block_size) >
           static_cast<int>(allocated_blocks);
}

std::vector<std::pair<int, int>> plan_prefill_chunks(
    int first_uncached_token,
    int prompt_tokens,
    int max_chunk_tokens) {
    std::vector<std::pair<int, int>> chunks;
    if (first_uncached_token < 0 || prompt_tokens <= 0 ||
        max_chunk_tokens <= 0 || first_uncached_token >= prompt_tokens) {
        return chunks;
    }

    int64_t offset = first_uncached_token;
    const int64_t prompt_end = prompt_tokens;
    const int64_t chunk_limit = max_chunk_tokens;
    while (offset < prompt_end) {
        const int64_t count = std::min(chunk_limit, prompt_end - offset);
        chunks.emplace_back(static_cast<int>(offset), static_cast<int>(count));
        offset += count;
    }
    return chunks;
}

void copy_offline_result_tokens(
    const OfflineRequest& request,
    OfflineChunkResult& result) {
    result.token_ids = request.tokens;
}

int resolve_paged_kv_num_blocks(const OfflineBatchConfig& config) {
    if (config.kv_num_blocks > 0) {
        return config.kv_num_blocks;
    }
    const int batch_size = std::max(1, config.batch_size);
    const int block_size = std::max(1, config.kv_block_size);
    const int ctx_size = std::max(1, config.ctx_size);
    return batch_size * ((ctx_size + block_size - 1) / block_size);
}

int resolve_paged_kv_physical_rows(const OfflineBatchConfig& config) {
    return resolve_paged_kv_num_blocks(config) *
           std::max(1, config.kv_block_size);
}

int paged_initial_admission_blocks(int prompt_tokens,
                                   int cached_prefix_tokens,
                                   int block_size) {
    if (prompt_tokens <= cached_prefix_tokens || block_size <= 0) {
        return 0;
    }
    return 1;
}

std::vector<size_t> plan_frontend_request_order(
    const std::vector<AudioChunk>& chunks,
    int frontend_batch_size,
    int bucket_window) {
    std::vector<size_t> order(chunks.size());
    std::iota(order.begin(), order.end(), 0);
    if (frontend_batch_size <= 1 || bucket_window <= frontend_batch_size ||
        chunks.size() <= static_cast<size_t>(frontend_batch_size)) {
        return order;
    }

    const size_t window = static_cast<size_t>(std::max(
        frontend_batch_size, bucket_window));
    for (size_t begin = 0; begin < order.size(); begin += window) {
        const size_t end = std::min(order.size(), begin + window);
        std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(begin),
                         order.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](size_t lhs, size_t rhs) {
            const AudioChunk& a = chunks[lhs];
            const AudioChunk& b = chunks[rhs];
            const size_t a_size = a.infer_end_sample > a.infer_start_sample
                ? a.infer_end_sample - a.infer_start_sample
                : 0;
            const size_t b_size = b.infer_end_sample > b.infer_start_sample
                ? b.infer_end_sample - b.infer_start_sample
                : 0;
            return a_size > b_size;
        });
    }
    return order;
}

OfflineAutoTuneResult tune_offline_runtime(
    const OfflineAutoTuneInput& input) {
    OfflineAutoTuneResult result;
    if (input.gpu_free_bytes == 0 || input.gpu_total_bytes == 0 ||
        input.kv_bytes_per_block == 0 || input.block_size <= 0) {
        return result;
    }

    constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    const uint64_t model_reserve = input.model_file_bytes > 0
        ? input.model_file_bytes + input.model_file_bytes / 8
        : 2ULL * kGiB;
    const uint64_t runtime_reserve = 768ULL * 1024ULL * 1024ULL;
    const uint64_t reserve = model_reserve + runtime_reserve;
    const uint64_t free_budget = input.gpu_free_bytes > reserve
        ? input.gpu_free_bytes - reserve
        : 0;
    // The 40% ceiling is measured against the 8 GB laptop profile. A
    // 224-block Q8 KV pool sustains batch 48 while leaving headroom for
    // weights, frontend workspaces, and CUDA graph allocations. Batch 56/256
    // remains available as an explicit throughput experiment, but did not
    // improve the measured median and was less deterministic on this device.
    const uint64_t total_budget = input.gpu_total_bytes * 40 / 100;
    const uint64_t kv_budget = std::min(free_budget, total_budget);
    uint64_t blocks = kv_budget / input.kv_bytes_per_block;
    blocks = std::min<uint64_t>(384, blocks);
    blocks = (blocks / 32) * 32;
    if (blocks < 96) {
        blocks = 96;
    }

    result.kv_num_blocks = static_cast<int>(blocks);
    if (result.kv_num_blocks >= 256) {
        result.batch_size = 56;
        result.max_frontend_requests = 8;
    } else if (result.kv_num_blocks >= 224) {
        result.batch_size = 48;
        result.max_frontend_requests = 6;
    } else if (result.kv_num_blocks >= 192) {
        result.batch_size = 40;
        result.max_frontend_requests = 4;
    } else if (result.kv_num_blocks >= 160) {
        result.batch_size = 32;
        result.max_frontend_requests = 4;
    } else if (result.kv_num_blocks >= 128) {
        result.batch_size = 24;
        result.max_frontend_requests = 4;
    } else {
        result.batch_size = 16;
        result.max_frontend_requests = 2;
    }

    // Long windows keep more prompt/decode KV resident per request. Keep one
    // tier of headroom instead of allowing an optimistic memory-only batch.
    if (input.chunk_seconds >= 45 && result.batch_size > 40) {
        result.batch_size = 40;
        result.max_frontend_requests = 4;
    }
    result.used_memory_probe = true;
    return result;
}

bool PagedPrefillAdmissionHints::can_prepare(
    size_t sample_count,
    int free_blocks) const {
    const int hint = required_blocks(sample_count);
    return hint == 0 || free_blocks >= hint;
}

void PagedPrefillAdmissionHints::observe(
    size_t sample_count,
    int required_blocks) {
    if (required_blocks <= 0) {
        return;
    }
    int& hint = required_blocks_by_sample_count_[sample_count];
    hint = std::max(hint, required_blocks);
}

int PagedPrefillAdmissionHints::required_blocks(size_t sample_count) const {
    const auto it = required_blocks_by_sample_count_.find(sample_count);
    return it == required_blocks_by_sample_count_.end() ? 0 : it->second;
}

void ContinuousKVSlotPool::reset(int num_slots, int ctx_size) {
    capacity_ = std::max(0, num_slots);
    ctx_size_ = std::max(0, ctx_size);
    in_use_.assign(static_cast<size_t>(capacity_), false);

    std::queue<int> empty;
    std::swap(free_slots_, empty);
    for (int i = 0; i < capacity_; i++) {
        free_slots_.push(i);
    }
}

bool ContinuousKVSlotPool::has_free_slot() const {
    return !free_slots_.empty();
}

bool ContinuousKVSlotPool::has_capacity(int max_tokens) const {
    (void)max_tokens;
    return has_free_slot();
}

KVHandle ContinuousKVSlotPool::acquire(int request_id) {
    return acquire(request_id, ctx_size_);
}

KVHandle ContinuousKVSlotPool::acquire(int request_id, int max_tokens) {
    (void)max_tokens;
    if (free_slots_.empty()) {
        return {};
    }

    int slot = free_slots_.front();
    free_slots_.pop();
    in_use_[static_cast<size_t>(slot)] = true;
    return KVHandle{request_id, slot, ctx_size_};
}

void ContinuousKVSlotPool::release(const KVHandle& handle) {
    if (!handle.valid() || handle.slot_id >= capacity_) {
        return;
    }

    size_t slot = static_cast<size_t>(handle.slot_id);
    if (!in_use_[slot]) {
        return;
    }

    in_use_[slot] = false;
    free_slots_.push(handle.slot_id);
}

void PagedKVBlockPool::reset(int num_blocks, int block_size) {
    capacity_ = std::max(0, num_blocks);
    block_size_ = std::max(1, block_size);
    ownership_errors_ = 0;
    ref_counts_.assign(static_cast<size_t>(capacity_), 0);
    in_free_queue_.assign(static_cast<size_t>(capacity_), true);

    std::queue<int> empty;
    std::swap(free_blocks_, empty);
    for (int i = 0; i < capacity_; i++) {
        free_blocks_.push(i);
    }
}

int PagedKVBlockPool::blocks_needed(int max_tokens) const {
    if (max_tokens <= 0) {
        return 0;
    }
    return (max_tokens + block_size_ - 1) / block_size_;
}

bool PagedKVBlockPool::has_capacity(int max_tokens) const {
    return static_cast<int>(free_blocks_.size()) >= blocks_needed(max_tokens);
}

KVHandle PagedKVBlockPool::acquire(int request_id, int max_tokens) {
    const int need = blocks_needed(max_tokens);
    if (need <= 0 || static_cast<int>(free_blocks_.size()) < need) {
        return {};
    }

    KVHandle handle;
    handle.request_id = request_id;
    handle.slot_id = -1;
    handle.ctx_size = need * block_size_;
    handle.block_size = block_size_;
    handle.block_table.reserve(static_cast<size_t>(need));

    for (int i = 0; i < need; i++) {
        int block = allocate_block();
        if (block < 0) {
            release(handle);
            return {};
        }
        handle.block_table.push_back(block);
    }
    return handle;
}

void PagedKVBlockPool::release(const KVHandle& handle) {
    for (int block : handle.block_table) {
        release_block(block);
    }
}

int PagedKVBlockPool::allocate_block() {
    if (free_blocks_.empty()) {
        return -1;
    }

    const int block = free_blocks_.front();
    free_blocks_.pop();
    if (block < 0 || block >= capacity_) {
        ownership_errors_++;
        return -1;
    }

    const size_t index = static_cast<size_t>(block);
    if (!in_free_queue_[index] || ref_counts_[index] != 0) {
        ownership_errors_++;
        return -1;
    }

    in_free_queue_[index] = false;
    ref_counts_[index] = 1;
    return block;
}

bool PagedKVBlockPool::append_block(KVHandle& handle) {
    const int block = allocate_block();
    if (block < 0) {
        return false;
    }
    handle.block_table.push_back(block);
    handle.block_size = block_size_;
    handle.ctx_size = static_cast<int>(handle.block_table.size()) * block_size_;
    return true;
}

bool PagedKVBlockPool::retain_block(int block_id) {
    if (block_id < 0 || block_id >= capacity_) {
        ownership_errors_++;
        return false;
    }

    const size_t index = static_cast<size_t>(block_id);
    if (in_free_queue_[index] || ref_counts_[index] <= 0) {
        ownership_errors_++;
        return false;
    }
    ref_counts_[index]++;
    return true;
}

bool PagedKVBlockPool::retain(const KVHandle& handle) {
    std::vector<int> retained;
    retained.reserve(handle.block_table.size());
    for (int block : handle.block_table) {
        if (!retain_block(block)) {
            for (int rollback : retained) {
                release_block(rollback);
            }
            return false;
        }
        retained.push_back(block);
    }
    return true;
}

bool PagedKVBlockPool::release_block(int block_id) {
    if (block_id < 0 || block_id >= capacity_) {
        ownership_errors_++;
        return false;
    }

    const size_t index = static_cast<size_t>(block_id);
    if (in_free_queue_[index] || ref_counts_[index] <= 0) {
        ownership_errors_++;
        return false;
    }

    ref_counts_[index]--;
    if (ref_counts_[index] == 0) {
        in_free_queue_[index] = true;
        free_blocks_.push(block_id);
    }
    return true;
}

bool PagedKVBlockPool::replace_block_after_cow(
    KVHandle& handle,
    size_t logical_block,
    int replacement_block) {
    if (logical_block >= handle.block_table.size() ||
        replacement_block < 0 || replacement_block >= capacity_) {
        ownership_errors_++;
        return false;
    }

    const int old_block = handle.block_table[logical_block];
    const size_t replacement_index = static_cast<size_t>(replacement_block);
    if (old_block == replacement_block ||
        in_free_queue_[replacement_index] ||
        ref_counts_[replacement_index] != 1) {
        ownership_errors_++;
        return false;
    }
    if (!release_block(old_block)) {
        return false;
    }
    handle.block_table[logical_block] = replacement_block;
    return true;
}

int PagedKVBlockPool::ref_count(int block_id) const {
    if (block_id < 0 || block_id >= capacity_) {
        return 0;
    }
    return ref_counts_[static_cast<size_t>(block_id)];
}

OfflineBatchTranscriber::OfflineBatchTranscriber(Recognizer& recognizer)
    : recognizer_(recognizer)
{}

void OfflineBatchTranscriber::reset_stats(
    size_t total_chunks,
    const OfflineBatchConfig& config) {
    last_stats_ = OfflineBatchStats{};
    last_stats_.total_chunks = static_cast<int>(total_chunks);
    last_stats_.batch_size = std::max(1, config.batch_size);
    last_stats_.use_gpu = config.use_gpu;
    last_stats_.use_paged_kv = config.use_paged_kv;
    last_stats_.kv_block_size = config.kv_block_size;
}

void OfflineBatchTranscriber::update_peak_blocks(const KVPool& kv_pool) {
    const int used = kv_pool.capacity() - kv_pool.free_count();
    last_stats_.peak_blocks_in_use = std::max(last_stats_.peak_blocks_in_use, used);
}

namespace {

int argmax_logits(const std::vector<float>& logits) {
    float max_val = -1e30f;
    int max_idx = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

bool tail_repeats(const std::vector<int>& tokens, int ngram, int repeats) {
    const int need = ngram * repeats;
    if (ngram <= 0 || repeats <= 1 || static_cast<int>(tokens.size()) < need) {
        return false;
    }
    const int start = static_cast<int>(tokens.size()) - need;
    for (int r = 1; r < repeats; r++) {
        for (int i = 0; i < ngram; i++) {
            if (tokens[static_cast<size_t>(start + i)] !=
                tokens[static_cast<size_t>(start + r * ngram + i)]) {
                return false;
            }
        }
    }
    return true;
}

bool has_repetition_loop_local(const std::vector<int>& tokens) {
    for (int ngram = 4; ngram <= 16; ngram++) {
        if (tail_repeats(tokens, ngram, 4)) {
            return true;
        }
    }
    return false;
}

int estimate_kv_budget_tokens(const AudioChunk& chunk,
                              const std::vector<float>& samples,
                              int sample_rate,
                              const OfflineBatchConfig& config) {
    if (!config.use_paged_kv) {
        return config.ctx_size;
    }

    const size_t begin = std::min(chunk.infer_start_sample, samples.size());
    const size_t end = std::min(chunk.infer_end_sample, samples.size());
    const float seconds = sample_rate > 0 && end > begin
        ? static_cast<float>(end - begin) / static_cast<float>(sample_rate)
        : 0.0f;

    // FunASR adaptor output is about 16-17 embeddings/sec for 16 kHz audio in
    // the common 30s window case. Use a conservative upper estimate so paged KV
    // does not allocate a full ctx slot while still leaving headroom for prompt
    // tokens, frontend boundary effects, and decode.
    const int estimated_prefill = static_cast<int>(seconds * 25.0f) + 128;
    const int budget = estimated_prefill + config.max_tokens;
    return std::max(1, std::min(config.ctx_size, budget));
}

} // namespace

std::vector<OfflineChunkResult> OfflineBatchTranscriber::transcribe(
    const std::vector<float>& samples,
    int sample_rate,
    const std::vector<AudioChunk>& chunks,
    const OfflineBatchConfig& config,
    OfflineProgressCallback progress) {
    reset_stats(chunks.size(), config);

    std::vector<OfflineChunkResult> results;
    results.reserve(chunks.size());

    if (config.use_gpu) {
        if (config.use_unified_scheduler) {
            return transcribe_unified_gpu(
                samples, sample_rate, chunks, config, progress);
        }
        return transcribe_continuous_gpu(samples, sample_rate, chunks, config, progress);
    }

    std::unique_ptr<KVPool> kv_pool;
    if (config.use_paged_kv) {
        const int num_blocks = resolve_paged_kv_num_blocks(config);
        kv_pool = std::make_unique<PagedKVBlockPool>(num_blocks, config.kv_block_size);
        last_stats_.kv_block_capacity = num_blocks;
        printf("[OfflineBatch] Paged KV scheduler pool: blocks=%d block_size=%d\n",
               num_blocks, config.kv_block_size);
        if (config.use_gpu) {
            printf("[OfflineBatch] WARNING: paged attention kernel is not wired yet; "
                   "GPU execution will use serial compatibility until block-table attention lands.\n");
        }
    } else {
        kv_pool = std::make_unique<ContinuousKVSlotPool>(
            std::max(1, config.batch_size), config.ctx_size);
    }

    int completed = 0;
    size_t next_request = 0;

    while (next_request < chunks.size()) {
        std::vector<OfflineRequest> active;
        active.reserve(static_cast<size_t>(std::max(1, config.batch_size)));

        while (next_request < chunks.size() &&
               static_cast<int>(active.size()) < std::max(1, config.batch_size)) {
            int request_id = static_cast<int>(next_request);
            const int kv_budget = estimate_kv_budget_tokens(
                chunks[next_request], samples, sample_rate, config);
            if (!kv_pool->has_capacity(kv_budget)) {
                break;
            }
            KVHandle handle = kv_pool->acquire(request_id, kv_budget);
            if (!handle.valid()) {
                break;
            }
            if (handle.paged()) {
                update_peak_blocks(*kv_pool);
            }

            OfflineRequest request;
            request.id = request_id;
            request.phase = OfflineRequestPhase::WaitingPrefill;
            request.kv = handle;
            request.max_tokens = config.max_tokens;
            request.start_sec = chunks[next_request].start_sec;
            request.end_sec = chunks[next_request].end_sec;
            active.push_back(request);
            next_request++;
        }

        // Multiple active request states are admitted together. Compute is
        // still serial, but each request now goes through prepare -> LLM run,
        // which gives later batched prefill/decode a clean insertion point.
        bool can_use_gpu_batch = config.use_gpu && active.size() > 1 &&
                                 !active.front().kv.paged();

        if (can_use_gpu_batch) {
            InferenceConfig inference;
            inference.max_new_tokens = config.max_tokens;
            inference.n_threads = config.n_threads;
            inference.kv_cache_size = config.ctx_size;
            inference.use_gpu = config.use_gpu;
            inference.gpu_id = config.gpu_id;
            inference.prompt = config.prompt;

            std::vector<AudioSpan> batch_audio;
            std::vector<int> slot_ids;
            batch_audio.reserve(active.size());
            slot_ids.reserve(active.size());

            for (auto& request : active) {
                const AudioChunk& chunk = chunks[static_cast<size_t>(request.id)];
                size_t begin = std::min(chunk.infer_start_sample, samples.size());
                size_t end = std::min(chunk.infer_end_sample, samples.size());
                if (end <= begin) {
                    batch_audio.push_back(AudioSpan{});
                } else {
                    batch_audio.push_back(AudioSpan{samples.data() + begin, end - begin});
                }
                slot_ids.push_back(request.kv.slot_id);
                request.phase = OfflineRequestPhase::Decoding;
            }

            std::vector<InferenceResult> asr_results =
                recognizer_.transcribe_audio_batch_gpu(batch_audio, slot_ids, inference);

            for (size_t i = 0; i < active.size(); i++) {
                auto& request = active[i];
                const AudioChunk& chunk = chunks[static_cast<size_t>(request.id)];
                const InferenceResult& asr = asr_results[i];

                OfflineChunkResult result;
                result.id = request.id;
                result.slot_id = request.kv.slot_id;
                result.start_sec = chunk.start_sec;
                result.end_sec = chunk.end_sec;
                result.text = asr.text;
                result.ok = !asr.text.empty();
                result.encoder_ms = asr.encoder_ms;
                result.prefill_ms = asr.prefill_ms;
                result.decode_ms = asr.decode_ms;
                result.total_ms = asr.total_ms;
                result.prefill_tokens = asr.prefill_tokens;
                result.decode_tokens = asr.decode_tokens;

                request.text = result.text;
                request.phase = result.ok ? OfflineRequestPhase::Finished
                                          : OfflineRequestPhase::Failed;

                results.push_back(result);
                kv_pool->release(request.kv);

                if (progress) {
                    progress(++completed, static_cast<int>(chunks.size()), result);
                }
            }
        } else {
            for (auto& request : active) {
                request.phase = OfflineRequestPhase::Decoding;
                OfflineChunkResult result = run_one(
                    samples, sample_rate, request,
                    chunks[static_cast<size_t>(request.id)], config, request.id);
                result.id = request.id;
                result.slot_id = request.kv.slot_id;

                request.text = result.text;
                request.phase = result.ok ? OfflineRequestPhase::Finished
                                          : OfflineRequestPhase::Failed;

                results.push_back(result);
                kv_pool->release(request.kv);

                if (progress) {
                    progress(++completed, static_cast<int>(chunks.size()), result);
                }
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const OfflineChunkResult& a, const OfflineChunkResult& b) {
                  return a.id < b.id;
              });
    last_stats_.sync_fallback_total();
    return results;
}

std::vector<OfflineChunkResult> OfflineBatchTranscriber::transcribe_unified_gpu(
    const std::vector<float>& samples,
    int sample_rate,
    const std::vector<AudioChunk>& chunks,
    const OfflineBatchConfig& config,
    OfflineProgressCallback progress) {
    (void)sample_rate;
    reset_stats(chunks.size(), config);

    std::vector<OfflineChunkResult> results;
    results.reserve(chunks.size());
    if (!config.use_gpu || !config.use_paged_kv ||
        !config.enable_dynamic_kv_blocks || config.kv_block_size <= 0 ||
        config.batch_size <= 0 || config.max_tokens <= 0 ||
        config.max_num_scheduled_tokens <= 0 ||
        config.max_prefill_chunk_tokens <= 0 ||
        config.max_frontend_requests_per_step <= 0) {
        printf("[OfflineBatch] ERROR: unified scheduler requires GPU, paged KV, "
               "dynamic blocks, and positive scheduling limits\n");
        return results;
    }
    if (!recognizer_.configure_mixed_graph_cache(
            config.mixed_graph_cache_entries)) {
        printf("[OfflineBatch] ERROR: cannot configure mixed graph cache\n");
        return results;
    }
    if (config.batch_size > std::numeric_limits<int>::max() -
            config.max_frontend_requests_per_step ||
        !recognizer_.reserve_prompt_embedding_slots(
            config.batch_size + config.max_frontend_requests_per_step)) {
        printf("[OfflineBatch] ERROR: cannot reserve unified prompt slots\n");
        return results;
    }
    const int num_blocks = resolve_paged_kv_num_blocks(config);
    PagedKVBlockPool paged_pool(num_blocks, config.kv_block_size);
    last_stats_.kv_block_capacity = num_blocks;
    printf("[OfflineBatch] Unified GPU scheduler: blocks=%d block_size=%d "
           "token_budget=%d prefill_chunk=%d frontend_per_step=%d\n",
           num_blocks, config.kv_block_size,
           config.max_num_scheduled_tokens,
           config.max_prefill_chunk_tokens,
           config.max_frontend_requests_per_step);

    InferenceConfig inference;
    inference.max_new_tokens = config.max_tokens;
    inference.n_threads = config.n_threads;
    inference.kv_cache_size = config.ctx_size;
    inference.use_gpu = true;
    inference.gpu_id = config.gpu_id;
    inference.prompt = config.prompt;

    TaskPrefixKVCache prefix_cache;
    if (config.enable_prefix_kv_cache) {
        prefix_cache.key.prefix_token_ids =
            recognizer_.prompt_prefix_ids(config.prompt);
        prefix_cache.key.block_size = config.kv_block_size;
        prefix_cache.key.model_generation = 1;
        prefix_cache.prefix_tokens = static_cast<int>(
            prefix_cache.key.prefix_token_ids.size());
        if (prefix_cache.prefix_tokens > 0) {
            prefix_cache.state = paged_pool.acquire(
                -1, prefix_cache.prefix_tokens);
            if (!prefix_cache.state.valid()) {
                printf("[OfflineBatch] ERROR: no KV capacity for unified prefix cache\n");
                return results;
            }
            GPUPrefillState prefix = recognizer_.gpu_prefill_prefix_paged(
                prefix_cache.key.prefix_token_ids, -1,
                prefix_cache.state.block_table,
                prefix_cache.state.block_size, inference);
            if (!prefix.ok) {
                paged_pool.release(prefix_cache.state);
                printf("[OfflineBatch] ERROR: unified prefix KV prefill failed\n");
                return results;
            }
            prefix_cache.ready = true;
            last_stats_.prefix_cache_builds = 1;
            update_peak_blocks(paged_pool);
        }
    }

    UnifiedSchedulerConfig scheduler_config;
    scheduler_config.max_num_seqs = std::min(
        config.batch_size, config.max_num_scheduled_tokens);
    scheduler_config.max_num_scheduled_tokens =
        config.max_num_scheduled_tokens;
    scheduler_config.max_prefill_chunk_tokens =
        config.max_prefill_chunk_tokens;
    UnifiedTokenScheduler scheduler(scheduler_config);

    struct UnifiedActiveRequest {
        OfflineRequest request;
        PreparedGPUPrompt prepared;
        UnifiedRequestProgress progress;
    };

    std::vector<UnifiedActiveRequest> active;
    active.reserve(static_cast<size_t>(config.batch_size));
    std::deque<UnifiedActiveRequest> prepared_queue;
    const std::vector<size_t> frontend_order = plan_frontend_request_order(
        chunks, config.max_frontend_requests_per_step,
        config.frontend_bucket_window);
    size_t next_request = 0;
    int completed = 0;
    const int eos_id = recognizer_.tokenizer().eos_id();

    auto release_prompt = [&](UnifiedActiveRequest& state) {
        if (state.prepared.embeddings.valid()) {
            recognizer_.release_prompt_gpu(state.prepared.embeddings);
            state.prepared.embeddings = {};
        }
    };

    auto emit_finished = [&](UnifiedActiveRequest& state) {
        release_prompt(state);
        state.request.tokens = state.progress.output_tokens;
        OfflineChunkResult result;
        result.id = state.request.id;
        result.slot_id = -1;
        result.start_sec = state.request.start_sec;
        result.end_sec = state.request.end_sec;
        result.text = recognizer_.tokenizer().decode(state.request.tokens);
        result.ok = state.request.phase != OfflineRequestPhase::Failed &&
                    !result.text.empty();
        result.token_ids = state.request.tokens;
        result.encoder_ms = state.request.encoder_ms;
        result.prefill_ms = state.request.prefill_ms;
        result.decode_ms = state.request.decode_ms;
        result.prefill_tokens = state.request.prefill_tokens;
        result.decode_tokens = static_cast<int>(state.request.tokens.size());
        result.total_ms = result.encoder_ms + result.prefill_ms + result.decode_ms;
        results.push_back(std::move(result));

        const int allocated_blocks = static_cast<int>(
            state.request.kv.block_table.size());
        const int used_positions = std::max(
            0, state.progress.num_computed_tokens);
        const int used_blocks = paged_blocks_for_tokens(
            used_positions, config.kv_block_size);
        const int clamped_used = std::min(allocated_blocks, used_blocks);
        const int wasted_blocks = std::max(0, allocated_blocks - clamped_used);
        last_stats_.paged_kv_released_requests++;
        last_stats_.kv_allocated_blocks_total += allocated_blocks;
        last_stats_.kv_used_blocks_total += clamped_used;
        last_stats_.kv_wasted_blocks_total += wasted_blocks;
        last_stats_.kv_allocated_positions_total +=
            allocated_blocks * config.kv_block_size;
        last_stats_.kv_used_positions_total += used_positions;
        last_stats_.kv_wasted_positions_total += std::max(
            0, allocated_blocks * config.kv_block_size - used_positions);
        last_stats_.kv_max_wasted_blocks = std::max(
            last_stats_.kv_max_wasted_blocks, wasted_blocks);
        paged_pool.release(state.request.kv);

        if (progress) {
            progress(++completed, static_cast<int>(chunks.size()), results.back());
        } else {
            ++completed;
        }
    };

    struct FrontendWorkResult {
        PreparedFbankBatch fbank;
        std::vector<PreparedGPUPrompt> prepared;
        double frontend_wall_ms = 0.0;
        bool gpu_prepared = false;
    };
    struct FrontendJob {
        std::vector<UnifiedActiveRequest> candidates;
        std::vector<AudioSpan> audio;
        std::vector<int> request_ids;
        std::future<FrontendWorkResult> frontend_future;
        bool async_frontend = false;
        bool async_gpu_frontend = false;
        bool active = false;
    };
    FrontendJob frontend_job;

    auto launch_frontend_job = [&]() -> int {
        if (frontend_job.active || next_request >= frontend_order.size()) {
            return 0;
        }
        const int remaining = static_cast<int>(std::min(
            frontend_order.size() - next_request,
            static_cast<size_t>(std::numeric_limits<int>::max())));
        const int target = std::max(0, std::min({
            config.max_frontend_requests_per_step,
            remaining,
        }));
        if (target <= 0) {
            return 0;
        }

        frontend_job = FrontendJob{};
        frontend_job.candidates.reserve(static_cast<size_t>(target));
        frontend_job.audio.reserve(static_cast<size_t>(target));
        frontend_job.request_ids.reserve(static_cast<size_t>(target));

        int consumed = 0;
        for (int count = 0; count < target; ++count) {
            const size_t chunk_index = frontend_order[next_request];
            const int request_id = static_cast<int>(chunk_index);
            const AudioChunk& chunk = chunks[chunk_index];
            const size_t begin = std::min(
                chunk.infer_start_sample, samples.size());
            const size_t end = std::min(
                chunk.infer_end_sample, samples.size());
            ++last_stats_.admit_attempts;
            ++next_request;
            ++consumed;

            UnifiedActiveRequest state;
            state.request.id = request_id;
            state.request.phase = OfflineRequestPhase::Prefilling;
            state.request.max_tokens = config.max_tokens;
            state.request.start_sec = chunk.start_sec;
            state.request.end_sec = chunk.end_sec;
            state.request.kv.request_id = request_id;
            state.request.kv.slot_id = -1;
            state.request.kv.block_size = config.kv_block_size;
            if (end <= begin) {
                state.request.phase = OfflineRequestPhase::Failed;
                emit_finished(state);
                continue;
            }
            frontend_job.candidates.push_back(std::move(state));
            frontend_job.audio.push_back(
                AudioSpan{samples.data() + begin, end - begin});
            frontend_job.request_ids.push_back(request_id);
        }

        frontend_job.active = !frontend_job.candidates.empty();
        frontend_job.async_frontend = frontend_job.active &&
            config.enable_frontend_batching &&
            config.enable_frontend_prefetch;
        frontend_job.async_gpu_frontend =
            frontend_job.async_frontend &&
            config.enable_gpu_frontend_overlap &&
            recognizer_.supports_gpu_frontend_overlap() &&
            recognizer_.free_prompt_embedding_slots() >=
                static_cast<int>(frontend_job.candidates.size());
        if (frontend_job.async_frontend) {
            const std::vector<AudioSpan> audio = frontend_job.audio;
            const std::vector<int> ids = frontend_job.request_ids;
            const int threads = config.n_threads;
            const bool prepare_on_gpu = frontend_job.async_gpu_frontend;
            const int cached_prefix_tokens = prefix_cache.ready
                ? prefix_cache.prefix_tokens
                : 0;
            const InferenceConfig frontend_inference = inference;
            frontend_job.frontend_future = std::async(
                std::launch::async,
                [this, audio, ids, threads, prepare_on_gpu,
                 cached_prefix_tokens, frontend_inference]() {
                    FrontendWorkResult work;
                    work.fbank = recognizer_.prepare_fbank_batch(
                        audio, ids, threads);
                    if (prepare_on_gpu) {
                        const auto begin = std::chrono::steady_clock::now();
                        work.prepared =
                            recognizer_.prepare_prompts_gpu_batch_from_fbank(
                                work.fbank, cached_prefix_tokens,
                                frontend_inference);
                        recognizer_.synchronize_gpu_frontend();
                        const auto end = std::chrono::steady_clock::now();
                        work.frontend_wall_ms =
                            std::chrono::duration<double, std::milli>(
                                end - begin).count();
                        work.gpu_prepared = true;
                    }
                    return work;
                });
            ++last_stats_.frontend_prefetch_launches;
        }
        if (consumed > 0) {
            last_stats_.admit_rounds++;
            last_stats_.admit_round_requests += consumed;
            last_stats_.max_admit_round_size = std::max(
                last_stats_.max_admit_round_size, consumed);
        }
        return consumed;
    };

    auto consume_frontend_job = [&](bool wait_for_ready) -> int {
        if (!frontend_job.active) {
            return 0;
        }
        const int prompt_slots = recognizer_.free_prompt_embedding_slots();
        if (!frontend_job.async_gpu_frontend &&
            prompt_slots < static_cast<int>(frontend_job.candidates.size())) {
            return 0;
        }

        PreparedFbankBatch fbank_batch;
        std::vector<PreparedGPUPrompt> prepared;
        double frontend_wall_ms = 0.0;
        bool gpu_prepared = false;
        if (config.enable_frontend_batching) {
            if (frontend_job.async_frontend) {
                const std::future_status status =
                    frontend_job.frontend_future.wait_for(
                        std::chrono::milliseconds(0));
                if (status != std::future_status::ready && !wait_for_ready) {
                    return 0;
                }
                if (status == std::future_status::ready) {
                    ++last_stats_.frontend_prefetch_ready_hits;
                }
                const auto wait_begin = std::chrono::steady_clock::now();
                try {
                    FrontendWorkResult work =
                        frontend_job.frontend_future.get();
                    fbank_batch = std::move(work.fbank);
                    prepared = std::move(work.prepared);
                    frontend_wall_ms = work.frontend_wall_ms;
                    gpu_prepared = work.gpu_prepared;
                } catch (const std::exception& error) {
                    printf("[OfflineBatch] WARNING: frontend prefetch failed: %s\n",
                           error.what());
                } catch (...) {
                    printf("[OfflineBatch] WARNING: frontend prefetch failed\n");
                }
                const auto wait_end = std::chrono::steady_clock::now();
                last_stats_.frontend_prefetch_wait_ms +=
                    std::chrono::duration<double, std::milli>(
                        wait_end - wait_begin).count();
            } else {
                fbank_batch = recognizer_.prepare_fbank_batch(
                    frontend_job.audio, frontend_job.request_ids,
                    config.n_threads);
            }
            if (!fbank_batch.ok) {
                fbank_batch = recognizer_.prepare_fbank_batch(
                    frontend_job.audio, frontend_job.request_ids,
                    config.n_threads);
                prepared.clear();
                gpu_prepared = false;
            }
            last_stats_.frontend_fbank_ms += fbank_batch.fbank_ms;
            last_stats_.frontend_input_frames += fbank_batch.input_frames;
            last_stats_.frontend_padded_frames += fbank_batch.padded_frames;
        }

        const int cached_prefix_tokens = prefix_cache.ready
            ? prefix_cache.prefix_tokens
            : 0;
        if (!gpu_prepared) {
            const auto frontend_begin =
                std::chrono::high_resolution_clock::now();
            if (config.enable_frontend_batching) {
                prepared = recognizer_.prepare_prompts_gpu_batch_from_fbank(
                    fbank_batch, cached_prefix_tokens, inference);
            } else {
                prepared.reserve(frontend_job.audio.size());
                for (size_t i = 0; i < frontend_job.audio.size(); ++i) {
                    prepared.push_back(recognizer_.prepare_prompt_gpu(
                        frontend_job.audio[i], frontend_job.request_ids[i],
                        cached_prefix_tokens, inference));
                }
            }
            recognizer_.synchronize_gpu_frontend();
            const auto frontend_end =
                std::chrono::high_resolution_clock::now();
            frontend_wall_ms =
                std::chrono::duration<double, std::milli>(
                    frontend_end - frontend_begin).count();
        }

        const bool requested_batch = config.enable_frontend_batching &&
            frontend_job.candidates.size() > 1;
            const bool any_batch_success = std::any_of(
                prepared.begin(), prepared.end(),
                [](const PreparedGPUPrompt& item) { return item.ok; });
            const bool used_batch = requested_batch && any_batch_success &&
                prepared.size() == frontend_job.candidates.size() &&
                std::all_of(
                    prepared.begin(), prepared.end(),
                    [&](const PreparedGPUPrompt& item) {
                        return !item.ok || item.frontend_batch_size ==
                            static_cast<int>(frontend_job.candidates.size());
                    });
            if (used_batch) {
                ++last_stats_.frontend_batch_calls;
                last_stats_.frontend_batched_requests += static_cast<int>(
                    std::count_if(
                        prepared.begin(), prepared.end(),
                        [](const PreparedGPUPrompt& item) { return item.ok; }));
                last_stats_.frontend_batch_wall_ms += frontend_wall_ms;
            } else {
                last_stats_.frontend_single_requests += static_cast<int>(
                    std::count_if(
                        prepared.begin(), prepared.end(),
                        [](const PreparedGPUPrompt& item) { return item.ok; }));
                if (requested_batch) {
                    ++last_stats_.frontend_batch_fallbacks;
                }
                last_stats_.frontend_single_wall_ms += frontend_wall_ms;
            }

            if (prepared.size() != frontend_job.candidates.size()) {
                prepared.resize(frontend_job.candidates.size());
            }
            for (size_t i = 0; i < frontend_job.candidates.size(); ++i) {
                UnifiedActiveRequest& state = frontend_job.candidates[i];
                const int request_id = state.request.id;
                state.prepared = std::move(prepared[i]);
                if (!state.prepared.ok ||
                    state.prepared.prompt_tokens <= 0 ||
                    state.prepared.prompt_tokens > config.ctx_size) {
                    state.request.phase = OfflineRequestPhase::Failed;
                    emit_finished(state);
                    continue;
                }

                state.request.encoder_ms = state.prepared.encoder_ms;
                state.request.prefill_tokens = state.prepared.prompt_tokens;
                prepared_queue.push_back(std::move(state));
                last_stats_.frontend_prepared_queue_peak = std::max(
                    last_stats_.frontend_prepared_queue_peak,
                    static_cast<int>(prepared_queue.size()));
            }
        const int consumed = static_cast<int>(
            frontend_job.candidates.size());
        frontend_job = FrontendJob{};
        return consumed;
    };

    auto activate_prepared = [&]() -> int {
        int activated = 0;
        while (!prepared_queue.empty() &&
               static_cast<int>(active.size()) < config.batch_size) {
            const UnifiedActiveRequest& candidate = prepared_queue.front();
            const int candidate_prefix_tokens = prefix_cache.ready
                ? candidate.prepared.cached_prefix_tokens
                : 0;
            const int initial_blocks = paged_initial_admission_blocks(
                candidate.prepared.prompt_tokens,
                candidate_prefix_tokens,
                config.kv_block_size);
            if (paged_pool.free_count() < initial_blocks) {
                ++last_stats_.admit_no_kv_capacity;
                break;
            }

            UnifiedActiveRequest state = std::move(prepared_queue.front());
            prepared_queue.pop_front();
            const int request_id = state.request.id;
            const int cached_prefix_tokens = state.prepared.cached_prefix_tokens;

            if (prefix_cache.ready) {
                state.request.kv = prefix_cache.state;
                state.request.kv.request_id = request_id;
                if (!paged_pool.retain(state.request.kv)) {
                    state.request.kv = {};
                    state.request.phase = OfflineRequestPhase::Failed;
                    emit_finished(state);
                    continue;
                }
                ++last_stats_.prefix_cache_hits;
                last_stats_.prefix_tokens_reused += cached_prefix_tokens;
                last_stats_.prefix_prefill_tokens_saved +=
                    cached_prefix_tokens;

                const int partial_rows = paged_prefix_partial_rows(
                    cached_prefix_tokens, config.kv_block_size);
                if (partial_rows > 0) {
                    const size_t logical_block =
                        state.request.kv.block_table.size() - 1;
                    const int source_block =
                        state.request.kv.block_table[logical_block];
                    const int replacement = paged_pool.allocate_block();
                    if (replacement < 0 ||
                        !recognizer_.copy_paged_kv_block_rows(
                            source_block, replacement, partial_rows,
                            config.kv_block_size) ||
                        !paged_pool.replace_block_after_cow(
                            state.request.kv, logical_block, replacement)) {
                        if (replacement >= 0 &&
                            paged_pool.ref_count(replacement) > 0) {
                            paged_pool.release_block(replacement);
                        }
                        state.request.phase = OfflineRequestPhase::Failed;
                        emit_finished(state);
                        continue;
                    }
                    ++last_stats_.prefix_cow_copies;
                }
            }

            if (initial_blocks > 0 &&
                state.request.kv.block_table.size() ==
                    static_cast<size_t>(paged_blocks_for_tokens(
                        cached_prefix_tokens, config.kv_block_size)) &&
                cached_prefix_tokens % config.kv_block_size == 0) {
                if (!paged_pool.append_block(state.request.kv)) {
                    ++last_stats_.admit_no_kv_capacity;
                    state.request.phase = OfflineRequestPhase::Failed;
                    emit_finished(state);
                    break;
                }
            }

            state.progress.request_id = request_id;
            state.progress.prompt_tokens = state.prepared.prompt_tokens;
            state.progress.num_computed_tokens = cached_prefix_tokens;
            state.progress.max_output_tokens = config.max_tokens;
            active.push_back(std::move(state));
            ++last_stats_.admit_success;
            ++activated;
        }
        return activated;
    };

    auto fill_active = [&]() -> int {
        int made_progress = 0;
        while (static_cast<int>(active.size()) < config.batch_size) {
            if (!prepared_queue.empty()) {
                const int activated = activate_prepared();
                made_progress += activated;
                if (activated == 0) {
                    break;
                }
                continue;
            }
            if (!frontend_job.active) {
                made_progress += launch_frontend_job();
            }
            if (!frontend_job.active) {
                break;
            }
            const int prepared = consume_frontend_job(active.empty());
            made_progress += prepared;
            if (prepared == 0) {
                break;
            }
        }
        if (!frontend_job.active && next_request < frontend_order.size()) {
            made_progress += launch_frontend_job();
        }
        return made_progress;
    };

    auto rollback_appended = [&](const std::vector<int>& appended) {
        for (size_t i = 0; i < active.size() && i < appended.size(); ++i) {
            for (int count = 0; count < appended[i]; ++count) {
                if (active[i].request.kv.block_table.empty()) {
                    break;
                }
                const int block = active[i].request.kv.block_table.back();
                active[i].request.kv.block_table.pop_back();
                paged_pool.release_block(block);
            }
            active[i].request.kv.ctx_size = static_cast<int>(
                active[i].request.kv.block_table.size()) *
                config.kv_block_size;
        }
    };

    while (!active.empty() || !prepared_queue.empty() ||
           frontend_job.active || next_request < frontend_order.size()) {
        for (size_t i = active.size(); i > 0; --i) {
            const size_t index = i - 1;
            if (active[index].progress.finished ||
                active[index].request.phase == OfflineRequestPhase::Finished ||
                active[index].request.phase == OfflineRequestPhase::Failed) {
                emit_finished(active[index]);
                active.erase(active.begin() +
                             static_cast<std::ptrdiff_t>(index));
            }
        }

        const int admission_progress = fill_active();
        if (active.empty()) {
            if (prepared_queue.empty() && !frontend_job.active &&
                next_request >= frontend_order.size()) {
                break;
            }
            if (admission_progress == 0) {
                printf("[OfflineBatch] ERROR: unified frontend cannot admit a request\n");
                break;
            }
            continue;
        }

        std::vector<UnifiedRequestProgress> progress_snapshot;
        progress_snapshot.reserve(active.size());
        for (auto& state : active) {
            state.progress.max_schedulable_tokens = std::max(
                0, config.ctx_size - state.progress.num_computed_tokens);
            progress_snapshot.push_back(state.progress);
        }
        MixedBatchPlan plan = scheduler.build_plan(progress_snapshot);
        if (!plan.ok()) {
            printf("[OfflineBatch] ERROR: unified scheduler rejected request state\n");
            for (auto& state : active) {
                state.request.phase = OfflineRequestPhase::Failed;
            }
            continue;
        }

        std::vector<ScheduledSequence> executable;
        std::vector<int> appended(active.size(), 0);
        executable.reserve(plan.sequences.size());
        for (ScheduledSequence scheduled : plan.sequences) {
            if (scheduled.request_index < 0 ||
                scheduled.request_index >= static_cast<int>(active.size())) {
                continue;
            }
            UnifiedActiveRequest& state = active[
                static_cast<size_t>(scheduled.request_index)];
            const int room = config.ctx_size - scheduled.token_offset;
            scheduled.num_tokens = std::min(scheduled.num_tokens, room);
            if (scheduled.num_tokens <= 0) {
                state.progress.finished = true;
                state.request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            int scheduled_end = scheduled.token_offset + scheduled.num_tokens;
            int needed_blocks = paged_blocks_for_tokens(
                scheduled_end, config.kv_block_size);
            while (static_cast<int>(state.request.kv.block_table.size()) <
                   needed_blocks) {
                if (!paged_pool.append_block(state.request.kv)) {
                    break;
                }
                ++appended[static_cast<size_t>(scheduled.request_index)];
                if (scheduled.input_kind == UnifiedInputKind::Prompt) {
                    last_stats_.dynamic_prefill_blocks++;
                } else {
                    last_stats_.dynamic_decode_appends++;
                }
            }

            const int backed_end = static_cast<int>(
                state.request.kv.block_table.size()) * config.kv_block_size;
            if (scheduled_end > backed_end) {
                if (scheduled.input_kind == UnifiedInputKind::Decode) {
                    continue;
                }
                scheduled.num_tokens = std::max(
                    0, backed_end - scheduled.token_offset);
                scheduled_end = scheduled.token_offset + scheduled.num_tokens;
            }
            if (scheduled.num_tokens <= 0) {
                continue;
            }
            scheduled.produces_logits =
                scheduled.input_kind == UnifiedInputKind::Decode ||
                scheduled_end == state.progress.prompt_tokens;
            executable.push_back(scheduled);
        }
        update_peak_blocks(paged_pool);

        if (executable.empty()) {
            rollback_appended(appended);
            printf("[OfflineBatch] ERROR: unified KV pool made no scheduling progress\n");
            active.front().request.phase = OfflineRequestPhase::Failed;
            continue;
        }

        std::vector<GPUMixedStepInput> mixed_inputs;
        mixed_inputs.reserve(executable.size());
        for (const ScheduledSequence& scheduled : executable) {
            UnifiedActiveRequest& state = active[
                static_cast<size_t>(scheduled.request_index)];
            GPUMixedStepInput input;
            input.source.scheduled = scheduled;
            input.source.prompt_tokens = state.progress.prompt_tokens;
            input.source.cached_prefix_tokens =
                state.prepared.cached_prefix_tokens;
            input.source.block_table = state.request.kv.block_table;
            if (scheduled.input_kind == UnifiedInputKind::Prompt) {
                input.source.decode_token_id = -1;
                input.prompt_embeddings = state.prepared.embeddings;
            } else {
                const int output_index =
                    scheduled.token_offset - state.progress.prompt_tokens;
                if (output_index < 0 || output_index >=
                        static_cast<int>(state.progress.output_tokens.size())) {
                    state.request.phase = OfflineRequestPhase::Failed;
                    continue;
                }
                input.source.decode_token_id = state.progress.output_tokens[
                    static_cast<size_t>(output_index)];
            }
            mixed_inputs.push_back(std::move(input));
        }
        if (mixed_inputs.size() != executable.size()) {
            rollback_appended(appended);
            continue;
        }

        GPUMixedStepResult step = recognizer_.gpu_mixed_step_paged(
            mixed_inputs, config.kv_block_size, inference);
        if (!step.ok) {
            rollback_appended(appended);
            printf("[OfflineBatch] ERROR: packed mixed GPU step failed\n");
            for (const ScheduledSequence& scheduled : executable) {
                active[static_cast<size_t>(scheduled.request_index)]
                    .request.phase = OfflineRequestPhase::Failed;
            }
            continue;
        }

        last_stats_.unified_steps++;
        last_stats_.unified_scheduled_tokens += step.total_tokens;
        last_stats_.unified_selected_rows += step.outputs.size();
        last_stats_.unified_staging_ms += step.staging_ms;
        last_stats_.unified_compute_ms += step.compute_ms;
        if (step.graph_cache_hit) {
            last_stats_.unified_graph_cache_hits++;
        } else {
            last_stats_.unified_graph_cache_misses++;
        }
        const uint64_t evictions = step.graph_cache_evictions;
        last_stats_.unified_graph_cache_evictions = static_cast<long long>(
            std::min<uint64_t>(
                evictions,
                static_cast<uint64_t>(std::numeric_limits<long long>::max())));
        last_stats_.unified_graph_cache_entries_peak = std::max(
            last_stats_.unified_graph_cache_entries_peak,
            step.graph_cache_entries);
        last_stats_.unified_graph_cache_capacity =
            step.graph_cache_capacity;
        OfflineMixedGraphShapeStat& shape =
            last_stats_.unified_graph_shapes[step.graph_shape_hash];
        shape.shape_hash = step.graph_shape_hash;
        ++shape.calls;
        shape.prefill_tokens = step.prefill_tokens;
        shape.decode_tokens = step.decode_tokens;
        shape.total_tokens = step.total_tokens;
        shape.max_blocks = step.graph_max_blocks;
        shape.max_n_kv = step.graph_max_n_kv;
        shape.prefill_sequences = step.graph_prefill_sequences;
        if (step.prefill_tokens > 0 && step.decode_tokens > 0) {
            last_stats_.unified_mixed_steps++;
        } else if (step.prefill_tokens > 0) {
            last_stats_.unified_prefill_only_steps++;
        } else {
            last_stats_.unified_decode_only_steps++;
        }
        if (step.decode_tokens > 0) {
            last_stats_.decode_steps++;
            last_stats_.active_batch_sum += step.decode_tokens;
            if (step.decode_tokens > 1) {
                last_stats_.decode_group_calls++;
            } else {
                last_stats_.fallback_reasons.single_request++;
            }
        }
        if (step.prefill_tokens > 0) {
            last_stats_.prefill_wall_ms += step.compute_ms;
        }

        bool commit_failed = false;
        for (const ScheduledSequence& scheduled : executable) {
            UnifiedActiveRequest& state = active[
                static_cast<size_t>(scheduled.request_index)];
            if (!UnifiedTokenScheduler::commit_sequence(
                    state.progress, scheduled)) {
                state.request.phase = OfflineRequestPhase::Failed;
                commit_failed = true;
                continue;
            }
            state.request.n_past = state.progress.num_computed_tokens;
            if (scheduled.input_kind == UnifiedInputKind::Prompt) {
                state.request.prefill_ms += step.compute_ms;
                state.request.phase = state.progress.prompt_complete()
                    ? OfflineRequestPhase::Decoding
                    : OfflineRequestPhase::Prefilling;
            } else {
                state.request.decode_ms += step.compute_ms;
            }
        }
        if (commit_failed) {
            continue;
        }

        for (const GPUMixedStepOutput& output : step.outputs) {
            const auto state_it = std::find_if(
                active.begin(), active.end(), [&](const UnifiedActiveRequest& state) {
                    return state.progress.request_id == output.request_id;
                });
            if (state_it == active.end() || output.next_token < 0) {
                continue;
            }
            const SampleCommitResult committed =
                UnifiedTokenScheduler::commit_sample(
                    state_it->progress, output.next_token, eos_id);
            if (committed == SampleCommitResult::Invalid) {
                state_it->request.phase = OfflineRequestPhase::Failed;
                continue;
            }
            state_it->request.tokens = state_it->progress.output_tokens;
            if (committed == SampleCommitResult::FinishedEos ||
                committed == SampleCommitResult::FinishedLimit) {
                state_it->request.phase = OfflineRequestPhase::Finished;
            } else if (has_repetition_loop_local(state_it->request.tokens)) {
                state_it->progress.finished = true;
                state_it->request.phase = OfflineRequestPhase::Finished;
            }
        }

        for (auto& state : active) {
            if (state.progress.prompt_complete()) {
                release_prompt(state);
            }
        }
    }

    if (frontend_job.active) {
        if (frontend_job.async_frontend &&
            frontend_job.frontend_future.valid()) {
            frontend_job.frontend_future.wait();
        }
        for (UnifiedActiveRequest& state : frontend_job.candidates) {
            state.request.phase = OfflineRequestPhase::Failed;
            emit_finished(state);
        }
        frontend_job = FrontendJob{};
    }
    while (!prepared_queue.empty()) {
        UnifiedActiveRequest state = std::move(prepared_queue.front());
        prepared_queue.pop_front();
        state.request.phase = OfflineRequestPhase::Failed;
        emit_finished(state);
    }
    for (auto& state : active) {
        if (state.request.phase != OfflineRequestPhase::Finished) {
            state.request.phase = OfflineRequestPhase::Failed;
        }
        emit_finished(state);
    }
    active.clear();
    if (prefix_cache.ready) {
        paged_pool.release(prefix_cache.state);
        prefix_cache.ready = false;
    }
    std::sort(results.begin(), results.end(),
              [](const OfflineChunkResult& a, const OfflineChunkResult& b) {
                  return a.id < b.id;
              });
    last_stats_.kv_ownership_errors = paged_pool.ownership_errors();
    last_stats_.kv_final_free_blocks = paged_pool.free_count();
    last_stats_.sync_fallback_total();
    return results;
}

std::vector<OfflineChunkResult> OfflineBatchTranscriber::transcribe_continuous_gpu(
    const std::vector<float>& samples,
    int sample_rate,
    const std::vector<AudioChunk>& chunks,
    const OfflineBatchConfig& config,
    OfflineProgressCallback progress) {
    (void)sample_rate;

    reset_stats(chunks.size(), config);

    std::vector<OfflineChunkResult> results;
    results.reserve(chunks.size());

    std::unique_ptr<KVPool> kv_pool;
    PagedKVBlockPool* paged_pool = nullptr;
    if (config.use_paged_kv) {
        const int num_blocks = resolve_paged_kv_num_blocks(config);
        kv_pool = std::make_unique<PagedKVBlockPool>(num_blocks, config.kv_block_size);
        paged_pool = static_cast<PagedKVBlockPool*>(kv_pool.get());
        last_stats_.kv_block_capacity = num_blocks;
        printf("[OfflineBatch] GPU paged KV scheduler: blocks=%d block_size=%d\n",
               num_blocks, config.kv_block_size);
    } else {
        kv_pool = std::make_unique<ContinuousKVSlotPool>(
            std::max(1, config.batch_size), config.ctx_size);
    }
    std::vector<OfflineRequest> active;
    active.reserve(static_cast<size_t>(std::max(1, config.batch_size)));

    InferenceConfig inference;
    inference.max_new_tokens = config.max_tokens;
    inference.n_threads = config.n_threads;
    inference.kv_cache_size = config.ctx_size;
    inference.use_gpu = true;
    inference.gpu_id = config.gpu_id;
    inference.prompt = config.prompt;

    const bool chunked_prefill_enabled =
        config.use_gpu && config.use_paged_kv &&
        config.enable_dynamic_kv_blocks &&
        config.prefill_chunk_tokens > 0;

    TaskPrefixKVCache prefix_cache;
    if (paged_pool && config.enable_prefix_kv_cache) {
        prefix_cache.key.prefix_token_ids = recognizer_.prompt_prefix_ids(config.prompt);
        prefix_cache.key.block_size = config.kv_block_size;
        prefix_cache.key.model_generation = 1;
        prefix_cache.prefix_tokens = static_cast<int>(
            prefix_cache.key.prefix_token_ids.size());
        if (prefix_cache.prefix_tokens > 0) {
            prefix_cache.state = paged_pool->acquire(-1, prefix_cache.prefix_tokens);
            if (!prefix_cache.state.valid()) {
                printf("[OfflineBatch] ERROR: no KV capacity for task prefix cache\n");
                return results;
            }
            GPUPrefillState prefix = recognizer_.gpu_prefill_prefix_paged(
                prefix_cache.key.prefix_token_ids, -1,
                prefix_cache.state.block_table, prefix_cache.state.block_size,
                inference);
            if (!prefix.ok) {
                printf("[OfflineBatch] ERROR: task prefix KV prefill failed\n");
                paged_pool->release(prefix_cache.state);
                return results;
            }
            prefix_cache.ready = true;
            last_stats_.prefix_cache_builds = 1;
            update_peak_blocks(*kv_pool);
        }
    }

    const int eos_id = recognizer_.tokenizer().eos_id();
    size_t next_request = 0;
    int completed = 0;
    PagedPrefillAdmissionHints prefill_admission_hints;

    auto emit_finished = [&](OfflineRequest& request) {
        OfflineChunkResult result;
        result.id = request.id;
        result.slot_id = request.kv.slot_id;
        result.start_sec = request.start_sec;
        result.end_sec = request.end_sec;
        result.text = recognizer_.tokenizer().decode(request.tokens);
        result.ok = request.phase != OfflineRequestPhase::Failed &&
                    !result.text.empty();
        copy_offline_result_tokens(request, result);
        result.encoder_ms = request.encoder_ms;
        result.prefill_ms = request.prefill_ms;
        result.decode_ms = request.decode_ms;
        result.decode_tokens = static_cast<int>(request.tokens.size());
        result.prefill_tokens = request.prefill_tokens;
        result.total_ms = result.encoder_ms + result.prefill_ms + result.decode_ms;
        results.push_back(result);
        if (request.kv.paged() && request.kv.block_size > 0) {
            const int allocated_blocks = static_cast<int>(request.kv.block_table.size());
            const int used_positions = std::max(0, result.prefill_tokens + result.decode_tokens);
            const int used_blocks = used_positions > 0
                ? (used_positions + request.kv.block_size - 1) / request.kv.block_size
                : 0;
            const int clamped_used_blocks = std::min(allocated_blocks, used_blocks);
            const int wasted_blocks = std::max(0, allocated_blocks - clamped_used_blocks);
            const int allocated_positions = allocated_blocks * request.kv.block_size;
            const int wasted_positions = std::max(0, allocated_positions - used_positions);

            last_stats_.paged_kv_released_requests++;
            last_stats_.kv_allocated_blocks_total += allocated_blocks;
            last_stats_.kv_used_blocks_total += clamped_used_blocks;
            last_stats_.kv_wasted_blocks_total += wasted_blocks;
            last_stats_.kv_allocated_positions_total += allocated_positions;
            last_stats_.kv_used_positions_total += used_positions;
            last_stats_.kv_wasted_positions_total += wasted_positions;
            last_stats_.kv_max_wasted_blocks =
                std::max(last_stats_.kv_max_wasted_blocks, wasted_blocks);
        }
        kv_pool->release(request.kv);
        if (progress) {
            progress(++completed, static_cast<int>(chunks.size()), results.back());
        } else {
            completed++;
        }
    };

    auto admit_one = [&]() -> bool {
        if (next_request >= chunks.size() ||
            static_cast<int>(active.size()) >= std::max(1, config.batch_size)) {
            return false;
        }

        int request_id = static_cast<int>(next_request);
        last_stats_.admit_attempts++;

        const AudioChunk& chunk = chunks[next_request];
        size_t begin = std::min(chunk.infer_start_sample, samples.size());
        size_t end = std::min(chunk.infer_end_sample, samples.size());

        OfflineRequest request;
        request.id = request_id;
        request.phase = OfflineRequestPhase::Prefilling;
        request.max_tokens = config.max_tokens;
        request.start_sec = chunk.start_sec;
        request.end_sec = chunk.end_sec;

        if (end > begin) {
            GPUPrefillState prefill;
            auto prefill_t0 = std::chrono::high_resolution_clock::now();
            if (chunked_prefill_enabled && paged_pool) {
                const size_t sample_count = end - begin;
                if (!prefill_admission_hints.can_prepare(
                        sample_count, paged_pool->free_count())) {
                    last_stats_.admit_no_kv_capacity++;
                    return false;
                }

                const int cached_prefix_tokens = prefix_cache.ready
                    ? prefix_cache.prefix_tokens
                    : 0;
                PreparedGPUPrompt prepared = recognizer_.prepare_prompt_gpu(
                    AudioSpan{samples.data() + begin, sample_count},
                    request_id, cached_prefix_tokens, inference);
                if (!prepared.ok) {
                    request.phase = OfflineRequestPhase::Failed;
                    emit_finished(request);
                    next_request++;
                    return true;
                }
                PromptEmbeddingLease embedding_lease(
                    recognizer_, prepared.embeddings);

                const int actual_prefill_tokens = prepared.prompt_tokens;
                const int required_blocks = paged_blocks_for_tokens(
                    actual_prefill_tokens, config.kv_block_size);
                const int prefix_blocks = prefix_cache.ready
                    ? static_cast<int>(prefix_cache.state.block_table.size())
                    : 0;
                const int partial_rows = prefix_cache.ready
                    ? paged_prefix_partial_rows(
                          cached_prefix_tokens, config.kv_block_size)
                    : 0;
                const int extra_blocks = prefix_cache.ready
                    ? std::max(0, required_blocks - prefix_blocks) +
                          (partial_rows > 0 ? 1 : 0)
                    : required_blocks;
                prefill_admission_hints.observe(sample_count, extra_blocks);
                if (paged_pool->free_count() < extra_blocks) {
                    last_stats_.admit_no_kv_capacity++;
                    return false;
                }

                KVHandle handle;
                handle.request_id = request_id;
                handle.slot_id = -1;
                handle.block_size = config.kv_block_size;
                if (prefix_cache.ready) {
                    handle = prefix_cache.state;
                    handle.request_id = request_id;
                    if (!paged_pool->retain(handle)) {
                        request.phase = OfflineRequestPhase::Failed;
                        emit_finished(request);
                        next_request++;
                        return true;
                    }
                    last_stats_.prefix_cache_hits++;
                    last_stats_.prefix_tokens_reused += cached_prefix_tokens;
                    last_stats_.prefix_prefill_tokens_saved += cached_prefix_tokens;

                    if (partial_rows > 0) {
                        const size_t logical_block = handle.block_table.size() - 1;
                        const int source_block = handle.block_table[logical_block];
                        const int replacement = paged_pool->allocate_block();
                        if (replacement < 0 ||
                            !recognizer_.copy_paged_kv_block_rows(
                                source_block, replacement, partial_rows,
                                config.kv_block_size) ||
                            !paged_pool->replace_block_after_cow(
                                handle, logical_block, replacement)) {
                            if (replacement >= 0 &&
                                paged_pool->ref_count(replacement) > 0) {
                                paged_pool->release_block(replacement);
                            }
                            paged_pool->release(handle);
                            request.phase = OfflineRequestPhase::Failed;
                            emit_finished(request);
                            next_request++;
                            return true;
                        }
                        last_stats_.prefix_cow_copies++;
                    }
                }

                request.kv = handle;
                const std::vector<std::pair<int, int>> chunks_to_run =
                    plan_prefill_chunks(
                        cached_prefix_tokens, actual_prefill_tokens,
                        config.prefill_chunk_tokens);
                bool chunk_failed = chunks_to_run.empty();
                float chunk_prefill_ms = 0.0f;
                std::vector<float> final_logits;

                for (size_t chunk_index = 0;
                     !chunk_failed && chunk_index < chunks_to_run.size();
                     ++chunk_index) {
                    const int chunk_offset = chunks_to_run[chunk_index].first;
                    const int chunk_tokens = chunks_to_run[chunk_index].second;
                    const int64_t chunk_end64 =
                        static_cast<int64_t>(chunk_offset) + chunk_tokens;
                    if (chunk_end64 <= 0 ||
                        chunk_end64 > std::numeric_limits<int>::max()) {
                        chunk_failed = true;
                        break;
                    }
                    const int chunk_end = static_cast<int>(chunk_end64);
                    const int chunk_required_blocks = paged_blocks_for_tokens(
                        chunk_end, config.kv_block_size);
                    while (static_cast<int>(handle.block_table.size()) <
                           chunk_required_blocks) {
                        if (!paged_pool->append_block(handle)) {
                            chunk_failed = true;
                            break;
                        }
                        last_stats_.dynamic_prefill_blocks++;
                        request.kv = handle;
                        update_peak_blocks(*kv_pool);
                    }
                    if (chunk_failed) {
                        break;
                    }

                    GPUPrefillState chunk_prefill =
                        recognizer_.gpu_prefill_prompt_chunk_paged(
                            prepared, chunk_offset, chunk_tokens,
                            handle.block_table, handle.block_size, inference);
                    const bool final_chunk =
                        chunk_index + 1 == chunks_to_run.size();
                    if (!chunk_prefill.ok ||
                        chunk_prefill.n_past != chunk_end ||
                        (final_chunk && chunk_prefill.logits.empty())) {
                        chunk_failed = true;
                        break;
                    }
                    chunk_prefill_ms += chunk_prefill.stats.prefill_ms;
                    if (final_chunk) {
                        final_logits = std::move(chunk_prefill.logits);
                    }
                }

                request.kv = handle;
                if (chunk_failed) {
                    request.phase = OfflineRequestPhase::Failed;
                    emit_finished(request);
                    next_request++;
                    return true;
                }

                prefill.ok = true;
                prefill.request_id = request_id;
                prefill.n_past = actual_prefill_tokens;
                prefill.block_table = handle.block_table;
                prefill.block_size = handle.block_size;
                prefill.logits = std::move(final_logits);
                prefill.stats.encoder_ms = prepared.encoder_ms;
                prefill.stats.prefill_ms = chunk_prefill_ms;
                prefill.stats.prefill_tokens = actual_prefill_tokens;
                update_peak_blocks(*kv_pool);
            } else if (paged_pool &&
                (config.enable_dynamic_kv_blocks || prefix_cache.ready)) {
                const size_t sample_count = end - begin;
                if (!prefill_admission_hints.can_prepare(
                        sample_count, paged_pool->free_count())) {
                    last_stats_.admit_no_kv_capacity++;
                    return false;
                }
                PreparedGPUAudio prepared = recognizer_.prepare_audio_gpu(
                    AudioSpan{samples.data() + begin, sample_count}, request_id);
                if (!prepared.ok) {
                    request.phase = OfflineRequestPhase::Failed;
                    emit_finished(request);
                    next_request++;
                    return true;
                }

                const int prefix_tokens = prefix_cache.ready
                    ? prefix_cache.prefix_tokens
                    : static_cast<int>(recognizer_.prompt_prefix_ids(config.prompt).size());
                const int suffix_tokens = static_cast<int>(
                    recognizer_.prompt_suffix_ids(config.prompt).size());
                const int actual_prefill_tokens =
                    prefix_tokens + prepared.audio_frames + suffix_tokens;
                const int static_budget = estimate_kv_budget_tokens(
                    chunk, samples, sample_rate, config);
                const int required_blocks = config.enable_dynamic_kv_blocks
                    ? paged_blocks_for_tokens(
                          actual_prefill_tokens, config.kv_block_size)
                    : paged_blocks_for_tokens(
                          static_budget, config.kv_block_size);
                const int prefix_blocks = prefix_cache.ready
                    ? static_cast<int>(prefix_cache.state.block_table.size())
                    : 0;
                const int partial_rows = prefix_cache.ready
                    ? paged_prefix_partial_rows(prefix_tokens, config.kv_block_size)
                    : 0;
                const int extra_blocks = prefix_cache.ready
                    ? std::max(0, required_blocks - prefix_blocks) +
                          (partial_rows > 0 ? 1 : 0)
                    : required_blocks;
                prefill_admission_hints.observe(sample_count, extra_blocks);
                if (paged_pool->free_count() < extra_blocks) {
                    last_stats_.admit_no_kv_capacity++;
                    return false;
                }

                KVHandle handle;
                handle.request_id = request_id;
                handle.slot_id = -1;
                handle.block_size = config.kv_block_size;
                if (prefix_cache.ready) {
                    handle = prefix_cache.state;
                    handle.request_id = request_id;
                    if (!paged_pool->retain(handle)) {
                        request.phase = OfflineRequestPhase::Failed;
                        emit_finished(request);
                        next_request++;
                        return true;
                    }
                    last_stats_.prefix_cache_hits++;
                    last_stats_.prefix_tokens_reused += prefix_tokens;
                    last_stats_.prefix_prefill_tokens_saved += prefix_tokens;

                    if (partial_rows > 0) {
                        const size_t logical_block = handle.block_table.size() - 1;
                        const int source_block = handle.block_table[logical_block];
                        const int replacement = paged_pool->allocate_block();
                        if (replacement < 0 ||
                            !recognizer_.copy_paged_kv_block_rows(
                                source_block, replacement, partial_rows,
                                config.kv_block_size) ||
                            !paged_pool->replace_block_after_cow(
                                handle, logical_block, replacement)) {
                            if (replacement >= 0 &&
                                paged_pool->ref_count(replacement) > 0) {
                                paged_pool->release_block(replacement);
                            }
                            paged_pool->release(handle);
                            request.phase = OfflineRequestPhase::Failed;
                            emit_finished(request);
                            next_request++;
                            return true;
                        }
                        last_stats_.prefix_cow_copies++;
                    }
                }

                while (static_cast<int>(handle.block_table.size()) < required_blocks) {
                    if (!paged_pool->append_block(handle)) {
                        paged_pool->release(handle);
                        last_stats_.admit_no_kv_capacity++;
                        return false;
                    }
                    if (config.enable_dynamic_kv_blocks) {
                        last_stats_.dynamic_prefill_blocks++;
                    }
                }
                request.kv = handle;
                update_peak_blocks(*kv_pool);
                prefill = recognizer_.gpu_prefill_prepared_audio_paged(
                    prepared, request_id, handle.block_table, handle.block_size,
                    prefix_cache.ready ? prefix_tokens : 0, inference);
            } else {
                const int kv_budget = estimate_kv_budget_tokens(
                    chunk, samples, sample_rate, config);
                if (!kv_pool->has_capacity(kv_budget)) {
                    last_stats_.admit_no_kv_capacity++;
                    return false;
                }
                request.kv = kv_pool->acquire(request_id, kv_budget);
                if (!request.kv.valid()) {
                    return false;
                }
                if (request.kv.paged()) {
                    update_peak_blocks(*kv_pool);
                    prefill = recognizer_.gpu_prefill_audio_paged(
                        AudioSpan{samples.data() + begin, end - begin},
                        request_id, request.kv.block_table,
                        request.kv.block_size, inference);
                } else {
                    prefill = recognizer_.gpu_prefill_audio_slot(
                        AudioSpan{samples.data() + begin, end - begin},
                        request_id, request.kv.slot_id, inference);
                }
            }
            auto prefill_t1 = std::chrono::high_resolution_clock::now();
            last_stats_.prefill_wall_ms +=
                std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();
            if (prefill.ok) {
                request.phase = OfflineRequestPhase::Decoding;
                request.n_past = prefill.n_past;
                request.logits = std::move(prefill.logits);
                request.next_token = argmax_logits(request.logits);
                if (request.kv.paged()) {
                    request.logits.clear();
                    request.logits.shrink_to_fit();
                }
                request.encoder_ms = prefill.stats.encoder_ms;
                request.prefill_ms = prefill.stats.prefill_ms;
                request.prefill_tokens = prefill.stats.prefill_tokens;
                active.push_back(std::move(request));
                last_stats_.admit_success++;
                next_request++;
                return true;
            }
        }

        request.phase = OfflineRequestPhase::Failed;
        emit_finished(request);
        next_request++;
        return true;
    };

    auto admit_round = [&]() -> int {
        int admitted = 0;
        while (next_request < chunks.size() &&
               static_cast<int>(active.size()) < std::max(1, config.batch_size)) {
            if (!admit_one()) {
                break;
            }
            admitted++;
        }
        if (admitted > 0) {
            last_stats_.admit_rounds++;
            last_stats_.admit_round_requests += admitted;
            last_stats_.max_admit_round_size =
                std::max(last_stats_.max_admit_round_size, admitted);
        }
        return admitted;
    };

    admit_round();

    while (!active.empty() || next_request < chunks.size()) {
        std::vector<GPUDecodeStepInput> decode_inputs;
        std::vector<size_t> decode_indices;
        decode_inputs.reserve(active.size());
        decode_indices.reserve(active.size());

        for (size_t i = 0; i < active.size(); i++) {
            OfflineRequest& request = active[i];
            if (request.phase != OfflineRequestPhase::Decoding) {
                continue;
            }
            if (static_cast<int>(request.tokens.size()) >= request.max_tokens ||
                request.n_past >= config.ctx_size) {
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            int token = request.next_token >= 0
                ? request.next_token
                : argmax_logits(request.logits);
            if (token == eos_id) {
                request.next_token = -1;
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            if (paged_pool && config.enable_dynamic_kv_blocks &&
                paged_decode_requires_append(
                    request.n_past, request.kv.block_table.size(),
                    request.kv.block_size)) {
                if (!paged_pool->append_block(request.kv)) {
                    request.waiting_for_kv_block = true;
                    continue;
                }
                request.waiting_for_kv_block = false;
                last_stats_.dynamic_decode_appends++;
                update_peak_blocks(*kv_pool);
            }
            request.next_token = -1;

            request.tokens.push_back(token);
            if (has_repetition_loop_local(request.tokens)) {
                printf("[OfflineBatch] WARNING: request %d stopped after detecting repetition loop\n",
                       request.id);
                request.phase = OfflineRequestPhase::Finished;
                continue;
            }

            GPUDecodeStepInput input;
            input.request_id = request.id;
            input.slot_id = request.kv.slot_id;
            input.n_past = request.n_past;
            input.token_id = token;
            input.block_table = request.kv.block_table;
            input.block_size = request.kv.block_size;
            decode_inputs.push_back(input);
            decode_indices.push_back(i);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<GPUDecodeStepOutput> decode_outputs;
        if (!decode_inputs.empty()) {
            last_stats_.decode_steps++;
            last_stats_.active_batch_sum += static_cast<long long>(decode_inputs.size());

            GPUDecodeDispatchStats dispatch_stats;
            decode_outputs = recognizer_.gpu_decode_step_slots(
                decode_inputs, inference, &dispatch_stats);
            last_stats_.fallback_reasons.token_id_fast_path_unavailable +=
                dispatch_stats.token_id_fast_path_unavailable;
            last_stats_.fallback_reasons.host_embedding_batch_unavailable +=
                dispatch_stats.host_embedding_batch_unavailable;
            last_stats_.fallback_reasons.serial_env_forced +=
                dispatch_stats.serial_env_forced;
            last_stats_.fallback_reasons.invalid_paged_input +=
                dispatch_stats.invalid_paged_input;
            if (decode_inputs.size() == 1) {
                last_stats_.fallback_reasons.single_request++;
            } else {
                last_stats_.decode_group_calls++;
            }
            last_stats_.sync_fallback_total();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (!decode_inputs.empty()) {
            last_stats_.decode_dispatch_wall_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
            last_stats_.scheduler_idle_steps++;
        }

        for (const auto& output : decode_outputs) {
            auto it = std::find_if(active.begin(), active.end(),
                [&](const OfflineRequest& request) {
                    return request.id == output.request_id;
                });
            if (it == active.end() || !output.ok) {
                continue;
            }
            it->n_past = output.n_past;
            it->next_token = output.next_token;
            if (!output.logits.empty()) {
                it->logits = output.logits;
            }
            it->decode_ms += step_ms;
            // This is wall time for the shared decode step. It is useful for
            // tracing but should not be summed as exclusive per-request time.
        }

        for (size_t i = 0; i < decode_inputs.size(); i++) {
            const int request_id = decode_inputs[i].request_id;
            const bool produced = std::any_of(
                decode_outputs.begin(), decode_outputs.end(),
                [&](const GPUDecodeStepOutput& output) {
                    return output.request_id == request_id && output.ok;
                });
            if (!produced && decode_indices[i] < active.size()) {
                printf("[OfflineBatch] ERROR: decode produced no output for request %d\n",
                       request_id);
                active[decode_indices[i]].phase = OfflineRequestPhase::Failed;
            }
        }

        if (decode_inputs.empty() && !active.empty()) {
            const bool has_releasable = std::any_of(
                active.begin(), active.end(), [](const OfflineRequest& request) {
                    return request.phase == OfflineRequestPhase::Finished ||
                           request.phase == OfflineRequestPhase::Failed;
                });
            const bool all_decoders_waiting = std::all_of(
                active.begin(), active.end(), [](const OfflineRequest& request) {
                    return request.phase != OfflineRequestPhase::Decoding ||
                           request.waiting_for_kv_block;
                });
            if (!has_releasable && all_decoders_waiting) {
                printf("[OfflineBatch] ERROR: paged KV pool exhausted during decode\n");
                for (auto& request : active) {
                    if (request.waiting_for_kv_block) {
                        request.phase = OfflineRequestPhase::Failed;
                    }
                }
            }
        }

        for (size_t i = active.size(); i > 0; i--) {
            size_t idx = i - 1;
            if (active[idx].phase == OfflineRequestPhase::Finished ||
                active[idx].phase == OfflineRequestPhase::Failed) {
                emit_finished(active[idx]);
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(idx));
            }
        }

        admit_round();

        if (decode_inputs.empty() && active.empty() && next_request < chunks.size()) {
            if (admit_round() == 0) {
                break;
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const OfflineChunkResult& a, const OfflineChunkResult& b) {
                  return a.id < b.id;
              });
    if (prefix_cache.ready && paged_pool) {
        paged_pool->release(prefix_cache.state);
        prefix_cache.ready = false;
    }
    if (paged_pool) {
        last_stats_.kv_ownership_errors = paged_pool->ownership_errors();
        last_stats_.kv_final_free_blocks = paged_pool->free_count();
    }
    last_stats_.sync_fallback_total();
    return results;
}

OfflineChunkResult OfflineBatchTranscriber::run_one(
    const std::vector<float>& samples,
    int sample_rate,
    const OfflineRequest& request,
    const AudioChunk& chunk,
    const OfflineBatchConfig& config,
    int request_id) {
    (void)sample_rate;

    OfflineChunkResult result;
    result.id = request_id;
    result.slot_id = request.kv.slot_id;
    result.start_sec = chunk.start_sec;
    result.end_sec = chunk.end_sec;

    size_t begin = std::min(chunk.infer_start_sample, samples.size());
    size_t end = std::min(chunk.infer_end_sample, samples.size());
    if (end <= begin) {
        return result;
    }

    InferenceConfig inference;
    inference.max_new_tokens = config.max_tokens;
    inference.n_threads = config.n_threads;
    inference.kv_cache_size = config.ctx_size;
    inference.use_gpu = config.use_gpu;
    inference.gpu_id = config.gpu_id;
    inference.prompt = config.prompt;

    PreparedLLMInput prepared = recognizer_.prepare_llm_input(
        samples.data() + begin,
        end - begin,
        inference);
    InferenceResult asr = recognizer_.run_prepared(prepared, inference);

    result.text = asr.text;
    result.ok = !asr.text.empty();
    result.encoder_ms = asr.encoder_ms;
    result.prefill_ms = asr.prefill_ms;
    result.decode_ms = asr.decode_ms;
    result.total_ms = asr.total_ms;
    result.prefill_tokens = asr.prefill_tokens;
    result.decode_tokens = asr.decode_tokens;
    return result;
}

} // namespace funasr
