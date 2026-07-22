// funasr/pipeline/offline_batching.hpp
// Offline transcription scheduler scaffolding.
//
// This module is intentionally separate from Pipeline::transcribe_audio().
// The realtime/single-chunk path remains unchanged. OfflineBatchTranscriber
// starts as a serial scheduler with vLLM-style request/KV-slot boundaries, so
// later continuous batching can be added without reshaping the CLI again.
#ifndef FUNASR_PIPELINE_OFFLINE_BATCHING_HPP
#define FUNASR_PIPELINE_OFFLINE_BATCHING_HPP

#include "pipeline/chunking.hpp"
#include "pipeline/pipeline.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace funasr {

class Recognizer;

int paged_blocks_for_tokens(int tokens, int block_size);
int paged_prefix_partial_rows(int prefix_tokens, int block_size);
bool paged_decode_requires_append(int n_past,
                                  size_t allocated_blocks,
                                  int block_size);
std::vector<std::pair<int, int>> plan_prefill_chunks(
    int first_uncached_token,
    int prompt_tokens,
    int max_chunk_tokens);

enum class OfflineRequestPhase {
    WaitingPrefill,
    Prefilling,
    Decoding,
    Finished,
    Failed,
};

struct OfflineBatchConfig {
    int batch_size = 1;
    int ctx_size = 2048;
    int max_tokens = 100;
    int n_threads = 4;
    bool use_gpu = false;
    int gpu_id = 0;
    bool use_paged_kv = false;
    int kv_block_size = 64;
    int kv_num_blocks = 0;
    PromptOptions prompt;
    bool enable_prefix_kv_cache = false;
    bool enable_dynamic_kv_blocks = false;
    int prefill_chunk_tokens = 0;
    bool use_unified_scheduler = false;
    int max_num_scheduled_tokens = 1024;
    int max_prefill_chunk_tokens = 512;
    int max_frontend_requests_per_step = 4;
    bool enable_frontend_batching = true;
    bool enable_frontend_prefetch = true;
    bool enable_gpu_frontend_overlap = false;
    int frontend_bucket_window = 16;
    int mixed_graph_cache_entries = 16;
};

int resolve_paged_kv_num_blocks(const OfflineBatchConfig& config);
int resolve_paged_kv_physical_rows(const OfflineBatchConfig& config);

// Minimum private capacity needed before a prepared request may enter the
// unified scheduler. A partial shared prefix needs one COW block; an aligned
// prefix (or no prefix) needs the first append block for the remaining prompt.
int paged_initial_admission_blocks(int prompt_tokens,
                                   int cached_prefix_tokens,
                                   int block_size);

std::vector<size_t> plan_frontend_request_order(
    const std::vector<AudioChunk>& chunks,
    int frontend_batch_size,
    int bucket_window);

struct OfflineAutoTuneInput {
    uint64_t gpu_free_bytes = 0;
    uint64_t gpu_total_bytes = 0;
    uint64_t model_file_bytes = 0;
    uint64_t kv_bytes_per_block = 0;
    int chunk_seconds = 30;
    int max_tokens = 220;
    int block_size = 128;
};

struct OfflineAutoTuneResult {
    int batch_size = 12;
    int kv_num_blocks = 192;
    int max_frontend_requests = 4;
    bool used_memory_probe = false;
};

OfflineAutoTuneResult tune_offline_runtime(
    const OfflineAutoTuneInput& input);

class PagedPrefillAdmissionHints {
public:
    bool can_prepare(size_t sample_count, int free_blocks) const;
    void observe(size_t sample_count, int required_blocks);
    int required_blocks(size_t sample_count) const;

private:
    std::unordered_map<size_t, int> required_blocks_by_sample_count_;
};

struct KVHandle {
    int request_id = -1;
    int slot_id = -1;
    int ctx_size = 0;
    int block_size = 0;
    std::vector<int> block_table;

    bool valid() const { return slot_id >= 0 || !block_table.empty(); }
    bool paged() const { return !block_table.empty(); }
};

struct PromptCacheKey {
    std::vector<int> prefix_token_ids;
    int block_size = 0;
    uint64_t model_generation = 0;

    bool operator==(const PromptCacheKey& other) const {
        return block_size == other.block_size &&
               model_generation == other.model_generation &&
               prefix_token_ids == other.prefix_token_ids;
    }
};

struct TaskPrefixKVCache {
    PromptCacheKey key;
    KVHandle state;
    int prefix_tokens = 0;
    bool ready = false;
};

class KVPool {
public:
    virtual ~KVPool() = default;
    virtual bool has_capacity(int max_tokens) const = 0;
    virtual KVHandle acquire(int request_id, int max_tokens) = 0;
    virtual void release(const KVHandle& handle) = 0;
    virtual int free_count() const = 0;
    virtual int capacity() const = 0;
};

class ContinuousKVSlotPool : public KVPool {
public:
    ContinuousKVSlotPool() = default;
    ContinuousKVSlotPool(int num_slots, int ctx_size) { reset(num_slots, ctx_size); }

    void reset(int num_slots, int ctx_size);
    bool has_free_slot() const;
    bool has_capacity(int max_tokens) const override;
    KVHandle acquire(int request_id);
    KVHandle acquire(int request_id, int max_tokens) override;
    void release(const KVHandle& handle) override;

    int capacity() const override { return capacity_; }
    int free_count() const override { return static_cast<int>(free_slots_.size()); }
    int ctx_size() const { return ctx_size_; }

private:
    int capacity_ = 0;
    int ctx_size_ = 0;
    std::vector<bool> in_use_;
    std::queue<int> free_slots_;
};

class PagedKVBlockPool : public KVPool {
public:
    PagedKVBlockPool() = default;
    PagedKVBlockPool(int num_blocks, int block_size) { reset(num_blocks, block_size); }

    void reset(int num_blocks, int block_size);
    bool has_capacity(int max_tokens) const override;
    KVHandle acquire(int request_id, int max_tokens) override;
    void release(const KVHandle& handle) override;
    int allocate_block();
    bool append_block(KVHandle& handle);
    bool retain_block(int block_id);
    bool retain(const KVHandle& handle);
    bool release_block(int block_id);
    bool replace_block_after_cow(KVHandle& handle,
                                 size_t logical_block,
                                 int replacement_block);

    int capacity() const override { return capacity_; }
    int free_count() const override { return static_cast<int>(free_blocks_.size()); }
    int block_size() const { return block_size_; }
    int ref_count(int block_id) const;
    int ownership_errors() const { return ownership_errors_; }

private:
    int blocks_needed(int max_tokens) const;

    int capacity_ = 0;
    int block_size_ = 64;
    int ownership_errors_ = 0;
    std::vector<int> ref_counts_;
    std::vector<bool> in_free_queue_;
    std::queue<int> free_blocks_;
};

struct OfflineRequest {
    int id = -1;
    OfflineRequestPhase phase = OfflineRequestPhase::WaitingPrefill;
    KVHandle kv;
    int n_past = 0;
    int next_token = -1;
    int max_tokens = 0;
    std::vector<int> tokens;
    std::vector<float> logits;
    std::string text;
    float start_sec = 0.0f;
    float end_sec = 0.0f;

    float encoder_ms = 0.0f;
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    int prefill_tokens = 0;
    bool waiting_for_kv_block = false;
};

struct OfflineChunkResult {
    int id = -1;
    int slot_id = -1;
    float start_sec = 0.0f;
    float end_sec = 0.0f;
    std::string text;
    bool ok = false;
    std::vector<int> token_ids;

    float encoder_ms = 0.0f;
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    float total_ms = 0.0f;
    int prefill_tokens = 0;
    int decode_tokens = 0;
};

void copy_offline_result_tokens(
    const OfflineRequest& request,
    OfflineChunkResult& result);

struct OfflineDecodeFallbackStats {
    int single_request = 0;
    int token_id_fast_path_unavailable = 0;
    int host_embedding_batch_unavailable = 0;
    int serial_env_forced = 0;
    int invalid_paged_input = 0;

    int total() const {
        return single_request +
               token_id_fast_path_unavailable +
               host_embedding_batch_unavailable +
               serial_env_forced +
               invalid_paged_input;
    }
};

struct OfflineMixedGraphShapeStat {
    uint64_t shape_hash = 0;
    long long calls = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int total_tokens = 0;
    int max_blocks = 0;
    int max_n_kv = 0;
    int prefill_sequences = 0;
};

struct OfflineBatchStats {
    int total_chunks = 0;
    int batch_size = 1;
    bool use_gpu = false;
    bool use_paged_kv = false;
    int kv_block_size = 0;
    int kv_block_capacity = 0;
    int peak_blocks_in_use = 0;
    int decode_steps = 0;
    int decode_group_calls = 0;
    int decode_fallback_calls = 0;
    int admit_attempts = 0;
    int admit_success = 0;
    int admit_no_kv_capacity = 0;
    int admit_rounds = 0;
    int admit_round_requests = 0;
    int max_admit_round_size = 0;
    int scheduler_idle_steps = 0;
    long long active_batch_sum = 0;
    long long paged_kv_released_requests = 0;
    long long kv_allocated_blocks_total = 0;
    long long kv_used_blocks_total = 0;
    long long kv_wasted_blocks_total = 0;
    long long kv_allocated_positions_total = 0;
    long long kv_used_positions_total = 0;
    long long kv_wasted_positions_total = 0;
    int kv_max_wasted_blocks = 0;
    int prefix_cache_builds = 0;
    int prefix_cache_hits = 0;
    long long prefix_tokens_reused = 0;
    long long prefix_prefill_tokens_saved = 0;
    int prefix_cow_copies = 0;
    int dynamic_prefill_blocks = 0;
    int dynamic_decode_appends = 0;
    int kv_ownership_errors = 0;
    int kv_final_free_blocks = 0;
    double prefill_wall_ms = 0.0;
    double decode_dispatch_wall_ms = 0.0;
    int unified_steps = 0;
    int unified_mixed_steps = 0;
    int unified_prefill_only_steps = 0;
    int unified_decode_only_steps = 0;
    long long unified_scheduled_tokens = 0;
    long long unified_selected_rows = 0;
    double unified_staging_ms = 0.0;
    double unified_compute_ms = 0.0;
    long long unified_graph_cache_hits = 0;
    long long unified_graph_cache_misses = 0;
    int frontend_batch_calls = 0;
    int frontend_batched_requests = 0;
    int frontend_single_requests = 0;
    int frontend_batch_fallbacks = 0;
    int frontend_prepared_queue_peak = 0;
    int frontend_prefetch_launches = 0;
    int frontend_prefetch_ready_hits = 0;
    long long frontend_input_frames = 0;
    long long frontend_padded_frames = 0;
    double frontend_fbank_ms = 0.0;
    double frontend_prefetch_wait_ms = 0.0;
    double frontend_batch_wall_ms = 0.0;
    double frontend_single_wall_ms = 0.0;
    long long unified_graph_cache_evictions = 0;
    int unified_graph_cache_entries_peak = 0;
    int unified_graph_cache_capacity = 0;
    std::unordered_map<uint64_t, OfflineMixedGraphShapeStat>
        unified_graph_shapes;
    OfflineDecodeFallbackStats fallback_reasons;

    double average_active_batch() const {
        return decode_steps > 0
            ? static_cast<double>(active_batch_sum) / static_cast<double>(decode_steps)
            : 0.0;
    }

    double average_prefill_wall_ms() const {
        return admit_success > 0
            ? prefill_wall_ms / static_cast<double>(admit_success)
            : 0.0;
    }

    double average_decode_dispatch_ms() const {
        return decode_steps > 0
            ? decode_dispatch_wall_ms / static_cast<double>(decode_steps)
            : 0.0;
    }

    double average_admit_round_size() const {
        return admit_rounds > 0
            ? static_cast<double>(admit_round_requests) / static_cast<double>(admit_rounds)
            : 0.0;
    }

    double average_allocated_blocks() const {
        return paged_kv_released_requests > 0
            ? static_cast<double>(kv_allocated_blocks_total) /
                  static_cast<double>(paged_kv_released_requests)
            : 0.0;
    }

    double average_used_blocks() const {
        return paged_kv_released_requests > 0
            ? static_cast<double>(kv_used_blocks_total) /
                  static_cast<double>(paged_kv_released_requests)
            : 0.0;
    }

    double average_wasted_blocks() const {
        return paged_kv_released_requests > 0
            ? static_cast<double>(kv_wasted_blocks_total) /
                  static_cast<double>(paged_kv_released_requests)
            : 0.0;
    }

    double kv_block_waste_rate() const {
        return kv_allocated_blocks_total > 0
            ? 100.0 * static_cast<double>(kv_wasted_blocks_total) /
                  static_cast<double>(kv_allocated_blocks_total)
            : 0.0;
    }

    void sync_fallback_total() {
        decode_fallback_calls = fallback_reasons.total();
    }
};

using OfflineProgressCallback = std::function<void(
    int completed, int total, const OfflineChunkResult& result)>;

class OfflineBatchTranscriber {
public:
    explicit OfflineBatchTranscriber(Recognizer& recognizer);

    const OfflineBatchStats& last_stats() const { return last_stats_; }

    std::vector<OfflineChunkResult> transcribe(
        const std::vector<float>& samples,
        int sample_rate,
        const std::vector<AudioChunk>& chunks,
        const OfflineBatchConfig& config,
        OfflineProgressCallback progress = nullptr);

private:
    std::vector<OfflineChunkResult> transcribe_continuous_gpu(
        const std::vector<float>& samples,
        int sample_rate,
        const std::vector<AudioChunk>& chunks,
        const OfflineBatchConfig& config,
        OfflineProgressCallback progress);

    std::vector<OfflineChunkResult> transcribe_unified_gpu(
        const std::vector<float>& samples,
        int sample_rate,
        const std::vector<AudioChunk>& chunks,
        const OfflineBatchConfig& config,
        OfflineProgressCallback progress);

    OfflineChunkResult run_one(
        const std::vector<float>& samples,
        int sample_rate,
        const OfflineRequest& request,
        const AudioChunk& chunk,
        const OfflineBatchConfig& config,
        int request_id);

    void reset_stats(size_t total_chunks, const OfflineBatchConfig& config);
    void update_peak_blocks(const KVPool& kv_pool);

    Recognizer& recognizer_;
    OfflineBatchStats last_stats_;
};

} // namespace funasr

#endif // FUNASR_PIPELINE_OFFLINE_BATCHING_HPP
