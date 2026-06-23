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
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace funasr {

class Recognizer;

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

    int capacity() const override { return capacity_; }
    int free_count() const override { return static_cast<int>(free_blocks_.size()); }
    int block_size() const { return block_size_; }

private:
    int blocks_needed(int max_tokens) const;

    int capacity_ = 0;
    int block_size_ = 64;
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
};

struct OfflineChunkResult {
    int id = -1;
    int slot_id = -1;
    float start_sec = 0.0f;
    float end_sec = 0.0f;
    std::string text;
    bool ok = false;

    float encoder_ms = 0.0f;
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    float total_ms = 0.0f;
    int prefill_tokens = 0;
    int decode_tokens = 0;
};

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
    double prefill_wall_ms = 0.0;
    double decode_dispatch_wall_ms = 0.0;
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
