#ifndef FUNASR_COMPUTE_GPU_MIXED_RUNNER_HPP
#define FUNASR_COMPUTE_GPU_MIXED_RUNNER_HPP

#include "compute/gpu_embedding_pool.hpp"
#include "pipeline/mixed_batch.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace funasr {

class GPUContext;

struct PackedPromptSlice {
    GPUEmbeddingHandle source;
    int source_token_offset = 0;
    int token_count = 0;
    int destination_token_offset = 0;
};

struct MixedGraphExecutionInfo {
    bool cache_hit = false;
    uint64_t shape_hash = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int total_tokens = 0;
    int max_blocks = 0;
    int max_n_kv = 0;
    int prefill_sequences = 0;
    int cache_entries = 0;
    int cache_capacity = 0;
    uint64_t cache_evictions = 0;
};

class GPUMixedRunner {
public:
    static constexpr int kMaxPackedTokens = 1 << 20;

    GPUMixedRunner(ggml_backend_t backend, int embedding_dim);
    ~GPUMixedRunner();

    GPUMixedRunner(const GPUMixedRunner&) = delete;
    GPUMixedRunner& operator=(const GPUMixedRunner&) = delete;

    bool stage_inputs(
        const GPUEmbeddingPool& pool,
        const std::vector<PackedPromptSlice>& prompt_slices,
        int prefill_tokens,
        const std::vector<int>& decode_token_ids);

    bool execute_paged(
        GPUContext& gpu_context,
        const PackedMixedMetadata& metadata,
        int block_size,
        std::vector<int>& next_tokens,
        MixedGraphExecutionInfo* graph_info = nullptr);

    void set_graph_cache_capacity(int capacity);
    int graph_cache_capacity() const;
    int graph_cache_entries() const;

    ggml_tensor* prompt_staging() const;
    int prefill_tokens() const;
    int decode_tokens() const;
    int total_tokens() const;
    const std::vector<int>& decode_token_ids() const;
    int capacity_tokens() const;
    size_t reserved_bytes() const;
    bool ready() const;

private:
    bool ensure_capacity(int requested_tokens);
    bool validate_inputs(
        const GPUEmbeddingPool& pool,
        const std::vector<PackedPromptSlice>& prompt_slices,
        int prefill_tokens,
        const std::vector<int>& decode_token_ids) const;
    void clear_batch_state();
    void clear_graph_cache();

    struct GraphSignature {
        int prefill_tokens = 0;
        int decode_tokens = 0;
        int total_tokens = 0;
        int max_blocks = 0;
        int block_size = 0;
        int max_n_kv = 0;
        int selected_rows = 0;
        std::vector<int> prefill_layout;

        bool operator==(const GraphSignature& other) const;
    };

    struct CachedGraph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* decode_ids_input = nullptr;
        ggml_tensor* block_table_input = nullptr;
        ggml_tensor* position_input = nullptr;
        ggml_tensor* kv_lens_input = nullptr;
        ggml_tensor* selected_rows_input = nullptr;
        std::vector<ggml_tensor*> prefill_past_row_inputs;
        std::vector<ggml_tensor*> prefill_mask_inputs;
        ggml_tensor* token = nullptr;
        GraphSignature signature;
        const void* cuda_graph_key = nullptr;
        size_t allocator_generation = 0;
        bool valid = false;
    };

    static uint64_t hash_signature(const GraphSignature& signature);
    void destroy_cached_graph(CachedGraph& cached);
    void evict_lru_graph();

    ggml_backend_t backend_ = nullptr;
    int embedding_dim_ = 0;
    ggml_context* staging_ctx_ = nullptr;
    ggml_backend_buffer_t staging_buffer_ = nullptr;
    ggml_tensor* staging_tensor_ = nullptr;
    ggml_gallocr_t graph_allocator_ = nullptr;
    int capacity_tokens_ = 0;
    int prefill_tokens_ = 0;
    int decode_tokens_ = 0;
    int total_tokens_ = 0;
    std::vector<int> decode_token_ids_;
    bool ready_ = false;
    std::list<CachedGraph> graph_cache_;
    std::unordered_map<uint64_t, uint32_t> graph_shape_observations_;
    int graph_cache_capacity_ = 8;
    uint64_t graph_cache_evictions_ = 0;
};

}  // namespace funasr

#endif  // FUNASR_COMPUTE_GPU_MIXED_RUNNER_HPP
