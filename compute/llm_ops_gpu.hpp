// GPU LLM Decoder 前向计算
//
// 与 CPU 版 (llm_ops.hpp) 的关键区别:
//   - 使用 GPULLMWeights (tensor 在 GPU 上)
//   - KV Cache 用 ggml_cpy 在图内更新（不需要图外 memcpy）
//   - 收集 kv_cpy_ops 供图构建器使用
#ifndef FUNASR_COMPUTE_LLM_OPS_GPU_HPP
#define FUNASR_COMPUTE_LLM_OPS_GPU_HPP

#include <ggml.h>
#include "core/config.hpp"
#include "compute/gpu_context.hpp"
#include <vector>

namespace funasr {

struct GPUPackedPrefillLayout {
    int row_offset = 0;
    int token_count = 0;
    int n_past = 0;
};

// GPU GQA Attention with KV Cache (ggml_cpy 方式)
ggml_tensor* gpu_gqa_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops    // 收集 cpy 节点
);

ggml_tensor* gpu_gqa_forward_slot(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    int slot_id,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
);

ggml_tensor* gpu_gqa_forward_paged(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const std::vector<int>& block_table,
    int block_size,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    std::vector<ggml_tensor*>* block_table_inputs = nullptr,
    std::vector<ggml_tensor*>* position_inputs = nullptr,
    ggml_tensor* shared_block_table_input = nullptr,
    ggml_tensor* shared_position_input = nullptr
);

// GPU 单层 Transformer
ggml_tensor* gpu_llm_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
);

ggml_tensor* gpu_llm_layer_forward_slot(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    int slot_id,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
);

ggml_tensor* gpu_llm_layer_forward_paged(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const std::vector<int>& block_table,
    int block_size,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    std::vector<ggml_tensor*>* block_table_inputs = nullptr,
    std::vector<ggml_tensor*>* position_inputs = nullptr,
    ggml_tensor* shared_block_table_input = nullptr,
    ggml_tensor* shared_position_input = nullptr
);

// GPU 完整 LLM Decoder
ggml_tensor* gpu_llm_forward(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
);

ggml_tensor* gpu_llm_forward_slot(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int n_past,
    int slot_id,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
);

ggml_tensor* gpu_llm_forward_paged(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int n_past,
    const std::vector<int>& block_table,
    int block_size,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    std::vector<ggml_tensor*>* block_table_inputs = nullptr,
    std::vector<ggml_tensor*>* position_inputs = nullptr
);

ggml_tensor* gpu_llm_forward_paged_batch_decode(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    const std::vector<int>& n_pasts,
    const std::vector<std::vector<int>>& block_tables,
    int block_size,
    int graph_max_n_kv,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    ggml_tensor** block_table_input,
    ggml_tensor** position_input,
    ggml_tensor** kv_lens_input
);

ggml_tensor* gpu_llm_forward_paged_packed(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int max_blocks,
    int block_size,
    int graph_max_n_kv,
    int selected_row_count,
    const std::vector<GPUPackedPrefillLayout>& prefill_layouts,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_write_ops,
    ggml_tensor** block_table_input,
    ggml_tensor** position_input,
    ggml_tensor** kv_lens_input,
    ggml_tensor** selected_rows_input,
    std::vector<ggml_tensor*>& prefill_past_row_inputs,
    std::vector<ggml_tensor*>& prefill_mask_inputs
);

} // namespace funasr

#endif // FUNASR_COMPUTE_LLM_OPS_GPU_HPP
