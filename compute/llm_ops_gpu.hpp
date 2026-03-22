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

} // namespace funasr

#endif // FUNASR_COMPUTE_LLM_OPS_GPU_HPP