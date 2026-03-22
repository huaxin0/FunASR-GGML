// funasr/compute/llm_ops.hpp
// LLM Decoder 前向计算 (Qwen3-0.6B with GQA + KV Cache)
//
// 数据流:
//   hidden_states [1024, seq_len]
//     → 28 × Transformer Layer:
//         RMSNorm → GQA (16Q/8KV, RoPE, CausalMask, KVCache) → Residual
//         RMSNorm → SwiGLU MLP (1024→3072→1024) → Residual
//     → Final RMSNorm
//     → LM Head (1024 → 151936)
//     → logits [151936, seq_len]
//
#ifndef FUNASR_COMPUTE_LLM_OPS_HPP
#define FUNASR_COMPUTE_LLM_OPS_HPP

#include <ggml.h>
#include "core/config.hpp"
#include "model/weights.hpp"
#include "compute/kv_cache.hpp"

namespace funasr {

// GQA Attention with KV Cache (拼接法)
ggml_tensor* gqa_forward(
    ggml_context* ctx,
    ggml_tensor* x,              // [1024, seq_len]
    const LLMLayerWeights& layer,
    KVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg
);

// 单层 Transformer (RMSNorm→GQA→Residual + RMSNorm→SwiGLU→Residual)
ggml_tensor* llm_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const LLMLayerWeights& layer,
    KVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg
);

// 完整 LLM Decoder forward
// 输入: hidden_states [1024, seq_len]
// 输出: logits [151936, seq_len]
ggml_tensor* llm_forward(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const LLMWeights& weights,
    KVCache& cache,
    int n_past,
    const LLMConfig& cfg
);

} // namespace funasr

#endif // FUNASR_COMPUTE_LLM_OPS_HPP