// Audio Adaptor 前向计算
//
// 数据流:
//   [512, T] → linear1 + ReLU (512→2048)
//           → linear2 (2048→1024)
//           → 2 × attention block (标准MHA 8heads, FFN 中间维256)
//           → [1024, T]
//
// 注意:
//   - LayerNorm eps = 1e-12 (不是 encoder 的 1e-5)
//   - FFN 中间维度 = 256 (不是 2048)
//   - 无 FSMN, 无 causal mask
//
#ifndef FUNASR_COMPUTE_ADAPTOR_OPS_HPP
#define FUNASR_COMPUTE_ADAPTOR_OPS_HPP

#include <ggml.h>
#include "core/config.hpp"
#include "model/weights.hpp"

namespace funasr {

// 标准 Multi-head Attention（无 FSMN，无 causal mask）
ggml_tensor* adaptor_attention_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorBlockWeights& block,
    const AdaptorConfig& cfg
);

// Adaptor 单层 block (norm → attn → residual → norm → FFN → residual)
ggml_tensor* adaptor_block_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorBlockWeights& block,
    const AdaptorConfig& cfg
);

// 完整 Adaptor forward
// 输入: [512, T]  (encoder 输出)
// 输出: [1024, T] (LLM embedding 空间)
ggml_tensor* adaptor_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorWeights& weights,
    const AdaptorConfig& cfg
);

} // namespace funasr

#endif // FUNASR_COMPUTE_ADAPTOR_OPS_HPP