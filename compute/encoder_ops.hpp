// Audio Encoder 前向计算 (SANM + FSMN)
//
// 数据流:
//   [560, T] → scale(√output_size)
//           → encoder0 (1层, 560→512, 无attention残差)
//           → 49 × encoder_layer (512→512, 有attention残差)
//           → after_norm
//           → 20 × tp_encoder_layer (512→512, 有attention残差)
//           → tp_norm
//           → [512, T]
//
//
#ifndef FUNASR_COMPUTE_ENCODER_OPS_HPP
#define FUNASR_COMPUTE_ENCODER_OPS_HPP

#include <ggml.h>
#include "core/config.hpp"
#include "model/weights.hpp"

namespace funasr {

// ============================================================
// FSMN: Feedforward Sequential Memory Networks
// depthwise conv1d, 手动展开为 K 次 slice-multiply-accumulate
//
// 输入: v [dim, T],  fsmn_w [K, 1, dim]
// 输出:   [dim, T]
// ============================================================
ggml_tensor* fsmn_forward(
    ggml_context* ctx,
    ggml_tensor* v,
    ggml_tensor* fsmn_w,
    int kernel_size,
    int pad
);

// ============================================================
// SANM Attention (含 FSMN 记忆)
//
// 流程: Q/K/V投影 → FSMN(V) → Multi-head Attention → Output投影
//       最终 = attn_output + fsmn_memory
//
// 输入: x [dim, T]
// 输出:   [dim, T] (encoder0: dim=512, 输入经投影从560降到512)
// ============================================================
ggml_tensor* sanm_attention_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderLayerWeights& layer,
    const EncoderConfig& cfg
);

// ============================================================
// 单层 Encoder
//
// 流程:
//   residual = x
//   x = LayerNorm1(x) → SANM_Attention(x)
//   x = (has_residual ? residual + x : x)   // encoder0 无残差
//   residual = x
//   x = LayerNorm2(x) → FFN(ReLU)(x)
//   x = residual + x
//
// 输入: x [dim_in, T]  (encoder0: 560, 其他: 512)
// 输出:   [512, T]
// ============================================================
ggml_tensor* encoder_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderLayerWeights& layer,
    const EncoderConfig& cfg,
    bool has_attention_residual
);

// ============================================================
// 完整 Encoder 前向传播
//
// 输入: x [560, T]  F32 (Fbank + LFR 输出)
// 输出:   [512, T]  F32
// ============================================================
ggml_tensor* encoder_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderWeights& weights,
    const EncoderConfig& cfg
);

} // namespace funasr

#endif // FUNASR_COMPUTE_ENCODER_OPS_HPP