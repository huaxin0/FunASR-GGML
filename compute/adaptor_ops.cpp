// Audio Adaptor 前向计算实现
#include "compute/adaptor_ops.hpp"
#include <cmath>
#include <cstdio>

namespace funasr {

// ============================================================
// 辅助: LayerNorm (eps = 1e-12 for adaptor)
// ============================================================
static ggml_tensor* adaptor_layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    float eps = 1e-12f
) {
    ggml_tensor* w = ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor* b = ggml_cast(ctx, bias,   GGML_TYPE_F32);
    ggml_tensor* normed = ggml_norm(ctx, x, eps);
    normed = ggml_mul(ctx, normed, w);
    normed = ggml_add(ctx, normed, b);
    return normed;
}

// ============================================================
// 辅助: 线性层 + bias
// ============================================================
static ggml_tensor* linear_with_bias(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias
) {
    ggml_tensor* out = ggml_mul_mat(ctx, weight, x);
    ggml_tensor* b   = ggml_cast(ctx, bias, GGML_TYPE_F32);
    return ggml_add(ctx, out, b);
}

// ============================================================
// 标准 Multi-head Attention
//
// 与 SANM 的区别: 没有 FSMN 记忆
// Q/K/V permute 方式完全相同
//
// 维度排列:
//   Q/K: [dim, T] → [head_dim, n_heads, T] → permute(0,2,1) → [head_dim, T, n_heads]
//   V:   [dim, T] → [head_dim, n_heads, T] → permute(1,2,0) → [T, head_dim, n_heads]
// ============================================================
ggml_tensor* adaptor_attention_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorBlockWeights& block,
    const AdaptorConfig& cfg
) {
    const int T       = static_cast<int>(x->ne[1]);
    const int dim     = static_cast<int>(x->ne[0]);  // 1024
    const int n_heads = cfg.attention_heads;           // 8
    const int head_dim = dim / n_heads;                // 128

    // Q/K/V 投影
    ggml_tensor* q = linear_with_bias(ctx, x, block.attn_q_w, block.attn_q_b);
    ggml_tensor* k = linear_with_bias(ctx, x, block.attn_k_w, block.attn_k_b);
    ggml_tensor* v = linear_with_bias(ctx, x, block.attn_v_w, block.attn_v_b);

    // Multi-head reshape + permute
    // Q/K: [dim, T] → [head_dim, n_heads, T] → [head_dim, T, n_heads]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // V: [dim, T] → [head_dim, n_heads, T] → [T, head_dim, n_heads]
    ggml_tensor* v_3d   = ggml_reshape_3d(ctx, v, head_dim, n_heads, T);
    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_3d, 1, 2, 0, 3));

    // Scaled dot-product attention (无 causal mask)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    q = ggml_scale(ctx, q, scale);

    ggml_tensor* scores  = ggml_mul_mat(ctx, k, q);       // [T, T, n_heads]
    ggml_tensor* attn    = ggml_soft_max(ctx, scores);
    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, attn); // [head_dim, T, n_heads]

    // 还原: [head_dim, T, n_heads] → [head_dim, n_heads, T] → [dim, T]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, dim, T);

    // 输出投影
    attn_out = linear_with_bias(ctx, attn_out, block.attn_out_w, block.attn_out_b);

    return attn_out;
}

// ============================================================
// Adaptor Block
//
// norm1(eps=1e-12) → attention → residual
// norm2(eps=1e-12) → FFN(ReLU, 中间维256) → residual
// ============================================================
ggml_tensor* adaptor_block_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorBlockWeights& block,
    const AdaptorConfig& cfg
) {
    // Part 1: norm1 → attention → residual
    ggml_tensor* residual = x;
    ggml_tensor* x1 = adaptor_layer_norm(ctx, x, block.norm1_w, block.norm1_b);
    ggml_tensor* attn_out = adaptor_attention_forward(ctx, x1, block, cfg);
    x = ggml_add(ctx, residual, attn_out);

    // Part 2: norm2 → FFN(ReLU) → residual
    residual = x;
    ggml_tensor* x2 = adaptor_layer_norm(ctx, x, block.norm2_w, block.norm2_b);

    ggml_tensor* ffn = linear_with_bias(ctx, x2, block.ffn_w1, block.ffn_b1);
    ffn = ggml_relu(ctx, ffn);
    ffn = linear_with_bias(ctx, ffn, block.ffn_w2, block.ffn_b2);

    x = ggml_add(ctx, residual, ffn);
    return x;
}

// ============================================================
// 完整 Adaptor Forward
//
// linear1(512→2048) + ReLU → linear2(2048→1024) → 2 blocks
// ============================================================
ggml_tensor* adaptor_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const AdaptorWeights& weights,
    const AdaptorConfig& cfg
) {
    // 升维: 512 → 2048 → 1024
    x = linear_with_bias(ctx, x, weights.linear1_w, weights.linear1_b);
    x = ggml_relu(ctx, x);
    x = linear_with_bias(ctx, x, weights.linear2_w, weights.linear2_b);

    // 2 × attention block
    for (int i = 0; i < cfg.num_blocks; i++) {
        x = adaptor_block_forward(ctx, x, weights.blocks[i], cfg);
    }

    return x;
}

} // namespace funasr