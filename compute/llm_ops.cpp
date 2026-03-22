// funasr/compute/llm_ops.cpp
// LLM Decoder 前向计算实现
//
// Qwen3-0.6B 架构:
//   - GQA: 16 Q heads, 8 KV heads (group=2)
//   - RoPE: freq_base=1e6, mode=2 (neox), head_dim=128
//   - QK Norm: RMSNorm on Q/K after projection
//   - SwiGLU: gate(silu) * up → down
//   - KV Cache: 拼接法 (concat 历史 + 当前, 图外 memcpy commit)
//
#include "compute/llm_ops.hpp"
#include <cmath>
#include <cstdio>

namespace funasr {

// ============================================================
// 辅助: RMSNorm
// ============================================================
static ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps = 1e-5f
) {
    ggml_tensor* normed = ggml_rms_norm(ctx, x, eps);
    ggml_tensor* w = ggml_cast(ctx, weight, GGML_TYPE_F32);
    return ggml_mul(ctx, normed, w);
}

// ============================================================
// GQA Attention with KV Cache (拼接法)
//
// 流程:
//   1. Q/K/V 投影 (无 bias)
//   2. Reshape 3D
//   3. Q/K RMSNorm + weight
//   4. RoPE (ggml_rope_ext)
//   5. K/V → 2D → pending (图外 commit)
//   6. 构建完整 K/V (concat 历史 + 当前)
//   7. GQA expansion (8 → 16 via ggml_repeat)
//   8. Scaled Dot-Product Attention + Causal Mask
//   9. Output projection
// ============================================================
ggml_tensor* gqa_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const LLMLayerWeights& layer,
    KVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg
) {
    const int seq_len    = static_cast<int>(x->ne[1]);
    const int n_heads    = cfg.head_count;       // 16
    const int n_kv_heads = cfg.head_count_kv;    // 8
    const int head_dim   = cfg.head_dim();       // 128
    const int kv_dim     = cfg.kv_dim();         // 1024
    const int n_kv       = n_past + seq_len;     // 总 KV 长度
    const float eps      = 1e-5f;

    // ===== 1. Q/K/V 投影 (无 bias) =====
    ggml_tensor* q     = ggml_mul_mat(ctx, layer.q_proj_w, x);
    ggml_tensor* k_cur = ggml_mul_mat(ctx, layer.k_proj_w, x);
    ggml_tensor* v_cur = ggml_mul_mat(ctx, layer.v_proj_w, x);

    // ===== 2. Reshape 3D =====
    q     = ggml_reshape_3d(ctx, q,     head_dim, n_heads,    seq_len);
    k_cur = ggml_reshape_3d(ctx, k_cur, head_dim, n_kv_heads, seq_len);
    v_cur = ggml_reshape_3d(ctx, v_cur, head_dim, n_kv_heads, seq_len);

    // ===== 3. Q/K RMSNorm =====
    q     = rms_norm(ctx, q,     layer.q_norm_w, eps);
    k_cur = rms_norm(ctx, k_cur, layer.k_norm_w, eps);

    // ===== 4. RoPE =====
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    if (pos->data) {
        int* pos_data = (int*)pos->data;
        for (int i = 0; i < seq_len; i++) {
            pos_data[i] = n_past + i;
        }
    }

    q     = ggml_rope_ext(ctx, q,     pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx, k_cur, pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // ===== 5. K/V → 2D [kv_dim, seq_len] → pending =====
    k_cur = ggml_reshape_2d(ctx, k_cur, kv_dim, seq_len);
    k_cur = ggml_cont(ctx, k_cur);
    v_cur = ggml_reshape_2d(ctx, v_cur, kv_dim, seq_len);
    v_cur = ggml_cont(ctx, v_cur);

    cache.set_pending(layer_idx, k_cur, v_cur);

    // ===== 6. 构建完整 K/V =====
    ggml_tensor* k_full;
    ggml_tensor* v_full;

    if (n_past > 0) {
        size_t layer_offset = static_cast<size_t>(layer_idx) * cache.n_ctx() * kv_dim * sizeof(float);

        ggml_tensor* k_hist = ggml_view_2d(ctx, cache.k_tensor(),
            kv_dim, n_past, kv_dim * sizeof(float), layer_offset);
        ggml_tensor* v_hist = ggml_view_2d(ctx, cache.v_tensor(),
            kv_dim, n_past, kv_dim * sizeof(float), layer_offset);

        k_full = ggml_concat(ctx, k_hist, k_cur, 1);
        v_full = ggml_concat(ctx, v_hist, v_cur, 1);
    } else {
        k_full = k_cur;
        v_full = v_cur;
    }

    // ===== 7. Reshape 3D + Permute for attention =====
    // K/V: [kv_dim, n_kv] → [head_dim, n_kv_heads, n_kv] → permute → [head_dim, n_kv, n_kv_heads]
    k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv_heads, n_kv);
    k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 2, 1, 3));

    v_full = ggml_reshape_3d(ctx, v_full, head_dim, n_kv_heads, n_kv);
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 2, 1, 3));

    // Q: [head_dim, n_heads, seq_len] → permute → [head_dim, seq_len, n_heads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // ===== 8. GQA Expansion (8 KV → 16 Q heads) =====
    if (n_kv_heads < n_heads) {
        int repeat_factor = n_heads / n_kv_heads;  // 2

        // K: [head_dim, n_kv, n_kv_heads] → repeat → [head_dim, n_kv, n_heads]
        k_full = ggml_reshape_4d(ctx, k_full, head_dim, n_kv, n_kv_heads, 1);
        ggml_tensor* k_target = ggml_new_tensor_4d(ctx, k_full->type,
            head_dim, n_kv, n_kv_heads, repeat_factor);
        k_full = ggml_repeat(ctx, k_full, k_target);
        k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 1, 3, 2));
        k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv, n_heads);
        k_full = ggml_cont(ctx, k_full);

        // V: same
        v_full = ggml_reshape_4d(ctx, v_full, head_dim, n_kv, n_kv_heads, 1);
        ggml_tensor* v_target = ggml_new_tensor_4d(ctx, v_full->type,
            head_dim, n_kv, n_kv_heads, repeat_factor);
        v_full = ggml_repeat(ctx, v_full, v_target);
        v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 1, 3, 2));
        v_full = ggml_reshape_3d(ctx, v_full, head_dim, n_kv, n_heads);
        v_full = ggml_cont(ctx, v_full);
    }

    // ===== 9. Attention =====
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    q = ggml_scale(ctx, q, scale);

    // scores = K^T @ Q → [n_kv, seq_len, n_heads]
    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q);

    // Causal mask
    scores = ggml_diag_mask_inf(ctx, scores, n_past);

    // Softmax
    ggml_tensor* attn = ggml_soft_max(ctx, scores);

    // V: [head_dim, n_kv, n_heads] → permute → [n_kv, head_dim, n_heads]
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));

    // attn_out = V^T @ attn → [head_dim, seq_len, n_heads]
    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_full, attn);

    // ===== 10. Output projection =====
    // [head_dim, seq_len, n_heads] → permute → [head_dim, n_heads, seq_len] → reshape [q_dim, seq_len]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, seq_len);

    return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
}

// ============================================================
// Single LLM Layer
// ============================================================
ggml_tensor* llm_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const LLMLayerWeights& layer,
    KVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg
) {
    const float eps = 1e-5f;

    // ===== Attention Block =====
    ggml_tensor* residual = x;
    ggml_tensor* x_norm = rms_norm(ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gqa_forward(ctx, x_norm, layer, cache, layer_idx, n_past, cfg);
    x = ggml_add(ctx, residual, attn_out);

    // ===== MLP Block (SwiGLU) =====
    residual = x;
    x_norm = rms_norm(ctx, x, layer.post_attn_norm_w, eps);

    ggml_tensor* gate = ggml_mul_mat(ctx, layer.gate_proj_w, x_norm);
    ggml_tensor* up   = ggml_mul_mat(ctx, layer.up_proj_w,   x_norm);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* hidden = ggml_mul(ctx, gate, up);
    ggml_tensor* mlp_out = ggml_mul_mat(ctx, layer.down_proj_w, hidden);

    x = ggml_add(ctx, residual, mlp_out);
    return x;
}

// ============================================================
// Full LLM Decoder Forward
// ============================================================
ggml_tensor* llm_forward(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const LLMWeights& weights,
    KVCache& cache,
    int n_past,
    const LLMConfig& cfg
) {
    const float eps = 1e-5f;
    ggml_tensor* x = hidden_states;

    // 28 Layers
    for (int i = 0; i < cfg.block_count; i++) {
        x = llm_layer_forward(ctx, x, weights.layers[i], cache, i, n_past, cfg);
    }

    // Final RMSNorm
    x = rms_norm(ctx, x, weights.model_norm_w, eps);

    // LM Head
    ggml_tensor* logits = ggml_mul_mat(ctx, weights.lm_head_w, x);

    return logits;
}

} // namespace funasr