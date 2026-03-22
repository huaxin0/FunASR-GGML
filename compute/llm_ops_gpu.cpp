// funasr/compute/llm_ops_gpu.cpp
// GPU LLM Decoder 前向计算实现
//
// 与 CPU 版的核心区别:
//   KV Cache 更新用 ggml_cpy（图内执行，在 GPU 上完成）
//   不需要 pending/commit 机制
//
#include "compute/llm_ops_gpu.hpp"
#include <cmath>

namespace funasr {

// ============================================================
// RMSNorm (GPU 版，逻辑与 CPU 相同)
// ============================================================
static ggml_tensor* gpu_rms_norm(
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
// GPU GQA Attention with KV Cache
//
// 与 CPU 版的区别:
//   步骤 5: 用 ggml_cpy 把 K/V 写入 cache slot（图内执行）
//           而不是 pending + memcpy（图外执行）
// ============================================================
ggml_tensor* gpu_gqa_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
) {
    const int seq_len    = static_cast<int>(x->ne[1]);
    const int n_heads    = cfg.head_count;
    const int n_kv_heads = cfg.head_count_kv;
    const int head_dim   = cfg.head_dim();
    const int kv_dim     = cfg.kv_dim();
    const int n_kv       = n_past + seq_len;
    const float eps      = 1e-5f;

    // ===== 1. Q/K/V 投影 =====
    ggml_tensor* q     = ggml_mul_mat(ctx, layer.q_proj_w, x);
    ggml_tensor* k_cur = ggml_mul_mat(ctx, layer.k_proj_w, x);
    ggml_tensor* v_cur = ggml_mul_mat(ctx, layer.v_proj_w, x);

    // ===== 2. Reshape 3D =====
    q     = ggml_reshape_3d(ctx, q,     head_dim, n_heads,    seq_len);
    k_cur = ggml_reshape_3d(ctx, k_cur, head_dim, n_kv_heads, seq_len);
    v_cur = ggml_reshape_3d(ctx, v_cur, head_dim, n_kv_heads, seq_len);

    // ===== 3. Q/K RMSNorm =====
    q     = gpu_rms_norm(ctx, q,     layer.q_norm_w, eps);
    k_cur = gpu_rms_norm(ctx, k_cur, layer.k_norm_w, eps);

    // ===== 4. RoPE =====
    // 注意: GPU 模式下 pos->data 可能是 NULL (no_alloc=true)
    // position 数据通过 ggml_backend_tensor_set 在图执行前设置
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(pos);  // 标记为输入，gallocr 会保留它

    q     = ggml_rope_ext(ctx, q,     pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx, k_cur, pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // ===== 5. K/V → 2D + ggml_cpy 写入 cache =====
    k_cur = ggml_reshape_2d(ctx, k_cur, kv_dim, seq_len);
    k_cur = ggml_cont(ctx, k_cur);
    v_cur = ggml_reshape_2d(ctx, v_cur, kv_dim, seq_len);
    v_cur = ggml_cont(ctx, v_cur);

    // Cache 中的目标 slot
    size_t layer_offset = static_cast<size_t>(layer_idx) * cache.n_ctx * kv_dim * sizeof(float);
    size_t pos_offset   = static_cast<size_t>(n_past) * kv_dim * sizeof(float);

    ggml_tensor* k_slot = ggml_view_2d(ctx, cache.k,
        kv_dim, seq_len, kv_dim * sizeof(float), layer_offset + pos_offset);
    ggml_tensor* v_slot = ggml_view_2d(ctx, cache.v,
        kv_dim, seq_len, kv_dim * sizeof(float), layer_offset + pos_offset);

    // ggml_cpy: 在图执行时把 k_cur/v_cur 拷贝到 cache slot
    ggml_tensor* k_cpy = ggml_cpy(ctx, k_cur, k_slot);
    ggml_tensor* v_cpy = ggml_cpy(ctx, v_cur, v_slot);
    kv_cpy_ops.push_back(k_cpy);
    kv_cpy_ops.push_back(v_cpy);

    // ===== 6. 从 cache 读取完整 K/V =====
    ggml_tensor* k_full = ggml_view_2d(ctx, cache.k,
        kv_dim, n_kv, kv_dim * sizeof(float), layer_offset);
    ggml_tensor* v_full = ggml_view_2d(ctx, cache.v,
        kv_dim, n_kv, kv_dim * sizeof(float), layer_offset);

    // ===== 7. Reshape + Permute =====
    k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv_heads, n_kv);
    k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 2, 1, 3));

    v_full = ggml_reshape_3d(ctx, v_full, head_dim, n_kv_heads, n_kv);
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 2, 1, 3));

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // ===== 8. GQA Expansion (8→16) =====
    if (n_kv_heads < n_heads) {
        int repeat_factor = n_heads / n_kv_heads;

        k_full = ggml_reshape_4d(ctx, k_full, head_dim, n_kv, n_kv_heads, 1);
        ggml_tensor* k_target = ggml_new_tensor_4d(ctx, k_full->type,
            head_dim, n_kv, n_kv_heads, repeat_factor);
        k_full = ggml_repeat(ctx, k_full, k_target);
        k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 1, 3, 2));
        k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv, n_heads);
        k_full = ggml_cont(ctx, k_full);

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

    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q);
    scores = ggml_diag_mask_inf(ctx, scores, n_past);
    ggml_tensor* attn = ggml_soft_max(ctx, scores);

    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_full, attn);

    // ===== 10. Output projection =====
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, seq_len);

    return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
}

// ============================================================
// GPU Single LLM Layer
// ============================================================
ggml_tensor* gpu_llm_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
) {
    const float eps = 1e-5f;

    // Attention
    ggml_tensor* residual = x;
    ggml_tensor* x_norm = gpu_rms_norm(ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gpu_gqa_forward(
        ctx, x_norm, layer, cache, layer_idx, n_past, cfg, kv_cpy_ops);
    x = ggml_add(ctx, residual, attn_out);

    // SwiGLU MLP
    residual = x;
    x_norm = gpu_rms_norm(ctx, x, layer.post_attn_norm_w, eps);
    ggml_tensor* gate = ggml_mul_mat(ctx, layer.gate_proj_w, x_norm);
    ggml_tensor* up   = ggml_mul_mat(ctx, layer.up_proj_w,   x_norm);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* mlp_out = ggml_mul_mat(ctx, layer.down_proj_w, ggml_mul(ctx, gate, up));

    return ggml_add(ctx, residual, mlp_out);
}

// ============================================================
// GPU Full LLM Decoder
// ============================================================
ggml_tensor* gpu_llm_forward(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int n_past,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
) {
    const float eps = 1e-5f;
    ggml_tensor* x = hidden_states;

    for (int i = 0; i < cfg.block_count; i++) {
        x = gpu_llm_layer_forward(ctx, x, weights.layers[i], cache,
                                   i, n_past, cfg, kv_cpy_ops);
    }

    x = gpu_rms_norm(ctx, x, weights.model_norm_w, eps);
    return ggml_mul_mat(ctx, weights.lm_head_w, x);
}

} // namespace funasr