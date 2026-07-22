// funasr/compute/llm_ops_gpu.cpp
// GPU LLM Decoder 前向计算实现
//
// 与 CPU 版的核心区别:
//   KV Cache 更新用 ggml_cpy（图内执行，在 GPU 上完成）
//   不需要 pending/commit 机制
//
#include "compute/llm_ops_gpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace funasr {

static bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

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
    return gpu_gqa_forward_slot(ctx, x, layer, cache, layer_idx, n_past, 0,
                                cfg, kv_cpy_ops);
}

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
    const size_t k_row_size = ggml_row_size(cache.k->type, kv_dim);
    const size_t v_row_size = ggml_row_size(cache.v->type, kv_dim);
    slot_id = std::max(0, std::min(slot_id, cache.n_slots - 1));
    size_t k_slot_base = static_cast<size_t>(slot_id) * cache.n_layers * cache.n_ctx * k_row_size;
    size_t v_slot_base = static_cast<size_t>(slot_id) * cache.n_layers * cache.n_ctx * v_row_size;
    size_t k_layer_offset = k_slot_base + static_cast<size_t>(layer_idx) * cache.n_ctx * k_row_size;
    size_t v_layer_offset = v_slot_base + static_cast<size_t>(layer_idx) * cache.n_ctx * v_row_size;
    size_t k_pos_offset   = static_cast<size_t>(n_past) * k_row_size;
    size_t v_pos_offset   = static_cast<size_t>(n_past) * v_row_size;

    ggml_tensor* k_slot = ggml_view_2d(ctx, cache.k,
        kv_dim, seq_len, k_row_size, k_layer_offset + k_pos_offset);
    ggml_tensor* v_slot = ggml_view_2d(ctx, cache.v,
        kv_dim, seq_len, v_row_size, v_layer_offset + v_pos_offset);

    // ggml_cpy: 在图执行时把 k_cur/v_cur 拷贝到 cache slot
    ggml_tensor* k_cpy = ggml_cpy(ctx, k_cur, k_slot);
    ggml_tensor* v_cpy = ggml_cpy(ctx, v_cur, v_slot);
    kv_cpy_ops.push_back(k_cpy);
    kv_cpy_ops.push_back(v_cpy);

    // ===== 6. 从 cache 读取完整 K/V =====
    ggml_tensor* k_full = ggml_view_2d(ctx, cache.k,
        kv_dim, n_kv, k_row_size, k_layer_offset);
    ggml_tensor* v_full = ggml_view_2d(ctx, cache.v,
        kv_dim, n_kv, v_row_size, v_layer_offset);

    // ===== 7. Reshape + Permute =====
    k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv_heads, n_kv);
    k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 2, 1, 3));

    v_full = ggml_reshape_3d(ctx, v_full, head_dim, n_kv_heads, n_kv);
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 2, 1, 3));

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // ===== 8. Attention =====
    ggml_tensor* attn_out = nullptr;
#ifdef FUNASR_USE_FLASH_ATTN
    q = ggml_reshape_4d(ctx, q, head_dim, seq_len, n_heads, 1);
    k_full = ggml_reshape_4d(ctx, k_full, head_dim, n_kv, n_kv_heads, 1);
    v_full = ggml_reshape_4d(ctx, v_full, head_dim, n_kv, n_kv_heads, 1);

    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, seq_len, 1, 1);
    ggml_set_name(mask, "flash_attn_mask");
    ggml_set_input(mask);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_out = ggml_flash_attn_ext(ctx, q, k_full, v_full, mask,
                                   scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
#else
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
    attn_out = ggml_mul_mat(ctx, v_full, attn);
#endif

    // ===== 10. Output projection =====
#ifdef FUNASR_USE_FLASH_ATTN
    // ggml_flash_attn_ext returns [head_dim, n_heads, seq_len, 1], already in
    // merged-head order for reshape_2d([n_heads * head_dim, seq_len]).
    attn_out = ggml_cont(ctx, attn_out);
#else
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
#endif
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, seq_len);

    return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
}

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
    std::vector<ggml_tensor*>* block_table_inputs,
    std::vector<ggml_tensor*>* position_inputs,
    ggml_tensor* shared_block_table_input,
    ggml_tensor* shared_position_input
) {
    const int seq_len    = static_cast<int>(x->ne[1]);
    const int n_heads    = cfg.head_count;
    const int n_kv_heads = cfg.head_count_kv;
    const int head_dim   = cfg.head_dim();
    const int kv_dim     = cfg.kv_dim();
    const int n_kv       = n_past + seq_len;
    const float eps      = 1e-5f;

    ggml_tensor* q     = ggml_mul_mat(ctx, layer.q_proj_w, x);
    ggml_tensor* k_cur = ggml_mul_mat(ctx, layer.k_proj_w, x);
    ggml_tensor* v_cur = ggml_mul_mat(ctx, layer.v_proj_w, x);

    q     = ggml_reshape_3d(ctx, q,     head_dim, n_heads,    seq_len);
    k_cur = ggml_reshape_3d(ctx, k_cur, head_dim, n_kv_heads, seq_len);
    v_cur = ggml_reshape_3d(ctx, v_cur, head_dim, n_kv_heads, seq_len);

    q     = gpu_rms_norm(ctx, q,     layer.q_norm_w, eps);
    k_cur = gpu_rms_norm(ctx, k_cur, layer.k_norm_w, eps);

    ggml_tensor* pos = shared_position_input;
    if (!pos) {
        pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
        ggml_set_input(pos);
        if (position_inputs) {
            position_inputs->push_back(pos);
        }
    }

    q     = ggml_rope_ext(ctx, q,     pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx, k_cur, pos, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k_cur = ggml_cont(ctx, ggml_reshape_2d(ctx, k_cur, kv_dim, seq_len));
    v_cur = ggml_cont(ctx, ggml_reshape_2d(ctx, v_cur, kv_dim, seq_len));

    const size_t k_row_size = ggml_row_size(cache.k->type, kv_dim);
    const size_t v_row_size = ggml_row_size(cache.v->type, kv_dim);
    const int physical_rows = cache.physical_rows > 0 ? cache.physical_rows : cache.n_ctx;
    size_t k_layer_offset = static_cast<size_t>(layer_idx) * physical_rows * k_row_size;
    size_t v_layer_offset = static_cast<size_t>(layer_idx) * physical_rows * v_row_size;

    ggml_tensor* k_layer = ggml_view_2d(ctx, cache.k,
        kv_dim, physical_rows, k_row_size, k_layer_offset);
    ggml_tensor* v_layer = ggml_view_2d(ctx, cache.v,
        kv_dim, physical_rows, v_row_size, v_layer_offset);

    // Row ids are filled by GPURunner before graph compute:
    // physical_row = block_table[logical_pos / block_size] * block_size
    //              + logical_pos % block_size.
    auto physical_row = [&](int logical_pos) -> int {
        int block_idx = logical_pos / block_size;
        int block_off = logical_pos % block_size;
        if (block_idx < 0 || block_idx >= static_cast<int>(block_table.size())) {
            return -1;
        }
        return block_table[static_cast<size_t>(block_idx)] * block_size + block_off;
    };

    auto add_segment_copies = [&](ggml_tensor* src_base, ggml_tensor* dst_base,
                                  int src_logical_start, int dst_logical_start,
                                  int count, size_t src_row_size, size_t dst_row_size) {
        int done = 0;
        while (done < count) {
            const int dst_logical = dst_logical_start + done;
            const int dst_physical = physical_row(dst_logical);
            if (dst_physical < 0) {
                break;
            }
            const int in_block = dst_logical % block_size;
            const int n_seg = std::min(count - done, block_size - in_block);

            ggml_tensor* src = ggml_view_2d(ctx, src_base,
                kv_dim, n_seg, src_row_size,
                static_cast<size_t>(src_logical_start + done) * src_row_size);
            ggml_tensor* dst = ggml_view_2d(ctx, dst_base,
                kv_dim, n_seg, dst_row_size,
                static_cast<size_t>(dst_physical) * dst_row_size);
            kv_cpy_ops.push_back(ggml_cpy(ctx, src, dst));
            done += n_seg;
        }
    };

    const size_t k_cur_row_size = ggml_row_size(k_cur->type, kv_dim);
    const size_t v_cur_row_size = ggml_row_size(v_cur->type, kv_dim);

    // Write current K/V into physical paged rows.
    add_segment_copies(k_cur, k_layer, 0, n_past, seq_len,
                       k_cur_row_size, k_row_size);
    add_segment_copies(v_cur, v_layer, 0, n_past, seq_len,
                       v_cur_row_size, v_row_size);

    // Decode path: keep KV paged and let the CUDA attention op follow the
    // block table directly. Prefill still uses the contiguous fallback below.
    if (seq_len == 1) {
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        q = ggml_reshape_4d(ctx, q, head_dim, seq_len, n_heads, 1);

        ggml_tensor* block_ids = shared_block_table_input;
        if (!block_ids) {
            block_ids = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, static_cast<int64_t>(block_table.size()));
            ggml_set_name(block_ids, "paged_block_table");
            ggml_set_input(block_ids);
            if (block_table_inputs) {
                block_table_inputs->push_back(block_ids);
            }
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor* attn_out = ggml_paged_attn_ext(
            ctx, q, k_layer, v_layer, block_ids, scale,
            block_size, n_kv, n_kv_heads);
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, seq_len);
        return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
    }

    // Gather the logical prefix into contiguous scratch K/V tensors, then use
    // the existing attention graph. This validates block-table semantics before
    // replacing it with a fused PagedAttention kernel.
    ggml_tensor* k_full = ggml_new_tensor_2d(ctx, cache.k->type, kv_dim, n_kv);
    ggml_tensor* v_full = ggml_new_tensor_2d(ctx, cache.v->type, kv_dim, n_kv);

    int logical = 0;
    while (logical < n_kv) {
        int physical = physical_row(logical);
        if (physical < 0) {
            break;
        }
        int in_block = logical % block_size;
        int n_seg = std::min(n_kv - logical, block_size - in_block);

        ggml_tensor* k_src = ggml_view_2d(ctx, k_layer,
            kv_dim, n_seg, k_row_size, static_cast<size_t>(physical) * k_row_size);
        ggml_tensor* v_src = ggml_view_2d(ctx, v_layer,
            kv_dim, n_seg, v_row_size, static_cast<size_t>(physical) * v_row_size);
        ggml_tensor* k_dst = ggml_view_2d(ctx, k_full,
            kv_dim, n_seg, k_row_size, static_cast<size_t>(logical) * k_row_size);
        ggml_tensor* v_dst = ggml_view_2d(ctx, v_full,
            kv_dim, n_seg, v_row_size, static_cast<size_t>(logical) * v_row_size);
        kv_cpy_ops.push_back(ggml_cpy(ctx, k_src, k_dst));
        kv_cpy_ops.push_back(ggml_cpy(ctx, v_src, v_dst));
        logical += n_seg;
    }

    k_full = ggml_reshape_3d(ctx, k_full, head_dim, n_kv_heads, n_kv);
    k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 2, 1, 3));

    v_full = ggml_reshape_3d(ctx, v_full, head_dim, n_kv_heads, n_kv);
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 2, 1, 3));

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    ggml_tensor* attn_out = nullptr;
#ifdef FUNASR_USE_FLASH_ATTN
    q = ggml_reshape_4d(ctx, q, head_dim, seq_len, n_heads, 1);
    k_full = ggml_reshape_4d(ctx, k_full, head_dim, n_kv, n_kv_heads, 1);
    v_full = ggml_reshape_4d(ctx, v_full, head_dim, n_kv, n_kv_heads, 1);

    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, seq_len, 1, 1);
    ggml_set_name(mask, "flash_attn_mask");
    ggml_set_input(mask);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_out = ggml_flash_attn_ext(ctx, q, k_full, v_full, mask,
                                   scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
#else
    if (n_kv_heads < n_heads) {
        int repeat_factor = n_heads / n_kv_heads;

        k_full = ggml_reshape_4d(ctx, k_full, head_dim, n_kv, n_kv_heads, 1);
        ggml_tensor* k_target = ggml_new_tensor_4d(ctx, k_full->type,
            head_dim, n_kv, n_kv_heads, repeat_factor);
        k_full = ggml_repeat(ctx, k_full, k_target);
        k_full = ggml_cont(ctx, ggml_permute(ctx, k_full, 0, 1, 3, 2));
        k_full = ggml_cont(ctx, ggml_reshape_3d(ctx, k_full, head_dim, n_kv, n_heads));

        v_full = ggml_reshape_4d(ctx, v_full, head_dim, n_kv, n_kv_heads, 1);
        ggml_tensor* v_target = ggml_new_tensor_4d(ctx, v_full->type,
            head_dim, n_kv, n_kv_heads, repeat_factor);
        v_full = ggml_repeat(ctx, v_full, v_target);
        v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 0, 1, 3, 2));
        v_full = ggml_cont(ctx, ggml_reshape_3d(ctx, v_full, head_dim, n_kv, n_heads));
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    q = ggml_scale(ctx, q, scale);
    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q);
    scores = ggml_diag_mask_inf(ctx, scores, n_past);
    ggml_tensor* attn = ggml_soft_max(ctx, scores);
    v_full = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
    attn_out = ggml_mul_mat(ctx, v_full, attn);
#endif

#ifdef FUNASR_USE_FLASH_ATTN
    attn_out = ggml_cont(ctx, attn_out);
#else
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
#endif
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, seq_len);
    (void)block_size;
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
    return gpu_llm_layer_forward_slot(ctx, x, layer, cache, layer_idx, n_past, 0,
                                      cfg, kv_cpy_ops);
}

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
) {
    const float eps = 1e-5f;

    // Attention
    ggml_tensor* residual = x;
    ggml_tensor* x_norm = gpu_rms_norm(ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gpu_gqa_forward_slot(
        ctx, x_norm, layer, cache, layer_idx, n_past, slot_id, cfg, kv_cpy_ops);
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
    std::vector<ggml_tensor*>* block_table_inputs,
    std::vector<ggml_tensor*>* position_inputs,
    ggml_tensor* shared_block_table_input,
    ggml_tensor* shared_position_input
) {
    const float eps = 1e-5f;

    ggml_tensor* residual = x;
    ggml_tensor* x_norm = gpu_rms_norm(ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gpu_gqa_forward_paged(
        ctx, x_norm, layer, cache, layer_idx, n_past, block_table, block_size,
        cfg, kv_cpy_ops, block_table_inputs, position_inputs,
        shared_block_table_input, shared_position_input);
    x = ggml_add(ctx, residual, attn_out);

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
    return gpu_llm_forward_slot(ctx, hidden_states, weights, cache, n_past, 0,
                                cfg, kv_cpy_ops);
}

ggml_tensor* gpu_llm_forward_slot(
    ggml_context* ctx,
    ggml_tensor* hidden_states,
    const GPULLMWeights& weights,
    GPUKVCache& cache,
    int n_past,
    int slot_id,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops
) {
    const float eps = 1e-5f;
    ggml_tensor* x = hidden_states;

    for (int i = 0; i < cfg.block_count; i++) {
        x = gpu_llm_layer_forward_slot(ctx, x, weights.layers[i], cache,
                                       i, n_past, slot_id, cfg, kv_cpy_ops);
    }

    x = gpu_rms_norm(ctx, x, weights.model_norm_w, eps);
    return ggml_mul_mat(ctx, weights.lm_head_w, x);
}

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
    std::vector<ggml_tensor*>* block_table_inputs,
    std::vector<ggml_tensor*>* position_inputs
) {
    const float eps = 1e-5f;
    ggml_tensor* x = hidden_states;
    ggml_tensor* shared_block_table_input = nullptr;
    ggml_tensor* shared_position_input = nullptr;
    if (hidden_states->ne[1] == 1 && block_table_inputs) {
        shared_block_table_input = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, static_cast<int64_t>(block_table.size()));
        ggml_set_name(shared_block_table_input, "paged_block_table");
        ggml_set_input(shared_block_table_input);
        block_table_inputs->push_back(shared_block_table_input);
    }
    if (hidden_states->ne[1] == 1 && position_inputs) {
        shared_position_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(shared_position_input, "paged_position");
        ggml_set_input(shared_position_input);
        position_inputs->push_back(shared_position_input);
    }

    for (int i = 0; i < cfg.block_count; i++) {
        x = gpu_llm_layer_forward_paged(ctx, x, weights.layers[i], cache,
                                        i, n_past, block_table, block_size, cfg,
                                        kv_cpy_ops, block_table_inputs, position_inputs,
                                        shared_block_table_input,
                                        shared_position_input);
    }

    x = gpu_rms_norm(ctx, x, weights.model_norm_w, eps);
    return ggml_mul_mat(ctx, weights.lm_head_w, x);
}

static ggml_tensor* gpu_gqa_forward_paged_batch_decode(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    const std::vector<int>& n_pasts,
    const std::vector<std::vector<int>>& block_tables,
    int block_size,
    int graph_max_n_kv,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    ggml_tensor* block_table_input,
    ggml_tensor* position_input,
    ggml_tensor* kv_lens_input
) {
    const int batch      = static_cast<int>(x->ne[1]);
    const int n_heads    = cfg.head_count;
    const int n_kv_heads = cfg.head_count_kv;
    const int head_dim   = cfg.head_dim();
    const int kv_dim     = cfg.kv_dim();
    int exact_max_n_kv = 0;
    for (int n_past : n_pasts) {
        exact_max_n_kv = std::max(exact_max_n_kv, n_past + 1);
    }
    const int max_n_kv = graph_max_n_kv > 0 ? graph_max_n_kv : exact_max_n_kv;
    const float eps      = 1e-5f;

    ggml_tensor* q     = ggml_mul_mat(ctx, layer.q_proj_w, x);
    ggml_tensor* k_cur = ggml_mul_mat(ctx, layer.k_proj_w, x);
    ggml_tensor* v_cur = ggml_mul_mat(ctx, layer.v_proj_w, x);

    q     = ggml_reshape_3d(ctx, q,     head_dim, n_heads,    batch);
    k_cur = ggml_reshape_3d(ctx, k_cur, head_dim, n_kv_heads, batch);
    v_cur = ggml_reshape_3d(ctx, v_cur, head_dim, n_kv_heads, batch);

    q     = gpu_rms_norm(ctx, q,     layer.q_norm_w, eps);
    k_cur = gpu_rms_norm(ctx, k_cur, layer.k_norm_w, eps);

    q     = ggml_rope_ext(ctx, q,     position_input, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx, k_cur, position_input, nullptr, head_dim, 2, 32768,
                           cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k_cur = ggml_cont(ctx, ggml_reshape_2d(ctx, k_cur, kv_dim, batch));
    v_cur = ggml_cont(ctx, ggml_reshape_2d(ctx, v_cur, kv_dim, batch));

    const size_t k_row_size = ggml_row_size(cache.k->type, kv_dim);
    const size_t v_row_size = ggml_row_size(cache.v->type, kv_dim);
    const int physical_rows = cache.physical_rows > 0 ? cache.physical_rows : cache.n_ctx;
    const size_t k_layer_offset = static_cast<size_t>(layer_idx) * physical_rows * k_row_size;
    const size_t v_layer_offset = static_cast<size_t>(layer_idx) * physical_rows * v_row_size;

    ggml_tensor* k_layer = ggml_view_2d(ctx, cache.k,
        kv_dim, physical_rows, k_row_size, k_layer_offset);
    ggml_tensor* v_layer = ggml_view_2d(ctx, cache.v,
        kv_dim, physical_rows, v_row_size, v_layer_offset);

    const size_t k_cur_row_size = ggml_row_size(k_cur->type, kv_dim);
    const size_t v_cur_row_size = ggml_row_size(v_cur->type, kv_dim);
    ggml_tensor* kv_write_dependency = nullptr;
    if (env_flag_enabled("FUNASR_PAGED_KV_WRITE_OP")) {
        kv_write_dependency = ggml_paged_kv_write_ext(
            ctx, k_cur, v_cur, k_layer, v_layer, block_table_input,
            position_input, block_size);
        kv_cpy_ops.push_back(kv_write_dependency);
    } else {
        for (int i = 0; i < batch; i++) {
            const auto& table = block_tables[static_cast<size_t>(i)];
            const int n_past = n_pasts[static_cast<size_t>(i)];
            const int block_idx = n_past / block_size;
            const int block_off = n_past % block_size;
            const int physical = table[static_cast<size_t>(block_idx)] * block_size + block_off;

            ggml_tensor* k_src = ggml_view_2d(ctx, k_cur,
                kv_dim, 1, k_cur_row_size, static_cast<size_t>(i) * k_cur_row_size);
            ggml_tensor* v_src = ggml_view_2d(ctx, v_cur,
                kv_dim, 1, v_cur_row_size, static_cast<size_t>(i) * v_cur_row_size);
            ggml_tensor* k_dst = ggml_view_2d(ctx, k_layer,
                kv_dim, 1, k_row_size, static_cast<size_t>(physical) * k_row_size);
            ggml_tensor* v_dst = ggml_view_2d(ctx, v_layer,
                kv_dim, 1, v_row_size, static_cast<size_t>(physical) * v_row_size);
            kv_cpy_ops.push_back(ggml_cpy(ctx, k_src, k_dst));
            kv_cpy_ops.push_back(ggml_cpy(ctx, v_src, v_dst));
        }
    }

    q = ggml_cont(ctx, q);
    q = ggml_reshape_4d(ctx, q, head_dim, 1, n_heads, batch);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor* attn_out = ggml_paged_attn_ext_v_dep(
        ctx, q, k_layer, v_layer, block_table_input, kv_lens_input,
        kv_write_dependency, scale, block_size, max_n_kv, n_kv_heads);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, batch);
    return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
}

static ggml_tensor* gpu_llm_layer_forward_paged_batch_decode(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    const std::vector<int>& n_pasts,
    const std::vector<std::vector<int>>& block_tables,
    int block_size,
    int graph_max_n_kv,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_cpy_ops,
    ggml_tensor* block_table_input,
    ggml_tensor* position_input,
    ggml_tensor* kv_lens_input
) {
    const float eps = 1e-5f;

    ggml_tensor* residual = x;
    ggml_tensor* x_norm = gpu_rms_norm(ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gpu_gqa_forward_paged_batch_decode(
        ctx, x_norm, layer, cache, layer_idx, n_pasts, block_tables,
        block_size, graph_max_n_kv, cfg, kv_cpy_ops, block_table_input, position_input,
        kv_lens_input);
    x = ggml_add(ctx, residual, attn_out);

    residual = x;
    x_norm = gpu_rms_norm(ctx, x, layer.post_attn_norm_w, eps);
    ggml_tensor* gate = ggml_mul_mat(ctx, layer.gate_proj_w, x_norm);
    ggml_tensor* up   = ggml_mul_mat(ctx, layer.up_proj_w,   x_norm);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* mlp_out = ggml_mul_mat(ctx, layer.down_proj_w, ggml_mul(ctx, gate, up));

    return ggml_add(ctx, residual, mlp_out);
}

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
) {
    const float eps = 1e-5f;
    const int batch = static_cast<int>(hidden_states->ne[1]);
    GGML_ASSERT(static_cast<int>(n_pasts.size()) == batch);
    int max_blocks = 0;
    for (const auto& table : block_tables) {
        max_blocks = std::max(max_blocks, static_cast<int>(table.size()));
    }

    ggml_tensor* block_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, batch);
    ggml_set_name(block_ids, "paged_batch_block_table");
    ggml_set_input(block_ids);
    *block_table_input = block_ids;

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    ggml_set_name(pos, "paged_batch_position");
    ggml_set_input(pos);
    *position_input = pos;

    ggml_tensor* kv_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    ggml_set_name(kv_lens, "paged_batch_kv_lens");
    ggml_set_input(kv_lens);
    *kv_lens_input = kv_lens;

    ggml_tensor* x = hidden_states;
    for (int i = 0; i < cfg.block_count; i++) {
        x = gpu_llm_layer_forward_paged_batch_decode(
            ctx, x, weights.layers[i], cache, i, n_pasts, block_tables,
            block_size, graph_max_n_kv, cfg, kv_cpy_ops, block_ids, pos, kv_lens);
    }

    x = gpu_rms_norm(ctx, x, weights.model_norm_w, eps);
    return ggml_mul_mat(ctx, weights.lm_head_w, x);
}

static ggml_tensor* gpu_gqa_forward_paged_packed(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int block_size,
    int graph_max_n_kv,
    const std::vector<GPUPackedPrefillLayout>& prefill_layouts,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_write_ops,
    const std::vector<ggml_tensor*>& prefill_past_row_inputs,
    const std::vector<ggml_tensor*>& prefill_mask_inputs,
    ggml_tensor* block_table_input,
    ggml_tensor* position_input,
    ggml_tensor* kv_lens_input
) {
    const int packed_tokens = static_cast<int>(x->ne[1]);
    const int n_heads = cfg.head_count;
    const int n_kv_heads = cfg.head_count_kv;
    const int head_dim = cfg.head_dim();
    const int kv_dim = cfg.kv_dim();
    const float eps = 1e-5f;

    ggml_tensor* q = ggml_mul_mat(ctx, layer.q_proj_w, x);
    ggml_tensor* k_cur = ggml_mul_mat(ctx, layer.k_proj_w, x);
    ggml_tensor* v_cur = ggml_mul_mat(ctx, layer.v_proj_w, x);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, packed_tokens);
    k_cur = ggml_reshape_3d(
        ctx, k_cur, head_dim, n_kv_heads, packed_tokens);
    v_cur = ggml_reshape_3d(
        ctx, v_cur, head_dim, n_kv_heads, packed_tokens);

    q = gpu_rms_norm(ctx, q, layer.q_norm_w, eps);
    k_cur = gpu_rms_norm(ctx, k_cur, layer.k_norm_w, eps);
    q = ggml_rope_ext(ctx, q, position_input, nullptr, head_dim, 2, 32768,
                      cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(
        ctx, k_cur, position_input, nullptr, head_dim, 2, 32768,
        cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k_cur = ggml_cont(
        ctx, ggml_reshape_2d(ctx, k_cur, kv_dim, packed_tokens));
    v_cur = ggml_cont(
        ctx, ggml_reshape_2d(ctx, v_cur, kv_dim, packed_tokens));

    const size_t k_row_size = ggml_row_size(cache.k->type, kv_dim);
    const size_t v_row_size = ggml_row_size(cache.v->type, kv_dim);
    const int physical_rows =
        cache.physical_rows > 0 ? cache.physical_rows : cache.n_ctx;
    const size_t k_layer_offset = static_cast<size_t>(layer_idx) *
                                  physical_rows * k_row_size;
    const size_t v_layer_offset = static_cast<size_t>(layer_idx) *
                                  physical_rows * v_row_size;
    ggml_tensor* k_layer = ggml_view_2d(
        ctx, cache.k, kv_dim, physical_rows, k_row_size, k_layer_offset);
    ggml_tensor* v_layer = ggml_view_2d(
        ctx, cache.v, kv_dim, physical_rows, v_row_size, v_layer_offset);

    ggml_tensor* kv_write = ggml_paged_kv_write_ext(
        ctx, k_cur, v_cur, k_layer, v_layer,
        block_table_input, position_input, block_size);
    kv_write_ops.push_back(kv_write);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#ifdef FUNASR_USE_FLASH_ATTN
    ggml_tensor* packed_attn_out = nullptr;
    int prefill_tokens = 0;
    GGML_ASSERT(prefill_past_row_inputs.size() == prefill_layouts.size());
    GGML_ASSERT(prefill_mask_inputs.size() == prefill_layouts.size());

    for (size_t i = 0; i < prefill_layouts.size(); ++i) {
        const GPUPackedPrefillLayout& layout = prefill_layouts[i];
        GGML_ASSERT(layout.row_offset == prefill_tokens);
        GGML_ASSERT(layout.token_count > 0 && layout.n_past >= 0);
        GGML_ASSERT(layout.row_offset + layout.token_count <= packed_tokens);

        ggml_tensor* q_seq = ggml_view_3d(
            ctx, q, head_dim, n_heads, layout.token_count,
            q->nb[1], q->nb[2],
            static_cast<size_t>(layout.row_offset) * q->nb[2]);
        q_seq = ggml_cont(
            ctx, ggml_permute(ctx, q_seq, 0, 2, 1, 3));
        q_seq = ggml_reshape_4d(
            ctx, q_seq, head_dim, layout.token_count, n_heads, 1);

        ggml_tensor* k_seq = ggml_view_2d(
            ctx, k_cur, kv_dim, layout.token_count, k_cur->nb[1],
            static_cast<size_t>(layout.row_offset) * k_cur->nb[1]);
        ggml_tensor* v_seq = ggml_view_2d(
            ctx, v_cur, kv_dim, layout.token_count, v_cur->nb[1],
            static_cast<size_t>(layout.row_offset) * v_cur->nb[1]);

        ggml_tensor* k_full = k_seq;
        ggml_tensor* v_full = v_seq;
        if (layout.n_past > 0) {
            GGML_ASSERT(prefill_past_row_inputs[i] != nullptr);
            ggml_tensor* k_past = ggml_get_rows(
                ctx, k_layer, prefill_past_row_inputs[i]);
            ggml_tensor* v_past = ggml_get_rows(
                ctx, v_layer, prefill_past_row_inputs[i]);
            k_full = ggml_concat(ctx, k_past, k_seq, 1);
            v_full = ggml_concat(ctx, v_past, v_seq, 1);
        }

        const int n_kv = layout.n_past + layout.token_count;
        k_full = ggml_reshape_3d(
            ctx, k_full, head_dim, n_kv_heads, n_kv);
        k_full = ggml_cont(
            ctx, ggml_permute(ctx, k_full, 0, 2, 1, 3));
        k_full = ggml_reshape_4d(
            ctx, k_full, head_dim, n_kv, n_kv_heads, 1);

        v_full = ggml_reshape_3d(
            ctx, v_full, head_dim, n_kv_heads, n_kv);
        v_full = ggml_cont(
            ctx, ggml_permute(ctx, v_full, 0, 2, 1, 3));
        v_full = ggml_reshape_4d(
            ctx, v_full, head_dim, n_kv, n_kv_heads, 1);

        GGML_ASSERT(prefill_mask_inputs[i] != nullptr);
        ggml_tensor* seq_out = ggml_flash_attn_ext(
            ctx, q_seq, k_full, v_full, prefill_mask_inputs[i],
            scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(seq_out, GGML_PREC_F32);
        seq_out = ggml_cont(ctx, seq_out);
        seq_out = ggml_reshape_2d(
            ctx, seq_out, n_heads * head_dim, layout.token_count);
        packed_attn_out = packed_attn_out
            ? ggml_concat(ctx, packed_attn_out, seq_out, 1)
            : seq_out;
        prefill_tokens += layout.token_count;
    }

    const int decode_tokens = packed_tokens - prefill_tokens;
    if (decode_tokens > 0) {
        ggml_tensor* q_decode = ggml_view_3d(
            ctx, q, head_dim, n_heads, decode_tokens,
            q->nb[1], q->nb[2],
            static_cast<size_t>(prefill_tokens) * q->nb[2]);
        q_decode = ggml_cont(ctx, q_decode);
        q_decode = ggml_reshape_4d(
            ctx, q_decode, head_dim, 1, n_heads, decode_tokens);

        ggml_tensor* decode_block_table = ggml_view_2d(
            ctx, block_table_input, block_table_input->ne[0],
            decode_tokens, block_table_input->nb[1],
            static_cast<size_t>(prefill_tokens) * block_table_input->nb[1]);
        ggml_tensor* decode_kv_lens = ggml_view_1d(
            ctx, kv_lens_input, decode_tokens,
            static_cast<size_t>(prefill_tokens) * kv_lens_input->nb[0]);
        ggml_tensor* decode_out = ggml_paged_attn_ext_v_dep(
            ctx, q_decode, k_layer, v_layer, decode_block_table,
            decode_kv_lens, kv_write, scale, block_size,
            graph_max_n_kv, n_kv_heads);
        decode_out = ggml_cont(ctx, decode_out);
        decode_out = ggml_reshape_2d(
            ctx, decode_out, n_heads * head_dim, decode_tokens);
        packed_attn_out = packed_attn_out
            ? ggml_concat(ctx, packed_attn_out, decode_out, 1)
            : decode_out;
    }
    GGML_ASSERT(packed_attn_out != nullptr);
    return ggml_mul_mat(ctx, layer.o_proj_w, packed_attn_out);
#else
    q = ggml_cont(ctx, q);
    q = ggml_reshape_4d(
        ctx, q, head_dim, 1, n_heads, packed_tokens);
    ggml_tensor* attn_out = ggml_paged_attn_ext_v_dep(
        ctx, q, k_layer, v_layer, block_table_input, kv_lens_input,
        kv_write, scale, block_size, graph_max_n_kv, n_kv_heads);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(
        ctx, attn_out, n_heads * head_dim, packed_tokens);
    return ggml_mul_mat(ctx, layer.o_proj_w, attn_out);
#endif
}

static ggml_tensor* gpu_llm_layer_forward_paged_packed(
    ggml_context* ctx,
    ggml_tensor* x,
    const GPULLMLayerWeights& layer,
    GPUKVCache& cache,
    int layer_idx,
    int block_size,
    int graph_max_n_kv,
    const std::vector<GPUPackedPrefillLayout>& prefill_layouts,
    const LLMConfig& cfg,
    std::vector<ggml_tensor*>& kv_write_ops,
    const std::vector<ggml_tensor*>& prefill_past_row_inputs,
    const std::vector<ggml_tensor*>& prefill_mask_inputs,
    ggml_tensor* block_table_input,
    ggml_tensor* position_input,
    ggml_tensor* kv_lens_input
) {
    const float eps = 1e-5f;
    ggml_tensor* residual = x;
    ggml_tensor* x_norm = gpu_rms_norm(
        ctx, x, layer.input_norm_w, eps);
    ggml_tensor* attn_out = gpu_gqa_forward_paged_packed(
        ctx, x_norm, layer, cache, layer_idx, block_size,
        graph_max_n_kv, prefill_layouts, cfg, kv_write_ops,
        prefill_past_row_inputs, prefill_mask_inputs,
        block_table_input, position_input, kv_lens_input);
    x = ggml_add(ctx, residual, attn_out);

    residual = x;
    x_norm = gpu_rms_norm(ctx, x, layer.post_attn_norm_w, eps);
    ggml_tensor* gate = ggml_mul_mat(ctx, layer.gate_proj_w, x_norm);
    ggml_tensor* up = ggml_mul_mat(ctx, layer.up_proj_w, x_norm);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* mlp_out = ggml_mul_mat(
        ctx, layer.down_proj_w, ggml_mul(ctx, gate, up));
    return ggml_add(ctx, residual, mlp_out);
}

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
) {
    const int packed_tokens = static_cast<int>(hidden_states->ne[1]);
    GGML_ASSERT(packed_tokens > 0);
    GGML_ASSERT(max_blocks > 0);
    GGML_ASSERT(block_size > 0);
    GGML_ASSERT(graph_max_n_kv > 0);
    GGML_ASSERT(selected_row_count >= 0 &&
                selected_row_count <= packed_tokens);

    ggml_tensor* block_ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, max_blocks, packed_tokens);
    ggml_set_name(block_ids, "packed_paged_block_table");
    ggml_set_input(block_ids);
    *block_table_input = block_ids;

    ggml_tensor* positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, packed_tokens);
    ggml_set_name(positions, "packed_paged_positions");
    ggml_set_input(positions);
    *position_input = positions;

    ggml_tensor* kv_lens = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, packed_tokens);
    ggml_set_name(kv_lens, "packed_paged_kv_lens");
    ggml_set_input(kv_lens);
    *kv_lens_input = kv_lens;

    ggml_tensor* selected_rows = nullptr;
    if (selected_row_count > 0) {
        selected_rows = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, selected_row_count);
        ggml_set_name(selected_rows, "packed_selected_rows");
        ggml_set_input(selected_rows);
    }
    *selected_rows_input = selected_rows;

#ifdef FUNASR_USE_FLASH_ATTN
    prefill_past_row_inputs.clear();
    prefill_mask_inputs.clear();
    prefill_past_row_inputs.reserve(prefill_layouts.size());
    prefill_mask_inputs.reserve(prefill_layouts.size());
    for (const GPUPackedPrefillLayout& layout : prefill_layouts) {
        GGML_ASSERT(layout.row_offset >= 0 && layout.token_count > 0 &&
                    layout.n_past >= 0 &&
                    layout.row_offset + layout.token_count <= packed_tokens);
        ggml_tensor* past_rows = nullptr;
        if (layout.n_past > 0) {
            past_rows = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, layout.n_past);
            ggml_set_name(past_rows, "packed_prefill_past_rows");
            ggml_set_input(past_rows);
        }
        ggml_tensor* mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16,
            layout.n_past + layout.token_count,
            layout.token_count);
        ggml_set_name(mask, "packed_prefill_mask");
        ggml_set_input(mask);
        prefill_past_row_inputs.push_back(past_rows);
        prefill_mask_inputs.push_back(mask);
    }
#else
    (void)prefill_layouts;
    prefill_past_row_inputs.clear();
    prefill_mask_inputs.clear();
#endif

    ggml_tensor* x = hidden_states;
    for (int i = 0; i < cfg.block_count; ++i) {
        x = gpu_llm_layer_forward_paged_packed(
            ctx, x, weights.layers[static_cast<size_t>(i)], cache, i,
            block_size, graph_max_n_kv, prefill_layouts, cfg, kv_write_ops,
            prefill_past_row_inputs, prefill_mask_inputs,
            block_ids, positions, kv_lens);
    }

    if (selected_rows) {
        x = ggml_get_rows(ctx, x, selected_rows);
        x = gpu_rms_norm(ctx, x, weights.model_norm_w, 1e-5f);
        return ggml_mul_mat(ctx, weights.lm_head_w, x);
    }
    return x;
}

} // namespace funasr
