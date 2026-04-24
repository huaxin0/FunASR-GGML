// Audio Encoder 前向计算实现
//
// 核心算法:
//   FSMN: depthwise conv1d → K 次 slice-multiply-accumulate
//   SANM: Q/K/V投影 → FSMN(V) → Multi-head Attention → Output投影
//   LayerNorm: ggml_norm(eps=1e-5) + affine
//   FFN: linear → ReLU → linear
#include "compute/encoder_ops.hpp"
#include <cmath>
#include <cstdio>

namespace funasr {

// ============================================================
// 辅助: LayerNorm (ggml_norm + affine)
// ============================================================
static ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    ggml_tensor* bias,
    float eps = 1e-5f
) {
    ggml_tensor* w = ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor* b = ggml_cast(ctx, bias,   GGML_TYPE_F32);
    ggml_tensor* normed = ggml_norm(ctx, x, eps);
    normed = ggml_mul(ctx, normed, w);
    normed = ggml_add(ctx, normed, b);
    return normed;
}

// ============================================================
// 辅助: 线性层 + bias (cast bias to F32)
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
// FSMN: Feedforward Sequential Memory Networks
//
// 等价于 Python: MultiHeadedAttentionSANM.forward_fsmn()
// 实现 depthwise conv1d, GGML 不支持 groups conv, 手动展开
//
// 公式: output[c, t] = Σ_{k=0}^{K-1} v[c, t+k-pad] * w[c, k] + v[c, t]
//
// 输入: v [dim, T] F32,  fsmn_w [K, 1, dim] (F16/BF16)
// 输出: [dim, T] F32
// ============================================================
ggml_tensor* fsmn_forward(
    ggml_context* ctx,
    ggml_tensor* v,
    ggml_tensor* fsmn_w,
    int kernel_size,
    int pad
) {
    const int dim = static_cast<int>(v->ne[0]);
    const int T   = static_cast<int>(v->ne[1]);
    const int K   = kernel_size;

    // 权重变换: [K, 1, dim] → cast F32 → reshape [K, dim] → transpose → cont → [dim, K]
    ggml_tensor* w    = ggml_cast(ctx, fsmn_w, GGML_TYPE_F32);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, K, dim);
    ggml_tensor* w_T  = ggml_cont(ctx, ggml_transpose(ctx, w_2d));  // [dim, K] 连续

    // 用 k=pad 位置的权重初始化 result (中心位置的贡献)
    // w_T 在内存中: w_T[c, k] = *(data + c + k*dim)
    // view_1d 取第 pad 列: 从 offset = pad*dim*sizeof(float) 开始的 dim 个 float
    ggml_tensor* w_center = ggml_view_1d(ctx, w_T, dim,
                                          pad * dim * sizeof(float));
    w_center = ggml_reshape_2d(ctx, w_center, dim, 1);   // [dim, 1] 用于广播
    ggml_tensor* result = ggml_mul(ctx, v, w_center);     // [dim, T]

    // 累加其余 K-1 个位置的贡献
    for (int k = 0; k < K; k++) {
        if (k == pad) continue;

        int shift     = k - pad;
        int src_start = (shift > 0) ? shift : 0;
        int dst_start = (shift < 0) ? (-shift) : 0;
        int len       = T - std::abs(shift);
        if (len <= 0) continue;

        // 第 k 个位置的权重列 [dim, 1]
        ggml_tensor* w_k = ggml_view_1d(ctx, w_T, dim,
                                          k * dim * sizeof(float));
        w_k = ggml_reshape_2d(ctx, w_k, dim, 1);

        // v 的对应时间窗口 [dim, len]
        ggml_tensor* v_slice = ggml_view_2d(ctx, v,
                                             dim, len,
                                             v->nb[1],
                                             src_start * v->nb[1]);

        // 逐元素相乘 [dim, len]
        ggml_tensor* term = ggml_mul(ctx, v_slice, w_k);

        // 累加到 result 的 dst_start 行开始的位置
        result = ggml_acc(ctx, result, term,
                          result->nb[1], result->nb[2], result->nb[3],
                          dst_start * result->nb[1]);
    }

    // 残差: output = conv_result + v
    result = ggml_add(ctx, result, v);
    return result;
}

// ============================================================
// SANM Attention
//
// 等价于 Python: MultiHeadedAttentionSANM.forward()
//
// 核心: 标准 Multi-head Attention + FSMN 记忆
//   最终输出 = attention_output + fsmn_memory
//
// GGML 维度排列 (关键!):
//   Q/K: [dim, T] → [head_dim, n_heads, T] → permute(0,2,1) → [head_dim, T, n_heads]
//   V:   [dim, T] → [head_dim, n_heads, T] → permute(1,2,0) → [T, head_dim, n_heads]
//
//   permute 语义: 旧维度 i → 新维度 axis_i
//     permute(0,2,1): 维度0不动, 维度1→位置2, 维度2→位置1
//     permute(1,2,0): 维度0→位置1, 维度1→位置2, 维度2→位置0
//
//   scores = mul_mat(K, Q): 沿 head_dim 收缩 → [T, T, n_heads]
//   attn_out = mul_mat(V_perm, softmax(scores)): 沿 T 收缩 → [head_dim, T, n_heads]
// ============================================================
ggml_tensor* sanm_attention_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderLayerWeights& layer,
    const EncoderConfig& cfg
) {
    const int T       = static_cast<int>(x->ne[1]);
    const int n_heads = cfg.attention_heads;  // 4

    // ===== Q/K/V 投影 =====
    ggml_tensor* q = linear_with_bias(ctx, x, layer.attn_q_w, layer.attn_q_b);
    ggml_tensor* k = linear_with_bias(ctx, x, layer.attn_k_w, layer.attn_k_b);
    ggml_tensor* v = linear_with_bias(ctx, x, layer.attn_v_w, layer.attn_v_b);

    // ===== FSMN on V =====
    ggml_tensor* fsmn_mem = fsmn_forward(ctx, v, layer.fsmn_w,
                                          cfg.kernel_size, cfg.fsmn_pad());

    // ===== Multi-head reshape + permute =====
    int qk_dim     = static_cast<int>(q->ne[0]);
    int head_dim   = qk_dim / n_heads;
    int v_dim      = static_cast<int>(v->ne[0]);
    int v_head_dim = v_dim / n_heads;

    // Q: [qk_dim, T] → [head_dim, n_heads, T] → permute(0,2,1) → [head_dim, T, n_heads]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // K: 同 Q
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // V: [v_dim, T] → [v_head_dim, n_heads, T] → permute(1,2,0) → [T, v_head_dim, n_heads]
    ggml_tensor* v_3d   = ggml_reshape_3d(ctx, v, v_head_dim, n_heads, T);
    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_3d, 1, 2, 0, 3));

    // ===== Scaled Dot-Product Attention =====
    // 注意: Encoder 是双向的，没有 causal mask
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    q = ggml_scale(ctx, q, scale);

    // scores = K^T @ Q (沿 head_dim 收缩) → [T, T, n_heads]
    ggml_tensor* scores = ggml_mul_mat(ctx, k, q);

    // softmax (作用在 ne[0]=T_k 上)
    ggml_tensor* attn = ggml_soft_max(ctx, scores);

    // attn_out = V_perm^T @ attn (沿 T 收缩) → [v_head_dim, T, n_heads]
    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, attn);

    // ===== 还原形状 =====
    // [v_head_dim, T, n_heads] → permute(0,2,1) → [v_head_dim, n_heads, T]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    // → reshape [v_dim, T]
    attn_out = ggml_reshape_2d(ctx, attn_out, v_dim, T);

    // ===== 输出投影 =====
    attn_out = linear_with_bias(ctx, attn_out, layer.attn_out_w, layer.attn_out_b);

    // ===== 合并: attention + FSMN 记忆 =====
    return ggml_add(ctx, attn_out, fsmn_mem);
}

// ============================================================
// 单层 Encoder 前向传播
//
// 等价于 Python: EncoderLayerSANMExport.forward()
//
// encoder0 特殊点:
//   - norm1 维度是 560 (输入维度), norm2 维度是 512 (输出维度)
//   - attention 后无残差连接 (因为维度从 560 变为 512)
//
// 其他层:
//   - norm1/norm2 都是 512
//   - attention 后有残差连接
// ============================================================
ggml_tensor* encoder_layer_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderLayerWeights& layer,
    const EncoderConfig& cfg,
    bool has_attention_residual
) {
    // ===== Part 1: Norm1 → SANM Attention → (可选)残差 =====
    ggml_tensor* residual = x;

    ggml_tensor* x1 = layer_norm(ctx, x, layer.norm1_w, layer.norm1_b);
    ggml_tensor* attn_out = sanm_attention_forward(ctx, x1, layer, cfg);

    if (has_attention_residual) {
        x = ggml_add(ctx, residual, attn_out);
    } else {
        // encoder0: 维度从 560→512, 不能做残差
        x = attn_out;
    }

    // ===== Part 2: Norm2 → FFN(ReLU) → 残差 =====
    residual = x;

    ggml_tensor* x2 = layer_norm(ctx, x, layer.norm2_w, layer.norm2_b);

    // FFN: linear1 → ReLU → linear2
    ggml_tensor* ffn = linear_with_bias(ctx, x2, layer.ffn_w1, layer.ffn_b1);
    ffn = ggml_relu(ctx, ffn);
    ffn = linear_with_bias(ctx, ffn, layer.ffn_w2, layer.ffn_b2);

    x = ggml_add(ctx, residual, ffn);
    return x;
}

// ============================================================
// 完整 Encoder 前向传播
//
// 等价于 Python: SANMEncoderExport.forward()
//
// 流程:
//   1. 输入缩放: x *= sqrt(output_size)
//   2. encoder0 (1层, 无attention残差)
//   3. main encoders (49层, 有attention残差)
//   4. after_norm (LayerNorm)
//   5. tp encoders (20层, 有attention残差)
//   6. tp_norm (LayerNorm)
// ============================================================
ggml_tensor* encoder_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const EncoderWeights& weights,
    const EncoderConfig& cfg
) {
    // Step 1: 输入缩放
    float scale = std::sqrt(static_cast<float>(cfg.output_size));
    x = ggml_scale(ctx, x, scale);

    // Step 2: encoder0 (无 attention 残差, 560→512)
    x = encoder_layer_forward(ctx, x, weights.encoder0, cfg, false);

    // Step 3: main encoders (有 attention 残差, 512→512)
    int n_main = cfg.num_main_blocks();
    for (int i = 0; i < n_main; i++) {
        x = encoder_layer_forward(ctx, x, weights.main_layers[i], cfg, true);
    }

    // Step 4: after_norm
    x = layer_norm(ctx, x, weights.after_norm_w, weights.after_norm_b);

    // Step 5: tp encoders (有 attention 残差, 512→512)
    for (int i = 0; i < cfg.tp_blocks; i++) {
        x = encoder_layer_forward(ctx, x, weights.tp_layers[i], cfg, true);
    }

    // Step 6: tp_norm
    x = layer_norm(ctx, x, weights.tp_norm_w, weights.tp_norm_b);

    return x;
}

} // namespace funasr