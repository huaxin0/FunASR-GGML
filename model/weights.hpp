
// 设计原则：
//   1. 每个结构体只包含 ggml_tensor* 指针，不管理生命周期
//   2. 指针的有效性由 GGUFReader 的生命周期保证
//   3. is_valid() 检查全部指针非空，一次性报告完整性
//   4. 字段命名统一：组件_角色_后缀 (如 attn_q_w)
//
// 命名对照（结构体字段 ↔ GGUF tensor name）：
//   attn_q_w   ↔  *.self_attn.linear_q.weight
//   attn_q_b   ↔  *.self_attn.linear_q.bias
//   ffn_w1     ↔  *.feed_forward.w_1.weight
//   fsmn_w     ↔  *.self_attn.fsmn_block.weight
//   等等...
//   字段名是代码内部的事，GGUF 里的名字由 loader 来构造
//
#ifndef FUNASR_MODEL_WEIGHTS_HPP
#define FUNASR_MODEL_WEIGHTS_HPP

#include <ggml.h>
#include <vector>
#include <cstdio>

namespace funasr {

// ============================================================
// Encoder 单层权重（SANM + FSMN）
// encoder0 / main encoders / tp encoders 共用
//
// 每层 17 个 tensor：
//   Attention: Q/K/V/Out × (weight+bias) + FSMN = 9
//   FFN: w1/w2 × (weight+bias) = 4
//   Norm: norm1/norm2 × (weight+bias) = 4
// ============================================================
struct EncoderLayerWeights {
    // Self-Attention (SANM)
    ggml_tensor* attn_q_w      = nullptr;
    ggml_tensor* attn_q_b      = nullptr;
    ggml_tensor* attn_k_w      = nullptr;
    ggml_tensor* attn_k_b      = nullptr;
    ggml_tensor* attn_v_w      = nullptr;
    ggml_tensor* attn_v_b      = nullptr;
    ggml_tensor* attn_out_w    = nullptr;
    ggml_tensor* attn_out_b    = nullptr;
    ggml_tensor* fsmn_w        = nullptr;  // [kernel_size, 1, dim]

    // Feed Forward (ReLU activation)
    ggml_tensor* ffn_w1        = nullptr;  // [dim, ffn_dim]
    ggml_tensor* ffn_b1        = nullptr;
    ggml_tensor* ffn_w2        = nullptr;  // [ffn_dim, dim]
    ggml_tensor* ffn_b2        = nullptr;

    // Layer Norm
    ggml_tensor* norm1_w       = nullptr;  // pre-attention norm
    ggml_tensor* norm1_b       = nullptr;
    ggml_tensor* norm2_w       = nullptr;  // pre-FFN norm
    ggml_tensor* norm2_b       = nullptr;

    bool is_valid() const {
        return attn_q_w && attn_q_b && attn_k_w && attn_k_b
            && attn_v_w && attn_v_b && attn_out_w && attn_out_b
            && fsmn_w
            && ffn_w1 && ffn_b1 && ffn_w2 && ffn_b2
            && norm1_w && norm1_b && norm2_w && norm2_b;
    }

    // 打印关键 tensor 的 shape（调试用）
    void print_shapes(const char* label) const {
        if (!attn_q_w) {
            printf("  %s: NOT LOADED\n", label);
            return;
        }
        printf("  %s:\n", label);
        printf("    attn_q_w:  [%lld, %lld]  (%s)\n",
               (long long)attn_q_w->ne[0], (long long)attn_q_w->ne[1],
               ggml_type_name(attn_q_w->type));
        printf("    fsmn_w:    [%lld, %lld, %lld]  (%s)\n",
               (long long)fsmn_w->ne[0], (long long)fsmn_w->ne[1],
               (long long)fsmn_w->ne[2], ggml_type_name(fsmn_w->type));
        printf("    ffn_w1:    [%lld, %lld]  (%s)\n",
               (long long)ffn_w1->ne[0], (long long)ffn_w1->ne[1],
               ggml_type_name(ffn_w1->type));
        printf("    norm1_w:   [%lld]  (%s)\n",
               (long long)norm1_w->ne[0], ggml_type_name(norm1_w->type));
    }
};

// ============================================================
// 完整 Audio Encoder 权重
// encoder0 (1层) + main encoders (49层) + tp encoders (20层)
// + after_norm + tp_norm
//
// 总计: 70 × 17 + 4 = 1194 个 tensor
// ============================================================
struct EncoderWeights {
    EncoderLayerWeights encoder0;                      // 特殊第一层 (input 560→512)
    std::vector<EncoderLayerWeights> main_layers;      // 49 层 (512→512)
    std::vector<EncoderLayerWeights> tp_layers;        // 20 层 (512→512)

    // 顶层 Norm
    ggml_tensor* after_norm_w  = nullptr;  // encoders 之后、tp 之前
    ggml_tensor* after_norm_b  = nullptr;
    ggml_tensor* tp_norm_w     = nullptr;  // tp_encoders 之后
    ggml_tensor* tp_norm_b     = nullptr;

    bool is_valid() const {
        if (!encoder0.is_valid()) return false;

        for (const auto& layer : main_layers) {
            if (!layer.is_valid()) return false;
        }
        for (const auto& layer : tp_layers) {
            if (!layer.is_valid()) return false;
        }

        return after_norm_w && after_norm_b
            && tp_norm_w && tp_norm_b;
    }

    // 统计 tensor 总数
    int tensor_count() const {
        int per_layer = 17;
        return per_layer                                   // encoder0
             + (int)main_layers.size() * per_layer         // main
             + (int)tp_layers.size() * per_layer           // tp
             + 4;                                          // norms
    }
};

// ============================================================
// Adaptor 单层权重（标准 Multi-head Attention，无 FSMN）
//
// 每层 16 个 tensor：
//   Attention: Q/K/V/Out × (weight+bias) = 8
//   FFN: w1/w2 × (weight+bias) = 4
//   Norm: norm1/norm2 × (weight+bias) = 4
// ============================================================
struct AdaptorBlockWeights {
    // Self-Attention（标准 MHA，无 FSMN）
    ggml_tensor* attn_q_w      = nullptr;  // [1024, 1024]
    ggml_tensor* attn_q_b      = nullptr;
    ggml_tensor* attn_k_w      = nullptr;
    ggml_tensor* attn_k_b      = nullptr;
    ggml_tensor* attn_v_w      = nullptr;
    ggml_tensor* attn_v_b      = nullptr;
    ggml_tensor* attn_out_w    = nullptr;
    ggml_tensor* attn_out_b    = nullptr;

    // Feed Forward (注意: 中间维度是 256, 不是 2048)
    ggml_tensor* ffn_w1        = nullptr;  // [1024, 256]
    ggml_tensor* ffn_b1        = nullptr;
    ggml_tensor* ffn_w2        = nullptr;  // [256, 1024]
    ggml_tensor* ffn_b2        = nullptr;

    // Layer Norm
    ggml_tensor* norm1_w       = nullptr;
    ggml_tensor* norm1_b       = nullptr;
    ggml_tensor* norm2_w       = nullptr;
    ggml_tensor* norm2_b       = nullptr;

    bool is_valid() const {
        return attn_q_w && attn_q_b && attn_k_w && attn_k_b
            && attn_v_w && attn_v_b && attn_out_w && attn_out_b
            && ffn_w1 && ffn_b1 && ffn_w2 && ffn_b2
            && norm1_w && norm1_b && norm2_w && norm2_b;
    }
};

// ============================================================
// 完整 Audio Adaptor 权重
// linear1 → ReLU → linear2 → 2 blocks (attention + FFN)
//
// 总计: 4 + 2 × 16 = 36 个 tensor
// ============================================================
struct AdaptorWeights {
    ggml_tensor* linear1_w     = nullptr;  // [512, 2048]
    ggml_tensor* linear1_b     = nullptr;
    ggml_tensor* linear2_w     = nullptr;  // [2048, 1024]
    ggml_tensor* linear2_b     = nullptr;

    std::vector<AdaptorBlockWeights> blocks;  // 2 层

    bool is_valid() const {
        if (!linear1_w || !linear1_b || !linear2_w || !linear2_b)
            return false;
        for (const auto& block : blocks) {
            if (!block.is_valid()) return false;
        }
        return true;
    }

    int tensor_count() const {
        return 4 + (int)blocks.size() * 16;
    }
};

// ============================================================
// LLM 单层权重（Qwen3 Transformer with GQA）
//
// 每层 11 个 tensor：
//   GQA Attention: q/k/v/o_proj weight + q/k_norm weight = 6
//   SwiGLU MLP: gate/up/down_proj weight = 3
//   RMS Norm: input_norm + post_attn_norm = 2
//
// 注意：Qwen3 的 attention 没有 bias
// ============================================================
struct LLMLayerWeights {
    // GQA Attention (no bias)
    ggml_tensor* q_proj_w      = nullptr;  // [1024, 2048] = [hidden, n_heads × head_dim]
    ggml_tensor* k_proj_w      = nullptr;  // [1024, 1024] = [hidden, n_kv_heads × head_dim]
    ggml_tensor* v_proj_w      = nullptr;  // [1024, 1024]
    ggml_tensor* o_proj_w      = nullptr;  // [2048, 1024]
    ggml_tensor* q_norm_w      = nullptr;  // [128] = [head_dim]
    ggml_tensor* k_norm_w      = nullptr;  // [128]

    // SwiGLU MLP (no bias)
    ggml_tensor* gate_proj_w   = nullptr;  // [1024, 3072]
    ggml_tensor* up_proj_w     = nullptr;  // [1024, 3072]
    ggml_tensor* down_proj_w   = nullptr;  // [3072, 1024]

    // RMS Norm
    ggml_tensor* input_norm_w       = nullptr;  // [1024]
    ggml_tensor* post_attn_norm_w   = nullptr;  // [1024]

    bool is_valid() const {
        return q_proj_w && k_proj_w && v_proj_w && o_proj_w
            && q_norm_w && k_norm_w
            && gate_proj_w && up_proj_w && down_proj_w
            && input_norm_w && post_attn_norm_w;
    }

    void print_shapes(const char* label) const {
        if (!q_proj_w) {
            printf("  %s: NOT LOADED\n", label);
            return;
        }
        printf("  %s:\n", label);
        printf("    q_proj_w:  [%lld, %lld]  (%s)\n",
               (long long)q_proj_w->ne[0], (long long)q_proj_w->ne[1],
               ggml_type_name(q_proj_w->type));
        printf("    k_proj_w:  [%lld, %lld]  (%s)\n",
               (long long)k_proj_w->ne[0], (long long)k_proj_w->ne[1],
               ggml_type_name(k_proj_w->type));
        printf("    o_proj_w:  [%lld, %lld]  (%s)\n",
               (long long)o_proj_w->ne[0], (long long)o_proj_w->ne[1],
               ggml_type_name(o_proj_w->type));
        printf("    q_norm_w:  [%lld]  (%s)\n",
               (long long)q_norm_w->ne[0], ggml_type_name(q_norm_w->type));
        printf("    gate_proj: [%lld, %lld]  (%s)\n",
               (long long)gate_proj_w->ne[0], (long long)gate_proj_w->ne[1],
               ggml_type_name(gate_proj_w->type));
    }
};

// ============================================================
// 完整 LLM Decoder 权重
// embed_tokens + 28 layers + model_norm + lm_head
//
// 总计: 1 + 28 × 11 + 1 + 1 = 311 个 tensor
// ============================================================
struct LLMWeights {
    ggml_tensor* embed_tokens   = nullptr;  // [1024, 151936]
    std::vector<LLMLayerWeights> layers;    // 28 层
    ggml_tensor* model_norm_w   = nullptr;  // [1024]
    ggml_tensor* lm_head_w      = nullptr;  // [1024, 151936]

    bool is_valid() const {
        if (!embed_tokens || !model_norm_w || !lm_head_w)
            return false;
        for (const auto& layer : layers) {
            if (!layer.is_valid()) return false;
        }
        return true;
    }

    int tensor_count() const {
        return 3 + (int)layers.size() * 11;
    }
};

} // namespace funasr

#endif // FUNASR_MODEL_WEIGHTS_HPP