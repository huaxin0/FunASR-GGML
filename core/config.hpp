#ifndef FUNASR_CORE_CONFIG_HPP
#define FUNASR_CORE_CONFIG_HPP

#include <gguf.h>
#include <string>
#include <cstdio>

namespace funasr {

// ============================================================
// GGUF metadata 安全读取辅助函数
// 找不到 key 时返回 default，不会崩溃
// ============================================================
namespace detail {

inline int gguf_get_i32_or(struct gguf_context* ctx, const char* key, int default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_i32(ctx, idx);
}

inline uint32_t gguf_get_u32_or(struct gguf_context* ctx, const char* key, uint32_t default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_u32(ctx, idx);
}

inline float gguf_get_f32_or(struct gguf_context* ctx, const char* key, float default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx, idx);
}

inline std::string gguf_get_str_or(struct gguf_context* ctx, const char* key, const char* default_val) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) return default_val;
    return gguf_get_val_str(ctx, idx);
}

// 必须存在的 key，找不到返回 false 并打印错误
inline bool gguf_require_i32(struct gguf_context* ctx, const char* key, int& out) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        printf("[Config] ERROR: required key '%s' not found in GGUF\n", key);
        return false;
    }
    out = gguf_get_val_i32(ctx, idx);
    return true;
}

inline bool gguf_require_u32(struct gguf_context* ctx, const char* key, uint32_t& out) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        printf("[Config] ERROR: required key '%s' not found in GGUF\n", key);
        return false;
    }
    out = gguf_get_val_u32(ctx, idx);
    return true;
}

} // namespace detail

// ============================================================
// Frontend 配置 (Fbank 特征提取)
// GGUF keys: frontend.*
// ============================================================
struct FrontendConfig {
    int sample_rate  = 16000;
    int num_mels     = 80;
    int frame_length = 25;     // ms
    int frame_shift  = 10;     // ms
    int lfr_m        = 7;      // LFR 拼接帧数
    int lfr_n        = 6;      // LFR 跳帧步长
    std::string window = "hamming";

    // 派生值: Fbank 经过 LFR 后的特征维度 = encoder0 的输入维度
    int input_dim() const { return num_mels * lfr_m; }  // 80 * 7 = 560
};

// ============================================================
// Audio Encoder 配置 (SANM 架构)
// GGUF keys: encoder.*
// ============================================================
struct EncoderConfig {
    int output_size     = 512;   // encoder 输出维度
    int attention_heads = 4;     // SANM attention 头数
    int linear_units   = 2048;   // FFN 中间维度
    int num_blocks     = 50;     // encoder0(1) + main encoders(49)
    int tp_blocks      = 20;     // TP encoder 层数
    int kernel_size    = 11;     // FSMN 卷积核大小
    int sanm_shift     = 0;      // FSMN shift

    // 派生值
    int num_main_blocks() const { return num_blocks - 1; }         // 49
    int head_dim()        const { return output_size / attention_heads; } // 128
    int fsmn_pad()        const { return (kernel_size - 1) / 2 - sanm_shift; } // 5
};

// ============================================================
// Audio Adaptor 配置
// GGUF metadata 中没有 adaptor 参数
// 这些值从 encoder/LLM 配置推导
// ============================================================
struct AdaptorConfig {
    int input_dim       = 512;   // = encoder.output_size
    int hidden_dim      = 2048;  // linear1 输出维度
    int output_dim      = 1024;  // = llm.embedding_length
    int num_blocks      = 2;     // attention block 数量
    int attention_heads = 8;     // output_dim / 128
    int ffn_dim         = 256;   // attention block 内 FFN 的中间维度

    int head_dim() const { return output_dim / attention_heads; }  // 128
};

// ============================================================
// LLM Decoder 配置 (Qwen3-0.6B)
// GGUF keys: FunAsr.*
// ============================================================
struct LLMConfig {
    int      block_count         = 28;
    int      embedding_length    = 1024;
    int      context_length      = 40960;
    int      feed_forward_length = 3072;
    int      head_count          = 16;     // Q heads
    int      head_count_kv       = 8;      // KV heads (GQA)
    int      key_length          = 128;    // head_dim
    int      value_length        = 128;
    float    rope_freq_base      = 1000000.0f;
    int      vocab_size          = 151936;

    // 派生值
    int   head_dim()   const { return key_length; }                     // 128
    int   q_dim()      const { return head_count * head_dim(); }        // 2048
    int   kv_dim()     const { return head_count_kv * head_dim(); }     // 1024
    int   gqa_groups() const { return head_count / head_count_kv; }     // 2
};

// ============================================================
// 完整模型配置
// ============================================================
struct ModelConfig {
    FrontendConfig frontend;
    EncoderConfig  encoder;
    AdaptorConfig  adaptor;
    LLMConfig      llm;
    int            file_type = 1;

    // 从 GGUF metadata 填充所有配置
    // 返回 false 表示缺少关键参数
    bool load_from_gguf(struct gguf_context* gctx) {
        if (!gctx) {
            printf("[Config] ERROR: gguf_context is null\n");
            return false;
        }

        bool ok = true;

        // ===== Frontend =====
        frontend.sample_rate  = detail::gguf_get_i32_or(gctx, "frontend.sample_rate", 16000);
        frontend.num_mels     = detail::gguf_get_i32_or(gctx, "frontend.num_mels", 80);
        frontend.frame_length = detail::gguf_get_i32_or(gctx, "frontend.frame_length", 25);
        frontend.frame_shift  = detail::gguf_get_i32_or(gctx, "frontend.frame_shift", 10);
        frontend.lfr_m        = detail::gguf_get_i32_or(gctx, "frontend.lfr_m", 7);
        frontend.lfr_n        = detail::gguf_get_i32_or(gctx, "frontend.lfr_n", 6);
        frontend.window       = detail::gguf_get_str_or(gctx, "frontend.window", "hamming");

        int tmp_i;
        if (detail::gguf_require_i32(gctx, "encoder.output_size", tmp_i))
            encoder.output_size = tmp_i;
        else ok = false;

        if (detail::gguf_require_i32(gctx, "encoder.attention_heads", tmp_i))
            encoder.attention_heads = tmp_i;
        else ok = false;

        if (detail::gguf_require_i32(gctx, "encoder.linear_units", tmp_i))
            encoder.linear_units = tmp_i;
        else ok = false;

        if (detail::gguf_require_i32(gctx, "encoder.num_blocks", tmp_i))
            encoder.num_blocks = tmp_i;
        else ok = false;

        if (detail::gguf_require_i32(gctx, "encoder.tp_blocks", tmp_i))
            encoder.tp_blocks = tmp_i;
        else ok = false;

        encoder.kernel_size = detail::gguf_get_i32_or(gctx, "encoder.kernel_size", 11);
        encoder.sanm_shift  = detail::gguf_get_i32_or(gctx, "encoder.sanm_shift", 0);

        // ===== LLM (从 FunAsr.* keys 读取，这些是必需的) =====
        uint32_t tmp_u;
        if (detail::gguf_require_u32(gctx, "FunAsr.block_count", tmp_u))
            llm.block_count = static_cast<int>(tmp_u);
        else ok = false;

        if (detail::gguf_require_u32(gctx, "FunAsr.embedding_length", tmp_u))
            llm.embedding_length = static_cast<int>(tmp_u);
        else ok = false;

        if (detail::gguf_require_u32(gctx, "FunAsr.context_length", tmp_u))
            llm.context_length = static_cast<int>(tmp_u);
        else ok = false;

        if (detail::gguf_require_u32(gctx, "FunAsr.feed_forward_length", tmp_u))
            llm.feed_forward_length = static_cast<int>(tmp_u);
        else ok = false;

        if (detail::gguf_require_u32(gctx, "FunAsr.attention.head_count", tmp_u))
            llm.head_count = static_cast<int>(tmp_u);
        else ok = false;

        if (detail::gguf_require_u32(gctx, "FunAsr.attention.head_count_kv", tmp_u))
            llm.head_count_kv = static_cast<int>(tmp_u);
        else ok = false;

        llm.rope_freq_base = detail::gguf_get_f32_or(gctx, "FunAsr.rope.freq_base", 1000000.0f);

        llm.key_length   = static_cast<int>(
            detail::gguf_get_u32_or(gctx, "FunAsr.attention.key_length", 128));
        llm.value_length = static_cast<int>(
            detail::gguf_get_u32_or(gctx, "FunAsr.attention.value_length", 128));

        
        llm.vocab_size = 151936;

        file_type = static_cast<int>(
            detail::gguf_get_u32_or(gctx, "general.file_type", 1));

        // ===== Adaptor (从 encoder 和 LLM 推导) =====
        adaptor.input_dim       = encoder.output_size;       // 512
        adaptor.output_dim      = llm.embedding_length;      // 1024
        adaptor.hidden_dim      = 2048;                      // 固定
        adaptor.num_blocks      = 2;                         // 固定
        adaptor.ffn_dim         = 256;                       // 固定
        adaptor.attention_heads = adaptor.output_dim / llm.key_length;  // 1024/128 = 8

        return ok;
    }

    // 打印所有配置参数（验证用）
    void print() const {
        printf("\n");
        printf("========================================\n");
        printf("  FunASR Model Configuration\n");
        printf("========================================\n");

        printf("\n--- Frontend ---\n");
        printf("  sample_rate:  %d\n", frontend.sample_rate);
        printf("  num_mels:     %d\n", frontend.num_mels);
        printf("  frame_length: %d ms\n", frontend.frame_length);
        printf("  frame_shift:  %d ms\n", frontend.frame_shift);
        printf("  lfr_m:        %d\n", frontend.lfr_m);
        printf("  lfr_n:        %d\n", frontend.lfr_n);
        printf("  window:       %s\n", frontend.window.c_str());
        printf("  input_dim:    %d (= %d × %d)\n",
               frontend.input_dim(), frontend.num_mels, frontend.lfr_m);

        printf("\n--- Encoder (SANM) ---\n");
        printf("  output_size:      %d\n", encoder.output_size);
        printf("  attention_heads:  %d\n", encoder.attention_heads);
        printf("  head_dim:         %d\n", encoder.head_dim());
        printf("  linear_units:     %d\n", encoder.linear_units);
        printf("  num_blocks:       %d (encoder0: 1, main: %d)\n",
               encoder.num_blocks, encoder.num_main_blocks());
        printf("  tp_blocks:        %d\n", encoder.tp_blocks);
        printf("  kernel_size:      %d\n", encoder.kernel_size);
        printf("  sanm_shift:       %d\n", encoder.sanm_shift);
        printf("  fsmn_pad:         %d\n", encoder.fsmn_pad());

        printf("\n--- Adaptor ---\n");
        printf("  input_dim:        %d\n", adaptor.input_dim);
        printf("  hidden_dim:       %d\n", adaptor.hidden_dim);
        printf("  output_dim:       %d\n", adaptor.output_dim);
        printf("  num_blocks:       %d\n", adaptor.num_blocks);
        printf("  attention_heads:  %d\n", adaptor.attention_heads);
        printf("  head_dim:         %d\n", adaptor.head_dim());
        printf("  ffn_dim:          %d\n", adaptor.ffn_dim);

        printf("\n--- LLM Decoder (Qwen3) ---\n");
        printf("  block_count:      %d\n", llm.block_count);
        printf("  embedding_length: %d\n", llm.embedding_length);
        printf("  context_length:   %d\n", llm.context_length);
        printf("  feed_forward:     %d\n", llm.feed_forward_length);
        printf("  head_count:       %d (Q heads)\n", llm.head_count);
        printf("  head_count_kv:    %d (KV heads, GQA-%d)\n",
               llm.head_count_kv, llm.gqa_groups());
        printf("  head_dim:         %d\n", llm.head_dim());
        printf("  q_dim:            %d (= %d × %d)\n",
               llm.q_dim(), llm.head_count, llm.head_dim());
        printf("  kv_dim:           %d (= %d × %d)\n",
               llm.kv_dim(), llm.head_count_kv, llm.head_dim());
        printf("  rope_freq_base:   %.1f\n", llm.rope_freq_base);
        printf("  vocab_size:       %d\n", llm.vocab_size);

        printf("\n--- General ---\n");
        printf("  file_type:        %d\n", file_type);
        printf("========================================\n\n");
    }
};

} // namespace funasr

#endif // FUNASR_CORE_CONFIG_HPP