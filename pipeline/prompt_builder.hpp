// funasr/pipeline/prompt_builder.hpp
// ChatML Prompt 构建器
//
// 职责:
//   1. 用 Tokenizer 将固定模板编码成 token IDs
//   2. 管理 [prefix] [audio_placeholder] [suffix] 的拼接
//   3. 构建 inputs_embeds: text embedding + audio features
//
// Prompt 结构:
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//   <|im_start|>user\n语音转写：[AUDIO_FEATURES]<|im_end|>\n
//   <|im_start|>assistant\n
//
#ifndef FUNASR_PIPELINE_PROMPT_BUILDER_HPP
#define FUNASR_PIPELINE_PROMPT_BUILDER_HPP

#include "model/tokenizer.hpp"
#include "model/weights.hpp"
#include <ggml.h>
#include <vector>
#include <string>
#include <cstring>

namespace funasr {

class PromptBuilder {
public:
    explicit PromptBuilder(const Tokenizer& tokenizer)
        : tokenizer_(tokenizer)
    {
        // 编码固定的 prefix 和 suffix
        std::string prefix_text =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n语音转写：";
        std::string suffix_text =
            "<|im_end|>\n<|im_start|>assistant\n";

        prefix_ids_ = tokenizer_.encode(prefix_text);
        suffix_ids_ = tokenizer_.encode(suffix_text);
    }

    // ============================================================
    // 构建 inputs_embeds: prefix_embed + audio_features + suffix_embed
    //
    // adaptor_out_data: adaptor 输出的 float 指针 [embed_dim, audio_frames]
    // audio_frames:     adaptor 输出帧数
    // embed_tokens:     LLM 的 embedding 权重 tensor
    // embed_dim:        embedding 维度 (1024)
    // out_embeds:       输出 buffer，调用者负责分配
    //                   大小 = total_len() * embed_dim
    //
    // 返回: 总 token 数
    // ============================================================
    int build_inputs_embeds(
        const float* adaptor_out_data,
        int audio_frames,
        ggml_tensor* embed_tokens,
        int embed_dim,
        float* out_embeds
    ) const {
        int total = total_len(audio_frames);
        int pos = 0;

        // prefix token embeddings
        for (int id : prefix_ids_) {
            get_token_embedding(embed_tokens, id, out_embeds + pos * embed_dim, embed_dim);
            pos++;
        }

        // audio features
        std::memcpy(out_embeds + pos * embed_dim,
                     adaptor_out_data,
                     audio_frames * embed_dim * sizeof(float));
        pos += audio_frames;

        // suffix token embeddings
        for (int id : suffix_ids_) {
            get_token_embedding(embed_tokens, id, out_embeds + pos * embed_dim, embed_dim);
            pos++;
        }

        return total;
    }

    // 查询
    int prefix_len()                   const { return static_cast<int>(prefix_ids_.size()); }
    int suffix_len()                   const { return static_cast<int>(suffix_ids_.size()); }
    int total_len(int audio_frames)    const { return prefix_len() + audio_frames + suffix_len(); }
    const std::vector<int>& prefix_ids() const { return prefix_ids_; }
    const std::vector<int>& suffix_ids() const { return suffix_ids_; }

    // 从 embed_tokens 获取单个 token embedding
    static void get_token_embedding(
        ggml_tensor* embed_tokens,
        int token_id,
        float* out,
        int embed_dim
    ) {
        if (embed_tokens->type == GGML_TYPE_BF16) {
            ggml_bf16_t* src = (ggml_bf16_t*)embed_tokens->data + token_id * embed_dim;
            for (int j = 0; j < embed_dim; j++) {
                out[j] = ggml_bf16_to_fp32(src[j]);
            }
        } else if (embed_tokens->type == GGML_TYPE_F32) {
            float* src = (float*)embed_tokens->data + token_id * embed_dim;
            std::memcpy(out, src, embed_dim * sizeof(float));
        } else if (embed_tokens->type == GGML_TYPE_F16) {
            ggml_fp16_t* src = (ggml_fp16_t*)embed_tokens->data + token_id * embed_dim;
            for (int j = 0; j < embed_dim; j++) {
                out[j] = ggml_fp16_to_fp32(src[j]);
            }
        }
    }

private:
    const Tokenizer& tokenizer_;
    std::vector<int> prefix_ids_;
    std::vector<int> suffix_ids_;
};

} // namespace funasr

#endif // FUNASR_PIPELINE_PROMPT_BUILDER_HPP