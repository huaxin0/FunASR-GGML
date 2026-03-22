// funasr/model/model.hpp
// 聚合模型结构体 — 把 Config + 三大模块权重打包
#ifndef FUNASR_MODEL_MODEL_HPP
#define FUNASR_MODEL_MODEL_HPP

#include "core/config.hpp"
#include "model/weights.hpp"

namespace funasr {

struct FunASRModel {
    ModelConfig    config;
    EncoderWeights encoder;
    AdaptorWeights adaptor;
    LLMWeights     llm;

    bool is_valid() const {
        return encoder.is_valid() && adaptor.is_valid() && llm.is_valid();
    }

    // 各模块 tensor 计数
    int tensor_count() const {
        return encoder.tensor_count() + adaptor.tensor_count() + llm.tensor_count();
    }

    void print_summary() const {
        printf("\n--- Model Summary ---\n");
        printf("  Encoder:  %d tensors  %s\n",
               encoder.tensor_count(), encoder.is_valid() ? "[OK]" : "[INCOMPLETE]");
        printf("  Adaptor:  %d tensors  %s\n",
               adaptor.tensor_count(), adaptor.is_valid() ? "[OK]" : "[INCOMPLETE]");
        printf("  LLM:      %d tensors  %s\n",
               llm.tensor_count(), llm.is_valid() ? "[OK]" : "[INCOMPLETE]");
        printf("  Total:    %d tensors  %s\n",
               tensor_count(), is_valid() ? "[ALL OK]" : "[INCOMPLETE]");
    }
};

} // namespace funasr

#endif // FUNASR_MODEL_MODEL_HPP