// funasr/model/loader.hpp
// 模型加载器 — 从 GGUFReader 绑定权重指针到结构体
//
// 设计原则：
//   1. 全部 static 方法，不持有状态
//   2. 接收 GGUFReader 引用，reader 必须在 model 使用期间保持存活
//   3. 利用 reader.require_tensor() 收集缺失的 tensor
//   4. 层数从 Config 读取，不硬编码
//
#ifndef FUNASR_MODEL_LOADER_HPP
#define FUNASR_MODEL_LOADER_HPP

#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/weights.hpp"
#include "model/model.hpp"
#include <string>

namespace funasr {

class ModelLoader {
public:
    // ============================================================
    // 一站式加载：Config + Encoder + Adaptor + LLM
    // ============================================================
    static bool load(GGUFReader& reader, FunASRModel& model);

    // ============================================================
    // 分步加载（可单独验证每个模块）
    // ============================================================
    static bool load_config(GGUFReader& reader, ModelConfig& config);

    static bool load_encoder(GGUFReader& reader, const EncoderConfig& cfg,
                             EncoderWeights& weights);

    static bool load_adaptor(GGUFReader& reader, const AdaptorConfig& cfg,
                             AdaptorWeights& weights);

    static bool load_llm(GGUFReader& reader, const LLMConfig& cfg,
                         LLMWeights& weights);

private:
    // 内部辅助：加载 encoder 单层（encoder0 / main / tp 共用）
    static void load_encoder_layer(GGUFReader& reader,
                                   EncoderLayerWeights& layer,
                                   const std::string& prefix);

    // 内部辅助：加载 adaptor block
    static void load_adaptor_block(GGUFReader& reader,
                                   AdaptorBlockWeights& block,
                                   int block_idx);

    // 内部辅助：加载 LLM 单层
    static void load_llm_layer(GGUFReader& reader,
                               LLMLayerWeights& layer,
                               int layer_idx);
};

} // namespace funasr

#endif // FUNASR_MODEL_LOADER_HPP