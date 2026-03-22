// funasr/pipeline/recognizer.hpp
// Recognizer — 面向用户的一站式语音识别 API
//
// 使用方式:
//   funasr::Recognizer recognizer;
//   recognizer.init("FunAsr_q8.bin");
//   auto result = recognizer.transcribe("audio.wav");
//   printf("%s\n", result.text.c_str());
//
// 一个 Recognizer 持有完整的生命周期链:
//   GGUFReader → FunASRModel → Tokenizer → Pipeline
//
#ifndef FUNASR_PIPELINE_RECOGNIZER_HPP
#define FUNASR_PIPELINE_RECOGNIZER_HPP

#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "model/tokenizer.hpp"
#include "pipeline/pipeline.hpp"
#include <memory>
#include <string>

namespace funasr {

class Recognizer {
public:
    Recognizer() = default;
    ~Recognizer() = default;

    // 不可拷贝（持有 GGUFReader）
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // ============================================================
    // 初始化: 加载模型 + 词表 + 创建 Pipeline
    // ============================================================
    bool init(const std::string& model_path, bool verbose = true) {
        if (verbose) printf("[Recognizer] Loading model: %s\n", model_path.c_str());

        // 1. 打开 GGUF
        reader_ = std::make_unique<GGUFReader>();
        if (!reader_->open(model_path)) {
            last_error_ = "Failed to open model file";
            return false;
        }

        // 2. 加载模型权重
        model_ = std::make_unique<FunASRModel>();
        if (!ModelLoader::load(*reader_, *model_)) {
            last_error_ = "Failed to load model weights";
            return false;
        }

        // 3. 加载 Tokenizer
        tokenizer_ = std::make_unique<Tokenizer>();
        if (!tokenizer_->load(*reader_)) {
            last_error_ = "Failed to load tokenizer";
            return false;
        }

        // 4. 创建 Pipeline
        pipeline_ = std::make_unique<Pipeline>(*model_, *tokenizer_);

        ready_ = true;
        if (verbose) {
            printf("[Recognizer] Ready!\n");
            printf("  Vocab: %d tokens\n", tokenizer_->vocab_size());
        }
        return true;
    }

    // ============================================================
    // GPU 初始化（可选，在 init() 之后调用）
    // ============================================================
    bool init_gpu(int n_ctx = 2048, int gpu_id = 0) {
        if (!ready_ || !pipeline_) {
            last_error_ = "Call init() before init_gpu()";
            return false;
        }
        if (!pipeline_->init_gpu(n_ctx, gpu_id)) {
            last_error_ = "Failed to init GPU";
            return false;
        }
        return true;
    }

    bool is_gpu_ready() const {
        return pipeline_ && pipeline_->is_gpu_ready();
    }

    // ============================================================
    // 从 WAV 文件识别
    // ============================================================
    InferenceResult transcribe(
        const std::string& wav_path,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    ) {
        if (!ready_) {
            last_error_ = "Not initialized";
            return InferenceResult{};
        }
        return pipeline_->transcribe(wav_path, config, callback);
    }

    // ============================================================
    // 从 float 音频识别
    // ============================================================
    InferenceResult transcribe_audio(
        const float* audio, size_t n_samples,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    ) {
        if (!ready_) {
            last_error_ = "Not initialized";
            return InferenceResult{};
        }
        return pipeline_->transcribe_audio(audio, n_samples, config, callback);
    }

    // ============================================================
    // 从 fbank 特征识别
    // ============================================================
    InferenceResult run(
        const float* fbank_data, int T, int D,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    ) {
        if (!ready_) {
            last_error_ = "Not initialized";
            return InferenceResult{};
        }
        return pipeline_->run(fbank_data, T, D, config, callback);
    }

    // 查询
    bool is_ready()                   const { return ready_; }
    const std::string& last_error()   const { return last_error_; }
    const ModelConfig& config()       const { return model_->config; }
    const Tokenizer& tokenizer()      const { return *tokenizer_; }

private:
    std::unique_ptr<GGUFReader>   reader_;
    std::unique_ptr<FunASRModel>  model_;
    std::unique_ptr<Tokenizer>    tokenizer_;
    std::unique_ptr<Pipeline>     pipeline_;

    bool ready_ = false;
    std::string last_error_;
};

} // namespace funasr

#endif // FUNASR_PIPELINE_RECOGNIZER_HPP