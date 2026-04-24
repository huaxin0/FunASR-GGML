// funasr/pipeline/pipeline.hpp
// 推理管线 — 封装完整的 Fbank → Encoder → Adaptor → LLM → Text 流程
//
// 设计原则:
//   1. 不持有模型权重（通过引用访问，GGUFReader 管理生命周期）
//   2. 每次推理创建临时 ggml_context，推理完释放
//   3. KV Cache 在 pipeline 内部管理
//   4. 支持流式回调（每生成一个 token 就通知调用者）
//
#ifndef FUNASR_PIPELINE_PIPELINE_HPP
#define FUNASR_PIPELINE_PIPELINE_HPP

#include "core/config.hpp"
#include "model/model.hpp"
#include "model/tokenizer.hpp"
#include "compute/fbank.hpp"
#include "compute/kv_cache.hpp"
#include "pipeline/prompt_builder.hpp"
#ifdef FUNASR_USE_CUDA
#include "compute/gpu_context.hpp"
#include "compute/gpu_runner.hpp"
#include "compute/encoder_adaptor_gpu.hpp"
#endif
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <memory>

namespace funasr {

// 流式 token 回调
// token_id: 当前 token ID
// text:     当前 token 解码后的文本
// is_final: true 表示 EOS，生成结束
using TokenCallback = std::function<void(int token_id, const std::string& text, bool is_final)>;

// 推理结果
struct InferenceResult {
    std::string text;               // 识别文本
    std::vector<int> token_ids;     // 生成的 token ID 序列
    float encoder_ms   = 0;        // Encoder + Adaptor 耗时
    float prefill_ms   = 0;        // LLM Prefill 耗时
    float decode_ms    = 0;        // LLM Decode 耗时
    float total_ms     = 0;        // 总耗时
    int   prefill_tokens = 0;      // Prefill token 数
    int   decode_tokens  = 0;      // 生成 token 数
};

// 推理配置
struct InferenceConfig {
    int max_new_tokens = 100;       // 最大生成 token 数
    int n_threads      = 4;         // CPU 线程数
    int kv_cache_size  = 2048;      // KV Cache 最大上下文长度
    size_t encoder_mem = 4ULL * 1024 * 1024 * 1024;  // Encoder context 内存
    size_t llm_mem     = 1024ULL * 1024 * 1024;      // LLM 每步 context 内存

    // GPU 选项
    bool use_gpu       = false;     // true = LLM 在 GPU 上跑
    int  gpu_id        = 0;         // CUDA 设备 ID
};

class Pipeline {
public:
    Pipeline(FunASRModel& model, Tokenizer& tokenizer);
    ~Pipeline();

    // ============================================================
    // GPU 初始化（可选，只在 use_gpu=true 时需要）
    // 必须在第一次 run() 之前调用
    // ============================================================
    bool init_gpu(int n_ctx = 2048, int gpu_id = 0);
    bool is_gpu_ready() const;

    // ============================================================
    // 核心接口：从 fbank 特征推理
    // ============================================================
    InferenceResult run(
        const float* fbank_data, int T, int D,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    );

    // ============================================================
    // 便捷接口：从 WAV 文件推理
    // ============================================================
    InferenceResult transcribe(
        const std::string& wav_path,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    );

    // ============================================================
    // 便捷接口：从 float 音频推理
    // ============================================================
    InferenceResult transcribe_audio(
        const float* audio, size_t n_samples,
        const InferenceConfig& config = InferenceConfig(),
        TokenCallback callback = nullptr
    );

private:
    FunASRModel& model_;
    Tokenizer& tokenizer_;
    PromptBuilder prompt_builder_;

    // CPU 推理路径
    InferenceResult run_cpu(
        const float* adaptor_data, int audio_frames,
        const InferenceConfig& config, TokenCallback callback);

    // GPU 推理路径 (adaptor 数据在 GPU 上)
    InferenceResult run_gpu(
        ggml_tensor* gpu_adaptor_tensor, int audio_frames,
        const float* cpu_adaptor_data,   // fallback: 没有 GPU tensor 时用
        const InferenceConfig& config, TokenCallback callback);

    static int argmax(const float* logits, int vocab_size);

#ifdef FUNASR_USE_CUDA
    std::unique_ptr<GPUContext> gpu_ctx_;
    std::unique_ptr<GPURunner>  gpu_runner_;
    std::unique_ptr<GPUEncoderAdaptorRunner> gpu_ea_runner_;

    // Prefill staging: 持久化的 GPU tensor 用于拼接 inputs_embeds
    // 避免每次推理都 alloc/free
    ggml_context* prefill_stg_ctx_ = nullptr;
    ggml_backend_buffer_t prefill_stg_buf_ = nullptr;
    ggml_tensor* prefill_stg_tensor_ = nullptr;
    int prefill_stg_max_len_ = 0;  // 当前 staging 能容纳的最大 total_len

    bool ensure_prefill_staging(int total_len, int embed_dim);
    void free_prefill_staging();
#endif
};

} // namespace funasr

#endif // FUNASR_PIPELINE_PIPELINE_HPP
