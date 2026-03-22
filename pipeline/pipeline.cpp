// funasr/pipeline/pipeline.cpp
// 推理管线实现 — 支持 CPU 和 GPU 两条路径
//
// CPU 路径: llm_forward + KVCache (memcpy commit)
// GPU 路径: GPURunner (ggml_cpy in graph, gallocr)
//
// Encoder + Adaptor 始终在 CPU 上跑
//
#include "pipeline/pipeline.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
#include "compute/llm_ops.hpp"
#include "compute/graph_runner.hpp"
#include <ggml.h>
#include <cstring>
#include <cstdio>

namespace funasr {

Pipeline::Pipeline(FunASRModel& model, Tokenizer& tokenizer)
    : model_(model)
    , tokenizer_(tokenizer)
    , prompt_builder_(tokenizer)
{}

Pipeline::~Pipeline() = default;

int Pipeline::argmax(const float* logits, int vocab_size) {
    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================
// GPU 初始化
// ============================================================
bool Pipeline::init_gpu(int n_ctx, int gpu_id) {
#ifdef FUNASR_USE_CUDA
    gpu_ctx_ = std::make_unique<GPUContext>();
    if (!gpu_ctx_->init(model_.llm, model_.config.llm, n_ctx, gpu_id)) {
        gpu_ctx_.reset();
        return false;
    }

    gpu_runner_ = std::make_unique<GPURunner>(*gpu_ctx_);
    if (!gpu_runner_->warmup(500, n_ctx - 100)) {
        printf("[Pipeline] WARNING: GPU warmup failed\n");
    }

    return true;
#else
    (void)n_ctx; (void)gpu_id;
    printf("[Pipeline] CUDA not compiled\n");
    return false;
#endif
}

bool Pipeline::is_gpu_ready() const {
#ifdef FUNASR_USE_CUDA
    return gpu_ctx_ && gpu_ctx_->is_initialized() && gpu_runner_;
#else
    return false;
#endif
}

// ============================================================
// 核心: fbank → encoder → adaptor → (CPU or GPU) LLM → text
// ============================================================
InferenceResult Pipeline::run(
    const float* fbank_data, int T, int D,
    const InferenceConfig& config,
    TokenCallback callback
) {
    InferenceResult result;
    auto t_total_start = std::chrono::high_resolution_clock::now();

    const int embed_dim = model_.config.llm.embedding_length;

    // ================================================================
    // Phase 0: Encoder + Adaptor (始终 CPU)
    // ================================================================
    std::vector<float> adaptor_data;
    int audio_frames = 0;
    {
        struct ggml_init_params params = { config.encoder_mem, nullptr, false };
        struct ggml_context* ctx_ea = ggml_init(params);
        if (!ctx_ea) {
            printf("[Pipeline] ERROR: failed to alloc encoder context\n");
            return result;
        }

        struct ggml_tensor* fbank_t = ggml_new_tensor_2d(ctx_ea, GGML_TYPE_F32, D, T);
        std::memcpy(fbank_t->data, fbank_data, T * D * sizeof(float));

        struct ggml_tensor* enc_out = encoder_forward(
            ctx_ea, fbank_t, model_.encoder, model_.config.encoder);
        struct ggml_tensor* adp_out = adaptor_forward(
            ctx_ea, enc_out, model_.adaptor, model_.config.adaptor);

        auto t0 = std::chrono::high_resolution_clock::now();
        run_graph(ctx_ea, adp_out, config.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        result.encoder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        audio_frames = static_cast<int>(adp_out->ne[1]);
        adaptor_data.resize(audio_frames * embed_dim);
        std::memcpy(adaptor_data.data(), adp_out->data,
                     audio_frames * embed_dim * sizeof(float));

        ggml_free(ctx_ea);
    }

    // ================================================================
    // Phase 1-3: LLM (CPU 或 GPU)
    // ================================================================
    InferenceResult llm_result;
    if (config.use_gpu && is_gpu_ready()) {
        llm_result = run_gpu(adaptor_data.data(), audio_frames, config, callback);
    } else {
        if (config.use_gpu && !is_gpu_ready()) {
            printf("[Pipeline] WARNING: GPU requested but not ready, falling back to CPU\n");
        }
        llm_result = run_cpu(adaptor_data.data(), audio_frames, config, callback);
    }

    // 合并结果
    result.text            = llm_result.text;
    result.token_ids       = llm_result.token_ids;
    result.prefill_ms      = llm_result.prefill_ms;
    result.decode_ms       = llm_result.decode_ms;
    result.prefill_tokens  = llm_result.prefill_tokens;
    result.decode_tokens   = llm_result.decode_tokens;

    auto t_total_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_total_end - t_total_start).count();

    return result;
}

// ============================================================
// CPU 路径
// ============================================================
InferenceResult Pipeline::run_cpu(
    const float* adaptor_data, int audio_frames,
    const InferenceConfig& config, TokenCallback callback
) {
    InferenceResult result;
    const int embed_dim  = model_.config.llm.embedding_length;
    const int vocab_size = model_.config.llm.vocab_size;
    const int eos_id     = tokenizer_.eos_id();

    // Build inputs_embeds
    int total_len = prompt_builder_.total_len(audio_frames);
    result.prefill_tokens = total_len;

    std::vector<float> inputs_embeds(total_len * embed_dim);
    prompt_builder_.build_inputs_embeds(
        adaptor_data, audio_frames,
        model_.llm.embed_tokens, embed_dim,
        inputs_embeds.data()
    );

    // Prefill
    KVCache cache;
    if (!cache.init(model_.config.llm, config.kv_cache_size)) return result;

    std::vector<float> logits_buf(vocab_size);

    {
        struct ggml_init_params params = { config.llm_mem, nullptr, false };
        struct ggml_context* ctx_pf = ggml_init(params);
        if (!ctx_pf) return result;

        struct ggml_tensor* embeds_t = ggml_new_tensor_2d(
            ctx_pf, GGML_TYPE_F32, embed_dim, total_len);
        std::memcpy(embeds_t->data, inputs_embeds.data(),
                     total_len * embed_dim * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        struct ggml_tensor* logits = llm_forward(
            ctx_pf, embeds_t, model_.llm, cache, 0, model_.config.llm);
        run_graph(ctx_pf, logits, config.n_threads);

        cache.commit(0, total_len);
        cache.set_n_past(total_len);

        auto t1 = std::chrono::high_resolution_clock::now();
        result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        float* last = (float*)logits->data + (total_len - 1) * vocab_size;
        std::memcpy(logits_buf.data(), last, vocab_size * sizeof(float));
        ggml_free(ctx_pf);
    }

    inputs_embeds.clear();
    inputs_embeds.shrink_to_fit();

    // Decode
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < config.max_new_tokens; step++) {
        int next_token = argmax(logits_buf.data(), vocab_size);
        if (next_token == eos_id) {
            if (callback) callback(next_token, "", true);
            break;
        }
        result.token_ids.push_back(next_token);
        if (callback) callback(next_token, tokenizer_.decode({next_token}), false);

        struct ggml_init_params params = { config.llm_mem, nullptr, false };
        struct ggml_context* ctx_dec = ggml_init(params);
        if (!ctx_dec) break;

        struct ggml_tensor* new_embed = ggml_new_tensor_2d(
            ctx_dec, GGML_TYPE_F32, embed_dim, 1);
        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, next_token,
            (float*)new_embed->data, embed_dim);

        struct ggml_tensor* logits = llm_forward(
            ctx_dec, new_embed, model_.llm, cache,
            cache.n_past(), model_.config.llm);
        run_graph(ctx_dec, logits, config.n_threads);

        cache.commit(cache.n_past(), 1);
        cache.set_n_past(cache.n_past() + 1);

        std::memcpy(logits_buf.data(), logits->data, vocab_size * sizeof(float));
        ggml_free(ctx_dec);
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_decode_end - t_decode_start).count();
    result.decode_tokens = static_cast<int>(result.token_ids.size());
    result.text = tokenizer_.decode(result.token_ids);
    return result;
}

// ============================================================
// GPU 路径
// ============================================================
InferenceResult Pipeline::run_gpu(
    const float* adaptor_data, int audio_frames,
    const InferenceConfig& config, TokenCallback callback
) {
    InferenceResult result;

#ifdef FUNASR_USE_CUDA
    const int embed_dim  = model_.config.llm.embedding_length;
    const int vocab_size = model_.config.llm.vocab_size;
    const int eos_id     = tokenizer_.eos_id();

    // Build inputs_embeds
    int total_len = prompt_builder_.total_len(audio_frames);
    result.prefill_tokens = total_len;

    std::vector<float> inputs_embeds(total_len * embed_dim);
    prompt_builder_.build_inputs_embeds(
        adaptor_data, audio_frames,
        model_.llm.embed_tokens, embed_dim,
        inputs_embeds.data()
    );

    // 清空 KV Cache
    gpu_ctx_->clear_kv_cache();

    // Prefill
    std::vector<float> logits;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!gpu_runner_->forward(inputs_embeds.data(), total_len, 0, logits)) {
        printf("[Pipeline] GPU prefill failed\n");
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    inputs_embeds.clear();
    inputs_embeds.shrink_to_fit();

    // Decode
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    std::vector<float> token_embed(embed_dim);

    for (int step = 0; step < config.max_new_tokens; step++) {
        int next_token = GPURunner::argmax(logits);
        if (next_token == eos_id) {
            if (callback) callback(next_token, "", true);
            break;
        }
        result.token_ids.push_back(next_token);
        if (callback) callback(next_token, tokenizer_.decode({next_token}), false);

        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, next_token, token_embed.data(), embed_dim);

        int n_past = gpu_ctx_->kv_cache().n_past;
        if (!gpu_runner_->forward(token_embed.data(), 1, n_past, logits)) {
            printf("[Pipeline] GPU decode step %d failed\n", step);
            break;
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_decode_end - t_decode_start).count();
    result.decode_tokens = static_cast<int>(result.token_ids.size());
    result.text = tokenizer_.decode(result.token_ids);
#else
    (void)adaptor_data; (void)audio_frames; (void)config; (void)callback;
    printf("[Pipeline] CUDA not compiled\n");
#endif

    return result;
}

// ============================================================
// 便捷接口
// ============================================================
InferenceResult Pipeline::transcribe(
    const std::string& wav_path,
    const InferenceConfig& config,
    TokenCallback callback
) {
    FbankProcessor fbank(model_.config.frontend);
    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process_wav(wav_path, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed for '%s'\n", wav_path.c_str());
        return InferenceResult{};
    }
    return run(fbank_data.data(), T, D, config, callback);
}

InferenceResult Pipeline::transcribe_audio(
    const float* audio, size_t n_samples,
    const InferenceConfig& config,
    TokenCallback callback
) {
    FbankProcessor fbank(model_.config.frontend);
    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process(audio, n_samples, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed\n");
        return InferenceResult{};
    }
    return run(fbank_data.data(), T, D, config, callback);
}

} // namespace funasr