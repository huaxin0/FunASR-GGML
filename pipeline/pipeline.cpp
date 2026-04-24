// funasr/pipeline/pipeline.cpp
// 推理管线实现 — 支持 CPU 和 GPU 两条路径
//
// CPU 路径: llm_forward + KVCache (memcpy commit)
// GPU 路径: GPURunner (ggml_cpy in graph, gallocr)
//
// GPU 路径优化:
//   Encoder/Adaptor 输出留在 GPU 上，零拷贝传给 LLM
//   只有 fbank 输入和 logits 输出会过 PCIe
//
#include "pipeline/pipeline.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
#include "compute/llm_ops.hpp"
#include "compute/graph_runner.hpp"
#include <ggml.h>
#include <cstring>
#include <cstdio>
#include <memory>

namespace funasr {

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx) ggml_free(ctx);
    }
};

using PipelineGgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

} // namespace

Pipeline::Pipeline(FunASRModel& model, Tokenizer& tokenizer)
    : model_(model)
    , tokenizer_(tokenizer)
    , prompt_builder_(tokenizer)
{}

Pipeline::~Pipeline() {
#ifdef FUNASR_USE_CUDA
    free_prefill_staging();
#endif
}

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

    gpu_ea_runner_ = std::make_unique<GPUEncoderAdaptorRunner>(gpu_ctx_->backend());
    if (!gpu_ea_runner_->init(
            model_.encoder, model_.adaptor,
            model_.config.encoder, model_.config.adaptor)) {
        printf("[Pipeline] WARNING: failed to init GPU encoder/adaptor, keeping CPU frontend\n");
        gpu_ea_runner_.reset();
    } else if (!gpu_ea_runner_->warmup(128)) {
        printf("[Pipeline] WARNING: GPU encoder/adaptor warmup failed\n");
    } else {
        printf("[Pipeline] GPU encoder/adaptor ready\n");
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
    // Phase 0: Encoder + Adaptor
    // ================================================================
    std::vector<float> adaptor_data;   // CPU 路径用
    int audio_frames = 0;
    bool ran_frontend_on_gpu = false;
    ggml_tensor* gpu_adaptor_tensor = nullptr;
#ifdef FUNASR_USE_CUDA
    if (config.use_gpu && gpu_ea_runner_ && gpu_ea_runner_->is_initialized()) {
        long frontend_ms = 0;
        gpu_adaptor_tensor = gpu_ea_runner_->forward_on_gpu(
            fbank_data, D, T, audio_frames, frontend_ms);
        if (gpu_adaptor_tensor && audio_frames > 0) {
            result.encoder_ms = static_cast<float>(frontend_ms);
            ran_frontend_on_gpu = true;
            printf("[Pipeline] Frontend ran on GPU (zero-copy): %d frames -> %d embeddings in %ld ms\n",
                   T, audio_frames, frontend_ms);
        } else {
            printf("[Pipeline] WARNING: GPU encoder/adaptor failed, falling back to CPU frontend\n");
            gpu_adaptor_tensor = nullptr;
        }
    }
#endif
    if (!ran_frontend_on_gpu) {
        struct ggml_init_params params = { config.encoder_mem, nullptr, false };
        PipelineGgmlContextPtr ctx_ea(ggml_init(params));
        if (!ctx_ea) {
            printf("[Pipeline] ERROR: failed to alloc encoder context\n");
            return result;
        }

        struct ggml_tensor* fbank_t = ggml_new_tensor_2d(ctx_ea.get(), GGML_TYPE_F32, D, T);
        std::memcpy(fbank_t->data, fbank_data, T * D * sizeof(float));

        struct ggml_tensor* enc_out = encoder_forward(
            ctx_ea.get(), fbank_t, model_.encoder, model_.config.encoder);
        struct ggml_tensor* adp_out = adaptor_forward(
            ctx_ea.get(), enc_out, model_.adaptor, model_.config.adaptor);

        auto t0 = std::chrono::high_resolution_clock::now();
        run_graph(ctx_ea.get(), adp_out, config.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        result.encoder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        audio_frames = static_cast<int>(adp_out->ne[1]);
        adaptor_data.resize(static_cast<size_t>(audio_frames) * embed_dim);
        std::memcpy(adaptor_data.data(), adp_out->data,
                     adaptor_data.size() * sizeof(float));

    }

    // ================================================================
    // Phase 1-3: LLM (CPU 或 GPU)
    // ================================================================
    InferenceResult llm_result;
    if (config.use_gpu && is_gpu_ready()) {
        llm_result = run_gpu(gpu_adaptor_tensor, audio_frames,
                             adaptor_data.empty() ? nullptr : adaptor_data.data(),
                             config, callback);
    } else {
        if (config.use_gpu && !is_gpu_ready()) {
            printf("[Pipeline] WARNING: GPU requested but not ready, falling back to CPU\n");
        }
        if (ran_frontend_on_gpu && adaptor_data.empty() && gpu_adaptor_tensor) {
            adaptor_data.resize(static_cast<size_t>(audio_frames) * embed_dim);
            ggml_backend_tensor_get(gpu_adaptor_tensor, adaptor_data.data(), 0,
                                     adaptor_data.size() * sizeof(float));
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
//
// GPU-resident path: adaptor 输出已在 GPU 上
//   prefix/suffix embedding 在 CPU 查表后传上去 (几十个 token, 很小)
//   audio 部分: GPU staging → GPU prefill staging (GPU→GPU, 无 PCIe)
//   prefill staging tensor 跨推理复用, 不每次 alloc/free
// ============================================================
InferenceResult Pipeline::run_gpu(
    ggml_tensor* gpu_adaptor_tensor, int audio_frames,
    const float* cpu_adaptor_data,
    const InferenceConfig& config, TokenCallback callback
) {
    InferenceResult result;

#ifdef FUNASR_USE_CUDA
    const int embed_dim  = model_.config.llm.embedding_length;
    const int eos_id     = tokenizer_.eos_id();

    int total_len = prompt_builder_.total_len(audio_frames);
    result.prefill_tokens = total_len;

    // 清空 KV Cache
    gpu_ctx_->clear_kv_cache();

    std::vector<float> logits;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (gpu_adaptor_tensor) {
        const auto& prefix_ids = prompt_builder_.prefix_ids();
        const auto& suffix_ids = prompt_builder_.suffix_ids();
        int prefix_len = static_cast<int>(prefix_ids.size());
        int suffix_len = static_cast<int>(suffix_ids.size());

        std::vector<float> prefix_embeds(prefix_len * embed_dim);
        for (int i = 0; i < prefix_len; i++) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, prefix_ids[i],
                prefix_embeds.data() + i * embed_dim, embed_dim);
        }
        std::vector<float> suffix_embeds(suffix_len * embed_dim);
        for (int i = 0; i < suffix_len; i++) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, suffix_ids[i],
                suffix_embeds.data() + i * embed_dim, embed_dim);
        }

        if (!ensure_prefill_staging(total_len, embed_dim)) {
            printf("[Pipeline] Failed to alloc prefill staging\n");
            return result;
        }

        ggml_backend_tensor_set(prefill_stg_tensor_, prefix_embeds.data(), 0,
                                 static_cast<size_t>(prefix_len) * embed_dim * sizeof(float));

        {
            size_t cpy_meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
            ggml_init_params cpy_params = { cpy_meta, nullptr, true };
            PipelineGgmlContextPtr ctx_cpy(ggml_init(cpy_params));
            if (!ctx_cpy) {
                printf("[Pipeline] Failed to create GPU audio copy context\n");
                return result;
            }

            ggml_tensor* src = ggml_view_2d(ctx_cpy.get(), gpu_adaptor_tensor,
                embed_dim, audio_frames,
                embed_dim * sizeof(float), 0);

            size_t audio_offset = static_cast<size_t>(prefix_len) * embed_dim * sizeof(float);
            ggml_tensor* dst = ggml_view_2d(ctx_cpy.get(), prefill_stg_tensor_,
                embed_dim, audio_frames,
                embed_dim * sizeof(float), audio_offset);

            ggml_tensor* cpy_op = ggml_cpy(ctx_cpy.get(), src, dst);

            ggml_cgraph* cpy_graph = ggml_new_graph(ctx_cpy.get());
            ggml_build_forward_expand(cpy_graph, cpy_op);

            auto buf_type = ggml_backend_get_default_buffer_type(gpu_ctx_->backend());
            ggml_gallocr_t cpy_allocr = ggml_gallocr_new(buf_type);
            if (!cpy_allocr) {
                printf("[Pipeline] Failed to create GPU audio copy allocator\n");
                return result;
            }

            if (!ggml_gallocr_alloc_graph(cpy_allocr, cpy_graph) ||
                ggml_backend_graph_compute(gpu_ctx_->backend(), cpy_graph) != GGML_STATUS_SUCCESS) {
                printf("[Pipeline] GPU audio copy graph failed, falling back to CPU round-trip\n");
                size_t audio_bytes = static_cast<size_t>(audio_frames) * embed_dim * sizeof(float);
                std::vector<float> audio_buf(audio_frames * embed_dim);
                ggml_backend_tensor_get(gpu_adaptor_tensor, audio_buf.data(), 0, audio_bytes);
                ggml_backend_tensor_set(prefill_stg_tensor_, audio_buf.data(), audio_offset, audio_bytes);
            }

            ggml_gallocr_free(cpy_allocr);
        }

        size_t suffix_offset = static_cast<size_t>(prefix_len + audio_frames)
                             * embed_dim * sizeof(float);
        ggml_backend_tensor_set(prefill_stg_tensor_, suffix_embeds.data(),
                                 suffix_offset,
                                 static_cast<size_t>(suffix_len) * embed_dim * sizeof(float));

        if (!gpu_runner_->forward_gpu_tensor(prefill_stg_tensor_, total_len, 0, logits)) {
            printf("[Pipeline] GPU prefill failed\n");
            return result;
        }

    } else if (cpu_adaptor_data) {
        std::vector<float> inputs_embeds(total_len * embed_dim);
        prompt_builder_.build_inputs_embeds(
            cpu_adaptor_data, audio_frames,
            model_.llm.embed_tokens, embed_dim,
            inputs_embeds.data()
        );

        if (!gpu_runner_->forward(inputs_embeds.data(), total_len, 0, logits)) {
            printf("[Pipeline] GPU prefill failed\n");
            return result;
        }
    } else {
        printf("[Pipeline] No adaptor data available\n");
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

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
    (void)gpu_adaptor_tensor; (void)audio_frames; (void)cpu_adaptor_data;
    (void)config; (void)callback;
    printf("[Pipeline] CUDA not compiled\n");
#endif

    return result;
}

// ============================================================
// Prefill staging 管理 (GPU-resident, 跨推理复用)
// ============================================================
#ifdef FUNASR_USE_CUDA
bool Pipeline::ensure_prefill_staging(int total_len, int embed_dim) {
    if (prefill_stg_buf_ && prefill_stg_max_len_ == total_len) {
        return true;
    }

    free_prefill_staging();

    int alloc_len = total_len;

    size_t ctx_size = ggml_tensor_overhead() * 2;
    ggml_init_params params = { ctx_size, nullptr, true };
    prefill_stg_ctx_ = ggml_init(params);
    if (!prefill_stg_ctx_) return false;

    prefill_stg_tensor_ = ggml_new_tensor_2d(
        prefill_stg_ctx_, GGML_TYPE_F32, embed_dim, alloc_len);

    prefill_stg_buf_ = ggml_backend_alloc_ctx_tensors(
        prefill_stg_ctx_, gpu_ctx_->backend());
    if (!prefill_stg_buf_) {
        ggml_free(prefill_stg_ctx_);
        prefill_stg_ctx_ = nullptr;
        prefill_stg_tensor_ = nullptr;
        return false;
    }

    prefill_stg_max_len_ = alloc_len;
    printf("[Pipeline] Prefill staging: %.2f MB (max_len=%d)\n",
           ggml_backend_buffer_get_size(prefill_stg_buf_) / 1e6, alloc_len);
    return true;
}

void Pipeline::free_prefill_staging() {
    if (prefill_stg_buf_) {
        ggml_backend_buffer_free(prefill_stg_buf_);
        prefill_stg_buf_ = nullptr;
    }
    if (prefill_stg_ctx_) {
        ggml_free(prefill_stg_ctx_);
        prefill_stg_ctx_ = nullptr;
    }
    prefill_stg_tensor_ = nullptr;
    prefill_stg_max_len_ = 0;
}
#endif

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
