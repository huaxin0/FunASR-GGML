// funasr/pipeline/pipeline.cpp
// 推理管线实现 — 支持 CPU 和 GPU 两条路径
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
#ifdef FUNASR_USE_CUDA
#include <ggml-cuda.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <future>
#include <limits>
#include <memory>
#include <utility>

namespace funasr {

namespace {

bool check_context_capacity(int total_len, const InferenceConfig& config) {
    if (total_len <= 0) {
        printf("[Pipeline] ERROR: invalid LLM input length: %d\n", total_len);
        return false;
    }
    if (total_len > config.kv_cache_size) {
        printf("[Pipeline] ERROR: LLM input too long: total_len=%d > n_ctx=%d. "
               "Use shorter VAD chunks.\n",
               total_len, config.kv_cache_size);
        return false;
    }
    if (total_len + config.max_new_tokens > config.kv_cache_size) {
        printf("[Pipeline] WARNING: total_len(%d) + max_new_tokens(%d) > n_ctx(%d); "
               "decode will stop at context limit.\n",
               total_len, config.max_new_tokens, config.kv_cache_size);
    }
    return true;
}

bool tail_repeats(const std::vector<int>& tokens, int ngram, int repeats) {
    const int need = ngram * repeats;
    if (ngram <= 0 || repeats <= 1 || static_cast<int>(tokens.size()) < need) {
        return false;
    }

    const int start = static_cast<int>(tokens.size()) - need;
    for (int r = 1; r < repeats; r++) {
        for (int i = 0; i < ngram; i++) {
            if (tokens[static_cast<size_t>(start + i)] !=
                tokens[static_cast<size_t>(start + r * ngram + i)]) {
                return false;
            }
        }
    }
    return true;
}

bool has_repetition_loop(const std::vector<int>& tokens) {
    for (int ngram = 4; ngram <= 16; ngram++) {
        if (tail_repeats(tokens, ngram, 4)) {
            return true;
        }
    }
    return false;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx) ggml_free(ctx);
    }
};

using PipelineGgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

#ifdef FUNASR_USE_CUDA
class GPUEmbeddingReleaseGuard {
public:
    GPUEmbeddingReleaseGuard(
            GPUEmbeddingPool* pool, const GPUEmbeddingHandle& handle)
        : pool_(pool), handle_(handle) {}

    ~GPUEmbeddingReleaseGuard() {
        if (active_ && pool_) {
            pool_->release(handle_);
        }
    }

    GPUEmbeddingReleaseGuard(const GPUEmbeddingReleaseGuard&) = delete;
    GPUEmbeddingReleaseGuard& operator=(
        const GPUEmbeddingReleaseGuard&) = delete;

    void dismiss() { active_ = false; }

private:
    GPUEmbeddingPool* pool_ = nullptr;
    GPUEmbeddingHandle handle_;
    bool active_ = true;
};
#endif

void add_gpu_dither(std::vector<float>& samples, uint32_t seed) {
    constexpr float amplitude = 0.0005f;
    uint32_t state = seed ? seed : 1u;
    for (float& sample : samples) {
        state = state * 1664525u + 1013904223u;
        float u = static_cast<float>((state >> 8) & 0x00FFFFFF) / 16777215.0f;
        float noise = (u * 2.0f - 1.0f) * amplitude;
        sample = std::max(-1.0f, std::min(1.0f, sample + noise));
    }
}

const float* prepare_audio_for_fbank(const float* audio, size_t n_samples,
                                     bool use_gpu_dither,
                                     std::vector<float>& scratch) {
    if (!use_gpu_dither || !audio || n_samples == 0) {
        return audio;
    }

    scratch.assign(audio, audio + n_samples);
    add_gpu_dither(scratch, 0x46554E41u);
    return scratch.data();
}

} // namespace

Pipeline::Pipeline(FunASRModel& model, Tokenizer& tokenizer)
    : model_(model)
    , tokenizer_(tokenizer)
    , prompt_builder_(tokenizer)
{}

Pipeline::~Pipeline() {
#ifdef FUNASR_USE_CUDA
    free_prefill_staging();
    gpu_mixed_runner_.reset();
    gpu_prompt_pool_.reset();
    gpu_ea_runner_.reset();
    if (gpu_frontend_backend_) {
        ggml_backend_free(gpu_frontend_backend_);
        gpu_frontend_backend_ = nullptr;
    }
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
bool Pipeline::init_gpu(
    int n_ctx,
    int gpu_id,
    int n_slots,
    int physical_kv_rows,
    int prompt_slots,
    bool enable_frontend_overlap) {
#ifdef FUNASR_USE_CUDA
    // init_gpu may be called again. Release every object backed by the old
    // backend before replacing GPUContext.
    free_prefill_staging();
    gpu_mixed_runner_.reset();
    gpu_prompt_pool_.reset();
    gpu_ea_runner_.reset();
    if (gpu_frontend_backend_) {
        ggml_backend_free(gpu_frontend_backend_);
        gpu_frontend_backend_ = nullptr;
    }
    gpu_runner_.reset();
    gpu_ctx_.reset();

    gpu_ctx_ = std::make_unique<GPUContext>();
    if (!gpu_ctx_->init(
            model_.llm, model_.config.llm, n_ctx, gpu_id, n_slots,
            physical_kv_rows)) {
        gpu_ctx_.reset();
        return false;
    }

    gpu_runner_ = std::make_unique<GPURunner>(*gpu_ctx_);
#ifdef FUNASR_USE_FLASH_ATTN
    printf("[Pipeline] GPU LLM FlashAttention enabled\n");
#else
    printf("[Pipeline] GPU LLM FlashAttention disabled\n");
#endif
    if (!gpu_runner_->warmup(500, n_ctx - 100)) {
        printf("[Pipeline] WARNING: GPU warmup failed\n");
    }

    gpu_frontend_backend_ = enable_frontend_overlap
        ? ggml_backend_cuda_init(gpu_id)
        : nullptr;
    ggml_backend_t frontend_backend = gpu_frontend_backend_
        ? gpu_frontend_backend_
        : gpu_ctx_->backend();
    if (gpu_frontend_backend_) {
        printf("[Pipeline] GPU frontend uses a dedicated CUDA stream\n");
    } else if (enable_frontend_overlap) {
        printf("[Pipeline] WARNING: dedicated GPU frontend stream unavailable\n");
    }

    gpu_ea_runner_ = std::make_unique<GPUEncoderAdaptorRunner>(frontend_backend);
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

    gpu_prompt_pool_ = std::make_unique<GPUEmbeddingPool>(
        frontend_backend,
        std::max(1, prompt_slots > 0 ? prompt_slots : n_slots));
    gpu_mixed_runner_ = std::make_unique<GPUMixedRunner>(
        gpu_ctx_->backend(), model_.config.llm.embedding_length);

    return true;
#else
    (void)n_ctx; (void)gpu_id; (void)n_slots; (void)physical_kv_rows;
    (void)prompt_slots; (void)enable_frontend_overlap;
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
    int total_len = prompt_builder_.total_len(audio_frames, config.prompt);
    result.prefill_tokens = total_len;
    if (!check_context_capacity(total_len, config)) {
        return result;
    }

    std::vector<float> inputs_embeds(total_len * embed_dim);
    prompt_builder_.build_inputs_embeds(
        adaptor_data, audio_frames,
        model_.llm.embed_tokens, embed_dim,
        inputs_embeds.data(),
        config.prompt
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
        std::vector<ggml_tensor*> kv_cpy_ops;
        struct ggml_tensor* logits = llm_forward(
            ctx_pf, embeds_t, model_.llm, cache, 0, model_.config.llm, kv_cpy_ops);
        run_graph(ctx_pf, kv_cpy_ops, logits, config.n_threads);

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
        if (cache.n_past() >= config.kv_cache_size) {
            printf("[Pipeline] WARNING: CPU decode stopped at context limit n_ctx=%d\n",
                   config.kv_cache_size);
            break;
        }

        int next_token = argmax(logits_buf.data(), vocab_size);
        if (next_token == eos_id) {
            if (callback) callback(next_token, "", true);
            break;
        }
        result.token_ids.push_back(next_token);
        if (callback) callback(next_token, tokenizer_.decode({next_token}), false);
        if (has_repetition_loop(result.token_ids)) {
            printf("[Pipeline] WARNING: CPU decode stopped after detecting repetition loop\n");
            break;
        }

        struct ggml_init_params params = { config.llm_mem, nullptr, false };
        struct ggml_context* ctx_dec = ggml_init(params);
        if (!ctx_dec) break;

        struct ggml_tensor* new_embed = ggml_new_tensor_2d(
            ctx_dec, GGML_TYPE_F32, embed_dim, 1);
        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, next_token,
            (float*)new_embed->data, embed_dim);

        std::vector<ggml_tensor*> kv_cpy_ops;
        struct ggml_tensor* logits = llm_forward(
            ctx_dec, new_embed, model_.llm, cache,
            cache.n_past(), model_.config.llm, kv_cpy_ops);
        run_graph(ctx_dec, kv_cpy_ops, logits, config.n_threads);

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

    gpu_ctx_->clear_kv_cache();

    std::vector<float> logits;
    if (!gpu_prefill_slot(gpu_adaptor_tensor, audio_frames, cpu_adaptor_data,
                          0, config, logits, result)) {
        return result;
    }

    // Decode
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    std::vector<float> token_embed(embed_dim);
    int n_past = result.prefill_tokens;

    for (int step = 0; step < config.max_new_tokens; step++) {
        if (n_past >= config.kv_cache_size) {
            printf("[Pipeline] WARNING: GPU decode stopped at context limit n_ctx=%d\n",
                   config.kv_cache_size);
            break;
        }

        int next_token = GPURunner::argmax(logits);
        if (next_token == eos_id) {
            if (callback) callback(next_token, "", true);
            break;
        }
        result.token_ids.push_back(next_token);
        if (callback) callback(next_token, tokenizer_.decode({next_token}), false);
        if (has_repetition_loop(result.token_ids)) {
            printf("[Pipeline] WARNING: GPU decode stopped after detecting repetition loop\n");
            break;
        }

        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, next_token, token_embed.data(), embed_dim);

        if (!gpu_runner_->forward_slot(token_embed.data(), 1, n_past, 0, logits)) {
            printf("[Pipeline] GPU decode step %d failed\n", step);
            break;
        }
        n_past++;
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

#ifdef FUNASR_USE_CUDA
bool Pipeline::gpu_prefill_slot(
    ggml_tensor* gpu_adaptor_tensor, int audio_frames,
    const float* cpu_adaptor_data,
    int slot_id,
    const InferenceConfig& config,
    std::vector<float>& logits,
    InferenceResult& result
) {
    const int embed_dim = model_.config.llm.embedding_length;

    int total_len = prompt_builder_.total_len(audio_frames, config.prompt);
    result.prefill_tokens = total_len;
    if (!check_context_capacity(total_len, config)) {
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (gpu_adaptor_tensor) {
        auto prefix_ids = prompt_builder_.prefix_ids(config.prompt);
        auto suffix_ids = prompt_builder_.suffix_ids(config.prompt);
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
            return false;
        }

        ggml_backend_tensor_set(prefill_stg_tensor_, prefix_embeds.data(), 0,
                                static_cast<size_t>(prefix_len) * embed_dim * sizeof(float));

        {
            size_t cpy_meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
            ggml_init_params cpy_params = { cpy_meta, nullptr, true };
            PipelineGgmlContextPtr ctx_cpy(ggml_init(cpy_params));
            if (!ctx_cpy) {
                printf("[Pipeline] Failed to create GPU audio copy context\n");
                return false;
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
                return false;
            }

            if (!ggml_gallocr_alloc_graph(cpy_allocr, cpy_graph) ||
                ggml_backend_graph_compute(gpu_ctx_->backend(), cpy_graph) != GGML_STATUS_SUCCESS) {
                printf("[Pipeline] GPU audio copy graph failed, falling back to CPU round-trip\n");
                size_t audio_bytes = static_cast<size_t>(audio_frames) * embed_dim * sizeof(float);
                std::vector<float> audio_buf(static_cast<size_t>(audio_frames) * embed_dim);
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

        if (!gpu_runner_->forward_gpu_tensor_slot(prefill_stg_tensor_, total_len, 0, slot_id, logits)) {
            printf("[Pipeline] GPU prefill failed\n");
            return false;
        }
    } else if (cpu_adaptor_data) {
        std::vector<float> inputs_embeds(static_cast<size_t>(total_len) * embed_dim);
        prompt_builder_.build_inputs_embeds(
            cpu_adaptor_data, audio_frames,
            model_.llm.embed_tokens, embed_dim,
            inputs_embeds.data(),
            config.prompt);

        if (!gpu_runner_->forward_slot(inputs_embeds.data(), total_len, 0, slot_id, logits)) {
            printf("[Pipeline] GPU prefill failed\n");
            return false;
        }
    } else {
        printf("[Pipeline] No adaptor data available\n");
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return true;
}

bool Pipeline::gpu_prefill_paged(
    ggml_tensor* gpu_adaptor_tensor, int audio_frames,
    const float* cpu_adaptor_data,
    const std::vector<int>& block_table,
    int block_size,
    const InferenceConfig& config,
    std::vector<float>& logits,
    InferenceResult& result
) {
    const int embed_dim = model_.config.llm.embedding_length;

    int total_len = prompt_builder_.total_len(audio_frames, config.prompt);
    result.prefill_tokens = total_len;
    if (!check_context_capacity(total_len, config)) {
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (gpu_adaptor_tensor) {
        auto prefix_ids = prompt_builder_.prefix_ids(config.prompt);
        auto suffix_ids = prompt_builder_.suffix_ids(config.prompt);
        int prefix_len = static_cast<int>(prefix_ids.size());
        int suffix_len = static_cast<int>(suffix_ids.size());

        std::vector<float> prefix_embeds(static_cast<size_t>(prefix_len) * embed_dim);
        for (int i = 0; i < prefix_len; i++) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, prefix_ids[i],
                prefix_embeds.data() + static_cast<size_t>(i) * embed_dim, embed_dim);
        }
        std::vector<float> suffix_embeds(static_cast<size_t>(suffix_len) * embed_dim);
        for (int i = 0; i < suffix_len; i++) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, suffix_ids[i],
                suffix_embeds.data() + static_cast<size_t>(i) * embed_dim, embed_dim);
        }

        if (!ensure_prefill_staging(total_len, embed_dim)) {
            printf("[Pipeline] Failed to alloc prefill staging\n");
            return false;
        }

        ggml_backend_tensor_set(prefill_stg_tensor_, prefix_embeds.data(), 0,
                                static_cast<size_t>(prefix_len) * embed_dim * sizeof(float));

        {
            size_t cpy_meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
            ggml_init_params cpy_params = { cpy_meta, nullptr, true };
            PipelineGgmlContextPtr ctx_cpy(ggml_init(cpy_params));
            if (!ctx_cpy) {
                return false;
            }
            ggml_tensor* src = ggml_view_2d(ctx_cpy.get(), gpu_adaptor_tensor,
                embed_dim, audio_frames, embed_dim * sizeof(float), 0);
            size_t audio_offset = static_cast<size_t>(prefix_len) * embed_dim * sizeof(float);
            ggml_tensor* dst = ggml_view_2d(ctx_cpy.get(), prefill_stg_tensor_,
                embed_dim, audio_frames, embed_dim * sizeof(float), audio_offset);
            ggml_tensor* cpy_op = ggml_cpy(ctx_cpy.get(), src, dst);
            ggml_cgraph* cpy_graph = ggml_new_graph(ctx_cpy.get());
            ggml_build_forward_expand(cpy_graph, cpy_op);
            auto buf_type = ggml_backend_get_default_buffer_type(gpu_ctx_->backend());
            ggml_gallocr_t cpy_allocr = ggml_gallocr_new(buf_type);
            if (!cpy_allocr) {
                return false;
            }
            bool copied = ggml_gallocr_alloc_graph(cpy_allocr, cpy_graph) &&
                          ggml_backend_graph_compute(gpu_ctx_->backend(), cpy_graph) == GGML_STATUS_SUCCESS;
            if (!copied) {
                size_t audio_bytes = static_cast<size_t>(audio_frames) * embed_dim * sizeof(float);
                std::vector<float> audio_buf(static_cast<size_t>(audio_frames) * embed_dim);
                ggml_backend_tensor_get(gpu_adaptor_tensor, audio_buf.data(), 0, audio_bytes);
                ggml_backend_tensor_set(prefill_stg_tensor_, audio_buf.data(), audio_offset, audio_bytes);
            }
            ggml_gallocr_free(cpy_allocr);
        }

        size_t suffix_offset = static_cast<size_t>(prefix_len + audio_frames)
                             * embed_dim * sizeof(float);
        ggml_backend_tensor_set(prefill_stg_tensor_, suffix_embeds.data(), suffix_offset,
                                static_cast<size_t>(suffix_len) * embed_dim * sizeof(float));

        if (!gpu_runner_->forward_gpu_tensor_paged(
                prefill_stg_tensor_, total_len, 0, block_table, block_size, logits)) {
            printf("[Pipeline] GPU paged prefill failed\n");
            return false;
        }
    } else if (cpu_adaptor_data) {
        std::vector<float> inputs_embeds(static_cast<size_t>(total_len) * embed_dim);
        prompt_builder_.build_inputs_embeds(
            cpu_adaptor_data, audio_frames,
            model_.llm.embed_tokens, embed_dim,
            inputs_embeds.data(),
            config.prompt);

        if (!gpu_runner_->forward_paged(
                inputs_embeds.data(), total_len, 0, block_table, block_size, logits)) {
            printf("[Pipeline] GPU paged prefill failed\n");
            return false;
        }
    } else {
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return true;
}
#endif

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
// Offline batching helpers
// ============================================================
PreparedLLMInput Pipeline::prepare_llm_input(
    const float* audio, size_t n_samples,
    const InferenceConfig& config
) {
    PreparedLLMInput prepared;
    if (!audio || n_samples == 0) {
        printf("[Pipeline] ERROR: empty audio for prepare\n");
        return prepared;
    }

    FbankProcessor fbank(model_.config.frontend);
    std::vector<float> audio_scratch;
    const float* audio_for_fbank = prepare_audio_for_fbank(
        audio, n_samples, config.use_gpu && is_gpu_ready(), audio_scratch);

    std::vector<float> fbank_data;
    int T = 0;
    int D = 0;
    if (!fbank.process(audio_for_fbank, n_samples, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed\n");
        return prepared;
    }

    const int embed_dim = model_.config.llm.embedding_length;
    std::vector<float> adaptor_data;

#ifdef FUNASR_USE_CUDA
    if (config.use_gpu && gpu_ea_runner_ && gpu_ea_runner_->is_initialized()) {
        int audio_frames = 0;
        long frontend_ms = 0;
        ggml_tensor* gpu_adaptor_tensor = gpu_ea_runner_->forward_on_gpu(
            fbank_data.data(), D, T, audio_frames, frontend_ms);
        if (gpu_adaptor_tensor && audio_frames > 0) {
            prepared.audio_frames = audio_frames;
            prepared.encoder_ms = static_cast<float>(frontend_ms);
            adaptor_data.resize(static_cast<size_t>(prepared.audio_frames) * embed_dim);
            ggml_backend_tensor_get(gpu_adaptor_tensor, adaptor_data.data(), 0,
                                    adaptor_data.size() * sizeof(float));
        } else {
            printf("[Pipeline] WARNING: GPU prepare frontend failed, falling back to CPU frontend\n");
        }
    }
#endif

    if (adaptor_data.empty()) {
        struct ggml_init_params params = { config.encoder_mem, nullptr, false };
        PipelineGgmlContextPtr ctx_ea(ggml_init(params));
        if (!ctx_ea) {
            printf("[Pipeline] ERROR: failed to alloc encoder context\n");
            return prepared;
        }

        ggml_tensor* fbank_t = ggml_new_tensor_2d(ctx_ea.get(), GGML_TYPE_F32, D, T);
        std::memcpy(fbank_t->data, fbank_data.data(), fbank_data.size() * sizeof(float));

        ggml_tensor* enc_out = encoder_forward(
            ctx_ea.get(), fbank_t, model_.encoder, model_.config.encoder);
        ggml_tensor* adp_out = adaptor_forward(
            ctx_ea.get(), enc_out, model_.adaptor, model_.config.adaptor);

        auto t0 = std::chrono::high_resolution_clock::now();
        run_graph(ctx_ea.get(), adp_out, config.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        prepared.encoder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        prepared.audio_frames = static_cast<int>(adp_out->ne[1]);
        adaptor_data.resize(static_cast<size_t>(prepared.audio_frames) * embed_dim);
        std::memcpy(adaptor_data.data(), adp_out->data,
                    adaptor_data.size() * sizeof(float));
    }

    prepared.total_len = prompt_builder_.total_len(prepared.audio_frames, config.prompt);
    prepared.inputs_embeds.resize(static_cast<size_t>(prepared.total_len) * embed_dim);
    prompt_builder_.build_inputs_embeds(
        adaptor_data.data(), prepared.audio_frames,
        model_.llm.embed_tokens, embed_dim,
        prepared.inputs_embeds.data(),
        config.prompt);

    prepared.ok = prepared.total_len > 0 && !prepared.inputs_embeds.empty();
    return prepared;
}

InferenceResult Pipeline::run_prepared(
    const PreparedLLMInput& prepared,
    const InferenceConfig& config,
    TokenCallback callback
) {
    InferenceResult result;
    auto t_total_start = std::chrono::high_resolution_clock::now();
    result.encoder_ms = prepared.encoder_ms;
    result.prefill_tokens = prepared.total_len;

    if (!prepared.ok || prepared.inputs_embeds.empty() || prepared.total_len <= 0) {
        printf("[Pipeline] ERROR: invalid prepared LLM input\n");
        return result;
    }
    if (!check_context_capacity(prepared.total_len, config)) {
        return result;
    }

    const int embed_dim = model_.config.llm.embedding_length;
    const int vocab_size = model_.config.llm.vocab_size;
    const int eos_id = tokenizer_.eos_id();

    if (config.use_gpu && is_gpu_ready()) {
#ifdef FUNASR_USE_CUDA
        gpu_ctx_->clear_kv_cache();
        std::vector<float> logits;

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!gpu_runner_->forward(
                prepared.inputs_embeds.data(), prepared.total_len, 0, logits)) {
            printf("[Pipeline] GPU prepared prefill failed\n");
            return result;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        auto t_decode_start = std::chrono::high_resolution_clock::now();
        std::vector<float> token_embed(embed_dim);
        for (int step = 0; step < config.max_new_tokens; step++) {
            if (gpu_ctx_->kv_cache().n_past >= config.kv_cache_size) {
                printf("[Pipeline] WARNING: GPU prepared decode stopped at context limit n_ctx=%d\n",
                       config.kv_cache_size);
                break;
            }

            int next_token = GPURunner::argmax(logits);
            if (next_token == eos_id) {
                if (callback) callback(next_token, "", true);
                break;
            }
            result.token_ids.push_back(next_token);
            if (callback) callback(next_token, tokenizer_.decode({next_token}), false);
            if (has_repetition_loop(result.token_ids)) {
                printf("[Pipeline] WARNING: GPU prepared decode stopped after detecting repetition loop\n");
                break;
            }

            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, next_token, token_embed.data(), embed_dim);

            int n_past = gpu_ctx_->kv_cache().n_past;
            if (!gpu_runner_->forward(token_embed.data(), 1, n_past, logits)) {
                printf("[Pipeline] GPU prepared decode step %d failed\n", step);
                break;
            }
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_decode_end - t_decode_start).count();
#else
        printf("[Pipeline] CUDA not compiled\n");
#endif
    } else {
        KVCache cache;
        if (!cache.init(model_.config.llm, config.kv_cache_size)) {
            return result;
        }

        std::vector<float> logits_buf(vocab_size);
        {
            ggml_init_params params = { config.llm_mem, nullptr, false };
            ggml_context* ctx_pf = ggml_init(params);
            if (!ctx_pf) return result;

            ggml_tensor* embeds_t = ggml_new_tensor_2d(
                ctx_pf, GGML_TYPE_F32, embed_dim, prepared.total_len);
            std::memcpy(embeds_t->data, prepared.inputs_embeds.data(),
                        prepared.inputs_embeds.size() * sizeof(float));

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<ggml_tensor*> kv_cpy_ops;
            ggml_tensor* logits = llm_forward(
                ctx_pf, embeds_t, model_.llm, cache, 0, model_.config.llm, kv_cpy_ops);
            run_graph(ctx_pf, kv_cpy_ops, logits, config.n_threads);
            cache.set_n_past(prepared.total_len);

            auto t1 = std::chrono::high_resolution_clock::now();
            result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            float* last = static_cast<float*>(logits->data) +
                          static_cast<size_t>(prepared.total_len - 1) * vocab_size;
            std::memcpy(logits_buf.data(), last, logits_buf.size() * sizeof(float));
            ggml_free(ctx_pf);
        }

        auto t_decode_start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < config.max_new_tokens; step++) {
            if (cache.n_past() >= config.kv_cache_size) {
                printf("[Pipeline] WARNING: CPU prepared decode stopped at context limit n_ctx=%d\n",
                       config.kv_cache_size);
                break;
            }

            int next_token = argmax(logits_buf.data(), vocab_size);
            if (next_token == eos_id) {
                if (callback) callback(next_token, "", true);
                break;
            }
            result.token_ids.push_back(next_token);
            if (callback) callback(next_token, tokenizer_.decode({next_token}), false);
            if (has_repetition_loop(result.token_ids)) {
                printf("[Pipeline] WARNING: CPU prepared decode stopped after detecting repetition loop\n");
                break;
            }

            ggml_init_params params = { config.llm_mem, nullptr, false };
            ggml_context* ctx_dec = ggml_init(params);
            if (!ctx_dec) break;

            ggml_tensor* new_embed = ggml_new_tensor_2d(
                ctx_dec, GGML_TYPE_F32, embed_dim, 1);
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, next_token,
                static_cast<float*>(new_embed->data), embed_dim);

            std::vector<ggml_tensor*> kv_cpy_ops;
            ggml_tensor* logits = llm_forward(
                ctx_dec, new_embed, model_.llm, cache,
                cache.n_past(), model_.config.llm, kv_cpy_ops);
            run_graph(ctx_dec, kv_cpy_ops, logits, config.n_threads);
            cache.set_n_past(cache.n_past() + 1);

            std::memcpy(logits_buf.data(), logits->data, logits_buf.size() * sizeof(float));
            ggml_free(ctx_dec);
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_decode_end - t_decode_start).count();
    }

    result.decode_tokens = static_cast<int>(result.token_ids.size());
    result.text = tokenizer_.decode(result.token_ids);

    auto t_total_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_total_end - t_total_start).count();
    return result;
}

std::vector<InferenceResult> Pipeline::run_prepared_batch(
    const std::vector<PreparedLLMInput>& prepared,
    const std::vector<int>& slot_ids,
    const InferenceConfig& config
) {
    const size_t batch = prepared.size();
    std::vector<InferenceResult> results(batch);
    if (batch == 0) {
        return results;
    }
    if (slot_ids.size() != batch) {
        printf("[Pipeline] ERROR: prepared batch slot count mismatch\n");
        return results;
    }

    if (!config.use_gpu || !is_gpu_ready() || batch == 1) {
        for (size_t i = 0; i < batch; i++) {
            results[i] = run_prepared(prepared[i], config, nullptr);
        }
        return results;
    }

#ifdef FUNASR_USE_CUDA
    auto t_total_start = std::chrono::high_resolution_clock::now();
    const int embed_dim = model_.config.llm.embedding_length;
    const int eos_id = tokenizer_.eos_id();

    struct DecodeState {
        bool done = false;
        int n_past = 0;
        std::vector<float> logits;
        std::vector<float> token_embed;
    };

    std::vector<DecodeState> states(batch);
    gpu_ctx_->clear_kv_cache();

    for (size_t i = 0; i < batch; i++) {
        results[i].encoder_ms = prepared[i].encoder_ms;
        results[i].prefill_tokens = prepared[i].total_len;
        states[i].token_embed.resize(embed_dim);

        if (!prepared[i].ok || prepared[i].inputs_embeds.empty() ||
            !check_context_capacity(prepared[i].total_len, config)) {
            states[i].done = true;
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!gpu_runner_->forward_slot(
                prepared[i].inputs_embeds.data(), prepared[i].total_len,
                0, slot_ids[i], states[i].logits)) {
            printf("[Pipeline] GPU batch prefill failed for request %zu slot=%d\n",
                   i, slot_ids[i]);
            states[i].done = true;
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        results[i].prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        states[i].n_past = prepared[i].total_len;
    }

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < config.max_new_tokens; step++) {
        std::vector<size_t> active_indices;
        std::vector<const float*> batch_embeds;
        std::vector<int> batch_pasts;
        std::vector<int> batch_slots;

        for (size_t i = 0; i < batch; i++) {
            auto& state = states[i];
            if (state.done) {
                continue;
            }
            if (state.n_past >= config.kv_cache_size) {
                printf("[Pipeline] WARNING: GPU batch request %zu stopped at context limit n_ctx=%d\n",
                       i, config.kv_cache_size);
                state.done = true;
                continue;
            }

            int next_token = GPURunner::argmax(state.logits);
            if (next_token == eos_id) {
                state.done = true;
                continue;
            }

            results[i].token_ids.push_back(next_token);
            if (has_repetition_loop(results[i].token_ids)) {
                printf("[Pipeline] WARNING: GPU batch request %zu stopped after detecting repetition loop\n",
                       i);
                state.done = true;
                continue;
            }

            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, next_token, state.token_embed.data(), embed_dim);
            active_indices.push_back(i);
            batch_embeds.push_back(state.token_embed.data());
            batch_pasts.push_back(state.n_past);
            batch_slots.push_back(slot_ids[i]);
        }

        if (active_indices.empty()) {
            break;
        }

        std::vector<std::vector<float>> next_logits;
        if (!gpu_runner_->forward_batch_decode_slots(
                batch_embeds, batch_pasts, batch_slots, next_logits)) {
            printf("[Pipeline] GPU batch decode step %d failed\n", step);
            break;
        }

        for (size_t j = 0; j < active_indices.size(); j++) {
            size_t i = active_indices[j];
            states[i].logits = std::move(next_logits[j]);
            states[i].n_past += 1;
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    float batch_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_decode_end - t_decode_start).count();
    auto t_total_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_total_end - t_total_start).count();

    for (size_t i = 0; i < batch; i++) {
        results[i].decode_ms = batch_decode_ms;
        results[i].total_ms = total_ms;
        results[i].decode_tokens = static_cast<int>(results[i].token_ids.size());
        results[i].text = tokenizer_.decode(results[i].token_ids);
    }
#else
    for (size_t i = 0; i < batch; i++) {
        results[i] = run_prepared(prepared[i], config, nullptr);
    }
#endif

    return results;
}

std::vector<InferenceResult> Pipeline::transcribe_audio_batch_gpu(
    const std::vector<AudioSpan>& audio,
    const std::vector<int>& slot_ids,
    const InferenceConfig& config
) {
    const size_t batch = audio.size();
    std::vector<InferenceResult> results(batch);
    if (batch == 0) {
        return results;
    }
    if (slot_ids.size() != batch) {
        printf("[Pipeline] ERROR: audio batch slot count mismatch\n");
        return results;
    }
    if (!config.use_gpu || !is_gpu_ready() || batch == 1) {
        for (size_t i = 0; i < batch; i++) {
            results[i] = transcribe_audio(audio[i].data, audio[i].n_samples, config, nullptr);
        }
        return results;
    }

#ifdef FUNASR_USE_CUDA
    auto t_total_start = std::chrono::high_resolution_clock::now();
    const int embed_dim = model_.config.llm.embedding_length;
    const int eos_id = tokenizer_.eos_id();

    struct DecodeState {
        bool done = false;
        int n_past = 0;
        std::vector<float> logits;
        std::vector<float> token_embed;
    };

    std::vector<DecodeState> states(batch);
    gpu_ctx_->clear_kv_cache();

    FbankProcessor fbank(model_.config.frontend);

    for (size_t i = 0; i < batch; i++) {
        states[i].token_embed.resize(embed_dim);
        if (!audio[i].data || audio[i].n_samples == 0) {
            states[i].done = true;
            continue;
        }

        std::vector<float> audio_scratch;
        const float* audio_for_fbank = prepare_audio_for_fbank(
            audio[i].data, audio[i].n_samples, true, audio_scratch);

        std::vector<float> fbank_data;
        int T = 0;
        int D = 0;
        if (!fbank.process(audio_for_fbank, audio[i].n_samples, fbank_data, T, D)) {
            printf("[Pipeline] ERROR: fbank failed for batch request %zu\n", i);
            states[i].done = true;
            continue;
        }

        int audio_frames = 0;
        long frontend_ms = 0;
        ggml_tensor* gpu_adaptor_tensor = nullptr;
        if (gpu_ea_runner_ && gpu_ea_runner_->is_initialized()) {
            gpu_adaptor_tensor = gpu_ea_runner_->forward_on_gpu(
                fbank_data.data(), D, T, audio_frames, frontend_ms);
        }
        if (!gpu_adaptor_tensor || audio_frames <= 0) {
            printf("[Pipeline] ERROR: GPU frontend failed for batch request %zu\n", i);
            states[i].done = true;
            continue;
        }

        results[i].encoder_ms = static_cast<float>(frontend_ms);
        printf("[Pipeline] Batch frontend ran on GPU (zero-copy): req=%zu slot=%d %d frames -> %d embeddings in %ld ms\n",
               i, slot_ids[i], T, audio_frames, frontend_ms);

        if (!gpu_prefill_slot(gpu_adaptor_tensor, audio_frames, nullptr,
                              slot_ids[i], config, states[i].logits, results[i])) {
            states[i].done = true;
            continue;
        }
        states[i].n_past = results[i].prefill_tokens;
    }

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < config.max_new_tokens; step++) {
        std::vector<size_t> active_indices;
        std::vector<const float*> batch_embeds;
        std::vector<int> batch_pasts;
        std::vector<int> batch_slots;

        for (size_t i = 0; i < batch; i++) {
            auto& state = states[i];
            if (state.done) {
                continue;
            }
            if (state.n_past >= config.kv_cache_size) {
                printf("[Pipeline] WARNING: GPU batch request %zu stopped at context limit n_ctx=%d\n",
                       i, config.kv_cache_size);
                state.done = true;
                continue;
            }

            int next_token = GPURunner::argmax(state.logits);
            if (next_token == eos_id) {
                state.done = true;
                continue;
            }

            results[i].token_ids.push_back(next_token);
            if (has_repetition_loop(results[i].token_ids)) {
                printf("[Pipeline] WARNING: GPU batch request %zu stopped after detecting repetition loop\n",
                       i);
                state.done = true;
                continue;
            }

            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, next_token, state.token_embed.data(), embed_dim);
            active_indices.push_back(i);
            batch_embeds.push_back(state.token_embed.data());
            batch_pasts.push_back(state.n_past);
            batch_slots.push_back(slot_ids[i]);
        }

        if (active_indices.empty()) {
            break;
        }

        std::vector<std::vector<float>> next_logits;
        if (!gpu_runner_->forward_batch_decode_slots(
                batch_embeds, batch_pasts, batch_slots, next_logits)) {
            printf("[Pipeline] GPU batch decode step %d failed\n", step);
            break;
        }

        for (size_t j = 0; j < active_indices.size(); j++) {
            size_t i = active_indices[j];
            states[i].logits = std::move(next_logits[j]);
            states[i].n_past += 1;
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    float batch_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_decode_end - t_decode_start).count();
    auto t_total_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_total_end - t_total_start).count();

    for (size_t i = 0; i < batch; i++) {
        results[i].decode_ms = batch_decode_ms;
        results[i].total_ms = total_ms;
        results[i].decode_tokens = static_cast<int>(results[i].token_ids.size());
        results[i].text = tokenizer_.decode(results[i].token_ids);
    }
#else
    for (size_t i = 0; i < batch; i++) {
        results[i] = transcribe_audio(audio[i].data, audio[i].n_samples, config, nullptr);
    }
#endif

    return results;
}

GPUPrefillState Pipeline::gpu_prefill_audio_slot(
    const AudioSpan& audio,
    int request_id,
    int slot_id,
    const InferenceConfig& config
) {
    GPUPrefillState state;
    state.request_id = request_id;
    state.slot_id = slot_id;

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready()) {
        printf("[Pipeline] ERROR: gpu_prefill_audio_slot requires initialized GPU\n");
        return state;
    }
    if (!audio.data || audio.n_samples == 0) {
        return state;
    }

    FbankProcessor fbank(model_.config.frontend);
    std::vector<float> audio_scratch;
    const float* audio_for_fbank = prepare_audio_for_fbank(
        audio.data, audio.n_samples, true, audio_scratch);

    std::vector<float> fbank_data;
    int T = 0;
    int D = 0;
    if (!fbank.process(audio_for_fbank, audio.n_samples, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed for request %d\n", request_id);
        return state;
    }

    int audio_frames = 0;
    long frontend_ms = 0;
    ggml_tensor* gpu_adaptor_tensor = nullptr;
    if (gpu_ea_runner_ && gpu_ea_runner_->is_initialized()) {
        gpu_adaptor_tensor = gpu_ea_runner_->forward_on_gpu(
            fbank_data.data(), D, T, audio_frames, frontend_ms);
    }
    if (!gpu_adaptor_tensor || audio_frames <= 0) {
        printf("[Pipeline] ERROR: GPU frontend failed for request %d\n", request_id);
        return state;
    }

    state.stats.encoder_ms = static_cast<float>(frontend_ms);
    printf("[Pipeline] Scheduler frontend ran on GPU (zero-copy): req=%d slot=%d %d frames -> %d embeddings in %ld ms\n",
           request_id, slot_id, T, audio_frames, frontend_ms);

    if (!gpu_prefill_slot(gpu_adaptor_tensor, audio_frames, nullptr,
                          slot_id, config, state.logits, state.stats)) {
        return state;
    }

    state.n_past = state.stats.prefill_tokens;
    state.ok = true;
#else
    (void)audio; (void)request_id; (void)slot_id; (void)config;
    printf("[Pipeline] CUDA not compiled\n");
#endif

    return state;
}

PreparedGPUAudio Pipeline::prepare_audio_gpu(
    const AudioSpan& audio,
    int request_id
) {
    PreparedGPUAudio prepared;
    prepared.request_id = request_id;

#ifdef FUNASR_USE_CUDA
    if (!is_gpu_ready()) {
        printf("[Pipeline] ERROR: prepare_audio_gpu requires initialized GPU\n");
        return prepared;
    }
    if (!audio.data || audio.n_samples == 0) {
        return prepared;
    }

    FbankProcessor fbank(model_.config.frontend);
    std::vector<float> audio_scratch;
    const float* audio_for_fbank = prepare_audio_for_fbank(
        audio.data, audio.n_samples, true, audio_scratch);

    std::vector<float> fbank_data;
    int T = 0;
    int D = 0;
    if (!fbank.process(audio_for_fbank, audio.n_samples, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed for paged request %d\n", request_id);
        return prepared;
    }

    int audio_frames = 0;
    long frontend_ms = 0;
    ggml_tensor* gpu_adaptor_tensor = nullptr;
    if (gpu_ea_runner_ && gpu_ea_runner_->is_initialized()) {
        gpu_adaptor_tensor = gpu_ea_runner_->forward_on_gpu(
            fbank_data.data(), D, T, audio_frames, frontend_ms);
    }
    if (!gpu_adaptor_tensor || audio_frames <= 0) {
        printf("[Pipeline] ERROR: GPU frontend failed for paged request %d\n", request_id);
        return prepared;
    }

    prepared.adaptor_tensor = gpu_adaptor_tensor;
    prepared.audio_frames = audio_frames;
    prepared.encoder_ms = static_cast<float>(frontend_ms);
    prepared.ok = true;
    printf("[Pipeline] Scheduler frontend prepared on GPU: req=%d %d frames -> %d embeddings in %ld ms\n",
           request_id, T, audio_frames, frontend_ms);
#else
    (void)audio; (void)request_id;
    printf("[Pipeline] CUDA not compiled\n");
#endif
    return prepared;
}

PreparedGPUPrompt Pipeline::prepare_prompt_gpu(
    AudioSpan audio,
    int request_id,
    int cached_prefix_tokens,
    const InferenceConfig& config
) {
    PreparedGPUPrompt result;
    result.request_id = request_id;

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready() || !gpu_prompt_pool_ ||
        !audio.data || audio.n_samples == 0) {
        return result;
    }

    const std::vector<int> prefix_ids =
        prompt_builder_.prefix_ids(config.prompt);
    const std::vector<int> suffix_ids =
        prompt_builder_.suffix_ids(config.prompt);
    if (prefix_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        suffix_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return result;
    }
    const int prefix_tokens = static_cast<int>(prefix_ids.size());
    const int suffix_tokens = static_cast<int>(suffix_ids.size());
    if (cached_prefix_tokens < 0 ||
        (cached_prefix_tokens != 0 &&
         cached_prefix_tokens != prefix_tokens)) {
        return result;
    }

    // The adaptor tensor is non-owning staging memory. No frontend invocation
    // may occur before it is copied into the owning prompt pool below.
    const PreparedGPUAudio prepared = prepare_audio_gpu(audio, request_id);
    if (!prepared.ok || !prepared.adaptor_tensor) {
        return result;
    }

    const GPUPromptLayout layout = make_gpu_prompt_layout(
        prefix_tokens, cached_prefix_tokens,
        prepared.audio_frames, suffix_tokens);
    const int embed_dim = model_.config.llm.embedding_length;
    if (!layout.valid || embed_dim <= 0) {
        return result;
    }

    const GPUEmbeddingHandle handle =
        gpu_prompt_pool_->acquire(layout.stored_tokens, embed_dim);
    if (!handle.valid()) {
        return result;
    }
    GPUEmbeddingReleaseGuard release_guard(gpu_prompt_pool_.get(), handle);

    if (!gpu_prompt_pool_->copy_tensor(
            handle, layout.stored_prefix_tokens,
            prepared.adaptor_tensor, 0, prepared.audio_frames)) {
        return result;
    }

    if (layout.stored_prefix_tokens > 0) {
        std::vector<float> prefix_embeds(
            static_cast<size_t>(layout.stored_prefix_tokens) * embed_dim);
        for (int i = 0; i < layout.stored_prefix_tokens; ++i) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, prefix_ids[static_cast<size_t>(i)],
                prefix_embeds.data() + static_cast<size_t>(i) * embed_dim,
                embed_dim);
        }
        if (!gpu_prompt_pool_->set_host(
                handle, 0, prefix_embeds.data(),
                layout.stored_prefix_tokens)) {
            return result;
        }
    }

    if (layout.suffix_tokens > 0) {
        std::vector<float> suffix_embeds(
            static_cast<size_t>(layout.suffix_tokens) * embed_dim);
        for (int i = 0; i < layout.suffix_tokens; ++i) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, suffix_ids[static_cast<size_t>(i)],
                suffix_embeds.data() + static_cast<size_t>(i) * embed_dim,
                embed_dim);
        }
        const int suffix_offset =
            layout.stored_prefix_tokens + layout.audio_tokens;
        if (!gpu_prompt_pool_->set_host(
                handle, suffix_offset, suffix_embeds.data(),
                layout.suffix_tokens)) {
            return result;
        }
    }

    result.embeddings = handle;
    result.prompt_tokens = layout.prompt_tokens;
    result.cached_prefix_tokens = layout.cached_prefix_tokens;
    result.audio_frames = prepared.audio_frames;
    result.encoder_ms = prepared.encoder_ms;
    result.ok = true;
    release_guard.dismiss();
#else
    (void)audio;
    (void)cached_prefix_tokens;
    (void)config;
#endif
    return result;
}

PreparedFbankBatch Pipeline::prepare_fbank_batch(
    const std::vector<AudioSpan>& audio,
    const std::vector<int>& request_ids,
    int n_threads
) {
    PreparedFbankBatch batch;
    batch.audio = audio;
    batch.request_ids = request_ids;
    batch.items.resize(audio.size());
    if (audio.empty() || audio.size() != request_ids.size()) {
        return batch;
    }

    const auto fbank_begin = std::chrono::high_resolution_clock::now();
    const size_t worker_limit = std::min(
        audio.size(), static_cast<size_t>(std::max(1, n_threads)));
    for (size_t base = 0; base < audio.size(); base += worker_limit) {
        const size_t wave_size = std::min(worker_limit, audio.size() - base);
        std::vector<std::future<FbankBatchItem>> fbank_tasks;
        fbank_tasks.reserve(wave_size);
        for (size_t offset = 0; offset < wave_size; ++offset) {
            const AudioSpan span = audio[base + offset];
            fbank_tasks.push_back(std::async(
                std::launch::async, [this, span]() {
                FbankBatchItem item;
                if (!span.data || span.n_samples == 0) {
                    return item;
                }
                FbankProcessor fbank(model_.config.frontend);
                std::vector<float> scratch;
                const float* source = prepare_audio_for_fbank(
                    span.data, span.n_samples, true, scratch);
                item.ok = fbank.process(
                    source, span.n_samples,
                    item.data, item.frames, item.dim) &&
                    item.frames > 0 && item.dim > 0;
                return item;
            }));
        }
        for (size_t offset = 0; offset < fbank_tasks.size(); ++offset) {
            const size_t index = base + offset;
            try {
                batch.items[index] = fbank_tasks[offset].get();
            } catch (const std::exception& error) {
                printf("[Pipeline] WARNING: batched Fbank request %zu failed: %s\n",
                       index, error.what());
            } catch (...) {
                printf("[Pipeline] WARNING: batched Fbank request %zu failed\n",
                       index);
            }
        }
    }
    const auto fbank_end = std::chrono::high_resolution_clock::now();
    batch.fbank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        fbank_end - fbank_begin).count();

    int max_frames = 0;
    int valid_items = 0;
    for (const FbankBatchItem& item : batch.items) {
        if (!item.ok) {
            continue;
        }
        ++valid_items;
        max_frames = std::max(max_frames, item.frames);
        batch.input_frames += item.frames;
    }
    batch.padded_frames = static_cast<long long>(max_frames) * valid_items -
                          batch.input_frames;
    batch.ok = valid_items > 0;
    return batch;
}

std::vector<PreparedGPUPrompt> Pipeline::prepare_prompts_gpu_batch(
    const std::vector<AudioSpan>& audio,
    const std::vector<int>& request_ids,
    int cached_prefix_tokens,
    const InferenceConfig& config
) {
    if (audio.size() == 1 && request_ids.size() == 1) {
        return {prepare_prompt_gpu(
            audio[0], request_ids[0], cached_prefix_tokens, config)};
    }
    const PreparedFbankBatch fbank_batch = prepare_fbank_batch(
        audio, request_ids, config.n_threads);
    return prepare_prompts_gpu_batch_from_fbank(
        fbank_batch, cached_prefix_tokens, config);
}

std::vector<PreparedGPUPrompt> Pipeline::prepare_prompts_gpu_batch_from_fbank(
    const PreparedFbankBatch& fbank_batch,
    int cached_prefix_tokens,
    const InferenceConfig& config
) {
    const std::vector<AudioSpan>& audio = fbank_batch.audio;
    const std::vector<int>& request_ids = fbank_batch.request_ids;
    std::vector<PreparedGPUPrompt> results(audio.size());
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].request_id = i < request_ids.size()
            ? request_ids[i]
            : -1;
    }

#ifdef FUNASR_USE_CUDA
    if (audio.empty() || audio.size() != request_ids.size() ||
        !config.use_gpu || !is_gpu_ready() || !gpu_prompt_pool_ ||
        !gpu_ea_runner_ || !gpu_ea_runner_->is_initialized()) {
        return results;
    }
    const std::vector<int> prefix_ids =
        prompt_builder_.prefix_ids(config.prompt);
    const std::vector<int> suffix_ids =
        prompt_builder_.suffix_ids(config.prompt);
    if (prefix_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        suffix_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return results;
    }
    const int prefix_tokens = static_cast<int>(prefix_ids.size());
    const int suffix_tokens = static_cast<int>(suffix_ids.size());
    if (cached_prefix_tokens < 0 ||
        (cached_prefix_tokens != 0 &&
         cached_prefix_tokens != prefix_tokens)) {
        return results;
    }

    const std::vector<FbankBatchItem>& fbank_storage = fbank_batch.items;
    const long fbank_ms = fbank_batch.fbank_ms;

    std::vector<const float*> fbank_pointers;
    std::vector<int> fbank_frames;
    std::vector<size_t> batch_to_result;
    fbank_pointers.reserve(audio.size());
    fbank_frames.reserve(audio.size());
    batch_to_result.reserve(audio.size());
    int fbank_dim = 0;

    for (size_t i = 0; i < audio.size(); ++i) {
        const FbankBatchItem& item = fbank_storage[i];
        if (!item.ok || (fbank_dim != 0 && item.dim != fbank_dim)) {
            continue;
        }
        fbank_dim = item.dim;
        fbank_pointers.push_back(item.data.data());
        fbank_frames.push_back(item.frames);
        batch_to_result.push_back(i);
    }

    if (batch_to_result.size() < 2) {
        for (size_t index : batch_to_result) {
            results[index] = prepare_prompt_gpu(
                audio[index], request_ids[index],
                cached_prefix_tokens, config);
        }
        return results;
    }

    std::vector<int> output_frames;
    int output_stride_frames = 0;
    long frontend_ms = 0;
    ggml_tensor* batched_adaptor = gpu_ea_runner_->forward_batch_on_gpu(
        fbank_pointers, fbank_dim, fbank_frames,
        output_frames, output_stride_frames, frontend_ms);
    if (!batched_adaptor ||
        output_frames.size() != batch_to_result.size() ||
        output_stride_frames <= 0) {
        printf("[Pipeline] WARNING: batched GPU frontend failed, "
               "falling back to single requests\n");
        for (size_t index : batch_to_result) {
            results[index] = prepare_prompt_gpu(
                audio[index], request_ids[index],
                cached_prefix_tokens, config);
        }
        return results;
    }

    const int embed_dim = model_.config.llm.embedding_length;
    std::vector<float> stored_prefix_embeds;
    if (cached_prefix_tokens == 0 && prefix_tokens > 0) {
        stored_prefix_embeds.resize(
            static_cast<size_t>(prefix_tokens) * embed_dim);
        for (int i = 0; i < prefix_tokens; ++i) {
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, prefix_ids[static_cast<size_t>(i)],
                stored_prefix_embeds.data() + static_cast<size_t>(i) * embed_dim,
                embed_dim);
        }
    }
    std::vector<float> suffix_embeds(
        static_cast<size_t>(suffix_tokens) * embed_dim);
    for (int i = 0; i < suffix_tokens; ++i) {
        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, suffix_ids[static_cast<size_t>(i)],
            suffix_embeds.data() + static_cast<size_t>(i) * embed_dim,
            embed_dim);
    }

    const ggml_init_params view_params = {
        ggml_tensor_overhead() * (batch_to_result.size() + 2),
        nullptr,
        true,
    };
    PipelineGgmlContextPtr view_ctx(ggml_init(view_params));
    if (!view_ctx) {
        for (size_t index : batch_to_result) {
            results[index] = prepare_prompt_gpu(
                audio[index], request_ids[index],
                cached_prefix_tokens, config);
        }
        return results;
    }

    std::vector<GPUEmbeddingHandle> acquired;
    acquired.reserve(batch_to_result.size());
    bool batch_copy_failed = false;
    for (size_t b = 0; b < batch_to_result.size(); ++b) {
        const size_t result_index = batch_to_result[b];
        const int audio_frames = output_frames[b];
        const GPUPromptLayout layout = make_gpu_prompt_layout(
            prefix_tokens, cached_prefix_tokens,
            audio_frames, suffix_tokens);
        if (!layout.valid || audio_frames <= 0 ||
            audio_frames > output_stride_frames) {
            batch_copy_failed = true;
            break;
        }

        const GPUEmbeddingHandle handle =
            gpu_prompt_pool_->acquire(layout.stored_tokens, embed_dim);
        if (!handle.valid()) {
            batch_copy_failed = true;
            break;
        }
        acquired.push_back(handle);

        ggml_tensor* sequence_view = ggml_view_2d(
            view_ctx.get(), batched_adaptor,
            embed_dim, audio_frames, batched_adaptor->nb[1],
            b * batched_adaptor->nb[2]);
        if (!sequence_view ||
            ggml_backend_view_init(sequence_view) != GGML_STATUS_SUCCESS ||
            !gpu_prompt_pool_->copy_tensor(
                handle, layout.stored_prefix_tokens,
                sequence_view, 0, audio_frames) ||
            (layout.stored_prefix_tokens > 0 &&
             !gpu_prompt_pool_->set_host(
                 handle, 0, stored_prefix_embeds.data(),
                 layout.stored_prefix_tokens)) ||
            (layout.suffix_tokens > 0 &&
             !gpu_prompt_pool_->set_host(
                 handle,
                 layout.stored_prefix_tokens + layout.audio_tokens,
                 suffix_embeds.data(), layout.suffix_tokens))) {
            batch_copy_failed = true;
            break;
        }

        PreparedGPUPrompt prepared;
        prepared.embeddings = handle;
        prepared.request_id = request_ids[result_index];
        prepared.prompt_tokens = layout.prompt_tokens;
        prepared.cached_prefix_tokens = layout.cached_prefix_tokens;
        prepared.audio_frames = audio_frames;
        prepared.encoder_ms = static_cast<float>(frontend_ms);
        prepared.frontend_batch_size = static_cast<int>(
            batch_to_result.size());
        prepared.ok = true;
        results[result_index] = prepared;
    }

    if (batch_copy_failed) {
        for (const GPUEmbeddingHandle& handle : acquired) {
            gpu_prompt_pool_->release(handle);
        }
        for (size_t index : batch_to_result) {
            results[index] = prepare_prompt_gpu(
                audio[index], request_ids[index],
                cached_prefix_tokens, config);
        }
        return results;
    }

    printf("[Pipeline] Batched frontend prepared: batch=%zu max_frames=%d "
           "fbank=%ldms gpu=%ldms total=%ldms\n",
           batch_to_result.size(), output_stride_frames,
           fbank_ms, frontend_ms, fbank_ms + frontend_ms);
#else
    (void)cached_prefix_tokens;
    (void)config;
#endif
    return results;
}

bool Pipeline::release_prompt_gpu(const GPUEmbeddingHandle& handle) {
#ifdef FUNASR_USE_CUDA
    return gpu_prompt_pool_ && gpu_prompt_pool_->release(handle);
#else
    (void)handle;
    return false;
#endif
}

bool Pipeline::reserve_prompt_embedding_slots(int min_capacity) {
#ifdef FUNASR_USE_CUDA
    return gpu_prompt_pool_ &&
           gpu_prompt_pool_->reserve_slots(min_capacity);
#else
    (void)min_capacity;
    return false;
#endif
}

int Pipeline::free_prompt_embedding_slots() const {
#ifdef FUNASR_USE_CUDA
    return gpu_prompt_pool_ ? gpu_prompt_pool_->free_count() : 0;
#else
    return 0;
#endif
}

int Pipeline::prompt_embedding_capacity() const {
#ifdef FUNASR_USE_CUDA
    return gpu_prompt_pool_ ? gpu_prompt_pool_->capacity() : 0;
#else
    return 0;
#endif
}

bool Pipeline::supports_gpu_frontend_overlap() const {
#ifdef FUNASR_USE_CUDA
    return gpu_frontend_backend_ && gpu_ea_runner_ && gpu_prompt_pool_;
#else
    return false;
#endif
}

void Pipeline::synchronize_gpu_frontend() {
#ifdef FUNASR_USE_CUDA
    if (gpu_frontend_backend_) {
        ggml_backend_synchronize(gpu_frontend_backend_);
    }
#endif
}

bool Pipeline::configure_mixed_graph_cache(int capacity) {
#ifdef FUNASR_USE_CUDA
    if (!gpu_mixed_runner_ || capacity <= 0) {
        return false;
    }
    gpu_mixed_runner_->set_graph_cache_capacity(capacity);
    return true;
#else
    (void)capacity;
    return false;
#endif
}

GPUMixedStepResult Pipeline::gpu_mixed_step_paged(
    const std::vector<GPUMixedStepInput>& inputs,
    int block_size,
    const InferenceConfig& config
) {
    GPUMixedStepResult result;
#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready() || !gpu_prompt_pool_ ||
        !gpu_mixed_runner_ || inputs.empty() || block_size <= 0) {
        return result;
    }

    std::vector<PackedSequenceSource> sources;
    sources.reserve(inputs.size());
    for (const auto& input : inputs) {
        sources.push_back(input.source);
    }
    PackedMixedMetadata metadata =
        build_packed_mixed_metadata(sources, block_size);
    if (!metadata.valid) {
        return result;
    }

    std::vector<PackedPromptSlice> prompt_slices;
    std::vector<int> decode_token_ids;
    std::vector<int> output_request_ids;
    prompt_slices.reserve(inputs.size());
    decode_token_ids.reserve(static_cast<size_t>(metadata.decode_tokens));
    output_request_ids.reserve(metadata.selected_rows.size());

    int packed_row = 0;
    for (const auto& packed_source : metadata.sequences) {
        const auto input_it = std::find_if(
            inputs.begin(), inputs.end(), [&](const GPUMixedStepInput& input) {
                return input.source.scheduled.request_id ==
                       packed_source.scheduled.request_id;
            });
        if (input_it == inputs.end()) {
            return result;
        }

        const ScheduledSequence& scheduled = packed_source.scheduled;
        if (scheduled.input_kind == UnifiedInputKind::Prompt) {
            const int64_t local_offset =
                static_cast<int64_t>(scheduled.token_offset) -
                packed_source.cached_prefix_tokens;
            if (local_offset < 0 || local_offset >
                    std::numeric_limits<int>::max() ||
                !gpu_prompt_pool_->owns(input_it->prompt_embeddings)) {
                return result;
            }
            prompt_slices.push_back(PackedPromptSlice{
                input_it->prompt_embeddings,
                static_cast<int>(local_offset),
                scheduled.num_tokens,
                packed_row,
            });
        } else {
            decode_token_ids.push_back(packed_source.decode_token_id);
        }
        if (scheduled.produces_logits) {
            output_request_ids.push_back(scheduled.request_id);
        }
        packed_row += scheduled.num_tokens;
    }
    if (packed_row != metadata.total_tokens ||
        output_request_ids.size() != metadata.selected_rows.size()) {
        return result;
    }

    const auto staging_begin = std::chrono::high_resolution_clock::now();
    if (!gpu_mixed_runner_->stage_inputs(
            *gpu_prompt_pool_, prompt_slices, metadata.prefill_tokens,
            decode_token_ids)) {
        return result;
    }
    const auto staging_end = std::chrono::high_resolution_clock::now();

    std::vector<int> next_tokens;
    MixedGraphExecutionInfo graph_info;
    if (!gpu_mixed_runner_->execute_paged(
            *gpu_ctx_, metadata, block_size, next_tokens,
            &graph_info) ||
        next_tokens.size() != output_request_ids.size()) {
        return result;
    }
    const auto compute_end = std::chrono::high_resolution_clock::now();

    result.outputs.reserve(next_tokens.size());
    for (size_t i = 0; i < next_tokens.size(); ++i) {
        result.outputs.push_back(
            GPUMixedStepOutput{output_request_ids[i], next_tokens[i]});
    }
    result.total_tokens = metadata.total_tokens;
    result.prefill_tokens = metadata.prefill_tokens;
    result.decode_tokens = metadata.decode_tokens;
    result.staging_ms = std::chrono::duration<float, std::milli>(
        staging_end - staging_begin).count();
    result.compute_ms = std::chrono::duration<float, std::milli>(
        compute_end - staging_end).count();
    result.graph_cache_hit = graph_info.cache_hit;
    result.graph_shape_hash = graph_info.shape_hash;
    result.graph_max_blocks = graph_info.max_blocks;
    result.graph_max_n_kv = graph_info.max_n_kv;
    result.graph_prefill_sequences = graph_info.prefill_sequences;
    result.graph_cache_entries = graph_info.cache_entries;
    result.graph_cache_capacity = graph_info.cache_capacity;
    result.graph_cache_evictions = graph_info.cache_evictions;
    result.ok = true;
#else
    (void)inputs;
    (void)block_size;
    (void)config;
#endif
    return result;
}

GPUPrefillState Pipeline::gpu_prefill_prompt_chunk_paged(
    const PreparedGPUPrompt& prepared,
    int absolute_token_offset,
    int requested_tokens,
    const std::vector<int>& block_table,
    int block_size,
    const InferenceConfig& config
) {
    GPUPrefillState state;
    state.request_id = prepared.request_id;
    state.block_table = block_table;
    state.block_size = block_size;
    state.stats.encoder_ms = prepared.encoder_ms;

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready() || !gpu_prompt_pool_ ||
        !gpu_prompt_pool_->owns(prepared.embeddings) ||
        block_table.empty() || block_size <= 0) {
        return state;
    }

    const GPUPromptChunkRange range = make_gpu_prompt_chunk_range(
        prepared, absolute_token_offset, requested_tokens);
    if (!range.valid) {
        return state;
    }

    const int64_t absolute_end =
        static_cast<int64_t>(range.absolute_offset) + range.token_count;
    if (absolute_end > std::numeric_limits<int>::max() ||
        !check_context_capacity(static_cast<int>(absolute_end), config)) {
        return state;
    }

    const size_t metadata_size = ggml_tensor_overhead() * 2;
    const ggml_init_params params = {metadata_size, nullptr, true};
    PipelineGgmlContextPtr view_ctx(ggml_init(params));
    if (!view_ctx) {
        return state;
    }
    ggml_tensor* chunk = gpu_prompt_pool_->view(
        view_ctx.get(), prepared.embeddings,
        range.local_offset, range.token_count);
    if (!chunk) {
        return state;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!gpu_runner_->forward_gpu_tensor_paged(
            chunk, range.token_count, range.absolute_offset,
            block_table, block_size, state.logits)) {
        return state;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    state.n_past = static_cast<int>(absolute_end);
    state.stats.prefill_tokens = range.token_count;
    state.stats.prefill_ms = std::chrono::duration<float, std::milli>(
        t1 - t0).count();
    if (!range.produces_logits) {
        state.logits.clear();
    }
    state.ok = true;
#else
    (void)absolute_token_offset;
    (void)requested_tokens;
    (void)config;
#endif
    return state;
}

GPUPrefillState Pipeline::gpu_prefill_prefix_paged(
    const std::vector<int>& prefix_ids,
    int request_id,
    const std::vector<int>& block_table,
    int block_size,
    const InferenceConfig& config
) {
    GPUPrefillState state;
    state.request_id = request_id;
    state.block_table = block_table;
    state.block_size = block_size;

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready() || prefix_ids.empty() ||
        block_table.empty() || block_size <= 0) {
        return state;
    }

    const int prefix_len = static_cast<int>(prefix_ids.size());
    if (!check_context_capacity(prefix_len, config)) {
        return state;
    }

    const int embed_dim = model_.config.llm.embedding_length;
    std::vector<float> prefix_embeds(static_cast<size_t>(prefix_len) * embed_dim);
    for (int i = 0; i < prefix_len; i++) {
        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, prefix_ids[static_cast<size_t>(i)],
            prefix_embeds.data() + static_cast<size_t>(i) * embed_dim, embed_dim);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!gpu_runner_->forward_paged(
            prefix_embeds.data(), prefix_len, 0,
            block_table, block_size, state.logits)) {
        return state;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    state.stats.prefill_tokens = prefix_len;
    state.stats.prefill_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    state.n_past = prefix_len;
    state.logits.clear();
    state.ok = true;
#else
    (void)prefix_ids; (void)request_id; (void)block_table;
    (void)block_size; (void)config;
#endif
    return state;
}

GPUPrefillState Pipeline::gpu_prefill_prepared_audio_paged(
    const PreparedGPUAudio& prepared,
    int request_id,
    const std::vector<int>& block_table,
    int block_size,
    int cached_prefix_tokens,
    const InferenceConfig& config
) {
    GPUPrefillState state;
    state.request_id = request_id;
    state.block_table = block_table;
    state.block_size = block_size;
    state.stats.encoder_ms = prepared.encoder_ms;

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready() || !prepared.ok ||
        !prepared.adaptor_tensor || prepared.audio_frames <= 0) {
        return state;
    }

    if (cached_prefix_tokens <= 0) {
        if (!gpu_prefill_paged(
                prepared.adaptor_tensor, prepared.audio_frames, nullptr,
                block_table, block_size, config, state.logits, state.stats)) {
            return state;
        }
        state.n_past = state.stats.prefill_tokens;
        state.ok = true;
        return state;
    }

    const int embed_dim = model_.config.llm.embedding_length;
    const std::vector<int> suffix_ids = prompt_builder_.suffix_ids(config.prompt);
    const int suffix_len = static_cast<int>(suffix_ids.size());
    const int tail_len = prepared.audio_frames + suffix_len;
    const int total_len = cached_prefix_tokens + tail_len;
    state.stats.prefill_tokens = total_len;
    if (!check_context_capacity(total_len, config) ||
        !ensure_prefill_staging(tail_len, embed_dim)) {
        return state;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t metadata_size = ggml_tensor_overhead() * 4;
    ggml_init_params copy_params = {metadata_size, nullptr, true};
    PipelineGgmlContextPtr copy_ctx(ggml_init(copy_params));
    if (!copy_ctx) {
        return state;
    }
    ggml_tensor* source = ggml_view_2d(
        copy_ctx.get(), prepared.adaptor_tensor,
        embed_dim, prepared.audio_frames, embed_dim * sizeof(float), 0);
    ggml_tensor* destination = ggml_view_2d(
        copy_ctx.get(), prefill_stg_tensor_,
        embed_dim, prepared.audio_frames, embed_dim * sizeof(float), 0);
    if (ggml_backend_view_init(source) != GGML_STATUS_SUCCESS ||
        ggml_backend_view_init(destination) != GGML_STATUS_SUCCESS) {
        return state;
    }
    ggml_backend_tensor_copy(source, destination);

    std::vector<float> suffix_embeds(static_cast<size_t>(suffix_len) * embed_dim);
    for (int i = 0; i < suffix_len; i++) {
        PromptBuilder::get_token_embedding(
            model_.llm.embed_tokens, suffix_ids[static_cast<size_t>(i)],
            suffix_embeds.data() + static_cast<size_t>(i) * embed_dim, embed_dim);
    }
    const size_t suffix_offset = static_cast<size_t>(prepared.audio_frames) *
                                 embed_dim * sizeof(float);
    ggml_backend_tensor_set(
        prefill_stg_tensor_, suffix_embeds.data(), suffix_offset,
        static_cast<size_t>(suffix_len) * embed_dim * sizeof(float));

    if (!gpu_runner_->forward_gpu_tensor_paged(
            prefill_stg_tensor_, tail_len, cached_prefix_tokens,
            block_table, block_size, state.logits)) {
        return state;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    state.stats.prefill_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    state.n_past = total_len;
    state.ok = true;
#else
    (void)prepared; (void)request_id; (void)block_table;
    (void)block_size; (void)cached_prefix_tokens; (void)config;
#endif
    return state;
}

bool Pipeline::copy_paged_kv_block_rows(
    int source_block,
    int destination_block,
    int valid_rows,
    int block_size
) {
#ifdef FUNASR_USE_CUDA
    return is_gpu_ready() && gpu_runner_->copy_paged_kv_block_rows(
        source_block, destination_block, valid_rows, block_size);
#else
    (void)source_block; (void)destination_block;
    (void)valid_rows; (void)block_size;
    return false;
#endif
}

GPUPrefillState Pipeline::gpu_prefill_audio_paged(
    const AudioSpan& audio,
    int request_id,
    const std::vector<int>& block_table,
    int block_size,
    const InferenceConfig& config
) {
    PreparedGPUAudio prepared = prepare_audio_gpu(audio, request_id);
    GPUPrefillState state = gpu_prefill_prepared_audio_paged(
        prepared, request_id, block_table, block_size, 0, config);

    return state;
}

std::vector<GPUDecodeStepOutput> Pipeline::gpu_decode_step_slots(
    const std::vector<GPUDecodeStepInput>& inputs,
    const InferenceConfig& config,
    GPUDecodeDispatchStats* dispatch_stats
) {
    std::vector<GPUDecodeStepOutput> outputs;
    outputs.reserve(inputs.size());
    if (inputs.empty()) {
        return outputs;
    }

#ifdef FUNASR_USE_CUDA
    if (!config.use_gpu || !is_gpu_ready()) {
        printf("[Pipeline] ERROR: gpu_decode_step_slots requires initialized GPU\n");
        return outputs;
    }

    bool has_paged = false;
    for (const auto& input : inputs) {
        if (!input.block_table.empty()) {
            has_paged = true;
            break;
        }
    }

    if (has_paged) {
        const int embed_dim = model_.config.llm.embedding_length;
        std::vector<const float*> batch_embeds;
        std::vector<std::vector<float>> owned_embeds;
        std::vector<int> batch_pasts;
        std::vector<std::vector<int>> batch_tables;
        std::vector<int> request_ids;
        std::vector<int> batch_token_ids;
        int block_size = 0;

        batch_embeds.reserve(inputs.size());
        owned_embeds.reserve(inputs.size());
        batch_pasts.reserve(inputs.size());
        batch_tables.reserve(inputs.size());
        request_ids.reserve(inputs.size());
        batch_token_ids.reserve(inputs.size());

        const bool serial_forced = env_flag_enabled("FUNASR_PAGED_DECODE_SERIAL");
        if (serial_forced && dispatch_stats) {
            dispatch_stats->serial_env_forced++;
        }

        for (const auto& input : inputs) {
            if (input.block_table.empty() || input.block_size <= 0 ||
                input.n_past >= config.kv_cache_size) {
                if (dispatch_stats) {
                    dispatch_stats->invalid_paged_input++;
                }
                continue;
            }
            if (input.token_id >= 0 &&
                !env_flag_enabled("FUNASR_PAGED_DECODE_CPU_EMBED") &&
                !serial_forced) {
                batch_token_ids.push_back(input.token_id);
            } else if (input.token_embed) {
                batch_embeds.push_back(input.token_embed);
            } else if (input.token_id >= 0) {
                owned_embeds.emplace_back(static_cast<size_t>(embed_dim));
                PromptBuilder::get_token_embedding(
                    model_.llm.embed_tokens, input.token_id,
                    owned_embeds.back().data(), embed_dim);
                batch_embeds.push_back(owned_embeds.back().data());
            } else {
                continue;
            }
            block_size = input.block_size;
            batch_pasts.push_back(input.n_past);
            batch_tables.push_back(input.block_table);
            request_ids.push_back(input.request_id);
        }

        std::vector<std::vector<float>> logits;
        std::vector<int> next_tokens;
        if (batch_token_ids.size() == batch_pasts.size() &&
            !serial_forced) {
            if (!gpu_runner_->forward_batch_decode_paged_token_ids(
                    batch_token_ids, batch_pasts, batch_tables, block_size, next_tokens)) {
                if (dispatch_stats) {
                    dispatch_stats->token_id_fast_path_unavailable++;
                }
                printf("[Pipeline] GPU scheduler paged token-id decode unavailable; falling back\n");
                for (int token_id : batch_token_ids) {
                    owned_embeds.emplace_back(static_cast<size_t>(embed_dim));
                    PromptBuilder::get_token_embedding(
                        model_.llm.embed_tokens, token_id,
                        owned_embeds.back().data(), embed_dim);
                    batch_embeds.push_back(owned_embeds.back().data());
                }
                batch_token_ids.clear();
            }
        }

        if (next_tokens.empty() && !batch_embeds.empty()) {
            if (serial_forced) {
                logits.resize(batch_embeds.size());
                for (size_t i = 0; i < batch_embeds.size(); i++) {
                    if (!gpu_runner_->forward_paged(
                            batch_embeds[i], 1, batch_pasts[i],
                            batch_tables[i], block_size, logits[i])) {
                        printf("[Pipeline] GPU scheduler paged serial decode failed\n");
                        return outputs;
                    }
                }
            } else if (!gpu_runner_->forward_batch_decode_paged_tokens(
                           batch_embeds, batch_pasts, batch_tables, block_size, next_tokens)) {
                if (dispatch_stats) {
                    dispatch_stats->host_embedding_batch_unavailable++;
                }
                printf("[Pipeline] GPU scheduler paged batch token decode failed\n");
                return outputs;
            }
        }

        const size_t n_outputs = serial_forced
            ? logits.size()
            : next_tokens.size();
        for (size_t i = 0; i < n_outputs; i++) {
            GPUDecodeStepOutput out;
            out.request_id = request_ids[i];
            out.n_past = batch_pasts[i] + 1;
            if (serial_forced) {
                out.next_token = GPURunner::argmax(logits[i]);
                out.logits = std::move(logits[i]);
            } else {
                out.next_token = next_tokens[i];
            }
            out.ok = true;
            outputs.push_back(std::move(out));
        }
        return outputs;
    }

    std::vector<const float*> batch_embeds;
    std::vector<std::vector<float>> owned_embeds;
    std::vector<int> batch_pasts;
    std::vector<int> batch_slots;
    std::vector<int> request_ids;
    batch_embeds.reserve(inputs.size());
    owned_embeds.reserve(inputs.size());
    batch_pasts.reserve(inputs.size());
    batch_slots.reserve(inputs.size());
    request_ids.reserve(inputs.size());

    const int embed_dim = model_.config.llm.embedding_length;
    for (const auto& input : inputs) {
        if (input.n_past >= config.kv_cache_size) {
            continue;
        }
        if (input.token_embed) {
            batch_embeds.push_back(input.token_embed);
        } else if (input.token_id >= 0) {
            owned_embeds.emplace_back(static_cast<size_t>(embed_dim));
            PromptBuilder::get_token_embedding(
                model_.llm.embed_tokens, input.token_id,
                owned_embeds.back().data(), embed_dim);
            batch_embeds.push_back(owned_embeds.back().data());
        } else {
            continue;
        }
        batch_pasts.push_back(input.n_past);
        batch_slots.push_back(input.slot_id);
        request_ids.push_back(input.request_id);
    }

    std::vector<std::vector<float>> logits;
    if (!batch_embeds.empty() &&
        !gpu_runner_->forward_batch_decode_slots(
            batch_embeds, batch_pasts, batch_slots, logits)) {
        printf("[Pipeline] GPU scheduler batch decode failed\n");
        return outputs;
    }

    for (size_t i = 0; i < logits.size(); i++) {
        GPUDecodeStepOutput out;
        out.request_id = request_ids[i];
        out.n_past = batch_pasts[i] + 1;
        out.logits = std::move(logits[i]);
        out.ok = true;
        outputs.push_back(std::move(out));
    }
#else
    (void)inputs; (void)config;
    printf("[Pipeline] CUDA not compiled\n");
#endif

    return outputs;
}

PagedDecodeProfile Pipeline::gpu_paged_decode_profile() const {
#ifdef FUNASR_USE_CUDA
    if (gpu_runner_) {
        return gpu_runner_->paged_decode_profile();
    }
#endif
    return PagedDecodeProfile{};
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
    std::vector<float> audio;
    int wav_sr = 0;
    if (!fbank.read_wav(wav_path, audio, wav_sr)) {
        printf("[Pipeline] ERROR: fbank failed for '%s'\n", wav_path.c_str());
        return InferenceResult{};
    }

    if (wav_sr != model_.config.frontend.sample_rate) {
        printf("[Fbank] WARNING: WAV sample rate %d != expected %d, no resampling\n",
               wav_sr, model_.config.frontend.sample_rate);
    }

    std::vector<float> audio_scratch;
    const float* audio_for_fbank = prepare_audio_for_fbank(
        audio.data(), audio.size(), config.use_gpu && is_gpu_ready(), audio_scratch);

    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process(audio_for_fbank, audio.size(), fbank_data, T, D)) {
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
    std::vector<float> audio_scratch;
    const float* audio_for_fbank = prepare_audio_for_fbank(
        audio, n_samples, config.use_gpu && is_gpu_ready(), audio_scratch);

    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process(audio_for_fbank, n_samples, fbank_data, T, D)) {
        printf("[Pipeline] ERROR: fbank failed\n");
        return InferenceResult{};
    }
    return run(fbank_data.data(), T, D, config, callback);
}

} // namespace funasr
