#ifndef FUNASR_COMPUTE_ENCODER_ADAPTOR_GPU_HPP
#define FUNASR_COMPUTE_ENCODER_ADAPTOR_GPU_HPP

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include "core/config.hpp"
#include "model/weights.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
#include <vector>
#include <memory>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace funasr {

struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx) ggml_free(ctx);
    }
};

using GgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

class GPUEncoderAdaptorRunner {
public:
    explicit GPUEncoderAdaptorRunner(ggml_backend_t backend)
        : backend_(backend) {
        auto buf_type = ggml_backend_get_default_buffer_type(backend_);
        allocr_encoder_ = ggml_gallocr_new(buf_type);
        allocr_adaptor_ = ggml_gallocr_new(buf_type);
    }

    ~GPUEncoderAdaptorRunner() {
        if (allocr_encoder_) ggml_gallocr_free(allocr_encoder_);
        if (allocr_adaptor_) ggml_gallocr_free(allocr_adaptor_);
        cleanup();
    }

    GPUEncoderAdaptorRunner(const GPUEncoderAdaptorRunner&) = delete;
    GPUEncoderAdaptorRunner& operator=(const GPUEncoderAdaptorRunner&) = delete;

    bool init(const EncoderWeights& cpu_encoder,
              const AdaptorWeights& cpu_adaptor,
              const EncoderConfig& encoder_cfg,
              const AdaptorConfig& adaptor_cfg) {
        cleanup();

        encoder_cfg_ = encoder_cfg;
        adaptor_cfg_ = adaptor_cfg;
        input_dim_ = cpu_encoder.encoder0.norm1_w
            ? static_cast<int>(cpu_encoder.encoder0.norm1_w->ne[0])
            : 0;

        if (!init_encoder_weights(cpu_encoder)) return false;
        if (!init_adaptor_weights(cpu_adaptor)) return false;

        if (input_dim_ <= 0) {
            printf("[GPU-EA] Failed to infer encoder input dim\n");
            cleanup();
            return false;
        }

        initialized_ = true;
        printf("[GPU-EA] Initialized, input_dim=%d, encoder_dim=%d, adaptor_dim=%d\n",
               input_dim_, encoder_cfg_.output_size, adaptor_cfg_.output_dim);
        return true;
    }

    bool warmup(int max_frames = 500) {
        if (!initialized_) return false;
        std::vector<float> dummy(static_cast<size_t>(input_dim_) * max_frames, 0.0f);
        std::vector<float> out;
        return forward(dummy.data(), input_dim_, max_frames, out) > 0;
    }

    // ============================================================
    // GPU-resident forward: 数据全程留在 GPU 上
    //
    // fbank_data: CPU 上的 fbank 输入 (只有这一次 CPU→GPU)
    // 返回: adaptor 输出 tensor (在 GPU staging buffer 上)
    //       调用者可直接传给 GPURunner，无需 GPU→CPU→GPU
    //       返回 nullptr 表示失败
    // out_frames: 输出帧数
    // ============================================================
    ggml_tensor* forward_on_gpu(const float* fbank_data, int fbank_dim, int frames,
                                int& out_frames, long& total_ms) {
        out_frames = 0;
        total_ms = 0;

        if (!initialized_ || !fbank_data || fbank_dim <= 0 || frames <= 0) {
            return nullptr;
        }
        if (fbank_dim != input_dim_) {
            printf("[GPU-EA] Invalid fbank dim: got %d, expected %d\n",
                   fbank_dim, input_dim_);
            return nullptr;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // ===== Phase 1: Encoder =====
        size_t enc_ctx_size = 256ULL * 1024 * 1024;
        ggml_init_params enc_params = { enc_ctx_size, nullptr, true };
        GgmlContextPtr ctx_enc(ggml_init(enc_params));
        if (!ctx_enc) return nullptr;

        ggml_tensor* fbank = ggml_new_tensor_2d(ctx_enc.get(), GGML_TYPE_F32, fbank_dim, frames);
        ggml_set_input(fbank);
        ggml_tensor* enc_out = encoder_forward(ctx_enc.get(), fbank, encoder_weights_, encoder_cfg_);
        ggml_set_output(enc_out);

        ggml_cgraph* graph_enc = ggml_new_graph_custom(ctx_enc.get(), 131072, false);
        ggml_build_forward_expand(graph_enc, enc_out);

        if (!ggml_gallocr_alloc_graph(allocr_encoder_, graph_enc)) {
            printf("[GPU-EA] Failed to alloc encoder graph\n");
            return nullptr;
        }

        // 唯一的 CPU→GPU: fbank 输入
        ggml_backend_tensor_set(fbank, fbank_data, 0,
                                static_cast<size_t>(fbank_dim) * frames * sizeof(float));

        if (ggml_backend_graph_compute(backend_, graph_enc) != GGML_STATUS_SUCCESS) {
            printf("[GPU-EA] Encoder graph compute failed\n");
            return nullptr;
        }

        const int enc_out_frames = static_cast<int>(enc_out->ne[1]);
        const int enc_out_dim = static_cast<int>(enc_out->ne[0]);

        // ===== Phase 1.5: Encoder 输出 → staging buffer (GPU→GPU) =====
        // enc_out 在 allocr_encoder_ 的临时 buffer 里，下次 alloc_graph 会覆盖
        // 所以我们需要把它拷到一个独立的 staging tensor
        if (!ensure_staging_buffer(enc_out_dim, enc_out_frames,
                                   adaptor_cfg_.output_dim, enc_out_frames)) {
            printf("[GPU-EA] Staging buffer alloc failed\n");
            return nullptr;
        }

        ggml_backend_tensor_copy(enc_out, staging_enc_);

        // ===== Phase 2: Adaptor =====
        ggml_init_params adp_params = { 128ULL * 1024 * 1024, nullptr, true };
        GgmlContextPtr ctx_adp(ggml_init(adp_params));
        if (!ctx_adp) return nullptr;

        ggml_tensor* adaptor_input = ggml_new_tensor_2d(
            ctx_adp.get(), GGML_TYPE_F32, enc_out_dim, enc_out_frames);
        ggml_set_input(adaptor_input);
        ggml_tensor* adp_out = adaptor_forward(
            ctx_adp.get(), adaptor_input, adaptor_weights_, adaptor_cfg_);
        ggml_set_output(adp_out);

        ggml_cgraph* graph_adp = ggml_new_graph_custom(ctx_adp.get(), 16384, false);
        ggml_build_forward_expand(graph_adp, adp_out);

        if (!ggml_gallocr_alloc_graph(allocr_adaptor_, graph_adp)) {
            printf("[GPU-EA] Failed to alloc adaptor graph\n");
            return nullptr;
        }

        // GPU→GPU: staging_enc_ → adaptor 输入 (无 PCIe)
        ggml_backend_tensor_copy(staging_enc_, adaptor_input);

        if (ggml_backend_graph_compute(backend_, graph_adp) != GGML_STATUS_SUCCESS) {
            printf("[GPU-EA] Adaptor graph compute failed\n");
            return nullptr;
        }

        // GPU→GPU: adaptor 输出 → staging_adp_
        const int adp_out_frames = static_cast<int>(adp_out->ne[1]);
        const int adp_out_dim = static_cast<int>(adp_out->ne[0]);

        if (!ensure_staging_buffer(enc_out_dim, enc_out_frames,
                                   adp_out_dim, adp_out_frames)) {
            printf("[GPU-EA] Staging buffer realloc failed\n");
            return nullptr;
        }
        ggml_backend_tensor_copy(adp_out, staging_adp_);

        out_frames = adp_out_frames;
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // 返回 GPU 上的 staging tensor，调用者可直接使用
        return staging_adp_;
    }

    int forward(const float* fbank_data, int fbank_dim, int frames,
                std::vector<float>& adaptor_out) {
        long ignored = 0;
        return forward_timed(fbank_data, fbank_dim, frames, adaptor_out, ignored);
    }

    int forward_encoder(const float* fbank_data, int fbank_dim, int frames,
                        std::vector<float>& encoder_out) {
        long ignored = 0;
        return forward_encoder_timed(fbank_data, fbank_dim, frames, encoder_out, ignored);
    }

    int forward_adaptor(const float* encoder_data, int encoder_dim, int frames,
                        std::vector<float>& adaptor_out) {
        long ignored = 0;
        return forward_adaptor_timed(encoder_data, encoder_dim, frames, adaptor_out, ignored);
    }

    int forward_timed(const float* fbank_data, int fbank_dim, int frames,
                      std::vector<float>& adaptor_out, long& total_ms) {
        std::vector<float> encoder_out;
        long encoder_ms = 0;
        const int encoder_frames = forward_encoder_timed(
            fbank_data, fbank_dim, frames, encoder_out, encoder_ms);
        if (encoder_frames <= 0) {
            adaptor_out.clear();
            total_ms = 0;
            return 0;
        }

        long adaptor_ms = 0;
        const int adaptor_frames = forward_adaptor_timed(
            encoder_out.data(), encoder_cfg_.output_size, encoder_frames, adaptor_out, adaptor_ms);
        if (adaptor_frames <= 0) {
            adaptor_out.clear();
            total_ms = 0;
            return 0;
        }

        total_ms = encoder_ms + adaptor_ms;
        return adaptor_frames;
    }

    int forward_encoder_timed(const float* fbank_data, int fbank_dim, int frames,
                              std::vector<float>& encoder_out, long& total_ms) {
        encoder_out.clear();
        total_ms = 0;

        if (!initialized_ || !fbank_data || fbank_dim <= 0 || frames <= 0) {
            return 0;
        }
        if (fbank_dim != input_dim_) {
            printf("[GPU-EA] Invalid fbank dim: got %d, expected %d\n",
                   fbank_dim, input_dim_);
            return 0;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        size_t ctx_size = 256ULL * 1024 * 1024;
        ggml_init_params enc_params = { ctx_size, nullptr, true };
        ggml_context* ctx_enc = ggml_init(enc_params);
        if (!ctx_enc) {
            printf("[GPU-EA] Failed to create encoder graph context\n");
            return 0;
        }

        ggml_tensor* fbank = ggml_new_tensor_2d(ctx_enc, GGML_TYPE_F32, fbank_dim, frames);
        ggml_set_input(fbank);
        ggml_tensor* enc_out = encoder_forward(ctx_enc, fbank, encoder_weights_, encoder_cfg_);
        ggml_set_output(enc_out);

        ggml_cgraph* graph_enc = ggml_new_graph_custom(ctx_enc, 131072, false);
        ggml_build_forward_expand(graph_enc, enc_out);

        if (!ggml_gallocr_alloc_graph(allocr_encoder_, graph_enc)) {
            printf("[GPU-EA] Failed to alloc encoder graph\n");
            ggml_free(ctx_enc);
            return 0;
        }

        ggml_backend_tensor_set(fbank, fbank_data, 0,
                                static_cast<size_t>(fbank_dim) * frames * sizeof(float));

        if (ggml_backend_graph_compute(backend_, graph_enc) != GGML_STATUS_SUCCESS) {
            printf("[GPU-EA] Encoder graph compute failed\n");
            ggml_free(ctx_enc);
            return 0;
        }

        const int out_frames = static_cast<int>(enc_out->ne[1]);
        const int out_dim = static_cast<int>(enc_out->ne[0]);
        encoder_out.resize(static_cast<size_t>(out_dim) * out_frames);
        ggml_backend_tensor_get(enc_out, encoder_out.data(), 0,
                                encoder_out.size() * sizeof(float));

        ggml_free(ctx_enc);

        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return out_frames;
    }

    int forward_adaptor_timed(const float* encoder_data, int encoder_dim, int frames,
                              std::vector<float>& adaptor_out, long& total_ms) {
        adaptor_out.clear();
        total_ms = 0;

        if (!initialized_ || !encoder_data || encoder_dim <= 0 || frames <= 0) {
            return 0;
        }
        if (encoder_dim != encoder_cfg_.output_size) {
            printf("[GPU-EA] Invalid encoder dim: got %d, expected %d\n",
                   encoder_dim, encoder_cfg_.output_size);
            return 0;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        ggml_init_params adp_params = { 128ULL * 1024 * 1024, nullptr, true };
        ggml_context* ctx_adp = ggml_init(adp_params);
        if (!ctx_adp) {
            printf("[GPU-EA] Failed to create adaptor graph context\n");
            return 0;
        }

        ggml_tensor* adaptor_input = ggml_new_tensor_2d(
            ctx_adp, GGML_TYPE_F32, encoder_dim, frames);
        ggml_set_input(adaptor_input);
        ggml_tensor* adp_out = adaptor_forward(
            ctx_adp, adaptor_input, adaptor_weights_, adaptor_cfg_);
        ggml_set_output(adp_out);

        ggml_cgraph* graph_adp = ggml_new_graph_custom(ctx_adp, 16384, false);
        ggml_build_forward_expand(graph_adp, adp_out);

        if (!ggml_gallocr_alloc_graph(allocr_adaptor_, graph_adp)) {
            printf("[GPU-EA] Failed to alloc adaptor graph\n");
            ggml_free(ctx_adp);
            return 0;
        }

        ggml_backend_tensor_set(adaptor_input, encoder_data, 0,
                                static_cast<size_t>(encoder_dim) * frames * sizeof(float));

        if (ggml_backend_graph_compute(backend_, graph_adp) != GGML_STATUS_SUCCESS) {
            printf("[GPU-EA] Adaptor graph compute failed\n");
            ggml_free(ctx_adp);
            return 0;
        }

        const int out_frames = static_cast<int>(adp_out->ne[1]);
        const int out_dim = static_cast<int>(adp_out->ne[0]);
        adaptor_out.resize(static_cast<size_t>(out_dim) * out_frames);
        ggml_backend_tensor_get(adp_out, adaptor_out.data(), 0,
                                adaptor_out.size() * sizeof(float));

        ggml_free(ctx_adp);

        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return out_frames;
    }

    bool is_initialized() const { return initialized_; }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t allocr_encoder_ = nullptr;
    ggml_gallocr_t allocr_adaptor_ = nullptr;

    EncoderConfig encoder_cfg_{};
    AdaptorConfig adaptor_cfg_{};
    EncoderWeights encoder_weights_{};
    AdaptorWeights adaptor_weights_{};
    int input_dim_ = 0;

    ggml_context* encoder_ctx_ = nullptr;
    ggml_context* adaptor_ctx_ = nullptr;
    ggml_backend_buffer_t encoder_buffer_ = nullptr;
    ggml_backend_buffer_t adaptor_buffer_ = nullptr;

    // GPU staging buffers: 用于在 encoder→adaptor→LLM 之间
    // 保持数据在 GPU 上，避免 GPU→CPU→GPU round-trip
    ggml_context* staging_ctx_ = nullptr;
    ggml_backend_buffer_t staging_buffer_ = nullptr;
    ggml_tensor* staging_enc_ = nullptr;
    ggml_tensor* staging_adp_ = nullptr;
    int staging_enc_dim_ = 0;
    int staging_enc_frames_ = 0;
    int staging_adp_dim_ = 0;
    int staging_adp_frames_ = 0;

    bool initialized_ = false;

    // 确保 staging buffer 足够大，返回 false 表示分配失败
    bool ensure_staging_buffer(int enc_dim, int enc_frames,
                               int adp_dim, int adp_frames) {
        if (staging_buffer_
            && staging_enc_dim_ == enc_dim && staging_enc_frames_ == enc_frames
            && staging_adp_dim_ == adp_dim && staging_adp_frames_ == adp_frames) {
            return true;
        }

        if (staging_buffer_) {
            ggml_backend_buffer_free(staging_buffer_);
            staging_buffer_ = nullptr;
        }
        if (staging_ctx_) {
            ggml_free(staging_ctx_);
            staging_ctx_ = nullptr;
        }
        staging_enc_ = nullptr;
        staging_adp_ = nullptr;

        staging_enc_dim_ = enc_dim;
        staging_enc_frames_ = enc_frames;
        staging_adp_dim_ = adp_dim;
        staging_adp_frames_ = adp_frames;

        size_t ctx_size = ggml_tensor_overhead() * 4;
        ggml_init_params params = { ctx_size, nullptr, true };
        staging_ctx_ = ggml_init(params);
        if (!staging_ctx_) {
            printf("[GPU-EA] Failed to create staging context\n");
            staging_enc_dim_ = 0; staging_enc_frames_ = 0;
            staging_adp_dim_ = 0; staging_adp_frames_ = 0;
            return false;
        }

        staging_enc_ = ggml_new_tensor_2d(staging_ctx_, GGML_TYPE_F32,
                                          enc_dim, enc_frames);
        staging_adp_ = ggml_new_tensor_2d(staging_ctx_, GGML_TYPE_F32,
                                          adp_dim, adp_frames);

        staging_buffer_ = ggml_backend_alloc_ctx_tensors(staging_ctx_, backend_);
        if (!staging_buffer_) {
            printf("[GPU-EA] Failed to alloc staging buffer\n");
            ggml_free(staging_ctx_);
            staging_ctx_ = nullptr;
            staging_enc_ = nullptr;
            staging_adp_ = nullptr;
            staging_enc_dim_ = 0; staging_enc_frames_ = 0;
            staging_adp_dim_ = 0; staging_adp_frames_ = 0;
            return false;
        }

        printf("[GPU-EA] Staging buffer: %.2f MB (enc=[%d,%d] adp=[%d,%d])\n",
               ggml_backend_buffer_get_size(staging_buffer_) / 1e6,
               enc_dim, enc_frames, adp_dim, adp_frames);
        return true;
    }

    static ggml_tensor* mirror_tensor(ggml_context* ctx, const ggml_tensor* src) {
        if (!src) return nullptr;
        const int nd = ggml_n_dims(src);
        if (nd == 1) return ggml_new_tensor_1d(ctx, src->type, src->ne[0]);
        if (nd == 2) return ggml_new_tensor_2d(ctx, src->type, src->ne[0], src->ne[1]);
        if (nd == 3) return ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], src->ne[2]);
        if (nd == 4) return ggml_new_tensor_4d(ctx, src->type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        return nullptr;
    }

    static void copy_tensor(ggml_tensor* dst, const ggml_tensor* src) {
        if (!dst || !src) return;
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(dst));
    }

    static void mirror_encoder_layer(ggml_context* ctx,
                                     EncoderLayerWeights& dst,
                                     const EncoderLayerWeights& src) {
        dst.attn_q_w   = mirror_tensor(ctx, src.attn_q_w);
        dst.attn_q_b   = mirror_tensor(ctx, src.attn_q_b);
        dst.attn_k_w   = mirror_tensor(ctx, src.attn_k_w);
        dst.attn_k_b   = mirror_tensor(ctx, src.attn_k_b);
        dst.attn_v_w   = mirror_tensor(ctx, src.attn_v_w);
        dst.attn_v_b   = mirror_tensor(ctx, src.attn_v_b);
        dst.attn_out_w = mirror_tensor(ctx, src.attn_out_w);
        dst.attn_out_b = mirror_tensor(ctx, src.attn_out_b);
        dst.fsmn_w     = mirror_tensor(ctx, src.fsmn_w);
        dst.ffn_w1     = mirror_tensor(ctx, src.ffn_w1);
        dst.ffn_b1     = mirror_tensor(ctx, src.ffn_b1);
        dst.ffn_w2     = mirror_tensor(ctx, src.ffn_w2);
        dst.ffn_b2     = mirror_tensor(ctx, src.ffn_b2);
        dst.norm1_w    = mirror_tensor(ctx, src.norm1_w);
        dst.norm1_b    = mirror_tensor(ctx, src.norm1_b);
        dst.norm2_w    = mirror_tensor(ctx, src.norm2_w);
        dst.norm2_b    = mirror_tensor(ctx, src.norm2_b);
    }

    static void copy_encoder_layer(EncoderLayerWeights& dst,
                                   const EncoderLayerWeights& src) {
        copy_tensor(dst.attn_q_w,   src.attn_q_w);
        copy_tensor(dst.attn_q_b,   src.attn_q_b);
        copy_tensor(dst.attn_k_w,   src.attn_k_w);
        copy_tensor(dst.attn_k_b,   src.attn_k_b);
        copy_tensor(dst.attn_v_w,   src.attn_v_w);
        copy_tensor(dst.attn_v_b,   src.attn_v_b);
        copy_tensor(dst.attn_out_w, src.attn_out_w);
        copy_tensor(dst.attn_out_b, src.attn_out_b);
        copy_tensor(dst.fsmn_w,     src.fsmn_w);
        copy_tensor(dst.ffn_w1,     src.ffn_w1);
        copy_tensor(dst.ffn_b1,     src.ffn_b1);
        copy_tensor(dst.ffn_w2,     src.ffn_w2);
        copy_tensor(dst.ffn_b2,     src.ffn_b2);
        copy_tensor(dst.norm1_w,    src.norm1_w);
        copy_tensor(dst.norm1_b,    src.norm1_b);
        copy_tensor(dst.norm2_w,    src.norm2_w);
        copy_tensor(dst.norm2_b,    src.norm2_b);
    }

    static void mirror_adaptor_block(ggml_context* ctx,
                                     AdaptorBlockWeights& dst,
                                     const AdaptorBlockWeights& src) {
        dst.attn_q_w   = mirror_tensor(ctx, src.attn_q_w);
        dst.attn_q_b   = mirror_tensor(ctx, src.attn_q_b);
        dst.attn_k_w   = mirror_tensor(ctx, src.attn_k_w);
        dst.attn_k_b   = mirror_tensor(ctx, src.attn_k_b);
        dst.attn_v_w   = mirror_tensor(ctx, src.attn_v_w);
        dst.attn_v_b   = mirror_tensor(ctx, src.attn_v_b);
        dst.attn_out_w = mirror_tensor(ctx, src.attn_out_w);
        dst.attn_out_b = mirror_tensor(ctx, src.attn_out_b);
        dst.ffn_w1     = mirror_tensor(ctx, src.ffn_w1);
        dst.ffn_b1     = mirror_tensor(ctx, src.ffn_b1);
        dst.ffn_w2     = mirror_tensor(ctx, src.ffn_w2);
        dst.ffn_b2     = mirror_tensor(ctx, src.ffn_b2);
        dst.norm1_w    = mirror_tensor(ctx, src.norm1_w);
        dst.norm1_b    = mirror_tensor(ctx, src.norm1_b);
        dst.norm2_w    = mirror_tensor(ctx, src.norm2_w);
        dst.norm2_b    = mirror_tensor(ctx, src.norm2_b);
    }

    static void copy_adaptor_block(AdaptorBlockWeights& dst,
                                   const AdaptorBlockWeights& src) {
        copy_tensor(dst.attn_q_w,   src.attn_q_w);
        copy_tensor(dst.attn_q_b,   src.attn_q_b);
        copy_tensor(dst.attn_k_w,   src.attn_k_w);
        copy_tensor(dst.attn_k_b,   src.attn_k_b);
        copy_tensor(dst.attn_v_w,   src.attn_v_w);
        copy_tensor(dst.attn_v_b,   src.attn_v_b);
        copy_tensor(dst.attn_out_w, src.attn_out_w);
        copy_tensor(dst.attn_out_b, src.attn_out_b);
        copy_tensor(dst.ffn_w1,     src.ffn_w1);
        copy_tensor(dst.ffn_b1,     src.ffn_b1);
        copy_tensor(dst.ffn_w2,     src.ffn_w2);
        copy_tensor(dst.ffn_b2,     src.ffn_b2);
        copy_tensor(dst.norm1_w,    src.norm1_w);
        copy_tensor(dst.norm1_b,    src.norm1_b);
        copy_tensor(dst.norm2_w,    src.norm2_w);
        copy_tensor(dst.norm2_b,    src.norm2_b);
    }

    bool init_encoder_weights(const EncoderWeights& cpu) {
        const int n_tensors = cpu.tensor_count() + 16;
        ggml_init_params params = {
            ggml_tensor_overhead() * static_cast<size_t>(n_tensors),
            nullptr,
            true,
        };
        encoder_ctx_ = ggml_init(params);
        if (!encoder_ctx_) {
            printf("[GPU-EA] Failed to create encoder weights context\n");
            return false;
        }

        mirror_encoder_layer(encoder_ctx_, encoder_weights_.encoder0, cpu.encoder0);
        encoder_weights_.main_layers.resize(cpu.main_layers.size());
        for (size_t i = 0; i < cpu.main_layers.size(); ++i) {
            mirror_encoder_layer(encoder_ctx_, encoder_weights_.main_layers[i], cpu.main_layers[i]);
        }
        encoder_weights_.tp_layers.resize(cpu.tp_layers.size());
        for (size_t i = 0; i < cpu.tp_layers.size(); ++i) {
            mirror_encoder_layer(encoder_ctx_, encoder_weights_.tp_layers[i], cpu.tp_layers[i]);
        }
        encoder_weights_.after_norm_w = mirror_tensor(encoder_ctx_, cpu.after_norm_w);
        encoder_weights_.after_norm_b = mirror_tensor(encoder_ctx_, cpu.after_norm_b);
        encoder_weights_.tp_norm_w    = mirror_tensor(encoder_ctx_, cpu.tp_norm_w);
        encoder_weights_.tp_norm_b    = mirror_tensor(encoder_ctx_, cpu.tp_norm_b);

        encoder_buffer_ = ggml_backend_alloc_ctx_tensors(encoder_ctx_, backend_);
        if (!encoder_buffer_) {
            printf("[GPU-EA] Failed to alloc encoder weights buffer\n");
            return false;
        }

        copy_encoder_layer(encoder_weights_.encoder0, cpu.encoder0);
        for (size_t i = 0; i < cpu.main_layers.size(); ++i) {
            copy_encoder_layer(encoder_weights_.main_layers[i], cpu.main_layers[i]);
        }
        for (size_t i = 0; i < cpu.tp_layers.size(); ++i) {
            copy_encoder_layer(encoder_weights_.tp_layers[i], cpu.tp_layers[i]);
        }
        copy_tensor(encoder_weights_.after_norm_w, cpu.after_norm_w);
        copy_tensor(encoder_weights_.after_norm_b, cpu.after_norm_b);
        copy_tensor(encoder_weights_.tp_norm_w,    cpu.tp_norm_w);
        copy_tensor(encoder_weights_.tp_norm_b,    cpu.tp_norm_b);

        printf("[GPU-EA] Encoder weights buffer: %.2f MB\n",
               ggml_backend_buffer_get_size(encoder_buffer_) / 1e6);
        return true;
    }

    bool init_adaptor_weights(const AdaptorWeights& cpu) {
        const int n_tensors = cpu.tensor_count() + 8;
        ggml_init_params params = {
            ggml_tensor_overhead() * static_cast<size_t>(n_tensors),
            nullptr,
            true,
        };
        adaptor_ctx_ = ggml_init(params);
        if (!adaptor_ctx_) {
            printf("[GPU-EA] Failed to create adaptor weights context\n");
            return false;
        }

        adaptor_weights_.linear1_w = mirror_tensor(adaptor_ctx_, cpu.linear1_w);
        adaptor_weights_.linear1_b = mirror_tensor(adaptor_ctx_, cpu.linear1_b);
        adaptor_weights_.linear2_w = mirror_tensor(adaptor_ctx_, cpu.linear2_w);
        adaptor_weights_.linear2_b = mirror_tensor(adaptor_ctx_, cpu.linear2_b);
        adaptor_weights_.blocks.resize(cpu.blocks.size());
        for (size_t i = 0; i < cpu.blocks.size(); ++i) {
            mirror_adaptor_block(adaptor_ctx_, adaptor_weights_.blocks[i], cpu.blocks[i]);
        }

        adaptor_buffer_ = ggml_backend_alloc_ctx_tensors(adaptor_ctx_, backend_);
        if (!adaptor_buffer_) {
            printf("[GPU-EA] Failed to alloc adaptor weights buffer\n");
            return false;
        }

        copy_tensor(adaptor_weights_.linear1_w, cpu.linear1_w);
        copy_tensor(adaptor_weights_.linear1_b, cpu.linear1_b);
        copy_tensor(adaptor_weights_.linear2_w, cpu.linear2_w);
        copy_tensor(adaptor_weights_.linear2_b, cpu.linear2_b);
        for (size_t i = 0; i < cpu.blocks.size(); ++i) {
            copy_adaptor_block(adaptor_weights_.blocks[i], cpu.blocks[i]);
        }

        printf("[GPU-EA] Adaptor weights buffer: %.2f MB\n",
               ggml_backend_buffer_get_size(adaptor_buffer_) / 1e6);
        return true;
    }

    void cleanup() {
        if (staging_buffer_) {
            ggml_backend_buffer_free(staging_buffer_);
        staging_buffer_ = nullptr;
        }
        if (staging_ctx_) {
            ggml_free(staging_ctx_);
            staging_ctx_ = nullptr;
        }
        staging_enc_ = nullptr;
        staging_adp_ = nullptr;
        staging_enc_dim_ = 0;
        staging_enc_frames_ = 0;
        staging_adp_dim_ = 0;
        staging_adp_frames_ = 0;

        if (encoder_buffer_) {
            ggml_backend_buffer_free(encoder_buffer_);
            encoder_buffer_ = nullptr;
        }
        if (adaptor_buffer_) {
            ggml_backend_buffer_free(adaptor_buffer_);
            adaptor_buffer_ = nullptr;
        }
        if (encoder_ctx_) {
            ggml_free(encoder_ctx_);
            encoder_ctx_ = nullptr;
        }
        if (adaptor_ctx_) {
            ggml_free(adaptor_ctx_);
            adaptor_ctx_ = nullptr;
        }
        encoder_weights_ = EncoderWeights{};
        adaptor_weights_ = AdaptorWeights{};
        input_dim_ = 0;
        initialized_ = false;
    }
};

} // namespace funasr

#endif // FUNASR_COMPUTE_ENCODER_ADAPTOR_GPU_HPP
