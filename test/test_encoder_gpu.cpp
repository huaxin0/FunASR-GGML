#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "compute/fbank.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
#include "compute/graph_runner.hpp"
#include "compute/gpu_context.hpp"
#include "compute/encoder_adaptor_gpu.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

static void compare_outputs(const float* cpu, const float* gpu, int n, const char* label) {
    float max_diff = 0.0f;
    double mean_diff = 0.0;
    float max_rel = 0.0f;
    double mean_rel = 0.0;
    int max_idx = 0;

    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(cpu[i] - gpu[i]);
        float denom = std::max(std::fabs(cpu[i]), 1e-6f);
        float rel = diff / denom;
        mean_diff += diff;
        mean_rel += rel;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (rel > max_rel) {
            max_rel = rel;
        }
    }

    mean_diff /= n;
    mean_rel /= n;
    printf("%s\n", label);
    printf("  Max abs diff:  %.6f (idx=%d)\n", max_diff, max_idx);
    printf("  Mean abs diff: %.6f\n", static_cast<float>(mean_diff));
    printf("  Max rel diff:  %.6f\n", max_rel);
    printf("  Mean rel diff: %.6f\n", static_cast<float>(mean_rel));
    printf("  CPU[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
    printf("  GPU[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]);
}

static bool run_cpu_adaptor(const funasr::FunASRModel& model,
                            const float* encoder_data,
                            int frames,
                            std::vector<float>& adaptor_out,
                            long& adaptor_ms) {
    adaptor_out.clear();
    adaptor_ms = 0;

    ggml_init_params params = { 1024ULL * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor* encoder_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, model.config.encoder.output_size, frames);
    std::memcpy(encoder_in->data, encoder_data,
                static_cast<size_t>(model.config.encoder.output_size) * frames * sizeof(float));

    ggml_tensor* adp_out = funasr::adaptor_forward(
        ctx, encoder_in, model.adaptor, model.config.adaptor);

    auto t0 = std::chrono::high_resolution_clock::now();
    funasr::run_graph(ctx, adp_out, 4);
    auto t1 = std::chrono::high_resolution_clock::now();
    adaptor_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    adaptor_out.resize(static_cast<size_t>(adp_out->ne[0]) * adp_out->ne[1]);
    std::memcpy(adaptor_out.data(), adp_out->data, adaptor_out.size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

#ifndef FUNASR_USE_CUDA
    printf("This test requires CUDA. Build with -DFUNASR_CUDA=ON\n");
    return 1;
#else
    const std::string model_path = argv[1];
    const std::string wav_path = argv[2];

    printf("========================================\n");
    printf("  FunASR Encoder/Adaptor GPU Test\n");
    printf("========================================\n");

    funasr::GGUFReader reader;
    if (!reader.open(model_path)) return 1;

    funasr::FunASRModel model;
    if (!funasr::ModelLoader::load(reader, model)) return 1;

    funasr::FbankProcessor fbank(model.config.frontend);
    std::vector<float> fbank_data;
    int T = 0;
    int D = 0;
    if (!fbank.process_wav(wav_path, fbank_data, T, D)) {
        printf("Failed to compute fbank\n");
        return 1;
    }

    std::vector<float> cpu_enc_out;
    std::vector<float> cpu_adp_out;
    long cpu_ms = 0;
    long cpu_encoder_ms = 0;
    long cpu_adaptor_ms = 0;
    {
        ggml_init_params params = { 4ULL * 1024 * 1024 * 1024, nullptr, false };
        ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("Failed to allocate CPU context\n");
            return 1;
        }

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
        std::memcpy(input->data, fbank_data.data(), fbank_data.size() * sizeof(float));

        ggml_tensor* enc_out = funasr::encoder_forward(
            ctx, input, model.encoder, model.config.encoder);
        ggml_tensor* adp_out = funasr::adaptor_forward(
            ctx, enc_out, model.adaptor, model.config.adaptor);

        auto t0 = std::chrono::high_resolution_clock::now();
        funasr::run_graph(ctx, enc_out, 4);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_encoder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        cpu_enc_out.resize(static_cast<size_t>(enc_out->ne[0]) * enc_out->ne[1]);
        std::memcpy(cpu_enc_out.data(), enc_out->data, cpu_enc_out.size() * sizeof(float));

        auto t2 = std::chrono::high_resolution_clock::now();
        funasr::run_graph(ctx, adp_out, 4);
        auto t3 = std::chrono::high_resolution_clock::now();
        cpu_adaptor_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        cpu_ms = cpu_encoder_ms + cpu_adaptor_ms;

        cpu_adp_out.resize(static_cast<size_t>(adp_out->ne[0]) * adp_out->ne[1]);
        std::memcpy(cpu_adp_out.data(), adp_out->data, cpu_adp_out.size() * sizeof(float));

        printf("CPU encoder: [%lld, %lld], %ld ms\n",
               (long long)enc_out->ne[0], (long long)enc_out->ne[1], cpu_encoder_ms);
        printf("CPU adaptor: [%lld, %lld], %ld ms\n",
               (long long)adp_out->ne[0], (long long)adp_out->ne[1], cpu_adaptor_ms);
        ggml_free(ctx);
    }

    funasr::GPUContext gpu_ctx;
    if (!gpu_ctx.init(model.llm, model.config.llm, 2048, 0)) {
        printf("Failed to init GPU context\n");
        return 1;
    }

    funasr::GPUEncoderAdaptorRunner runner(gpu_ctx.backend());
    if (!runner.init(model.encoder, model.adaptor,
                     model.config.encoder, model.config.adaptor)) {
        printf("Failed to init GPU encoder/adaptor runner\n");
        return 1;
    }

    if (!runner.warmup(128)) {
        printf("GPU warmup failed\n");
    }

    std::vector<float> gpu_enc_out;
    std::vector<float> gpu_adp_out;
    long gpu_encoder_ms = 0;
    long gpu_adaptor_ms = 0;
    int enc_frames = runner.forward_encoder_timed(
        fbank_data.data(), D, T, gpu_enc_out, gpu_encoder_ms);
    if (enc_frames <= 0) {
        printf("GPU encoder failed\n");
        return 1;
    }

    int out_frames = runner.forward_adaptor_timed(
        gpu_enc_out.data(), model.config.encoder.output_size, enc_frames,
        gpu_adp_out, gpu_adaptor_ms);
    if (out_frames <= 0) {
        printf("GPU adaptor failed\n");
        return 1;
    }

    long gpu_ms = gpu_encoder_ms + gpu_adaptor_ms;
    printf("GPU encoder: [512, %d], %ld ms\n", enc_frames, gpu_encoder_ms);
    printf("GPU adaptor: [1024, %d], %ld ms\n", out_frames, gpu_adaptor_ms);
    compare_outputs(cpu_enc_out.data(), gpu_enc_out.data(),
                    static_cast<int>(cpu_enc_out.size()), "\n[Encoder Diff]");
    compare_outputs(cpu_adp_out.data(), gpu_adp_out.data(),
                    static_cast<int>(cpu_adp_out.size()), "\n[Adaptor Diff]");

    std::vector<float> cpu_enc_gpu_adp;
    std::vector<float> gpu_enc_cpu_adp;
    long cpu_enc_gpu_adp_ms = 0;
    long gpu_enc_cpu_adp_ms = 0;

    if (!runner.forward_adaptor_timed(cpu_enc_out.data(), model.config.encoder.output_size,
                                      enc_frames, cpu_enc_gpu_adp, cpu_enc_gpu_adp_ms)) {
        printf("CPU enc -> GPU adaptor failed\n");
        return 1;
    }
    if (!run_cpu_adaptor(model, gpu_enc_out.data(), enc_frames,
                         gpu_enc_cpu_adp, gpu_enc_cpu_adp_ms)) {
        printf("GPU enc -> CPU adaptor failed\n");
        return 1;
    }

    compare_outputs(cpu_adp_out.data(), cpu_enc_gpu_adp.data(),
                    static_cast<int>(cpu_adp_out.size()), "\n[CPU enc -> GPU adp vs CPU baseline]");
    compare_outputs(cpu_adp_out.data(), gpu_enc_cpu_adp.data(),
                    static_cast<int>(cpu_adp_out.size()), "\n[GPU enc -> CPU adp vs CPU baseline]");
    compare_outputs(cpu_enc_gpu_adp.data(), gpu_adp_out.data(),
                    static_cast<int>(cpu_adp_out.size()), "\n[Same GPU adp, CPU enc vs GPU enc]");

    if (gpu_ms > 0) {
        printf("Speedup: %.2fx\n", static_cast<float>(cpu_ms) / gpu_ms);
    }

    return 0;
#endif
}
