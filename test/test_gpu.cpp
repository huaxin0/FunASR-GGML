// GPU 推理测试
//
// 构建:
//   cmake .. -DFUNASR_CUDA=ON && make test_gpu
// 运行:
//   ./test_gpu ../FunAsr_q8.bin ../zh.wav
//
#include "pipeline/recognizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav> [audio2.wav ...]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    printf("========================================\n");
    printf("  FunASR GPU Inference Test\n");
    printf("========================================\n");

    // 1. 初始化模型
    funasr::Recognizer recognizer;
    if (!recognizer.init(model_path)) {
        printf("Init failed\n");
        return 1;
    }

    // 2. 初始化 GPU
    printf("\nInitializing GPU...\n");
    if (!recognizer.init_gpu(2048, 0)) {
        printf("GPU init failed (is CUDA compiled and GPU available?)\n");
        return 1;
    }
    printf("GPU ready: %s\n", recognizer.is_gpu_ready() ? "yes" : "no");

    // 3. GPU 配置
    funasr::InferenceConfig config;
    config.use_gpu = true;
    config.max_new_tokens = 100;

    // 4. 推理每个文件
    for (int i = 2; i < argc; i++) {
        std::string wav_path = argv[i];
        printf("\n--- %s ---\n", wav_path.c_str());

        // 流式输出
        auto result = recognizer.transcribe(wav_path, config,
            [](int id, const std::string& text, bool is_final) {
                if (!is_final) {
                    printf("%s", text.c_str());
                    fflush(stdout);
                } else {
                    printf("\n");
                }
            }
        );

        printf("  Text:     '%s'\n", result.text.c_str());
        printf("  Encoder:  %.0f ms\n", result.encoder_ms);
        printf("  Prefill:  %.0f ms (%d tokens)\n", result.prefill_ms, result.prefill_tokens);
        printf("  Decode:   %.0f ms (%d tokens)\n", result.decode_ms, result.decode_tokens);
        printf("  Total:    %.0f ms\n", result.total_ms);
        if (result.decode_tokens > 0 && result.decode_ms > 0) {
            printf("  Speed:    %.1f tok/s\n", result.decode_tokens * 1000.0 / result.decode_ms);
        }
    }

    // 5. CPU vs GPU 对比
    printf("\n--- CPU vs GPU Comparison ---\n");
    {
        funasr::InferenceConfig cpu_cfg;
        cpu_cfg.use_gpu = false;

        funasr::InferenceConfig gpu_cfg;
        gpu_cfg.use_gpu = true;

        auto cpu_result = recognizer.transcribe(argv[2], cpu_cfg);
        auto gpu_result = recognizer.transcribe(argv[2], gpu_cfg);

        printf("  CPU: '%s' (%.0f ms, decode %.1f tok/s)\n",
               cpu_result.text.c_str(), cpu_result.total_ms,
               cpu_result.decode_tokens > 0 ? cpu_result.decode_tokens * 1000.0 / cpu_result.decode_ms : 0);
        printf("  GPU: '%s' (%.0f ms, decode %.1f tok/s)\n",
               gpu_result.text.c_str(), gpu_result.total_ms,
               gpu_result.decode_tokens > 0 ? gpu_result.decode_tokens * 1000.0 / gpu_result.decode_ms : 0);

        bool text_match = (cpu_result.text == gpu_result.text);
        printf("  Text match: %s\n", text_match ? "YES" : "NO (may differ due to float precision)");
    }

    printf("\n========================================\n");
    printf("  Done!\n");
    printf("========================================\n");

    return 0;
}