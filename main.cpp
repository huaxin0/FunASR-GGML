// main.cpp — FunASR-GGML minimal example
//
// Build:
//   cmake .. [-DFUNASR_CUDA=ON] && make funasr_main
// Run:
//   ./funasr_main FunAsr_q8.bin audio.wav [--gpu]
//
#include "pipeline/recognizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav> [--gpu]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];
    bool use_gpu = false;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--gpu") use_gpu = true;
    }

    // Initialize
    funasr::Recognizer recognizer;
    if (!recognizer.init(model_path)) return 1;

    if (use_gpu) {
        if (!recognizer.init_gpu()) {
            printf("GPU init failed, using CPU\n");
            use_gpu = false;
        }
    }

    // Transcribe
    funasr::InferenceConfig config;
    config.use_gpu = use_gpu;

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

    // Summary
    printf("\nText:    %s\n", result.text.c_str());
    printf("Tokens:  %d\n", result.decode_tokens);
    printf("Time:    %.0f ms (encoder %.0f + prefill %.0f + decode %.0f)\n",
           result.total_ms, result.encoder_ms, result.prefill_ms, result.decode_ms);

    return 0;
}