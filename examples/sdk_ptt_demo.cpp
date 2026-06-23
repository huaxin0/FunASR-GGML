#include "funasr_sdk.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

void setup_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

int main(int argc, char** argv) {
    setup_console_utf8();

    if (argc < 2) {
        std::printf("Usage: %s <model.bin> [--gpu]\n", argv[0]);
        return 1;
    }

    FunasrConfig config;
    funasr_get_default_config(&config);
    config.model_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--gpu") {
            config.use_gpu = 1;
        }
    }

    FunasrHandle handle = funasr_create();
    int rc = funasr_init(handle, &config);
    if (rc != 0) {
        std::printf("init failed: %s\n", funasr_last_error(handle));
        funasr_destroy(handle);
        return 1;
    }

    // This demo feeds one second of silence through the SDK API. Host
    // applications should feed real microphone PCM from their audio callback.
    std::vector<float> audio(static_cast<size_t>(config.sample_rate), 0.0f);
    char text[4096] = {};
    FunasrResult result = {};
    rc = funasr_transcribe_f32(handle, audio.data(), static_cast<int>(audio.size()),
                               text, sizeof(text), &result);
    std::printf("rc=%d text=\"%s\" audio=%.1fs total=%.0fms ttft=%.0fms rtf=%.2f\n",
                rc, text, result.audio_sec, result.total_ms,
                result.first_token_ms, result.rtf);

    funasr_destroy(handle);
    return 0;
}
