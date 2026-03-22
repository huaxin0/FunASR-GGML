// 实时麦克风语音识别 Demo
//
// 构建:
//   cmake .. [-DFUNASR_CUDA=ON] && make test_realtime
// 运行:
//   ./test_realtime ../FunAsr_q8.bin           # CPU 模式
//   ./test_realtime ../FunAsr_q8.bin --gpu     # GPU 模式
//
#include "pipeline/recognizer.hpp"
#include "pipeline/audio_capture.hpp"
#include "pipeline/realtime.hpp"
#include <cstdio>
#include <csignal>
#include <string>
#include <atomic>

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.bin> [--gpu]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    bool use_gpu = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--gpu") use_gpu = true;
    }

    signal(SIGINT, signal_handler);

    printf("========================================\n");
    printf("  FunASR Realtime Demo\n");
    printf("  Mode: %s\n", use_gpu ? "GPU" : "CPU");
    printf("========================================\n\n");

    // ===== 1. 初始化模型 =====
    printf("[1] Loading model...\n");
    funasr::Recognizer recognizer;
    if (!recognizer.init(model_path)) {
        printf("Failed to init model\n");
        return 1;
    }

    // ===== 2. GPU 初始化（可选）=====
    if (use_gpu) {
        printf("\n[2] Initializing GPU...\n");
        if (!recognizer.init_gpu(2048, 0)) {
            printf("GPU init failed, falling back to CPU\n");
            use_gpu = false;
        } else {
            printf("  GPU ready!\n");
        }
    }

    // ===== 3. 初始化麦克风 =====
    printf("\n[3] Initializing microphone...\n");
    funasr::AudioCapture mic;

    // ===== 4. 创建实时识别器 =====
    funasr::RealtimeRecognizer realtime(recognizer);

    // 连接麦克风 → 实时识别器
    mic.set_callback([&](const float* samples, size_t count) {
        realtime.feed_audio(samples, count);
    });

    if (!mic.init(16000)) {
        printf("Failed to init microphone\n");
        return 1;
    }

    // ===== 5. 配置 =====
    funasr::RealtimeConfig config;
    config.inference.use_gpu = use_gpu;
    config.inference.max_new_tokens = 50;

    // ===== 6. 启动 =====
    realtime.start(config,
        [](int id, const std::string& text, float audio_sec, float inference_ms) {
            float rtf = inference_ms / (audio_sec * 1000.0f);
            printf("[%d] %s  (%.1fs, %.0fms, RTF=%.2f)\n",
                   id, text.c_str(), audio_sec, inference_ms, rtf);
        }
    );

    if (!mic.start()) {
        printf("Failed to start microphone\n");
        return 1;
    }

    // ===== 7. 主循环（等待 Ctrl+C）=====
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ===== 8. 清理 =====
    printf("\n\nShutting down...\n");
    mic.stop();
    realtime.stop();

    // 统计
    printf("\n========================================\n");
    printf("  Session Statistics\n");
    printf("========================================\n");
    printf("  Utterances: %d\n", realtime.utterance_count());
    printf("  Total audio: %.1f sec\n", realtime.total_audio_ms() / 1000.0f);
    printf("  Total inference: %.1f sec\n", realtime.total_inference_ms() / 1000.0f);
    if (realtime.total_audio_ms() > 0) {
        printf("  Average RTF: %.2f\n",
               static_cast<float>(realtime.total_inference_ms()) / realtime.total_audio_ms());
    }
    printf("========================================\n");

    return 0;
}