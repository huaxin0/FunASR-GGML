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
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

void setup_console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

std::string trim_ascii(std::string value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(begin, end - begin);
}

std::vector<std::string> parse_hotwords_text(const std::string& text) {
    std::vector<std::string> hotwords;
    std::string token;
    for (char ch : text) {
        if (ch == '\n' || ch == '\r' || ch == ',' || ch == ';' || ch == '\t') {
            std::string word = trim_ascii(token);
            if (!word.empty()) {
                hotwords.push_back(word);
            }
            token.clear();
        } else {
            token.push_back(ch);
        }
    }
    std::string word = trim_ascii(token);
    if (!word.empty()) {
        hotwords.push_back(word);
    }
    return hotwords;
}

bool load_hotwords_file(const std::string& path, std::vector<std::string>& hotwords) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    hotwords = parse_hotwords_text(buffer.str());
    return true;
}

int main(int argc, char* argv[]) {
    setup_console_utf8();

    if (argc < 2) {
        printf("Usage: %s <model.bin> [--gpu] [--push-to-talk] [--max-record-sec <n>] "
               "[--hotwords-file <path>] [--hotwords <text>]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    bool use_gpu = false;
    bool push_to_talk = false;
    int max_record_sec = 30;
    std::vector<std::string> hotwords;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--push-to-talk") {
            push_to_talk = true;
        } else if (arg == "--max-record-sec" && i + 1 < argc) {
            int parsed = std::atoi(argv[++i]);
            if (parsed > 0) {
                max_record_sec = parsed;
            }
        } else if (arg == "--hotwords-file" && i + 1 < argc) {
            std::string path = argv[++i];
            if (!load_hotwords_file(path, hotwords)) {
                printf("Failed to load hotwords file: %s\n", path.c_str());
                return 1;
            }
        } else if (arg == "--hotwords" && i + 1 < argc) {
            hotwords = parse_hotwords_text(argv[++i]);
        }
    }

    signal(SIGINT, signal_handler);

    printf("========================================\n");
    printf("  FunASR Realtime Demo\n");
    printf("  Mode: %s\n", use_gpu ? "GPU" : "CPU");
    printf("  Input: %s\n", push_to_talk ? "push-to-talk" : "energy VAD");
    printf("  Hotwords: %zu\n", hotwords.size());
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
        int gpu_ctx = push_to_talk ? 4096 : 2048;
        if (!recognizer.init_gpu(gpu_ctx, 0)) {
            printf("GPU init failed, falling back to CPU\n");
            use_gpu = false;
        } else {
            printf("  GPU ready! ctx=%d\n", gpu_ctx);
        }
    }

    // ===== 3. 初始化麦克风 =====
    printf("\n[3] Initializing microphone...\n");
    funasr::AudioCapture mic;

    funasr::PushToTalkRecorder ptt_recorder(16000, max_record_sec * 1000);
    funasr::RealtimeRecognizer realtime(recognizer);

    if (push_to_talk) {
        mic.set_callback([&](const float* samples, size_t count) {
            ptt_recorder.feed_audio(samples, count);
        });
    } else {
        mic.set_callback([&](const float* samples, size_t count) {
            realtime.feed_audio(samples, count);
        });
    }

    if (!mic.init(16000)) {
        printf("Failed to init microphone\n");
        return 1;
    }

    // ===== 5. 配置 =====
    funasr::RealtimeConfig config;
    config.inference.use_gpu = use_gpu;
    config.inference.max_new_tokens = push_to_talk ? 160 : 50;
    config.inference.kv_cache_size = push_to_talk ? 4096 : 2048;
    config.inference.prompt.hotwords = hotwords;
    if (!hotwords.empty()) {
        config.inference.prompt.language = "中文";
        config.inference.prompt.itn = true;
        printf("[Hotwords] ");
        for (size_t i = 0; i < hotwords.size(); i++) {
            printf("%s%s", i == 0 ? "" : ", ", hotwords[i].c_str());
        }
        printf("\n");
    }

    int utterance_count = 0;
    long total_audio_ms = 0;
    long total_inference_ms = 0;

    auto transcribe_recording = [&](const std::vector<float>& audio, bool truncated) {
        if (audio.empty()) {
            printf("No audio captured.\n");
            return;
        }

        float audio_sec = audio.size() / 16000.0f;
        float first_token_ms = -1.0f;
        auto t_start = std::chrono::high_resolution_clock::now();
        auto result = recognizer.transcribe_audio(
            audio.data(), audio.size(), config.inference,
            [&](int, const std::string&, bool is_final) {
                if (!is_final && first_token_ms < 0.0f) {
                    auto t_first = std::chrono::high_resolution_clock::now();
                    first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        t_first - t_start).count();
                }
            });
        auto t_end = std::chrono::high_resolution_clock::now();
        float inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();
        float rtf = inference_ms / (audio_sec * 1000.0f);

        utterance_count++;
        total_audio_ms += static_cast<long>(audio_sec * 1000);
        total_inference_ms += static_cast<long>(inference_ms);

        if (first_token_ms >= 0.0f) {
            printf("[%d] %s  (%.1fs%s, %.0fms, TTFT=%.0fms, RTF=%.2f)\n",
                   utterance_count, result.text.c_str(), audio_sec,
                   truncated ? ", truncated" : "",
                   inference_ms, first_token_ms, rtf);
        } else {
            printf("[%d] %s  (%.1fs%s, %.0fms, TTFT=n/a, RTF=%.2f)\n",
                   utterance_count, result.text.c_str(), audio_sec,
                   truncated ? ", truncated" : "",
                   inference_ms, rtf);
        }
    };

    // ===== 6. 启动 =====
    if (!push_to_talk) {
        realtime.start(config,
            [](int id, const std::string& text, float audio_sec,
               float inference_ms, float first_token_ms) {
                float rtf = inference_ms / (audio_sec * 1000.0f);
                if (first_token_ms >= 0.0f) {
                    printf("[%d] %s  (%.1fs, %.0fms, TTFT=%.0fms, RTF=%.2f)\n",
                           id, text.c_str(), audio_sec, inference_ms, first_token_ms, rtf);
                } else {
                    printf("[%d] %s  (%.1fs, %.0fms, TTFT=n/a, RTF=%.2f)\n",
                           id, text.c_str(), audio_sec, inference_ms, rtf);
                }
            }
        );
    }

    if (!mic.start()) {
        printf("Failed to start microphone\n");
        return 1;
    }

    // ===== 7. 主循环 =====
    if (push_to_talk) {
        printf("\nPush-to-talk mode:\n");
        printf("  Press Enter to start recording.\n");
        printf("  Press Enter again to stop and transcribe.\n");
        printf("  Max recording length: %d sec.\n", max_record_sec);
        printf("  Type q then Enter when idle to quit.\n\n");

        std::string line;
        while (g_running) {
            printf("[idle] Enter=start, q=quit > ");
            std::cout.flush();
            if (!std::getline(std::cin, line)) break;
            if (line == "q" || line == "Q") break;

            ptt_recorder.start();
            printf("[recording] Enter=stop (cap=%d sec)...\n", max_record_sec);
            if (!std::getline(std::cin, line)) break;
            std::vector<float> audio = ptt_recorder.stop();
            bool truncated = ptt_recorder.was_truncated();
            printf("[processing] captured %.1fs%s\n",
                   audio.size() / 16000.0f,
                   truncated ? " (truncated at cap)" : "");
            transcribe_recording(audio, truncated);
        }
    } else {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // ===== 8. 清理 =====
    printf("\n\nShutting down...\n");
    mic.stop();
    realtime.stop();

    // 统计
    printf("\n========================================\n");
    printf("  Session Statistics\n");
    printf("========================================\n");
    int printed_utterances = push_to_talk ? utterance_count : realtime.utterance_count();
    long printed_audio_ms = push_to_talk ? total_audio_ms : realtime.total_audio_ms();
    long printed_inference_ms = push_to_talk ? total_inference_ms : realtime.total_inference_ms();
    printf("  Utterances: %d\n", printed_utterances);
    printf("  Total audio: %.1f sec\n", printed_audio_ms / 1000.0f);
    printf("  Total inference: %.1f sec\n", printed_inference_ms / 1000.0f);
    if (printed_audio_ms > 0) {
        printf("  Average RTF: %.2f\n",
               static_cast<float>(printed_inference_ms) / printed_audio_ms);
    }
    printf("========================================\n");

    return 0;
}
