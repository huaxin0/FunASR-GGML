// test/test_benchmark.cpp
// AISHELL-1 批量评测
//
// 用法:
//   ./test_benchmark <model.bin> <wav_dir> <output.txt> [--gpu] [--limit N]
//
// 示例:
//   ./test_benchmark ../FunAsr_q8.bin ~/data_aishell/wav/test results.txt --gpu --limit 100
//
#include "pipeline/recognizer.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

// 递归收集目录下所有 .wav 文件
std::vector<std::string> collect_wav_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") {
                files.push_back(path);
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}


std::string get_utterance_id(const std::string& path) {
    std::string filename = fs::path(path).stem().string();
    return filename;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <model.bin> <wav_dir> <output.txt> [--gpu] [--limit N]\n", argv[0]);
        printf("Example: %s ../FunAsr_q8.bin ~/data_aishell/wav/test results.txt --gpu --limit 100\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_dir    = argv[2];
    const std::string output_path = argv[3];
    bool use_gpu = false;
    int limit = -1;
    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "--gpu") use_gpu = true;
        if (std::string(argv[i]) == "--limit" && i + 1 < argc) {
            limit = std::max(1, std::atoi(argv[++i]));
        }
    }

    // ===== 1. 收集 WAV 文件 =====
    printf("[1] Scanning WAV files in: %s\n", wav_dir.c_str());
    auto wav_files = collect_wav_files(wav_dir);
    if (limit > 0 && static_cast<int>(wav_files.size()) > limit) {
        wav_files.resize(limit);
    }
    printf("    Found %zu files\n\n", wav_files.size());

    if (wav_files.empty()) {
        printf("No WAV files found!\n");
        return 1;
    }

    // ===== 2. 初始化模型 =====
    printf("[2] Loading model: %s\n", model_path.c_str());
    funasr::Recognizer recognizer;
    if (!recognizer.init(model_path)) {
        printf("Failed to init model: %s\n", recognizer.last_error().c_str());
        return 1;
    }

    if (use_gpu) {
        printf("    Initializing GPU...\n");
        if (!recognizer.init_gpu(2048, 0)) {
            printf("    GPU init failed, falling back to CPU\n");
            use_gpu = false;
        } else {
            printf("    GPU ready!\n");
        }
    }
    printf("    Mode: %s\n\n", use_gpu ? "GPU" : "CPU");

    // ===== 3. 推理配置 =====
    funasr::InferenceConfig config;
    config.use_gpu = use_gpu;
    config.max_new_tokens = 100;

    // ===== 4. 批量推理 =====
    printf("[3] Running inference on %zu files...\n", wav_files.size());

    std::ofstream out(output_path);
    if (!out.is_open()) {
        printf("Cannot open output file: %s\n", output_path.c_str());
        return 1;
    }

    int total = static_cast<int>(wav_files.size());
    int success = 0;
    int failed = 0;
    float total_inference_ms = 0.0f;
    float total_inference_ms_excl_first = 0.0f;
    float first_file_ms = 0.0f;

    auto t_all_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total; i++) {
        const std::string& wav_path = wav_files[i];
        std::string utt_id = get_utterance_id(wav_path);

        auto t_start = std::chrono::high_resolution_clock::now();
        auto result = recognizer.transcribe(wav_path, config);
        auto t_end = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        if (!result.text.empty()) {
            // 输出格式: utterance_id \t 识别结果
            out << utt_id << "\t" << result.text << "\n";
            success++;
            total_inference_ms += ms;
            if (i == 0) {
                first_file_ms = ms;
            } else {
                total_inference_ms_excl_first += ms;
            }

            // 估算音频时长（从 encoder_ms 和 RTF 反推，或直接用文件大小估算）
            // 简单方法: 读 WAV header 获取采样数
        } else {
            out << utt_id << "\t" << "[EMPTY]" << "\n";
            failed++;
        }

        // 进度显示
        if ((i + 1) % 100 == 0 || i == total - 1) {
            float progress = (i + 1) * 100.0f / total;
            printf("\r    [%d/%d] %.1f%% | OK: %d, Failed: %d | Avg: %.0f ms/file",
                   i + 1, total, progress, success, failed,
                   success > 0 ? total_inference_ms / success : 0);
            fflush(stdout);
        }
    }

    auto t_all_end = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration_cast<std::chrono::milliseconds>(t_all_end - t_all_start).count() / 1000.0f;

    printf("\n\n");
    printf("========================================\n");
    printf("  Benchmark Results\n");
    printf("========================================\n");
    printf("  Total files:    %d\n", total);
    printf("  Success:        %d\n", success);
    printf("  Failed:         %d\n", failed);
    printf("  Total time:     %.1f sec\n", total_sec);
    printf("  Avg per file:   %.0f ms\n", success > 0 ? total_inference_ms / success : 0);
    if (success > 0) {
        printf("  First file:     %.0f ms\n", first_file_ms);
    }
    if (success > 1) {
        printf("  Avg excl first: %.0f ms\n", total_inference_ms_excl_first / (success - 1));
    }
    printf("  Throughput:     %.1f files/sec\n", total / total_sec);
    printf("  Output:         %s\n", output_path.c_str());
    printf("========================================\n");
    printf("\nRun CER evaluation:\n");
    printf("  python3 tools/eval_cer.py %s ~/data_aishell/transcript/aishell_transcript_v0.8.txt\n",
           output_path.c_str());

    out.close();
    return 0;
}
