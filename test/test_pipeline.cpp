// Pipeline 层测试 — 用最简洁的 Recognizer API 做端到端推理
//
// 构建:
//   cd build && cmake .. && make test_pipeline
// 运行:
//   ./test_pipeline ../FunAsr_q8.bin ../zh.wav
//   ./test_pipeline ../FunAsr_q8.bin ../zh.wav ../yue.wav   (多文件)
//
#include "pipeline/recognizer.hpp"
#include <cstdio>
#include <string>
#include <vector>

// ============================================================
// 简易测试框架
// ============================================================
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

// ============================================================
// Test 1: Recognizer 初始化
// ============================================================
void test_init(const std::string& model_path) {
    printf("\n--- Test 1: Recognizer Init ---\n");

    funasr::Recognizer recognizer;
    TEST_ASSERT(!recognizer.is_ready(), "not ready before init");

    bool ok = recognizer.init(model_path);
    TEST_ASSERT(ok, "init succeeds");
    TEST_ASSERT(recognizer.is_ready(), "ready after init");

    printf("  [OK] Init passed\n");
}

// ============================================================
// Test 2: 最简调用 — 一行 transcribe
// ============================================================
void test_simple_transcribe(funasr::Recognizer& recognizer, const std::string& wav_path) {
    printf("\n--- Test 2: Simple Transcribe ---\n");

    auto result = recognizer.transcribe(wav_path);

    printf("  Text:   '%s'\n", result.text.c_str());
    printf("  Tokens: %zu\n", result.token_ids.size());

    TEST_ASSERT(!result.text.empty(), "text non-empty");
    TEST_ASSERT(!result.token_ids.empty(), "tokens non-empty");

    printf("  [OK] Simple transcribe passed\n");
}

// ============================================================
// Test 3: 流式回调
// ============================================================
void test_streaming_callback(funasr::Recognizer& recognizer, const std::string& wav_path) {
    printf("\n--- Test 3: Streaming Callback ---\n");

    int callback_count = 0;
    bool saw_eos = false;

    auto result = recognizer.transcribe(wav_path, funasr::InferenceConfig(),
        [&](int token_id, const std::string& text, bool is_final) {
            callback_count++;
            if (is_final) {
                saw_eos = true;
            } else {
                printf("%s", text.c_str());
                fflush(stdout);
            }
        }
    );

    printf("\n");
    printf("  Callbacks: %d, EOS: %s\n", callback_count, saw_eos ? "yes" : "no");
    printf("  Text: '%s'\n", result.text.c_str());

    TEST_ASSERT(callback_count > 0, "got callbacks");
    TEST_ASSERT(saw_eos, "saw EOS");
    TEST_ASSERT(!result.text.empty(), "text non-empty");

    printf("  [OK] Streaming callback passed\n");
}

// ============================================================
// Test 4: 性能统计
// ============================================================
void test_performance(funasr::Recognizer& recognizer, const std::string& wav_path) {
    printf("\n--- Test 4: Performance ---\n");

    auto result = recognizer.transcribe(wav_path);

    printf("  Encoder+Adaptor: %.0f ms\n", result.encoder_ms);
    printf("  Prefill:         %.0f ms (%d tokens)\n", result.prefill_ms, result.prefill_tokens);
    printf("  Decode:          %.0f ms (%d tokens)\n", result.decode_ms, result.decode_tokens);
    printf("  Total:           %.0f ms\n", result.total_ms);

    if (result.decode_tokens > 0 && result.decode_ms > 0) {
        float tok_per_sec = result.decode_tokens * 1000.0f / result.decode_ms;
        printf("  Speed:           %.1f tok/s\n", tok_per_sec);
    }

    TEST_ASSERT(result.total_ms > 0, "total_ms > 0");
    TEST_ASSERT(result.encoder_ms > 0, "encoder_ms > 0");

    printf("  [OK] Performance passed\n");
}

// ============================================================
// Test 5: 自定义配置
// ============================================================
void test_custom_config(funasr::Recognizer& recognizer, const std::string& wav_path) {
    printf("\n--- Test 5: Custom Config ---\n");

    funasr::InferenceConfig config;
    config.max_new_tokens = 5;  // 只生成 5 个 token
    config.n_threads = 2;       // 2 线程

    auto result = recognizer.transcribe(wav_path, config);

    printf("  Text (max 5 tokens): '%s'\n", result.text.c_str());
    printf("  Tokens generated: %d\n", result.decode_tokens);

    // 应该最多 5 个（可能更少如果遇到 EOS）
    TEST_ASSERT(result.decode_tokens <= 5, "max_new_tokens respected");
    TEST_ASSERT(!result.text.empty(), "still produces text");

    printf("  [OK] Custom config passed\n");
}

// ============================================================
// Test 6: 多文件连续推理
// ============================================================
void test_multi_file(funasr::Recognizer& recognizer, const std::vector<std::string>& wav_files) {
    printf("\n--- Test 6: Multi-File ---\n");

    for (const auto& wav : wav_files) {
        auto result = recognizer.transcribe(wav);
        printf("  %s → '%s' (%.0f ms)\n", wav.c_str(), result.text.c_str(), result.total_ms);
        TEST_ASSERT(!result.text.empty(), ("non-empty for " + wav).c_str());
    }

    printf("  [OK] Multi-file passed\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav> [audio2.wav ...]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    std::vector<std::string> wav_files;
    for (int i = 2; i < argc; i++) {
        wav_files.push_back(argv[i]);
    }

    printf("========================================\n");
    printf("  FunASR Pipeline Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());
    printf("WAV files: %zu\n", wav_files.size());

    // Test 1: init 用独立的 recognizer（测完析构）
    test_init(model_path);

    // 后续测试共用一个 recognizer
    funasr::Recognizer recognizer;
    if (!recognizer.init(model_path, false)) {
        printf("Init failed: %s\n", recognizer.last_error().c_str());
        return 1;
    }

    test_simple_transcribe(recognizer, wav_files[0]);
    test_streaming_callback(recognizer, wav_files[0]);
    test_performance(recognizer, wav_files[0]);
    test_custom_config(recognizer, wav_files[0]);

    if (wav_files.size() > 1) {
        test_multi_file(recognizer, wav_files);
    }

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}