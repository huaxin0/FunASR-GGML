// SDK C ABI smoke tests that do not require a model file.
#include "funasr_sdk.h"
#include "pipeline/prompt_builder.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

void test_default_config() {
    printf("\n--- Test 1: default config ---\n");

    FunasrConfig config;
    funasr_get_default_config(&config);

    TEST_ASSERT(config.model_path == nullptr, "model path defaults to null");
    TEST_ASSERT(config.use_gpu == 0, "GPU defaults off");
    TEST_ASSERT(config.ctx_size == 4096, "ctx size defaults to 4096");
    TEST_ASSERT(config.max_new_tokens == 220, "max_new_tokens defaults to 220");
    TEST_ASSERT(config.max_record_ms == 60000, "max record defaults to 60s");
    TEST_ASSERT(config.sample_rate == 16000, "sample rate defaults to 16k");
}

void test_create_destroy_and_errors() {
    printf("\n--- Test 2: create/destroy and errors ---\n");

    FunasrHandle handle = funasr_create();
    TEST_ASSERT(handle != nullptr, "create returns handle");
    TEST_ASSERT(funasr_last_error(handle) != nullptr, "last_error is never null");

    FunasrConfig config;
    funasr_get_default_config(&config);
    int rc = funasr_init(handle, &config);
    TEST_ASSERT(rc < 0, "init without model path fails");
    TEST_ASSERT(std::strlen(funasr_last_error(handle)) > 0, "init failure sets last_error");

    funasr_destroy(handle);
    funasr_destroy(nullptr);
}

void test_uninitialized_transcribe_fails() {
    printf("\n--- Test 3: uninitialized transcribe fails ---\n");

    FunasrHandle handle = funasr_create();
    float samples[4] = {0.0f, 0.1f, -0.1f, 0.0f};
    char text[32] = {};
    FunasrResult result = {};

    int rc = funasr_transcribe_f32(handle, samples, 4, text, sizeof(text), &result);
    TEST_ASSERT(rc < 0, "transcribe before init fails");
    TEST_ASSERT(text[0] == '\0', "text remains empty on failure");

    funasr_destroy(handle);
}

void test_ptt_feed_is_callable_before_inference() {
    printf("\n--- Test 4: push-to-talk feed API ---\n");

    FunasrHandle handle = funasr_create();
    float f32_samples[4] = {0.0f, 0.5f, -0.5f, 0.0f};
    int16_t i16_samples[4] = {0, 16384, -16384, 0};

    TEST_ASSERT(funasr_ptt_start(handle) == 0, "ptt_start succeeds");
    TEST_ASSERT(funasr_ptt_feed_f32(handle, f32_samples, 4) == 0, "feed_f32 succeeds");
    TEST_ASSERT(funasr_ptt_feed_i16(handle, i16_samples, 4) == 0, "feed_i16 succeeds");

    char text[32] = {};
    FunasrResult result = {};
    int rc = funasr_ptt_stop_and_transcribe(handle, text, sizeof(text), &result);
    TEST_ASSERT(rc < 0, "stop_and_transcribe before init fails");
    TEST_ASSERT(result.audio_sec > 0.0f, "result reports captured audio duration");

    funasr_destroy(handle);
}

void test_hotword_prompt_matches_official_nano_shape() {
    printf("\n--- Test 5: hotword prompt shape ---\n");

    std::vector<std::string> hotwords = {"开放时间", "FunASR", "张三"};
    std::string prompt = funasr::PromptBuilder::build_user_prompt(
        hotwords, "中文", true);

    TEST_ASSERT(prompt.find("热词列表：[开放时间, FunASR, 张三]") != std::string::npos,
                "prompt includes hotword list");
    TEST_ASSERT(prompt.find("语音转写成中文：") != std::string::npos,
                "prompt includes Chinese transcription request");
}

void test_set_and_load_hotwords_are_callable_before_init() {
    printf("\n--- Test 6: hotword SDK API ---\n");

    FunasrHandle handle = funasr_create();
    TEST_ASSERT(funasr_set_hotwords(handle, "开放时间\nFunASR\n张三") == 0,
                "set_hotwords accepts newline-separated text");
    TEST_ASSERT(funasr_set_hotwords(handle, "") == 0,
                "set_hotwords accepts empty text to clear hotwords");

    const char* path = "test_hotwords_sdk.txt";
    {
        std::ofstream out(path);
        out << "广东话\n开放时间\n";
    }
    TEST_ASSERT(funasr_load_hotwords_file(handle, path) == 0,
                "load_hotwords_file accepts UTF-8 text file");
    TEST_ASSERT(funasr_load_hotwords_file(handle, "missing_hotwords.txt") < 0,
                "load_hotwords_file fails for missing file");

    std::remove(path);
    funasr_destroy(handle);
}

int main() {
    test_default_config();
    test_create_destroy_and_errors();
    test_uninitialized_transcribe_fails();
    test_ptt_feed_is_callable_before_inference();
    test_hotword_prompt_matches_official_nano_shape();
    test_set_and_load_hotwords_are_callable_before_init();

    printf("\n========================================\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================\n");

    return tests_failed == 0 ? 0 : 1;
}
