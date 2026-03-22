// 验证 Fbank 特征提取
//
// 构建:
//   cd build && cmake .. && make test_fbank
// 运行:
//   ./test_fbank ../zh.wav
#include "core/config.hpp"
#include "compute/fbank.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

// ============================================================
// 简易测试框架
// ============================================================
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define TEST_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
           name, (int)(expected), (int)(actual), __LINE__); } \
} while(0)

// ============================================================
// Test 1: WAV 读取
// ============================================================
void test_wav_read(const std::string& wav_path) {
    printf("\n--- Test 1: WAV Read ---\n");

    funasr::FrontendConfig config;  // 默认值
    funasr::FbankProcessor fbank(config);

    std::vector<float> audio;
    int sr;
    bool ok = fbank.read_wav(wav_path, audio, sr);
    TEST_ASSERT(ok, "read_wav succeeds");
    TEST_EQ(sr, 16000, "sample_rate = 16000");
    TEST_ASSERT(!audio.empty(), "audio non-empty");

    // 音频应该在 ±1.0 范围
    float max_abs = 0.0f;
    for (float s : audio) {
        float a = std::fabs(s);
        if (a > max_abs) max_abs = a;
    }
    TEST_ASSERT(max_abs <= 1.0f, "audio normalized to ±1.0");
    TEST_ASSERT(max_abs > 0.001f, "audio is not silence");

    float duration = audio.size() / 16000.0f;
    printf("  Audio: %zu samples, %.2f sec, max_abs=%.4f\n",
           audio.size(), duration, max_abs);

    printf("  [OK] WAV read passed\n");
}

// ============================================================
// Test 2: Fbank 基本功能
// ============================================================
void test_fbank_basic(const std::string& wav_path) {
    printf("\n--- Test 2: Fbank Basic ---\n");

    funasr::FrontendConfig config;
    funasr::FbankProcessor fbank(config);

    std::vector<float> fbank_out;
    int T, D;
    bool ok = fbank.process_wav(wav_path, fbank_out, T, D);
    TEST_ASSERT(ok, "process_wav succeeds");

    // D 应该是 560 = 80 × 7
    TEST_EQ(D, 560, "D = 560 (80 × 7)");

    // T > 0
    TEST_ASSERT(T > 0, "T > 0");

    // fbank_out 大小 = T × D
    TEST_EQ((int)fbank_out.size(), T * D, "output size = T × D");

    printf("  Fbank output: [%d, %d] (%d floats)\n", T, D, T * D);

    printf("  [OK] Fbank basic passed\n");
}

// ============================================================
// Test 3: 数值有效性
// ============================================================
void test_fbank_values(const std::string& wav_path) {
    printf("\n--- Test 3: Fbank Values ---\n");

    funasr::FrontendConfig config;
    funasr::FbankProcessor fbank(config);

    std::vector<float> fbank_out;
    int T, D;
    fbank.process_wav(wav_path, fbank_out, T, D);

    // 检查无 NaN / Inf
    int nan_count = 0, inf_count = 0;
    float min_val = 1e30f, max_val = -1e30f;
    double sum = 0.0;

    for (int i = 0; i < T * D; i++) {
        float v = fbank_out[i];
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }

    TEST_EQ(nan_count, 0, "no NaN");
    TEST_EQ(inf_count, 0, "no Inf");

    float mean_val = static_cast<float>(sum / (T * D));

    printf("  Value range: [%.4f, %.4f]\n", min_val, max_val);
    printf("  Mean: %.4f\n", mean_val);

    // log-mel 特征的合理范围大约是 [-25, 20]
    TEST_ASSERT(min_val > -50.0f, "min > -50 (reasonable log-mel)");
    TEST_ASSERT(max_val < 30.0f,  "max < 30 (reasonable log-mel)");

    // 打印前几帧的前几个值（方便与旧代码对比）
    printf("\n  First frame (first 10 values):\n    ");
    for (int i = 0; i < std::min(10, D); i++) {
        printf("%.4f ", fbank_out[i]);
    }
    printf("\n");

    if (T > 1) {
        printf("  Second frame (first 10 values):\n    ");
        for (int i = 0; i < std::min(10, D); i++) {
            printf("%.4f ", fbank_out[D + i]);
        }
        printf("\n");
    }

    printf("  [OK] Fbank values passed\n");
}

// ============================================================
// Test 4: 用 Config 参数验证维度关系
// ============================================================
void test_fbank_config(const std::string& wav_path) {
    printf("\n--- Test 4: Config-Driven Dimensions ---\n");

    funasr::FrontendConfig config;
    // 验证默认配置下的派生值
    TEST_EQ(config.input_dim(), 560, "input_dim() = 560");
    TEST_EQ(config.sample_rate, 16000, "sample_rate = 16000");
    TEST_EQ(config.lfr_m, 7, "lfr_m = 7");
    TEST_EQ(config.lfr_n, 6, "lfr_n = 6");

    funasr::FbankProcessor fbank(config);

    std::vector<float> fbank_out;
    int T, D;
    fbank.process_wav(wav_path, fbank_out, T, D);

    // D 必须等于 config.input_dim()
    TEST_EQ(D, config.input_dim(), "D matches config.input_dim()");

    // 读取 WAV 计算预期帧数
    std::vector<float> audio;
    int sr;
    fbank.read_wav(wav_path, audio, sr);

    int frame_len   = config.sample_rate * config.frame_length / 1000;  // 400
    int frame_shift = config.sample_rate * config.frame_shift  / 1000;  // 160
    int raw_frames  = 1 + static_cast<int>((audio.size() - frame_len) / frame_shift);

    // LFR 后的帧数
    int expected_T = 0;
    for (int i = 0; i + config.lfr_m <= raw_frames; i += config.lfr_n) {
        expected_T++;
    }

    TEST_EQ(T, expected_T, "T matches LFR calculation");

    printf("  Raw frames: %d, LFR frames: %d\n", raw_frames, T);
    printf("  Frame len: %d, Frame shift: %d\n", frame_len, frame_shift);

    printf("  [OK] Config dimensions passed\n");
}

// ============================================================
// Test 5: 性能基准
// ============================================================
void test_fbank_performance(const std::string& wav_path) {
    printf("\n--- Test 5: Performance ---\n");

    funasr::FrontendConfig config;
    funasr::FbankProcessor fbank(config);

    std::vector<float> fbank_out;
    int T, D;

    // 预热
    fbank.process_wav(wav_path, fbank_out, T, D);

    // 计时
    const int N_RUNS = 10;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        fbank.process_wav(wav_path, fbank_out, T, D);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    printf("  %d runs: %ld ms total, %.1f ms avg\n",
           N_RUNS, total_ms, (float)total_ms / N_RUNS);

    // 读 WAV 获取音频时长
    std::vector<float> audio;
    int sr;
    fbank.read_wav(wav_path, audio, sr);
    float audio_ms = audio.size() * 1000.0f / sr;
    float rtf = (total_ms / (float)N_RUNS) / audio_ms;

    printf("  Audio: %.0f ms, RTF: %.4f\n", audio_ms, rtf);
    TEST_ASSERT(rtf < 1.0f, "RTF < 1.0 (real-time capable)");

    printf("  [OK] Performance passed\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <audio.wav>\n", argv[0]);
        printf("  Example: %s ../zh.wav\n", argv[0]);
        return 1;
    }

    const std::string wav_path = argv[1];

    printf("========================================\n");
    printf("  FunASR Fbank Test\n");
    printf("========================================\n");
    printf("WAV: %s\n", wav_path.c_str());

    test_wav_read(wav_path);
    test_fbank_basic(wav_path);
    test_fbank_values(wav_path);
    test_fbank_config(wav_path);
    test_fbank_performance(wav_path);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}