// 验证 Encoder 前向计算
//
// 构建:
//   cd build && cmake .. && make test_encoder
// 运行:
//   ./test_encoder ../FunAsr_q8.bin ../zh.wav
#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "compute/fbank.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/graph_runner.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

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
// Test 1: 单层 Encoder Forward (encoder0)
// 用小 context 快速验证基本正确性
// ============================================================
void test_single_layer(funasr::FunASRModel& model, const std::vector<float>& fbank_data,
                       int T_fbank, int D_fbank) {
    printf("\n--- Test 1: Single Layer (encoder0) ---\n");

    // 分配计算 context (单层足够小)
    const size_t mem_size = 512ULL * 1024 * 1024;  // 512MB
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "alloc compute context");
    if (!ctx) return;

    // 创建输入 tensor [560, T]
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_fbank, T_fbank);
    memcpy(input->data, fbank_data.data(), T_fbank * D_fbank * sizeof(float));

    // 缩放 (encoder forward 第一步)
    float scale = std::sqrt(static_cast<float>(model.config.encoder.output_size));
    struct ggml_tensor* scaled = ggml_scale(ctx, input, scale);

    // 跑 encoder0 单层
    struct ggml_tensor* output = funasr::encoder_layer_forward(
        ctx, scaled, model.encoder.encoder0, model.config.encoder, false
    );

    // 执行
    funasr::run_graph(ctx, output, 4);

    // 检查输出
    TEST_EQ((int)output->ne[0], model.config.encoder.output_size, "encoder0 output dim = 512");
    TEST_EQ((int)output->ne[1], T_fbank, "encoder0 output T preserved");

    // 检查无 NaN
    float* out_data = (float*)output->data;
    int total = static_cast<int>(output->ne[0] * output->ne[1]);
    bool has_nan = false;
    for (int i = 0; i < total; i++) {
        if (std::isnan(out_data[i]) || std::isinf(out_data[i])) {
            has_nan = true;
            break;
        }
    }
    TEST_ASSERT(!has_nan, "encoder0 output no NaN/Inf");

    // 打印前几个值
    printf("  Output shape: [%lld, %lld]\n",
           (long long)output->ne[0], (long long)output->ne[1]);
    printf("  First 5 values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", out_data[i]);
    }
    printf("\n");

    ggml_free(ctx);
    printf("  [OK] Single layer passed\n");
}

// ============================================================
// Test 2: 完整 Encoder Forward (70 层)
// ============================================================
void test_full_encoder(funasr::FunASRModel& model, const std::vector<float>& fbank_data,
                       int T_fbank, int D_fbank) {
    printf("\n--- Test 2: Full Encoder (70 layers) ---\n");

    // 70 层需要较大 context
    const size_t mem_size = 4ULL * 1024 * 1024 * 1024;  // 4GB
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "alloc 4GB compute context");
    if (!ctx) return;

    // 创建输入 tensor [560, T]
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_fbank, T_fbank);
    memcpy(input->data, fbank_data.data(), T_fbank * D_fbank * sizeof(float));

    printf("  Input:  [%d, %d]\n", D_fbank, T_fbank);

    // 构建计算图
    auto t_build_start = std::chrono::high_resolution_clock::now();

    struct ggml_tensor* output = funasr::encoder_forward(
        ctx, input, model.encoder, model.config.encoder
    );

    auto t_build_end = std::chrono::high_resolution_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_build_end - t_build_start).count();
    printf("  Graph build: %ld ms\n", build_ms);

    // 执行计算图
    printf("  Computing (this may take a while)...\n");
    auto t_compute_start = std::chrono::high_resolution_clock::now();

    funasr::run_graph(ctx, output, 4);

    auto t_compute_end = std::chrono::high_resolution_clock::now();
    auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_compute_end - t_compute_start).count();
    printf("  Compute: %ld ms\n", compute_ms);

    // 验证输出 shape
    TEST_EQ((int)output->ne[0], model.config.encoder.output_size, "encoder output dim = 512");
    TEST_EQ((int)output->ne[1], T_fbank, "encoder output T = input T");

    printf("  Output: [%lld, %lld]\n",
           (long long)output->ne[0], (long long)output->ne[1]);

    // 检查无 NaN/Inf
    float* out_data = (float*)output->data;
    int total = static_cast<int>(output->ne[0] * output->ne[1]);
    int nan_count = 0, inf_count = 0;
    float min_val = 1e30f, max_val = -1e30f;
    double sum = 0.0;

    for (int i = 0; i < total; i++) {
        float v = out_data[i];
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }

    TEST_EQ(nan_count, 0, "no NaN");
    TEST_EQ(inf_count, 0, "no Inf");

    float mean_val = static_cast<float>(sum / total);
    printf("  Value range: [%.4f, %.4f], mean: %.4f\n", min_val, max_val, mean_val);

    // 值应该在合理范围（经过 70 层 + LayerNorm 后应该是有界的）
    TEST_ASSERT(min_val > -100.0f, "min > -100 (bounded)");
    TEST_ASSERT(max_val < 100.0f,  "max < 100 (bounded)");

    // 打印前几个值（方便与旧代码对比）
    printf("\n  First frame (first 10 of 512 values):\n    ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", out_data[i]);
    }
    printf("\n");

    // 打印最后一帧的前几个值
    printf("  Last frame (first 10 of 512 values):\n    ");
    float* last_frame = out_data + (T_fbank - 1) * 512;
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", last_frame[i]);
    }
    printf("\n");

    ggml_free(ctx);
    printf("  [OK] Full encoder passed\n");
}

// ============================================================
// Test 3: 维度一致性检查
// 验证 Config 驱动的参数是否正确传递
// ============================================================
void test_dimensions(funasr::FunASRModel& model) {
    printf("\n--- Test 3: Dimension Consistency ---\n");

    auto& cfg = model.config.encoder;

    // encoder0 的 Q weight: [input_dim(560), output_size(512)]
    TEST_EQ((int)model.encoder.encoder0.attn_q_w->ne[0],
            model.config.frontend.input_dim(), "encoder0 Q input = 560");
    TEST_EQ((int)model.encoder.encoder0.attn_q_w->ne[1],
            cfg.output_size, "encoder0 Q output = 512");

    // encoder0 的 norm1: [560]
    TEST_EQ((int)model.encoder.encoder0.norm1_w->ne[0],
            model.config.frontend.input_dim(), "encoder0 norm1 = 560");

    // encoder0 的 norm2: [512]
    TEST_EQ((int)model.encoder.encoder0.norm2_w->ne[0],
            cfg.output_size, "encoder0 norm2 = 512");

    // main encoder 的 Q weight: [512, 512]
    TEST_EQ((int)model.encoder.main_layers[0].attn_q_w->ne[0],
            cfg.output_size, "main[0] Q input = 512");

    // FSMN weight: [kernel_size, 1, 512]
    TEST_EQ((int)model.encoder.encoder0.fsmn_w->ne[0],
            cfg.kernel_size, "fsmn K = kernel_size");

    // 派生值验证
    TEST_EQ(cfg.num_main_blocks(), (int)model.encoder.main_layers.size(),
            "num_main_blocks matches");
    TEST_EQ(cfg.tp_blocks, (int)model.encoder.tp_layers.size(),
            "tp_blocks matches");

    printf("  [OK] Dimensions consistent\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav>\n", argv[0]);
        printf("  Example: %s ../FunAsr_q8.bin ../zh.wav\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];

    printf("========================================\n");
    printf("  FunASR Encoder Forward Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());
    printf("WAV:   %s\n", wav_path.c_str());

    // ===== Step 1: 加载模型 =====
    printf("\nLoading model...\n");
    funasr::GGUFReader reader;
    if (!reader.open(model_path)) return 1;

    funasr::FunASRModel model;
    if (!funasr::ModelLoader::load(reader, model)) return 1;

    // ===== Step 2: Fbank 特征提取 =====
    printf("\nExtracting fbank...\n");
    funasr::FbankProcessor fbank(model.config.frontend);
    std::vector<float> fbank_data;
    int T_fbank, D_fbank;
    if (!fbank.process_wav(wav_path, fbank_data, T_fbank, D_fbank)) {
        printf("Fbank failed\n");
        return 1;
    }
    printf("Fbank: [%d, %d]\n", T_fbank, D_fbank);

    // ===== Step 3: 运行测试 =====
    test_dimensions(model);
    test_single_layer(model, fbank_data, T_fbank, D_fbank);
    test_full_encoder(model, fbank_data, T_fbank, D_fbank);

    // ===== 汇总 =====
    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}