// funasr/test/test_adaptor.cpp
// 验证 Adaptor 前向计算 (接在 Encoder 后面)
//
// 构建:
//   cd build && cmake .. && make test_adaptor
// 运行:
//   ./test_adaptor ../FunAsr_q8.bin ../zh.wav
//
#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "compute/fbank.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
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
// Test 1: Encoder + Adaptor 端到端
// ============================================================
void test_encoder_adaptor(funasr::FunASRModel& model,
                          const std::vector<float>& fbank_data,
                          int T_fbank, int D_fbank) {
    printf("\n--- Test 1: Encoder + Adaptor ---\n");

    const size_t mem_size = 4ULL * 1024 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "alloc context");
    if (!ctx) return;

    // 输入 [560, T]
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_fbank, T_fbank);
    memcpy(input->data, fbank_data.data(), T_fbank * D_fbank * sizeof(float));

    // Encoder
    printf("  Running Encoder...\n");
    struct ggml_tensor* enc_out = funasr::encoder_forward(
        ctx, input, model.encoder, model.config.encoder
    );

    // Adaptor
    printf("  Running Adaptor...\n");
    struct ggml_tensor* adp_out = funasr::adaptor_forward(
        ctx, enc_out, model.adaptor, model.config.adaptor
    );

    // 执行
    printf("  Computing...\n");
    auto t_start = std::chrono::high_resolution_clock::now();
    funasr::run_graph(ctx, adp_out, 4);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    printf("  Compute: %ld ms\n", ms);

    // 验证输出 shape: [1024, T]
    TEST_EQ((int)adp_out->ne[0], model.config.adaptor.output_dim, "adaptor output dim = 1024");
    TEST_EQ((int)adp_out->ne[1], T_fbank, "adaptor output T preserved");

    printf("  Output: [%lld, %lld]\n",
           (long long)adp_out->ne[0], (long long)adp_out->ne[1]);

    // 检查数值
    float* out_data = (float*)adp_out->data;
    int total = static_cast<int>(adp_out->ne[0] * adp_out->ne[1]);
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

    // 值应该有界
    TEST_ASSERT(min_val > -200.0f, "min bounded");
    TEST_ASSERT(max_val < 200.0f,  "max bounded");

    // 打印前几个值
    printf("\n  First frame (first 10 of 1024):\n    ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", out_data[i]);
    }
    printf("\n");

    printf("  Last frame (first 10 of 1024):\n    ");
    float* last = out_data + (T_fbank - 1) * 1024;
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", last[i]);
    }
    printf("\n");

    ggml_free(ctx);
    printf("  [OK] Encoder + Adaptor passed\n");
}

// ============================================================
// Test 2: Adaptor 单独测试 (用 fake encoder 输出)
// 验证 adaptor 本身的维度变换正确性
// ============================================================
void test_adaptor_standalone(funasr::FunASRModel& model) {
    printf("\n--- Test 2: Adaptor Standalone ---\n");

    const size_t mem_size = 1024ULL * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, false };
    struct ggml_context* ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "alloc context");
    if (!ctx) return;

    // 构造 fake encoder 输出 [512, 10]
    const int T = 10;
    const int dim_in = model.config.encoder.output_size;  // 512
    struct ggml_tensor* fake_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_in, T);
    float* data = (float*)fake_input->data;
    for (int i = 0; i < dim_in * T; i++) {
        data[i] = (float)(i % 100) * 0.01f - 0.5f;  // 值在 [-0.5, 0.49]
    }

    // Adaptor forward
    struct ggml_tensor* output = funasr::adaptor_forward(
        ctx, fake_input, model.adaptor, model.config.adaptor
    );

    funasr::run_graph(ctx, output, 4);

    // 验证输出维度
    TEST_EQ((int)output->ne[0], model.config.adaptor.output_dim, "output dim = 1024");
    TEST_EQ((int)output->ne[1], T, "output T = 10");

    // 检查无 NaN
    float* out = (float*)output->data;
    bool has_nan = false;
    for (int i = 0; i < 1024 * T; i++) {
        if (std::isnan(out[i]) || std::isinf(out[i])) { has_nan = true; break; }
    }
    TEST_ASSERT(!has_nan, "no NaN/Inf");

    printf("  Fake input: [%d, %d], Output: [%lld, %lld]\n",
           dim_in, T, (long long)output->ne[0], (long long)output->ne[1]);

    ggml_free(ctx);
    printf("  [OK] Adaptor standalone passed\n");
}

// ============================================================
// Test 3: 维度链验证
// 确认 encoder → adaptor 的维度衔接正确
// ============================================================
void test_dimension_chain(funasr::FunASRModel& model) {
    printf("\n--- Test 3: Dimension Chain ---\n");

    auto& enc_cfg = model.config.encoder;
    auto& adp_cfg = model.config.adaptor;
    auto& llm_cfg = model.config.llm;

    // encoder output = adaptor input
    TEST_EQ(enc_cfg.output_size, adp_cfg.input_dim,
            "encoder.output = adaptor.input (512)");

    // adaptor output = LLM embedding dim
    TEST_EQ(adp_cfg.output_dim, llm_cfg.embedding_length,
            "adaptor.output = llm.embedding (1024)");

    // adaptor linear1: [512, 2048]
    TEST_EQ((int)model.adaptor.linear1_w->ne[0], adp_cfg.input_dim,
            "linear1 input = 512");
    TEST_EQ((int)model.adaptor.linear1_w->ne[1], adp_cfg.hidden_dim,
            "linear1 output = 2048");

    // adaptor linear2: [2048, 1024]
    TEST_EQ((int)model.adaptor.linear2_w->ne[0], adp_cfg.hidden_dim,
            "linear2 input = 2048");
    TEST_EQ((int)model.adaptor.linear2_w->ne[1], adp_cfg.output_dim,
            "linear2 output = 1024");

    // block FFN: 中间维度 256
    TEST_EQ((int)model.adaptor.blocks[0].ffn_w1->ne[1], adp_cfg.ffn_dim,
            "block FFN mid = 256");

    printf("  512 → [linear1] → 2048 → [linear2] → 1024 → [2 blocks] → 1024\n");
    printf("  [OK] Dimension chain passed\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];

    printf("========================================\n");
    printf("  FunASR Adaptor Forward Test\n");
    printf("========================================\n");

    // 加载模型
    printf("\nLoading model...\n");
    funasr::GGUFReader reader;
    if (!reader.open(model_path)) return 1;
    funasr::FunASRModel model;
    if (!funasr::ModelLoader::load(reader, model)) return 1;

    // Fbank
    printf("\nExtracting fbank...\n");
    funasr::FbankProcessor fbank(model.config.frontend);
    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process_wav(wav_path, fbank_data, T, D)) return 1;
    printf("Fbank: [%d, %d]\n", T, D);

    // 测试
    test_dimension_chain(model);
    test_adaptor_standalone(model);
    test_encoder_adaptor(model, fbank_data, T, D);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}