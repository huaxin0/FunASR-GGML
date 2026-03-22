// 验证第一步：Config 读取 + GGUFReader 生命周期
//
// 构建 (CMake):
//   cd build && cmake .. && make test_config
//
// 运行:
//   ./test_config ../FunAsr_q8.bin
//
#include "core/config.hpp"
#include "core/gguf_reader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>

// ============================================================
// 简易测试框架
// ============================================================
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, msg) do { \
    if (condition) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
    } \
} while(0)

#define TEST_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
               name, (int)(expected), (int)(actual), __LINE__); \
    } \
} while(0)

#define TEST_FEQ(actual, expected, name) do { \
    float diff = (actual) - (expected); \
    if (diff < 0) diff = -diff; \
    if (diff < 0.01f) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  [FAIL] %s: expected %.1f, got %.1f (line %d)\n", \
               name, (float)(expected), (float)(actual), __LINE__); \
    } \
} while(0)

// ============================================================
// Test 1: GGUFReader 基础功能
// ============================================================
void test_reader_basics(const std::string& model_path) {
    printf("\n--- Test 1: GGUFReader Basics ---\n");

    funasr::GGUFReader reader;

    // 打开前应该是关闭状态
    TEST_ASSERT(!reader.is_open(), "reader should be closed initially");
    TEST_EQ(reader.tensor_count(), 0, "tensor_count before open");

    // 打开文件
    bool ok = reader.open(model_path);
    TEST_ASSERT(ok, "open() should succeed");
    TEST_ASSERT(reader.is_open(), "reader should be open after open()");
    TEST_EQ(reader.tensor_count(), 1541, "tensor_count");
    TEST_ASSERT(!reader.has_errors(), "no errors after open");

    // 能找到已知的 tensor
    auto* t = reader.get_tensor("llm.model.embed_tokens.weight");
    TEST_ASSERT(t != nullptr, "embed_tokens should exist");

    // 找不到不存在的 tensor
    auto* t2 = reader.get_tensor("does_not_exist");
    TEST_ASSERT(t2 == nullptr, "nonexistent tensor should be null");

    // require_tensor 记录缺失
    reader.clear_errors();
    auto* t3 = reader.require_tensor("audio_encoder.encoders0.0.self_attn.linear_q.weight");
    TEST_ASSERT(t3 != nullptr, "encoder0 linear_q should exist");
    TEST_ASSERT(!reader.has_errors(), "no errors for valid tensor");

    auto* t4 = reader.require_tensor("this.does.not.exist");
    TEST_ASSERT(t4 == nullptr, "missing tensor returns null");
    TEST_ASSERT(reader.has_errors(), "error recorded for missing tensor");
    TEST_EQ((int)reader.missing_tensors().size(), 1, "one missing tensor");

    printf("  [OK] GGUFReader basics passed\n");
}

// ============================================================
// Test 2: Config 从 GGUF 读取
// 对照 funasr_gguf.md 里的精确值
// ============================================================
void test_config_loading(const std::string& model_path) {
    printf("\n--- Test 2: Config Loading ---\n");

    funasr::GGUFReader reader;
    bool ok = reader.open(model_path);
    TEST_ASSERT(ok, "open for config test");

    funasr::ModelConfig config;
    ok = config.load_from_gguf(reader.gguf_ctx());
    TEST_ASSERT(ok, "load_from_gguf should succeed");

    // ===== Frontend 验证 =====
    // GGUF md 文档: frontend.sample_rate = 16000, num_mels = 80, ...
    TEST_EQ(config.frontend.sample_rate,  16000, "frontend.sample_rate");
    TEST_EQ(config.frontend.num_mels,     80,    "frontend.num_mels");
    TEST_EQ(config.frontend.frame_length, 25,    "frontend.frame_length");
    TEST_EQ(config.frontend.frame_shift,  10,    "frontend.frame_shift");
    TEST_EQ(config.frontend.lfr_m,        7,     "frontend.lfr_m");
    TEST_EQ(config.frontend.lfr_n,        6,     "frontend.lfr_n");
    TEST_EQ(config.frontend.input_dim(),  560,   "frontend.input_dim (derived)");

    // ===== Encoder 验证 =====
    // GGUF md: encoder.output_size = 512, attention_heads = 4, ...
    TEST_EQ(config.encoder.output_size,      512,  "encoder.output_size");
    TEST_EQ(config.encoder.attention_heads,  4,    "encoder.attention_heads");
    TEST_EQ(config.encoder.linear_units,     2048, "encoder.linear_units");
    TEST_EQ(config.encoder.num_blocks,       50,   "encoder.num_blocks");
    TEST_EQ(config.encoder.tp_blocks,        20,   "encoder.tp_blocks");
    TEST_EQ(config.encoder.kernel_size,      11,   "encoder.kernel_size");
    TEST_EQ(config.encoder.sanm_shift,       0,    "encoder.sanm_shift");
    // 派生值
    TEST_EQ(config.encoder.num_main_blocks(), 49,  "encoder.num_main_blocks (derived)");
    TEST_EQ(config.encoder.head_dim(),        128, "encoder.head_dim (derived)");
    TEST_EQ(config.encoder.fsmn_pad(),        5,   "encoder.fsmn_pad (derived)");

    // ===== LLM 验证 =====
    // GGUF md: FunAsr.block_count = 28, embedding_length = 1024, ...
    TEST_EQ(config.llm.block_count,         28,      "llm.block_count");
    TEST_EQ(config.llm.embedding_length,    1024,    "llm.embedding_length");
    TEST_EQ(config.llm.context_length,      40960,   "llm.context_length");
    TEST_EQ(config.llm.feed_forward_length, 3072,    "llm.feed_forward_length");
    TEST_EQ(config.llm.head_count,          16,      "llm.head_count");
    TEST_EQ(config.llm.head_count_kv,       8,       "llm.head_count_kv");
    TEST_EQ(config.llm.key_length,          128,     "llm.key_length");
    TEST_EQ(config.llm.value_length,        128,     "llm.value_length");
    TEST_FEQ(config.llm.rope_freq_base,     1000000.0f, "llm.rope_freq_base");
    // 派生值
    TEST_EQ(config.llm.head_dim(),    128,  "llm.head_dim (derived)");
    TEST_EQ(config.llm.q_dim(),       2048, "llm.q_dim (derived)");
    TEST_EQ(config.llm.kv_dim(),      1024, "llm.kv_dim (derived)");
    TEST_EQ(config.llm.gqa_groups(),  2,    "llm.gqa_groups (derived)");

    // ===== Adaptor 验证 (从 encoder/LLM 推导) =====
    TEST_EQ(config.adaptor.input_dim,        512,  "adaptor.input_dim");
    TEST_EQ(config.adaptor.output_dim,       1024, "adaptor.output_dim");
    TEST_EQ(config.adaptor.hidden_dim,       2048, "adaptor.hidden_dim");
    TEST_EQ(config.adaptor.num_blocks,       2,    "adaptor.num_blocks");
    TEST_EQ(config.adaptor.attention_heads,  8,    "adaptor.attention_heads");
    TEST_EQ(config.adaptor.ffn_dim,          256,  "adaptor.ffn_dim");

    // 打印完整配置（方便人工对照）
    config.print();

    printf("  [OK] Config loading passed\n");
}

// ============================================================
// Test 3: GGUFReader 移动语义
// 验证 RAII 和所有权转移的正确性
// ============================================================
void test_reader_move(const std::string& model_path) {
    printf("\n--- Test 3: GGUFReader Move Semantics ---\n");

    struct ggml_tensor* tensor_ptr = nullptr;

    {
        funasr::GGUFReader reader1;
        reader1.open(model_path);

        // 记住一个 tensor 指针
        tensor_ptr = reader1.get_tensor("llm.model.norm.weight");
        TEST_ASSERT(tensor_ptr != nullptr, "tensor found before move");

        // 移动到 reader2
        funasr::GGUFReader reader2 = std::move(reader1);

        // reader1 应该已清空
        TEST_ASSERT(!reader1.is_open(), "reader1 closed after move");
        TEST_EQ(reader1.tensor_count(), 0, "reader1 tensor_count after move");

        // reader2 应该接管了一切
        TEST_ASSERT(reader2.is_open(), "reader2 open after move");
        TEST_EQ(reader2.tensor_count(), 1541, "reader2 tensor_count after move");

        // 通过 reader2 能找到同一个 tensor
        auto* t2 = reader2.get_tensor("llm.model.norm.weight");
        TEST_ASSERT(t2 != nullptr, "tensor accessible through reader2");
        TEST_ASSERT(t2 == tensor_ptr, "same tensor pointer after move");

        // reader2 析构时释放资源
    }

    // 离开作用域后，tensor_ptr 指向的内存已释放
    // （我们不去访问它，只验证生命周期管理的正确性）
    printf("  [OK] Move semantics passed\n");
}

// ============================================================
// Test 4: 验证关键 tensor 存在且 shape 正确
// 抽查几个代表性的 tensor，确认能正确绑定
// ============================================================
void test_tensor_spot_check(const std::string& model_path) {
    printf("\n--- Test 4: Tensor Spot Check ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);

    // Encoder0: Q weight 应该是 [560, 512]
    auto* enc0_q = reader.get_tensor("audio_encoder.encoders0.0.self_attn.linear_q.weight");
    TEST_ASSERT(enc0_q != nullptr, "encoder0 linear_q exists");
    if (enc0_q) {
        TEST_EQ((int)enc0_q->ne[0], 560, "encoder0 linear_q ne[0]");
        TEST_EQ((int)enc0_q->ne[1], 512, "encoder0 linear_q ne[1]");
    }

    // Encoder0: norm1 应该是 [560]（特殊维度）
    auto* enc0_n1 = reader.get_tensor("audio_encoder.encoders0.0.norm1.weight");
    TEST_ASSERT(enc0_n1 != nullptr, "encoder0 norm1 exists");
    if (enc0_n1) {
        TEST_EQ((int)enc0_n1->ne[0], 560, "encoder0 norm1 ne[0]");
    }

    // Main encoder[0]: Q weight 应该是 [512, 512]
    auto* enc_q = reader.get_tensor("audio_encoder.encoders.0.self_attn.linear_q.weight");
    TEST_ASSERT(enc_q != nullptr, "encoders.0 linear_q exists");
    if (enc_q) {
        TEST_EQ((int)enc_q->ne[0], 512, "encoders.0 linear_q ne[0]");
        TEST_EQ((int)enc_q->ne[1], 512, "encoders.0 linear_q ne[1]");
    }

    // Adaptor: linear2 应该是 [2048, 1024]
    auto* adp_l2 = reader.get_tensor("audio_adaptor.linear2.weight");
    TEST_ASSERT(adp_l2 != nullptr, "adaptor linear2 exists");
    if (adp_l2) {
        TEST_EQ((int)adp_l2->ne[0], 2048, "adaptor linear2 ne[0]");
        TEST_EQ((int)adp_l2->ne[1], 1024, "adaptor linear2 ne[1]");
    }

    // Adaptor block FFN: w1 应该是 [1024, 256]（不是 2048!）
    auto* adp_ffn = reader.get_tensor("audio_adaptor.blocks.0.feed_forward.w_1.weight");
    TEST_ASSERT(adp_ffn != nullptr, "adaptor block0 ffn_w1 exists");
    if (adp_ffn) {
        TEST_EQ((int)adp_ffn->ne[0], 1024, "adaptor block0 ffn_w1 ne[0]");
        TEST_EQ((int)adp_ffn->ne[1], 256,  "adaptor block0 ffn_w1 ne[1]");
    }

    // LLM: q_proj 应该是 [1024, 2048]
    auto* llm_q = reader.get_tensor("llm.model.layers.0.self_attn.q_proj.weight");
    TEST_ASSERT(llm_q != nullptr, "llm layer0 q_proj exists");
    if (llm_q) {
        TEST_EQ((int)llm_q->ne[0], 1024, "llm q_proj ne[0]");
        TEST_EQ((int)llm_q->ne[1], 2048, "llm q_proj ne[1]");
    }

    // LLM: embed_tokens 应该是 [1024, 151936]
    auto* embed = reader.get_tensor("llm.model.embed_tokens.weight");
    TEST_ASSERT(embed != nullptr, "embed_tokens exists");
    if (embed) {
        TEST_EQ((int)embed->ne[0], 1024,   "embed_tokens ne[0]");
        TEST_EQ((int)embed->ne[1], 151936, "embed_tokens ne[1]");
    }

    // 最后一个 TP encoder layer (19)
    auto* tp19 = reader.get_tensor("audio_encoder.tp_encoders.19.self_attn.linear_q.weight");
    TEST_ASSERT(tp19 != nullptr, "tp_encoders.19 exists");

    // 不存在的 layer (encoder 50, 超出范围)
    auto* bad = reader.get_tensor("audio_encoder.encoders.49.self_attn.linear_q.weight");
    TEST_ASSERT(bad == nullptr, "encoders.49 should NOT exist (only 0..48)");

    printf("  [OK] Tensor spot check passed\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        printf("  Example: %s ../../FunAsr_q8.bin\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    printf("========================================\n");
    printf("  FunASR Config & Reader Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());

    test_reader_basics(model_path);
    test_config_loading(model_path);
    test_reader_move(model_path);
    test_tensor_spot_check(model_path);

    // ===== 汇总 =====
    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n",
           tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}