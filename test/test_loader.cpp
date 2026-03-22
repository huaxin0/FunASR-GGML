// 验证第二步：完整模型加载 + 逐模块权重验证
//
// 构建:
//   cd build && cmake .. && make test_loader
// 运行:
//   ./test_loader ../FunAsr_q8.bin
//
#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

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
// Test 1: 一站式加载
// ============================================================
void test_full_load(const std::string& model_path) {
    printf("\n--- Test 1: Full Model Load ---\n");

    funasr::GGUFReader reader;
    TEST_ASSERT(reader.open(model_path), "open GGUF file");

    funasr::FunASRModel model;
    bool ok = funasr::ModelLoader::load(reader, model);
    TEST_ASSERT(ok, "ModelLoader::load succeeds");
    TEST_ASSERT(model.is_valid(), "model.is_valid()");

    // 验证 tensor 总数: 1194 (encoder) + 36 (adaptor) + 311 (llm) = 1541
    TEST_EQ(model.encoder.tensor_count(), 1194, "encoder tensor count");
    TEST_EQ(model.adaptor.tensor_count(), 36,   "adaptor tensor count");
    TEST_EQ(model.llm.tensor_count(),     311,  "llm tensor count");
    TEST_EQ(model.tensor_count(),         1541, "total tensor count");

    printf("  [OK] Full load passed\n");
}

// ============================================================
// Test 2: Encoder 详细验证
// ============================================================
void test_encoder_detail(const std::string& model_path) {
    printf("\n--- Test 2: Encoder Detail ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);

    funasr::ModelConfig config;
    funasr::ModelLoader::load_config(reader, config);

    funasr::EncoderWeights enc;
    bool ok = funasr::ModelLoader::load_encoder(reader, config.encoder, enc);
    TEST_ASSERT(ok, "encoder load succeeds");
    TEST_ASSERT(enc.is_valid(), "encoder.is_valid()");

    // encoder0: 特殊维度 (input 560)
    TEST_ASSERT(enc.encoder0.is_valid(), "encoder0 valid");
    TEST_EQ((int)enc.encoder0.attn_q_w->ne[0], 560, "encoder0 q_w ne[0] = 560");
    TEST_EQ((int)enc.encoder0.attn_q_w->ne[1], 512, "encoder0 q_w ne[1] = 512");
    TEST_EQ((int)enc.encoder0.norm1_w->ne[0],  560, "encoder0 norm1 = 560");
    TEST_EQ((int)enc.encoder0.norm2_w->ne[0],  512, "encoder0 norm2 = 512");
    // FSMN: [11, 1, 512]
    TEST_EQ((int)enc.encoder0.fsmn_w->ne[0], 11,  "encoder0 fsmn ne[0]");
    TEST_EQ((int)enc.encoder0.fsmn_w->ne[2], 512, "encoder0 fsmn ne[2]");

    // main encoders: 49 层, 标准维度 512
    TEST_EQ((int)enc.main_layers.size(), 49, "49 main layers");
    TEST_EQ((int)enc.main_layers[0].attn_q_w->ne[0], 512, "main[0] q_w ne[0]");
    TEST_EQ((int)enc.main_layers[0].attn_q_w->ne[1], 512, "main[0] q_w ne[1]");
    TEST_EQ((int)enc.main_layers[0].ffn_w1->ne[0], 512,  "main[0] ffn_w1 ne[0]");
    TEST_EQ((int)enc.main_layers[0].ffn_w1->ne[1], 2048, "main[0] ffn_w1 ne[1]");
    // 最后一层
    TEST_ASSERT(enc.main_layers[48].is_valid(), "main[48] valid");

    // tp encoders: 20 层
    TEST_EQ((int)enc.tp_layers.size(), 20, "20 tp layers");
    TEST_ASSERT(enc.tp_layers[0].is_valid(),  "tp[0] valid");
    TEST_ASSERT(enc.tp_layers[19].is_valid(), "tp[19] valid");

    // 顶层 norm: [512]
    TEST_EQ((int)enc.after_norm_w->ne[0], 512, "after_norm [512]");
    TEST_EQ((int)enc.tp_norm_w->ne[0],    512, "tp_norm [512]");

    // 打印代表性 layer
    enc.encoder0.print_shapes("encoder0");
    enc.main_layers[0].print_shapes("main_encoder[0]");

    printf("  [OK] Encoder detail passed\n");
}

// ============================================================
// Test 3: Adaptor 详细验证
// ============================================================
void test_adaptor_detail(const std::string& model_path) {
    printf("\n--- Test 3: Adaptor Detail ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);

    funasr::ModelConfig config;
    funasr::ModelLoader::load_config(reader, config);

    funasr::AdaptorWeights adp;
    bool ok = funasr::ModelLoader::load_adaptor(reader, config.adaptor, adp);
    TEST_ASSERT(ok, "adaptor load succeeds");
    TEST_ASSERT(adp.is_valid(), "adaptor.is_valid()");

    // linear1: [512, 2048]
    TEST_EQ((int)adp.linear1_w->ne[0], 512,  "linear1 ne[0]");
    TEST_EQ((int)adp.linear1_w->ne[1], 2048, "linear1 ne[1]");

    // linear2: [2048, 1024]
    TEST_EQ((int)adp.linear2_w->ne[0], 2048, "linear2 ne[0]");
    TEST_EQ((int)adp.linear2_w->ne[1], 1024, "linear2 ne[1]");

    // 2 blocks
    TEST_EQ((int)adp.blocks.size(), 2, "2 adaptor blocks");

    // block[0] attention: [1024, 1024]
    TEST_EQ((int)adp.blocks[0].attn_q_w->ne[0], 1024, "block0 q ne[0]");
    TEST_EQ((int)adp.blocks[0].attn_q_w->ne[1], 1024, "block0 q ne[1]");

    // block[0] FFN: w1 [1024, 256], w2 [256, 1024]
    TEST_EQ((int)adp.blocks[0].ffn_w1->ne[0], 1024, "block0 ffn_w1 ne[0]");
    TEST_EQ((int)adp.blocks[0].ffn_w1->ne[1], 256,  "block0 ffn_w1 ne[1]");
    TEST_EQ((int)adp.blocks[0].ffn_w2->ne[0], 256,  "block0 ffn_w2 ne[0]");
    TEST_EQ((int)adp.blocks[0].ffn_w2->ne[1], 1024, "block0 ffn_w2 ne[1]");

    // block[0] norm: [1024]
    TEST_EQ((int)adp.blocks[0].norm1_w->ne[0], 1024, "block0 norm1");

    // block[1] 也应该有效
    TEST_ASSERT(adp.blocks[1].is_valid(), "block[1] valid");

    printf("  [OK] Adaptor detail passed\n");
}

// ============================================================
// Test 4: LLM 详细验证
// ============================================================
void test_llm_detail(const std::string& model_path) {
    printf("\n--- Test 4: LLM Detail ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);

    funasr::ModelConfig config;
    funasr::ModelLoader::load_config(reader, config);

    funasr::LLMWeights llm;
    bool ok = funasr::ModelLoader::load_llm(reader, config.llm, llm);
    TEST_ASSERT(ok, "llm load succeeds");
    TEST_ASSERT(llm.is_valid(), "llm.is_valid()");

    // embed_tokens: [1024, 151936]
    TEST_EQ((int)llm.embed_tokens->ne[0], 1024,   "embed ne[0]");
    TEST_EQ((int)llm.embed_tokens->ne[1], 151936, "embed ne[1]");

    // 28 layers
    TEST_EQ((int)llm.layers.size(), 28, "28 llm layers");

    // layer[0] shapes
    auto& L0 = llm.layers[0];
    TEST_ASSERT(L0.is_valid(), "layer[0] valid");
    TEST_EQ((int)L0.q_proj_w->ne[0], 1024, "L0 q_proj ne[0]");
    TEST_EQ((int)L0.q_proj_w->ne[1], 2048, "L0 q_proj ne[1]");
    TEST_EQ((int)L0.k_proj_w->ne[0], 1024, "L0 k_proj ne[0]");
    TEST_EQ((int)L0.k_proj_w->ne[1], 1024, "L0 k_proj ne[1]");
    TEST_EQ((int)L0.v_proj_w->ne[0], 1024, "L0 v_proj ne[0]");
    TEST_EQ((int)L0.v_proj_w->ne[1], 1024, "L0 v_proj ne[1]");
    TEST_EQ((int)L0.o_proj_w->ne[0], 2048, "L0 o_proj ne[0]");
    TEST_EQ((int)L0.o_proj_w->ne[1], 1024, "L0 o_proj ne[1]");
    TEST_EQ((int)L0.q_norm_w->ne[0], 128,  "L0 q_norm");
    TEST_EQ((int)L0.k_norm_w->ne[0], 128,  "L0 k_norm");
    TEST_EQ((int)L0.gate_proj_w->ne[0], 1024, "L0 gate ne[0]");
    TEST_EQ((int)L0.gate_proj_w->ne[1], 3072, "L0 gate ne[1]");
    TEST_EQ((int)L0.up_proj_w->ne[0],   1024, "L0 up ne[0]");
    TEST_EQ((int)L0.up_proj_w->ne[1],   3072, "L0 up ne[1]");
    TEST_EQ((int)L0.down_proj_w->ne[0], 3072, "L0 down ne[0]");
    TEST_EQ((int)L0.down_proj_w->ne[1], 1024, "L0 down ne[1]");
    TEST_EQ((int)L0.input_norm_w->ne[0],     1024, "L0 input_norm");
    TEST_EQ((int)L0.post_attn_norm_w->ne[0], 1024, "L0 post_attn_norm");

    // 最后一层也应该有效
    TEST_ASSERT(llm.layers[27].is_valid(), "layer[27] valid");

    // model_norm: [1024]
    TEST_EQ((int)llm.model_norm_w->ne[0], 1024, "model_norm");

    // lm_head: [1024, 151936]
    TEST_EQ((int)llm.lm_head_w->ne[0], 1024,   "lm_head ne[0]");
    TEST_EQ((int)llm.lm_head_w->ne[1], 151936, "lm_head ne[1]");

    // 打印代表性 layer
    L0.print_shapes("llm_layer[0]");

    printf("  [OK] LLM detail passed\n");
}

// ============================================================
// Test 5: 分步加载 vs 一站式加载一致性
// ============================================================
void test_consistency(const std::string& model_path) {
    printf("\n--- Test 5: Consistency Check ---\n");

    // 一站式
    funasr::GGUFReader reader1;
    reader1.open(model_path);
    funasr::FunASRModel model1;
    funasr::ModelLoader::load(reader1, model1);

    // 分步
    funasr::GGUFReader reader2;
    reader2.open(model_path);
    funasr::ModelConfig config;
    funasr::ModelLoader::load_config(reader2, config);

    funasr::EncoderWeights enc;
    funasr::ModelLoader::load_encoder(reader2, config.encoder, enc);
    funasr::AdaptorWeights adp;
    funasr::ModelLoader::load_adaptor(reader2, config.adaptor, adp);
    funasr::LLMWeights llm;
    funasr::ModelLoader::load_llm(reader2, config.llm, llm);

    // 两种方式的层数应该一致
    TEST_EQ((int)model1.encoder.main_layers.size(), (int)enc.main_layers.size(),
            "encoder main layers count match");
    TEST_EQ((int)model1.adaptor.blocks.size(), (int)adp.blocks.size(),
            "adaptor blocks count match");
    TEST_EQ((int)model1.llm.layers.size(), (int)llm.layers.size(),
            "llm layers count match");

    // 两种方式绑定的 tensor 指针应该相同（指向同一块 GGUF 数据）
    // 但因为是不同的 reader 打开了不同的内存，指针值不同
    // 我们只验证 shape 一致
    TEST_EQ((int)model1.encoder.encoder0.attn_q_w->ne[0],
            (int)enc.encoder0.attn_q_w->ne[0],
            "encoder0 q_w shape match");
    TEST_EQ((int)model1.llm.layers[0].q_proj_w->ne[1],
            (int)llm.layers[0].q_proj_w->ne[1],
            "llm L0 q_proj shape match");

    printf("  [OK] Consistency check passed\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    printf("========================================\n");
    printf("  FunASR Model Loader Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());

    test_full_load(model_path);
    test_encoder_detail(model_path);
    test_adaptor_detail(model_path);
    test_llm_detail(model_path);
    test_consistency(model_path);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}