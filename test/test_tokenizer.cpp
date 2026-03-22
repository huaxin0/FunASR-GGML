// 验证 Tokenizer：加载、encode、decode、round-trip、特殊 token
//
// 构建:
//   cd build && cmake .. && make test_tokenizer
// 运行:
//   ./test_tokenizer ../FunAsr_q8.bin
//
#include "core/gguf_reader.hpp"
#include "model/tokenizer.hpp"
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

#define TEST_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s: expected %d, got %d (line %d)\n", \
           name, (int)(expected), (int)(actual), __LINE__); } \
} while(0)

#define TEST_STR_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { tests_passed++; } \
    else { tests_failed++; printf("  [FAIL] %s: expected '%s', got '%s' (line %d)\n", \
           name, (expected).c_str(), (actual).c_str(), __LINE__); } \
} while(0)

// ============================================================
// Test 1: 基础加载
// ============================================================
void test_load(const std::string& model_path) {
    printf("\n--- Test 1: Tokenizer Load ---\n");

    funasr::GGUFReader reader;
    TEST_ASSERT(reader.open(model_path), "open GGUF");

    funasr::Tokenizer tokenizer;
    TEST_ASSERT(!tokenizer.is_loaded(), "not loaded initially");

    bool ok = tokenizer.load(reader);
    TEST_ASSERT(ok, "load succeeds");
    TEST_ASSERT(tokenizer.is_loaded(), "loaded after load()");

    TEST_EQ(tokenizer.vocab_size(), 151936, "vocab_size");
    TEST_EQ(tokenizer.eos_id(), 151645, "eos_id");
    TEST_EQ(tokenizer.bos_id(), 151643, "bos_id");

    printf("  [OK] Load passed\n");
}

// ============================================================
// Test 2: 简单 decode 验证
// ============================================================
void test_decode_basic(const std::string& model_path) {
    printf("\n--- Test 2: Decode Basic ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    // 空序列
    std::string result = tokenizer.decode({});
    TEST_STR_EQ(result, std::string(""), "empty decode");

    // 特殊 token 应被跳过
    result = tokenizer.decode({151645});  // eos
    TEST_STR_EQ(result, std::string(""), "eos skipped");

    result = tokenizer.decode({151643});  // bos
    TEST_STR_EQ(result, std::string(""), "bos skipped");

    // 单个 token 的 decode
    // "Hello" 的 token 在 Qwen 词表中通常是一个完整 token
    // 我们先 encode 再 decode 来验证
    std::string hello_decoded = tokenizer.decode_token(9707);
    printf("  token 9707 = '%s'\n", hello_decoded.c_str());
    // 不做精确值断言（依赖词表），但不应为空
    TEST_ASSERT(!hello_decoded.empty(), "token 9707 decodes to non-empty");

    printf("  [OK] Decode basic passed\n");
}

// ============================================================
// Test 3: Encode + Decode round-trip（英文）
// ============================================================
void test_roundtrip_english(const std::string& model_path) {
    printf("\n--- Test 3: Round-trip English ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    std::string text = "Hello world";
    auto ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    printf("  Text:    '%s'\n", text.c_str());
    printf("  IDs:     [");
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%d", ids[i]);
        if (i + 1 < ids.size()) printf(", ");
    }
    printf("]\n");
    printf("  Decoded: '%s'\n", decoded.c_str());

    TEST_ASSERT(!ids.empty(), "encode produces tokens");
    TEST_STR_EQ(decoded, text, "english round-trip");

    printf("  [OK] Round-trip English passed\n");
}

// ============================================================
// Test 4: Encode + Decode round-trip（中文）
// ============================================================
void test_roundtrip_chinese(const std::string& model_path) {
    printf("\n--- Test 4: Round-trip Chinese ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    std::string text = "你好世界";
    auto ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    printf("  Text:    '%s'\n", text.c_str());
    printf("  IDs:     [");
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%d", ids[i]);
        if (i + 1 < ids.size()) printf(", ");
    }
    printf("]\n");
    printf("  Decoded: '%s'\n", decoded.c_str());

    TEST_ASSERT(!ids.empty(), "chinese encode produces tokens");
    TEST_STR_EQ(decoded, text, "chinese round-trip");

    printf("  [OK] Round-trip Chinese passed\n");
}

// ============================================================
// Test 5: 特殊 token 在 encode 中的处理
// ============================================================
void test_special_tokens(const std::string& model_path) {
    printf("\n--- Test 5: Special Token Encoding ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    // 含特殊 token 的完整 prompt
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    auto ids = tokenizer.encode(prompt);

    printf("  Prompt: '%s'\n", prompt.c_str());
    printf("  IDs (%zu tokens): [", ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%d", ids[i]);
        if (i + 1 < ids.size()) printf(", ");
    }
    printf("]\n");

    TEST_ASSERT(!ids.empty(), "prompt encode non-empty");

    // 第一个 token 应该是 <|im_start|> = 151644
    TEST_EQ(ids[0], 151644, "first token is <|im_start|>");

    // 最后一个 token 应该是 <|im_end|> = 151645
    TEST_EQ(ids.back(), 151645, "last token is <|im_end|>");

    printf("  [OK] Special tokens passed\n");
}

// ============================================================
// Test 6: ChatML prompt 完整 encode/decode
// 模拟真实推理时的 prompt 构建
// ============================================================
void test_chatml_prompt(const std::string& model_path) {
    printf("\n--- Test 6: ChatML Prompt ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    // 这是实际推理时用的 prompt 结构
    std::string prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                         "<|im_start|>user\n语音转写：";
    std::string suffix = "<|im_end|>\n<|im_start|>assistant\n";

    auto prefix_ids = tokenizer.encode(prefix);
    auto suffix_ids = tokenizer.encode(suffix);

    printf("  Prefix: %zu tokens\n", prefix_ids.size());
    printf("    First 5: [");
    for (size_t i = 0; i < std::min((size_t)5, prefix_ids.size()); i++) {
        printf("%d", prefix_ids[i]);
        if (i + 1 < std::min((size_t)5, prefix_ids.size())) printf(", ");
    }
    printf("]\n");

    printf("  Suffix: %zu tokens\n", suffix_ids.size());
    printf("    All: [");
    for (size_t i = 0; i < suffix_ids.size(); i++) {
        printf("%d", suffix_ids[i]);
        if (i + 1 < suffix_ids.size()) printf(", ");
    }
    printf("]\n");

    // 基本验证
    TEST_ASSERT(prefix_ids.size() > 5, "prefix has enough tokens");
    TEST_ASSERT(suffix_ids.size() > 3, "suffix has enough tokens");
    TEST_EQ(prefix_ids[0], 151644, "prefix starts with <|im_start|>");

    // suffix 应该以 <|im_end|> 开头
    TEST_EQ(suffix_ids[0], 151645, "suffix starts with <|im_end|>");

    printf("  [OK] ChatML prompt passed\n");
}

// ============================================================
// Test 7: 多种文本 round-trip 压力测试
// ============================================================
void test_roundtrip_various(const std::string& model_path) {
    printf("\n--- Test 7: Various Round-trips ---\n");

    funasr::GGUFReader reader;
    reader.open(model_path);
    funasr::Tokenizer tokenizer;
    tokenizer.load(reader);

    std::vector<std::string> test_cases = {
        "Hello",
        "hello world",
        "The quick brown fox",
        "12345",
        "Hello, World! 123",
        "你好",
        "语音识别测试",
    };

    int pass_count = 0;
    for (const auto& text : test_cases) {
        auto ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);

        bool match = (decoded == text);
        if (match) {
            pass_count++;
        } else {
            printf("  MISMATCH: '%s' → [%zu ids] → '%s'\n",
                   text.c_str(), ids.size(), decoded.c_str());
        }
    }

    printf("  %d/%zu round-trips passed\n", pass_count, test_cases.size());
    TEST_EQ(pass_count, (int)test_cases.size(), "all round-trips match");

    printf("  [OK] Various round-trips passed\n");
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
    printf("  FunASR Tokenizer Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());

    test_load(model_path);
    test_decode_basic(model_path);
    test_roundtrip_english(model_path);
    test_roundtrip_chinese(model_path);
    test_special_tokens(model_path);
    test_chatml_prompt(model_path);
    test_roundtrip_various(model_path);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}