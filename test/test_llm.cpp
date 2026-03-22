// 端到端测试：WAV → Fbank → Encoder → Adaptor → LLM → 文字
//
// 构建:
//   cd build && cmake .. && make test_llm
// 运行:
//   ./test_llm ../FunAsr_q8.bin ../zh.wav
//
#include "core/gguf_reader.hpp"
#include "core/config.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "model/tokenizer.hpp"
#include "compute/fbank.hpp"
#include "compute/encoder_ops.hpp"
#include "compute/adaptor_ops.hpp"
#include "compute/llm_ops.hpp"
#include "compute/kv_cache.hpp"
#include "compute/graph_runner.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
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
// 辅助: 从 embed_tokens 获取单个 token 的 embedding
// ============================================================
static void get_token_embedding(
    ggml_tensor* embed_tokens,
    int token_id,
    float* out,
    int embed_dim
) {
    if (embed_tokens->type == GGML_TYPE_BF16) {
        ggml_bf16_t* src = (ggml_bf16_t*)embed_tokens->data + token_id * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            out[j] = ggml_bf16_to_fp32(src[j]);
        }
    } else if (embed_tokens->type == GGML_TYPE_F32) {
        float* src = (float*)embed_tokens->data + token_id * embed_dim;
        memcpy(out, src, embed_dim * sizeof(float));
    } else if (embed_tokens->type == GGML_TYPE_F16) {
        ggml_fp16_t* src = (ggml_fp16_t*)embed_tokens->data + token_id * embed_dim;
        for (int j = 0; j < embed_dim; j++) {
            out[j] = ggml_fp16_to_fp32(src[j]);
        }
    }
}

// ============================================================
// 辅助: 构建 inputs_embeds
// prompt 结构: [prefix_ids] [audio_features] [suffix_ids]
// ============================================================
static void build_inputs_embeds(
    const std::vector<int>& prefix_ids,
    const std::vector<int>& suffix_ids,
    float* adaptor_data,
    int audio_frames,
    ggml_tensor* embed_tokens,
    int embed_dim,
    float* out_embeds,    // 输出: [total_len × embed_dim]
    int& total_len
) {
    total_len = static_cast<int>(prefix_ids.size()) + audio_frames
              + static_cast<int>(suffix_ids.size());

    int pos = 0;

    // prefix token embeddings
    for (int id : prefix_ids) {
        get_token_embedding(embed_tokens, id, out_embeds + pos * embed_dim, embed_dim);
        pos++;
    }

    // audio features (from adaptor output)
    memcpy(out_embeds + pos * embed_dim,
           adaptor_data,
           audio_frames * embed_dim * sizeof(float));
    pos += audio_frames;

    // suffix token embeddings
    for (int id : suffix_ids) {
        get_token_embedding(embed_tokens, id, out_embeds + pos * embed_dim, embed_dim);
        pos++;
    }
}

// ============================================================
// 辅助: Argmax
// ============================================================
static int argmax(const float* logits, int vocab_size) {
    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================
// Test 1: 完整端到端推理
// WAV → Fbank → Encoder → Adaptor → build_embeds → LLM → decode
// ============================================================
void test_end_to_end(
    funasr::FunASRModel& model,
    funasr::Tokenizer& tokenizer,
    const std::vector<float>& fbank_data,
    int T_fbank, int D_fbank
) {
    printf("\n--- Test 1: End-to-End Inference ---\n");

    const int embed_dim  = model.config.llm.embedding_length;  // 1024
    const int vocab_size = model.config.llm.vocab_size;
    const int eos_id     = tokenizer.eos_id();

    // ===== Phase 0: Encoder + Adaptor =====
    printf("  Phase 0: Encoder + Adaptor...\n");
    {
        const size_t mem_size = 4ULL * 1024 * 1024 * 1024;
        struct ggml_init_params params = { mem_size, nullptr, false };
        struct ggml_context* ctx_ea = ggml_init(params);
        TEST_ASSERT(ctx_ea != nullptr, "alloc encoder+adaptor context");
        if (!ctx_ea) return;

        struct ggml_tensor* fbank_t = ggml_new_tensor_2d(ctx_ea, GGML_TYPE_F32, D_fbank, T_fbank);
        memcpy(fbank_t->data, fbank_data.data(), T_fbank * D_fbank * sizeof(float));

        struct ggml_tensor* enc_out = funasr::encoder_forward(
            ctx_ea, fbank_t, model.encoder, model.config.encoder);
        struct ggml_tensor* adp_out = funasr::adaptor_forward(
            ctx_ea, enc_out, model.adaptor, model.config.adaptor);

        auto t0 = std::chrono::high_resolution_clock::now();
        funasr::run_graph(ctx_ea, adp_out, 4);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  Encoder+Adaptor: %ld ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        printf("  Adaptor output: [%lld, %lld]\n",
               (long long)adp_out->ne[0], (long long)adp_out->ne[1]);

        TEST_EQ((int)adp_out->ne[0], embed_dim, "adaptor out dim = 1024");

        int audio_frames = static_cast<int>(adp_out->ne[1]);

        // ===== Phase 1: Build inputs_embeds =====
        printf("  Phase 1: Build inputs_embeds...\n");

        std::string prefix_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                  "<|im_start|>user\n语音转写：";
        std::string suffix_text = "<|im_end|>\n<|im_start|>assistant\n";

        auto prefix_ids = tokenizer.encode(prefix_text);
        auto suffix_ids = tokenizer.encode(suffix_text);

        int total_len = static_cast<int>(prefix_ids.size()) + audio_frames
                      + static_cast<int>(suffix_ids.size());

        printf("  Prefix: %zu tokens, Audio: %d frames, Suffix: %zu tokens, Total: %d\n",
               prefix_ids.size(), audio_frames, suffix_ids.size(), total_len);

        std::vector<float> inputs_embeds(total_len * embed_dim);
        int actual_len = 0;
        build_inputs_embeds(
            prefix_ids, suffix_ids,
            (float*)adp_out->data, audio_frames,
            model.llm.embed_tokens, embed_dim,
            inputs_embeds.data(), actual_len
        );
        TEST_EQ(actual_len, total_len, "inputs_embeds length");

        // 保存 inputs_embeds 到 vector，释放 encoder/adaptor context
        ggml_free(ctx_ea);

        // ===== Phase 2: LLM Prefill =====
        printf("  Phase 2: LLM Prefill...\n");

        funasr::KVCache cache;
        TEST_ASSERT(cache.init(model.config.llm, 2048), "init KV cache");

        // 持久化 logits buffer
        std::vector<float> logits_buf(vocab_size);

        {
            const size_t llm_mem = 1024ULL * 1024 * 1024;
            struct ggml_init_params llm_params = { llm_mem, nullptr, false };
            struct ggml_context* ctx_prefill = ggml_init(llm_params);
            TEST_ASSERT(ctx_prefill != nullptr, "alloc prefill context");
            if (!ctx_prefill) return;

            struct ggml_tensor* embeds_t = ggml_new_tensor_2d(
                ctx_prefill, GGML_TYPE_F32, embed_dim, total_len);
            memcpy(embeds_t->data, inputs_embeds.data(),
                   total_len * embed_dim * sizeof(float));

            auto t2 = std::chrono::high_resolution_clock::now();

            struct ggml_tensor* logits = funasr::llm_forward(
                ctx_prefill, embeds_t, model.llm, cache, 0, model.config.llm);

            funasr::run_graph(ctx_prefill, logits, 4);

            // commit KV cache (在 free 之前!)
            cache.commit(0, total_len);
            cache.set_n_past(total_len);

            auto t3 = std::chrono::high_resolution_clock::now();
            printf("  Prefill: %ld ms (%d tokens)\n",
                   std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count(),
                   total_len);

            // 验证 logits shape
            TEST_EQ((int)logits->ne[0], vocab_size, "logits dim = vocab_size");
            TEST_EQ((int)logits->ne[1], total_len,  "logits seq = total_len");

            // 保存最后一个位置的 logits
            float* last_logits = (float*)logits->data + (total_len - 1) * vocab_size;
            memcpy(logits_buf.data(), last_logits, vocab_size * sizeof(float));

            ggml_free(ctx_prefill);
        }

        // ===== Phase 3: Autoregressive Decode =====
        printf("  Phase 3: Decode...\n");
        auto t_decode_start = std::chrono::high_resolution_clock::now();

        std::vector<int> generated_ids;
        const int max_new_tokens = 100;
        const size_t decode_mem = 1024ULL * 1024 * 1024;

        for (int step = 0; step < max_new_tokens; step++) {
            int next_token = argmax(logits_buf.data(), vocab_size);

            if (next_token == eos_id) {
                printf("  [EOS at step %d]\n", step);
                break;
            }

            generated_ids.push_back(next_token);

            // 打印流式输出
            std::string tok_str = tokenizer.decode({next_token});
            printf("%s", tok_str.c_str());
            fflush(stdout);

            // Decode step: 单个 token
            struct ggml_init_params dp = { decode_mem, nullptr, false };
            struct ggml_context* ctx_dec = ggml_init(dp);
            if (!ctx_dec) { printf("\n  [ERROR] alloc decode ctx\n"); break; }

            struct ggml_tensor* new_embed = ggml_new_tensor_2d(
                ctx_dec, GGML_TYPE_F32, embed_dim, 1);
            get_token_embedding(model.llm.embed_tokens, next_token,
                                (float*)new_embed->data, embed_dim);

            struct ggml_tensor* logits = funasr::llm_forward(
                ctx_dec, new_embed, model.llm, cache, cache.n_past(), model.config.llm);

            funasr::run_graph(ctx_dec, logits, 4);

            // commit + save logits (在 free 之前!)
            cache.commit(cache.n_past(), 1);
            cache.set_n_past(cache.n_past() + 1);

            float* last_logits = (float*)logits->data;  // seq_len=1, 只有一个位置
            memcpy(logits_buf.data(), last_logits, vocab_size * sizeof(float));

            ggml_free(ctx_dec);
        }

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_decode_end - t_decode_start).count();

        printf("\n");

        // ===== 结果 =====
        std::string result = tokenizer.decode(generated_ids);

        printf("\n  ========== Result ==========\n");
        printf("  Text: '%s'\n", result.c_str());
        printf("  Tokens: %zu\n", generated_ids.size());
        printf("  Decode: %ld ms", decode_ms);
        if (!generated_ids.empty() && decode_ms > 0) {
            printf(" (%.1f tok/s)", generated_ids.size() * 1000.0 / decode_ms);
        }
        printf("\n");
        printf("  ============================\n");

        // 基本验证
        TEST_ASSERT(!generated_ids.empty(), "generated at least 1 token");
        TEST_ASSERT(result.size() > 0, "decoded text non-empty");
    }

    printf("  [OK] End-to-end passed\n");
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
    printf("  FunASR End-to-End Test\n");
    printf("========================================\n");
    printf("Model: %s\n", model_path.c_str());
    printf("WAV:   %s\n", wav_path.c_str());

    // ===== 加载模型 =====
    printf("\nLoading model...\n");
    funasr::GGUFReader reader;
    if (!reader.open(model_path)) return 1;

    funasr::FunASRModel model;
    if (!funasr::ModelLoader::load(reader, model)) return 1;

    // ===== 加载 Tokenizer =====
    printf("\nLoading tokenizer...\n");
    funasr::Tokenizer tokenizer;
    if (!tokenizer.load(reader)) return 1;

    // ===== Fbank =====
    printf("\nExtracting fbank...\n");
    funasr::FbankProcessor fbank(model.config.frontend);
    std::vector<float> fbank_data;
    int T, D;
    if (!fbank.process_wav(wav_path, fbank_data, T, D)) {
        printf("Fbank failed\n");
        return 1;
    }
    printf("Fbank: [%d, %d]\n", T, D);

    // ===== 端到端测试 =====
    test_end_to_end(model, tokenizer, fbank_data, T, D);

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}