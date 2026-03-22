// GPU 图执行器 — 封装 ggml_gallocr + warmup + forward
//
// 关键设计:
//   1. 分离 prefill (seq>1) 和 decode (seq=1) 的 allocator
//   2. warmup 预分配 GPU buffer，避免推理时突然分配
//   3. 遍历图找 ROPE 节点设置 position tensor
//   4. kv_cpy_ops 必须加入图中才会执行
//
#ifndef FUNASR_COMPUTE_GPU_RUNNER_HPP
#define FUNASR_COMPUTE_GPU_RUNNER_HPP

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include "compute/gpu_context.hpp"
#include "compute/llm_ops_gpu.hpp"
#include <vector>
#include <cstdio>

namespace funasr {

class GPURunner {
public:
    explicit GPURunner(GPUContext& ctx)
        : gpu_ctx_(ctx)
    {
        auto buf_type = ggml_backend_get_default_buffer_type(gpu_ctx_.backend());
        allocr_prefill_ = ggml_gallocr_new(buf_type);
        allocr_decode_  = ggml_gallocr_new(buf_type);
    }

    ~GPURunner() {
        if (allocr_prefill_) ggml_gallocr_free(allocr_prefill_);
        if (allocr_decode_)  ggml_gallocr_free(allocr_decode_);
    }

    GPURunner(const GPURunner&) = delete;
    GPURunner& operator=(const GPURunner&) = delete;

    // ============================================================
    // Warmup: 用 dummy 数据跑一次 prefill + decode
    // 让 gallocr 确定各自的 buffer 大小
    // ============================================================
    bool warmup(int max_prefill_len = 500, int max_past = 2048) {
        printf("[GPURunner] Warming up...\n");

        const int embed_dim = gpu_ctx_.config().embedding_length;
        std::vector<float> dummy_logits;

        // Prefill warmup
        printf("[GPURunner]   Prefill (max_len=%d)...\n", max_prefill_len);
        std::vector<float> dummy_prefill(embed_dim * max_prefill_len, 0.1f);
        if (!forward(dummy_prefill.data(), max_prefill_len, 0, dummy_logits)) {
            printf("[GPURunner] Prefill warmup failed\n");
            return false;
        }

        // Decode warmup
        printf("[GPURunner]   Decode (max_past=%d)...\n", max_past);
        std::vector<float> dummy_decode(embed_dim, 0.1f);
        if (!forward(dummy_decode.data(), 1, max_past, dummy_logits)) {
            printf("[GPURunner] Decode warmup failed\n");
            return false;
        }

        // 清空 warmup 产生的 KV Cache
        gpu_ctx_.clear_kv_cache();

        warmed_up_ = true;
        printf("[GPURunner] Warmup complete!\n");
        return true;
    }

    // ============================================================
    // Forward: CPU float → GPU 计算 → CPU logits
    //
    // input_embeds: CPU 上的 [embed_dim, seq_len] 行优先 float
    // seq_len:      当前序列长度 (prefill: >1, decode: =1)
    // n_past:       已缓存的 token 数
    // logits_out:   输出最后一个位置的 logits [vocab_size]
    // ============================================================
    bool forward(
        const float* input_embeds,
        int seq_len,
        int n_past,
        std::vector<float>& logits_out
    ) {
        const auto& cfg = gpu_ctx_.config();
        const int embed_dim  = cfg.embedding_length;
        const int vocab_size = cfg.vocab_size;

        // 选择 allocator
        bool is_decode = (seq_len == 1);
        ggml_gallocr_t allocr = is_decode ? allocr_decode_ : allocr_prefill_;

        // 准备 position 数据
        pos_data_.resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            pos_data_[i] = n_past + i;
        }

        // ===== 1. 创建图构建 context (no_alloc=true) =====
        size_t ctx_size = 256ULL * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("[GPURunner] Failed to create context\n");
            return false;
        }

        // ===== 2. 创建输入 tensor =====
        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, seq_len);
        ggml_set_name(input, "input_embeds");
        ggml_set_input(input);

        // ===== 3. 构建计算图 =====
        std::vector<ggml_tensor*> kv_cpy_ops;

        ggml_tensor* logits = gpu_llm_forward(
            ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
            n_past, cfg, kv_cpy_ops
        );
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        // 创建图，先加 kv_cpy 操作再加 logits
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 131072, false);
        for (auto* cpy_op : kv_cpy_ops) {
            ggml_build_forward_expand(graph, cpy_op);
        }
        ggml_build_forward_expand(graph, logits);

        // ===== 4. 分配 GPU buffer =====
        if (!ggml_gallocr_alloc_graph(allocr, graph)) {
            printf("[GPURunner] Failed to alloc graph\n");
            ggml_free(ctx);
            return false;
        }

        // ===== 5. 设置输入数据 (CPU → GPU) =====
        ggml_backend_tensor_set(input, input_embeds, 0,
                                 embed_dim * seq_len * sizeof(float));

        // ===== 6. 设置 RoPE position tensor =====
        // GPU 模式下 pos->data 是 NULL，需要遍历图找 ROPE 节点
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_ROPE) {
                ggml_tensor* pos_tensor = node->src[1];
                if (pos_tensor && pos_tensor->type == GGML_TYPE_I32) {
                    ggml_backend_tensor_set(pos_tensor, pos_data_.data(), 0,
                                             seq_len * sizeof(int));
                }
            }
        }

        // ===== 7. 执行计算图 =====
        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            printf("[GPURunner] Graph compute failed\n");
            ggml_free(ctx);
            return false;
        }

        // ===== 8. 获取输出 (GPU → CPU) =====
        logits_out.resize(vocab_size);
        int last_pos = seq_len - 1;
        ggml_backend_tensor_get(logits, logits_out.data(),
                                 last_pos * vocab_size * sizeof(float),
                                 vocab_size * sizeof(float));

        // 更新 KV Cache position
        gpu_ctx_.kv_cache().n_past = n_past + seq_len;

        ggml_free(ctx);
        return true;
    }

    // 简单 argmax
    static int argmax(const std::vector<float>& logits) {
        float max_val = -1e30f;
        int max_idx = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }

    bool is_warmed_up() const { return warmed_up_; }

private:
    GPUContext& gpu_ctx_;
    ggml_gallocr_t allocr_prefill_ = nullptr;
    ggml_gallocr_t allocr_decode_  = nullptr;
    std::vector<int> pos_data_;
    bool warmed_up_ = false;
};

} // namespace funasr

#endif // FUNASR_COMPUTE_GPU_RUNNER_HPP