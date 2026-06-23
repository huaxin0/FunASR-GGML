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
#include "compute/gpu_profile.hpp"
#include "compute/llm_ops_gpu.hpp"
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <utility>

namespace funasr {

extern "C" void ggml_cuda_paged_attn_profile_reset();
extern "C" void ggml_cuda_paged_attn_profile_get(long * calls, double * ms);

class GPURunner {
public:
    explicit GPURunner(GPUContext& ctx)
        : gpu_ctx_(ctx)
    {
        auto buf_type = ggml_backend_get_default_buffer_type(gpu_ctx_.backend());
        allocr_prefill_ = ggml_gallocr_new(buf_type);
        allocr_decode_  = ggml_gallocr_new(buf_type);
        ggml_cuda_paged_attn_profile_reset();
    }

    ~GPURunner() {
        clear_paged_decode_graph_cache();
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
        return forward_impl(input_embeds, nullptr, seq_len, n_past, 0, logits_out);
    }

    bool forward_slot(
        const float* input_embeds,
        int seq_len,
        int n_past,
        int slot_id,
        std::vector<float>& logits_out
    ) {
        return forward_impl(input_embeds, nullptr, seq_len, n_past, slot_id, logits_out);
    }

    // ============================================================
    // Forward (GPU tensor 输入): 直接接受 GPU 上的 tensor
    //
    // gpu_embeds:   GPU 上的 [embed_dim, seq_len] tensor
    //               从 GPUEncoderAdaptorRunner::forward_on_gpu() 获得
    // 其余参数同上
    // ============================================================
    bool forward_gpu_tensor(
        ggml_tensor* gpu_embeds,
        int seq_len,
        int n_past,
        std::vector<float>& logits_out
    ) {
        return forward_impl(nullptr, gpu_embeds, seq_len, n_past, 0, logits_out);
    }

    bool forward_gpu_tensor_slot(
        ggml_tensor* gpu_embeds,
        int seq_len,
        int n_past,
        int slot_id,
        std::vector<float>& logits_out
    ) {
        return forward_impl(nullptr, gpu_embeds, seq_len, n_past, slot_id, logits_out);
    }

    bool forward_paged(
        const float* input_embeds,
        int seq_len,
        int n_past,
        const std::vector<int>& block_table,
        int block_size,
        std::vector<float>& logits_out
    ) {
        return forward_paged_impl(input_embeds, nullptr, seq_len, n_past,
                                  block_table, block_size, logits_out);
    }

    bool forward_gpu_tensor_paged(
        ggml_tensor* gpu_embeds,
        int seq_len,
        int n_past,
        const std::vector<int>& block_table,
        int block_size,
        std::vector<float>& logits_out
    ) {
        return forward_paged_impl(nullptr, gpu_embeds, seq_len, n_past,
                                  block_table, block_size, logits_out);
    }

    bool forward_batch_decode_slots(
        const std::vector<const float*>& input_embeds,
        const std::vector<int>& n_pasts,
        const std::vector<int>& slot_ids,
        std::vector<std::vector<float>>& logits_out
    ) {
        const int batch = static_cast<int>(input_embeds.size());
        if (batch <= 0 || static_cast<int>(n_pasts.size()) != batch ||
            static_cast<int>(slot_ids.size()) != batch) {
            return false;
        }

        const auto& cfg = gpu_ctx_.config();
        const int embed_dim  = cfg.embedding_length;
        const int vocab_size = cfg.vocab_size;

        size_t ctx_size = 384ULL * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("[GPURunner] Failed to create batch decode context\n");
            return false;
        }

        std::vector<ggml_tensor*> inputs;
        std::vector<ggml_tensor*> logits_nodes;
        std::vector<ggml_tensor*> kv_cpy_ops;
        inputs.reserve(static_cast<size_t>(batch));
        logits_nodes.reserve(static_cast<size_t>(batch));

        for (int i = 0; i < batch; i++) {
            ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, 1);
            ggml_set_name(input, "batch_decode_input");
            ggml_set_input(input);
            inputs.push_back(input);

            ggml_tensor* logits = gpu_llm_forward_slot(
                ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
                n_pasts[i], slot_ids[i], cfg, kv_cpy_ops);
            ggml_set_name(logits, "batch_decode_logits");
            ggml_set_output(logits);
            logits_nodes.push_back(logits);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 262144, false);
        for (auto* cpy_op : kv_cpy_ops) {
            ggml_build_forward_expand(graph, cpy_op);
        }
        for (auto* logits : logits_nodes) {
            ggml_build_forward_expand(graph, logits);
        }

        clear_paged_decode_graph_cache();
        if (!ggml_gallocr_alloc_graph(allocr_decode_, graph)) {
            printf("[GPURunner] Failed to alloc batch decode graph\n");
            ggml_free(ctx);
            return false;
        }

        for (int i = 0; i < batch; i++) {
            ggml_backend_tensor_set(inputs[i], input_embeds[i], 0,
                                    embed_dim * sizeof(float));
        }

        const int ropes_per_request = cfg.block_count * 2;
        int rope_seen = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_ROPE) {
                continue;
            }
            int request_idx = std::min(batch - 1, rope_seen / ropes_per_request);
            int pos = n_pasts[request_idx];
            ggml_tensor* pos_tensor = node->src[1];
            if (pos_tensor && pos_tensor->type == GGML_TYPE_I32) {
                ggml_backend_tensor_set(pos_tensor, &pos, 0, sizeof(int));
            }
            rope_seen++;
        }

#ifdef FUNASR_USE_FLASH_ATTN
        int mask_seen = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }

            ggml_tensor* mask = node->src[3];
            if (!mask || mask->type != GGML_TYPE_F16) {
                continue;
            }

            const int n_kv_mask = static_cast<int>(mask->ne[0]);
            const int n_tokens_mask = static_cast<int>(mask->ne[1]);
            const int n_broadcast = static_cast<int>(mask->ne[2] * mask->ne[3]);
            const size_t n_mask = static_cast<size_t>(n_kv_mask)
                                * n_tokens_mask * n_broadcast;
            flash_mask_data_.assign(n_mask, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(mask, flash_mask_data_.data(), 0,
                                    flash_mask_data_.size() * sizeof(ggml_fp16_t));
            mask_seen++;
        }
#endif

        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            printf("[GPURunner] Batch decode graph compute failed\n");
            ggml_free(ctx);
            return false;
        }

        logits_out.assign(static_cast<size_t>(batch), std::vector<float>(vocab_size));
        for (int i = 0; i < batch; i++) {
            ggml_backend_tensor_get(logits_nodes[i], logits_out[static_cast<size_t>(i)].data(),
                                    0, vocab_size * sizeof(float));
        }

        ggml_free(ctx);
        return true;
    }

    bool forward_batch_decode_paged(
        const std::vector<const float*>& input_embeds,
        const std::vector<int>& n_pasts,
        const std::vector<std::vector<int>>& block_tables,
        int block_size,
        std::vector<std::vector<float>>& logits_out
    ) {
        const int batch = static_cast<int>(input_embeds.size());
        if (batch <= 0 || static_cast<int>(n_pasts.size()) != batch ||
            static_cast<int>(block_tables.size()) != batch || block_size <= 0) {
            return false;
        }

        const auto& cfg = gpu_ctx_.config();
        const int embed_dim  = cfg.embedding_length;
        const int vocab_size = cfg.vocab_size;
        const int physical_rows = gpu_ctx_.kv_cache().physical_rows > 0
            ? gpu_ctx_.kv_cache().physical_rows
            : gpu_ctx_.kv_cache().n_ctx;

        for (int i = 0; i < batch; i++) {
            const int needed_blocks = (n_pasts[i] + 1 + block_size - 1) / block_size;
            if (static_cast<int>(block_tables[i].size()) < needed_blocks) {
                printf("[GPURunner] Batch paged KV block table too small\n");
                return false;
            }
            for (int block : block_tables[i]) {
                if (block < 0 || (block + 1) * block_size > physical_rows) {
                    printf("[GPURunner] Batch paged KV block out of range\n");
                    return false;
                }
            }
        }

        size_t ctx_size = 512ULL * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("[GPURunner] Failed to create batch paged decode context\n");
            return false;
        }

        std::vector<ggml_tensor*> inputs;
        std::vector<ggml_tensor*> logits_nodes;
        std::vector<ggml_tensor*> kv_cpy_ops;
        std::vector<std::vector<ggml_tensor*>> block_table_inputs(static_cast<size_t>(batch));
        std::vector<std::vector<ggml_tensor*>> position_inputs(static_cast<size_t>(batch));
        inputs.reserve(static_cast<size_t>(batch));
        logits_nodes.reserve(static_cast<size_t>(batch));

        for (int i = 0; i < batch; i++) {
            ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, 1);
            ggml_set_name(input, "batch_paged_decode_input");
            ggml_set_input(input);
            inputs.push_back(input);

            ggml_tensor* logits = gpu_llm_forward_paged(
                ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
                n_pasts[i], block_tables[i], block_size, cfg, kv_cpy_ops,
                &block_table_inputs[static_cast<size_t>(i)],
                &position_inputs[static_cast<size_t>(i)]);
            ggml_set_name(logits, "batch_paged_decode_logits");
            ggml_set_output(logits);
            logits_nodes.push_back(logits);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 262144, false);
        for (auto* cpy_op : kv_cpy_ops) {
            ggml_build_forward_expand(graph, cpy_op);
        }
        for (auto* logits : logits_nodes) {
            ggml_build_forward_expand(graph, logits);
        }

        clear_paged_decode_graph_cache();
        if (!ggml_gallocr_alloc_graph(allocr_decode_, graph)) {
            printf("[GPURunner] Failed to alloc batch paged decode graph\n");
            ggml_free(ctx);
            return false;
        }

        for (int i = 0; i < batch; i++) {
            ggml_backend_tensor_set(inputs[i], input_embeds[i], 0,
                                    embed_dim * sizeof(float));
        }

        for (int i = 0; i < batch; i++) {
            int pos = n_pasts[i];
            for (ggml_tensor* pos_tensor : position_inputs[static_cast<size_t>(i)]) {
                ggml_backend_tensor_set(pos_tensor, &pos, 0, sizeof(int));
            }
        }

        for (int i = 0; i < batch; i++) {
            for (ggml_tensor* table : block_table_inputs[static_cast<size_t>(i)]) {
                ggml_backend_tensor_set(table, block_tables[static_cast<size_t>(i)].data(), 0,
                    static_cast<size_t>(block_tables[static_cast<size_t>(i)].size()) * sizeof(int));
            }
        }

        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            printf("[GPURunner] Batch paged decode graph compute failed\n");
            ggml_free(ctx);
            return false;
        }

        logits_out.assign(static_cast<size_t>(batch), std::vector<float>(vocab_size));
        for (int i = 0; i < batch; i++) {
            ggml_backend_tensor_get(logits_nodes[i], logits_out[static_cast<size_t>(i)].data(),
                                    0, vocab_size * sizeof(float));
        }

        ggml_free(ctx);
        return true;
    }

    bool forward_batch_decode_paged_tokens(
        const std::vector<const float*>& input_embeds,
        const std::vector<int>& n_pasts,
        const std::vector<std::vector<int>>& block_tables,
        int block_size,
        std::vector<int>& tokens_out
    ) {
        const int batch = static_cast<int>(input_embeds.size());
        if (batch <= 0 || static_cast<int>(n_pasts.size()) != batch ||
            static_cast<int>(block_tables.size()) != batch || block_size <= 0) {
            return false;
        }

        const auto& cfg = gpu_ctx_.config();
        const int embed_dim = cfg.embedding_length;
        const int physical_rows = gpu_ctx_.kv_cache().physical_rows > 0
            ? gpu_ctx_.kv_cache().physical_rows
            : gpu_ctx_.kv_cache().n_ctx;
        for (int i = 0; i < batch; i++) {
            const int needed_blocks = (n_pasts[i] + 1 + block_size - 1) / block_size;
            if (static_cast<int>(block_tables[i].size()) < needed_blocks) {
                printf("[GPURunner] Batch paged KV block table too small\n");
                return false;
            }
            for (int block : block_tables[i]) {
                if (block < 0 || (block + 1) * block_size > physical_rows) {
                    printf("[GPURunner] Batch paged KV block out of range\n");
                    return false;
                }
            }
        }

        if (env_enabled("FUNASR_PAGED_DECODE_LEGACY_GRAPH")) {
            std::vector<std::vector<float>> logits;
            if (!forward_batch_decode_paged(
                    input_embeds, n_pasts, block_tables, block_size, logits)) {
                return false;
            }
            tokens_out.resize(static_cast<size_t>(batch));
            for (int i = 0; i < batch; i++) {
                tokens_out[static_cast<size_t>(i)] = argmax(logits[static_cast<size_t>(i)]);
            }
            return true;
        }

        auto t_build0 = std::chrono::high_resolution_clock::now();

        size_t ctx_size = 512ULL * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("[GPURunner] Failed to create batch paged token decode context\n");
            return false;
        }

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, batch);
        ggml_set_name(input, "batch_paged_decode_input");
        ggml_set_input(input);

        ggml_tensor* token = nullptr;
        std::vector<ggml_tensor*> kv_cpy_ops;
        ggml_tensor* block_table_input = nullptr;
        ggml_tensor* position_input = nullptr;
        ggml_tensor* kv_lens_input = nullptr;
        ggml_tensor* logits = gpu_llm_forward_paged_batch_decode(
            ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
            n_pasts, block_tables, block_size, 0, cfg, kv_cpy_ops,
            &block_table_input, &position_input, &kv_lens_input);
        token = ggml_argmax(ctx, logits);
        ggml_set_name(token, "batch_paged_decode_token");
        ggml_set_output(token);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 262144, false);
        for (auto* cpy_op : kv_cpy_ops) {
            ggml_build_forward_expand(graph, cpy_op);
        }
        ggml_build_forward_expand(graph, token);
        auto t_build1 = std::chrono::high_resolution_clock::now();

        clear_paged_decode_graph_cache();
        if (!ggml_gallocr_alloc_graph(allocr_decode_, graph)) {
            printf("[GPURunner] Failed to alloc batch paged token decode graph\n");
            ggml_free(ctx);
            return false;
        }
        auto t_alloc1 = std::chrono::high_resolution_clock::now();

        batch_input_data_.resize(static_cast<size_t>(embed_dim) * batch);
        for (int i = 0; i < batch; i++) {
            std::memcpy(batch_input_data_.data() + static_cast<size_t>(i) * embed_dim,
                        input_embeds[i], static_cast<size_t>(embed_dim) * sizeof(float));
        }
        ggml_backend_tensor_set(input, batch_input_data_.data(), 0,
                                batch_input_data_.size() * sizeof(float));

        batch_pos_data_.assign(n_pasts.begin(), n_pasts.end());
        ggml_backend_tensor_set(position_input, batch_pos_data_.data(), 0,
                                batch_pos_data_.size() * sizeof(int));

        batch_kv_lens_data_.resize(static_cast<size_t>(batch));
        for (int i = 0; i < batch; i++) {
            batch_kv_lens_data_[static_cast<size_t>(i)] = n_pasts[static_cast<size_t>(i)] + 1;
        }
        ggml_backend_tensor_set(kv_lens_input, batch_kv_lens_data_.data(), 0,
                                batch_kv_lens_data_.size() * sizeof(int));

        int max_blocks = 0;
        for (const auto& table : block_tables) {
            max_blocks = std::max(max_blocks, static_cast<int>(table.size()));
        }
        batch_block_table_data_.assign(static_cast<size_t>(max_blocks) * batch, 0);
        for (int i = 0; i < batch; i++) {
            std::memcpy(batch_block_table_data_.data() + static_cast<size_t>(i) * max_blocks,
                        block_tables[static_cast<size_t>(i)].data(),
                        block_tables[static_cast<size_t>(i)].size() * sizeof(int));
        }
        ggml_backend_tensor_set(block_table_input, batch_block_table_data_.data(), 0,
                                batch_block_table_data_.size() * sizeof(int));
        auto t_set1 = std::chrono::high_resolution_clock::now();

        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            printf("[GPURunner] Batch paged token decode graph compute failed\n");
            ggml_free(ctx);
            return false;
        }
        auto t_compute1 = std::chrono::high_resolution_clock::now();

        tokens_out.resize(static_cast<size_t>(batch));
        ggml_backend_tensor_get(token, tokens_out.data(), 0,
                                static_cast<size_t>(batch) * sizeof(int));
        auto t_get1 = std::chrono::high_resolution_clock::now();

        record_paged_decode_profile(
            elapsed_ms(t_build0, t_build1),
            elapsed_ms(t_build1, t_alloc1),
            elapsed_ms(t_alloc1, t_set1),
            elapsed_ms(t_set1, t_compute1),
            elapsed_ms(t_compute1, t_get1));

        ggml_free(ctx);
        return true;
    }

    bool forward_batch_decode_paged_token_ids(
        const std::vector<int>& input_tokens,
        const std::vector<int>& n_pasts,
        const std::vector<std::vector<int>>& block_tables,
        int block_size,
        std::vector<int>& tokens_out
    ) {
        const int batch = static_cast<int>(input_tokens.size());
        if (batch <= 0 || static_cast<int>(n_pasts.size()) != batch ||
            static_cast<int>(block_tables.size()) != batch || block_size <= 0) {
            return false;
        }

        const auto& cfg = gpu_ctx_.config();
        const int physical_rows = gpu_ctx_.kv_cache().physical_rows > 0
            ? gpu_ctx_.kv_cache().physical_rows
            : gpu_ctx_.kv_cache().n_ctx;
        for (int i = 0; i < batch; i++) {
            const int needed_blocks = (n_pasts[i] + 1 + block_size - 1) / block_size;
            if (static_cast<int>(block_tables[i].size()) < needed_blocks) {
                return false;
            }
            for (int block : block_tables[i]) {
                if (block < 0 || (block + 1) * block_size > physical_rows) {
                    return false;
                }
            }
        }

        int max_blocks = 0;
        int exact_max_n_kv = 0;
        const bool dynamic_kv_write = env_enabled("FUNASR_PAGED_KV_WRITE_OP");
        std::vector<int> copy_rows;
        if (!dynamic_kv_write) {
            copy_rows.reserve(static_cast<size_t>(batch));
        }
        for (int i = 0; i < batch; i++) {
            const auto& table = block_tables[static_cast<size_t>(i)];
            max_blocks = std::max(max_blocks, static_cast<int>(table.size()));
            exact_max_n_kv = std::max(exact_max_n_kv, n_pasts[static_cast<size_t>(i)] + 1);
            const int block_idx = n_pasts[static_cast<size_t>(i)] / block_size;
            const int block_off = n_pasts[static_cast<size_t>(i)] % block_size;
            if (!dynamic_kv_write) {
                copy_rows.push_back(table[static_cast<size_t>(block_idx)] * block_size + block_off);
            }
        }
        const int graph_max_n_kv = paged_decode_graph_max_n_kv(
            exact_max_n_kv, max_blocks, block_size,
            env_enabled("FUNASR_PAGED_DECODE_BUCKET_MAX_KV"));
        record_paged_graph_cache_probe(batch, max_blocks, block_size, graph_max_n_kv, copy_rows);

        auto t_build0 = std::chrono::high_resolution_clock::now();
        PagedGraphProbeSignature graph_sig;
        graph_sig.batch = batch;
        graph_sig.max_blocks = max_blocks;
        graph_sig.block_size = block_size;
        graph_sig.max_n_kv = graph_max_n_kv;
        graph_sig.copy_rows = copy_rows;

        const bool use_graph_cache =
            env_enabled("FUNASR_PAGED_DECODE_GRAPH_CACHE") &&
            dynamic_kv_write &&
            env_enabled("FUNASR_PAGED_DECODE_BUCKET_MAX_KV") &&
            graph_max_n_kv > 0 &&
            copy_rows.empty();

        struct ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* token_input = nullptr;
        ggml_tensor* block_table_input = nullptr;
        ggml_tensor* position_input = nullptr;
        ggml_tensor* kv_lens_input = nullptr;
        ggml_tensor* token = nullptr;
        bool graph_cache_hit = false;

        if (use_graph_cache &&
            cached_paged_decode_graph_.valid &&
            same_paged_graph_signature(cached_paged_decode_graph_.signature, graph_sig)) {
            graph_cache_hit = true;
            paged_decode_profile_.graph_cache_hits++;
            ctx = cached_paged_decode_graph_.ctx;
            graph = cached_paged_decode_graph_.graph;
            token_input = cached_paged_decode_graph_.token_input;
            block_table_input = cached_paged_decode_graph_.block_table_input;
            position_input = cached_paged_decode_graph_.position_input;
            kv_lens_input = cached_paged_decode_graph_.kv_lens_input;
            token = cached_paged_decode_graph_.token;
        } else {
            if (use_graph_cache) {
                paged_decode_profile_.graph_cache_misses++;
                clear_paged_decode_graph_cache();
            } else {
                clear_paged_decode_graph_cache();
            }

            size_t ctx_size = 384ULL * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            ctx = ggml_init(params);
            if (!ctx) {
                return false;
            }

            token_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
            ggml_set_name(token_input, "batch_paged_decode_token_input");
            ggml_set_input(token_input);

            ggml_tensor* input4 = ggml_get_rows(ctx, gpu_ctx_.weights().embed_tokens, token_input);
            ggml_tensor* input = ggml_reshape_2d(ctx, input4, cfg.embedding_length, batch);

            std::vector<ggml_tensor*> kv_cpy_ops;
            ggml_tensor* logits = gpu_llm_forward_paged_batch_decode(
                ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
                n_pasts, block_tables, block_size, graph_max_n_kv, cfg, kv_cpy_ops,
                &block_table_input, &position_input, &kv_lens_input);
            token = ggml_argmax(ctx, logits);
            ggml_set_name(token, "batch_paged_decode_token");
            ggml_set_output(token);

            graph = ggml_new_graph_custom(ctx, 262144, false);
            for (auto* cpy_op : kv_cpy_ops) {
                ggml_build_forward_expand(graph, cpy_op);
            }
            ggml_build_forward_expand(graph, token);
        }
        auto t_build1 = std::chrono::high_resolution_clock::now();

        if (!graph_cache_hit) {
            if (!ggml_gallocr_alloc_graph(allocr_decode_, graph)) {
                if (use_graph_cache) {
                    clear_paged_decode_graph_cache();
                } else {
                    ggml_free(ctx);
                }
                return false;
            }

            if (use_graph_cache) {
                cached_paged_decode_graph_.ctx = ctx;
                cached_paged_decode_graph_.graph = graph;
                cached_paged_decode_graph_.token_input = token_input;
                cached_paged_decode_graph_.block_table_input = block_table_input;
                cached_paged_decode_graph_.position_input = position_input;
                cached_paged_decode_graph_.kv_lens_input = kv_lens_input;
                cached_paged_decode_graph_.token = token;
                cached_paged_decode_graph_.signature = std::move(graph_sig);
                cached_paged_decode_graph_.valid = true;
            }
        }
        auto t_alloc1 = std::chrono::high_resolution_clock::now();

        ggml_backend_tensor_set(token_input, input_tokens.data(), 0,
                                static_cast<size_t>(batch) * sizeof(int));

        batch_pos_data_.assign(n_pasts.begin(), n_pasts.end());
        ggml_backend_tensor_set(position_input, batch_pos_data_.data(), 0,
                                batch_pos_data_.size() * sizeof(int));

        batch_kv_lens_data_.resize(static_cast<size_t>(batch));
        for (int i = 0; i < batch; i++) {
            batch_kv_lens_data_[static_cast<size_t>(i)] = n_pasts[static_cast<size_t>(i)] + 1;
        }
        ggml_backend_tensor_set(kv_lens_input, batch_kv_lens_data_.data(), 0,
                                batch_kv_lens_data_.size() * sizeof(int));

        batch_block_table_data_.assign(static_cast<size_t>(max_blocks) * batch, 0);
        for (int i = 0; i < batch; i++) {
            std::memcpy(batch_block_table_data_.data() + static_cast<size_t>(i) * max_blocks,
                        block_tables[static_cast<size_t>(i)].data(),
                        block_tables[static_cast<size_t>(i)].size() * sizeof(int));
        }
        ggml_backend_tensor_set(block_table_input, batch_block_table_data_.data(), 0,
                                batch_block_table_data_.size() * sizeof(int));
        auto t_set1 = std::chrono::high_resolution_clock::now();

        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            if (use_graph_cache) {
                clear_paged_decode_graph_cache();
            } else {
                ggml_free(ctx);
            }
            return false;
        }
        auto t_compute1 = std::chrono::high_resolution_clock::now();

        tokens_out.resize(static_cast<size_t>(batch));
        ggml_backend_tensor_get(token, tokens_out.data(), 0,
                                static_cast<size_t>(batch) * sizeof(int));
        auto t_get1 = std::chrono::high_resolution_clock::now();

        record_paged_decode_profile(
            elapsed_ms(t_build0, t_build1),
            elapsed_ms(t_build1, t_alloc1),
            elapsed_ms(t_alloc1, t_set1),
            elapsed_ms(t_set1, t_compute1),
            elapsed_ms(t_compute1, t_get1));

        if (!use_graph_cache) {
            ggml_free(ctx);
        }
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

    PagedDecodeProfile paged_decode_profile() const {
        PagedDecodeProfile profile = paged_decode_profile_;
        ggml_cuda_paged_attn_profile_get(
            &profile.paged_attn_calls,
            &profile.paged_attn_ms);
        return profile;
    }

private:
    GPUContext& gpu_ctx_;
    ggml_gallocr_t allocr_prefill_ = nullptr;
    ggml_gallocr_t allocr_decode_  = nullptr;
    std::vector<int> pos_data_;
    std::vector<float> batch_input_data_;
    std::vector<int> batch_pos_data_;
    std::vector<int> batch_kv_lens_data_;
    std::vector<int> batch_block_table_data_;
    std::vector<ggml_fp16_t> flash_mask_data_;
    bool warmed_up_ = false;

    PagedDecodeProfile paged_decode_profile_;
    struct PagedGraphProbeSignature {
        int batch = 0;
        int max_blocks = 0;
        int block_size = 0;
        int max_n_kv = 0;
        std::vector<int> copy_rows;
    };
    PagedGraphProbeSignature last_paged_graph_probe_;
    bool has_last_paged_graph_probe_ = false;

    struct CachedPagedDecodeGraph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* token_input = nullptr;
        ggml_tensor* block_table_input = nullptr;
        ggml_tensor* position_input = nullptr;
        ggml_tensor* kv_lens_input = nullptr;
        ggml_tensor* token = nullptr;
        PagedGraphProbeSignature signature;
        bool valid = false;
    };
    CachedPagedDecodeGraph cached_paged_decode_graph_;

    static bool env_enabled(const char* name) {
        const char* value = std::getenv(name);
        return value && std::strcmp(value, "0") != 0;
    }

    static double elapsed_ms(std::chrono::high_resolution_clock::time_point begin,
                             std::chrono::high_resolution_clock::time_point end) {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
    }

    void record_paged_decode_profile(double build_ms, double alloc_ms,
                                     double set_input_ms, double compute_ms,
                                     double get_ms) {
        paged_decode_profile_.calls++;
        paged_decode_profile_.build_ms += build_ms;
        paged_decode_profile_.alloc_ms += alloc_ms;
        paged_decode_profile_.set_input_ms += set_input_ms;
        paged_decode_profile_.compute_ms += compute_ms;
        paged_decode_profile_.get_ms += get_ms;

        if (env_enabled("FUNASR_PROFILE_PAGED_DECODE") &&
            paged_decode_profile_.calls % 50 == 0) {
            const double calls = static_cast<double>(paged_decode_profile_.calls);
            printf("[GPURunner] paged decode profile calls=%ld avg_ms: "
                   "build=%.3f alloc=%.3f set=%.3f compute=%.3f get=%.3f total=%.3f\n",
                   paged_decode_profile_.calls,
                   paged_decode_profile_.build_ms / calls,
                   paged_decode_profile_.alloc_ms / calls,
                   paged_decode_profile_.set_input_ms / calls,
                   paged_decode_profile_.compute_ms / calls,
                   paged_decode_profile_.get_ms / calls,
                   paged_decode_profile_.avg_total_ms());
        }
    }

    void record_paged_graph_cache_probe(
        int batch,
        int max_blocks,
        int block_size,
        int max_n_kv,
        const std::vector<int>& copy_rows) {
        PagedGraphProbeSignature current;
        current.batch = batch;
        current.max_blocks = max_blocks;
        current.block_size = block_size;
        current.max_n_kv = max_n_kv;
        current.copy_rows = copy_rows;

        paged_decode_profile_.graph_cache_probe_calls++;
        if (has_last_paged_graph_probe_) {
            const bool same_shape =
                last_paged_graph_probe_.batch == current.batch &&
                last_paged_graph_probe_.max_blocks == current.max_blocks &&
                last_paged_graph_probe_.block_size == current.block_size;
            const bool same_params =
                same_shape &&
                last_paged_graph_probe_.max_n_kv == current.max_n_kv;
            const bool same_full_graph =
                same_params &&
                last_paged_graph_probe_.copy_rows == current.copy_rows;
            if (same_shape) {
                paged_decode_profile_.shape_cache_probe_hits++;
            }
            if (same_params) {
                paged_decode_profile_.param_cache_probe_hits++;
            }
            if (same_full_graph) {
                paged_decode_profile_.full_graph_cache_probe_hits++;
            }
        }
        last_paged_graph_probe_ = std::move(current);
        has_last_paged_graph_probe_ = true;
    }

    static bool same_paged_graph_signature(
        const PagedGraphProbeSignature& lhs,
        const PagedGraphProbeSignature& rhs) {
        return lhs.batch == rhs.batch &&
               lhs.max_blocks == rhs.max_blocks &&
               lhs.block_size == rhs.block_size &&
               lhs.max_n_kv == rhs.max_n_kv &&
               lhs.copy_rows == rhs.copy_rows;
    }

    void clear_paged_decode_graph_cache() {
        if (cached_paged_decode_graph_.ctx) {
            ggml_free(cached_paged_decode_graph_.ctx);
        }
        cached_paged_decode_graph_ = CachedPagedDecodeGraph{};
    }

    // ============================================================
    // 统一的 forward 实现
    //
    // cpu_embeds:  非 null 时从 CPU 拷入 (传统路径)
    // gpu_embeds:  非 null 时从 GPU tensor 拷入 (GPU-resident 路径)
    // 两者互斥，gpu_embeds 优先
    // ============================================================
    bool forward_impl(
        const float* cpu_embeds,
        ggml_tensor* gpu_embeds,
        int seq_len,
        int n_past,
        int slot_id,
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

        ggml_tensor* logits = gpu_llm_forward_slot(
            ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
            n_past, slot_id, cfg, kv_cpy_ops
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
        if (allocr == allocr_decode_) {
            clear_paged_decode_graph_cache();
        }
        if (!ggml_gallocr_alloc_graph(allocr, graph)) {
            printf("[GPURunner] Failed to alloc graph\n");
            ggml_free(ctx);
            return false;
        }

        // ===== 5. 设置输入数据 =====
        if (gpu_embeds) {
            // GPU→GPU: 数据已在 GPU 上
            ggml_backend_tensor_copy(gpu_embeds, input);
        } else if (cpu_embeds) {
            // CPU→GPU: 传统路径
            ggml_backend_tensor_set(input, cpu_embeds, 0,
                                     embed_dim * seq_len * sizeof(float));
        }

        // ===== 6. 设置 RoPE position tensor =====
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

#ifdef FUNASR_USE_FLASH_ATTN
        // ===== 6.5 设置 FlashAttention causal mask =====
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }

            ggml_tensor* mask = node->src[3];
            if (!mask || mask->type != GGML_TYPE_F16) {
                continue;
            }

            const int n_kv_mask = static_cast<int>(mask->ne[0]);
            const int n_tokens_mask = static_cast<int>(mask->ne[1]);
            const int n_broadcast = static_cast<int>(mask->ne[2] * mask->ne[3]);
            const size_t n_mask = static_cast<size_t>(n_kv_mask)
                                * n_tokens_mask * n_broadcast;
            flash_mask_data_.assign(n_mask, ggml_fp32_to_fp16(0.0f));

            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int b = 0; b < n_broadcast; b++) {
                const size_t b_offset = static_cast<size_t>(b) * n_kv_mask * n_tokens_mask;
                for (int q_pos = 0; q_pos < n_tokens_mask; q_pos++) {
                    const int max_k_pos = n_past + q_pos;
                    for (int k_pos = max_k_pos + 1; k_pos < n_kv_mask; k_pos++) {
                        flash_mask_data_[b_offset
                            + static_cast<size_t>(q_pos) * n_kv_mask
                            + k_pos] = neg_inf;
                    }
                }
            }

            ggml_backend_tensor_set(mask, flash_mask_data_.data(), 0,
                                    flash_mask_data_.size() * sizeof(ggml_fp16_t));
        }
#endif

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

    bool forward_paged_impl(
        const float* cpu_embeds,
        ggml_tensor* gpu_embeds,
        int seq_len,
        int n_past,
        const std::vector<int>& block_table,
        int block_size,
        std::vector<float>& logits_out
    ) {
        const auto& cfg = gpu_ctx_.config();
        const int embed_dim  = cfg.embedding_length;
        const int vocab_size = cfg.vocab_size;
        const int n_kv = n_past + seq_len;

        if (block_size <= 0 || block_table.empty()) {
            printf("[GPURunner] Invalid paged KV block table\n");
            return false;
        }

        ggml_gallocr_t allocr = seq_len == 1 ? allocr_decode_ : allocr_prefill_;

        pos_data_.resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            pos_data_[i] = n_past + i;
        }

        const int needed_blocks = (n_kv + block_size - 1) / block_size;
        if (needed_blocks > static_cast<int>(block_table.size())) {
            printf("[GPURunner] Paged KV block table too small\n");
            return false;
        }
        for (int block : block_table) {
            const int physical_rows = gpu_ctx_.kv_cache().physical_rows > 0
                ? gpu_ctx_.kv_cache().physical_rows
                : gpu_ctx_.kv_cache().n_ctx;
            if (block < 0 || (block + 1) * block_size > physical_rows) {
                printf("[GPURunner] Paged KV block out of range\n");
                return false;
            }
        }

        size_t ctx_size = 384ULL * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("[GPURunner] Failed to create paged context\n");
            return false;
        }

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, seq_len);
        ggml_set_name(input, "input_embeds");
        ggml_set_input(input);

        std::vector<ggml_tensor*> kv_cpy_ops;
        std::vector<ggml_tensor*> block_table_inputs;
        std::vector<ggml_tensor*> position_inputs;
        ggml_tensor* logits = gpu_llm_forward_paged(
            ctx, input, gpu_ctx_.weights(), gpu_ctx_.kv_cache(),
            n_past, block_table, block_size, cfg, kv_cpy_ops,
            &block_table_inputs, &position_inputs);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 262144, false);
        for (auto* op : kv_cpy_ops) {
            ggml_build_forward_expand(graph, op);
        }
        ggml_build_forward_expand(graph, logits);

        if (!ggml_gallocr_alloc_graph(allocr, graph)) {
            printf("[GPURunner] Failed to alloc paged graph\n");
            ggml_free(ctx);
            return false;
        }

        if (gpu_embeds) {
            ggml_backend_tensor_copy(gpu_embeds, input);
        } else if (cpu_embeds) {
            ggml_backend_tensor_set(input, cpu_embeds, 0,
                                    static_cast<size_t>(embed_dim) * seq_len * sizeof(float));
        }

        for (ggml_tensor* pos_tensor : position_inputs) {
            ggml_backend_tensor_set(pos_tensor, pos_data_.data(), 0,
                                     seq_len * sizeof(int));
        }

        for (ggml_tensor* table : block_table_inputs) {
            ggml_backend_tensor_set(table, block_table.data(), 0,
                static_cast<size_t>(block_table.size()) * sizeof(int));
        }

#ifdef FUNASR_USE_FLASH_ATTN
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_tensor* mask = node->src[3];
            if (!mask || mask->type != GGML_TYPE_F16) {
                continue;
            }
            const int n_kv_mask = static_cast<int>(mask->ne[0]);
            const int n_tokens_mask = static_cast<int>(mask->ne[1]);
            const int n_broadcast = static_cast<int>(mask->ne[2] * mask->ne[3]);
            const size_t n_mask = static_cast<size_t>(n_kv_mask) * n_tokens_mask * n_broadcast;
            flash_mask_data_.assign(n_mask, ggml_fp32_to_fp16(0.0f));
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int b = 0; b < n_broadcast; b++) {
                const size_t b_offset = static_cast<size_t>(b) * n_kv_mask * n_tokens_mask;
                for (int q_pos = 0; q_pos < n_tokens_mask; q_pos++) {
                    const int max_k_pos = n_past + q_pos;
                    for (int k_pos = max_k_pos + 1; k_pos < n_kv_mask; k_pos++) {
                        flash_mask_data_[b_offset
                            + static_cast<size_t>(q_pos) * n_kv_mask
                            + k_pos] = neg_inf;
                    }
                }
            }
            ggml_backend_tensor_set(mask, flash_mask_data_.data(), 0,
                                    flash_mask_data_.size() * sizeof(ggml_fp16_t));
        }
#endif

        if (ggml_backend_graph_compute(gpu_ctx_.backend(), graph) != GGML_STATUS_SUCCESS) {
            printf("[GPURunner] Paged graph compute failed\n");
            ggml_free(ctx);
            return false;
        }

        logits_out.resize(vocab_size);
        int last_pos = seq_len - 1;
        ggml_backend_tensor_get(logits, logits_out.data(),
                                 static_cast<size_t>(last_pos) * vocab_size * sizeof(float),
                                 vocab_size * sizeof(float));

        ggml_free(ctx);
        return true;
    }
};

} // namespace funasr

#endif // FUNASR_COMPUTE_GPU_RUNNER_HPP
