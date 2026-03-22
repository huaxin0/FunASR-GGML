// KV Cache — RAII 管理，用于 LLM 自回归解码
//
// 内存布局: 一个大的 1D F32 tensor
//   K: [n_layers × n_ctx × kv_dim] (kv_dim = n_kv_heads × head_dim = 1024)
//   V: 同上
//
// 工作流:
//   1. init() 分配内存
//   2. forward 时 set_pending() 记录当前步计算出的 K/V tensor 指针
//   3. run_graph() 执行计算图
//   4. commit() 把 pending 数据 memcpy 到 cache (必须在 ggml_free 之前!)
//   5. 下一步 forward 时从 cache 读取历史 K/V
//
#ifndef FUNASR_COMPUTE_KV_CACHE_HPP
#define FUNASR_COMPUTE_KV_CACHE_HPP

#include <ggml.h>
#include "core/config.hpp"
#include <vector>
#include <cstring>
#include <cstdio>

namespace funasr {

class KVCache {
public:
    KVCache() = default;

    ~KVCache() { free(); }

    // 禁用拷贝
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    // ============================================================
    // 初始化
    // ============================================================
    bool init(const LLMConfig& cfg, int n_ctx) {
        n_ctx_    = n_ctx;
        n_past_   = 0;
        n_layers_ = cfg.block_count;
        kv_dim_   = cfg.kv_dim();  // n_kv_heads * head_dim = 1024

        // 清空 pending
        pending_.resize(n_layers_);
        for (auto& p : pending_) { p.k = nullptr; p.v = nullptr; }

        // 分配内存
        size_t n_elements = static_cast<size_t>(kv_dim_) * n_ctx_ * n_layers_;
        size_t tensor_size = n_elements * sizeof(float);
        size_t mem_size = 2 * tensor_size + 2 * ggml_tensor_overhead() + 4096;

        buf_.resize(mem_size);

        struct ggml_init_params params = {
            mem_size,
            buf_.data(),
            false,
        };

        ctx_ = ggml_init(params);
        if (!ctx_) {
            printf("[KVCache] Failed to init context\n");
            return false;
        }

        k_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, n_elements);
        v_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, n_elements);

        memset(k_->data, 0, ggml_nbytes(k_));
        memset(v_->data, 0, ggml_nbytes(v_));

        initialized_ = true;
        return true;
    }

    // ============================================================
    // 记录当前步的 K/V (forward 时调用)
    // 这些指针指向计算图中的 tensor，必须在 graph compute 后、
    // ggml_free(ctx_compute) 前调用 commit()
    // ============================================================
    void set_pending(int layer_idx, ggml_tensor* k, ggml_tensor* v) {
        if (layer_idx >= 0 && layer_idx < n_layers_) {
            pending_[layer_idx].k = k;
            pending_[layer_idx].v = v;
        }
    }

    // ============================================================
    // 提交: 把 pending K/V memcpy 到 cache
    // 必须在 ggml_graph_compute 之后、ggml_free(ctx_compute) 之前调用
    // ============================================================
    void commit(int n_past, int seq_len) {
        for (int i = 0; i < n_layers_; i++) {
            if (pending_[i].k && pending_[i].v) {
                size_t layer_offset = static_cast<size_t>(i) * n_ctx_ * kv_dim_;
                size_t pos_offset   = static_cast<size_t>(n_past) * kv_dim_;
                size_t copy_size    = static_cast<size_t>(seq_len) * kv_dim_ * sizeof(float);

                float* dst_k = (float*)k_->data + layer_offset + pos_offset;
                float* dst_v = (float*)v_->data + layer_offset + pos_offset;

                memcpy(dst_k, pending_[i].k->data, copy_size);
                memcpy(dst_v, pending_[i].v->data, copy_size);

                pending_[i].k = nullptr;
                pending_[i].v = nullptr;
            }
        }
    }

    // ============================================================
    // 清空
    // ============================================================
    void clear() {
        n_past_ = 0;
        if (k_ && k_->data) memset(k_->data, 0, ggml_nbytes(k_));
        if (v_ && v_->data) memset(v_->data, 0, ggml_nbytes(v_));
        for (auto& p : pending_) { p.k = nullptr; p.v = nullptr; }
    }

    // ============================================================
    // 释放
    // ============================================================
    void free() {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        k_ = nullptr;
        v_ = nullptr;
        buf_.clear();
        pending_.clear();
        initialized_ = false;
        n_past_ = 0;
    }

    // ============================================================
    // 查询
    // ============================================================
    int  n_past()         const { return n_past_; }
    int  n_ctx()          const { return n_ctx_; }
    int  n_layers()       const { return n_layers_; }
    int  kv_dim()         const { return kv_dim_; }
    bool is_initialized() const { return initialized_; }

    void set_n_past(int n) { n_past_ = n; }

    // 供 forward 函数访问底层 tensor
    ggml_tensor* k_tensor() { return k_; }
    ggml_tensor* v_tensor() { return v_; }

private:
    ggml_context* ctx_ = nullptr;
    ggml_tensor*  k_   = nullptr;
    ggml_tensor*  v_   = nullptr;
    std::vector<uint8_t> buf_;

    int n_ctx_    = 0;
    int n_past_   = 0;
    int n_layers_ = 0;
    int kv_dim_   = 0;

    struct Pending { ggml_tensor* k; ggml_tensor* v; };
    std::vector<Pending> pending_;

    bool initialized_ = false;
};

} // namespace funasr

#endif // FUNASR_COMPUTE_KV_CACHE_HPP