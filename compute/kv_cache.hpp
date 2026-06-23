// KV Cache — RAII 管理，用于 LLM 自回归解码
//
// 内存布局: 一个大的 1D F16 tensor
//   K: [n_layers × n_ctx × kv_dim] (kv_dim = n_kv_heads × head_dim = 1024)
//   V: 同上
//
// 工作流:
//   1. init() 分配内存
//   2. forward 时创建当前 K/V 到 cache slot 的 ggml_cpy 节点
//   3. 调用方把 cpy 节点加入 graph，再加入 logits 节点
//   4. run_graph() 执行计算图，K/V 在图内写入 cache
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

        // 分配内存
        const ggml_type kv_type = GGML_TYPE_F16;
        size_t n_elements = static_cast<size_t>(kv_dim_) * n_ctx_ * n_layers_;
        size_t tensor_size = ggml_row_size(kv_type, kv_dim_) * n_ctx_ * n_layers_;
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

        k_ = ggml_new_tensor_1d(ctx_, kv_type, n_elements);
        v_ = ggml_new_tensor_1d(ctx_, kv_type, n_elements);

        memset(k_->data, 0, ggml_nbytes(k_));
        memset(v_->data, 0, ggml_nbytes(v_));

        initialized_ = true;
        printf("[KVCache] CPU KV Cache: K=%.1f MB, V=%.1f MB (%s)\n",
               ggml_nbytes(k_) / 1e6, ggml_nbytes(v_) / 1e6,
               ggml_type_name(k_->type));
        return true;
    }

    // ============================================================
    // 清空
    // ============================================================
    void clear() {
        n_past_ = 0;
        if (k_ && k_->data) memset(k_->data, 0, ggml_nbytes(k_));
        if (v_ && v_->data) memset(v_->data, 0, ggml_nbytes(v_));
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

    bool initialized_ = false;
};

} // namespace funasr

#endif // FUNASR_COMPUTE_KV_CACHE_HPP
