// funasr/compute/gpu_context.hpp
// GPU 资源管理器 — 统一管理 CUDA backend、LLM 权重、KV Cache
//
// 职责:
//   1. 初始化 CUDA backend
//   2. 把 CPU 上的 LLM 权重拷贝到 GPU
//   3. 在 GPU 上分配 KV Cache
//   4. 提供清理/重置接口
//
// 生命周期:
//   GPUContext 创建 → init() → 多次推理 → cleanup()/析构
//   init() 只调用一次（慢），之后的推理只用 forward（快）
//
#ifndef FUNASR_COMPUTE_GPU_CONTEXT_HPP
#define FUNASR_COMPUTE_GPU_CONTEXT_HPP

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#ifdef FUNASR_USE_CUDA
#include <ggml-cuda.h>
#endif

#include "core/config.hpp"
#include "model/weights.hpp"
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace funasr {

// ============================================================
// GPU LLM 单层权重（镜像 CPU 的 LLMLayerWeights）
// tensor 元数据在 CPU，数据在 GPU buffer
// ============================================================
struct GPULLMLayerWeights {
    ggml_tensor* q_proj_w      = nullptr;
    ggml_tensor* k_proj_w      = nullptr;
    ggml_tensor* v_proj_w      = nullptr;
    ggml_tensor* o_proj_w      = nullptr;
    ggml_tensor* q_norm_w      = nullptr;
    ggml_tensor* k_norm_w      = nullptr;
    ggml_tensor* gate_proj_w   = nullptr;
    ggml_tensor* up_proj_w     = nullptr;
    ggml_tensor* down_proj_w   = nullptr;
    ggml_tensor* input_norm_w  = nullptr;
    ggml_tensor* post_attn_norm_w = nullptr;
};

// ============================================================
// GPU LLM 完整权重
// ============================================================
struct GPULLMWeights {
    ggml_tensor* embed_tokens   = nullptr;
    std::vector<GPULLMLayerWeights> layers;
    ggml_tensor* model_norm_w   = nullptr;
    ggml_tensor* lm_head_w      = nullptr;
};

// ============================================================
// GPU KV Cache
// 与 CPU 版的区别: 不需要 pending/commit，用 ggml_cpy 在图内更新
// ============================================================
struct GPUKVCache {
    int n_ctx       = 0;
    int n_past      = 0;
    int n_layers    = 0;
    int kv_dim      = 0;      // n_kv_heads * head_dim = 1024

    ggml_tensor*  k = nullptr;
    ggml_tensor*  v = nullptr;
    ggml_context* ctx     = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    bool initialized = false;
};

// ============================================================
// GPUContext — 统一管理所有 GPU 资源
// ============================================================
class GPUContext {
public:
    GPUContext() = default;
    ~GPUContext() { cleanup(); }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    // ============================================================
    // 完整初始化: backend + weights + kv_cache
    // cpu_weights: CPU 上加载的 LLM 权重
    // cfg: LLM 配置
    // n_ctx: KV Cache 最大上下文长度
    // gpu_id: CUDA 设备 ID
    // ============================================================
    bool init(const LLMWeights& cpu_weights, const LLMConfig& cfg,
              int n_ctx = 2048, int gpu_id = 0)
    {
        printf("\n========== GPUContext Init ==========\n");
        cfg_ = cfg;

        if (!init_backend(gpu_id)) return false;
        if (!load_weights(cpu_weights)) return false;
        if (!init_kv_cache(n_ctx)) return false;

        initialized_ = true;
        printf("========== GPUContext Ready ==========\n\n");
        return true;
    }

    // ============================================================
    // 清空 KV Cache（每句话前调用）
    // ============================================================
    void clear_kv_cache() {
        if (!kv_cache_.initialized) return;
        kv_cache_.n_past = 0;

        size_t n_elements = static_cast<size_t>(kv_cache_.kv_dim)
                          * kv_cache_.n_ctx * kv_cache_.n_layers;
        std::vector<float> zeros(n_elements, 0.0f);
        ggml_backend_tensor_set(kv_cache_.k, zeros.data(), 0, n_elements * sizeof(float));
        ggml_backend_tensor_set(kv_cache_.v, zeros.data(), 0, n_elements * sizeof(float));
    }

    // ============================================================
    // 释放所有 GPU 资源
    // ============================================================
    void cleanup() {
        // KV Cache
        if (kv_cache_.buffer) {
            ggml_backend_buffer_free(kv_cache_.buffer);
            kv_cache_.buffer = nullptr;
        }
        if (kv_cache_.ctx) {
            ggml_free(kv_cache_.ctx);
            kv_cache_.ctx = nullptr;
        }
        kv_cache_.initialized = false;

        // Weights
        if (weights_buffer_) {
            ggml_backend_buffer_free(weights_buffer_);
            weights_buffer_ = nullptr;
        }
        if (weights_ctx_) {
            ggml_free(weights_ctx_);
            weights_ctx_ = nullptr;
        }

        // Backend
        if (backend_) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }

        initialized_ = false;
    }

    // ============================================================
    // 访问接口
    // ============================================================
    bool is_initialized()            const { return initialized_; }
    ggml_backend_t backend()         const { return backend_; }
    GPULLMWeights& weights()               { return weights_; }
    const GPULLMWeights& weights()   const { return weights_; }
    GPUKVCache& kv_cache()                 { return kv_cache_; }
    const GPUKVCache& kv_cache()     const { return kv_cache_; }
    const LLMConfig& config()        const { return cfg_; }

private:
    ggml_backend_t backend_ = nullptr;
    LLMConfig cfg_;

    // Weights on GPU
    GPULLMWeights weights_;
    ggml_context* weights_ctx_ = nullptr;
    ggml_backend_buffer_t weights_buffer_ = nullptr;

    // KV Cache on GPU
    GPUKVCache kv_cache_;

    bool initialized_ = false;

    // ============================================================
    // 初始化 CUDA backend
    // ============================================================
    bool init_backend(int gpu_id) {
#ifdef FUNASR_USE_CUDA
        int device_count = ggml_backend_cuda_get_device_count();
        if (device_count == 0) {
            printf("[GPUContext] No CUDA devices found\n");
            return false;
        }
        if (gpu_id >= device_count) {
            printf("[GPUContext] Invalid device %d (only %d available)\n", gpu_id, device_count);
            return false;
        }

        size_t free_mem, total_mem;
        ggml_backend_cuda_get_device_memory(gpu_id, &free_mem, &total_mem);
        printf("[GPUContext] Device %d: %.2f GB free / %.2f GB total\n",
               gpu_id, free_mem / 1e9, total_mem / 1e9);

        backend_ = ggml_backend_cuda_init(gpu_id);
        if (!backend_) {
            printf("[GPUContext] Failed to init CUDA backend\n");
            return false;
        }

        printf("[GPUContext] CUDA backend: %s\n", ggml_backend_name(backend_));
        return true;
#else
        printf("[GPUContext] CUDA not compiled (build with -DFUNASR_CUDA=ON)\n");
        return false;
#endif
    }

    // ============================================================
    // 把 CPU 权重拷贝到 GPU
    // ============================================================
    bool load_weights(const LLMWeights& cpu) {
        printf("[GPUContext] Loading LLM weights to GPU...\n");
        auto t_start = std::chrono::high_resolution_clock::now();

        int n_layers = cfg_.block_count;
        int n_tensors = 3 + n_layers * 11;  // embed + norm + head + layers

        // 创建 tensor 元数据 context (no_alloc=true)
        size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 10);
        struct ggml_init_params params = { ctx_size, nullptr, true };
        weights_ctx_ = ggml_init(params);
        if (!weights_ctx_) return false;

        // 创建 GPU tensor 元数据（shape/type 与 CPU 一致）
        weights_.embed_tokens = ggml_new_tensor_2d(weights_ctx_,
            cpu.embed_tokens->type, cpu.embed_tokens->ne[0], cpu.embed_tokens->ne[1]);
        weights_.model_norm_w = ggml_new_tensor_1d(weights_ctx_,
            cpu.model_norm_w->type, cpu.model_norm_w->ne[0]);
        weights_.lm_head_w = ggml_new_tensor_2d(weights_ctx_,
            cpu.lm_head_w->type, cpu.lm_head_w->ne[0], cpu.lm_head_w->ne[1]);

        weights_.layers.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            auto& gl = weights_.layers[i];
            auto& cl = cpu.layers[i];

            gl.q_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.q_proj_w->type,
                cl.q_proj_w->ne[0], cl.q_proj_w->ne[1]);
            gl.k_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.k_proj_w->type,
                cl.k_proj_w->ne[0], cl.k_proj_w->ne[1]);
            gl.v_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.v_proj_w->type,
                cl.v_proj_w->ne[0], cl.v_proj_w->ne[1]);
            gl.o_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.o_proj_w->type,
                cl.o_proj_w->ne[0], cl.o_proj_w->ne[1]);
            gl.q_norm_w = ggml_new_tensor_1d(weights_ctx_, cl.q_norm_w->type,
                cl.q_norm_w->ne[0]);
            gl.k_norm_w = ggml_new_tensor_1d(weights_ctx_, cl.k_norm_w->type,
                cl.k_norm_w->ne[0]);
            gl.gate_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.gate_proj_w->type,
                cl.gate_proj_w->ne[0], cl.gate_proj_w->ne[1]);
            gl.up_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.up_proj_w->type,
                cl.up_proj_w->ne[0], cl.up_proj_w->ne[1]);
            gl.down_proj_w = ggml_new_tensor_2d(weights_ctx_, cl.down_proj_w->type,
                cl.down_proj_w->ne[0], cl.down_proj_w->ne[1]);
            gl.input_norm_w = ggml_new_tensor_1d(weights_ctx_, cl.input_norm_w->type,
                cl.input_norm_w->ne[0]);
            gl.post_attn_norm_w = ggml_new_tensor_1d(weights_ctx_, cl.post_attn_norm_w->type,
                cl.post_attn_norm_w->ne[0]);
        }

        // 在 GPU 上分配 buffer
        weights_buffer_ = ggml_backend_alloc_ctx_tensors(weights_ctx_, backend_);
        if (!weights_buffer_) {
            printf("[GPUContext] Failed to alloc GPU buffer for weights\n");
            return false;
        }

        printf("[GPUContext] GPU weights buffer: %.2f MB\n",
               ggml_backend_buffer_get_size(weights_buffer_) / 1e6);

        // CPU → GPU 数据拷贝
        auto copy_tensor = [](ggml_tensor* dst, const ggml_tensor* src) {
            ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(dst));
        };

        copy_tensor(weights_.embed_tokens, cpu.embed_tokens);
        copy_tensor(weights_.model_norm_w, cpu.model_norm_w);
        copy_tensor(weights_.lm_head_w,    cpu.lm_head_w);

        for (int i = 0; i < n_layers; i++) {
            auto& gl = weights_.layers[i];
            auto& cl = cpu.layers[i];

            copy_tensor(gl.q_proj_w,        cl.q_proj_w);
            copy_tensor(gl.k_proj_w,        cl.k_proj_w);
            copy_tensor(gl.v_proj_w,        cl.v_proj_w);
            copy_tensor(gl.o_proj_w,        cl.o_proj_w);
            copy_tensor(gl.q_norm_w,        cl.q_norm_w);
            copy_tensor(gl.k_norm_w,        cl.k_norm_w);
            copy_tensor(gl.gate_proj_w,     cl.gate_proj_w);
            copy_tensor(gl.up_proj_w,       cl.up_proj_w);
            copy_tensor(gl.down_proj_w,     cl.down_proj_w);
            copy_tensor(gl.input_norm_w,    cl.input_norm_w);
            copy_tensor(gl.post_attn_norm_w, cl.post_attn_norm_w);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        printf("[GPUContext] Weights loaded: %ld ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        return true;
    }

    // ============================================================
    // 初始化 GPU KV Cache
    // ============================================================
    bool init_kv_cache(int n_ctx) {
        printf("[GPUContext] Init GPU KV Cache (n_ctx=%d)...\n", n_ctx);

        kv_cache_.n_ctx    = n_ctx;
        kv_cache_.n_past   = 0;
        kv_cache_.n_layers = cfg_.block_count;
        kv_cache_.kv_dim   = cfg_.kv_dim();

        size_t n_elements = static_cast<size_t>(kv_cache_.kv_dim)
                          * n_ctx * kv_cache_.n_layers;

        // 创建 tensor 元数据
        size_t ctx_size = ggml_tensor_overhead() * 4;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        kv_cache_.ctx = ggml_init(params);
        if (!kv_cache_.ctx) return false;

        kv_cache_.k = ggml_new_tensor_1d(kv_cache_.ctx, GGML_TYPE_F32, n_elements);
        kv_cache_.v = ggml_new_tensor_1d(kv_cache_.ctx, GGML_TYPE_F32, n_elements);

        // GPU 分配
        kv_cache_.buffer = ggml_backend_alloc_ctx_tensors(kv_cache_.ctx, backend_);
        if (!kv_cache_.buffer) {
            printf("[GPUContext] Failed to alloc GPU KV buffer\n");
            return false;
        }

        // 清零
        std::vector<float> zeros(n_elements, 0.0f);
        ggml_backend_tensor_set(kv_cache_.k, zeros.data(), 0, n_elements * sizeof(float));
        ggml_backend_tensor_set(kv_cache_.v, zeros.data(), 0, n_elements * sizeof(float));

        kv_cache_.initialized = true;
        printf("[GPUContext] KV Cache: K=%.1f MB, V=%.1f MB\n",
               ggml_nbytes(kv_cache_.k) / 1e6, ggml_nbytes(kv_cache_.v) / 1e6);
        return true;
    }
};

} // namespace funasr

#endif // FUNASR_COMPUTE_GPU_CONTEXT_HPP