# Owning GPU Prompt Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Unified Scheduler 增加可跨 scheduler iteration 持有的 Prompt Embedding，并用现有 Paged Prefill Runner 验证单请求分块 Prefill 与一次性 Prefill 等价。

**Architecture:** 新增后端无关的 `GPUEmbeddingPool`，每个 in-flight request 获得带 generation 的稳定 tensor slot；Pipeline 在下一次 Encoder/Adaptor 覆盖 staging 之前，把 `[prefix?][audio][suffix]` 复制进 slot。Chunk executor 只创建 slot 的 tensor view，并复用现有 `forward_gpu_tensor_paged(view, chunk_len, n_past, ...)`，本阶段不实现多请求 Packed Mixed Runner，也不改变默认离线路径。

**Tech Stack:** C++17、GGML backend/tensor API、CUDA/CPU backend、现有 Paged KV Runner、手写单元测试、CMake

---

## 范围和文件

- Create: `compute/gpu_embedding_pool.hpp`
  - 定义 `GPUEmbeddingHandle` 和持有 GGML backend tensor 的固定 slot pool。
- Create: `compute/gpu_embedding_pool.cpp`
  - 实现 acquire/release、generation 校验、host 写入、tensor copy 和 chunk view。
- Create: `test/test_gpu_embedding_pool.cpp`
  - 使用 GGML CPU backend 验证资源生命周期和数据切片，不依赖真实 GPU。
- Modify: `CMakeLists.txt`
  - 将 pool 加入 `funasr_compute`，新增测试 target。
- Modify: `pipeline/pipeline.hpp`, `pipeline/pipeline.cpp`
  - 增加 owning prompt handle、Prompt 组装、chunk prefill 和显式 release API。
- Modify: `pipeline/recognizer.hpp`
  - 暴露 Pipeline 新 API 给离线 scheduler。
- Modify: `pipeline/offline_batching.hpp`, `pipeline/offline_batching.cpp`
  - 增加纯 chunk range helper；在显式配置开启时用 sequential chunked prefill 做正确性验证。
- Modify: `test/test_offline_scheduler.cpp`
  - 覆盖 chunk range、prefix offset 和尾 chunk。
- Modify: `test/test_offline_batching.cpp`
  - 增加实验参数解析，不改变默认值。

## 明确不做

- 不把多个 request 打成一个 packed tensor。
- 不实现 varlen/paged prefill attention kernel。
- 不在一个 iteration 内混合 Decode 和 Prefill graph。
- 不把 Unified 模式设为默认。
- 不优化中间 Prompt chunk 的 LM Head；本阶段 Runner 仍计算 logits，但调用方只保留最后一个 chunk 的 logits。

### Task 1: Stable Backend Embedding Pool

**Files:**
- Create: `compute/gpu_embedding_pool.hpp`
- Create: `compute/gpu_embedding_pool.cpp`
- Create: `test/test_gpu_embedding_pool.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 写失败测试，固定 slot 生命周期和 stale handle 语义**

测试必须使用 `ggml_backend_cpu_init()`，避免要求 CI 具有 CUDA：

```cpp
void test_pool_capacity_and_stale_handles() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    {
        funasr::GPUEmbeddingPool pool(backend, 2);

        auto a = pool.acquire(4, 3);
        auto b = pool.acquire(2, 3);
        auto blocked = pool.acquire(1, 3);
        TEST_ASSERT(a.valid() && b.valid(), "two slots acquired");
        TEST_ASSERT(!blocked.valid(), "third acquire is backpressured");

        TEST_ASSERT(pool.release(a), "first handle released");
        auto reused = pool.acquire(3, 3);
        TEST_EQ(reused.slot, a.slot, "slot is reused");
        TEST_ASSERT(reused.generation != a.generation,
                    "generation changes on reuse");
        TEST_ASSERT(!pool.release(a), "stale handle cannot release new owner");
    }

    ggml_backend_free(backend);
}
```

第二个测试写入 `[token_count, embedding_dim]` 的已知 float，读取一个非零 offset 的 view，验证 token 顺序没有转置：

```cpp
std::vector<float> values = {
    0, 1, 2,
    10, 11, 12,
    20, 21, 22,
    30, 31, 32,
};
TEST_ASSERT(pool.set_host(handle, 0, values.data(), 4), "host data set");
ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
ggml_context* view_ctx = ggml_init(params);
ggml_tensor* view = pool.view(view_ctx, handle, 1, 2);
std::vector<float> slice(6);
ggml_backend_tensor_get(view, slice.data(), 0,
                        slice.size() * sizeof(float));
TEST_EQ(slice[0], 10.0f, "slice begins at token one");
TEST_EQ(slice[5], 22.0f, "slice ends at token two");
ggml_free(view_ctx);
```

- [ ] **Step 2: 运行 RED**

```bash
cmake -S . -B build-cuda -DFUNASR_USE_CUDA=ON -DFUNASR_BUILD_TESTS=ON
cmake --build build-cuda --target test_gpu_embedding_pool -j 8
```

Expected: FAIL，缺少 `GPUEmbeddingPool` 和 target。

- [ ] **Step 3: 实现 handle 和 pool**

公开 API 固定为：

```cpp
struct GPUEmbeddingHandle {
    int slot = -1;
    uint64_t generation = 0;
    int token_count = 0;
    int embedding_dim = 0;

    bool valid() const {
        return slot >= 0 && generation != 0 &&
               token_count > 0 && embedding_dim > 0;
    }
};

class GPUEmbeddingPool {
public:
    GPUEmbeddingPool(ggml_backend_t backend, int max_slots);
    ~GPUEmbeddingPool();
    GPUEmbeddingPool(const GPUEmbeddingPool&) = delete;
    GPUEmbeddingPool& operator=(const GPUEmbeddingPool&) = delete;

    GPUEmbeddingHandle acquire(int token_count, int embedding_dim);
    bool release(const GPUEmbeddingHandle& handle);
    bool owns(const GPUEmbeddingHandle& handle) const;
    int free_count() const;
    int capacity() const;
    size_t reserved_bytes() const;

    bool set_host(const GPUEmbeddingHandle& handle,
                  int token_offset,
                  const float* source,
                  int token_count);
    bool copy_tensor(const GPUEmbeddingHandle& handle,
                     int destination_token_offset,
                     ggml_tensor* source,
                     int source_token_offset,
                     int token_count);
    ggml_tensor* view(ggml_context* ctx,
                      const GPUEmbeddingHandle& handle,
                      int token_offset,
                      int token_count) const;
};
```

每个 slot 保留自己的 `ggml_context*`、`ggml_backend_buffer_t` 和二维 F32 tensor。释放 slot 只释放所有权，允许同维度且容量足够时复用 backing buffer；pool 析构时按 `buffer -> context` 顺序释放。`acquire()` 分配 backing 失败时必须把 slot 放回 free queue。

所有 range 计算先转为 `int64_t`，并验证：

```cpp
token_offset >= 0
token_count > 0
token_offset + token_count <= handle.token_count
```

`view()` 使用 `ggml_view_2d()` 和 `ggml_backend_view_init()`，stride 为 `embedding_dim * sizeof(float)`，byte offset 为 `token_offset * embedding_dim * sizeof(float)`。

- [ ] **Step 4: 运行 GREEN 和回归**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool test_unified_scheduler test_offline_scheduler -j 8
./build-cuda/test_gpu_embedding_pool
./build-cuda/test_unified_scheduler
./build-cuda/test_offline_scheduler
git diff --check
```

Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add CMakeLists.txt compute/gpu_embedding_pool.hpp compute/gpu_embedding_pool.cpp test/test_gpu_embedding_pool.cpp
git commit -m "feat: add owning GPU embedding pool"
```

### Task 2: Build an Owning Prompt Tensor Before Frontend Reuse

**Files:**
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_gpu_embedding_pool.cpp`

- [ ] **Step 1: 写失败测试，定义 Prompt layout 计算**

先增加与 GPU 无关的布局 helper，避免通过集成测试猜 offset：

```cpp
struct GPUPromptLayout {
    int cached_prefix_tokens = 0;
    int stored_prefix_tokens = 0;
    int audio_tokens = 0;
    int suffix_tokens = 0;
    int prompt_tokens = 0;
    int stored_tokens = 0;
    bool valid = false;
};

GPUPromptLayout make_gpu_prompt_layout(int prefix_tokens,
                                       int cached_prefix_tokens,
                                       int audio_tokens,
                                       int suffix_tokens);
```

测试覆盖：

```text
无 cache: prefix=11, cached=0, audio=499, suffix=12
          stored=522, prompt=522
命中 cache: prefix=11, cached=11, audio=499, suffix=12
            stored=511, prompt=522
部分/超长 cache: invalid
```

- [ ] **Step 2: 运行 RED**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool -j 8
```

Expected: FAIL，helper 尚不存在。

- [ ] **Step 3: 增加 owning Prompt API**

在 `pipeline/pipeline.hpp` 定义：

```cpp
struct PreparedGPUPrompt {
    GPUEmbeddingHandle embeddings;
    int request_id = -1;
    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int audio_frames = 0;
    float encoder_ms = 0.0f;
    bool ok = false;
};
```

Pipeline/Recognizer 增加：

```cpp
PreparedGPUPrompt prepare_prompt_gpu(const AudioSpan& audio,
                                     int request_id,
                                     int cached_prefix_tokens,
                                     const InferenceConfig& config);
bool release_prompt_gpu(const GPUEmbeddingHandle& handle);
int free_prompt_embedding_slots() const;
```

`Pipeline::init_gpu()` 使用 `n_slots` 创建 pool；析构时先销毁 pool，再销毁 `GPUContext` backend。

`prepare_prompt_gpu()` 必须在一次调用内完成：

1. 调用现有 `prepare_audio_gpu()` 得到临时 adaptor tensor。
2. 计算 prefix/suffix IDs 和 `GPUPromptLayout`。
3. acquire 稳定 slot；slot 不足返回 `ok=false`，不再次调用 frontend。
4. 未命中 cache 时把 prefix host embeddings 写到 slot 起点。
5. 用 `copy_tensor()` 立即复制 audio embeddings。
6. 写入 suffix embeddings。
7. 任一步失败都 release slot；成功后才返回 handle。

不要让 `PreparedGPUPrompt` 自动析构释放；request 状态可能 move，所有权由 scheduler 显式 release，stale generation 负责防 double release。

- [ ] **Step 4: 构建并运行 CPU 测试和现有回归**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool test_offline_scheduler -j 8
./build-cuda/test_gpu_embedding_pool
./build-cuda/test_offline_scheduler
git diff --check
```

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline.hpp pipeline/pipeline.cpp pipeline/recognizer.hpp test/test_gpu_embedding_pool.cpp
git commit -m "feat: retain prompt embeddings across scheduler steps"
```

### Task 3: Execute One Paged Prompt Chunk From an Owned Slot

**Files:**
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_gpu_embedding_pool.cpp`

- [ ] **Step 1: 写失败测试，定义 absolute/local chunk range**

增加纯 helper：

```cpp
struct GPUPromptChunkRange {
    int absolute_offset = 0;
    int local_offset = 0;
    int token_count = 0;
    bool produces_logits = false;
    bool valid = false;
};

GPUPromptChunkRange make_gpu_prompt_chunk_range(
    const GPUPromptLayout& layout,
    int absolute_offset,
    int requested_tokens);
```

命中 11-token prefix cache、prompt=522 时测试：

```text
offset=11,count=128  -> local=0,count=128,logits=false
offset=512,count=128 -> local=501,count=10,logits=true
offset=10            -> invalid（不能重算 cache）
offset=522           -> invalid（无新 token）
```

- [ ] **Step 2: 运行 RED**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool -j 8
```

- [ ] **Step 3: 增加 chunk executor**

Pipeline/Recognizer 增加：

```cpp
GPUPrefillState gpu_prefill_prompt_chunk_paged(
    const PreparedGPUPrompt& prepared,
    int absolute_token_offset,
    int requested_tokens,
    const std::vector<int>& block_table,
    int block_size,
    const InferenceConfig& config);
```

执行流程：

1. 校验 prepared handle 仍归 pool 所有。
2. 用 helper 把 absolute offset 转为 slot local offset，并裁剪最后一个 chunk。
3. 创建短生命周期 metadata context，调用 `pool.view()`。
4. 调用现有：

```cpp
gpu_runner_->forward_gpu_tensor_paged(
    chunk_view,
    range.token_count,
    range.absolute_offset,
    block_table,
    block_size,
    state.logits);
```

5. 成功时 `state.n_past = absolute_offset + token_count`。
6. 中间 chunk 清空 logits；只有 `produces_logits=true` 的最后一个 chunk 保留 logits。
7. 失败不 release embedding、不提交 token 进度，交给 scheduler 决定重试或终止。

- [ ] **Step 4: 运行测试和回归**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool test_unified_scheduler test_offline_scheduler -j 8
./build-cuda/test_gpu_embedding_pool
./build-cuda/test_unified_scheduler
./build-cuda/test_offline_scheduler
git diff --check
```

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline.hpp pipeline/pipeline.cpp pipeline/recognizer.hpp test/test_gpu_embedding_pool.cpp
git commit -m "feat: execute paged prompt chunks from owned embeddings"
```

### Task 4: Experimental Sequential Chunked Prefill Equivalence

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_scheduler.cpp`
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: 写失败测试，定义 chunk 列表**

增加：

```cpp
std::vector<std::pair<int, int>> plan_prefill_chunks(
    int first_uncached_token,
    int prompt_tokens,
    int max_chunk_tokens);
```

测试：

```text
(0,522,128)  -> (0,128),(128,128),(256,128),(384,128),(512,10)
(11,522,256) -> (11,256),(267,255)
invalid 输入 -> 空列表
```

- [ ] **Step 2: 增加显式实验配置**

在 `OfflineBatchConfig` 增加：

```cpp
int prefill_chunk_tokens = 0; // 0 keeps the existing one-shot path
```

`test_offline_batching` 增加 `--prefill-chunk-tokens <n>`；默认 0，README 和正式 CLI 暂不暴露。

- [ ] **Step 3: 接入 sequential chunked prefill**

仅在下列条件同时成立时走新路径：

```text
use_gpu
use_paged_kv
enable_dynamic_kv_blocks
prefill_chunk_tokens > 0
```

admission 中：

1. frontend 完成后得到 `PreparedGPUPrompt`，而不是立即执行整段 prefill。
2. 从 `cached_prefix_tokens` 开始遍历 `plan_prefill_chunks()`。
3. 每个 chunk 执行前按 `offset + count` 事务式 append 所需 KV block。
4. 调用 `gpu_prefill_prompt_chunk_paged()`；失败时释放 embedding 和 KV，request Failed。
5. 只有最后一个 chunk 读取 logits 并生成第一个 token。
6. Prompt 全部完成后立即 release embedding slot。
7. 现有 one-shot 路径保持原样。

本任务仍在一次 admission 调用内顺序执行 chunks；它只验证 tensor 生命周期、position、Paged KV 和输出正确性。跨 request 的 scheduler interleave 留给 Packed Mixed Runner 阶段。

- [ ] **Step 4: 单元测试和小样本 GPU 等价验证**

```bash
cmake --build build-cuda --target test_offline_scheduler test_offline_batching -j 8
./build-cuda/test_offline_scheduler

FUNASR_PAGED_KV_WRITE_OP=1 \
./build-cuda/test_offline_batching \
  FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 1 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220 --prefill-chunk-tokens 128
```

再跑相同命令但 `--prefill-chunk-tokens 0`。比较前 24 chunks 时必须满足：

```text
ok 数一致
逐 chunk token/text 完全一致
无 Paged KV ownership error
embedding pool 最终全部归还
```

如果当前测试程序仍不支持 `--max-chunks`，不要伪造该参数；使用现有 benchmark suite 的截断音频或完整输入。

- [ ] **Step 5: 提交**

```bash
git add pipeline/offline_batching.hpp pipeline/offline_batching.cpp test/test_offline_scheduler.cpp test/test_offline_batching.cpp
git commit -m "feat: validate sequential chunked paged prefill"
```

## 阶段完成门槛

- 所有 CPU 单元测试通过，pool stale handle/double release 无所有权错误。
- Chunk range 永远不重算 cached prefix，也不跨 prompt 尾部。
- Owned embedding 在下一次 frontend 调用后仍保持数据不变。
- One-shot 与 128-token sequential chunked prefill 的逐 chunk 输出一致。
- 默认 `prefill_chunk_tokens=0`，现有 benchmark 行为和性能不变。
- 只有达到这些门槛后，才进入 Packed Mixed Runner 和 Varlen Paged Attention。
