# Packed Mixed Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Unified Scheduler 的 Prompt chunk 与 Decode token 组装为无 padding packed batch，在一次 Transformer forward 中共享 Q8 projection/MLP，并以 Paged KV metadata 隔离每个 request。

**Architecture:** 第一版先实现可验证的 flattened-query backend：Prompt token 排在前、Decode token 排在后，每个 packed token 携带自己的 absolute position、request row、KV length 和 block table；Transformer projection/MLP 对 `[hidden, total_tokens]` 一次计算，Paged Attention 暂时把每个 token 当作 `q_len=1` query 批量执行。稳定后再把 Prompt query 替换为真正的 varlen Paged Prefill kernel，并保留现有 Decode Fast Path。

**Tech Stack:** C++17、GGML graph/backend API、CUDA、现有 Paged KV/Embedding Pool、手写单元测试、Nsight Systems/Compute

---

## 文件边界

- Create: `pipeline/mixed_batch.hpp`, `pipeline/mixed_batch.cpp`
  - 纯 CPU 的 plan 校验、packed row 布局和 metadata 展开。
- Create: `test/test_mixed_batch.cpp`
  - 不依赖 GPU，覆盖 request 隔离、selected rows、position 和 block-table 展开。
- Create: `compute/gpu_mixed_runner.hpp`, `compute/gpu_mixed_runner.cpp`
  - 持有 packed prompt staging tensor，执行 mixed GGML graph。
- Modify: `compute/llm_ops_gpu.hpp`, `compute/llm_ops_gpu.cpp`
  - 增加 packed paged Transformer forward 和 selective LM Head。
- Modify: `compute/gpu_runner.hpp`
  - 暴露 mixed runner 所需的 backend/config/KV/weights 只读访问，不复制所有权。
- Modify: `third_party/ggml/include/ggml.h`, `third_party/ggml/src/ggml.c`
  - 为 Paged Attention 增加显式 KV-write dependency 输入。
- Modify: `third_party/ggml/src/ggml-cuda/paged-attn.cu`
  - 校验 dependency tensor，并保持同一 CUDA stream 上 write-before-attention。
- Modify: `pipeline/offline_batching.hpp`, `pipeline/offline_batching.cpp`
  - 增加 experimental unified 模式及 prepare/execute/commit 循环。
- Modify: `test/test_offline_batching.cpp`, `test/test_offline_scheduler.cpp`
  - CLI、回收、失败回滚及端到端审计。

## 正确性口径

必须分三层验证，不能只看最终文本：

```text
结构正确性：position / request row / block table / selected row 精确一致
资源正确性：final_free == capacity, ownership_errors == 0, embedding 全归还
数值正确性：同配置重复执行 token hash 一致；与 one-shot 记录 token mismatch、
             logits max_abs/max_rel；完整 ASR 记录 CER/WER，不隐瞒 shape drift
```

Unified 模式成为默认前仍要求质量不退化。由于 Q8 CUDA kernel 对 shape 敏感，
跨 execution shape 的 bitwise token equality 作为审计指标保留，但不再单独等同于
KV 逻辑正确性。

### Task 1: Pure Packed Metadata Builder

**Files:**
- Create: `pipeline/mixed_batch.hpp`
- Create: `pipeline/mixed_batch.cpp`
- Create: `test/test_mixed_batch.cpp`
- Modify: `CMakeLists.txt`

- [x] **Step 1: 写 RED 测试固定 packed 顺序和 metadata**

定义两个 Prompt sequence 和两个 Decode sequence，输入 scheduler plan 的顺序故意
交错。期望 packer 输出 Prompt-first、Decode-second，同时保持每类内部稳定顺序：

```cpp
TEST_EQ(packed.total_tokens, 7, "packed token count");
TEST_EQ(packed.prefill_tokens, 5, "prompt rows first");
TEST_EQ(packed.decode_tokens, 2, "decode rows last");
TEST_EQ(packed.positions, std::vector<int>({11, 12, 13, 20, 21, 522, 601}),
        "absolute positions");
TEST_EQ(packed.sequence_rows, std::vector<int>({0, 0, 0, 1, 1, 2, 3}),
        "each token selects exactly one request row");
TEST_EQ(packed.kv_lens, std::vector<int>({12, 13, 14, 21, 22, 523, 602}),
        "causal visible prefix ends at current token");
TEST_EQ(packed.selected_rows, std::vector<int>({4, 5, 6}),
        "last prompt row and both decode rows produce logits");
```

- [x] **Step 2: 运行 RED**

```bash
cmake --build build-cuda --target test_mixed_batch -j 8
```

Expected: target/type 尚不存在。

- [x] **Step 3: 实现严格校验的 metadata builder**

公开类型：

```cpp
struct PackedSequenceSource {
    ScheduledSequence scheduled;
    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int decode_token_id = -1;
    std::vector<int> block_table;
};

struct PackedMixedMetadata {
    std::vector<PackedSequenceSource> sequences;
    std::vector<int> positions;
    std::vector<int> sequence_rows;
    std::vector<int> kv_lens;
    std::vector<int> selected_rows;
    std::vector<int> expanded_block_table;
    int max_blocks = 0;
    int total_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    bool valid = false;
};
```

`expanded_block_table` 使用 `[max_blocks, total_tokens]`，每个 token row 复制所属
request 的 table。所有加法用 `int64_t`；任意 position 没有预留 block、Decode
sequence 不是单 token、或 selected row 重复时返回 invalid。

- [x] **Step 4: GREEN 与回归**

```bash
cmake --build build-cuda --target test_mixed_batch test_unified_scheduler test_offline_scheduler -j 8
./build-cuda/test_mixed_batch
./build-cuda/test_unified_scheduler
./build-cuda/test_offline_scheduler
git diff --check
```

### Task 2: Explicit KV Write Dependency

**Files:**
- Modify: `third_party/ggml/include/ggml.h`
- Modify: `third_party/ggml/src/ggml.c`
- Modify: `third_party/ggml/src/ggml-cuda/paged-attn.cu`
- Modify: `compute/llm_ops_gpu.cpp`
- Test: `test/test_mixed_batch.cpp`

- [x] **Step 1: 写 graph-structure RED 测试**

构造最小 GGML graph，要求 `GGML_OP_PAGED_ATTN_EXT` 的 dependency source 指向
同层 `GGML_OP_PAGED_KV_WRITE_EXT`，且 graph traversal 只构建 logits 时仍包含 write。

- [x] **Step 2: 增加 dependency-aware API**

```cpp
GGML_API struct ggml_tensor * ggml_paged_attn_ext_v_dep(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k_cache,
    struct ggml_tensor * v_cache,
    struct ggml_tensor * block_table,
    struct ggml_tensor * kv_lens,
    struct ggml_tensor * kv_write_dependency,
    float scale,
    int block_size,
    int max_n_kv,
    int n_head_kv);
```

CUDA kernel 不读取 dependency 的数值；依赖边用于 GGML 拓扑排序，保证同 stream
先完成当前层所有 packed K/V 写入，再读取 cache。旧 API 调用新 API 并传 `NULL`，
保持现有路径兼容。

- [x] **Step 3: GREEN 与 CUDA build**

```bash
cmake --build build-cuda --target ggml-cuda test_mixed_batch test_offline_batching -j 8
./build-cuda/test_mixed_batch
git diff --check
```

### Task 3: Packed Input Staging

**Files:**
- Create: `compute/gpu_mixed_runner.hpp`
- Create: `compute/gpu_mixed_runner.cpp`
- Modify: `compute/gpu_embedding_pool.hpp`, `compute/gpu_embedding_pool.cpp`
- Modify: `CMakeLists.txt`
- Test: `test/test_gpu_embedding_pool.cpp`

- [ ] **Step 1: 写 CPU-backend RED 测试**

用两个 pool slot 构造已知 Prompt embedding，按非连续 local offset 打包到 staging，
验证 Prompt rows 在前；Decode token 只保留 token id metadata，不从 host 提取 embedding。

- [ ] **Step 2: 实现 GPU-resident packer**

`GPUMixedRunner` 持有可增长、可复用的 F32 prompt staging tensor。每个 Prompt
sequence 通过 source/destination view 执行 backend-to-backend copy；Decode IDs
在 graph 内通过 `ggml_get_rows(embed_tokens, token_ids)` 得到 embedding，然后沿
token 维与 prompt staging `ggml_concat(..., 1)`。

- [ ] **Step 3: 验证异常路径**

stale embedding handle、越界 local offset、staging 扩容失败都必须在 graph 构建前
失败，不提交 scheduler 进度，不释放 request 所有权。

- [ ] **Step 4: GREEN**

```bash
cmake --build build-cuda --target test_gpu_embedding_pool test_mixed_batch -j 8
./build-cuda/test_gpu_embedding_pool
./build-cuda/test_mixed_batch
```

### Task 4: Flattened Packed Transformer and Selective LM Head

**Files:**
- Modify: `compute/llm_ops_gpu.hpp`
- Modify: `compute/llm_ops_gpu.cpp`
- Modify: `compute/gpu_mixed_runner.hpp`, `compute/gpu_mixed_runner.cpp`
- Modify: `compute/gpu_runner.hpp`

- [ ] **Step 1: 写 runner validation RED 测试**

覆盖 total token、position、expanded table、KV length、selected row 的 tensor shape；
拒绝空 selected rows、不同 block size 和超出 physical pool 的 block id。

- [ ] **Step 2: 实现 packed layer**

每层对 `[hidden, total_tokens]` 一次执行 RMSNorm、QKV projection、RoPE、O projection
和 MLP。K/V 通过 expanded table 与 per-token position 写入各自 request 的物理块。
Q reshape 为 `[head_dim, 1, n_heads, total_tokens]`，复用 batched Paged Attention；
每个 token 的 `kv_len=position+1` 保证 causal 可见范围。

- [ ] **Step 3: 实现 Selective LM Head**

Transformer 最后一层后先执行：

```cpp
ggml_tensor * selected = ggml_get_rows(ctx, hidden, selected_rows);
selected = gpu_rms_norm(ctx, selected, weights.model_norm_w, 1e-5f);
return ggml_mul_mat(ctx, weights.lm_head_w, selected);
```

只为 Prompt 完成位置和 Decode rows 生成 `[vocab, logits_rows]`。

- [ ] **Step 4: GPU 小图验证**

先关闭 graph cache，比较：

```text
decode-only packed vs existing token-id fast path
single prompt chunk packed vs sequential chunk executor
one prompt + one decode mixed vs 两次 reference execution
```

记录 logits `max_abs`、`max_rel`、top-1、top-5，不只比较文本。

### Task 5: Unified Offline Prepare / Execute / Commit

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_scheduler.cpp`
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: 写 request lifecycle RED 测试**

覆盖 frontend 每请求只执行一次、Prompt 跨轮推进、Prompt 完成只 sample 一次、首个
生成 token 下一轮才写 KV、执行失败回滚本轮新增 block、KV backpressure 不空转。

- [ ] **Step 2: 增加显式 experimental 配置**

```cpp
bool use_unified_scheduler = false;
int max_num_scheduled_tokens = 1024;
int max_prefill_chunk_tokens = 512;
int max_frontend_requests_per_step = 4;
```

默认继续使用现有 decode-only scheduler。

> 2026-07-21 实测更新：`frontend_per_step=1/4/12` 的 24-chunk wall
> 分别为 `6.021/5.921/6.046s`，因此 experimental 默认值调整为 4。

- [ ] **Step 3: 接入循环**

每轮按以下顺序执行：reap -> limited frontend -> build plan -> transactional KV append
-> pack -> execute -> commit -> sample -> release completed prompt embedding。GPU 失败时
rollback 本轮新增 blocks，`num_computed_tokens` 不变。

- [ ] **Step 4: 增加指标**

输出 `unified_steps/mixed_steps/prefill_only/decode_only/scheduled tokens/packed build/
packed compute/selected rows/frontend wait/embedding peak/kv backpressure`。

### Task 6: Varlen Paged Prefill Attention

当前已先落地一个不新增 GGML op 的混合实现：Q8 projection、O projection 和 MLP
继续对 packed tokens 合批；每条 Prompt sequence 使用 FlashAttention，历史 K/V 通过
运行时 physical-row tensor 从 Paged KV gather，当前 chunk K/V 直接参与 attention；
Decode suffix 继续使用已有 Paged Attention。该实现把 24-chunk wall 从旧 scheduler 的
`6.206s` 降至 `5.921s`。下面的自定义 varlen kernel 保留为进一步减少 gather/concat
和多 FlashAttention node 的后续优化，不再是接通 unified runtime 的前置条件。

**Files:**
- Modify: `third_party/ggml/include/ggml.h`
- Modify: `third_party/ggml/src/ggml.c`
- Modify: `third_party/ggml/src/ggml-cuda/paged-attn.cuh`
- Modify: `third_party/ggml/src/ggml-cuda/paged-attn.cu`
- Modify: `compute/llm_ops_gpu.cpp`

- [ ] **Step 1: 用 flattened backend 建 reference**

测试 query length `1/2/127/128/129/256`，随机 block table 与非连续 physical block，
比较每个 query row 的 attention 输出误差。

- [ ] **Step 2: 实现 varlen kernel**

输入 `query_start`、`positions`、`sequence_rows`、request block table 和 context lens。
Prompt query 使用多-query kernel；Decode request 继续走现有 cached-row decode kernel。
禁止让 `q_len>1` 退回 contiguous gather。

- [ ] **Step 3: profile 后决定 tile**

用 Nsight Compute 记录 DRAM throughput、SM throughput、occupancy 和 kernel duration，
比较 flattened reference；没有测量收益时不删除 reference fallback。

### Task 7: Graph Bucketing and Benchmark Gate

**Files:**
- Modify: `compute/gpu_mixed_runner.hpp`, `compute/gpu_mixed_runner.cpp`
- Modify: `tools/benchmark_suite.py`
- Modify: `docs/benchmarking.md`

- [ ] **Step 1: 正确性 smoke**

```text
24/24 complete
same-config run A/B token hash identical
final_free == capacity
ownership_errors == 0
embedding pool fully free
```

- [ ] **Step 2: 恢复 graph cache**

signature 使用 `total_token_bucket/sequence_count/logits_rows/max_kv_bucket`；position、
block table、KV lens、selected rows 和 token IDs 全部为运行时 input tensor。

- [ ] **Step 3: 204 chunk 三轮矩阵**

```text
max_num_scheduled_tokens: 256 / 512 / 1024 / 2048
max_prefill_chunk_tokens: 128 / 256 / 512
max_num_seqs: 12 / 24 / 32 / 40
```

同时记录 wall、RTF、Prefill/Decode wall、峰值显存、CER/WER、exact mismatch、
同配置确定性和资源指标。只有 wall 中位数优于 `40.812s` 至少 5%，且质量不退化，
才把 Unified 模式设为默认。
