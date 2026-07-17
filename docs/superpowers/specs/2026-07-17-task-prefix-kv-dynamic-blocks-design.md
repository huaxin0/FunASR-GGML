# Task-Scoped Prefix KV Cache and Dynamic Paged Blocks Design

**Date:** 2026-07-17

**Status:** Approved

## 1. Purpose

The offline long-audio scheduler currently reserves a request's estimated prefill and maximum decode capacity when the request is admitted. It also runs the fixed ChatML prefix through the LLM for every audio chunk. This design replaces those behaviors with:

1. A task-scoped prefix KV cache that computes a fixed ChatML prefix once and reuses its KV state across all chunks in one transcription task.
2. Reference-counted paged KV blocks with copy-on-write (COW) for a shared partial prefix block.
3. Exact prefill allocation after the GPU frontend reports the real audio embedding length.
4. Decode-time block append only when a request crosses a block boundary.

The change targets the paged offline GPU path. The continuous-slot path and realtime/PTT path retain their current behavior.

## 2. Goals

- Compute the task's fixed ChatML prefix KV once instead of once per chunk.
- Preserve the exact logical token order and positions:

  ```text
  [prefix at 0..P-1] [audio at P..P+A-1] [suffix at P+A..P+A+S-1]
  ```

- Use `max_tokens` only as a generation stopping limit, not as admission-time KV capacity.
- Allocate enough blocks for the real prefill length and append one block at a decode boundary.
- Make every physical block's ownership explicit through a reference count.
- Prevent duplicate release, duplicate free-queue insertion, and accidental concurrent writes to a shared block.
- Preserve current dynamic paged KV write, KV-length bucketing, and decode graph cache behavior.
- Provide feature switches and metrics for reproducible A/B comparisons.

## 3. Non-Goals

- Process-global or cross-process prefix caching.
- An LRU cache, external KV cache service, or distributed KV connector.
- Prefill/decode disaggregation.
- Speculative decoding.
- Batched encoder/adaptor execution.
- A fused CUDA COW-copy kernel in the first implementation.
- Changes to model weights, quantization, tokenizer semantics, or ASR chunking.

These are separate follow-up projects. The next likely performance project after this work is batched frontend/prefill plus frontend CUDA Graph evaluation.

## 4. Current Behavior and Problems

### 4.1 Admission reserves estimated future capacity

The current paged scheduler computes:

```text
estimated_prefill = audio_seconds * 25 + 128
kv_budget = estimated_prefill + max_tokens
```

It then allocates all blocks for `kv_budget` before frontend execution. On the current 6119-second, 204-chunk workload this produced:

```text
allocated blocks: 1836
used blocks:      1024
wasted blocks:     812
waste rate:      44.23%
```

The estimate is not authoritative because the actual adaptor output length becomes available after frontend execution, while decode length is unknown until EOS or another stopping condition.

### 4.2 The fixed ChatML prefix is recomputed per chunk

`PromptBuilder` already caches default prefix and suffix token IDs, but the paged prefill path still:

- rebuilds prefix and suffix embeddings for every chunk;
- executes all decoder layers for the prefix on every chunk;
- writes a new copy of the resulting prefix KV for every chunk.

The prefix is much shorter than the audio embedding sequence, so the expected wall-time gain is modest. The architectural and memory-lifecycle value is still important and must be measured rather than assumed.

### 4.3 The paged block pool has no ownership tracking

The current free queue accepts every valid block ID passed to `release()`. A duplicate release can enqueue the same physical block twice. The new reference-counted pool is a prerequisite for safe prefix sharing and also closes this correctness gap.

## 5. Architecture

The paged offline path is split into five responsibilities:

```text
Offline Scheduler
    orchestrates requests and stopping/error policy

Paged KV Block Pool
    allocate / retain / release / append / COW metadata

Task Prefix KV Cache
    owns the task's prompt key and one retained prefix KV state

Prefill Engine
    builds prefix state once and resumes from it for audio + suffix

Decode Engine
    consumes a PagedKVState without knowing whether blocks are shared
```

No plugin framework is introduced. The interfaces are kept narrow enough that a different decode backend or a future external KV connector would not require rewriting scheduler policy.

## 6. Core Data Structures

### 6.1 Paged KV state

```cpp
struct PagedKVState {
    std::vector<int> block_table;
    int block_size = 0;
    int token_count = 0;

    bool valid() const;
};
```

`block_table` contains only physically allocated blocks. Vector capacity or unallocated placeholder entries are not ownership.

### 6.2 Prompt cache key

```cpp
struct PromptCacheKey {
    std::vector<int> prefix_token_ids;
    int block_size = 0;
    uint64_t model_generation = 0;

    bool operator==(const PromptCacheKey& other) const;
};
```

The task-scoped implementation normally has one entry, but the key is explicit to prevent reuse after prompt, block-size, or model changes. Equality compares the actual token IDs; a hash alone is not accepted as proof of equality.

### 6.3 Prefix cache entry

```cpp
struct PrefixKVEntry {
    PromptCacheKey key;
    PagedKVState state;
    int prefix_tokens = 0;
    bool ready = false;
};
```

The cache owns one reference to every block in `state.block_table`. A request acquires additional references when it attaches the entry.

### 6.4 Block metadata

The paged pool maintains one record per physical block:

```cpp
struct PagedBlockMeta {
    uint32_t ref_count = 0;
    bool in_free_queue = true;
};
```

Required operations:

```cpp
int allocate_block();
bool append_block(PagedKVState& state);
bool retain_block(int block_id);
bool release_block(int block_id);
bool release_state(PagedKVState& state);
bool replace_block_after_cow(PagedKVState& state,
                             int logical_block,
                             int replacement_block);
```

`release_block()` returns a block to the free queue only on the `1 -> 0` reference transition. Releasing a zero-reference block is reported as an ownership error and does not mutate the queue.

## 7. Prefix Cache Lifecycle

### 7.1 Task initialization

Before admitting audio requests, the scheduler:

1. Builds the prefix token IDs from the task's `PromptOptions`.
2. Creates a `PromptCacheKey`.
3. Allocates `ceil(prefix_tokens / block_size)` blocks.
4. Runs a prefix-only paged LLM prefill with `n_past = 0`.
5. Stores the resulting block table and `token_count = prefix_tokens` in `PrefixKVEntry`.
6. Keeps one cache-owned reference to each prefix block.

The prefix-only logits are not used for ASR decoding.

If allocation or GPU prefix prefill fails, all acquired blocks are released and the transcription task fails before request admission.

### 7.2 Request attachment

When a request has completed frontend execution:

1. Copy the prefix block IDs into the request's `PagedKVState`.
2. Retain each copied block ID.
3. Set `request.state.token_count = prefix_tokens`.
4. Prepare the state for appending the request-specific audio tail.

### 7.3 Partial-block copy-on-write

If `prefix_tokens % block_size != 0`, the final prefix block is partially occupied. Before audio KV is written:

1. Allocate a private replacement block.
2. Copy exactly `prefix_tokens % block_size` K rows and V rows for every decoder layer from the shared block to the replacement.
3. Replace the request's final block-table entry with the replacement.
4. Release the request's reference to the shared partial block.
5. Leave the cache-owned shared block unchanged.

If the prefix ends on a block boundary, all complete prefix blocks remain shared and the first audio token starts in a newly appended private block. Padding is never inserted because it would change positions and model semantics.

### 7.4 Task teardown

After every request has finished or failed, the task releases the cache-owned prefix references. The final invariant is:

```text
free block count == pool capacity
```

Any mismatch is emitted as an ownership/leak error in test and benchmark output.

## 8. Exact Prefill Allocation

The existing combined `gpu_prefill_audio_paged()` path is split so the scheduler can allocate after observing the actual adaptor output length.

```cpp
struct PreparedGPUAudio {
    ggml_tensor* adaptor_tensor = nullptr;
    int audio_frames = 0;
    float encoder_ms = 0.0f;
    bool ok = false;
};
```

The adaptor tensor is non-owning and remains valid only until the next frontend invocation. The scheduler must consume it immediately; it cannot store it across requests.

Per request:

```text
prepare_audio_gpu(audio)
    -> actual audio_frames

tail_tokens = audio_frames + suffix_tokens
required_total_blocks = ceil((prefix_tokens + tail_tokens) / block_size)

attach prefix blocks
COW the shared partial block when required
append only the missing prefill blocks

prefill [audio + suffix] with n_past = prefix_tokens
```

The prefill staging tensor contains only `[audio embeddings + suffix embeddings]`. Position inputs begin at `prefix_tokens`, and attention reads the prefix through the request block table.

The implementation must validate:

```text
prefix_tokens + audio_frames + suffix_tokens <= ctx_size
```

It must not require room for all `max_tokens` before admitting the request. Decode later stops at the earlier of EOS, `max_tokens`, repetition protection, or `ctx_size`.

## 9. Decode-Time Block Append

Before dispatching a request's next token:

```text
required_blocks = ceil((n_past + 1) / block_size)
```

If `required_blocks > block_table.size()`, the scheduler appends one private block. No block is appended when the current final block still has room.

If append cannot allocate immediately:

1. Finished and failed requests from the current iteration are released first.
2. The blocked request remains active and is retried on the next scheduler iteration.
3. The scheduler tracks whether any request made progress.
4. If every active request is waiting for a block and no release or decode progress is possible, the scheduler terminates those requests with an explicit `kv_exhausted` result rather than spinning forever.

Preemption and recomputation are out of scope because the configured physical pool currently has substantial headroom. The failure is still explicit and observable.

## 10. GPU KV Copy

The first implementation adds a device-to-device copy operation for a contiguous prefix row range across all decoder layers and both K and V tensors.

```cpp
bool copy_paged_kv_rows(int source_block,
                        int destination_block,
                        int valid_rows,
                        int block_size);
```

Requirements:

- `0 < valid_rows < block_size` for the COW path.
- Source and destination block IDs must be distinct and in range.
- The copy includes every KV head dimension, decoder layer, and both K and V.
- The operation completes on the same backend/stream ordering domain before tail prefill writes the destination block.
- A partial failure leaves the request failed and releases the newly allocated destination block.

The first version may use GGML device copies or explicit backend copies. A custom fused CUDA kernel is considered only if profiling shows COW copy time is material.

## 11. Decode Graph Compatibility

The existing decode fast path remains enabled:

- GPU token-ID embedding lookup;
- dynamic paged KV write;
- bucketed `max_n_kv`;
- decode graph cache.

Actual block-table length can grow at a block boundary, so a graph signature may change at append time. That is acceptable for the first implementation. Metrics must show whether graph-cache hit rate changes materially.

The implementation must not pad ownership block tables with fake physical block IDs merely to preserve graph shape. A future upload-specific padded table may be introduced behind a separate metadata abstraction if measurements justify it.

## 12. Configuration and Rollout

Add typed configuration fields and matching offline-test CLI options:

```cpp
PromptOptions prompt;
bool enable_prefix_kv_cache = false;
bool enable_dynamic_kv_blocks = false;
```

```text
--prefix-kv-cache <on|off>
--dynamic-kv-blocks <on|off>
```

During development and A/B validation, both options are explicit and default to the existing behavior. After correctness and full benchmark acceptance, the paged offline preset enables both by default while retaining `off` for regression comparison. Continuous KV mode ignores both options with a clear diagnostic.

The scheduler copies `OfflineBatchConfig::prompt` into the `InferenceConfig`
used for prefix construction and tail prefill. The same prompt value is used for
the cache key and all requests in that transcription task, preventing the cache
from being built with one ChatML prefix while requests use another.

## 13. Metrics

Add these task-level statistics:

```text
prefix_cache_builds
prefix_cache_hits
prefix_tokens_computed
prefix_tokens_reused
prefix_tokens_saved_vs_baseline
prefix_build_ms
prefix_cow_copies
prefix_cow_rows
prefix_cow_bytes
prefix_cow_ms
blocks_appended_prefill
blocks_appended_decode
append_retries
append_failures
kv_ownership_errors
final_free_blocks
```

`prefix_cache_hits` increments when a request successfully attaches the ready
task cache. `prefix_tokens_computed` counts the one prefix build, while
`prefix_tokens_reused` adds `prefix_tokens` for every successful request
attachment. These counters describe actual reuse events rather than cache-key
lookups. `prefix_tokens_saved_vs_baseline` is
`prefix_tokens * max(0, prefix_cache_hits - 1)`, because the one task-level
prefix build must still be computed.

Memory reporting separates:

1. Whole-block over-reservation: allocated blocks beyond current logical requirements. This must be zero in dynamic mode.
2. Internal fragmentation: unused rows in each request's currently allocated final block.
3. Cache residency: blocks held by the task prefix cache.

The previous aggregate `waste_rate` must not combine these categories.

## 14. Error Handling and Invariants

Every request exit path uses one release routine. This includes EOS, `max_tokens`, context limit, repetition protection, frontend failure, prefill failure, decode failure, and KV exhaustion.

Required invariants:

- A free block has `ref_count == 0` and appears exactly once in the free queue.
- An allocated block has `ref_count > 0` and does not appear in the free queue.
- A block with `ref_count > 1` is never written.
- Before writing logical position `p`, the request owns a writable block for `p / block_size`.
- `state.token_count` equals request `n_past` after successful prefill/decode.
- A failed GPU decode output marks the corresponding request failed; it cannot silently continue with an invalid next token.
- Task teardown returns every block to the pool.

Debug builds and unit tests treat invariant violations as failures. Release builds report the error and fail the affected task/request without corrupting another request's KV.

## 15. Testing

### 15.1 CPU unit tests

Use small artificial pools to cover:

```text
block_size=4, prefix_len=3   partial block COW
block_size=4, prefix_len=4   exact boundary, no COW
block_size=4, prefix_len=9   two shared full blocks plus partial COW
```

Tests cover:

- allocate and append reduce the free count exactly once;
- retain increments a block reference;
- release returns a block only on `1 -> 0`;
- duplicate release reports an ownership error without duplicating the free entry;
- request attachment retains all prefix blocks;
- COW replaces only the partial logical block;
- full prefix blocks remain shared;
- request release preserves the cache-owned reference;
- task cache release restores `free_count == capacity`;
- append failure leaves existing ownership intact;
- `max_tokens` does not affect dynamic admission allocation.

### 15.2 GPU correctness tests

Compare baseline and optimized modes on identical model/audio inputs:

```text
baseline: prefix cache off, dynamic blocks off
optimized: prefix cache on, dynamic blocks on
```

Validate:

- all chunks complete successfully;
- prefill token counts and logical positions match;
- short test inputs generate identical token sequences;
- copied prefix KV rows match independently computed rows within the storage type's tolerance;
- the long-video transcript and CER do not regress;
- no block ownership errors or leaks occur.

Splitting prefill can change floating-point execution order. Small logits differences are acceptable, but unexplained token instability or CER degradation is not.

### 15.3 Benchmark correctness prerequisite

Fix the `--max-chunks` accounting bug before using short benchmarks: reported audio duration must cover the selected chunks, not the entire source file.

## 16. Performance Evaluation

Run four variants:

```text
A: static reservation, prefix cache off
B: dynamic blocks,    prefix cache off
C: static reservation, prefix cache on
D: dynamic blocks,    prefix cache on
```

Fixed final workload:

```text
GPU:          RTX 4070 Laptop 8 GB
audio:        current 6119-second source
chunks:       204
batch size:   12
context:      4096
block size:   128
max tokens:   220
KV mode:      paged
```

Use a warmup and at least three measured repetitions per variant. Report medians for:

```text
wall time
RTF and audio seconds/second
prefill wall time
decode dispatch time
tokens/second
peak blocks
internal fragmentation
prefix build and COW time
decode graph-cache hit rate
```

Acceptance criteria:

- Recognition output and CER do not regress.
- Dynamic mode has zero whole-block future over-reservation.
- Prefix cache builds once and reuses the prefix for all admitted chunks.
- Append failures and ownership errors are zero on the target workload.
- Graph-cache behavior is measured and any material loss is explained.
- Median full wall time does not regress by more than 3% without an understood and documented cause.
- Performance results are reported even when prefix reuse produces no measurable wall-time gain.

## 17. Expected Outcome

The optimized scheduler will no longer depend on an audio-seconds heuristic or reserve `max_tokens` worth of KV at admission. It will own a correct, reference-counted block lifecycle and a task-scoped prefix cache with safe partial-block COW.

The likely immediate result is a large reduction in reserved KV blocks and a smaller reduction in repeated prefix prefill work. The design intentionally treats that speedup as an experimental question. Its larger value is a sound KV-state boundary that supports later frontend batching, richer prefix caching, or a different decode backend without reworking scheduler semantics.
