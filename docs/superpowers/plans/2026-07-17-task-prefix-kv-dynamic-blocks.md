# Task-Scoped Prefix KV Cache and Dynamic Paged Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace paged offline admission-time max-token reservation with exact prefill allocation and decode-time append, while computing the task's fixed ChatML prefix KV once and safely reusing it through reference counting and partial-block copy-on-write.

**Architecture:** Keep scheduler policy in `offline_batching`, add ownership semantics to the existing paged block pool, and expose narrow Pipeline/GPURunner operations for prefix-only prefill, ephemeral GPU audio preparation, tail prefill, and device-to-device KV row copy. Preserve the old path behind explicit switches so all four static/dynamic and cache-on/off combinations remain benchmarkable.

**Tech Stack:** C++17, GGML backend API, CUDA backend, CMake, existing custom test executables, Nsight-compatible benchmark output.

---

## Baseline

The current CUDA build completed before implementation. Fresh baseline results:

```text
./build-cuda/test_offline_scheduler: 34 passed, 0 failed
./build-cuda/test_cli_options: all passed
```

Implementation is authorized in the current `funasr` branch so the existing uncommitted KV-waste instrumentation remains available.

### Task 1: Fix selected-audio benchmark accounting

**Files:**
- Modify: `pipeline/chunking.hpp`
- Modify: `pipeline/chunking.cpp`
- Modify: `test/test_offline_scheduler.cpp`
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: Add a failing selected-duration test**

Add a scheduler unit test that constructs two selected chunks covering samples `[0, 16000)` and `[16000, 32000)` at 16 kHz while the hypothetical source is longer. Assert that `selected_audio_duration_seconds(chunks, 16000)` returns 2 seconds.

- [ ] **Step 2: Run the scheduler test and verify RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler
```

Expected: compilation fails because `selected_audio_duration_seconds` is not defined.

- [ ] **Step 3: Implement the duration helper**

Declare and define:

```cpp
float selected_audio_duration_seconds(const std::vector<AudioChunk>& chunks,
                                      int sample_rate);
```

The helper returns the sum of each selected chunk's nonnegative `[start_sample, end_sample)` duration. Invalid sample rates return zero.

- [ ] **Step 4: Use the helper in the offline benchmark**

After `--max-chunks` truncation, compute `audio_sec` from selected chunks. Fall back to full samples only when no chunks exist.

- [ ] **Step 5: Build and run GREEN verification**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler test_offline_batching
./build-cuda/test_offline_scheduler
```

Expected: all scheduler tests pass.

### Task 2: Make the paged block pool reference-counted

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_scheduler.cpp`

- [ ] **Step 1: Add failing ownership and append tests**

Add focused tests for:

```text
retain increments ref_count
first release preserves a retained block
final release returns it once
duplicate release does not grow free_count
append_block grows the table by one
failed append leaves the table unchanged
replace_block_after_cow swaps one table entry and releases the old reference
```

- [ ] **Step 2: Build and verify RED**

Run the scheduler target. Expected: compilation fails on the new pool API.

- [ ] **Step 3: Add block metadata and APIs**

Add per-block reference counts and free-queue flags. Add:

```cpp
int ref_count(int block_id) const;
int ownership_errors() const;
bool retain_block(int block_id);
bool retain(const KVHandle& handle);
bool append_block(KVHandle& handle);
bool replace_block_after_cow(KVHandle& handle,
                             size_t logical_block,
                             int replacement_block);
```

Keep `acquire(request_id, tokens)` for the static path. Every acquired block starts with one reference. `release()` only queues a block on `1 -> 0`; invalid and zero-reference release attempts increment `ownership_errors_` without changing the queue.

- [ ] **Step 4: Run GREEN verification**

Build and run `test_offline_scheduler`. Expected: all old and new ownership tests pass.

### Task 3: Add prompt/cache state and configuration surfaces

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `cli/funasr_cli.cpp`
- Modify: `test/test_cli_options.cpp`
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: Add failing CLI/config tests**

Test parsing and mapping for:

```text
--prefix-kv-cache on|off
--dynamic-kv-blocks on|off
```

Assert both default off in raw options and enabled in the final long-video paged preset only after rollout code explicitly enables them. Assert continuous mode disables both.

- [ ] **Step 2: Verify RED**

Build `test_cli_options`. Expected: compilation or parsing assertions fail because fields/options do not exist.

- [ ] **Step 3: Add typed configuration and task cache metadata**

Add to `OfflineBatchConfig`:

```cpp
PromptOptions prompt;
bool enable_prefix_kv_cache = false;
bool enable_dynamic_kv_blocks = false;
```

Add task-scoped metadata using exact prefix token IDs, block size, prefix token count, readiness, and an owning prefix `KVHandle`. Add Pipeline/Recognizer accessors for prefix and suffix IDs without exposing `PromptBuilder` itself.

- [ ] **Step 4: Add CLI parsing and benchmark flags**

Implement strict `on|off` parsing, usage text, config mapping, and equivalent flags in `test_offline_batching` so A/B modes are available from both executables.

- [ ] **Step 5: Run GREEN verification**

Build and run `test_cli_options` and `test_offline_scheduler`.

### Task 4: Add GPU prefix prefill and device-to-device COW copy

**Files:**
- Modify: `compute/gpu_runner.hpp`
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_gpu.cpp` or create `test/test_paged_kv_gpu.cpp`
- Modify: `CMakeLists.txt` when a new test target is used

- [ ] **Step 1: Add a GPU COW-copy regression test**

Use a small initialized GPU KV cache, write a known byte/FP16 row pattern into a source block, invoke the new copy operation for a partial row count, and assert:

```text
destination valid rows equal source rows
destination rows after valid_rows remain unchanged
source rows remain unchanged
invalid block/range arguments return false
```

- [ ] **Step 2: Build and verify RED**

Expected: compilation fails because `copy_paged_kv_block_rows` does not exist.

- [ ] **Step 3: Implement device-to-device KV row copy**

Add:

```cpp
bool copy_paged_kv_block_rows(int source_block,
                              int destination_block,
                              int valid_rows,
                              int block_size);
```

Create GGML views for every decoder layer and K/V tensor and invoke backend tensor copies on the active GPU backend. Validate ranges and preserve backend stream ordering. Do not stage KV through host memory.

- [ ] **Step 4: Add prefix-only paged prefill**

Expose a Pipeline/Recognizer method that embeds prefix IDs once and calls paged prefill with `n_past=0`. Return prefix token count, elapsed time, and success; prefix logits are discarded.

- [ ] **Step 5: Run GPU GREEN verification**

Build and run the GPU copy test on the RTX 4070. Expected: all copy and validation assertions pass.

### Task 5: Split GPU frontend preparation from paged tail prefill

**Files:**
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_pipeline.cpp` or add focused GPU integration coverage

- [ ] **Step 1: Add a failing API/length test**

Add coverage showing that prepared GPU audio reports the actual adaptor frame count and that cached-prefix tail prefill reports:

```text
prefill_tokens = prefix_len + audio_frames + suffix_len
n_past after prefill = prefill_tokens
```

- [ ] **Step 2: Verify RED**

Build the focused target. Expected: new preparation/tail APIs are missing.

- [ ] **Step 3: Extract ephemeral GPU audio preparation**

Add `PreparedGPUAudio` with a non-owning adaptor tensor, actual frame count, encoder time, and success flag. Reuse the current Fbank and GPU encoder/adaptor code. Document and enforce immediate consumption before the next frontend call.

- [ ] **Step 4: Implement full and cached-prefix prepared prefill**

Support two modes:

```text
no prefix cache: stage [prefix + audio + suffix], n_past=0
prefix cache:    stage [audio + suffix],          n_past=prefix_len
```

Use the same logical positions and return logits from the final suffix token. Retain the existing combined API as the baseline wrapper.

- [ ] **Step 5: Run GREEN verification**

Build and run focused pipeline/GPU tests.

### Task 6: Integrate task prefix cache, COW, exact prefill, and decode append

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_scheduler.cpp`

- [ ] **Step 1: Add failing lifecycle tests around pure helpers**

Extract/test helper-level decisions for:

```text
required blocks from actual token count
partial prefix block index and valid rows
decode append only at a boundary
max_tokens does not alter dynamic prefill allocation
```

- [ ] **Step 2: Verify RED**

Build scheduler tests and confirm failures are due to missing helpers.

- [ ] **Step 3: Build and own one task prefix entry**

At paged GPU task start, allocate prefix blocks, run prefix-only prefill, and retain one cache-owned state. On any build failure, release all prefix blocks and fail before admission.

- [ ] **Step 4: Change dynamic admission order**

For dynamic mode:

```text
run frontend first
obtain actual audio_frames
attach/retain prefix blocks when enabled
COW a shared partial block before writing
append exact blocks for actual prefill
run full or tail prefill
```

For static mode, preserve estimated reservation so A/C comparisons remain possible.

- [ ] **Step 5: Append during decode and handle exhaustion**

Before writing token position `n_past`, ensure `ceil((n_past + 1)/block_size)` blocks exist. Retry after releasing completed requests. Detect no-progress exhaustion and fail affected requests with an explicit reason instead of looping.

- [ ] **Step 6: Unify request and task release paths**

Every success/failure exit releases request references once. After all requests complete, release the cache-owned prefix state and verify the free count equals capacity.

- [ ] **Step 7: Run scheduler and short GPU GREEN verification**

Build all touched targets. Run unit tests and a two/three-chunk optimized transcription. Assert all chunks succeed and ownership errors are zero.

### Task 7: Add metrics and four-way benchmark support

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_batching.cpp`
- Create: `tools/bench_prefix_kv_dynamic_blocks.sh`

- [ ] **Step 1: Add failing stats helper tests**

Cover cache hit/reuse/saved-token formulas, whole-block over-reservation, internal fragmentation, append counters, ownership errors, and final free blocks.

- [ ] **Step 2: Verify RED then implement counters**

Add the approved metrics and print one stable summary line for parsing.

- [ ] **Step 3: Add the A/B shell harness**

Run variants A/B/C/D with shared model/audio/config, warmup, configurable repeats, per-run logs, and TSV summary. Use selected chunk duration for short runs.

- [ ] **Step 4: Run unit GREEN verification**

Build and run scheduler and CLI tests.

### Task 8: Final correctness and performance verification

**Files:**
- Update: `docs/superpowers/notes/2026-06-17-offline-asr-vllm-optimization-summary.md` only after measured results exist

- [ ] **Step 1: Run complete build and test suite available for the change**

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler test_cli_options test_offline_batching
./build-cuda/test_offline_scheduler
./build-cuda/test_cli_options
```

Run focused GPU tests as added above.

- [ ] **Step 2: Run short four-way functional A/B**

Use 24 chunks, compare successful chunk count, token/text stability, block ownership, prefix build count, COW count, append count, and graph-cache hit rate.

- [ ] **Step 3: Run full optimized workload**

Run the 204-chunk optimized variant on the RTX 4070 and compare against a fresh baseline. If time permits, run at least three measured repetitions per variant and report medians.

- [ ] **Step 4: Inspect diff and document actual findings**

Run `git diff --check`, review all changed files, and write measured outcomes without claiming a speedup when the data does not show one.

