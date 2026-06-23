# Paged Continuous Decode MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stable paged-KV continuous decode path that improves long-video offline transcription throughput.

**Architecture:** Keep `Pipeline::transcribe_audio()` and realtime paths unchanged. Make `OfflineBatchTranscriber` the offline scheduler owner, using `PagedKVBlockPool` for block allocation and grouped batched paged decode for active requests. Add focused stats so throughput gains and fallback behavior are visible.

**Tech Stack:** C++17, GGML CUDA backend, existing `test_offline_batching` executable, `rg`, CMake.

---

## File Structure

- Modify `pipeline/offline_batching.hpp`: add scheduler stats and expose them on `OfflineBatchTranscriber`.
- Modify `pipeline/offline_batching.cpp`: track paged KV usage, group active decode inputs by `n_past`, and record fallback counts.
- Modify `test/test_offline_batching.cpp`: print summary stats after the run.
- Modify `compute/gpu_runner.hpp`: preserve existing token-id paged decode fast path and make fallback behavior easier to detect if needed.

The first implementation stays inside existing files because the current offline scheduler is already localized there.

---

### Task 1: Add Offline Scheduler Stats

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Test: `test/test_offline_batching.cpp`

- [ ] **Step 1: Add a stats struct to the header**

Add this near `OfflineChunkResult` in `pipeline/offline_batching.hpp`:

```cpp
struct OfflineBatchStats {
    int total_chunks = 0;
    int batch_size = 1;
    bool use_gpu = false;
    bool use_paged_kv = false;
    int kv_block_size = 0;
    int kv_block_capacity = 0;
    int peak_blocks_in_use = 0;
    int decode_steps = 0;
    int decode_group_calls = 0;
    int decode_fallback_calls = 0;
    long long active_batch_sum = 0;

    double average_active_batch() const {
        return decode_steps > 0
            ? static_cast<double>(active_batch_sum) / static_cast<double>(decode_steps)
            : 0.0;
    }
};
```

Add this public accessor to `OfflineBatchTranscriber`:

```cpp
const OfflineBatchStats& last_stats() const { return last_stats_; }
```

Add this private field:

```cpp
OfflineBatchStats last_stats_;
```

- [ ] **Step 2: Initialize stats at the start of transcription**

In both `OfflineBatchTranscriber::transcribe()` and
`OfflineBatchTranscriber::transcribe_continuous_gpu()`, reset `last_stats_`:

```cpp
last_stats_ = OfflineBatchStats{};
last_stats_.total_chunks = static_cast<int>(chunks.size());
last_stats_.batch_size = std::max(1, config.batch_size);
last_stats_.use_gpu = config.use_gpu;
last_stats_.use_paged_kv = config.use_paged_kv;
last_stats_.kv_block_size = config.kv_block_size;
```

When creating a `PagedKVBlockPool`, also set:

```cpp
last_stats_.kv_block_capacity = num_blocks;
```

- [ ] **Step 3: Track peak block usage**

After each successful paged `acquire()`, update peak usage:

```cpp
if (handle.paged()) {
    const int used = kv_pool->capacity() - kv_pool->free_count();
    last_stats_.peak_blocks_in_use = std::max(last_stats_.peak_blocks_in_use, used);
}
```

- [ ] **Step 4: Build and run the existing offline batching target**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
```

Expected: build succeeds, or fails only on pre-existing CUDA/toolchain issues.

---

### Task 2: Group Active Paged Decode Inputs by `n_past`

**Files:**
- Modify: `pipeline/offline_batching.cpp`
- Test: `test/test_offline_batching.cpp`

- [ ] **Step 1: Add grouping helper includes**

Add to `pipeline/offline_batching.cpp`:

```cpp
#include <map>
```

- [ ] **Step 2: Replace the single decode call with grouped calls**

Inside `transcribe_continuous_gpu()`, replace:

```cpp
std::vector<GPUDecodeStepOutput> decode_outputs =
    recognizer_.gpu_decode_step_slots(decode_inputs, inference);
```

with:

```cpp
std::vector<GPUDecodeStepOutput> decode_outputs;
if (!decode_inputs.empty()) {
    last_stats_.decode_steps++;
    last_stats_.active_batch_sum += static_cast<long long>(decode_inputs.size());

    std::map<int, std::vector<GPUDecodeStepInput>> by_past;
    for (const auto& input : decode_inputs) {
        by_past[input.n_past].push_back(input);
    }

    for (auto& entry : by_past) {
        std::vector<GPUDecodeStepOutput> group_outputs =
            recognizer_.gpu_decode_step_slots(entry.second, inference);
        if (entry.second.size() == 1) {
            last_stats_.decode_fallback_calls++;
        } else {
            last_stats_.decode_group_calls++;
        }
        decode_outputs.insert(
            decode_outputs.end(),
            group_outputs.begin(),
            group_outputs.end());
    }
}
```

- [ ] **Step 3: Verify ordered output is preserved**

Run a short smoke test:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --gpu --kv-mode paged --batch-size 2 \
  --ctx-size 2048 --chunk-mode window --chunk-sec 3 \
  --max-tokens 40
```

Expected: chunk progress prints in completion order, final results are sorted by id.

---

### Task 3: Print Throughput and Scheduler Summary

**Files:**
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: Print scheduler stats after transcription**

After `auto results = transcriber.transcribe(...)`, read stats:

```cpp
const funasr::OfflineBatchStats& stats = transcriber.last_stats();
```

After the existing totals are computed, print:

```cpp
std::printf("[OfflineTest] scheduler: gpu=%d kv=%s batch=%d chunks=%d "
            "decode_steps=%d grouped_calls=%d fallback_calls=%d "
            "avg_active=%.2f blocks_peak=%d/%d block_size=%d\n",
            stats.use_gpu ? 1 : 0,
            stats.use_paged_kv ? "paged" : "continuous",
            stats.batch_size,
            stats.total_chunks,
            stats.decode_steps,
            stats.decode_group_calls,
            stats.decode_fallback_calls,
            stats.average_active_batch(),
            stats.peak_blocks_in_use,
            stats.kv_block_capacity,
            stats.kv_block_size);
```

- [ ] **Step 2: Rebuild the test executable**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 3: Run paged and continuous comparison**

Run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --gpu --kv-mode paged --batch-size 2 \
  --ctx-size 2048 --chunk-mode window --chunk-sec 3 \
  --max-tokens 40
```

Run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --gpu --kv-mode continuous --batch-size 1 \
  --ctx-size 2048 --chunk-mode window --chunk-sec 3 \
  --max-tokens 40
```

Expected: both runs complete; paged run prints `kv=paged` and nonzero scheduler counters.

---

### Task 4: Prefer Token-ID Paged Decode Fast Path

**Files:**
- Modify: `pipeline/pipeline.cpp`
- Inspect: `compute/gpu_runner.hpp`

- [ ] **Step 1: Confirm paged decode inputs carry token IDs**

In `OfflineBatchTranscriber::transcribe_continuous_gpu()`, each paged decode
input should set:

```cpp
input.token_id = token;
input.block_table = request.kv.block_table;
input.block_size = request.kv.block_size;
```

- [ ] **Step 2: Confirm `Pipeline::gpu_decode_step_slots()` enters token-id path**

In `pipeline/pipeline.cpp`, paged decode should collect token IDs and call:

```cpp
gpu_runner_->forward_batch_decode_paged_token_ids(
    batch_token_ids, batch_pasts, batch_tables, block_size, next_tokens)
```

The function should only fall back when `n_past` differs within that group or
when `FUNASR_PAGED_DECODE_SERIAL` is set.

- [ ] **Step 3: Add a diagnostic fallback counter if needed**

If the existing code cannot tell whether the token-id fast path ran, add a
single `printf` on failure:

```cpp
printf("[Pipeline] GPU scheduler paged token-id decode unavailable; falling back\n");
```

Keep this message only on fallback, not per successful step.

- [ ] **Step 4: Rebuild and smoke test**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --gpu --kv-mode paged --batch-size 2 \
  --ctx-size 2048 --chunk-mode window --chunk-sec 3 \
  --max-tokens 40
```

Expected: run completes; no repeated fallback message in the normal fast path.

---

### Task 5: Validate Long-Video Throughput Shape

**Files:**
- No source edits expected.
- Use: `test/test_offline_batching.cpp`

- [ ] **Step 1: Run a multi-chunk paged benchmark**

Use the longest local 16 kHz WAV available, for example:

```bash
./build-cuda/test_offline_batching outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 4 \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

If the executable expects model first, use:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 4 \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

Expected: summary prints `avg_active` greater than `1.00` when the input has
multiple chunks.

- [ ] **Step 2: Run single-slot baseline**

Run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode continuous --batch-size 1 \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

Expected: baseline completes and provides wall time/RTF comparison.

- [ ] **Step 3: Record findings**

Update the final response with:

```text
Paged run wall time:
Baseline wall time:
Paged avg_active:
Grouped decode calls:
Fallback decode calls:
Observed issue, if any:
```

Do not claim throughput improvement unless the measured wall time shows it.

---

## Self-Review

- Spec coverage: scheduler stats, paged KV, grouped decode, token-id fast path,
  validation, and non-goals are covered.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: all names match existing types or types introduced in Task 1.

## Git Note

The current directory is not a valid Git repository, so commit steps are omitted
for this workspace. If this code is moved into a valid checkout, commit after
each task with a focused message.
