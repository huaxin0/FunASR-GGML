# Paged Offline Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add diagnostics, throughput summaries, and memory-budget warnings for the paged offline continuous decode path.

**Architecture:** Keep the current paged scheduler and decode implementation unchanged. Extend existing stats structs, expose the current paged decode profile through accessors, and make `test_offline_batching` print enough information to explain fallback behavior and batch-size performance.

**Tech Stack:** C++17, CMake, GGML CUDA backend, existing `test_offline_scheduler` and `test_offline_batching` executables.

---

## File Structure

- Modify `pipeline/offline_batching.hpp`: add fallback reason stats and helper methods.
- Modify `pipeline/offline_batching.cpp`: track scheduler-visible fallback reasons and keep aggregate fallback counts synchronized.
- Modify `pipeline/pipeline.hpp`: expose GPU paged decode profile and add optional fallback status plumbing for paged decode.
- Modify `pipeline/pipeline.cpp`: classify token-id fast-path fallback, host-embedding fallback, serial-env fallback, and invalid paged inputs.
- Modify `compute/gpu_runner.hpp`: move `PagedDecodeProfile` to namespace scope, expose a read-only snapshot, and add helper methods for average timings.
- Modify `test/test_offline_scheduler.cpp`: add unit tests for fallback totals and stats synchronization.
- Modify `test/test_offline_batching.cpp`: print fallback reasons, throughput summary, memory-budget hints, and paged profile summary.

This plan mostly stays within existing files. It adds `compute/gpu_profile.hpp` so CUDA and non-CUDA headers can share the lightweight profile type safely. The current codebase already localizes offline scheduling in `pipeline/offline_batching.*`, GPU dispatch in `pipeline/pipeline.*`, and the benchmark harness in `test/test_offline_batching.cpp`.

---

### Task 1: Add Fallback Reason Stats

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Test: `test/test_offline_scheduler.cpp`

- [x] **Step 1: Add fallback reason struct**

In `pipeline/offline_batching.hpp`, add this before `OfflineBatchStats`:

```cpp
struct OfflineDecodeFallbackStats {
    int single_request = 0;
    int token_id_fast_path_unavailable = 0;
    int host_embedding_batch_unavailable = 0;
    int serial_env_forced = 0;
    int invalid_paged_input = 0;

    int total() const {
        return single_request +
               token_id_fast_path_unavailable +
               host_embedding_batch_unavailable +
               serial_env_forced +
               invalid_paged_input;
    }
};
```

- [x] **Step 2: Add fallback reasons to OfflineBatchStats**

In `OfflineBatchStats`, add:

```cpp
OfflineDecodeFallbackStats fallback_reasons;

void sync_fallback_total() {
    decode_fallback_calls = fallback_reasons.total();
}
```

Keep the existing `decode_fallback_calls` field so the current scheduler summary does not change shape.

- [x] **Step 3: Add unit test for fallback total**

In `test/test_offline_scheduler.cpp`, add:

```cpp
void test_fallback_reason_total() {
    std::printf("\n--- Test 3: OfflineDecodeFallbackStats total ---\n");

    funasr::OfflineDecodeFallbackStats stats;
    TEST_ASSERT(stats.total() == 0, "empty fallback stats total is zero");

    stats.single_request = 2;
    stats.token_id_fast_path_unavailable = 3;
    stats.host_embedding_batch_unavailable = 5;
    stats.serial_env_forced = 7;
    stats.invalid_paged_input = 11;

    TEST_ASSERT(stats.total() == 28, "fallback reason total sums all counters");
}
```

Call it from `main()` after the existing tests:

```cpp
test_fallback_reason_total();
```

- [x] **Step 4: Add unit test for sync_fallback_total**

In `test/test_offline_scheduler.cpp`, add:

```cpp
void test_offline_batch_stats_sync_fallback_total() {
    std::printf("\n--- Test 4: OfflineBatchStats fallback total sync ---\n");

    funasr::OfflineBatchStats stats;
    stats.decode_fallback_calls = 99;
    stats.fallback_reasons.single_request = 4;
    stats.fallback_reasons.token_id_fast_path_unavailable = 6;

    stats.sync_fallback_total();

    TEST_ASSERT(stats.decode_fallback_calls == 10,
                "sync_fallback_total replaces aggregate fallback count");
}
```

Call it from `main()`:

```cpp
test_offline_batch_stats_sync_fallback_total();
```

- [x] **Step 5: Build and run scheduler tests**

Run:

```bash
cmake --build build-cuda --target test_offline_scheduler -j$(nproc)
./build-cuda/test_offline_scheduler
```

Expected: all scheduler tests pass.

---

### Task 2: Track Scheduler-Level Fallback Reasons

**Files:**
- Modify: `pipeline/offline_batching.cpp`
- Test: `test/test_offline_scheduler.cpp`

- [x] **Step 1: Count single-request decode steps**

In `OfflineBatchTranscriber::transcribe_continuous_gpu()`, replace the current aggregate fallback increment:

```cpp
if (decode_inputs.size() == 1) {
    last_stats_.decode_fallback_calls++;
} else {
    last_stats_.decode_group_calls++;
}
```

with:

```cpp
if (decode_inputs.size() == 1) {
    last_stats_.fallback_reasons.single_request++;
} else {
    last_stats_.decode_group_calls++;
}
last_stats_.sync_fallback_total();
```

This preserves the existing meaning where single-request decode is counted as fallback for batching-efficiency purposes.

- [x] **Step 2: Keep aggregate fallback total synchronized at the end**

Before returning from `OfflineBatchTranscriber::transcribe_continuous_gpu()`, add:

```cpp
last_stats_.sync_fallback_total();
```

Also add the same call before returning from the non-GPU `transcribe()` path after all results are collected, so summary output is always internally consistent:

```cpp
last_stats_.sync_fallback_total();
```

- [x] **Step 3: Re-run scheduler tests**

Run:

```bash
cmake --build build-cuda --target test_offline_scheduler -j$(nproc)
./build-cuda/test_offline_scheduler
```

Expected: all scheduler tests pass.

---

### Task 3: Expose Pipeline Paged Fallback Reasons

**Files:**
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/offline_batching.cpp`

- [x] **Step 1: Add decode dispatch status struct**

In `pipeline/pipeline.hpp`, near `GPUDecodeStepOutput`, add:

```cpp
struct GPUDecodeDispatchStats {
    int token_id_fast_path_unavailable = 0;
    int host_embedding_batch_unavailable = 0;
    int serial_env_forced = 0;
    int invalid_paged_input = 0;
};
```

- [x] **Step 2: Add optional stats parameter to gpu_decode_step_slots**

Change the declaration in `pipeline/pipeline.hpp` from:

```cpp
std::vector<GPUDecodeStepOutput> gpu_decode_step_slots(
    const std::vector<GPUDecodeStepInput>& inputs,
    const InferenceConfig& config);
```

to:

```cpp
std::vector<GPUDecodeStepOutput> gpu_decode_step_slots(
    const std::vector<GPUDecodeStepInput>& inputs,
    const InferenceConfig& config,
    GPUDecodeDispatchStats* dispatch_stats = nullptr);
```

Update the matching declaration in `pipeline/recognizer.hpp` so it forwards the optional pointer:

```cpp
std::vector<GPUDecodeStepOutput> gpu_decode_step_slots(
    const std::vector<GPUDecodeStepInput>& inputs,
    const InferenceConfig& config,
    GPUDecodeDispatchStats* dispatch_stats = nullptr) {
    return pipeline_->gpu_decode_step_slots(inputs, config, dispatch_stats);
}
```

- [x] **Step 3: Count invalid paged inputs**

In `Pipeline::gpu_decode_step_slots()`, inside the paged input collection loop, replace:

```cpp
if (input.block_table.empty() || input.n_past >= config.kv_cache_size) {
    continue;
}
```

with:

```cpp
if (input.block_table.empty() || input.block_size <= 0 ||
    input.n_past >= config.kv_cache_size) {
    if (dispatch_stats) {
        dispatch_stats->invalid_paged_input++;
    }
    continue;
}
```

- [x] **Step 4: Count serial env fallback**

In `Pipeline::gpu_decode_step_slots()`, before the serial paged decode branch:

```cpp
if (env_flag_enabled("FUNASR_PAGED_DECODE_SERIAL")) {
```

add:

```cpp
if (env_flag_enabled("FUNASR_PAGED_DECODE_SERIAL") && dispatch_stats) {
    dispatch_stats->serial_env_forced++;
}
```

- [x] **Step 5: Count token-id fast-path fallback**

Where token-id decode returns false and currently prints:

```cpp
printf("[Pipeline] GPU scheduler paged token-id decode unavailable; falling back\n");
```

add:

```cpp
if (dispatch_stats) {
    dispatch_stats->token_id_fast_path_unavailable++;
}
```

immediately before the print.

- [x] **Step 6: Count host-embedding batch fallback**

Where host-embedding paged batch decode returns false and currently prints:

```cpp
printf("[Pipeline] GPU scheduler paged batch token decode failed\n");
return outputs;
```

add:

```cpp
if (dispatch_stats) {
    dispatch_stats->host_embedding_batch_unavailable++;
}
```

immediately before the print.

- [x] **Step 7: Pass dispatch stats from scheduler**

In `pipeline/offline_batching.cpp`, replace:

```cpp
decode_outputs = recognizer_.gpu_decode_step_slots(decode_inputs, inference);
```

with:

```cpp
GPUDecodeDispatchStats dispatch_stats;
decode_outputs = recognizer_.gpu_decode_step_slots(
    decode_inputs, inference, &dispatch_stats);
last_stats_.fallback_reasons.token_id_fast_path_unavailable +=
    dispatch_stats.token_id_fast_path_unavailable;
last_stats_.fallback_reasons.host_embedding_batch_unavailable +=
    dispatch_stats.host_embedding_batch_unavailable;
last_stats_.fallback_reasons.serial_env_forced +=
    dispatch_stats.serial_env_forced;
last_stats_.fallback_reasons.invalid_paged_input +=
    dispatch_stats.invalid_paged_input;
last_stats_.sync_fallback_total();
```

- [x] **Step 8: Build offline batching target**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
```

Expected: build succeeds.

---

### Task 4: Print Fallback Reasons and Throughput Summary

**Files:**
- Modify: `test/test_offline_batching.cpp`

- [x] **Step 1: Count decoded tokens**

In `test/test_offline_batching.cpp`, before the result loop, add:

```cpp
long long total_decode_tokens = 0;
```

Inside the loop over `results`, add:

```cpp
total_decode_tokens += result.decode_tokens;
```

- [x] **Step 2: Print fallback reason line**

After the existing scheduler line, add:

```cpp
std::printf("[OfflineTest] fallback_reasons: single=%d token_id=%d "
            "host_embed=%d serial_env=%d invalid=%d\n",
            stats.fallback_reasons.single_request,
            stats.fallback_reasons.token_id_fast_path_unavailable,
            stats.fallback_reasons.host_embedding_batch_unavailable,
            stats.fallback_reasons.serial_env_forced,
            stats.fallback_reasons.invalid_paged_input);
```

- [x] **Step 3: Print throughput line**

After the fallback reason line, add:

```cpp
const double wall_sec = wall_ms / 1000.0;
std::printf("[OfflineTest] throughput: audio_sec=%.2f wall_sec=%.2f "
            "audio_sec/s=%.2f rtf=%.4f tokens/s=%.1f\n",
            audio_sec,
            wall_sec,
            wall_sec > 0.0 ? audio_sec / wall_sec : 0.0,
            audio_sec > 0.0f ? wall_sec / audio_sec : 0.0,
            wall_sec > 0.0 ? static_cast<double>(total_decode_tokens) / wall_sec : 0.0);
```

- [x] **Step 4: Build and run CPU smoke**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav --kv-mode paged --batch-size 2 --ctx-size 256 --chunk-mode window --chunk-sec 3 --max-tokens 3
```

Expected: output includes `fallback_reasons` and `throughput` lines.

---

### Task 5: Print GPU Memory Budget Hint

**Files:**
- Modify: `test/test_offline_batching.cpp`

- [x] **Step 1: Add KV estimate helper**

In `test/test_offline_batching.cpp`, add this helper near the option parsing helpers:

```cpp
static double estimate_kv_cache_mb(const funasr::ModelConfig& model_cfg,
                                   int ctx_size,
                                   int slots) {
    const double bytes_per_f16 = 2.0;
    const double layer_count = static_cast<double>(model_cfg.llm.block_count);
    const double kv_heads = static_cast<double>(model_cfg.llm.head_count_kv);
    const double head_dim = static_cast<double>(model_cfg.llm.head_dim());
    const double rows = static_cast<double>(ctx_size) * static_cast<double>(slots);
    const double k_bytes = layer_count * rows * kv_heads * head_dim * bytes_per_f16;
    const double v_bytes = k_bytes;
    return (k_bytes + v_bytes) / (1024.0 * 1024.0);
}
```

- [x] **Step 2: Print budget before transcription**

After GPU initialization succeeds and before chunk processing begins, add:

```cpp
if (opt.use_gpu) {
    const double kv_mb = estimate_kv_cache_mb(recognizer.config(),
                                              opt.ctx_size,
                                              opt.batch_size);
    const double weights_mb = 945.0;
    std::printf("[OfflineTest] gpu budget: ctx=%d slots=%d kv_est=%.0fMB "
                "weights_est=%.0fMB mode=%s\n",
                opt.ctx_size,
                opt.batch_size,
                kv_mb,
                weights_mb,
                opt.use_paged_kv ? "paged" : "continuous");
    if (kv_mb + weights_mb > 6500.0) {
        std::printf("[OfflineTest] WARNING: requested ctx=%d slots=%d may exceed "
                    "the comfortable memory budget on 8GB GPUs; try batch=8/12 "
                    "or ctx=2048.\n",
                    opt.ctx_size,
                    opt.batch_size);
    }
}
```

- [x] **Step 3: Build**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
```

Expected: build succeeds.

---

### Task 6: Expose and Print Paged Decode Profile Summary

**Files:**
- Modify: `compute/gpu_runner.hpp`
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_offline_batching.cpp`

- [x] **Step 1: Move PagedDecodeProfile to namespace scope and add helpers**

In `compute/gpu_runner.hpp`, move the existing private nested
`PagedDecodeProfile` struct to namespace scope before `class GPURunner` and
define it as:

```cpp
struct PagedDecodeProfile {
    long calls = 0;
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double set_input_ms = 0.0;
    double compute_ms = 0.0;
    double get_ms = 0.0;

    double total_ms() const {
        return build_ms + alloc_ms + set_input_ms + compute_ms + get_ms;
    }

    double avg_total_ms() const {
        return calls > 0 ? total_ms() / static_cast<double>(calls) : 0.0;
    }
};
```

Remove the old private nested struct definition from inside `GPURunner`.

- [x] **Step 2: Add GPURunner accessor**

In the public section of `GPURunner`, add:

```cpp
const PagedDecodeProfile& paged_decode_profile() const {
    return paged_decode_profile_;
}
```

- [x] **Step 3: Keep accumulation always on**

In `GPURunner::record_paged_decode_profile()`, remove the early return:

```cpp
if (!env_enabled("FUNASR_PROFILE_PAGED_DECODE")) {
    return;
}
```

Keep the existing periodic `printf` guarded by:

```cpp
if (env_enabled("FUNASR_PROFILE_PAGED_DECODE") &&
    paged_decode_profile_.calls % 50 == 0) {
```

This makes the final benchmark summary available by default while preserving
the opt-in noisy per-50-call debug log.

- [x] **Step 4: Add Pipeline accessor**

In `pipeline/pipeline.hpp`, add:

```cpp
PagedDecodeProfile gpu_paged_decode_profile() const;
```

In `pipeline/pipeline.cpp`, implement:

```cpp
PagedDecodeProfile Pipeline::gpu_paged_decode_profile() const {
#ifdef FUNASR_USE_CUDA
    if (gpu_runner_) {
        return gpu_runner_->paged_decode_profile();
    }
#endif
    return PagedDecodeProfile{};
}
```

- [x] **Step 5: Add Recognizer forwarding accessor**

In `pipeline/recognizer.hpp`, add:

```cpp
PagedDecodeProfile gpu_paged_decode_profile() const {
    return pipeline_->gpu_paged_decode_profile();
}
```

- [x] **Step 6: Print profile summary**

In `test/test_offline_batching.cpp`, after the throughput line, add:

```cpp
if (opt.use_gpu && opt.use_paged_kv) {
    const funasr::PagedDecodeProfile profile = recognizer.gpu_paged_decode_profile();
    if (profile.calls > 0) {
        const double calls = static_cast<double>(profile.calls);
        std::printf("[OfflineTest] paged_profile: calls=%ld build=%.3fms "
                    "alloc=%.3fms set=%.3fms compute=%.3fms get=%.3fms total=%.3fms\n",
                    profile.calls,
                    profile.build_ms / calls,
                    profile.alloc_ms / calls,
                    profile.set_input_ms / calls,
                    profile.compute_ms / calls,
                    profile.get_ms / calls,
                    profile.avg_total_ms());
    }
}
```

- [x] **Step 7: Build and smoke test**

Run:

```bash
cmake --build build-cuda --target test_offline_batching -j$(nproc)
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav --kv-mode paged --batch-size 2 --ctx-size 256 --chunk-mode window --chunk-sec 3 --max-tokens 3
```

Expected: CPU smoke still passes. The `paged_profile` line may be absent without GPU paged decode calls.

---

### Task 7: Final Verification

**Files:**
- Verify all modified files.

- [x] **Step 1: Build both targets**

Run:

```bash
cmake --build build-cuda --target test_offline_scheduler test_offline_batching -j$(nproc)
```

Expected: build succeeds.

- [x] **Step 2: Run scheduler tests**

Run:

```bash
./build-cuda/test_offline_scheduler
```

Expected:

```text
Tests failed: 0
```

- [x] **Step 3: Run CPU smoke**

Run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav --kv-mode paged --batch-size 2 --ctx-size 256 --chunk-mode window --chunk-sec 3 --max-tokens 3
```

Expected:

```text
[OfflineTest] ok=2/2
[OfflineTest] fallback_reasons:
[OfflineTest] throughput:
```

- [x] **Step 4: Ask user for GPU verification**

Ask the user to run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --chunk-mode window --chunk-sec 30 --max-tokens 220
```

Expected GPU signals:

```text
[OfflineTest] fallback_reasons:
[OfflineTest] throughput:
[OfflineTest] paged_profile:
```

For `batch=16 --ctx-size 4096` on an 8 GB GPU, expected signal:

```text
[OfflineTest] WARNING: requested ctx=4096 slots=16 may exceed the comfortable memory budget on 8GB GPUs
```

---

## Self-Review

- Spec coverage: fallback reasons, throughput summary, memory budget hints, profile summary, tests, and GPU validation are all covered.
- Scope check: graph cache, batch prefill, and KV allocation redesign are intentionally excluded.
- Placeholder scan: no TBD/TODO placeholders remain; Task 5 uses the confirmed `Recognizer::config()` accessor.
- Type consistency: `OfflineDecodeFallbackStats`, `OfflineBatchStats`, `GPUDecodeDispatchStats`, and `PagedDecodeProfile` are introduced before use.
