# Scheduler Admission Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scheduler-level admission and prefill/decode stall diagnostics so the next vLLM-style optimization is guided by measured data.

**Architecture:** Keep CUDA kernels and decode behavior unchanged. Extend `OfflineBatchStats`, count admission outcomes in `OfflineBatchTranscriber`, time prefill/admission and decode dispatch separately, then print one compact scheduler profile line from `test_offline_batching`.

**Tech Stack:** C++17, CMake, existing offline scheduler tests and `test_offline_batching`.

---

## File Structure

- Modify `pipeline/offline_batching.hpp`: add scheduler admission/profile counters and average helpers.
- Modify `test/test_offline_scheduler.cpp`: add unit coverage for the new helper methods.
- Modify `pipeline/offline_batching.cpp`: increment admission counters and accumulate prefill/decode wall time.
- Modify `test/test_offline_batching.cpp`: print the new `scheduler_profile` summary line.

### Task 1: Add Stats Helpers

- [x] Add failing unit test for empty and populated admission/profile helper methods.
- [x] Add fields and helper methods to `OfflineBatchStats`.
- [x] Run `test_offline_scheduler`.

### Task 2: Wire Scheduler Counters

- [x] Count admission attempts, successes, and no-KV-capacity outcomes in `admit_one()`.
- [x] Accumulate prefill wall time around GPU prefill.
- [x] Accumulate decode dispatch wall time around `gpu_decode_step_slots()`.
- [x] Count idle scheduler loops where no decode input is available.

### Task 3: Print Summary

- [x] Add `[OfflineTest] scheduler_profile: ...` output after fallback reasons.
- [x] Build `test_offline_batching`.
- [x] Run CPU paged smoke test and scheduler unit test.

### Task 4: Add Batched Prefill Opportunity Counters

- [x] Add failing unit test for admission round helper methods.
- [x] Track each natural scheduler refill round.
- [x] Print `admit_rounds`, `avg_admit_round`, and `max_admit_round`.
- [x] Build and run local verification.

### Task 5: Add Decode Graph Cache Feasibility Probe

- [x] Add failing unit test for paged decode cache probe hit-rate helpers.
- [x] Track paged token decode shape-cache and full-graph-cache probe hits.
- [x] Print `paged_graph_cache_probe` when paged GPU decode was used.
- [x] Build and run local verification.

### Task 6: Split Cache Probe Params From Copy Rows

- [x] Add failing unit test for `param_cache_probe_hit_rate()`.
- [x] Track `param_cache_probe_hits` as shape plus `max_n_kv` stability.
- [x] Print `param_hits` and `param_hit_rate` in `paged_graph_cache_probe`.
- [x] Build and run local verification.

### Task 7: Add Bucketed `max_n_kv` Experiment

- [x] Add failing unit test for `paged_decode_graph_max_n_kv()`.
- [x] Gate bucketed paged decode graph `max_n_kv` behind `FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1`.
- [x] Pass the graph `max_n_kv` from `GPURunner` into the paged batch decode graph builder.
- [x] Build and run local verification.

### Task 8: Add Dynamic Paged KV Write Experiment

- [x] Add CUDA-only `GGML_OP_PAGED_KV_WRITE_EXT` and public constructor.
- [x] Implement a minimal F32-to-F16 paged KV write CUDA kernel.
- [x] Gate the new write path behind `FUNASR_PAGED_KV_WRITE_OP=1`.
- [x] Keep the old per-row `ggml_cpy` view path as default fallback.
- [x] Build and run local verification.

### Task 9: Add Token Decode Graph Cache Experiment

- [x] Add actual graph-cache hit/miss profile helpers.
- [x] Cache the last paged token-id decode graph when dynamic KV write and bucketed `max_n_kv` make the graph signature stable.
- [x] Gate the experiment behind `FUNASR_PAGED_DECODE_GRAPH_CACHE=1`.
- [x] Invalidate the cached graph before any other decode path reuses the shared decode allocator.
- [x] Build and run local verification.

### Validation

```bash
cmake --build build-cuda --target test_offline_scheduler test_offline_batching -j$(nproc)
./build-cuda/test_offline_scheduler
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav --kv-mode paged --batch-size 2 --ctx-size 256 --chunk-mode window --chunk-sec 3 --max-tokens 3
```

GPU validation is user-run:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220
```
