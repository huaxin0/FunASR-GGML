# Paged Decode Performance Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repeatable block-size benchmarking for the paged continuous offline decode path and apply the measured block-size default.

**Architecture:** Keep model execution unchanged. Add a shell benchmark helper that runs `test_offline_batching` in a fresh process for each KV block size and extracts the existing scheduler, throughput, fallback, and paged profile metrics into a TSV summary.

**Tech Stack:** Bash, CMake, existing `test_offline_batching`, existing paged benchmark output.

---

## File Structure

- Modify `pipeline/offline_batching.hpp`: set the paged KV default block size.
- Modify `test/test_offline_batching.cpp`: align the benchmark harness default and usage text.
- Modify `tools/bench_paged_block_sizes.sh`: runs isolated block-size sweep and writes logs plus `summary.tsv`.
- Create `docs/superpowers/specs/2026-05-26-paged-decode-performance-phase1-design.md`: describes the measurement-first optimization stage.
- Create `docs/superpowers/plans/2026-05-26-paged-decode-performance-phase1.md`: tracks this implementation.

---

### Task 1: Add Block Size Sweep Helper

**Files:**
- Create: `tools/bench_paged_block_sizes.sh`

- [x] **Step 1: Add CLI wrapper**

The script accepts:

```bash
tools/bench_paged_block_sizes.sh <model.bin> <audio.wav/mp3/flac> \
  --batch-size 12 --ctx-size 4096 --sizes "32 64 96 128"
```

- [x] **Step 2: Run each block size in a fresh process**

Each block size invokes:

```bash
./build-cuda/test_offline_batching <model> <audio> \
  --gpu --kv-mode paged --kv-block-size <size> \
  --batch-size <batch> --ctx-size <ctx> \
  --chunk-mode window --chunk-sec <chunk-sec> --max-tokens <max-tokens>
```

- [x] **Step 3: Save per-run logs and summary**

The script writes:

```text
outputs/bench_paged_blocks/<timestamp>/block_<size>.log
outputs/bench_paged_blocks/<timestamp>/summary.tsv
```

The summary columns are:

```text
block_size status wall_ms rtf audio_sec_per_s tokens_per_s avg_active fallback_calls compute_ms total_step_ms log
```

---

### Task 2: Document Phase 1

**Files:**
- Create: `docs/superpowers/specs/2026-05-26-paged-decode-performance-phase1-design.md`
- Create: `docs/superpowers/plans/2026-05-26-paged-decode-performance-phase1.md`

- [x] **Step 1: Record current state**

The design notes that paged continuous decode MVP and stabilization are already
working, with fallback causes clean on the latest `batch=12` run.

- [x] **Step 2: Define decision rules**

The design specifies how to choose between block sizes using wall time,
`tokens/s`, `paged_profile.compute`, memory pressure, and fallback counts.

---

### Task 3: Local Verification

**Files:**
- Verify: `tools/bench_paged_block_sizes.sh`

- [x] **Step 1: Check shell syntax**

Run:

```bash
bash -n tools/bench_paged_block_sizes.sh
```

Expected: exit code 0.

- [x] **Step 2: Check help output**

Run:

```bash
tools/bench_paged_block_sizes.sh --help
```

Expected: usage text prints and exit code 0.

- [x] **Step 3: Confirm GPU validation command**

Ask the user to run:

```bash
tools/bench_paged_block_sizes.sh FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --batch-size 12 --ctx-size 4096 --sizes "8 16 32 64"
```

Expected: `summary.tsv` contains one row per block size.

---

### Task 4: Apply Measured Block-Size Default

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `test/test_offline_batching.cpp`
- Modify: `docs/superpowers/specs/2026-05-26-paged-decode-performance-phase1-design.md`

- [x] **Step 1: Change core default to 64**

`OfflineBatchConfig::kv_block_size` now defaults to `64`.

- [x] **Step 2: Change benchmark harness default to 64**

`test_offline_batching` now defaults to `--kv-block-size 64` and prints that in
usage text.

- [x] **Step 3: Record measured recommendation**

The design now records:

```text
default: 64
long-video throughput: 128
```
