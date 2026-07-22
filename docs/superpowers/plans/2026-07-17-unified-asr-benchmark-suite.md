# Unified ASR Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reproducible command that benchmarks the local C++ runtime and optional FunASR PyTorch, FunASR-vLLM, and sherpa-onnx engines on identical fixed audio windows, then emits machine-readable records and a resume-ready Markdown table.

**Architecture:** A dependency-free Python common module owns WAV splitting, local log parsing, record validation, median aggregation, and CSV/Markdown reporting. The suite process orchestrates the local C++ executable and optional Python-engine subprocesses; the external runner lazily imports each framework so unavailable environments become explicit skipped engines instead of breaking local measurements.

**Tech Stack:** Python 3 standard library, `unittest`, existing C++ `test_offline_batching`, optional FunASR/vLLM/sherpa-onnx Python environments, `nvidia-smi` metadata capture.

---

### Task 1: Freeze the result contract and parser behavior

**Files:**
- Create: `test/test_benchmark_suite.py`
- Create: `tools/benchmark_common.py`

- [ ] **Step 1: Write failing tests for `parse_offline_log()`**

The fixture must contain the current `[OfflineTest]` wall, scheduler, KV runtime, scheduler profile, throughput, paged profile, and graph-cache lines. Assert numeric extraction for `wall_sec`, `rtf`, `rtfx`, `tokens_s`, `prefill_sec`, `decode_sec`, `fallback_calls`, `peak_blocks`, and `graph_hit_rate`.

- [ ] **Step 2: Run the parser test and verify RED**

Run: `python3 -m unittest test.test_benchmark_suite.BenchmarkCommonTests.test_parse_offline_log -v`

Expected: import failure because `tools.benchmark_common` does not exist.

- [ ] **Step 3: Implement strict local log parsing**

Implement helpers that parse `key=value` fields without depending on field order. Reject logs without a successful `ok=N/N` line or wall/throughput metrics.

- [ ] **Step 4: Run the parser test and verify GREEN**

Run: `python3 -m unittest test.test_benchmark_suite.BenchmarkCommonTests.test_parse_offline_log -v`

Expected: one passing test.

### Task 2: Add reproducible workload preparation and aggregation

**Files:**
- Modify: `test/test_benchmark_suite.py`
- Modify: `tools/benchmark_common.py`

- [ ] **Step 1: Write failing WAV split and aggregation tests**

Generate a tiny PCM WAV in a temporary directory. Assert exact fixed-window frame counts, stable chunk IDs, preserved WAV parameters, median RTFx, min/max wall time, and Markdown labels.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python3 -m unittest test.test_benchmark_suite -v`

Expected: missing `split_wav_fixed`, `aggregate_records`, or report functions.

- [ ] **Step 3: Implement minimal workload and reporting helpers**

Use the standard-library `wave`, `csv`, `json`, and `statistics` modules. Write `records.csv`, `summary.csv`, `summary.md`, and `metadata.json`; preserve failed/skipped records and their reasons.

- [ ] **Step 4: Run all common-module tests and verify GREEN**

Run: `python3 -m unittest test.test_benchmark_suite -v`

Expected: all tests pass.

### Task 3: Orchestrate local C++ baselines

**Files:**
- Modify: `test/test_benchmark_suite.py`
- Create: `tools/benchmark_suite.py`

- [ ] **Step 1: Write failing command-construction and dry-run tests**

Assert that `cpp-b1` selects continuous KV and batch 1, while `cpp-b12` selects paged KV, batch 12, dynamic blocks on, prefix cache off, and the common graph-cache environment.

- [ ] **Step 2: Run command tests and verify RED**

Run: `python3 -m unittest test.test_benchmark_suite.BenchmarkSuiteTests -v`

Expected: missing suite module or command builder.

- [ ] **Step 3: Implement the local runner and CLI**

Support `--model`, `--audio`, `--bin`, `--engines`, `--repeat`, `--warmup`, `--max-chunks`, `--chunk-sec`, `--max-tokens`, `--out`, `--build`, and `--dry-run`. Stream output to per-run log files and parse each completed run into the common record schema.

- [ ] **Step 4: Verify dry-run and local command tests**

Run: `python3 tools/benchmark_suite.py --model FunAsr_q8.bin --audio outputs/video_asr/20260502_130430/media/source_16k.wav --engines cpp-b1,cpp-b12 --repeat 1 --max-chunks 2 --dry-run`

Expected: both commands print without launching inference.

### Task 4: Add optional framework adapters

**Files:**
- Modify: `test/test_benchmark_suite.py`
- Create: `tools/benchmark_python_engines.py`
- Modify: `tools/benchmark_suite.py`

- [ ] **Step 1: Write failing engine-plan and result-contract tests**

Assert lazy dependency checks, model-path validation for sherpa-onnx, and compatible JSON output for `funasr-pytorch`, `funasr-vllm`, and `sherpa-onnx`.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest test.test_benchmark_suite.PythonEngineTests -v`

Expected: missing external runner functions.

- [ ] **Step 3: Implement framework runners**

FunASR PyTorch uses `AutoModel`; FunASR-vLLM uses `AutoModelVLLM`; sherpa-onnx uses `OfflineRecognizer.from_funasr_nano` and grouped `decode_streams`. Imports occur only inside the selected runner. Model loading is reported separately and excluded from timed inference. Warmup and repeats remain in one process so engine caches survive measured repeats.

- [ ] **Step 4: Verify dependency-missing behavior**

Run each adapter with `--check-only` in the current environment. Expected: exit code 0 with JSON status `available` or `skipped`, never a Python traceback for a missing optional dependency.

### Task 5: Document the fair-comparison protocol

**Files:**
- Create: `docs/benchmarking.md`
- Modify: `README.md`

- [ ] **Step 1: Document one-command examples**

Include local quick/full commands, separate Python executable flags for isolated conda environments, expected output files, and sherpa model filenames.

- [ ] **Step 2: Document fairness rules**

State the fixed-window engine benchmark versus natural full-pipeline benchmark distinction, model/precision labels, warmup/repeat policy, model-load timing scope, CER requirement, GPU power/thermal metadata, and why Whisper-family numbers are a separate system-reference table.

- [ ] **Step 3: Verify commands and links**

Run every `--help`/`--dry-run` command shown in the documentation and check local links.

### Task 6: End-to-end verification

**Files:**
- No production file changes expected.

- [ ] **Step 1: Run the complete Python test set**

Run: `python3 -m unittest discover -s test -p 'test_*.py' -v`

Expected: all Python tests pass.

- [ ] **Step 2: Run a two-chunk local GPU smoke benchmark**

Run the suite with `cpp-b12`, one repeat, no warmup, and `--max-chunks 2`.

Expected: status `ok`, valid `records.csv`, `summary.csv`, `summary.md`, and local log.

- [ ] **Step 3: Inspect generated artifacts**

Confirm the Markdown and CSV values agree with the raw `[OfflineTest]` output and that GPU metadata identifies the RTX 4070 Laptop GPU.
