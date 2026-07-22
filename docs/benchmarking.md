# Unified ASR Benchmarking

`tools/benchmark_suite.py` runs the local C++ baselines and optional external
Fun-ASR engines with one result schema. It prepares identical fixed WAV windows
for external engines and keeps every raw log beside CSV and Markdown summaries.

## Quick Local Check

Build the latest local binary and process two 30-second windows:

```bash
python3 tools/benchmark_suite.py \
  --model FunAsr_q8.bin \
  --audio outputs/video_asr/20260502_130430/media/source_16k.wav \
  --engines cpp-b12 \
  --repeat 1 --warmup 0 --max-chunks 2 \
  --build
```

This is a smoke test, not a resume number. Use the complete workload and at
least three measured repeats for a reportable result:

```bash
python3 tools/benchmark_suite.py \
  --model FunAsr_q8.bin \
  --audio outputs/video_asr/20260502_130430/media/source_16k.wav \
  --engines cpp-b1,cpp-b12 \
  --repeat 3 --warmup 1 \
  --build --quiet
```

## Optional Engines

The supported engine names are:

```text
cpp-b1          local Q8_0 C++ runtime, continuous KV, batch 1
cpp-b12         local Q8_0 C++ runtime, paged KV, batch 12
funasr-pytorch  official AutoModel path
funasr-vllm     official AutoModelVLLM path
sherpa-onnx     sherpa-onnx FunASR Nano INT8 path
```

Keep vLLM and sherpa-onnx in separate environments so their CUDA, PyTorch, and
ONNX Runtime dependencies do not replace one another:

```bash
python3 tools/benchmark_suite.py \
  --model FunAsr_q8.bin \
  --audio outputs/video_asr/20260502_130430/media/source_16k.wav \
  --engines cpp-b1,cpp-b12,funasr-pytorch,funasr-vllm,sherpa-onnx \
  --pytorch-python /path/to/funasr-env/bin/python \
  --vllm-python /path/to/vllm-env/bin/python \
  --sherpa-python /path/to/sherpa-env/bin/python \
  --sherpa-model-dir /path/to/sherpa-onnx-funasr-nano-int8-2025-12-30 \
  --repeat 3 --warmup 1 --build --quiet
```

### Current WSL Environments

The following interpreters have been checked on the current workstation:

```text
/home/hua/miniconda3/envs/funasr_env/bin/python
  FunASR 1.3.1, PyTorch 2.5.1+cu121

/home/hua/miniconda3/envs/funasr_vllm/bin/python
  FunASR 1.3.15, vLLM 0.20.1, PyTorch 2.11.0+cu130
```

The vLLM combination follows the FunASR guidance for a driver advertising
CUDA 13 support: let vLLM own the matching PyTorch CUDA packages, then install
FunASR. Both the `AutoModelVLLM` import and CUDA device discovery have been
verified.

Run the fixed-window comparison with:

```bash
python3 tools/benchmark_suite.py \
  --model FunAsr_q8.bin \
  --audio outputs/video_asr/20260502_130430/media/source_16k.wav \
  --engines cpp-b1,cpp-b12,funasr-pytorch,funasr-vllm \
  --pytorch-python /home/hua/miniconda3/envs/funasr_env/bin/python \
  --vllm-python /home/hua/miniconda3/envs/funasr_vllm/bin/python \
  --official-hub ms \
  --repeat 3 --warmup 1 --build --quiet
```

On this machine, loading the official PyTorch model was killed by the WSL OOM
killer while WSL was limited to about 7.3 GiB RAM and 2 GiB swap. The model
checkpoint is about 2 GiB, and the legacy FunASR loader temporarily holds the
constructed model, the loaded checkpoint, and a deep-copied state dictionary.
This is host-memory pressure, not an 8 GiB GPU-memory failure.

Before running the Python baselines, raise the WSL limit in the Windows user
profile file `%UserProfile%\.wslconfig`, for example:

```ini
[wsl2]
memory=16GB
swap=8GB
```

Then run `wsl --shutdown` from PowerShell and reopen WSL. Confirm with
`free -h` before starting the benchmark. If an engine exits with code `-9`
without a result JSON, inspect `dmesg` for an OOM-killer record before changing
CUDA or model settings.

The sherpa model directory must contain:

```text
encoder_adaptor.int8.onnx
llm.int8.onnx
embedding.int8.onnx
Qwen3-0.6 B/tokenizer.json
```

An unavailable package or model is recorded as `skipped` with a reason. Other
engines continue running. Use `--dry-run` to inspect every command before a
long benchmark.

## Artifacts

Each run creates `outputs/benchmark_suite/<timestamp>/` containing:

```text
records.csv              every measured repeat
summary.csv              median and wall-time range by engine
summary.md               human-readable comparison table
metadata.json            GPU, Git revision, workload and timing scope
logs/                    complete stdout/stderr for every engine
engine_results/          raw JSON from optional Python engines
texts/                   per-window hypotheses from optional engines
workload/manifest.json   exact fixed-window boundaries
```

`peak_vram_mib` is the maximum total memory-used value observed by
`nvidia-smi` during the process. Close unrelated GPU applications before a
formal run because this is a device-level measurement rather than isolated
process accounting.

## Fair Comparison Rules

Use two separate result tables.

### Fixed-Window Engine Benchmark

- Use the same PCM WAV and external 30-second boundaries.
- Disable VAD differences by feeding the prepared windows to external engines.
- Keep greedy decoding, generation cap, language/hotword settings and success
  criteria aligned.
- Exclude model loading from inference wall time, but report load time.
- Warm up first, then report the median of at least three repeats.
- Record model identity, precision, GPU, driver, batch policy and Git revision.

This table is suitable for local C++, official FunASR PyTorch/vLLM, and
sherpa-onnx because they run Fun-ASR-Nano-2512.

### Full-Pipeline Product Benchmark

- Feed the same unsplit long recording to each product's natural API.
- Allow each engine to use its intended VAD/chunking and timestamp pipeline.
- Report feature differences such as speaker diarization and forced alignment.

Do not mix this table with the fixed-window engine table. It answers product
throughput, not decoder-runtime efficiency.

Whisper.cpp and faster-whisper use different Whisper models. Put them in a
separate system-reference table with explicit model, precision, beam size, CER
or WER, memory, and timing scope. Their raw RTFx is not evidence that one
Fun-ASR runtime implementation is faster than another.

## Accuracy

Long-video throughput without a reference transcript is not enough for a
resume claim. Run AISHELL-1 separately and calculate CER with:

```bash
python3 tools/eval_cer.py results.txt \
  /path/to/data_aishell/transcript/aishell_transcript_v0.8.txt
```

Only compare throughput when the CER difference remains within an explicitly
reported tolerance.

### Strict CER/WER gate

Run the AISHELL split through the same optimized unified scheduler used by the
long-video benchmark:

```bash
tools/bench_asr_accuracy.sh
```

The default gate requires 100% hypothesis coverage, no missing, empty,
duplicate, extra, or malformed records, and corpus CER no higher than 2.2%.
It writes `hypotheses.tsv`, the complete `expected_ids.txt`, inference logs,
and JSON/Markdown accuracy reports under
`outputs/accuracy_benchmark/<timestamp>/`.
Strict gates require the expected-ID manifest; this prevents a crashed or
skipped request from being hidden by evaluating only intersecting IDs.

For a quick smoke test, limit the sorted corpus and set a threshold appropriate
for that subset explicitly:

```bash
tools/bench_asr_accuracy.sh --limit 100 --max-cer 10
```

The evaluator also supports whitespace-tokenized WER and regression against a
saved baseline:

```bash
python3 tools/eval_cer.py candidate.tsv references.txt \
  --expected-ids expected_ids.txt \
  --metrics both --baseline baseline.tsv --gate \
  --max-cer 3.0 --max-cer-regression 0.1 \
  --max-wer 8.0 --max-wer-regression 0.2
```

AISHELL Chinese hypotheses do not carry reliable word boundaries, so CER is
the release metric for that corpus. WER is intended for corpora where both
reference and hypothesis contain real whitespace-delimited words.

## Optimized Runtime Matrix

Use the dedicated matrix to compare graph-cache capacity, scheduler
concurrency, physical KV budget, CPU Fbank prefetch, frontend length buckets,
and the experimental second CUDA stream under one fixed workload:

```bash
tools/bench_unified_runtime.sh --repeat 3 \
  --profiles \
  graph1:40:192:1:off:4:off,optimized40:40:192:16:on:32:off,optimized48:48:224:16:on:32:off,optimized56:56:256:16:on:32:off
```

Each profile is
`name:batch:blocks:graph_entries:prefetch:bucket_window:gpu_overlap`.
Raw logs and a tab-separated summary are written below
`outputs/bench_unified_runtime/<timestamp>/`.

The product CLI exposes the selected configuration without requiring users to
reproduce the matrix manually:

```bash
./build-cuda/funasr-cli -m FunAsr_q8.bin -f long_video.wav \
  --gpu --offline-preset long-video -o transcript.txt
```

On the RTX 4070 Laptop GPU, the preset currently selects batch 48 and 224
128-token physical blocks. Explicit CLI batch/KV options override auto tuning;
`long-video-legacy` preserves the old path for regression isolation.

### 2026-07-22 measured results

For the 6119.29-second, 204-window long-video workload, three-repeat medians
were:

| Profile | Median wall | Audio xRT | Peak blocks | Repeat consistency |
|---|---:|---:|---:|---|
| batch 48 / 224 blocks | **23.097 s** | 264.94x | 224 | 0/204 token mismatches |
| batch 56 / 256 blocks | 23.263 s | 263.05x | 256 | first run differed on 12/204 chunks |

Batch 48 is therefore the automatic default: batch 56 consumes more KV memory
without a median throughput gain. The manual 56/256 profile remains useful for
other GPUs and workload experiments.

The short-request matrix selected batch 48 with 112 physical blocks. The
complete AISHELL-1 test produced 7176/7176 hypotheses, no empty or missing
records, CER 2.0035%, wall 151.778 s, and 237.91x realtime throughput. Relative
to the earlier 179.684-second run, wall time fell by 15.53%. Relative to the
same runtime at batch 40 / 192 blocks (166.235 s), it fell by another 8.70%
while the physical KV pool shrank by 41.67%. Frontend padding was 5.57%, CPU
Fbank prefetch was ready without blocking on 98.38% of batches, and mixed CUDA
Graph hit rate rose from the original 4.24% to 6.22%.

The graph cache uses a bounded LRU and only retains a shape after its second
observation. This prevents one-off variable-length prefill graphs from evicting
hot decode graphs. True tensor padding across arbitrary mixed prefill shapes is
not enabled: safe padding needs dummy KV ownership and an accuracy gate, rather
than changing graph signatures alone.

# Paged KV Pool and Concurrency Matrix

After building the CUDA benchmark target, run all scheduler concurrency points
against the same 160-block physical KV pool:

```bash
tools/bench_paged_kv_pool_concurrency.sh \
  --build --warmup 1 --repeat 3 --blocks 160
```

Use `--max-chunks 24` for a short correctness smoke test. A valid comparison
keeps the model, audio, chunk plan, block count, block size, and CUDA graph
environment identical across batch sizes.
