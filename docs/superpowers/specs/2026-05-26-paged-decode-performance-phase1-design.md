# Paged Decode Performance Phase 1 Design

## Goal

Improve the already-working paged continuous offline decode path by identifying
the best low-risk performance lever before changing CUDA kernels. Phase 1 starts
with repeatable block-size benchmarking and uses the existing paged decode
profile to decide whether kernel work or graph caching should come next.

## Current State

The vLLM-style offline MVP is functionally in place:

- paged KV block allocation is active;
- continuous decode keeps the active batch nearly full;
- variable `n_past` paged decode works through per-request KV lengths;
- token-id GPU fast path is clean, with latest fallback reasons showing
  `token_id=0`, `host_embed=0`, and `invalid=0`;
- `batch=12`, `ctx=4096`, `block_size=16` on the tested 8 GB GPU reached about
  `RTF=0.0126` and roughly `79x` realtime audio throughput before the sweep.
- `block_size=64` and `block_size=128` improved the same workload to about
  `RTF=0.0114`, or roughly `87-88x` realtime audio throughput.

The latest profile shows compute dominates:

```text
build=1.631ms alloc=0.278ms set=0.194ms compute=23.396ms get=0.103ms total=25.601ms
```

That makes CUDA attention compute and memory access more important than graph
construction for the next optimization decision.

## Phase 1A: Block Size Sweep

Run the same workload with several paged KV block sizes:

```text
8, 16, 32, 64
48, 64, 96, 128
```

Compare:

- wall time;
- RTF;
- audio seconds per wall second;
- decoded tokens per wall second;
- `paged_profile.compute`;
- `paged_profile.total`;
- `fallback_calls`;
- `blocks_peak`;
- output sanity.

The sweep must run each block size in a fresh process so GPU context, allocator
state, KV cache state, and profile counters do not leak across runs.

## Decision Rules

Choose the best block size by wall time and `tokens/s`, not by block utilization
alone.

- If `64` improves wall time without a large memory penalty, make it the
  conservative paged offline default.
- If `128` improves long-window wall time and fallback counts remain unchanged,
  make it the long-video throughput recommendation.
- If `8` improves wall time, fragmentation was more important than block-table
  overhead.
- If all sizes are close, keep `16` as the conservative default and move to
  kernel-level profiling.
- If larger block sizes trigger memory pressure or worse wall time, keep them as
  high-memory-only options.

## Measured Result

On the tested 8 GB RTX 4070 Laptop GPU with `batch=12`, `ctx=4096`,
`chunk_sec=30`, and `max_tokens=220`:

| block | wall | RTF | audio x realtime | tokens/s | fallback |
|---:|---:|---:|---:|---:|---:|
| 16 | 77.27s | 0.0126 | 79.20x | 248.8 | 6 |
| 64 | 70.02s | 0.0114 | 87.40x | 274.5 | 6 |
| 128 | 69.53s | 0.0114 | 88.01x | 276.4 | 6 |

The project default should move from `16` to `64`. For long-video 30-second
window workloads, `128` is the current throughput recommendation.

## Phase 1B: Kernel Profiling Direction

If the sweep shows `compute` remains dominant, inspect the paged attention CUDA
kernel for:

- repeated block table reads;
- non-coalesced K/V loads;
- softmax reduction cost;
- batch-to-thread-block mapping;
- special handling for decode-only `q_len=1`.

Kernel changes are intentionally outside Phase 1A.

The original baseline kernel can still be forced with:

```bash
FUNASR_PAGED_ATTN_V1=1
```

This keeps a simple correctness and performance comparison path after the
optimized kernel becomes the default.

The intermediate warp-QK variant can still be forced with:

```bash
FUNASR_PAGED_ATTN_V2=1
```

It changes only the QK score phase: one warp cooperates on one KV position dot
product, spreading `head_dim=128` across lanes. Softmax and V aggregation remain
the same as the v1 kernel. This keeps the experiment narrow and makes A/B
comparison straightforward.

The default paged attention kernel is now the former v3 path. It includes the
v2 warp-QK path and additionally caches each logical position's physical KV row
in shared memory. The V aggregation phase then reuses this row table instead of
recomputing `block_table[pos / block_size] * block_size + pos % block_size` for
every output dimension.

The V-split experiment was rejected after measurement: it regressed wall time
and paged-attention kernel time, so its code and environment switch were
removed to avoid future confusion.

Measured kernel direction on the tested workload:

| kernel | wall | tokens/s | avg paged-attn kernel | graph compute share |
|---|---:|---:|---:|---:|
| v1 forced | 68.19s | 281.9 | 0.262ms | 32.1% |
| v2 forced | 66.62s | 287.9 | 0.208ms | 26.7% |
| default v3 | 61.77s | 310.5 | 0.149ms | 21.6% |
| rejected V-split | 68.27s | 282.7 | 0.238ms | 29.4% |

### Paged Attention Event Profile

Use explicit diagnostic profiling before changing the kernel:

```bash
FUNASR_PROFILE_PAGED_ATTN=1 ./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220
```

This mode inserts CUDA events around each `GGML_OP_PAGED_ATTN_EXT` kernel and
synchronizes after the event. It is meant to estimate the paged attention share
inside decode graph compute. Do not compare its wall time against normal
benchmarks because the per-kernel synchronization changes scheduling behavior.

Expected additional output:

```text
[OfflineTest] paged_attn_profile: calls=46900 total=...ms avg_kernel=...ms graph_compute_share=...%
```

Compare the default optimized kernel against forced v1 and v2 with:

```bash
FUNASR_PROFILE_PAGED_ATTN=1 ./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220

FUNASR_PROFILE_PAGED_ATTN=1 FUNASR_PAGED_ATTN_V1=1 ./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220

FUNASR_PROFILE_PAGED_ATTN=1 FUNASR_PAGED_ATTN_V2=1 ./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 --max-tokens 220
```

## Validation

Use the new sweep helper:

```bash
tools/bench_paged_block_sizes.sh FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --batch-size 12 --ctx-size 4096 --sizes "32 64 96 128"
```

The script writes per-run logs and a TSV summary under
`outputs/bench_paged_blocks/<timestamp>/`.

## Completion Criteria

- The sweep script runs each block size as an isolated process.
- The summary contains wall, RTF, throughput, fallback, and profile metrics.
- The next kernel/cache optimization is chosen from measured data rather than
  intuition.
