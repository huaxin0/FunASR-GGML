# Offline Decode Kernel Profiling Notes

Date: 2026-06-11

## Summary

This profiling pass investigated the current long-video offline ASR path after
the vLLM-style scheduler, paged KV cache, bucketed decode shape, and graph cache
were already working.

The main result is that the scheduler layer is healthy. The next performance
work should focus on decoder compute kernels, especially Q8 matmul/fixup and
paged attention, rather than more scheduler-level changes.

## Baseline Configuration

The current long-video throughput configuration is:

```bash
FUNASR_PAGED_KV_WRITE_OP=1
FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1
FUNASR_PAGED_DECODE_GRAPH_CACHE=1

./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

Representative full-run profile:

```text
audio_sec=6119.29
chunks=204
wall total ~= 56-59s
rtf ~= 0.009
audio_sec/s ~= 100+
avg_active ~= 11.47
fallback_calls = 2
graph_cache_hit_rate = 80.74%
```

This confirms:

- continuous batching is active;
- batch size 12 is mostly filled;
- paged KV is active;
- token-id paged decode fast path is active;
- fallback is not the bottleneck;
- graph cache is working.

## Graph Cache And Bucket A/B

Three variants were compared on the same workload:

| Variant | Wall | Tokens/s | Notes |
|---|---:|---:|---|
| All optimizations on | 56.28s | 340.9 | bucket + graph cache + dynamic KV write |
| Graph cache off | 57.63s | 332.9 | build/alloc higher |
| Bucket off | 59.34s | 323.3 | cache hit disappears |

Key observations:

- graph cache gives about 2% wall-time improvement;
- bucketed max KV shape plus graph cache gives about 5% improvement;
- graph build/alloc are no longer the main bottleneck;
- paged decode compute is still the dominant part of each decode step.

In the all-on run:

```text
paged_profile:
build=0.184ms
alloc=0.021ms
set=0.162ms
compute=15.708ms
get=0.102ms
total=16.177ms
```

Compute is about 97% of the profiled paged decode step.

## Profiling Workflow

### Nsight Systems

`nsys` was first used to inspect the full CUDA timeline. In this WSL setup,
`nsys stats` showed CUDA API activity but did not report CUDA kernel summaries:

```text
cudaLaunchKernel      ~= 355k calls
cudaStreamSynchronize ~= 4.5k calls
cudaMemcpyAsync       ~= 35k calls
```

This is useful for seeing that the graph contains many CUDA launches and
host/device interactions, but it was not enough to identify GPU kernel hotspots.

Important lesson: do not shorten profiling by changing `--chunk-sec` from 30 to
120. That changes the workload shape, inflates prefill tokens, and damages ASR
quality. Profiling should preserve the real 30-second chunk shape and only limit
the number of chunks.

### Test Harness Change

`test_offline_batching` now supports:

```text
--max-chunks <n>
```

This keeps the real long-video configuration while limiting the amount of work
for profilers:

```bash
FUNASR_PAGED_KV_WRITE_OP=1 FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
nsys profile --force-overwrite true --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none -o funasr_batch12_24chunks \
  ./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220 --max-chunks 24
```

### Nsight Compute

Because `nsys` did not expose GPU kernel summaries, `ncu` was used for targeted
kernel profiling.

Useful pattern:

```bash
FUNASR_PAGED_KV_WRITE_OP=1 FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
ncu --target-processes all --replay-mode application \
  --metrics gpu__time_duration.sum \
  --kernel-name 'regex:.*(paged_attn|paged_kv_write|mul_mat_q).*' \
  --launch-count 80 \
  -o funasr_ncu_decode_kernels_time \
  ./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220 --max-chunks 24
```

When using shell regex alternatives, quote the argument. Otherwise `|` is
treated as a shell pipe.

## Kernel Findings

### Paged KV Write

Kernel:

```text
paged_kv_write_f32_to_f16_kernel
```

Representative profile:

```text
duration ~= 3.74us
memory throughput ~= 27.52 GB/s
SM throughput ~= 6%
achieved occupancy ~= 21.5%
```

There are some non-ideal store/coalescing hints, but the kernel is tiny. It is
not the primary bottleneck.

### Paged Attention

Kernel:

```text
paged_attn_decode_f32_f16_warp_qk_cached_rows
```

Representative light profile:

```text
duration ~= 176.03us
SM throughput ~= 50.82%
DRAM throughput ~= 57.38%
L2 sector throughput ~= 13.81%
active warps ~= 47.92%
```

This is not a pure memory-bound or pure compute-bound kernel. It is a mixed
compute and KV-memory-access bottleneck.

### Q8 Matmul

Kernels:

```text
mul_mat_q
mul_mat_q_stream_k_fixup
```

Representative durations:

```text
mul_mat_q ~= 40us / 66us / 88us / 96us
mul_mat_q_stream_k_fixup ~= 17-21us
```

The multiple duration bands likely correspond to different projection sizes in
the transformer decoder. The fixup kernel follows many Q8 matmul launches, so
the total cost is not just the main matmul kernel.

## Interpretation

A decode step contains many kernels per transformer layer:

```text
RMSNorm
Q/K/V matmul
RoPE
paged KV write
paged attention
O projection matmul
MLP gate/up/down matmul
copy/cast/fixup kernels
```

With 28 decoder layers, rough kernel-level cost estimates are:

```text
paged attention: 28 * 176us ~= 4.9ms
paged KV write: 28 * 3.7us ~= 0.1ms
Q8 matmul + fixup: likely 10ms-scale total
```

This matches the observed paged decode compute time of about 15-16ms per step.

The important conclusion is:

```text
The next bottleneck is not only paged attention.
Q8 matmul, its fixup kernel, and the large number of small decode kernels are
probably at least as important as paged attention.
```

## MMQ Stream-K Experiment

After seeing `mul_mat_q_stream_k_fixup` in NCU, we tested whether disabling MMQ
stream-k could reduce decode overhead. The experiment added two runtime knobs in
`third_party/ggml/src/ggml-cuda/mmq.cu`:

```bash
FUNASR_DISABLE_MMQ_STREAM_K=1
FUNASR_MMQ_STREAM_K_MIN_NCOLS=<n>
```

The first knob disables MMQ stream-k globally. The second was intended as a
shape-aware policy: keep stream-k for larger matmuls and disable it only for
small `ncols` decode shapes.

To avoid copying results manually, the benchmark harness was added:

```bash
tools/bench_mmq_stream_k.sh
```

It runs:

```text
baseline
disable_all
min_ncols_8
min_ncols_16
min_ncols_32
```

and writes raw logs plus `summary.tsv` under:

```text
outputs/bench_mmq_streamk/<timestamp>/
```

### Global Disable Result

Global disable did reduce the decode-dispatch time, but it badly hurt prefill:

```text
baseline:
wall=56.45s
prefill=29.38s
decode_dispatch=27.03s
paged_total=16.15ms
tokens/s=339.8

disable_all:
wall=70.25s
prefill=45.57s
decode_dispatch=24.63s
paged_total=14.67ms
tokens/s=273.3
```

Interpretation:

```text
Disabling MMQ stream-k globally can make decode about 9% faster, but prefill
becomes about 55% slower. This is a net loss for the full long-video pipeline.
```

This means stream-k is useful for prefill/large-matrix work even if it may have
overhead in some decode shapes.

### Threshold Result

The initial `MIN_NCOLS` implementation affected too much of the matmul path and
made prefill slower. It was then narrowed so the threshold only affects
`ggml_cuda_op_mul_mat_q`, while the regular `ggml_cuda_mul_mat_q` path only
honors the global disable switch.

After that correction, the threshold variants were still worse than baseline:

```text
variant        wall_ms  prefill_wall_ms  decode_dispatch_ms  paged_total_ms  tokens_s
baseline       56449    29376            27026               16.152          339.8
min_ncols_8    60388    31878            28458               17.009          317.7
min_ncols_16   60779    32139            28586               17.085          315.6
min_ncols_32   62952    33501            29396               17.568          304.7
```

Conclusion:

```text
Using src1_ncols as the threshold signal does not cleanly separate prefill and
decode work. The shape-aware MMQ stream-k policy did not improve this workload.
Keep baseline stream-k behavior as the default.
```

This negative result is still useful: it prevents a misleading optimization
from being merged and shows that stream-k behavior is workload-shape dependent,
not universally good or bad.

## Optimization Direction

Recommended priority:

1. Keep default MMQ stream-k behavior for now.
2. Build a decode-step kernel category summary for paged attention, MMQ matmul,
   KV write, flash-attention fixup, quantize/copy, RMSNorm, and RoPE.
3. Continue improving paged attention KV access, QK, softmax, and V aggregation.
4. Inspect whether decode-batch Q8 matmul can use a better small-GEMM/GEMV path
   without harming prefill.
5. Reduce the number of small kernels in the decode graph.
6. Keep bucketed max KV and graph cache enabled because their benefit is proven.

## Interview-Ready Explanation

The profiling story can be summarized as:

```text
I first confirmed that the scheduler was healthy: batch=12 keeps avg_active
around 11.5, fallback is only 2 calls, and graph cache hit rate is about 80%.

Then I ran A/B tests and found graph cache gives about 2% and bucketed shapes
about 5%, but paged decode compute still accounts for about 97% of the decode
step profile.

Nsight Systems showed many CUDA launches and host/device interactions, but did
not expose GPU kernel summaries in my WSL setup. I added --max-chunks to keep
the real 30-second chunk workload while shortening profiler runs, then used
Nsight Compute for targeted kernel analysis.

NCU showed paged KV write is tiny at about 3.7us. Paged attention is important,
around 176us per launch with roughly 51% SM and 57% DRAM throughput, so it is a
mixed compute/memory bottleneck. Q8 matmul appears in many launches with
40-96us duration plus a 17-21us fixup kernel. Given the number of projection and
MLP matmuls per layer, Q8 matmul/fixup plus paged attention together explain
the 15-16ms decode compute time.

Therefore the next optimization target moves from scheduler and graph cache to
decoder kernel work. A follow-up experiment tried disabling MMQ stream-k:
decode alone became faster, but prefill became much slower, so the full
pipeline regressed. A simple ncols threshold did not fix this. The right next
step is to profile decode kernels by category instead of guessing from one
fixup kernel name.
```
