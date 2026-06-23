# Paged Offline Stabilization Design

## Goal

Turn the paged continuous offline decode MVP into an observable and
operator-friendly path for long-video transcription. This stabilization phase
does not add new decode algorithms. It makes the existing paged scheduler easier
to diagnose, tune, and recommend.

The immediate target is Milestone A:

```text
offline paged run
  -> fallback reason counters
  -> throughput summary
  -> KV/GPU memory budget hints
  -> concise benchmark output
```

## Non-Goals

- Do not implement decode graph caching in this milestone.
- Do not implement batched prefill in this milestone.
- Do not replace the current GPU KV cache allocation model.
- Do not change realtime recognition.
- Do not change the normal single-request pipeline.
- Do not make paged mode implicit for all GPU runs yet.

## Current Baseline

The paged continuous decode path is now validated on a long-video workload:

- `batch=4`: `wall=105.8s`, `rtf=0.0173`, `avg_active=3.91`
- `batch=8`: `wall=80.0s`, `rtf=0.0131`, `avg_active=7.75`
- `batch=12`: `wall=73.6s`, `rtf=0.0120`, `avg_active=11.47`
- `batch=16`: `wall=316.9s`, `rtf=0.0518`, `avg_active=15.19`

On the tested 8 GB RTX 4070 Laptop GPU, `batch=12` is the best observed
throughput point and `batch=16` crosses a memory pressure boundary. The code
already exposes scheduler counters, but it does not explain fallback causes or
memory pressure clearly enough.

## Observability Additions

### Fallback Reasons

`OfflineBatchStats` should split `decode_fallback_calls` into reason counters.
The first version only needs reasons that can be detected at the scheduler and
pipeline boundaries:

- `single_request`: only one active request was decoded in a step.
- `token_id_fast_path_unavailable`: paged token-id batch decode returned false.
- `host_embedding_batch_unavailable`: host-embedding paged batch decode returned
  false after token-id fallback.
- `serial_env_forced`: `FUNASR_PAGED_DECODE_SERIAL` forced serial decode.
- `invalid_paged_input`: a request had an empty block table, invalid block size,
  or exceeded context before dispatch.

The scheduler-level counter should remain simple:

```text
fallback_calls = sum(reason counters)
```

### Throughput Summary

`test_offline_batching` should print a compact benchmark line in addition to the
existing totals:

```text
[OfflineTest] throughput: audio_sec=6119.29 wall_sec=73.65 audio_sec/s=83.08 rtf=0.0120 tokens/s=275.1
```

`tokens/s` should use total decoded tokens divided by wall seconds. This is not
a model-quality metric. It is a decode-throughput signal for comparing batch
sizes and future kernel changes.

### Memory Budget Hints

The offline test should print an estimated GPU memory summary before the long
run when GPU mode is enabled:

```text
[OfflineTest] gpu budget: ctx=4096 slots=12 kv_est=5637MB weights_est=945MB mode=paged
```

If an estimate is not available from public APIs, compute the KV estimate from
the known model config and scheduler settings. The warning threshold can be
conservative:

- warn when estimated KV cache alone is larger than 60% of detected free GPU
  memory;
- warn when estimated KV cache plus known weights is larger than 85% of detected
  free GPU memory;
- if free memory is not available, still print the estimate without a warning.

The warning should be advisory, not fatal:

```text
[OfflineTest] WARNING: requested ctx=4096 slots=16 may exceed the comfortable memory budget; try batch=8/12 or ctx=2048.
```

### Decode Profile Summary

`GPURunner` already accumulates paged decode profile pieces internally. Expose a
read-only summary so the offline test can print one final line:

```text
[OfflineTest] paged_profile: calls=1669 build=0.42ms alloc=0.08ms set=0.03ms compute=39.2ms get=0.02ms total=39.75ms
```

This should be a summary line, not noisy per-step logging. It will decide
whether graph caching is worth the next milestone.

## Data Model

Add focused structs instead of spreading raw counters through unrelated classes:

```cpp
struct OfflineDecodeFallbackStats {
    int single_request = 0;
    int token_id_fast_path_unavailable = 0;
    int host_embedding_batch_unavailable = 0;
    int serial_env_forced = 0;
    int invalid_paged_input = 0;

    int total() const;
};
```

`OfflineBatchStats` should own this struct and keep the existing
`decode_fallback_calls` field for compatibility with current logs. At the end
of transcription, `decode_fallback_calls` should equal
`fallback_reasons.total()`.

For GPU profiling, expose a lightweight immutable copy:

```cpp
struct PagedDecodeProfile {
    long calls = 0;
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double set_input_ms = 0.0;
    double compute_ms = 0.0;
    double get_ms = 0.0;
};
```

The profile already exists inside `GPURunner`; the stabilization work should add
an accessor rather than changing how timing is collected.

## Output Format

Keep the existing scheduler line so old benchmark notes remain comparable:

```text
[OfflineTest] scheduler: gpu=1 kv=paged batch=12 chunks=204 decode_steps=1675 grouped_calls=1669 fallback_calls=6 avg_active=11.47 blocks_peak=828/3072 block_size=16
```

Add new lines after it:

```text
[OfflineTest] fallback_reasons: single=6 token_id=0 host_embed=0 serial_env=0 invalid=0
[OfflineTest] throughput: audio_sec=6119.29 wall_sec=73.65 audio_sec/s=83.08 rtf=0.0120 tokens/s=275.1
[OfflineTest] paged_profile: calls=1669 build=... alloc=... set=... compute=... get=... total=...
```

Only print `paged_profile` when paged GPU decode was used.

## Validation

Unit tests should cover:

- `OfflineDecodeFallbackStats::total()`.
- `OfflineBatchStats::average_active_batch()` remains unchanged.
- `decode_fallback_calls` can be synchronized from reason counters.

Smoke tests should cover:

```bash
cmake --build build-cuda --target test_offline_scheduler test_offline_batching -j$(nproc)
./build-cuda/test_offline_scheduler
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav --kv-mode paged --batch-size 2 --ctx-size 256 --chunk-mode window --chunk-sec 3 --max-tokens 3
```

GPU validation is user-run because the sandbox cannot access the GPU:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --chunk-mode window --chunk-sec 30 --max-tokens 220
```

Expected signal:

- scheduler line remains comparable to previous runs;
- fallback reason totals match `fallback_calls`;
- throughput line prints wall, RTF, audio throughput, and tokens/s;
- memory warning appears for configurations similar to `batch=16`, `ctx=4096`
  on 8 GB GPUs.

## Completion Criteria

- Fallbacks are explained by reason counters.
- Benchmark output can compare batch sizes without manual arithmetic.
- Memory-pressure configurations produce a clear warning before the long run.
- Existing paged and continuous scheduler paths still build and pass the current
  scheduler tests.
