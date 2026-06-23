# Offline Video Entry Productization Design

Date: 2026-06-03

## Goal

Make the existing vLLM-style offline ASR scheduler usable from the normal long
video entry points. The first productized path targets fixed-window long video
transcription on GPU, using the measured high-throughput configuration:

```text
--gpu
--chunk-mode window
--chunk-sec 30
--ctx-size 4096
--max-tokens 220
--offline-scheduler
--batch-size 12
--kv-mode paged
--kv-block-size 128
--offline-profile
```

The user-facing result should be the same kind of transcript files as today:
plain text, SRT, TSV, and the video helper's sidecar JSON/stats files. The
performance-critical difference is that chunked long audio is sent through
`OfflineBatchTranscriber` instead of being recognized segment by segment.

## Non-Goals

- Do not change the realtime microphone path.
- Do not add token-level partial streaming.
- Do not force every `funasr-cli` long-audio run through the offline scheduler.
- Do not remove the existing serial per-segment chunk transcription path.
- Do not make VAD the default high-throughput path in the first version.
- Do not change CUDA kernels or scheduler internals as part of this entry work.

## Current State

`test_offline_batching` already exercises the high-throughput path:

```text
audio
  -> AudioChunk list
  -> OfflineBatchTranscriber
  -> GPU paged KV scheduler
  -> paged token-id batch decode
  -> ordered OfflineChunkResult list
```

`funasr-cli` currently handles long audio by splitting into chunks and then
calling `recognizer.transcribe_audio()` once per segment. That means the official
CLI entry does not use continuous batching, paged KV, dynamic paged KV write, or
the paged decode profile/stats path.

`tools/funasr_video_ui.py` currently prepares media and calls `funasr-cli`. Its
`long` preset already matches the benchmark workload shape in broad strokes
(`window`, `30s`, `ctx=4096`, `max_tokens=220`), but it cannot request the
offline scheduler yet because the CLI has no such option.

## Proposed CLI Interface

Add these options to `funasr-cli`:

```text
--offline-scheduler       Use OfflineBatchTranscriber for chunked audio.
--offline-profile         Print offline scheduler and paged decode profile lines.
--offline-preset <name>   Apply a named offline preset. First preset: long-video.
--batch-size <n>          Offline scheduler active request limit.
--kv-mode <mode>          continuous or paged.
--kv-block-size <n>       Paged KV block size.
--kv-num-blocks <n>       Optional explicit paged KV block pool size.
```

The `long-video` preset applies:

```text
offline_scheduler = true
offline_profile = true
batch_size = 12
kv_mode = paged
kv_block_size = 128
ctx_size = 4096
chunk_mode = window
chunk_sec = 30
max_tokens = 220
```

Explicit flags provided after the preset should override preset values. The
parser can keep the existing simple left-to-right behavior: the last assignment
wins.

## CLI Routing

Use the offline scheduler only when all of these are true:

```text
--offline-scheduler is set
chunk mode is window or vad
audio file was loaded successfully
there is at least one chunk
```

GPU is strongly recommended but not required by the type system. If the user
asks for `--kv-mode paged` without GPU, the scheduler can still run through its
existing compatibility behavior, but the CLI should print a warning that the
high-throughput paged decode path requires GPU.

For `--chunk-mode none`, keep the current single-call path:

```text
recognizer.transcribe_audio(samples.data(), samples.size(), config)
```

For chunked audio without `--offline-scheduler`, keep the existing serial
per-segment path.

For chunked audio with `--offline-scheduler`:

```text
samples + AudioChunk list
  -> OfflineBatchConfig
  -> OfflineBatchTranscriber::transcribe()
  -> OfflineChunkResult list
  -> SegmentResult list
  -> item.text via append_without_overlap()
```

The output writer should not need a new format. It should consume the same
`TranscriptionItem` structure it already uses.

## Stats And Profile Output

When `--offline-profile` is set, print compact profile lines to stderr after
each file finishes. At minimum include:

```text
[OfflineCLI] offline_stats:
  chunks, ok, batch, kv_mode, block_size, blocks_peak, decode_steps,
  grouped_calls, fallback_calls, avg_active

[OfflineCLI] fallback_reasons:
  single, token_id, host_embed, serial_env, invalid

[OfflineCLI] scheduler_profile:
  admit, no_kv, admit_rounds, avg_admit_round, max_admit_round,
  avg_prefill_wall_ms, avg_decode_dispatch_ms, idle_steps
```

When GPU paged KV is used, also print the existing paged decode profile:

```text
[OfflineCLI] paged_profile:
  calls, build, alloc, set, compute, get, total

[OfflineCLI] paged_attn_profile:
  calls, total, avg_kernel, graph_compute_share

[OfflineCLI] paged_graph_cache_probe:
  calls, shape_hits, shape_hit_rate, param_hits, param_hit_rate,
  full_hits, full_hit_rate, cache_hits, cache_misses, cache_hit_rate
```

These lines should mirror `test_offline_batching` so benchmark logs remain easy
to compare.

## Video UI Changes

Update `tools/funasr_video_ui.py` so the `long` preset defaults to the high
throughput offline path:

```text
--offline-preset long-video
```

or the equivalent explicit flags:

```text
--offline-scheduler
--batch-size 12
--kv-mode paged
--kv-block-size 128
--offline-profile
```

The first version should keep `balanced` and `vad` conservative unless the user
opts in. This keeps the default benchmark-backed path focused on fixed 30-second
windows, where the current best measurements were made.

The video helper should add the offline settings to `stats.json`:

```json
{
  "offline_scheduler": true,
  "offline_profile": true,
  "batch_size": 12,
  "kv_mode": "paged",
  "kv_block_size": 128
}
```

If CLI profile lines are available in the subprocess output later, they can be
parsed into structured stats in a follow-up. The first version can simply print
them and record the configured offline options.

## Error Handling

- If `--offline-scheduler` is requested with `--chunk-mode none`, print a warning
  and use the normal single-call path.
- If `--kv-mode` is not `continuous` or `paged`, fail argument parsing.
- If `--batch-size`, `--kv-block-size`, or `--kv-num-blocks` is invalid, fail
  argument parsing.
- If `OfflineBatchTranscriber` returns empty text for a chunk, skip that segment
  in the same spirit as the existing serial chunk path.
- Preserve the current behavior where a file is marked failed when no final text
  is produced.

## Realtime Boundary

The realtime path stays as it is:

```text
microphone callback
  -> feed_audio()
  -> 100 ms process loop
  -> energy VAD endpointing
  -> one completed utterance
  -> recognizer.transcribe_audio()
  -> callback emits final utterance text
```

This is utterance-level streaming, not token-level streaming. It is optimized
for low-latency local microphone interaction, while the offline scheduler is
optimized for throughput across many independent chunks. Mixing the two should
be a separate design.

## Testing

Build targets:

```bash
cmake --build build-cuda --target funasr-cli test_offline_scheduler test_offline_batching -j$(nproc)
```

Local non-GPU checks:

```bash
./build-cuda/test_offline_scheduler
./build-cuda/funasr-cli -m FunAsr_q8.bin -f zh.wav --chunk-mode window \
  --chunk-sec 3 --offline-scheduler --batch-size 2 --kv-mode paged \
  --ctx-size 256 --max-tokens 3 -osrt -o /tmp/funasr_offline_cli_smoke.srt
```

GPU benchmark-style check:

```bash
FUNASR_PAGED_KV_WRITE_OP=1 \
FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
./build-cuda/funasr-cli -m FunAsr_q8.bin \
  -f outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --offline-preset long-video -osrt -o /tmp/funasr_long_video.srt
```

Expected success criteria:

- Existing `funasr-cli` serial chunk behavior still works without
  `--offline-scheduler`.
- `--offline-scheduler` routes chunked audio through `OfflineBatchTranscriber`.
- SRT/TXT/TSV output remains ordered by chunk start time.
- `--offline-profile` emits comparable scheduler and paged profile lines.
- The video UI `long` preset invokes the offline long-video path by default.
