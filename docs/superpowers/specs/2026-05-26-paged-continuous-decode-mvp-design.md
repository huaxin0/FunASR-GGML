# Paged Continuous Decode MVP Design

## Goal

Improve long-video offline transcription throughput by keeping multiple ASR
chunks active during autoregressive decode. The first version optimizes the
decode phase with paged KV continuous batching. Prefill may remain per-request.

The target path is explicit and offline-only:

```text
audio/video
  -> chunk planner
  -> paged KV scheduler
  -> per-request paged prefill
  -> batched paged decode loop
  -> ordered chunk results
```

## Non-Goals

- Do not change realtime recognition.
- Do not change the normal single-request `Pipeline::transcribe_audio()` path.
- Do not implement batch prefill in this milestone.
- Do not rework CPU graph allocation in this milestone.
- Do not make paged mode the default CLI path until benchmarked.

## Existing Pieces

The current code already has most of the needed shape:

- `OfflineBatchTranscriber` owns the offline scheduler loop.
- `PagedKVBlockPool` allocates and releases fixed-size KV blocks.
- `KVHandle.block_table` carries the logical-to-physical KV mapping.
- `Pipeline::gpu_prefill_audio_paged()` can prefill one request into paged KV.
- `Pipeline::gpu_decode_step_slots()` can dispatch paged decode inputs.
- `GPURunner::forward_batch_decode_paged_token_ids()` can run a batched decode
  graph when active requests share the same `n_past`.

## MVP Behavior

The scheduler should keep up to `batch_size` requests active.

Admission:

1. Select the next chunk.
2. Estimate the KV block budget for that chunk.
3. Acquire blocks from `PagedKVBlockPool`.
4. Run paged prefill for the request.
5. Store `n_past`, first `next_token`, timings, and block table.

Decode loop:

1. For every active request, consume `next_token`.
2. Finish requests on EOS, max tokens, context limit, or repetition loop.
3. Group remaining active requests by `n_past`.
4. Run one batched paged decode step per `n_past` group.
5. Update each request with returned `next_token` and incremented `n_past`.
6. Release finished requests and immediately admit new chunks.

Grouping by `n_past` is required because the current fused paged decode path
expects a shared position tensor. Requests with different `n_past` should still
benefit from batching inside each group instead of forcing the whole active set
through a serial fallback.

## Data Flow

```text
OfflineRequest
  id
  phase
  KVHandle { block_table, block_size }
  n_past
  next_token
  tokens
  timing counters
```

Paged decode input:

```text
request_id
n_past
token_id
block_table
block_size
```

Paged decode output:

```text
request_id
n_past + 1
next_token
ok
```

The fast path should prefer token IDs over host token embeddings so the decode
step can gather embeddings on GPU and return only next token IDs.

## Stats

The offline test path should print enough summary data to judge throughput:

- total wall time
- audio duration and RTF
- number of chunks
- batch size
- KV mode
- average active batch size during decode
- decode step count
- grouped decode call count
- fallback decode call count
- paged KV block capacity and peak blocks in use

Per-chunk progress should remain available, but the summary is the main signal
for long-video optimization.

## Validation

Use `test_offline_batching` first:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin long.wav \
  --gpu --kv-mode paged --batch-size 4 \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

Compare against:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin long.wav \
  --gpu --kv-mode continuous --batch-size 1 \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

Correctness target:

- Output chunks are ordered by id.
- No request reads or writes outside its allocated block table.
- Paged mode text is close to the existing single-request GPU path for the same
  chunks.

Performance target:

- Paged mode should reduce total wall time on multi-chunk long audio when
  `batch_size > 1`.
- Average active batch should be greater than 1 for long enough input.

## Risks

- Variable chunk lengths create different `n_past` values, so grouping is
  necessary for the current fused decode graph.
- Paged prefill is still per-request and may dominate short chunks.
- Rebuilding GGML graphs every decode step may limit speedup. This MVP measures
  that cost before adding graph caching.
- `ggml_paged_attn_ext` behavior must be validated carefully against contiguous
  slot attention.

## Completion Criteria

- Paged mode runs the continuous scheduler without crashing on multi-chunk audio.
- Active requests are decoded in `n_past` groups, not all forced through a serial
  fallback.
- Summary stats expose whether batching is actually happening.
- Existing continuous-slot and single-request paths still build.
