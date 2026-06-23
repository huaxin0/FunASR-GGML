# Offline Continuous Batching Design

This design is for long-video / offline transcription. It must not replace the
existing `Pipeline::transcribe_audio()` path, which remains the simple realtime
and single-chunk path.

## Goal

Process one long video as many ASR chunk requests:

```text
audio/video
  -> chunk planner (window or VAD)
  -> request queue
  -> continuous batching scheduler
  -> GPU inference
  -> ordered transcript / SRT / JSON
```

The design borrows the vLLM shape:

```text
Request state + scheduler + KV memory manager
```

For this project, an ASR chunk is the request.

## Non-Goals

- Do not break realtime recognition.
- Do not require paged attention kernels in the first implementation.
- Do not force every CLI path through the offline scheduler.
- Do not remove the current continuous single-slot KV cache until the new path
  is verified.

## Request Model

```cpp
struct AsrChunk {
    int id;
    float start_sec;
    float end_sec;
    std::vector<float> samples;
};

enum class RequestPhase {
    WaitingPrefill,
    Decoding,
    Finished,
    Failed,
};

struct OfflineRequest {
    int id;
    RequestPhase phase;
    KVHandle kv;
    int n_past;
    int max_tokens;
    std::vector<int> tokens;
    std::string text;
    float start_sec;
    float end_sec;
};
```

## Scheduler

The scheduler owns three queues:

```text
waiting_prefill: chunks not started
active_decode:   chunks with KV allocated and logits ready
finished:        completed chunks
```

The loop is iteration based:

```text
while work remains:
  1. admit waiting requests while slots are available
  2. run prefill for newly admitted requests
  3. run one decode step for active requests
  4. finish requests that emitted EOS / hit repeat guard / hit max tokens
  5. release their KV slots
```

The first implementation can use static batches:

```text
admit B chunks -> prefill B -> decode B until all finish -> next B
```

The target implementation uses continuous batching:

```text
when one request finishes, immediately admit another request into the freed slot
```

## KV Memory Manager

The interface should be independent from the physical layout:

```cpp
struct KVHandle {
    int request_id;
    int slot_id;
    int ctx_size;
};

class KVPool {
public:
    virtual KVHandle acquire(int request_id, int ctx_size) = 0;
    virtual void release(KVHandle handle) = 0;
    virtual void clear(KVHandle handle) = 0;
};
```

Backends:

```text
1. ContinuousSlotPool
   - N fixed slots.
   - Each slot owns a full continuous KV cache.
   - Easiest path for batch=2/4.

2. ArenaKVPool
   - One large continuous arena.
   - Each request gets a contiguous token range.
   - Less waste than fixed full slots, but fragmented over time.

3. PagedKVPool
   - Fixed-size KV blocks, e.g. 16 or 32 tokens.
   - Each request has a block table.
   - Needs gather fallback or a paged attention kernel.
```

## Why Not Jump Directly To PagedAttention

Paged KV cache is not just an allocator change. Current attention reads K/V by
creating contiguous `ggml_view_2d()` tensors. Paged KV means historical K/V for a
request is not contiguous:

```text
logical tokens 0..15   -> physical block 7
logical tokens 16..31  -> physical block 2
logical tokens 32..47  -> physical block 9
```

That requires one of:

```text
A. gather paged blocks into a temporary contiguous K/V tensor, then use current attention
B. implement a block-table-aware attention kernel
```

A is a good stepping stone. B is the real vLLM-style endpoint.

## First Working Milestone

Keep current `Pipeline` untouched for realtime. Add an offline engine:

```cpp
class OfflineBatchTranscriber {
public:
    std::vector<ChunkResult> transcribe(
        const std::vector<AsrChunk>& chunks,
        const OfflineBatchConfig& cfg);
};
```

Initial behavior:

```text
- chunk planner creates chunks
- scheduler owns request states
- KV pool has one slot, so behavior equals current serial execution
- stats are emitted per chunk
```

This milestone proves the architecture without changing performance.

Current scaffold:

```text
pipeline/offline_batching.hpp
pipeline/offline_batching.cpp
```

It includes:

```text
- OfflineBatchConfig
- OfflineChunkResult
- OfflineRequestPhase / OfflineRequest
- ContinuousKVSlotPool
- OfflineBatchTranscriber
```

The first implementation is intentionally serial internally, but the call shape
is already request/scheduler/KV-slot based.

Milestone 2 adds a clean stage boundary:

```text
audio chunk
  -> prepare_llm_input()
       fbank + encoder + adaptor + prompt embedding build
  -> run_prepared()
       LLM prefill + autoregressive decode
```

This still computes each active request serially, but the offline scheduler can
now hold prepared requests as independent work items. The next step is to
replace the serial `run_prepared()` loop with batched prefill/decode for active
requests.

Milestone 3 adds the first real GPU batch path:

```text
active chunks
  -> prepare each chunk
  -> prefill each request into its own GPU KV slot
  -> run decode steps as one multi-request ggml graph
```

This requires the GPU KV cache to be slot-aware:

```text
[slot][layer][token][kv_dim]
```

It is not PagedAttention yet. It is a continuous-slot pool: each active ASR
chunk owns a full continuous KV slot. That is deliberately simpler than vLLM's
paged block table, but it creates the same request-state boundary and lets the
decode loop process multiple active requests together.

Current GPU batch behavior:

```text
- batch prefill: still one request at a time, but into separate KV slots
- batch decode: multiple request next-token steps in one ggml graph
- old single-request GPU path: unchanged, uses slot 0
```

Milestone 4 starts the Paged KV shape:

```text
- KVPool interface
- ContinuousKVSlotPool backend
- PagedKVBlockPool backend
- KVHandle.block_table
```

The paged allocator can already reserve/release fixed-size KV blocks for each
ASR request:

```text
request 17 logical tokens 0..511
  -> block_table = [42, 7, 91, ...]
```

Important distinction:

```text
Paged KV allocator is implemented.
Paged attention is not implemented yet.
```

Current GGML attention still expects contiguous K/V views. Real vLLM-like speed
requires a CUDA/GGML op that accepts `block_table` and reads K/V through block
indirection. Until that lands, `--kv-mode paged` is useful for scheduler and
memory-manager validation, not for GPU throughput.

The existing normal Pipeline::run() path keeps the GPU encoder/adaptor zero-copy
flow unchanged.

Milestone 5 splits the GPU path into scheduler-callable steps:

```cpp
GPUPrefillState Pipeline::gpu_prefill_audio_slot(
    AudioSpan audio, int request_id, int slot_id, InferenceConfig cfg);

std::vector<GPUDecodeStepOutput> Pipeline::gpu_decode_step_slots(
    std::vector<GPUDecodeStepInput> inputs, InferenceConfig cfg);
```

The offline scheduler can now run a true continuous-batching loop:

```text
admit request into free KV slot
prefill one request
while active:
  sample next token for every active request
  run one batched decode step
  finish/release completed requests
  immediately admit newly waiting requests
```

This is still continuous-slot attention. To turn the paged allocator into real
PagedAttention, the next low-level change is:

```text
gpu_gqa_forward_slot(...)
  current: slot_id -> contiguous [layer][token] view

gpu_gqa_forward_paged(...)
  target: block_table -> physical KV blocks
```

A safe intermediate implementation is gather fallback:

```text
for each layer/request:
  gather block_table blocks into contiguous K/V scratch
  run existing flash attention on scratch
```

That validates block-table correctness before replacing it with a fused CUDA
PagedAttention kernel.

Validation target:

```text
test_offline_batching
```

Example:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin zh.wav \
  --chunk-mode window --chunk-sec 30 \
  --batch-size 2 --ctx-size 2048 --max-tokens 40
```

GPU shape:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin source_16k.wav \
  --gpu --ctx-size 4096 \
  --chunk-mode window --chunk-sec 30 \
  --batch-size 4 --max-tokens 220
```

Expected shape:

```text
[OfflineTest] chunks=1 chunk_sec=30
[OfflineTest] 1/1 chunk=0 ok=1 ...
```

## Second Milestone

`ContinuousSlotPool` with multiple slots:

```text
slot 0 -> active request A
slot 1 -> active request B
slot 2 -> active request C
slot 3 -> active request D
```

Implementation can still execute per-slot graphs serially at first, but the
state model becomes batch-ready.

Current status:

```text
Done as scheduler shape.
```

`OfflineBatchTranscriber` now admits up to `batch_size` requests into an active
set and assigns distinct continuous KV slots:

```text
chunk 0 -> slot 0
chunk 1 -> slot 1
...
```

The active requests are still computed serially by calling the existing
`Recognizer::transcribe_audio()` path. This preserves correctness while making
the next milestone, static batched prefill/decode, a local replacement inside
the active request loop.

## Third Milestone

Static batch prefill/decode:

```text
- batch requests with similar prefill length
- pad inputs to a shared length
- maintain finished masks
- decode active requests step by step
```

This is the first milestone that should improve GPU utilization.

## Fourth Milestone

Continuous batching:

```text
- active batch has fixed capacity
- finished request releases slot
- scheduler admits a new waiting request immediately
- decode loop keeps GPU batch occupied
```

## Fifth Milestone

Paged KV:

```text
- block allocator
- request block table
- free list
- optional prefix cache
- gather fallback first
- custom paged attention later
```

## Fit For FunASR-GGML

Recommended split:

```text
Pipeline:
  realtime / single chunk / simple CLI

OfflineBatchTranscriber:
  long video / many chunks / continuous batching / dynamic KV pool
```

This keeps the existing product usable while creating a path toward
vLLM-style scheduling.
