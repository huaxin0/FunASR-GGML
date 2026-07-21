# Unified Chunked-Prefill Scheduler Design

**Date:** 2026-07-21

**Status:** Approved

## 1. Purpose

The offline long-audio runtime currently has two different scheduling models:

```text
Encoder / Adaptor / Prefill: one whole request at a time
Decode:                      continuous batched execution
```

After decoupling scheduler concurrency from the physical paged-KV pool, the
best `batch=40, blocks=192` result for 6119.29 seconds of audio is:

```text
median wall:          40.812 s
prefill wall:         27.910 s
decode dispatch wall: 12.859 s
average active batch: 32.95
```

Prefill now accounts for about 68% of end-to-end wall time. The next runtime
revision therefore replaces whole-request admission-time prefill with a
vLLM-style token-budget scheduler. Prompt chunks and decode tokens share an
iteration budget and are represented by one mixed execution plan.

The design deliberately separates two ideas:

1. Scheduling and input packing are unified.
2. Attention kernels remain specialized for multi-query prefill and
   single-query decode.

A single universal attention kernel is not a goal. The runtime should keep the
existing optimized decode kernel while adding a chunked-prefill path behind a
common executor interface.

## 2. Goals

- Replace whole-request prefill admission with token-budget scheduling.
- Track progress with `num_computed_tokens`, without a hard scheduler boundary
  between prefill and decode.
- Split request-specific prompt embeddings into bounded chunks.
- Pack scheduled prompt and decode tokens without padding.
- Preserve request isolation through per-request block tables and positions.
- Allocate paged-KV blocks only for positions scheduled in the current step.
- Preserve task-scoped prefix KV reuse and partial-block copy-on-write.
- Keep the existing paged decode fast path for one-query requests.
- Add a multi-query paged attention path for prompt chunks.
- Run LM Head only for positions that must produce a sampled token.
- Retain the current scheduler as a reference path for correctness and A/B
  performance comparison.
- Leave explicit interfaces for a future multi-GPU PD connector.

## 3. Non-Goals

This implementation does not include:

- multi-node request routing;
- prefill/decode disaggregation or remote KV transport;
- tensor parallelism;
- speculative decoding;
- preemption and host KV swap;
- automatic batching of the acoustic Encoder and Adaptor;
- replacing GGML with TensorRT, Triton, PyTorch, or another runtime;
- claiming a performance improvement before three-run benchmark validation.

Encoder/Adaptor batching and multi-GPU execution remain follow-up projects.
The interfaces introduced here must not prevent either.

## 4. Reference Behavior

The target follows current serving-runtime concepts:

- vLLM V1 uses a maximum scheduled-token budget, prioritizes decode requests,
  and fills remaining budget with chunked prefills.
- TensorRT-LLM in-flight batching packs context and generation sequences
  without padding and bounds each iteration with `max_num_tokens`.
- SGLang exposes chunked prefill and an optional mixed-chunk mode, showing that
  unified scheduling does not require every backend to use one attention
  kernel.

References:

- https://docs.vllm.ai/en/stable/configuration/optimization/
- https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md

## 5. Request State Model

### 5.1 Lifecycle state

The scheduler-visible lifecycle becomes:

```cpp
enum class OfflineRequestState {
    WaitingFrontend,
    Ready,
    Running,
    Finished,
    Failed,
};
```

`Prefilling` and `Decoding` are not lifecycle states. They are derived from
token progress.

### 5.2 Token progress

Each request stores:

```cpp
struct UnifiedOfflineRequest {
    int request_id = -1;
    OfflineRequestState state = OfflineRequestState::WaitingFrontend;

    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int num_computed_tokens = 0;
    int max_output_tokens = 0;

    GPUEmbeddingHandle prompt_tail;
    KVHandle kv;
    std::vector<int> output_tokens;

    bool waiting_for_kv = false;
};
```

The number of tokens currently available to compute is:

```text
num_tokens_with_output = prompt_tokens + output_tokens.size()
num_new_tokens = num_tokens_with_output - num_computed_tokens
```

Examples:

```text
Prompt not complete:
    prompt_tokens=522, output_tokens=0, num_computed=256
    num_new_tokens=266 -> schedule a prompt chunk

Prompt just completed:
    prompt_tokens=522, output_tokens=1, num_computed=522
    the newest sampled token is not in KV
    num_new_tokens=1 -> schedule one decode token
```

When the final prompt position is computed, its LM Head output produces the
first output token. That token is appended to `output_tokens`, but is not added
to KV until the following scheduled iteration.

EOS finishes a request without scheduling EOS as a new decoder input. The
generation limit counts sampled output tokens and remains independent of KV
capacity.

### 5.3 Prefix cache initialization

On a task-prefix cache hit:

```text
num_computed_tokens = cached_prefix_tokens
prompt tail starts at position cached_prefix_tokens
```

The request retains the shared prefix blocks. If the cached prefix ends inside
a block, the existing COW operation creates a private final block before any
request-specific prompt token is written.

## 6. Owning Prompt Embedding Pool

`PreparedGPUAudio` is currently a non-owning view into a reusable Encoder /
Adaptor staging tensor. A second frontend invocation overwrites the first
result, so it cannot support partial prefills that survive across scheduler
iterations.

Introduce a bounded `GPUEmbeddingPool`:

```cpp
struct GPUEmbeddingHandle {
    int slot = -1;
    int token_count = 0;
    int embedding_dim = 0;
};
```

The pool owns stable GPU storage for request-specific:

```text
[audio embeddings][ChatML suffix embeddings]
```

The fixed ChatML prefix is represented by the existing shared prefix KV and is
not copied into every request buffer.

Rules:

- A request acquires one embedding slot after frontend completion.
- The adaptor output and suffix embeddings are copied into that slot.
- The slot remains valid across all prompt-chunk iterations.
- It is released immediately after the full prompt has been computed.
- Pool exhaustion delays new frontend work instead of overwriting live data.
- Failure and cancellation release both embedding and KV ownership.

At the current shape, 40 request tails require roughly 80-90 MiB, which is
small relative to the reduced KV allocation but must still be measured.

## 7. Scheduler Policy

### 7.1 Configuration

Add explicit scheduler controls:

```cpp
int max_num_seqs;
int max_num_scheduled_tokens;
int max_prefill_chunk_tokens;
int max_frontend_requests_per_step;
```

Suggested initial sweep values for the RTX 4070 Laptop GPU:

```text
max_num_seqs:              40
max_num_scheduled_tokens:  256 / 512 / 1024 / 2048
max_prefill_chunk_tokens:  128 / 256 / 512
frontend requests/step:    1
```

These are benchmark parameters, not compile-time assumptions.

### 7.2 Per-iteration scheduling

Each iteration performs:

1. Reclaim finished and failed requests.
2. Prepare bounded frontend work when embedding slots and request capacity are
   available.
3. Start with `token_budget = max_num_scheduled_tokens`.
4. Schedule one token for each runnable decode request, preserving decode ITL.
5. Use the remaining budget for partial prefills and newly ready requests in
   FCFS order.
6. Cap each prompt allocation by `max_prefill_chunk_tokens`.
7. Allocate only the KV blocks needed by the scheduled token range.
8. Build and execute one `MixedBatchPlan`.
9. Commit token progress and sampled outputs only after successful execution.

Intermediate prompt chunks should normally end on a KV block boundary when
that does not waste the remaining token budget. The final prompt chunk may be
shorter.

### 7.3 KV backpressure

Before a request is included in a plan:

```text
required_positions = num_computed_tokens + num_scheduled_tokens
required_blocks = ceil(required_positions / block_size)
```

Missing blocks are appended transactionally. If the pool cannot satisfy the
full chunk, the scheduler first shrinks the prompt chunk to the available
capacity. A decode token is either fully admitted or deferred.

The scheduler never executes a token whose physical KV row has not already
been reserved. Allocation failure cannot fall back to another request's block.

## 8. Mixed Batch Plan

The scheduler produces a backend-neutral plan:

```cpp
struct ScheduledSequence {
    int request_id = -1;
    int token_offset = 0;
    int num_tokens = 0;
    int num_computed_tokens = 0;
    bool produces_logits = false;
    const KVHandle* kv = nullptr;
};

struct MixedBatchPlan {
    std::vector<ScheduledSequence> sequences;
    int total_tokens = 0;
    int logits_rows = 0;
};
```

The model runner materializes packed tensors:

```text
input_embeds       [embedding_dim, total_tokens]
positions          [total_tokens]
token_request_ids  [total_tokens]
query_start        [num_sequences + 1]
context_lens       [num_sequences]
block_table        [num_sequences, max_blocks]
selected_rows      [logits_rows]
```

Tokens belonging to one request are contiguous within the packed input.
Prompt chunks are placed before decode-only sequences to simplify attention
dispatch, without changing decode-first scheduling priority.

## 9. Packed Model Runner

### 9.1 Input assembly

For each scheduled absolute position:

```text
position < prompt_tokens:
    copy embedding from prompt_tail[position - cached_prefix_tokens]

position >= prompt_tokens:
    gather the corresponding generated-token embedding
```

Positions below `cached_prefix_tokens` are already represented by retained
prefix KV blocks and are never scheduled from the request's prompt-tail slot.

Input assembly remains on GPU. Generated token IDs must not force a host
embedding round trip.

### 9.2 Shared transformer work

Projection and MLP operations consume the packed token dimension:

```text
[hidden, total_tokens]
```

This lets Q8 matrix multiplication see a larger effective `ncols` than the
current one-request prefill or single-token decode path.

### 9.3 Specialized attention dispatch

The mixed attention backend receives common paged metadata but keeps two
optimized subpaths:

```text
query length == 1:
    existing paged decode attention fast path

query length > 1:
    new causal varlen paged prefill attention path
```

For every query token, the backend derives its request row and reads only that
request's block table. Its visible KV range is:

```text
[0, absolute_query_position]
```

Current-chunk K/V rows are written before attention reads them. The GGML graph
must express that dependency explicitly; ordering two unrelated output nodes
is not accepted as a correctness guarantee.

The first implementation may dispatch the two attention subpaths internally.
The public runner still executes one mixed plan. A universal attention CUDA
kernel is neither required nor preferred.

### 9.4 Selective LM Head

Most intermediate prompt tokens do not need vocabulary logits. After the final
Transformer layer, gather only rows that:

- represent a decode input token; or
- complete a request's full prompt.

Run final RMSNorm and LM Head on the gathered matrix:

```text
[hidden, logits_rows]
```

This avoids producing a `[vocab_size, prompt_chunk_tokens]` tensor for prompt
positions whose logits will never be sampled.

## 10. CUDA Graph Strategy

Correctness is established with CUDA Graph reuse disabled for the new runner.
Graph caching is reintroduced after packed execution is stable.

Candidate graph signature:

```text
total_token_bucket
sequence_count_bucket
logits_row_bucket
max_kv_bucket
```

Positions, request IDs, block tables, context lengths, and selected rows are
runtime tensor inputs and must not be captured as immutable host parameters.

The existing decode graph cache remains available to the reference scheduler
and to decode-only mixed iterations until the new cache proves faster.

## 11. Error Handling and Commit Semantics

Scheduler state changes follow prepare/execute/commit semantics:

1. Build a tentative plan and reserve required blocks.
2. Execute the GPU plan.
3. On success, increment `num_computed_tokens` and append sampled tokens.
4. On failure, roll back newly reserved blocks and leave token progress
   unchanged.

Required invariants:

```text
0 <= num_computed_tokens <= prompt_tokens + output_tokens.size()
every scheduled position has one reserved physical KV row
exclusive blocks have ref_count == 1
shared prefix blocks are never modified without COW
finished task: final_free == capacity
finished task: ownership_errors == 0
```

If every runnable request is waiting for KV and no request can finish or
release memory, emit a deterministic pool-exhaustion error instead of spinning.

## 12. Observability

Add scheduler and executor metrics:

```text
unified_steps
mixed_steps
decode_only_steps
prefill_only_steps
scheduled_prefill_tokens
scheduled_decode_tokens
average_tokens_per_step
average_prefill_chunk_tokens
frontend_wait_steps
embedding_pool_peak
kv_backpressure_steps
packed_build_ms
packed_compute_ms
attention_prefill_ms
attention_decode_ms
selected_lm_head_rows
```

Existing wall, RTF, tokens/s, active batch, block ownership, graph hit, and
per-request output metrics remain available.

## 13. Compatibility and Rollout

The existing scheduler remains selectable:

```text
--scheduler-mode decode-only
--scheduler-mode unified
```

Implementation proceeds in reviewable milestones while targeting the complete
architecture:

1. Introduce request progress state, pure token-budget scheduling, and unit
   tests.
2. Add the owning GPU embedding pool and chunk-lifetime tests.
3. Add chunked paged prefill correctness using the new plan interface.
4. Add packed mixed execution and specialized attention dispatch.
5. Add selective LM Head.
6. Restore graph bucketing and run the tuning matrix.
7. Make unified mode the default only after all correctness and performance
   gates pass.

Intermediate milestones are not presented as the final unified runtime.

## 14. Test Strategy

### 14.1 Scheduler unit tests

- Decode requests consume budget before prefills.
- Remaining budget is filled by one or more prompt chunks.
- A prompt larger than the budget is split over multiple iterations.
- Prompt completion produces exactly one first sampled token.
- The sampled token is written to KV only in the following iteration.
- Block allocation occurs exactly at logical block boundaries.
- KV shortage shrinks or defers work without corrupting request progress.
- Prefix-cache progress starts at `cached_prefix_tokens`.
- Partial shared prefix blocks perform COW before request-specific writes.
- Completion and failure release embedding and KV ownership exactly once.

### 14.2 Kernel and runner tests

- Single-request chunked prefill matches whole prefill logits.
- Chunk boundaries `1`, `127`, `128`, `129`, and final-short-chunk are covered.
- Mixed batches cannot read or overwrite another request's KV.
- Packed prefill plus decode matches independently executed references.
- Decode-only plans preserve current output tokens.
- Selective LM Head matches the corresponding rows of the full LM Head.

### 14.3 End-to-end validation

- Run the 24-chunk smoke workload first.
- Run all 204 chunks with deterministic greedy decoding.
- Compare every chunk's token IDs and text against the current C++ reference.
- Run three performance repetitions after one warmup.
- Record peak VRAM, final block count, and ownership errors.

## 15. Acceptance Criteria

Correctness gates:

- `204/204` requests complete.
- Per-chunk output token IDs match the current C++ reference.
- Three unified runs produce identical output hashes.
- `final_free == capacity` and `ownership_errors == 0`.
- No request performs frontend work more than once.
- No intermediate prompt chunk samples a token.

Performance gates before unified mode becomes default:

- Three-run median wall is at least 5% below the current `40.812 s` baseline,
  i.e. no more than `38.771 s` on the same machine and workload.
- Prefill-related wall time decreases rather than being hidden in another
  metric.
- Peak VRAM remains within the 8 GiB device limit and is reported.
- Decode output quality is unchanged.

If correctness passes but performance does not, unified mode remains
experimental and profiling determines whether the bottleneck is packed Q8
MatMul, prefill attention, input assembly, LM Head, or graph replay.

## 16. Two-GPU and PD Extension

Two GPUs do not automatically imply prefill/decode disaggregation.

### 16.1 Offline data parallelism

For this workload, 204 chunks are independent and the model fits on one GPU.
The first two-GPU strategy should therefore be two complete runtime replicas,
with chunks divided between them.

Benefits:

- no cross-GPU KV transfer;
- no P/D pipeline imbalance;
- simple failure isolation;
- close to linear throughput scaling when CPU frontend and I/O do not become
  bottlenecks.

### 16.2 Prefill/decode disaggregation

PD becomes attractive for online serving when long prompts and decode traffic
have different latency objectives, or when P and D workers can be scaled
independently.

For this model, one P GPU would run Fbank/Encoder/Adaptor/Prompt Prefill and one
D GPU would run autoregressive generation. The P worker must transfer all
decoder-layer prompt K/V to the D worker before generation can start.

Physical block IDs are local to a GPU. A PD implementation cannot transfer the
P worker's block table directly. The D worker must:

1. allocate local destination blocks;
2. create its own block table;
3. receive K/V rows into those blocks;
4. acknowledge completion before decode admission.

Reserve the following future boundary:

```cpp
struct KVTransferDescriptor {
    int request_id;
    int token_count;
    int block_size;
};

class KVTransferBackend {
public:
    virtual bool send(const KVTransferDescriptor&, const KVHandle&) = 0;
    virtual bool receive(const KVTransferDescriptor&, KVHandle&) = 0;
};
```

The descriptor carries logical metadata, not source-device physical pointers.

With the current approximately `28 s` prefill and `13 s` decode totals, a
fixed one-P/one-D split would be prefill-heavy and leave the D GPU underused.
That ratio must be measured after unified scheduling before a PD topology is
selected. On exactly two GPUs, data parallelism is expected to be the stronger
offline baseline; PD remains a serving-oriented comparison rather than part of
this implementation.

## 17. Interview Claim Boundary

After the acceptance gates pass, the accurate project statement is:

> Implemented a token-budget unified scheduler for offline ASR that performs
> chunked prefill and continuous decode in mixed packed batches, dynamically
> allocates paged KV capacity, preserves a specialized decode attention fast
> path, and avoids LM Head work for intermediate prompt tokens.

Before those gates pass, the current accurate statement remains:

> Implemented decode-centric continuous batching with sequential request
> prefill, then identified prefill as the dominant remaining stage and designed
> a unified chunked-prefill scheduler from measured profiles.
