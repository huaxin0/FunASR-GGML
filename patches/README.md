# GGML Runtime Extensions

The repository pins upstream GGML as a submodule and carries the FunASR-specific
runtime changes as `ggml-funasr-runtime.patch`. This keeps a fresh clone
reproducible without depending on an unpublished submodule commit.

Apply the patch once after initializing submodules:

```bash
git submodule update --init --recursive
tools/apply_ggml_runtime_patch.sh
```

The script is idempotent and exits successfully when the patch is already
present. The patch currently adds:

- FSMN CPU/CUDA graph operations for the batched acoustic encoder
- Paged KV Write and Paged Attention CUDA operations
- GGML allocator generation tracking for cached graph invalidation
- per-graph CUDA cache reset support
- MMQ Stream-K profiling controls used by the Q8 benchmark matrix

The patch is based on the GGML commit pinned by `third_party/ggml`. When that
submodule revision changes, rebase and verify this patch before updating the
parent gitlink.
