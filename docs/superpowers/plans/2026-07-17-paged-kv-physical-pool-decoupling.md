# Paged KV Physical Pool Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple offline paged-KV GPU allocation from `batch_size * ctx_size`, then benchmark higher scheduler concurrency against a fixed global block pool.

**Architecture:** Keep `ctx_size` as the per-request limit and `batch_size` as the scheduler active-request cap. Add an optional `physical_kv_rows` argument through `Recognizer -> Pipeline -> GPUContext`; paged callers resolve it from the same block-count helper used to construct `PagedKVBlockPool`, while all existing three-argument and continuous-KV calls retain the old allocation rule.

**Tech Stack:** C++17, ggml CUDA backend, CMake/CTest-style executables, Bash benchmark tooling.

---

### Task 1: Centralize paged block-capacity resolution

**Files:**
- Modify: `pipeline/offline_batching.hpp`
- Modify: `pipeline/offline_batching.cpp`
- Modify: `test/test_offline_scheduler.cpp`

- [ ] **Step 1: Write failing resolver tests**

Add tests that specify explicit and derived capacities:

```cpp
void test_paged_kv_capacity_resolution() {
    funasr::OfflineBatchConfig cfg;
    cfg.batch_size = 12;
    cfg.ctx_size = 4096;
    cfg.kv_block_size = 128;

    cfg.kv_num_blocks = 160;
    TEST_EQ(funasr::resolve_paged_kv_num_blocks(cfg), 160,
            "explicit paged block count");
    TEST_EQ(funasr::resolve_paged_kv_physical_rows(cfg), 20480,
            "explicit paged physical rows");

    cfg.kv_num_blocks = 0;
    cfg.batch_size = 3;
    cfg.ctx_size = 4100;
    TEST_EQ(funasr::resolve_paged_kv_num_blocks(cfg), 99,
            "derived blocks round each context upward");
    TEST_EQ(funasr::resolve_paged_kv_physical_rows(cfg), 12672,
            "derived rows match resolved blocks");
}
```

Register the test in `main()`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler
./build-cuda/test_offline_scheduler
```

Expected: compilation fails because the two resolver functions do not exist.

- [ ] **Step 3: Implement the minimal shared resolver**

Declare in `offline_batching.hpp`:

```cpp
int resolve_paged_kv_num_blocks(const OfflineBatchConfig& config);
int resolve_paged_kv_physical_rows(const OfflineBatchConfig& config);
```

Implement in `offline_batching.cpp` using positive clamping and ceil division. Replace both duplicated `num_blocks` formulas in `transcribe()` and `transcribe_continuous_gpu()` with `resolve_paged_kv_num_blocks(config)`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same two commands and require zero failures.

- [ ] **Step 5: Inspect the diff checkpoint**

Run:

```bash
git diff --check -- pipeline/offline_batching.hpp pipeline/offline_batching.cpp test/test_offline_scheduler.cpp
```

Do not create an intermediate commit because these files already contain required uncommitted dynamic-block work; preserve that baseline intact.

### Task 2: Thread explicit physical rows through GPU initialization

**Files:**
- Modify: `compute/gpu_context.hpp`
- Modify: `pipeline/pipeline.hpp`
- Modify: `pipeline/pipeline.cpp`
- Modify: `pipeline/recognizer.hpp`
- Modify: `test/test_offline_scheduler.cpp`

- [ ] **Step 1: Write a failing pure allocation-rule test**

Expose a side-effect-free helper in `GPUContext` so the compatibility rule can be tested without a GPU:

```cpp
TEST_EQ(funasr::GPUContext::resolve_physical_kv_rows(4096, 12, 0),
        49152, "default rows preserve slot allocation");
TEST_EQ(funasr::GPUContext::resolve_physical_kv_rows(4096, 32, 20480),
        20480, "explicit rows decouple scheduler concurrency");
```

- [ ] **Step 2: Build and verify RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler
```

Expected: compilation fails because `resolve_physical_kv_rows` does not exist.

- [ ] **Step 3: Add the compatibility helper and optional parameter**

Implement:

```cpp
static int resolve_physical_kv_rows(int n_ctx,
                                    int n_slots,
                                    int physical_kv_rows) {
    if (physical_kv_rows > 0) {
        return physical_kv_rows;
    }
    return std::max(1, n_ctx) * std::max(1, n_slots);
}
```

Extend the initialization signatures with a final defaulted argument:

```cpp
bool Recognizer::init_gpu(int n_ctx = 2048,
                          int gpu_id = 0,
                          int n_slots = 1,
                          int physical_kv_rows = 0);

bool Pipeline::init_gpu(int n_ctx = 2048,
                        int gpu_id = 0,
                        int n_slots = 1,
                        int physical_kv_rows = 0);

bool GPUContext::init(const LLMWeights&,
                      const LLMConfig&,
                      int n_ctx = 2048,
                      int gpu_id = 0,
                      int n_slots = 1,
                      int physical_kv_rows = 0);
```

Pass the value unchanged through the chain. `init_kv_cache` must retain logical `n_ctx` and `n_slots` metadata but allocate K/V using the resolved physical rows.

- [ ] **Step 4: Run focused and existing scheduler tests**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler
./build-cuda/test_offline_scheduler
```

Expected: all tests pass, including old pool ownership and dynamic append tests.

- [ ] **Step 5: Verify old call sites remain source compatible**

Run:

```bash
rg -n "init_gpu\(" pipeline cli examples test
cmake --build build-cuda -j2 --target funasr_pipeline
```

Expected: existing one-, two-, and three-argument calls compile without edits.

### Task 3: Make paged CLI and benchmark initialization use the global pool

**Files:**
- Modify: `cli/funasr_cli.cpp`
- Modify: `test/test_cli_options.cpp`
- Modify: `test/test_offline_batching.cpp`

- [ ] **Step 1: Add failing CLI physical-row routing tests**

Add a helper test around the options already used by `gpu_init_slots`:

```cpp
opt.use_gpu = true;
opt.offline_scheduler = true;
opt.chunk_mode = ChunkMode::Window;
opt.offline_kv_mode = OfflineKVMode::Paged;
opt.offline_batch_size = 32;
opt.ctx_size = 4096;
opt.offline_kv_block_size = 128;
opt.offline_kv_num_blocks = 160;
TEST_ASSERT(gpu_init_physical_kv_rows(opt) == 20480,
            "paged init uses explicit global pool");

opt.offline_kv_mode = OfflineKVMode::Continuous;
TEST_ASSERT(gpu_init_physical_kv_rows(opt) == 0,
            "continuous init preserves default allocation");
```

- [ ] **Step 2: Build `test_cli_options` and verify RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_cli_options
```

Expected: compilation fails because `gpu_init_physical_kv_rows` does not exist.

- [ ] **Step 3: Implement CLI routing**

Add `gpu_init_physical_kv_rows(const Options&)`. Return zero outside offline chunked GPU paged mode; otherwise build `OfflineBatchConfig` and return `resolve_paged_kv_physical_rows(config)`.

Change CLI initialization to:

```cpp
recognizer.init_gpu(opt.ctx_size,
                    opt.gpu_id,
                    gpu_init_slots(opt),
                    gpu_init_physical_kv_rows(opt));
```

- [ ] **Step 4: Update `test_offline_batching` initialization and memory estimate**

Construct its `OfflineBatchConfig` before GPU initialization. In paged mode pass `resolve_paged_kv_physical_rows(cfg)`; in continuous mode pass zero. Change the diagnostic estimator to consume resolved physical rows so its printed MB matches the actual allocation.

- [ ] **Step 5: Build and run CLI/scheduler tests**

Run:

```bash
cmake --build build-cuda -j2 --target test_cli_options test_offline_scheduler test_offline_batching
./build-cuda/test_cli_options
./build-cuda/test_offline_scheduler
```

Expected: all assertions pass and `test_offline_batching --help` still documents `--kv-num-blocks`.

### Task 4: Add a reproducible concurrency/block benchmark matrix

**Files:**
- Create: `tools/bench_paged_kv_pool_concurrency.sh`
- Modify: `docs/benchmarking.md`

- [ ] **Step 1: Add a shell syntax verification that fails before creation**

Run:

```bash
bash -n tools/bench_paged_kv_pool_concurrency.sh
```

Expected: failure because the script does not exist.

- [ ] **Step 2: Create the benchmark script**

Follow the existing benchmark scripts and provide:

```text
--model
--audio
--bin
--out
--repeat
--warmup
--max-chunks
--blocks
--build
```

Default to block size 128, 160 blocks, and variants `batch=12/16/24/32`. Keep dynamic blocks on and prefix cache off. Capture these fields into `summary.tsv`:

```text
batch repeat ok wall_ms rtf tokens_s avg_active no_kv
peak_blocks capacity final_free ownership_errors peak_vram_mib
prefill_wall_ms decode_dispatch_ms avg_decode_step_ms graph_hit_rate
```

Use `nvidia-smi` sampling in a background process only for device-level peak VRAM; always stop and wait for the sampler on normal completion or error.

- [ ] **Step 3: Verify shell syntax and help output**

Run:

```bash
bash -n tools/bench_paged_kv_pool_concurrency.sh
tools/bench_paged_kv_pool_concurrency.sh --help
```

Expected: exit zero and documented defaults.

- [ ] **Step 4: Document the matrix command**

Add to `docs/benchmarking.md`:

```bash
tools/bench_paged_kv_pool_concurrency.sh \
  --build --warmup 1 --repeat 3 --blocks 160
```

State that comparisons require identical block count, chunk plan, graph environment, model, and audio.

### Task 5: Verify correctness and run the GPU experiment

**Files:**
- Verify all modified files
- Generate: `outputs/bench_paged_kv_pool_concurrency/<timestamp>/`
- Update: `docs/superpowers/notes/2026-07-17-unified-asr-benchmark-results.md`

- [ ] **Step 1: Run the complete non-GPU verification set**

Run:

```bash
cmake --build build-cuda -j2 --target test_offline_scheduler test_cli_options test_offline_batching
./build-cuda/test_offline_scheduler
./build-cuda/test_cli_options
python3 -m unittest discover -s test -p 'test_*.py'
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Run a 24-chunk GPU smoke matrix**

Run:

```bash
tools/bench_paged_kv_pool_concurrency.sh \
  --warmup 0 --repeat 1 --max-chunks 24 --blocks 160
```

Require `ok=24/24`, `final_free=160/160`, and zero ownership errors for every variant.

- [ ] **Step 3: Compare smoke-test text outputs**

Extract each variant's chunk text and compare it with the batch-12 output after normalization. Reject variants with repetition degeneration or missing chunks before running the full matrix.

- [ ] **Step 4: Run the full fixed-pool matrix**

Run:

```bash
tools/bench_paged_kv_pool_concurrency.sh \
  --warmup 1 --repeat 3 --blocks 160
```

Do not leave the benchmark process running when the turn ends.

- [ ] **Step 5: Analyze and document the result**

Report median wall time, RTFx, peak VRAM, effective active batch, no-KV admission events, peak blocks, decode-step latency, graph hit rate, and text consistency. If batch 32 is block-limited, run only the additional controlled point `batch=32, blocks=192` and label it as a separate capacity experiment.

- [ ] **Step 6: Final verification**

Re-run the complete test commands from Step 1 after documentation edits, inspect `git status`, and state clearly whether the change improved memory only or both memory and throughput.
