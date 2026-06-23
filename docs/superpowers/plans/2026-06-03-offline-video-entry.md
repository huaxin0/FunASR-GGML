# Offline Video Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productize the existing vLLM-style offline ASR scheduler through `funasr-cli` and the video helper long preset.

**Architecture:** Keep the current serial CLI paths intact and add an explicit `--offline-scheduler` route for chunked audio. The CLI will map chunks into `OfflineBatchTranscriber`, convert ordered `OfflineChunkResult` values back into the existing `TranscriptionItem`/`SegmentResult` output model, and optionally print profile lines mirroring `test_offline_batching`. The video helper will request the new long-video preset and record the offline configuration in `stats.json`.

**Tech Stack:** C++17, CMake, existing FunASR pipeline classes, miniaudio, Python 3 standard library.

---

## File Structure

- Modify `cli/funasr_cli.cpp`: add offline CLI options, long-video preset, offline scheduler routing, and profile printing.
- Create `test/test_cli_options.cpp`: unit-style coverage for CLI option parsing and preset behavior.
- Modify `CMakeLists.txt`: build `test_cli_options`.
- Modify `tools/funasr_video_ui.py`: add offline scheduler fields to config, long preset defaults, command arguments, and `stats.json`.

## Task 1: Add CLI Option Parsing Coverage

**Files:**
- Create: `test/test_cli_options.cpp`
- Modify: `CMakeLists.txt`
- Modify: `cli/funasr_cli.cpp`

- [ ] **Step 1: Create a failing CLI option test**

Create `test/test_cli_options.cpp` with this content:

```cpp
#define FUNASR_CLI_TESTING
#define main funasr_cli_main_for_test
#include "../cli/funasr_cli.cpp"
#undef main

#include <cstdio>
#include <cstring>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("FAIL: %s\n", msg); \
            return false; \
        } \
        std::printf("PASS: %s\n", msg); \
    } while (0)

namespace {

char* arg(char* value) {
    return value;
}

bool parse_with_args(std::initializer_list<char*> raw_args, Options& opt) {
    std::vector<char*> args(raw_args);
    return parse_args(static_cast<int>(args.size()), args.data(), opt);
}

bool test_offline_flags_parse() {
    std::printf("\n--- CLI offline flags parse ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"zh.wav"),
        arg((char*)"--offline-scheduler"),
        arg((char*)"--offline-profile"),
        arg((char*)"--batch-size"), arg((char*)"12"),
        arg((char*)"--kv-mode"), arg((char*)"paged"),
        arg((char*)"--kv-block-size"), arg((char*)"128"),
        arg((char*)"--kv-num-blocks"), arg((char*)"384"),
    }, opt);

    TEST_ASSERT(ok, "offline flags parse successfully");
    TEST_ASSERT(opt.offline_scheduler, "offline scheduler enabled");
    TEST_ASSERT(opt.offline_profile, "offline profile enabled");
    TEST_ASSERT(opt.offline_batch_size == 12, "batch size parsed");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "paged kv mode parsed");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "kv block size parsed");
    TEST_ASSERT(opt.offline_kv_num_blocks == 384, "kv block count parsed");
    return true;
}

bool test_long_video_preset() {
    std::printf("\n--- CLI long-video preset ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--gpu"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
    }, opt);

    TEST_ASSERT(ok, "long-video preset parses");
    TEST_ASSERT(opt.offline_scheduler, "preset enables offline scheduler");
    TEST_ASSERT(opt.offline_profile, "preset enables offline profile");
    TEST_ASSERT(opt.offline_batch_size == 12, "preset batch size is 12");
    TEST_ASSERT(opt.offline_kv_mode == OfflineKVMode::Paged, "preset uses paged kv");
    TEST_ASSERT(opt.offline_kv_block_size == 128, "preset block size is 128");
    TEST_ASSERT(opt.ctx_size == 4096, "preset ctx size is 4096");
    TEST_ASSERT(opt.chunk_mode == ChunkMode::Window, "preset uses window chunks");
    TEST_ASSERT(opt.chunk_sec == 30, "preset chunk size is 30 seconds");
    TEST_ASSERT(opt.max_tokens == 220, "preset max tokens is 220");
    return true;
}

bool test_preset_can_be_overridden_afterwards() {
    std::printf("\n--- CLI preset override ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"video.wav"),
        arg((char*)"--offline-preset"), arg((char*)"long-video"),
        arg((char*)"--batch-size"), arg((char*)"8"),
        arg((char*)"--kv-block-size"), arg((char*)"64"),
    }, opt);

    TEST_ASSERT(ok, "preset with overrides parses");
    TEST_ASSERT(opt.offline_batch_size == 8, "following batch size overrides preset");
    TEST_ASSERT(opt.offline_kv_block_size == 64, "following block size overrides preset");
    return true;
}

bool test_invalid_kv_mode_fails() {
    std::printf("\n--- CLI invalid kv mode ---\n");
    Options opt;
    bool ok = parse_with_args({
        arg((char*)"funasr-cli"),
        arg((char*)"-m"), arg((char*)"FunAsr_q8.bin"),
        arg((char*)"-f"), arg((char*)"zh.wav"),
        arg((char*)"--kv-mode"), arg((char*)"bad"),
    }, opt);

    TEST_ASSERT(!ok, "invalid kv mode fails parsing");
    return true;
}

} // namespace

int main() {
    int failed = 0;
    if (!test_offline_flags_parse()) failed++;
    if (!test_long_video_preset()) failed++;
    if (!test_preset_can_be_overridden_afterwards()) failed++;
    if (!test_invalid_kv_mode_fails()) failed++;

    if (failed == 0) {
        std::printf("\nAll CLI option tests passed!\n");
        return 0;
    }
    std::printf("\n%d CLI option tests failed.\n", failed);
    return 1;
}
```

- [ ] **Step 2: Add the test target**

Modify `CMakeLists.txt` after the `funasr-cli` target:

```cmake
add_executable(test_cli_options test/test_cli_options.cpp)
target_link_libraries(test_cli_options PRIVATE funasr_pipeline)
```

- [ ] **Step 3: Run the test and confirm it fails**

Run:

```bash
cmake --build build-cuda --target test_cli_options -j$(nproc)
```

Expected: compile failure because `OfflineKVMode`, `offline_scheduler`, `offline_profile`, `offline_batch_size`, `offline_kv_mode`, `offline_kv_block_size`, `offline_kv_num_blocks`, and `--offline-preset` parsing do not exist.

- [ ] **Step 4: Add CLI option fields and parsing**

Modify `cli/funasr_cli.cpp`.

Add this include near the other pipeline includes:

```cpp
#include "pipeline/offline_batching.hpp"
```

Add this enum after `ChunkMode`:

```cpp
enum class OfflineKVMode {
    Continuous,
    Paged,
};
```

Add these fields to `Options`:

```cpp
    bool offline_scheduler = false;
    bool offline_profile = false;
    int offline_batch_size = 12;
    OfflineKVMode offline_kv_mode = OfflineKVMode::Continuous;
    int offline_kv_block_size = 64;
    int offline_kv_num_blocks = 0;
```

Add this helper after `parse_chunk_mode()`:

```cpp
bool parse_offline_kv_mode(const std::string& value, OfflineKVMode& out) {
    std::string mode = to_lower(value);
    if (mode == "continuous") {
        out = OfflineKVMode::Continuous;
        return true;
    }
    if (mode == "paged") {
        out = OfflineKVMode::Paged;
        return true;
    }
    return false;
}

bool apply_offline_preset(const std::string& value, Options& opt) {
    std::string preset = to_lower(value);
    if (preset != "long-video") {
        return false;
    }

    opt.offline_scheduler = true;
    opt.offline_profile = true;
    opt.offline_batch_size = 12;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    opt.offline_kv_block_size = 128;
    opt.ctx_size = 4096;
    opt.chunk_mode = ChunkMode::Window;
    opt.chunk_mode_set = true;
    opt.chunk_sec = 30;
    opt.max_tokens = 220;
    return true;
}
```

Add these branches to `parse_args()`:

```cpp
        } else if (arg == "--offline-scheduler") {
            opt.offline_scheduler = true;
        } else if (arg == "--offline-profile") {
            opt.offline_profile = true;
        } else if (arg == "--offline-preset") {
            const char* value = need_value(arg.c_str());
            if (!value || !apply_offline_preset(value, opt)) {
                std::fprintf(stderr, "Invalid --offline-preset value: use long-video\n");
                return false;
            }
        } else if (arg == "--batch-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.offline_batch_size)) {
                std::fprintf(stderr, "Invalid --batch-size value\n");
                return false;
            }
        } else if (arg == "--kv-mode") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_offline_kv_mode(value, opt.offline_kv_mode)) {
                std::fprintf(stderr, "Invalid --kv-mode value: use continuous or paged\n");
                return false;
            }
        } else if (arg == "--kv-block-size") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_positive_int(value, opt.offline_kv_block_size)) {
                std::fprintf(stderr, "Invalid --kv-block-size value\n");
                return false;
            }
        } else if (arg == "--kv-num-blocks") {
            const char* value = need_value(arg.c_str());
            if (!value || !parse_nonnegative_int(value, opt.offline_kv_num_blocks)) {
                std::fprintf(stderr, "Invalid --kv-num-blocks value\n");
                return false;
            }
```

Update `print_usage()` with:

```cpp
        "Offline scheduler:\n"
        "  --offline-scheduler    Use continuous offline scheduler for chunked audio\n"
        "  --offline-profile      Print scheduler and paged decode profile lines\n"
        "  --offline-preset <n>   Preset: long-video\n"
        "  --batch-size <n>       Offline scheduler active request limit (default: 12)\n"
        "  --kv-mode <mode>       Offline KV mode: continuous, paged (default: continuous)\n"
        "  --kv-block-size <n>    Paged KV block size (default: 64)\n"
        "  --kv-num-blocks <n>    Paged KV block count (default: derived)\n"
        "\n"
```

- [ ] **Step 5: Run the option test**

Run:

```bash
cmake --build build-cuda --target test_cli_options -j$(nproc)
./build-cuda/test_cli_options
```

Expected: all CLI option tests pass.

- [ ] **Step 6: Commit if git is available**

Run:

```bash
git status --short
git add CMakeLists.txt cli/funasr_cli.cpp test/test_cli_options.cpp
git commit -m "feat: add offline scheduler cli options"
```

Expected in this workspace: `git status` may fail because `.git` is not a usable repository. If it fails, record that and continue without committing.

## Task 2: Route Chunked CLI Audio Through OfflineBatchTranscriber

**Files:**
- Modify: `cli/funasr_cli.cpp`
- Test: `test/test_cli_options.cpp`

- [ ] **Step 1: Add small pure helpers for testable routing decisions**

Modify `cli/funasr_cli.cpp` by adding these helpers after `chunk_mode_name()`:

```cpp
const char* offline_kv_mode_name(OfflineKVMode mode) {
    switch (mode) {
        case OfflineKVMode::Paged: return "paged";
        case OfflineKVMode::Continuous:
        default: return "continuous";
    }
}

bool should_use_offline_scheduler(const Options& opt, bool loaded, size_t chunk_count) {
    return opt.offline_scheduler &&
           loaded &&
           chunk_count > 0 &&
           opt.chunk_mode != ChunkMode::None;
}

funasr::OfflineBatchConfig make_offline_batch_config(const Options& opt) {
    funasr::OfflineBatchConfig cfg;
    cfg.batch_size = opt.offline_batch_size;
    cfg.ctx_size = opt.ctx_size;
    cfg.max_tokens = opt.max_tokens;
    cfg.n_threads = opt.n_threads;
    cfg.use_gpu = opt.use_gpu;
    cfg.gpu_id = opt.gpu_id;
    cfg.use_paged_kv = opt.offline_kv_mode == OfflineKVMode::Paged;
    cfg.kv_block_size = opt.offline_kv_block_size;
    cfg.kv_num_blocks = opt.offline_kv_num_blocks;
    return cfg;
}
```

- [ ] **Step 2: Add failing helper tests**

Add these tests to `test/test_cli_options.cpp`:

```cpp
bool test_offline_routing_requires_chunks() {
    std::printf("\n--- CLI offline routing helper ---\n");
    Options opt;
    opt.offline_scheduler = true;
    opt.chunk_mode = ChunkMode::Window;

    TEST_ASSERT(should_use_offline_scheduler(opt, true, 3),
                "offline scheduler routes loaded chunked audio");
    TEST_ASSERT(!should_use_offline_scheduler(opt, true, 0),
                "offline scheduler does not route empty chunk list");
    TEST_ASSERT(!should_use_offline_scheduler(opt, false, 3),
                "offline scheduler does not route unloaded audio");

    opt.chunk_mode = ChunkMode::None;
    TEST_ASSERT(!should_use_offline_scheduler(opt, true, 3),
                "offline scheduler does not route chunk-mode none");
    return true;
}

bool test_make_offline_batch_config() {
    std::printf("\n--- CLI offline config helper ---\n");
    Options opt;
    opt.offline_batch_size = 12;
    opt.ctx_size = 4096;
    opt.max_tokens = 220;
    opt.n_threads = 6;
    opt.use_gpu = true;
    opt.gpu_id = 0;
    opt.offline_kv_mode = OfflineKVMode::Paged;
    opt.offline_kv_block_size = 128;
    opt.offline_kv_num_blocks = 384;

    funasr::OfflineBatchConfig cfg = make_offline_batch_config(opt);
    TEST_ASSERT(cfg.batch_size == 12, "offline config batch size copied");
    TEST_ASSERT(cfg.ctx_size == 4096, "offline config ctx copied");
    TEST_ASSERT(cfg.max_tokens == 220, "offline config max tokens copied");
    TEST_ASSERT(cfg.n_threads == 6, "offline config threads copied");
    TEST_ASSERT(cfg.use_gpu, "offline config gpu copied");
    TEST_ASSERT(cfg.gpu_id == 0, "offline config gpu id copied");
    TEST_ASSERT(cfg.use_paged_kv, "offline config paged kv copied");
    TEST_ASSERT(cfg.kv_block_size == 128, "offline config block size copied");
    TEST_ASSERT(cfg.kv_num_blocks == 384, "offline config block count copied");
    return true;
}
```

Call them from `main()`:

```cpp
    if (!test_offline_routing_requires_chunks()) failed++;
    if (!test_make_offline_batch_config()) failed++;
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
cmake --build build-cuda --target test_cli_options -j$(nproc)
./build-cuda/test_cli_options
```

Expected: pass after Step 1 helpers are present.

- [ ] **Step 4: Implement offline transcription branch**

In `cli/funasr_cli.cpp`, replace the existing per-segment loop inside `else if (loaded) { ... }` with a branch before that loop:

```cpp
            if (should_use_offline_scheduler(opt, loaded, segments.size())) {
                if (opt.offline_kv_mode == OfflineKVMode::Paged && !opt.use_gpu) {
                    std::fprintf(stderr,
                                 "[OfflineCLI] WARNING: paged offline mode is fastest with GPU; "
                                 "current run is CPU/compatibility mode\n");
                }

                funasr::OfflineBatchConfig offline_cfg = make_offline_batch_config(opt);
                funasr::OfflineBatchTranscriber transcriber(recognizer);
                std::vector<funasr::OfflineChunkResult> offline_results;
                {
                    ScopedStdoutToStderr redirect;
                    offline_results = transcriber.transcribe(
                        samples,
                        static_cast<int>(sample_rate),
                        segments,
                        offline_cfg);
                }

                for (const auto& result : offline_results) {
                    if (!result.ok || result.text.empty()) {
                        continue;
                    }
                    std::string cleaned_text = cleanup_repeated_text(result.text);
                    if (cleaned_text.empty()) {
                        continue;
                    }
                    SegmentResult seg_result;
                    seg_result.text = cleaned_text;
                    seg_result.start_sec = result.start_sec;
                    seg_result.end_sec = result.end_sec;
                    item.segments.push_back(seg_result);
                    item.text = append_without_overlap(item.text, cleaned_text);
                }

                if (opt.offline_profile) {
                    print_offline_profile(transcriber.last_stats(), recognizer, opt);
                }
            } else {
                for (size_t seg_idx = 0; seg_idx < segments.size(); seg_idx++) {
                    const auto& segment = segments[seg_idx];
                    size_t seg_len = segment.infer_end_sample > segment.infer_start_sample
                        ? segment.infer_end_sample - segment.infer_start_sample
                        : 0;
                    if (seg_len == 0) continue;

                    std::vector<float> segment_samples(
                        samples.begin() + static_cast<std::ptrdiff_t>(segment.infer_start_sample),
                        samples.begin() + static_cast<std::ptrdiff_t>(segment.infer_end_sample));

                    funasr::InferenceResult result;
                    {
                        ScopedStdoutToStderr redirect;
                        result = recognizer.transcribe_audio(
                            segment_samples.data(), segment_samples.size(), config);
                    }

                    if (!result.text.empty()) {
                        std::string cleaned_text = cleanup_repeated_text(result.text);
                        if (cleaned_text.empty()) {
                            continue;
                        }
                        SegmentResult seg_result;
                        seg_result.text = cleaned_text;
                        seg_result.start_sec = segment.start_sec;
                        seg_result.end_sec = segment.end_sec;
                        item.segments.push_back(seg_result);
                        item.text = append_without_overlap(item.text, cleaned_text);
                    }

                    print_vad_progress(static_cast<int>(seg_idx + 1),
                                       static_cast<int>(segments.size()),
                                       segment, item.duration_sec);
                }
            }
```

Add a forward declaration before `main()` if the compiler needs it:

```cpp
void print_offline_profile(
    const funasr::OfflineBatchStats& stats,
    const funasr::Recognizer& recognizer,
    const Options& opt);
```

Task 3 defines the full `print_offline_profile()` body. Until Task 3, add this minimal body to compile:

```cpp
void print_offline_profile(
    const funasr::OfflineBatchStats&,
    const funasr::Recognizer&,
    const Options&) {}
```

- [ ] **Step 5: Build CLI**

Run:

```bash
cmake --build build-cuda --target funasr-cli test_cli_options -j$(nproc)
./build-cuda/test_cli_options
```

Expected: build succeeds and option tests pass.

- [ ] **Step 6: Run CPU smoke test**

Run:

```bash
./build-cuda/funasr-cli -m FunAsr_q8.bin -f zh.wav --chunk-mode window \
  --chunk-sec 3 --offline-scheduler --batch-size 2 --kv-mode paged \
  --ctx-size 256 --max-tokens 3 -osrt -o /tmp/funasr_offline_cli_smoke.srt
```

Expected: command exits `0` and writes `/tmp/funasr_offline_cli_smoke.srt`.

- [ ] **Step 7: Commit if git is available**

Run:

```bash
git status --short
git add cli/funasr_cli.cpp test/test_cli_options.cpp
git commit -m "feat: route cli chunks through offline scheduler"
```

Expected in this workspace: `git status` may fail because `.git` is not a usable repository. If it fails, record that and continue without committing.

## Task 3: Print Offline Scheduler And Paged Decode Profiles

**Files:**
- Modify: `cli/funasr_cli.cpp`

- [ ] **Step 1: Replace the minimal profile stub**

Replace the stub from Task 2 with:

```cpp
void print_offline_profile(
    const funasr::OfflineBatchStats& stats,
    const funasr::Recognizer& recognizer,
    const Options& opt) {
    std::fprintf(stderr,
                 "[OfflineCLI] offline_stats: chunks=%d batch=%d kv=%s "
                 "block_size=%d blocks_peak=%d/%d decode_steps=%d "
                 "grouped_calls=%d fallback_calls=%d avg_active=%.2f\n",
                 stats.total_chunks,
                 stats.batch_size,
                 stats.use_paged_kv ? "paged" : "continuous",
                 stats.kv_block_size,
                 stats.peak_blocks_in_use,
                 stats.kv_block_capacity,
                 stats.decode_steps,
                 stats.decode_group_calls,
                 stats.decode_fallback_calls,
                 stats.average_active_batch());

    std::fprintf(stderr,
                 "[OfflineCLI] fallback_reasons: single=%d token_id=%d "
                 "host_embed=%d serial_env=%d invalid=%d\n",
                 stats.fallback_reasons.single_request,
                 stats.fallback_reasons.token_id_fast_path_unavailable,
                 stats.fallback_reasons.host_embedding_batch_unavailable,
                 stats.fallback_reasons.serial_env_forced,
                 stats.fallback_reasons.invalid_paged_input);

    std::fprintf(stderr,
                 "[OfflineCLI] scheduler_profile: admit=%d/%d no_kv=%d "
                 "admit_rounds=%d avg_admit_round=%.2f max_admit_round=%d "
                 "avg_prefill_wall=%.3fms avg_decode_dispatch=%.3fms idle_steps=%d\n",
                 stats.admit_success,
                 stats.admit_attempts,
                 stats.admit_no_kv_capacity,
                 stats.admit_rounds,
                 stats.average_admit_round_size(),
                 stats.max_admit_round_size,
                 stats.average_prefill_wall_ms(),
                 stats.average_decode_dispatch_ms(),
                 stats.scheduler_idle_steps);

    if (!(opt.use_gpu && stats.use_paged_kv)) {
        return;
    }

    const funasr::PagedDecodeProfile profile = recognizer.gpu_paged_decode_profile();
    if (profile.calls <= 0) {
        return;
    }

    std::fprintf(stderr,
                 "[OfflineCLI] paged_profile: calls=%ld build=%.3fms "
                 "alloc=%.3fms set=%.3fms compute=%.3fms get=%.3fms total=%.3fms\n",
                 profile.calls,
                 profile.build_ms / static_cast<double>(profile.calls),
                 profile.alloc_ms / static_cast<double>(profile.calls),
                 profile.set_input_ms / static_cast<double>(profile.calls),
                 profile.compute_ms / static_cast<double>(profile.calls),
                 profile.get_ms / static_cast<double>(profile.calls),
                 profile.avg_total_ms());

    if (profile.paged_attn_calls > 0) {
        std::fprintf(stderr,
                     "[OfflineCLI] paged_attn_profile: calls=%ld total=%.3fms "
                     "avg_kernel=%.6fms graph_compute_share=%.2f%%\n",
                     profile.paged_attn_calls,
                     profile.paged_attn_ms,
                     profile.avg_paged_attn_ms(),
                     profile.compute_ms > 0.0
                        ? 100.0 * profile.paged_attn_ms / profile.compute_ms
                        : 0.0);
    }

    if (profile.graph_cache_probe_calls > 0) {
        std::fprintf(stderr,
                     "[OfflineCLI] paged_graph_cache_probe: calls=%ld "
                     "shape_hits=%ld shape_hit_rate=%.2f%% "
                     "param_hits=%ld param_hit_rate=%.2f%% "
                     "full_hits=%ld full_hit_rate=%.2f%% "
                     "cache_hits=%ld cache_misses=%ld cache_hit_rate=%.2f%%\n",
                     profile.graph_cache_probe_calls,
                     profile.shape_cache_probe_hits,
                     profile.shape_cache_probe_hit_rate(),
                     profile.param_cache_probe_hits,
                     profile.param_cache_probe_hit_rate(),
                     profile.full_graph_cache_probe_hits,
                     profile.full_graph_cache_probe_hit_rate(),
                     profile.graph_cache_hits,
                     profile.graph_cache_misses,
                     profile.graph_cache_hit_rate());
    }
}
```

- [ ] **Step 2: Build CLI**

Run:

```bash
cmake --build build-cuda --target funasr-cli -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 3: Run profile smoke test**

Run:

```bash
./build-cuda/funasr-cli -m FunAsr_q8.bin -f zh.wav --chunk-mode window \
  --chunk-sec 3 --offline-scheduler --offline-profile --batch-size 2 \
  --kv-mode paged --ctx-size 256 --max-tokens 3 -osrt \
  -o /tmp/funasr_offline_cli_profile.srt
```

Expected: stderr includes these lines:

```text
[OfflineCLI] offline_stats:
[OfflineCLI] fallback_reasons:
[OfflineCLI] scheduler_profile:
```

- [ ] **Step 4: Commit if git is available**

Run:

```bash
git status --short
git add cli/funasr_cli.cpp
git commit -m "feat: print offline cli profiles"
```

Expected in this workspace: `git status` may fail because `.git` is not a usable repository. If it fails, record that and continue without committing.

## Task 4: Productize Video Helper Long Preset

**Files:**
- Modify: `tools/funasr_video_ui.py`

- [ ] **Step 1: Extend config and preset data**

Modify `Config`:

```python
@dataclass
class Config:
    source: str
    model: Path
    funasr_cli: Path
    output_dir: Path
    use_gpu: bool
    use_vad: bool
    chunk_mode: str
    chunk_sec: str
    vad_model: Path | None
    vad_threshold: str
    vad_min_silence_ms: str
    vad_speech_pad_ms: str
    vad_max_speech_sec: str
    ctx_size: str
    max_tokens: str
    srt_max_chars: str
    offline_scheduler: bool
    offline_profile: bool
    offline_batch_size: str
    offline_kv_mode: str
    offline_kv_block_size: str
    keep_media: bool
```

Modify `Preset`:

```python
@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    chunk_mode: str
    chunk_sec: str
    vad_min_silence_ms: str
    vad_speech_pad_ms: str
    vad_max_speech_sec: str
    ctx_size: str
    max_tokens: str
    srt_max_chars: str
    offline_scheduler: bool
    offline_profile: bool
    offline_batch_size: str
    offline_kv_mode: str
    offline_kv_block_size: str
```

Set the long preset offline values:

```python
    "1": Preset(
        name="long",
        description="fixed 30s windows, ctx 4096, offline paged batch",
        chunk_mode="window",
        chunk_sec="30",
        vad_min_silence_ms="800",
        vad_speech_pad_ms="120",
        vad_max_speech_sec="30",
        ctx_size="4096",
        max_tokens="220",
        srt_max_chars="28",
        offline_scheduler=True,
        offline_profile=True,
        offline_batch_size="12",
        offline_kv_mode="paged",
        offline_kv_block_size="128",
    ),
```

Set `balanced` and `vad` offline values:

```python
        offline_scheduler=False,
        offline_profile=False,
        offline_batch_size="12",
        offline_kv_mode="continuous",
        offline_kv_block_size="64",
```

- [ ] **Step 2: Collect offline config**

In `collect_config()`, add:

```python
    offline_scheduler = prompt_bool("Use offline scheduler", preset.offline_scheduler)
    offline_profile = False
    offline_batch_size = preset.offline_batch_size
    offline_kv_mode = preset.offline_kv_mode
    offline_kv_block_size = preset.offline_kv_block_size
    if offline_scheduler:
        offline_profile = prompt_bool("Print offline profile", preset.offline_profile)
        offline_batch_size = prompt("Offline batch size", offline_batch_size)
        offline_kv_mode = prompt("Offline KV mode (continuous/paged)", offline_kv_mode).lower()
        offline_kv_block_size = prompt("Offline KV block size", offline_kv_block_size)
```

Add these fields to the returned `Config`:

```python
        offline_scheduler=offline_scheduler,
        offline_profile=offline_profile,
        offline_batch_size=offline_batch_size,
        offline_kv_mode=offline_kv_mode,
        offline_kv_block_size=offline_kv_block_size,
```

- [ ] **Step 3: Pass offline flags to CLI**

In `run_asr()`, before `run_command(cmd)`, add:

```python
    if cfg.offline_scheduler:
        cmd.append("--offline-scheduler")
        cmd.extend(["--batch-size", cfg.offline_batch_size])
        cmd.extend(["--kv-mode", cfg.offline_kv_mode])
        cmd.extend(["--kv-block-size", cfg.offline_kv_block_size])
        if cfg.offline_profile:
            cmd.append("--offline-profile")
```

- [ ] **Step 4: Write offline settings to stats**

In `write_side_outputs()`, add these fields to the `stats.json` object:

```python
                "offline_scheduler": cfg.offline_scheduler,
                "offline_profile": cfg.offline_profile,
                "offline_batch_size": int(cfg.offline_batch_size) if cfg.offline_scheduler else None,
                "offline_kv_mode": cfg.offline_kv_mode if cfg.offline_scheduler else None,
                "offline_kv_block_size": int(cfg.offline_kv_block_size) if cfg.offline_scheduler else None,
```

- [ ] **Step 5: Run Python syntax check**

Run:

```bash
python3 -m py_compile tools/funasr_video_ui.py
```

Expected: no output and exit `0`.

- [ ] **Step 6: Commit if git is available**

Run:

```bash
git status --short
git add tools/funasr_video_ui.py
git commit -m "feat: enable offline video preset"
```

Expected in this workspace: `git status` may fail because `.git` is not a usable repository. If it fails, record that and continue without committing.

## Task 5: Final Verification

**Files:**
- Verify: `cli/funasr_cli.cpp`
- Verify: `tools/funasr_video_ui.py`
- Verify: `docs/superpowers/specs/2026-06-03-offline-video-entry-design.md`

- [ ] **Step 1: Build all touched C++ targets**

Run:

```bash
cmake --build build-cuda --target funasr-cli test_cli_options test_offline_scheduler test_offline_batching -j$(nproc)
```

Expected: all targets build successfully.

- [ ] **Step 2: Run unit and smoke checks**

Run:

```bash
./build-cuda/test_cli_options
./build-cuda/test_offline_scheduler
python3 -m py_compile tools/funasr_video_ui.py
```

Expected: CLI option tests pass, offline scheduler tests pass, Python compile check passes.

- [ ] **Step 3: Run CLI offline smoke test**

Run:

```bash
./build-cuda/funasr-cli -m FunAsr_q8.bin -f zh.wav --chunk-mode window \
  --chunk-sec 3 --offline-scheduler --offline-profile --batch-size 2 \
  --kv-mode paged --ctx-size 256 --max-tokens 3 -osrt \
  -o /tmp/funasr_offline_cli_final.srt
```

Expected:

```text
exit code 0
/tmp/funasr_offline_cli_final.srt exists
stderr includes [OfflineCLI] offline_stats
```

- [ ] **Step 4: Run benchmark command when GPU data is available**

Run:

```bash
FUNASR_PAGED_KV_WRITE_OP=1 \
FUNASR_PAGED_DECODE_BUCKET_MAX_KV=1 \
FUNASR_PAGED_DECODE_GRAPH_CACHE=1 \
./build-cuda/funasr-cli -m FunAsr_q8.bin \
  -f outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --offline-preset long-video -osrt -o /tmp/funasr_long_video.srt
```

Expected:

```text
exit code 0
/tmp/funasr_long_video.srt exists
stderr includes [OfflineCLI] paged_profile when paged GPU decode is used
```

- [ ] **Step 5: Final commit if git is available**

Run:

```bash
git status --short
git add CMakeLists.txt cli/funasr_cli.cpp test/test_cli_options.cpp tools/funasr_video_ui.py
git commit -m "feat: productize offline video scheduler entry"
```

Expected in this workspace: `git status` may fail because `.git` is not a usable repository. If it fails, record that in the final implementation summary.

## Self-Review

Spec coverage:

- CLI options and long-video preset are covered in Task 1.
- Offline scheduler routing is covered in Task 2.
- Profile and stats output is covered in Task 3.
- Video UI long preset and `stats.json` fields are covered in Task 4.
- Realtime remains unchanged because no realtime files are modified.
- Verification is covered in Task 5.

Placeholder scan:

- No task contains unresolved marker text or unnamed deferred work.
- All changed files are named explicitly.
- Each code-changing task includes concrete code snippets.

Type consistency:

- `OfflineKVMode`, `Options` fields, helper names, and test references match across Tasks 1 and 2.
- `print_offline_profile()` signature is introduced before use and fully defined in Task 3.
- Python `Config` and `Preset` field names match the `run_asr()` and `write_side_outputs()` references.
