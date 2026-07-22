# FunASR-GGML

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900)
![GGML](https://img.shields.io/badge/runtime-GGML-4B5563)
![Model](https://img.shields.io/badge/model-Fun--ASR--Nano--2512-0F766E)
![License](https://img.shields.io/badge/license-MIT-2563EB)

A native C++17/GGML/CUDA inference runtime for **Fun-ASR-Nano-2512**, focused
on local speech recognition and high-throughput offline long-video
transcription.

> On an RTX 4070 Laptop GPU, the optimized Q8 runtime transcribes a
> 6119.29-second recording in **23.097 seconds** median wall time
> (**264.94x realtime**), versus the original 321.009-second single-request
> path. Full AISHELL-1 evaluation reaches **2.0035% CER** with 100% coverage.

[Quick Start](#quick-start) · [Runtime Design](#runtime-design) ·
[Benchmarks](#verified-performance) · [Accuracy](#accuracy-and-stability) ·
[Benchmark Protocol](docs/benchmarking.md)

The core ASR engine runs locally without a Python runtime. Optional video tools
add URL ingestion, subtitles, summaries, mind maps, key frames, and a Bilibili
browser sidebar.

## Performance At A Glance

Verified on RTX 4070 Laptop GPU + Intel i7-13700H:

| Workload | Runtime configuration | Wall time | Throughput / quality |
| --- | --- | ---: | ---: |
| 6119.29 s long video, 204 fixed windows | Q8, unified scheduler, batch 48, 224 KV blocks | **23.097 s median** | **264.94x realtime** |
| Same long video, original single-request path | Q8, batch 1, continuous KV | 321.009 s | 19.06x realtime |
| AISHELL-1 test, 7176 utterances / 36108.919 s | Q8, batch 48, 112 KV blocks | **151.778 s** | **237.91x realtime, 2.0035% CER** |

The long-video path is **13.90x faster** than the original single-request
baseline. The AISHELL gate produced 7176/7176 hypotheses, with no missing or
empty outputs. Numbers are workload and hardware specific; benchmark scope and
comparison rules are documented in [docs/benchmarking.md](docs/benchmarking.md).

## Features

- **Pure C++17** — no Python runtime, no external dependencies (except GGML)
- **GGUF model format** — single file contains weights + tokenizer + config
- **CPU + GPU** — CPU fallback plus an offline CUDA path for Encoder, Adaptor, and LLM
- **VAD segmentation** — energy VAD by default, optional Silero VAD model via ggml
- **Real-time microphone** — VAD-based streaming recognition with [miniaudio](https://github.com/mackron/miniaudio)
- **Unified offline scheduler** — chunked Prefill and continuous Decode share one token budget
- **Paged KV runtime** — global physical pool, logical block tables, prefix sharing, COW, and dynamic append
- **CUDA fast path** — batched Encoder/Adaptor, Q8 Decoder, Paged Attention, Paged KV Write, and CUDA Graph LRU
- **Measured quality gate** — corpus-level CER, strict coverage checks, and repeat consistency auditing
- **Video workflow** — URL/local media helpers for Bilibili, YouTube, Douyin, and local files
- **Bilibili learning sidebar** — Chrome/Edge MV3 extension for transcripts, DeepSeek summaries, mind maps, key frames, and Q&A
- **C ABI SDK** — `funasr_sdk` wrapper for Qt/Windows integration and hotword injection
- **~985M parameters** — Audio Encoder (SANM) + Audio Adaptor + LLM Decoder (Qwen3-0.6B)

## What Is In This Snapshot

This repository is no longer only the original single-file demo path. The
current snapshot contains three working tracks:

1. **Core local ASR**: C++17 GGUF loader, CPU/GPU decoder, realtime microphone,
   CLI transcription, VAD/window chunking, and a C ABI SDK.
2. **Offline long-video throughput**: a unified request scheduler that packs
   prefill and decode work under one token budget, uses a global paged KV pool,
   batches the acoustic frontend, and retains frequently reused CUDA graphs.
3. **Video learning workflow**: a local FastAPI web service and a no-build
   Bilibili sidebar extension that can analyze the current video, reuse cached
   results, generate study notes, render a clickable mind map, extract key
   frames, and answer questions against the transcript.

## Runtime Design

```text
Audio / long video
    |
    v
Window or VAD chunk planner
    |
    +---- request 0 ---- request 1 ---- ... ---- request N
    |
    v
Frontend pipeline
    |- bounded length bucket
    |- asynchronous CPU Fbank workers             [80-dim Mel + LFR]
    `- batched CUDA Encoder + Adaptor              [560,T] -> [1024,T]
    |
    v
Prompt preparation
    |- shared task/ChatML prefix KV
    `- request-owned audio prompt embeddings
    |
    v
Unified Scheduler                         token budget per step: 1024
    |- active Decode request               1 token / request / step
    `- waiting Prefill request          <= 512 tokens / request / step
    |
    v
Packed Mixed GPU Batch
    |- Q8 Qwen3 28-layer Decoder
    |- custom Paged KV Write
    |- custom Paged Attention
    `- LM Head -> argmax -> next token
    |
    v
EOS / token limit -> BPE text -> release Prompt and KV blocks
```

### Request-centric scheduling

The scheduler operates on requests rather than worker threads. In one Decode
step, up to 48 independent requests contribute one token row to the same GPU
batch. Per-request positions, KV lengths, and block tables keep their attention
histories isolated.

Decode is scheduled first to keep active generation moving. Remaining token
budget is used for chunked Prefill, allowing Prefill and Decode work to share a
single mixed GPU forward instead of running as two globally serialized phases.

### Paged KV ownership

```text
Request A logical blocks [0, 1, 2] -> physical blocks [17, 31,  8]
Request B logical blocks [0, 1, 2] -> physical blocks [42, 11, 29]
Shared task prefix       [0]       -> physical block  [ 5] (ref-counted)
```

Scheduler concurrency and physical KV capacity are independent. Blocks are
appended on demand, shared prefix blocks use reference counts, partial shared
blocks use copy-on-write before mutation, and completed requests return blocks
to the global free list. KV backpressure keeps prepared requests queued when
capacity is temporarily unavailable.

### CUDA Graph cache

Mixed graphs are stored in a bounded multi-entry LRU. A graph signature covers
Prefill/Decode token counts, KV bucket, selected rows, and Prefill layout. A new
shape executes transiently on its first observation and enters the LRU only
after it repeats, preventing one-off variable-length Prefill shapes from
evicting hot Decode graphs.

## Quick Start

### Build (CPU only)

```bash
git clone --recursive https://github.com/huaxin0/FunASR-GGML.git
cd FunASR-GGML
tools/apply_ggml_runtime_patch.sh
cmake -S . -B build -DFUNASR_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### Build (with CUDA GPU)

```bash
cmake -S . -B build-cuda \
  -DFUNASR_CUDA=ON \
  -DFUNASR_BUILD_TESTS=ON
cmake --build build-cuda -j$(nproc)
```

### Run

```bash
# File transcription
./build/funasr-cli -m FunAsr_q8.bin -f audio.wav -otxt

# GPU inference
./build-cuda/funasr-cli -m FunAsr_q8.bin -f audio.wav --gpu -otxt

# Silero VAD subtitles
wget https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
./build-cuda/funasr-cli -m FunAsr_q8.bin -f audio.mp3 --gpu --chunk-mode vad \
  --vad-model ggml-silero-v6.2.0.bin -osrt -o audio.srt

# Optimized long-video transcription
./build-cuda/funasr-cli -m FunAsr_q8.bin -f long_video.wav --gpu \
  --offline-preset long-video -osrt -o long_video.srt

# Video URL transcription helper
python3 tools/funasr_video_ui.py

# Real-time microphone (CPU)
./build/test_realtime FunAsr_q8.bin

# Real-time microphone (GPU)
./build-cuda/test_realtime FunAsr_q8.bin --gpu
```

`long-video` enables the verified unified scheduler, task-prefix KV reuse,
dynamic blocks, frontend batching/prefetch, length bucketing, a 16-entry Graph
LRU, and GPU-memory-aware concurrency tuning. Use `long-video-legacy` to isolate
regressions against the older Decode-only path.

## Usage (C++ API)

### Minimal example

```cpp
#include "pipeline/recognizer.hpp"

int main() {
    funasr::Recognizer recognizer;
    recognizer.init("FunAsr_q8.bin");

    auto result = recognizer.transcribe("audio.wav");
    printf("%s\n", result.text.c_str());
    // Output: 开饭时间早上九点至下午五点。

    return 0;
}
```

### GPU inference

```cpp
funasr::Recognizer recognizer;
recognizer.init("FunAsr_q8.bin");
recognizer.init_gpu();  // Load weights to GPU + warmup

funasr::InferenceConfig config;
config.use_gpu = true;

auto result = recognizer.transcribe("audio.wav", config);
printf("%s\n", result.text.c_str());
```

### Streaming callback

```cpp
auto result = recognizer.transcribe("audio.wav", config,
    [](int id, const std::string& text, bool is_final) {
        if (!is_final) printf("%s", text.c_str());
    }
);
```

## Windows Qt SDK Package

### Rebuild after SDK API changes

If only `funasr_sdk.h`, `sdk/funasr_sdk.cpp`, or prompt/pipeline SDK glue changed,
you do not need to rebuild CUDA or `ggml-cuda.dll`. Rebuild only the SDK target:

```bat
cd /d D:\FunASR-GGML
cmake --build build-msvc --config Release --target funasr_sdk
```

If you also want to run the SDK smoke test:

```bat
cmake --build build-msvc --config Release --target test_sdk_api
.\build-msvc\Release\test_sdk_api.exe
```

After rebuilding, refresh these files in the SDK package:

```bat
copy D:\FunASR-GGML\include\funasr_sdk.h D:\FunASR_SDK\include\
copy D:\FunASR-GGML\build-msvc\Release\funasr_sdk.lib D:\FunASR_SDK\lib\
copy D:\FunASR-GGML\build-msvc\Release\funasr_sdk.dll D:\FunASR_SDK\bin\
```

The existing `ggml*.dll`, CUDA runtime DLLs, and model file can stay unchanged
unless those components were rebuilt separately.

### Runtime package

Put these files next to the Qt program `.exe`:

```text
funasr_sdk.dll
ggml.dll
ggml-base.dll
ggml-cpu.dll
ggml-cuda.dll
cudart64_110.dll
cublas64_11.dll
cublasLt64_11.dll
FunAsr_q8.bin
hotwords.txt
```

`hotwords.txt` is optional. Use one hotword per line:

```text
无人机
航点
返航
QGroundControl
```

Hotwords are plain UTF-8 text. No audio samples are needed.

### Development package

If the Qt program is compiled against this SDK, provide:

```text
FunASR_SDK/
  include/
    funasr_sdk.h

  lib/
    funasr_sdk.lib

  bin/
    funasr_sdk.dll
    ggml.dll
    ggml-base.dll
    ggml-cpu.dll
    ggml-cuda.dll
    cudart64_110.dll
    cublas64_11.dll
    cublasLt64_11.dll

  model/
    FunAsr_q8.bin

  hotwords.txt
```

Qt `.pro` example:

```qmake
INCLUDEPATH += path/to/FunASR_SDK/include
LIBS += -Lpath/to/FunASR_SDK/lib -lfunasr_sdk
```

### SDK usage demo

```cpp
#include "funasr_sdk.h"

// 1. Call once when the program starts.
FunasrConfig cfg;
funasr_get_default_config(&cfg);
cfg.model_path = "FunAsr_q8.bin";
cfg.use_gpu = 1;
cfg.gpu_id = 0;
cfg.ctx_size = 4096;
cfg.max_new_tokens = 220;

FunasrHandle h = funasr_create();
int rc = funasr_init(h, &cfg);
if (rc != 0) {
    const char* err = funasr_last_error(h);
    // Print err.
}

// Optional: load hotwords after init. It can also be called before init.
rc = funasr_load_hotwords_file(h, "hotwords.txt");
if (rc != 0) {
    const char* err = funasr_last_error(h);
    // Hotword loading failed. You may print err and continue without hotwords.
}

// Or set hotwords directly with UTF-8 text.
// funasr_set_hotwords(h, "无人机\n航点\n返航\nQGroundControl");

// 2. Call once for each audio segment.
// audio: 16 kHz mono float32, range -1.0 to 1.0.
char text[8192] = {};
FunasrResult result = {};

rc = funasr_transcribe_f32(
    h,
    audio,
    sample_count,
    text,
    sizeof(text),
    &result
);

if (rc >= 0) {
    // text is UTF-8.
    QString qtext = QString::fromUtf8(text);
}

// 3. Call once before program exit.
funasr_destroy(h);
```

Important notes:

```text
funasr_init loads the model and GPU weights once.
funasr_load_hotwords_file loads UTF-8 text hotwords from a file.
funasr_set_hotwords sets UTF-8 text hotwords directly.
funasr_transcribe_f32 runs once per audio segment.
funasr_destroy releases the SDK handle once at program exit.
Input audio must be 16000 Hz, mono, float32.
Output text is UTF-8.
```

If the input is int16 PCM, convert it to float first:

```cpp
std::vector<float> audioF32(sampleCount);
for (int i = 0; i < sampleCount; ++i) {
    audioF32[i] = pcmI16[i] / 32768.0f;
}
```

### Real-time microphone

```cpp
#include "pipeline/recognizer.hpp"
#include "pipeline/audio_capture.hpp"
#include "pipeline/realtime.hpp"

funasr::Recognizer recognizer;
recognizer.init("FunAsr_q8.bin");
recognizer.init_gpu();

funasr::AudioCapture mic;
funasr::RealtimeRecognizer realtime(recognizer);

mic.set_callback([&](const float* samples, size_t count) {
    realtime.feed_audio(samples, count);
});

funasr::RealtimeConfig config;
config.inference.use_gpu = true;

realtime.start(config, [](int id, const std::string& text,
                          float sec, float ms, float first_ms) {
    if (first_ms >= 0.0f) {
        printf("[%d] %s (%.1fs, %.0fms, TTFT=%.0fms)\n",
               id, text.c_str(), sec, ms, first_ms);
    } else {
        printf("[%d] %s (%.1fs, %.0fms, TTFT=n/a)\n",
               id, text.c_str(), sec, ms);
    }
});

mic.init();
mic.start();
// ... wait for Ctrl+C ...
mic.stop();
realtime.stop();
```

## Voice Activity Detection

`funasr-cli` supports two VAD modes:

```bash
# Energy VAD (default when --vad is set)
./funasr-cli -m FunAsr_q8.bin -f meeting.mp3 --gpu --vad -osrt -o meeting.srt

# Silero VAD (enabled by --vad-model)
./funasr-cli -m FunAsr_q8.bin -f meeting.mp3 --gpu --chunk-mode vad \
  --vad-model ggml-silero-v6.2.0.bin -osrt -o meeting.srt
```

Long audio can also be processed without VAD by using fixed windows:

```bash
./funasr-cli -m FunAsr_q8.bin -f meeting.mp3 --gpu \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  -osrt -o meeting.srt
```

Chunk modes:

```bash
--chunk-mode none      # transcribe the whole input as one segment
--chunk-mode window    # split by fixed --chunk-sec windows
--chunk-mode vad       # split by energy VAD or Silero VAD
```

`--vad` remains as a shortcut for `--chunk-mode vad`.

Download the Silero VAD ggml model:

```bash
wget https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
```

Silero VAD options:

```bash
--vad-threshold <f>        Speech probability threshold (default: 0.5)
--vad-min-speech-ms <n>    Minimum speech duration (default: 250)
--vad-min-silence-ms <n>   Minimum silence duration (default: 100)
--vad-max-speech-sec <f>   Maximum speech duration (default: 30)
--vad-speech-pad-ms <n>    Padding around speech segments (default: 30)
```

The Silero VAD implementation is adapted from [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and runs on CPU. The VAD model file uses whisper.cpp's custom ggml binary format, not GGUF.

## Offline Long-Video Batching

For long recordings, the expensive part is not just one utterance of inference;
it is keeping many independent chunks moving through the decoder without
rebuilding and reallocating everything at each decode step. The offline path in
`pipeline/offline_batching.*` and `test/test_offline_batching.cpp` is the
throughput-oriented route.

The product-facing fast preset enables the measured scheduler configuration and
automatically sizes concurrency and the physical KV pool from available GPU
memory:

```bash
./build-cuda/funasr-cli \
  -m FunAsr_q8.bin -f long_video.wav --gpu \
  --offline-preset long-video -o transcript.txt
```

Use `--offline-preset long-video-legacy` as the regression escape hatch for the
older decode-only scheduler.

The current fast path combines:

- a unified token budget shared by chunked prefill and continuous decode
- task-prefix KV reuse, copy-on-write, and dynamic block append
- a physical paged KV pool independent of scheduler concurrency
- batched Encoder/Adaptor plus asynchronous CPU Fbank prefetch
- bounded length bucketing for variable-duration frontend batches
- a multi-entry LRU CUDA Graph cache with repeated-shape admission
- GPU-memory-aware batch/KV-pool auto tuning

Representative profiling result on RTX 4070 Laptop GPU + i7-13700H:

| Workload | Config | Wall time | Throughput | Notes |
| --- | --- | ---: | ---: | --- |
| ~6119 s audio (~102 min) | Q8, batch=48, 224 blocks, 16 graphs | **23.10 s** median | **264.94x realtime** | Three measured repeats |
| Same workload | previous batched-frontend path, batch=40 | 27.69 s median | 220.99x realtime | Before prefetch/LRU tuning |
| Same workload | single-request continuous baseline | 321.01 s | 19.06x realtime | Before scheduler batching |
| AISHELL-1 test, 7176 utterances | Q8 unified path, batch=48, 112 blocks | 151.78 s | 237.91x realtime | CER 2.0035%, strict gate passed |

The experimental second CUDA stream for Encoder/Adaptor is available through
`--gpu-frontend-overlap on`, but is intentionally off in the preset: on the
RTX 4070 Laptop GPU it competed with decoder kernels and increased wall time.

These numbers are workload and GPU dependent. They are included to show the
order of magnitude and the benchmark shape, not as a universal guarantee. See
`docs/superpowers/notes/2026-05-27-vllm-style-offline-asr-retrospective.md` and
`docs/superpowers/notes/2026-06-11-offline-decode-kernel-profiling.md` for the
full optimization notes.

For reproducible local C++, official FunASR-vLLM, and sherpa-onnx comparisons,
use `tools/benchmark_suite.py`. The complete protocol, commands, result schema,
and fairness rules are documented in [docs/benchmarking.md](docs/benchmarking.md).

## Video URL Transcription

`tools/funasr_video_ui.py` is a small terminal UI for long-video testing. Paste a Bilibili, YouTube, Douyin, or local media path, and it will:

1. download the media with `yt-dlp` when the input is a URL
2. extract 16 kHz mono WAV with `ffmpeg`
3. call `funasr-cli`
4. write `transcript.srt`, `transcript.txt`, `transcript.json`, and `stats.json`

Install the external tools first:

```bash
pip install yt-dlp
# ffmpeg must also be available in PATH
```

Run:

```bash
python3 tools/funasr_video_ui.py
```

The helper is intentionally separate from the C++ inference path, so normal `funasr-cli` usage and builds are unchanged.

### Local Web UI

For the native local browser UI, install FastAPI dependencies and `yt-dlp`:

```bash
pip install fastapi uvicorn python-multipart yt-dlp
# ffmpeg must also be available in PATH
```

Run:

```bash
python3 tools/funasr_web_app.py
```

Open `http://127.0.0.1:8008`. The page accepts either a video URL or a local
uploaded video/audio file, streams the download/ffmpeg/ASR logs, then uses the
long-video offline preset to produce `transcript.srt`, `transcript.txt`,
`transcript.json`, and `stats.json` under `outputs/video_asr_web/`.

If Bilibili rejects anonymous downloads, export a `cookies.txt` file from the
browser where you are logged in and put its path in the `cookies.txt path`
setting. This is the most reliable option under WSL2 because Linux `yt-dlp`
cannot automatically read Windows Edge/Chrome cookies. The default path is:

```text
<repo>/cookies.txt
```

The `cookies-from-browser` setting is still available for Linux browsers inside
WSL, such as `chrome`, `edge`, or `firefox`.

### Bilibili Browser Sidebar MVP

There is also a no-build browser extension for validating the Bilibili in-page
learning flow:

```text
browser-extension/bilibili-funasr-sidebar
```

Start the local service first:

```bash
FUNASR_WEB_APP_PORT=8008 python3 tools/funasr_web_app.py
```

For summary, mind map, key-frame explanation, and video Q&A generation, provide
a DeepSeek API key through the environment or the sidebar:

```bash
DEEPSEEK_API_KEY=sk-... FUNASR_WEB_APP_PORT=8008 python3 tools/funasr_web_app.py
```

Then open `chrome://extensions` or `edge://extensions`, enable Developer mode,
click "Load unpacked", and select `browser-extension/bilibili-funasr-sidebar`.

On a Bilibili video page, the extension injects a right-side FunASR sidebar. It
sends the current URL to the local service, waits for transcription, renders
timestamped transcript nodes, and clicking a timestamp jumps the current page's
`<video>` element to that time.

The sidebar currently supports:

- local service auto-detection on `127.0.0.1:8008`, `:8009`, or `:8010`
- cached-result lookup by Bilibili video id/page, so previous transcripts can be
  loaded without re-running ASR
- grouped subtitle nodes, search, active playback highlight, and clickable
  timestamp jumps
- DeepSeek-generated takeaways, timestamped highlights, and Markdown study notes
- sidebar mind map with collapsible nodes, evidence, questions, and timestamp chips
- key-frame extraction from saved video media based on summary highlights
- video Q&A backed by the saved transcript and summary, with persisted chat history

Generated artifacts are kept under `outputs/video_asr_web/`, including
`transcript.json`, `summary.json`, `summary.md`, `mindmap.json`, extracted
`frames/`, and `chat_sessions/default.json`.

## Model

The model file `FunAsr_q8.bin` is a GGUF-format file containing:

- 1541 tensors (~985M parameters, Q8_0 quantized)
- BPE tokenizer (151936 tokens)
- Architecture config (all hyperparameters)

### Convert from HuggingFace

```bash
python hf_convert_ggml_q8.py --model-dir <huggingface_model> --output FunAsr_q8.bin
```

## Verified Performance

### Long-video optimization history

Fixed workload: 6119.29 seconds of audio, 204 externally fixed 30-second
windows, Q8 model, greedy Decode, RTX 4070 Laptop GPU.

| Runtime stage | Wall time | Speedup vs. original | Main change |
| --- | ---: | ---: | --- |
| Original single-request runtime | 321.009 s | 1.00x | Continuous KV, batch 1 |
| Paged Decode batching, batch 12 | 55.951 s | 5.74x | Request scheduler + Paged Attention |
| Unified scheduler + batched frontend | 27.691 s median | 11.59x | Mixed Prefill/Decode + GPU Encoder/Adaptor |
| Current optimized runtime | **23.097 s median** | **13.90x** | Fbank prefetch + Graph LRU + batch/KV tuning |

Current batch-48 repeats were `23.340 / 23.080 / 23.097 s`; all three produced
identical token sequences for all 204 chunks. Batch 56 consumed 256 blocks but
had a slower `23.263 s` median, so the 8GB auto-tuned profile uses batch 48 and
224 blocks.

The repository's existing same-machine fixed-window reference measured the
official FunASR-vLLM BF16 path at `42.770 s` median. The current C++ Q8 path is
about 1.85x faster for this specific workload. This is not a precision-matched
universal comparison: report Q8 vs. BF16, feature scope, chunking, and timing
boundaries together with the number.

### Runtime observability

| Signal | Long video | AISHELL-1 | Interpretation |
| --- | ---: | ---: | --- |
| Average active Decode batch | 36.98 / 48 | workload dependent | GPU batch remains populated |
| Mixed CUDA Graph hit rate | 66.60% | 6.22% | Variable short requests create many more shapes |
| Fbank prefetch ready rate | >98% | 98.38% | CPU feature work is mostly hidden |
| Frontend padding rate | 0.01% | 5.57% | Length bucket is most useful for variable audio |
| KV ownership errors | 0 | 0 | Block tables/refcounts returned cleanly |

Graph hit rate is a diagnostic, not the optimization target by itself. AISHELL
has 1862 observed mixed shapes; batch 48 is faster than batch 40 even though its
Graph hit rate is lower.

### Reproduce the measurements

```bash
# Long-video batch/KV/Graph matrix
tools/bench_unified_runtime.sh --repeat 3 \
  --profiles optimized48:48:224:16:on:32:off,optimized56:56:256:16:on:32:off

# Full AISHELL-1 coverage and CER gate
tools/bench_asr_accuracy.sh

# Local C++, official PyTorch/vLLM, and optional sherpa-onnx comparison
python3 tools/benchmark_suite.py --help
```

See [docs/benchmarking.md](docs/benchmarking.md) for timing scope, environment
isolation, artifacts, and fair-comparison rules.

## Accuracy And Stability

Full AISHELL-1 test result:

| Metric | Verified result |
| --- | ---: |
| Utterance coverage | **7176 / 7176 (100%)** |
| Missing / empty / malformed | **0 / 0 / 0** |
| Corpus CER | **2.0035%** |
| Audio duration | 36108.919 s |
| End-to-end wall | 151.778 s |
| Audio throughput | 237.91x realtime |
| Peak physical KV usage | 85 / 112 blocks |

The previous path measured 1.9902% CER; the optimized path changes CER by only
0.0133 percentage points and passes the strict 2.2% release gate. Core runtime
verification also covers scheduler planning, mixed metadata, GPU embedding
ownership, block COW/refcounts, CLI presets, result coverage, and benchmark
parsing.

The verified boundary is single-GPU offline batch transcription. Multi-process
serving, request cancellation, long-duration fault injection, arbitrary mixed
Prefill padding, multi-GPU execution, and broader multilingual/noisy-domain
quality suites remain future work. Experimental single-GPU Frontend/LLM stream
overlap is implemented but disabled by default because it regressed performance
on the RTX 4070 Laptop GPU.

## Project Structure

```
FunASR-GGML/
├── core/                    # Infrastructure (config, GGUF reader)
│   ├── config.hpp           #   All params from GGUF metadata, zero hardcoding
│   └── gguf_reader.hpp      #   RAII lifecycle management
├── model/                   # Model loading
│   ├── weights.hpp          #   Weight structs (1541 tensors)
│   ├── model.hpp            #   Aggregate model struct
│   ├── loader.hpp/.cpp      #   Tensor binding
│   └── tokenizer.hpp/.cpp   #   BPE encode/decode
├── compute/                 # Forward computation
│   ├── fbank.hpp/.cpp       #   Audio feature extraction
│   ├── silero_vad.hpp/.cpp  #   Optional Silero VAD segmentation
│   ├── encoder_ops.hpp/.cpp #   70-layer SANM + FSMN encoder
│   ├── adaptor_ops.hpp/.cpp #   Linear + 2-block MHA adaptor
│   ├── kv_cache.hpp         #   CPU KV cache (RAII)
│   ├── llm_ops.hpp/.cpp     #   28-layer Qwen3 decoder (CPU)
│   ├── gpu_context.hpp      #   GPU resource management
│   ├── encoder_adaptor_gpu.hpp # Batched CUDA acoustic frontend
│   ├── gpu_embedding_pool.* #   Owning GPU prompt-embedding pool
│   ├── gpu_mixed_runner.*   #   Packed Prefill/Decode + Graph LRU
│   ├── llm_ops_gpu.hpp/.cpp #   Q8 Decoder + Paged KV/Attention graph
│   ├── gpu_runner.hpp       #   GPU graph execution (gallocr)
│   └── graph_runner.hpp     #   CPU graph execution helper
├── pipeline/                # User-facing API
│   ├── prompt_builder.hpp   #   ChatML prompt construction
│   ├── mixed_batch.*        #   Per-request packed GPU metadata
│   ├── offline_batching.*   #   Unified scheduler + Paged KV ownership
│   ├── pipeline.hpp/.cpp    #   CPU/GPU inference pipeline
│   ├── recognizer.hpp       #   One-line API
│   ├── audio_capture.hpp/.cpp # Microphone capture (miniaudio)
│   └── realtime.hpp         #   Real-time VAD + recognition
├── tools/                   # Benchmark, CER gate, and video helpers
├── test/                    # Unit tests, accuracy runner, and demos
├── patches/                 # Reproducible GGML runtime extensions
└── third_party/
    ├── ggml/                #   GGML library (submodule)
    └── miniaudio.h          #   Audio I/O (header-only)
```

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- GGML (included as submodule)
- miniaudio.h (included, for real-time microphone)
- CUDA Toolkit (optional, for GPU)

## License

Apache License 2.0. See [LICENSE](https://claude.ai/chat/LICENSE).

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) — Tensor computation library
- [FunASR](https://github.com/modelscope/FunASR) — Original speech recognition model
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Reference for KV Cache patterns
- [whisper.cpp Silero VAD](https://github.com/ggml-org/whisper.cpp) — ggml Silero VAD implementation (MIT)
- [miniaudio](https://github.com/mackron/miniaudio) — Audio I/O
