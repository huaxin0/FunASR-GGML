# FunASR-GGML

C++ speech recognition inference engine using [GGML](https://github.com/ggerganov/ggml), powered by FunASR's SenseVoice architecture.

**Audio → Text**, fully local for core ASR, with optional local web/video tools.

This branch also includes a long-video offline path inspired by vLLM-style
continuous batching and paged KV cache, plus a Bilibili browser sidebar for
turning videos into clickable transcripts, summaries, mind maps, key frames, and
video Q&A.

## Features

- **Pure C++17** — no Python runtime, no external dependencies (except GGML)
- **GGUF model format** — single file contains weights + tokenizer + config
- **CPU + GPU** — LLM decoder runs on CUDA GPU (Encoder/Adaptor stay on CPU)
- **VAD segmentation** — energy VAD by default, optional Silero VAD model via ggml
- **Real-time microphone** — VAD-based streaming recognition with [miniaudio](https://github.com/mackron/miniaudio)
- **Long-video offline batching** — fixed-window or VAD chunks with scheduler batching, paged KV, and graph cache
- **Video workflow** — URL/local media helpers for Bilibili, YouTube, Douyin, and local files
- **Bilibili learning sidebar** — Chrome/Edge MV3 extension for transcripts, DeepSeek summaries, mind maps, key frames, and Q&A
- **C ABI SDK** — `funasr_sdk` wrapper for Qt/Windows integration and hotword injection
- **~985M parameters** — Audio Encoder (SANM) + Audio Adaptor + LLM Decoder (Qwen3-0.6B)

## What Is In This Snapshot

This repository is no longer only the original single-file demo path. The
current snapshot contains three working tracks:

1. **Core local ASR**: C++17 GGUF loader, CPU/GPU decoder, realtime microphone,
   CLI transcription, VAD/window chunking, and a C ABI SDK.
2. **Offline long-video throughput**: an offline scheduler that batches chunk
   decode steps, uses paged KV cache for decode requests, keeps a stable decode
   graph shape, and caches the hottest graph.
3. **Video learning workflow**: a local FastAPI web service and a no-build
   Bilibili sidebar extension that can analyze the current video, reuse cached
   results, generate study notes, render a clickable mind map, extract key
   frames, and answer questions against the transcript.

## Architecture

```
WAV (16kHz)
  → Fbank (80-dim mel + LFR) → [560, T]
  → Audio Encoder (50 SANM + 20 TP layers, FSMN) → [512, T]        [CPU]
  → Audio Adaptor (linear + 2 attention blocks) → [1024, T]         [CPU]
  → LLM Decoder (28-layer Qwen3, GQA-8, RoPE, KV Cache) → logits   [CPU or GPU]
  → BPE Decode → Text
```

## Quick Start

### Build (CPU only)

```bash
git clone --recursive https://github.com/huaxin0/FunASR-GGML.git
cd FunASR-GGML
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build (with CUDA GPU)

```bash
cmake .. -DFUNASR_CUDA=ON
make -j$(nproc)
```

### Run

```bash
# File transcription
./test_pipeline ../FunAsr_q8.bin audio.wav

# GPU inference
./test_gpu ../FunAsr_q8.bin audio.wav

# Silero VAD subtitles
wget https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
./funasr-cli -m ../FunAsr_q8.bin -f audio.mp3 --gpu --chunk-mode vad \
  --vad-model ggml-silero-v6.2.0.bin -osrt -o audio.srt

# Fixed-window long audio transcription
./funasr-cli -m ../FunAsr_q8.bin -f audio.mp3 --gpu \
  --ctx-size 4096 --chunk-mode window --chunk-sec 30 \
  -osrt -o audio.srt

# Video URL transcription helper
python3 tools/funasr_video_ui.py

# Real-time microphone (CPU)
./test_realtime ../FunAsr_q8.bin

# Real-time microphone (GPU)
./test_realtime ../FunAsr_q8.bin --gpu
```

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

Recommended RTX 4070 Laptop 8GB command used during profiling:

```bash
./build-cuda/test_offline_batching FunAsr_q8.bin \
  outputs/video_asr/20260502_130430/media/source_16k.wav \
  --gpu --kv-mode paged --batch-size 12 --ctx-size 4096 \
  --kv-block-size 128 --chunk-mode window --chunk-sec 30 \
  --max-tokens 220
```

The current fast path combines:

- scheduler batching across independent audio chunks
- paged KV cache for decode requests
- token-id paged batch decode, avoiding host embedding fallback
- dynamic paged KV write so the decode graph does not depend on physical KV rows
- bucketed decode shapes and graph cache for the hottest batch shape

Representative profiling result on RTX 4070 Laptop GPU + i7-13700H:

| Workload | Config | Wall time | Throughput | Notes |
| --- | --- | ---: | ---: | --- |
| ~6119 s audio (~102 min) | paged KV, batch=12, block=128, 30 s chunks | 52-59 s | 100-117 audio-sec/s | End-to-end offline test path |
| Same workload, graph cache snapshot | paged KV, batch=12 | 52.12 s | 117.42 audio-sec/s | Best recorded local run |
| Single-request continuous baseline | batch=1 | 308.63 s | 19.8 audio-sec/s | Before scheduler batching |

These numbers are workload and GPU dependent. They are included to show the
order of magnitude and the benchmark shape, not as a universal guarantee. See
`docs/superpowers/notes/2026-05-27-vllm-style-offline-asr-retrospective.md` and
`docs/superpowers/notes/2026-06-11-offline-decode-kernel-profiling.md` for the
full optimization notes.

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

## Performance

Tested on RTX 4070 Laptop GPU + i7-13700H:

| Mode | Workload | Result |
| ---- | -------- | ------ |
| CPU realtime | Single utterance | Prefill 1052 ms, decode ~12 tok/s, RTF ~0.91 |
| GPU realtime | Single utterance | Prefill 23 ms, decode 55-62 tok/s, RTF ~0.28-0.35 |
| GPU offline baseline | ~6119 s audio, batch=1 | 308.63 s wall, ~19.8 audio-sec/s |
| GPU offline paged batch | ~6119 s audio, batch=12, block=128 | 52-59 s wall, ~100-117 audio-sec/s |

The offline numbers come from the dedicated long-video benchmark path and use
30-second fixed windows. For comparable results, keep the same chunk shape and
report `wall`, `rtf`, `audio_sec/s`, `tokens/s`, `batch_size`, `kv_mode`, and
`kv_block_size`.

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
│   ├── llm_ops_gpu.hpp/.cpp #   LLM decoder (GPU, ggml_cpy)
│   ├── gpu_runner.hpp       #   GPU graph execution (gallocr)
│   └── graph_runner.hpp     #   CPU graph execution helper
├── pipeline/                # User-facing API
│   ├── prompt_builder.hpp   #   ChatML prompt construction
│   ├── pipeline.hpp/.cpp    #   CPU/GPU inference pipeline
│   ├── recognizer.hpp       #   One-line API
│   ├── audio_capture.hpp/.cpp # Microphone capture (miniaudio)
│   └── realtime.hpp         #   Real-time VAD + recognition
├── test/                    # Tests and demos
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
