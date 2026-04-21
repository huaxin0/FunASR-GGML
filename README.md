# FunASR-GGML

**Audio → Text**, fully local, no Python, no server.

## Features

- **Pure C++17** — no Python runtime, no external dependencies (except GGML)
- **GGUF model format** — single file contains weights + tokenizer + config
- **CPU + GPU** — LLM decoder runs on CUDA GPU (Encoder/Adaptor stay on CPU)
- **Real-time microphone** — VAD-based streaming recognition with [miniaudio](https://github.com/mackron/miniaudio)
- **~985M parameters** — Audio Encoder (SANM) + Audio Adaptor + LLM Decoder (Qwen3-0.6B)

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

realtime.start(config, [](int id, const std::string& text, float sec, float ms) {
    printf("[%d] %s (%.1fs, %.0fms)\n", id, text.c_str(), sec, ms);
});

mic.init();
mic.start();
// ... wait for Ctrl+C ...
mic.stop();
realtime.stop();
```

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

Tested on RTX 4070 Laptop GPU + AMD-7745H:

| Mode | Prefill (116 tok) | Decode Speed | RTF (realtime) |
| ---- | ----------------- | ------------ | -------------- |
| CPU  | 1052 ms           | 12.0 tok/s   | 0.91           |
| GPU  | 23 ms             | 55-62 tok/s  | 0.28-0.35      |

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

 MIT. See [LICENSE](https://claude.ai/chat/LICENSE).

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) — Tensor computation library
- [FunASR](https://github.com/modelscope/FunASR) — Original speech recognition model
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Reference for KV Cache patterns
- [miniaudio](https://github.com/mackron/miniaudio) — Audio I/O
