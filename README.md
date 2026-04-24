# FunASR-GGML

C++ speech recognition inference engine using [GGML](https://github.com/ggerganov/ggml), powered by [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) (0.8B).

**Audio → Text**, fully local, no Python, no server.

## Features

- **Pure C++17** — no Python runtime, no framework dependencies
- **GGUF model format** — single file contains weights + tokenizer + config
- **Full GPU pipeline** — Encoder, Adaptor, and LLM all run on CUDA GPU with GPU-resident data path
- **CPU fallback** — runs entirely on CPU when CUDA is unavailable
- **Real-time microphone** — VAD-based streaming recognition with [miniaudio](https://github.com/mackron/miniaudio)
- **~985M parameters** — Audio Encoder (SANM 70L) + Audio Adaptor (2L) + LLM Decoder (Qwen3-0.6B 28L)

## Architecture

```
WAV / Mic (16kHz mono)
  → Fbank (80-dim mel + LFR) → [560, T]                              CPU
  → Audio Encoder (50 SANM + 20 TP layers, FSMN) → [512, T]          GPU ★
  → Audio Adaptor (linear + 2 attention blocks) → [1024, T]          GPU ★
  → LLM Decoder (28-layer Qwen3, GQA, RoPE, KV Cache) → logits      GPU ★
  → BPE Decode → Text

★ GPU-resident path: data stays on GPU between Encoder → Adaptor → LLM.
  Only fbank input (CPU→GPU) and logits output (GPU→CPU) cross PCIe.
```

## Benchmark

### Accuracy — AISHELL-1 Test Set (7176 utterances)

| Implementation | CER (%) | Quantization |
|---|---|---|
| Fun-ASR-nano (official paper) | 1.80 | FP16 |
| Fun-ASR-nano (official Python, this machine) | 1.91 | FP16 |
| **FunASR-GGML (this project)** | **1.98** | **Q8_0** |

CER evaluated with digit normalization (阿拉伯数字→中文数字) and punctuation removal.

### Speed — RTX 4070 Laptop GPU + AMD Ryzen 7 7745HX

| Metric | Official Python (PyTorch) | C++ GGML Q8 | Speedup |
|---|---|---|---|
| Avg latency | 1753 ms | 221 ms | **7.9×** |
| Throughput | 0.6 files/sec | 4.5 files/sec | **7.5×** |
| RTF | 0.348 | 0.044 | **7.9×** |
| Total time (7176 files) | 12594 sec (3.5h) | 1586 sec (26min) | **7.9×** |

Both tested on the same machine, same GPU, same test set, no VAD.

Staging buffers allocated once at warmup; no per-inference GPU memory allocation.

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
# File transcription (CPU)
./test_pipeline ../FunAsr_q8.bin audio.wav

# File transcription (GPU)
./test_gpu ../FunAsr_q8.bin audio.wav

# Batch benchmark
./test_benchmark ../FunAsr_q8.bin ~/data_aishell/wav/test results.txt --gpu

# Real-time microphone
./test_realtime ../FunAsr_q8.bin --gpu
```

## C++ API

### Minimal example

```cpp
#include "pipeline/recognizer.hpp"

int main() {
    funasr::Recognizer recognizer;
    recognizer.init("FunAsr_q8.bin");

    auto result = recognizer.transcribe("audio.wav");
    printf("%s\n", result.text.c_str());
    return 0;
}
```

### GPU inference

```cpp
funasr::Recognizer recognizer;
recognizer.init("FunAsr_q8.bin");
recognizer.init_gpu();  // Uploads weights + warmup

funasr::InferenceConfig config;
config.use_gpu = true;

auto result = recognizer.transcribe("audio.wav", config);
printf("%.0f ms | %s\n", result.total_ms, result.text.c_str());
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

funasr::RealtimeConfig rt_config;
rt_config.inference.use_gpu = true;

realtime.start(rt_config, [](int id, const std::string& text, float sec, float ms) {
    printf("[%d] %s (%.1fs, %.0fms)\n", id, text.c_str(), sec, ms);
});

mic.init();
mic.start();
// ... Ctrl+C to stop ...
mic.stop();
realtime.stop();
```

## Model

The model file `FunAsr_q8.bin` is in GGUF format containing:

- 1541 tensors (~985M parameters, Q8_0 quantized)
- BPE tokenizer (151936 tokens)
- Architecture config (all hyperparameters)

### Convert from HuggingFace

```bash
python hf_convert_ggml_q8.py --model-dir <huggingface_model> --output FunAsr_q8.bin
```

## GPU Pipeline Details

The GPU inference path is optimized to minimize PCIe transfers:

- **KV Cache clear**: `ggml_backend_buffer_clear` on GPU, no CPU-side allocation
- **Encoder → Adaptor**: GPU-to-GPU staging buffer, no round-trip
- **Adaptor → LLM**: GPU-resident prefill staging with `ggml_cpy` graph for device-to-device copy
- **Buffer reuse**: Staging buffers allocated at warmup (max size), reused across inferences via partial `tensor_set`/`tensor_copy`
- **Separate compute graphs**: Encoder and Adaptor use independent `ggml_gallocr` instances (merged graph is slower due to gallocr planning overhead on 70+ layer graphs)

## Project Structure

```
FunASR-GGML/
├── core/                    # Config, GGUF reader
│   ├── config.hpp           #   All params from GGUF metadata
│   └── gguf_reader.hpp      #   RAII GGUF lifecycle
├── model/                   # Model loading
│   ├── weights.hpp          #   Weight structs (1541 tensors)
│   ├── model.hpp            #   Aggregate model struct
│   ├── loader.hpp/.cpp      #   Tensor binding from GGUF
│   └── tokenizer.hpp/.cpp   #   BPE tokenizer (encode/decode)
├── compute/                 # Forward computation
│   ├── fbank.hpp/.cpp       #   Mel filterbank + LFR
│   ├── encoder_ops.hpp/.cpp #   70-layer SANM + FSMN encoder
│   ├── adaptor_ops.hpp/.cpp #   Linear + 2-block MHA adaptor
│   ├── llm_ops.hpp/.cpp     #   28-layer Qwen3 (CPU path)
│   ├── kv_cache.hpp         #   CPU KV cache
│   ├── gpu_context.hpp      #   CUDA backend + GPU weights + GPU KV cache
│   ├── llm_ops_gpu.hpp/.cpp #   28-layer Qwen3 (GPU path)
│   ├── gpu_runner.hpp       #   GPU graph executor (gallocr)
│   ├── encoder_adaptor_gpu.hpp  # GPU encoder/adaptor with staging
│   └── graph_runner.hpp     #   CPU graph executor
├── pipeline/                # User-facing API
│   ├── prompt_builder.hpp   #   ChatML prompt construction
│   ├── pipeline.hpp/.cpp    #   CPU/GPU inference pipeline
│   ├── recognizer.hpp       #   One-line recognizer API
│   ├── audio_capture.hpp/.cpp # Microphone (miniaudio)
│   └── realtime.hpp         #   VAD + streaming recognition
├── test/                    # Tests and benchmarks
│   ├── test_pipeline.cpp    #   Basic inference test
│   ├── test_gpu.cpp         #   GPU inference test
│   ├── test_benchmark.cpp   #   AISHELL-1 batch evaluation
│   └── test_realtime.cpp    #   Real-time microphone demo
├── tools/
│   ├── hf_convert_ggml_q8.py    # HuggingFace → GGUF converter
│   └── eval_cer_v2.py           # CER evaluation script
└── third_party/
    ├── ggml/                #   GGML (submodule)
    └── miniaudio.h          #   Audio I/O (header-only)
```

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- GGML (included as submodule)
- CUDA Toolkit 11+ (optional, for GPU)
- miniaudio.h (included, for microphone)

## License

MIT

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) — Tensor computation library
- [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) / [FunASR](https://github.com/modelscope/FunASR) — Original speech recognition model and toolkit
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Reference for GGML-based ASR implementation patterns
- [zjkhahah/tokenizer-json](https://github.com/zjkhahah/tokenizer-json) — HuggingFace tokenizer JSON reader for C++
- [miniaudio](https://github.com/mackron/miniaudio) — Cross-platform audio I/O
