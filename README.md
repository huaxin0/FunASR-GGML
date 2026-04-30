# FunASR-GGML

High-accuracy Chinese offline ASR in pure C++: Fun-ASR-Nano on GGML, no Python, no server.

**7.9× faster than official PyTorch** | **CER 1.98% on AISHELL-1** | **Single binary + single model file**

## Features

- **Pure C++17** — no Python runtime, no framework dependencies
- **GGUF model format** — single file contains weights + tokenizer + config
- **Full GPU pipeline** — Encoder, Adaptor, and LLM all run on CUDA GPU
- **CPU fallback** — runs entirely on CPU when CUDA is unavailable
- **CLI tool** — `funasr-cli` for file transcription, batch processing, SRT subtitles
- **Long audio** — Silero VAD segmentation + punctuation-based subtitle splitting
- **Multiple formats** — WAV, MP3, FLAC input; TXT, SRT, TSV output
- **Real-time microphone** — VAD-based streaming recognition with [miniaudio](https://github.com/mackron/miniaudio)
- **~985M parameters** — Audio Encoder (SANM 70L) + Audio Adaptor (2L) + LLM Decoder (Qwen3-0.6B 28L)

## Quick Start

```bash
# 1. Build
git clone --recursive https://github.com/huaxin0/FunASR-GGML.git
cd FunASR-GGML && mkdir build && cd build
cmake .. -DFUNASR_CUDA=ON    # or just `cmake ..` for CPU-only
make -j$(nproc)

# 2. Download models
wget https://huggingface.co/huaxin0x/funasr-ggml-models/resolve/main/FunAsr_q8.bin
wget https://huggingface.co/huaxin0x/funasr-ggml-models/resolve/main/ggml-silero-v6.2.0.bin

# 3. Transcribe
./funasr-cli -m FunAsr_q8.bin -f audio.wav --gpu
```

## CLI Usage

```bash
cmake .. -DFUNASR_CUDA=ON
make -j$(nproc)
```

### Build options

| Option                       | Default | Description                                                       |
| ---------------------------- | ------- | ----------------------------------------------------------------- |
| `FUNASR_CUDA`                | `OFF`   | Enable CUDA GPU support (requires CUDA Toolkit).                  |
| `FUNASR_BUILD_AUDIO_CAPTURE` | `ON`    | Build microphone capture module (depends on `miniaudio`).         |
| `FUNASR_BUILD_REALTIME`      | `ON`    | Build the real-time recognizer (built-in VAD + streaming).        |

Turn off the last two to get a minimal build with no microphone input and no VAD (file transcription only):

```bash
cmake .. -DFUNASR_BUILD_AUDIO_CAPTURE=OFF -DFUNASR_BUILD_REALTIME=OFF
make -j$(nproc)
```

Note: `test_realtime` is only built when both `FUNASR_BUILD_AUDIO_CAPTURE` and `FUNASR_BUILD_REALTIME` are `ON`.

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
recognizer.init_gpu();

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

## Model

| File | Size | Description |
|---|---|---|
| `FunAsr_q8.bin` | 1.2 GB | ASR model (Q8_0 quantized, 1541 tensors) |
| `ggml-silero-v6.2.0.bin` | 860 KB | Silero VAD model (optional, for long audio) |

Download from [HuggingFace](https://huggingface.co/huaxin0x/funasr-ggml-models):

```bash
wget https://huggingface.co/huaxin0x/funasr-ggml-models/resolve/main/FunAsr_q8.bin
wget https://huggingface.co/huaxin0x/funasr-ggml-models/resolve/main/ggml-silero-v6.2.0.bin
```

### Convert from HuggingFace (optional)

```bash
python hf_convert_ggml_q8.py --model-dir <huggingface_model> --output FunAsr_q8.bin
```

## GPU Pipeline Details

- **KV Cache clear**: `ggml_backend_buffer_clear` on GPU, no CPU-side allocation
- **Encoder → Adaptor**: GPU-to-GPU staging buffer, no PCIe round-trip
- **Adaptor → LLM**: GPU-resident prefill staging with `ggml_cpy` graph
- **Buffer reuse**: Staging buffers allocated at warmup, reused across inferences
- **GPU dither**: Automatic low-amplitude dither prevents GPU encoder NaN on near-silent audio
- **Separate compute graphs**: Encoder and Adaptor use independent `ggml_gallocr` (merged graph is slower)

## Project Structure

```
FunASR-GGML/
├── cli/                     # Command-line tool
│   └── funasr_cli.cpp       #   funasr-cli: transcribe, SRT, batch
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
│   ├── silero_vad.hpp/.cpp  #   Silero VAD (ggml, from whisper.cpp)
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
- miniaudio.h (included, for audio decoding and microphone)

## License

MIT

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) — Tensor computation library
- [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) / [FunASR](https://github.com/modelscope/FunASR) — Original speech recognition model and toolkit
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — Silero VAD ggml implementation and GGML-based ASR patterns
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection model (MIT)
- [zjkhahah/tokenizer-json](https://github.com/zjkhahah/tokenizer-json) — HuggingFace tokenizer JSON reader for C++
- [miniaudio](https://github.com/mackron/miniaudio) — Cross-platform audio I/O
