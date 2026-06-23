# Gitee Release Scope

This repository is published as a clean source snapshot for the FunASR-GGML
engine and local video workflow.

## Included

- C++ inference engine: `core/`, `model/`, `compute/`, `pipeline/`
- CLI and SDK entry points: `cli/`, `sdk/`, `include/`, `examples/`
- Local video transcription Web app: `tools/funasr_web_app.py`,
  `tools/funasr_video_ui.py`, and supporting utility scripts
- Bilibili browser sidebar extension: `browser-extension/bilibili-funasr-sidebar/`
- Source tests under `test/`
- Essential documentation: `README.md`, `docs/offline_continuous_batching.md`,
  `docs/sdk_hotwords_usage.md`, and this release scope note
- Vendored GGML source under `third_party/ggml/`

## Excluded

- Model binaries such as `FunAsr.bin`, `FunAsr_q8.bin`, and VAD `.bin` files
- Build directories such as `build/`, `build-cuda/`, and CMake generated files
- Generated test executables and local test media/results
- Transcription outputs under `outputs/`
- Local cookies, API keys, cache files, and CodeGraph state
- Local audio/video/subtitle samples

## Runtime Notes

Download or place model files outside git, then pass the model path to
`funasr-cli`, the SDK, or the local Web app. For Bilibili downloads that require
login, provide a local `cookies.txt` at runtime; do not commit it.
