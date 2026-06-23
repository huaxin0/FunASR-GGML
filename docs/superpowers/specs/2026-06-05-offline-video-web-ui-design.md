# Offline Video Web UI Design

Date: 2026-06-05

## Goal

Add a first-version local Web UI for offline video transcription. The page lets
the user paste a Bilibili or other `yt-dlp` supported URL, or upload a local
video/audio file, then runs the existing long-video offline ASR path and returns
SRT, TXT, JSON, and stats outputs.

The first version is for local validation, not a hosted product.

## Non-Goals

- Do not change C++ inference, CUDA kernels, or the offline scheduler.
- Do not add accounts, remote storage, multi-user auth, or a persistent job
  database.
- Do not implement online microphone recognition in this UI.
- Do not parse Bilibili pages directly; URL support is delegated to `yt-dlp`.
- Do not add partial streaming or token-level output.

## Current State

`tools/funasr_video_ui.py` already provides a terminal workflow:

```text
URL or local media path
  -> yt-dlp when the source is a URL
  -> ffmpeg extracts 16 kHz mono WAV
  -> funasr-cli
  -> transcript.srt, transcript.txt, transcript.json, stats.json
```

The CLI now has a productized long-video offline preset:

```text
--gpu
--offline-preset long-video
```

That preset enables window chunking, batch 12, paged KV, block size 128, and
profile/stats defaults.

## Proposed Approach

Create `tools/funasr_web_ui.py` as a small Gradio app. It should reuse the
existing terminal helper's media preparation and side-output helpers where
possible. The Web UI owns only browser-facing concerns:

- collecting a URL or uploaded file
- choosing a model path, CLI path, output directory, and GPU flag
- showing progress messages
- previewing transcript text
- exposing result files for download

## User Flow

1. User opens the local Gradio page.
2. User either pastes a video URL or uploads a local media file.
3. User clicks Transcribe.
4. The backend creates an output directory under `outputs/video_asr_web/`.
5. If a URL was supplied, it downloads audio/video through `yt-dlp`.
6. If a file was uploaded, it uses that uploaded file directly.
7. `ffmpeg` extracts `source_16k.wav`.
8. `funasr-cli --offline-preset long-video -osrt` creates `transcript.srt`.
9. Existing side-output logic writes TXT, JSON, and stats.
10. The page displays the transcript preview and downloadable paths.

If both URL and upload are supplied, the uploaded file wins. If neither is
supplied, the UI returns a validation error without starting work.

## Command Defaults

The first version should keep options minimal:

```text
model path:      ./FunAsr_q8.bin
funasr-cli path: ./build-cuda/funasr-cli, then build-cpu/build fallbacks
output root:     ./outputs/video_asr_web
GPU:             enabled by default
preset:          long-video, always
SRT chars:       28
keep media:      enabled by default
```

The app can expose model path, CLI path, output root, GPU, and keep-media as
advanced controls. Offline scheduler internals stay hidden in this first UI.

## Error Handling

Validation should catch missing model, missing `funasr-cli`, missing `ffmpeg`,
and missing `yt-dlp` for URL jobs. Runtime errors from subprocesses should be
shown in the progress text instead of causing a silent page failure.

Uploaded files should be copied into the job media directory before processing
so the output directory is self-contained.

## Testing

Use Python unit tests for the Web UI wrapper logic without invoking real ASR:

- URL jobs create a config that uses the URL source.
- Upload jobs copy the uploaded file into the job media directory and prefer it
  over a URL.
- Missing input returns a clear validation error.
- The long-video command path uses `--offline-preset long-video`.

Manual validation for the full pipeline remains:

```bash
python3 tools/funasr_web_ui.py
```

Then paste a Bilibili URL or upload a local media file and verify the generated
SRT/TXT/JSON/stats files.
