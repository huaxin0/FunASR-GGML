# Offline Video Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version local Gradio Web UI for offline video transcription from a URL or uploaded local media file.

**Architecture:** Keep the C++ ASR path unchanged. Add a Python Web UI wrapper that reuses `tools/funasr_video_ui.py` helpers for downloading, ffmpeg extraction, `funasr-cli`, and side-output generation. Cover wrapper behavior with focused Python unit tests that avoid running real ASR.

**Tech Stack:** Python 3, Gradio, yt-dlp, ffmpeg, existing `funasr-cli`.

---

## File Structure

- Create `tools/funasr_web_ui.py`: Gradio app, source selection, progress wrapper, result preview.
- Create `test/test_funasr_web_ui.py`: Python unit tests for source selection and command/config behavior.
- Modify `README.md`: add a short Web UI section and install command.

## Task 1: Web UI Test Scaffolding

**Files:**
- Create: `test/test_funasr_web_ui.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `tools.funasr_web_ui` and verify:

```python
def test_select_source_requires_url_or_upload():
    result = web.select_source("", None, Path("/tmp/job"))
    assert not result.ok
    assert "Paste a URL or upload" in result.message

def test_select_source_prefers_uploaded_file(tmp_path):
    uploaded = tmp_path / "clip.mp4"
    uploaded.write_bytes(b"fake")
    job_dir = tmp_path / "job"
    result = web.select_source("https://example.com/video", str(uploaded), job_dir)
    assert result.ok
    assert result.source == str(job_dir / "media" / "upload" / "clip.mp4")
    assert Path(result.source).read_bytes() == b"fake"

def test_build_web_config_uses_long_video_defaults(tmp_path):
    cfg = web.build_web_config(
        source="https://example.com/video",
        model=str(tmp_path / "model.bin"),
        funasr_cli=str(tmp_path / "funasr-cli"),
        output_dir=tmp_path / "out",
        use_gpu=True,
        keep_media=True,
    )
    assert cfg.chunk_mode == "window"
    assert cfg.chunk_sec == "30"
    assert cfg.offline_scheduler is True
    assert cfg.offline_kv_mode == "paged"
    assert cfg.offline_kv_block_size == "128"
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python3 -m pytest test/test_funasr_web_ui.py -q
```

Expected: import or attribute failure because `tools/funasr_web_ui.py` does not exist.

## Task 2: Implement Web UI Wrapper Functions

**Files:**
- Create: `tools/funasr_web_ui.py`

- [ ] **Step 1: Add minimal implementation**

Implement:

- `SourceSelection`
- `select_source(url, uploaded_path, job_dir)`
- `build_web_config(...)`
- `transcribe_job(...)`
- `build_app()`
- `main()`

`build_web_config` should create a `funasr_video_ui.Config` equivalent to the
long preset and use `--offline-preset long-video` through the existing CLI
helper path.

- [ ] **Step 2: Verify focused tests pass**

Run:

```bash
python3 -m pytest test/test_funasr_web_ui.py -q
```

Expected: all tests pass.

## Task 3: CLI Command Preset Support

**Files:**
- Modify: `tools/funasr_video_ui.py`
- Modify: `test/test_funasr_web_ui.py`

- [ ] **Step 1: Add a test for long-video preset argument**

Use a fake `run_command` to capture the `funasr-cli` command and assert it
contains:

```text
--offline-preset long-video
```

- [ ] **Step 2: Update helper command construction**

Allow `tools/funasr_video_ui.py` to include an optional `offline_preset` value
in `Config`. When set to `long-video`, append `--offline-preset long-video`
instead of spelling out scheduler internals.

- [ ] **Step 3: Verify tests pass**

Run:

```bash
python3 -m pytest test/test_funasr_web_ui.py -q
```

Expected: all tests pass.

## Task 4: Documentation And Syntax Checks

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README instructions**

Document:

```bash
pip install gradio yt-dlp
python3 tools/funasr_web_ui.py
```

Mention that `ffmpeg` must be in `PATH` and the page runs locally.

- [ ] **Step 2: Run verification**

Run:

```bash
python3 -m py_compile tools/funasr_video_ui.py tools/funasr_web_ui.py
python3 -m pytest test/test_funasr_web_ui.py -q
```

Expected: compile succeeds and tests pass.
