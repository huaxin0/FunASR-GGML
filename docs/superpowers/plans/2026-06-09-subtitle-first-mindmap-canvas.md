# Subtitle-First Mindmap Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subtitle-first loading for URL jobs and replace the extension mindmap with a draggable, zoomable canvas while preserving existing transcript and mindmap data contracts.

**Architecture:** The backend gets a best-effort subtitle import path before the existing ASR flow. It writes the same `transcript.srt`, `transcript.txt`, `transcript.json`, and `stats.json` outputs, so the extension and library lookup keep working. The frontend keeps `renderMindmap()` as the public entry point but delegates to a new canvas renderer with local pan/zoom/collapse state.

**Tech Stack:** Python 3, FastAPI, `yt-dlp`, existing `tools/funasr_web_app.py` and `tools/funasr_video_ui.py`, vanilla content-script JavaScript, CSS, `unittest`.

---

## File Structure

- Modify `tools/funasr_web_app.py`
  - Add subtitle metadata dataclass.
  - Add subtitle download command builder.
  - Add VTT parsing, SRT serialization, and shared transcript output writer.
  - Add `try_import_subtitles()` and call it before media download in `run_job()`.
  - Add `transcript_source` metadata to ASR stats.
- Modify `tools/funasr_video_ui.py`
  - Keep current SRT parser.
  - Add optional metadata parameters to `write_side_outputs()`.
- Modify `browser-extension/bilibili-funasr-sidebar/content.js`
  - Add better status messages for subtitle phases.
  - Replace nested mindmap rendering with canvas renderer while preserving `renderMindmap()`.
- Modify `browser-extension/bilibili-funasr-sidebar/sidebar.css`
  - Add canvas, toolbar, node, connector, and transform styles.
- Modify `test/test_funasr_web_app.py`
  - Add subtitle command, parsing, transcript output, import success, and fallback tests.
- Modify `test/test_bilibili_extension.py`
  - Add tests for canvas UI markers, pan/zoom handlers, and `jumpTo` integration.

## Task 1: Shared Transcript Output Metadata

**Files:**
- Modify: `tools/funasr_video_ui.py`
- Test: `test/test_funasr_web_app.py`

- [ ] **Step 1: Write failing stats metadata test**

Add this test near the existing web app transcript/library tests in `test/test_funasr_web_app.py`:

```python
    def test_write_transcript_outputs_records_transcript_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web_app.build_job_config(
                source="https://www.bilibili.com/video/BV1abc/",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )
            cfg.output_dir.mkdir()
            segments = [
                {"index": 1, "start": 1.0, "end": 3.5, "text": "这里介绍 attention。"}
            ]

            web_app.write_transcript_outputs_from_segments(
                cfg,
                segments,
                elapsed_sec=0.25,
                source_info={
                    "transcript_source": "subtitle",
                    "subtitle_language": "zh-Hans",
                    "subtitle_kind": "manual",
                },
            )

            stats = json.loads((cfg.output_dir / "stats.json").read_text(encoding="utf-8"))
            transcript = json.loads((cfg.output_dir / "transcript.json").read_text(encoding="utf-8"))

        self.assertEqual(stats["transcript_source"], "subtitle")
        self.assertEqual(stats["subtitle_language"], "zh-Hans")
        self.assertEqual(stats["subtitle_kind"], "manual")
        self.assertEqual(transcript["segments"][0]["text"], "这里介绍 attention。")
        self.assertIn("attention", (cfg.output_dir / "transcript.txt").read_text(encoding="utf-8"))
        self.assertIn("-->", (cfg.output_dir / "transcript.srt").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest test.test_funasr_web_app.FunasrWebAppTests.test_write_transcript_outputs_records_transcript_source
```

Expected: FAIL with `AttributeError: module 'tools.funasr_web_app' has no attribute 'write_transcript_outputs_from_segments'`.

- [ ] **Step 3: Add transcript output helper**

In `tools/funasr_web_app.py`, add this helper after `build_asr_command()`:

```python
def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    millis = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"


def segments_to_srt(segments: list[dict[str, object]]) -> str:
    blocks = []
    for index, raw in enumerate(segments, start=1):
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or start)
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{format_srt_time(start)} --> {format_srt_time(end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def normalize_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for raw in segments:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or start)
        normalized.append(
            {
                "index": len(normalized) + 1,
                "start": start,
                "end": max(start, end),
                "text": text,
            }
        )
    return normalized


def write_transcript_outputs_from_segments(
    cfg: video_ui.Config,
    segments: list[dict[str, object]],
    elapsed_sec: float,
    source_info: dict[str, object],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_segments(segments)
    (cfg.output_dir / "transcript.srt").write_text(segments_to_srt(normalized), encoding="utf-8")
    text = "\n".join(str(item["text"]) for item in normalized)
    (cfg.output_dir / "transcript.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    (cfg.output_dir / "transcript.json").write_text(
        json.dumps({"source": cfg.source, "segments": normalized}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    duration = max((float(item["end"]) for item in normalized), default=0.0)
    stats = {
        "source": cfg.source,
        "elapsed_sec": round(elapsed_sec, 3),
        "audio_duration_sec": round(duration, 3),
        "rtf": round(elapsed_sec / duration, 4) if duration > 0 else None,
        "segments": len(normalized),
        "gpu": cfg.use_gpu,
        "chunk_mode": cfg.chunk_mode,
        "chunk_sec": cfg.chunk_sec if cfg.chunk_mode == "window" else None,
        "vad": cfg.use_vad,
        "vad_model": str(cfg.vad_model) if cfg.vad_model else None,
        "offline_scheduler": cfg.offline_scheduler,
        "offline_profile": cfg.offline_profile,
        "offline_batch_size": int(cfg.offline_batch_size) if cfg.offline_scheduler else None,
        "offline_kv_mode": cfg.offline_kv_mode if cfg.offline_scheduler else None,
        "offline_kv_block_size": int(cfg.offline_kv_block_size) if cfg.offline_scheduler else None,
        "transcript_source": source_info.get("transcript_source", "asr"),
        "subtitle_language": source_info.get("subtitle_language"),
        "subtitle_kind": source_info.get("subtitle_kind"),
    }
    (cfg.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Update ASR side output metadata**

In `tools/funasr_video_ui.py`, modify `write_side_outputs()` signature and stats block:

```python
def write_side_outputs(
    cfg: Config,
    srt_path: Path,
    elapsed_sec: float,
    source_info: dict[str, object] | None = None,
) -> None:
    source_info = source_info or {"transcript_source": "asr"}
```

Add these fields to the stats dictionary:

```python
                "transcript_source": source_info.get("transcript_source", "asr"),
                "subtitle_language": source_info.get("subtitle_language"),
                "subtitle_kind": source_info.get("subtitle_kind"),
```

No existing callers need to change because the parameter is optional.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest test.test_funasr_web_app.FunasrWebAppTests.test_write_transcript_outputs_records_transcript_source
python3 -m unittest test.test_funasr_web_ui test.test_funasr_web_app
```

Expected: PASS.

- [ ] **Step 6: Commit if git is available**

Run:

```bash
git status --short
git add tools/funasr_web_app.py tools/funasr_video_ui.py test/test_funasr_web_app.py
git commit -m "feat: add transcript source metadata"
```

Expected in this workspace: git may report `not a git repository`; if so, record that and continue without committing.

## Task 2: Subtitle Download and Parsing

**Files:**
- Modify: `tools/funasr_web_app.py`
- Test: `test/test_funasr_web_app.py`

- [ ] **Step 1: Write failing subtitle command and parser tests**

Add these tests to `test/test_funasr_web_app.py`:

```python
    def test_subtitle_download_command_prefers_subtitle_only_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = web_app.build_subtitle_download_command(
                "https://www.bilibili.com/video/BV1abc/",
                Path(tmp),
                cookies_from_browser="chrome",
                cookies_path="",
            )

        self.assertIn("--skip-download", cmd)
        self.assertIn("--write-subs", cmd)
        self.assertIn("--write-auto-subs", cmd)
        self.assertIn("--sub-format", cmd)
        self.assertIn("vtt/srt/best", cmd)
        self.assertIn("--cookies-from-browser", cmd)

    def test_parse_vtt_extracts_segments_and_removes_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.zh-Hans.vtt"
            path.write_text(
                "WEBVTT\n\n"
                "00:00:01.000 --> 00:00:03.500\n"
                "<c>这里介绍 attention。</c>\n\n"
                "00:00:04.000 --> 00:00:05.000\n"
                "第二句\n",
                encoding="utf-8",
            )

            segments = web_app.parse_vtt(path)

        self.assertEqual(segments[0]["index"], 1)
        self.assertEqual(segments[0]["start"], 1.0)
        self.assertEqual(segments[0]["end"], 3.5)
        self.assertEqual(segments[0]["text"], "这里介绍 attention。")
        self.assertEqual(segments[1]["text"], "第二句")

    def test_find_downloaded_subtitle_prefers_manual_chinese(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
            (root / "source.zh-Hans.vtt").write_text("WEBVTT\n", encoding="utf-8")
            (root / "source.zh-Hans.auto.vtt").write_text("WEBVTT\n", encoding="utf-8")

            candidate = web_app.find_downloaded_subtitle(root)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.language, "zh-Hans")
        self.assertEqual(candidate.kind, "manual")
        self.assertTrue(candidate.path.name.endswith("source.zh-Hans.vtt"))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest \
  test.test_funasr_web_app.FunasrWebAppTests.test_subtitle_download_command_prefers_subtitle_only_download \
  test.test_funasr_web_app.FunasrWebAppTests.test_parse_vtt_extracts_segments_and_removes_tags \
  test.test_funasr_web_app.FunasrWebAppTests.test_find_downloaded_subtitle_prefers_manual_chinese
```

Expected: FAIL with missing helper attributes.

- [ ] **Step 3: Add subtitle helpers**

In `tools/funasr_web_app.py`, add imports:

```python
import html
```

Add these definitions near `build_download_command()`:

```python
CHINESE_SUBTITLE_LANGS = ("zh-Hans", "zh-CN", "zh", "zh-Hant", "zh-TW")


@dataclass(frozen=True)
class SubtitleCandidate:
    path: Path
    language: str
    kind: str


def build_subtitle_download_command(
    source: str,
    subtitle_dir: Path,
    cookies_from_browser: str = "",
    cookies_path: str = "",
) -> list[str]:
    template = str(subtitle_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-format",
        "vtt/srt/best",
        "--no-playlist",
        "-o",
        template,
    ]
    cookies_path = cookies_path.strip()
    cookies_from_browser = cookies_from_browser.strip()
    if cookies_path:
        cmd.extend(["--cookies", str(Path(cookies_path).expanduser())])
    elif cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.append(source)
    return cmd


def _subtitle_language_from_name(path: Path) -> str:
    name = path.name
    for lang in CHINESE_SUBTITLE_LANGS:
        if f".{lang}." in name:
            return lang
    parts = name.split(".")
    if len(parts) >= 3:
        return parts[-2] if parts[-2] != "auto" else parts[-3]
    return ""


def _subtitle_kind_from_name(path: Path) -> str:
    return "auto" if ".auto." in path.name or path.name.endswith(".auto.vtt") else "manual"


def find_downloaded_subtitle(subtitle_dir: Path) -> SubtitleCandidate | None:
    candidates = []
    for path in sorted(subtitle_dir.glob("*")):
        if path.suffix.lower() not in {".vtt", ".srt"} or not path.is_file():
            continue
        language = _subtitle_language_from_name(path)
        kind = _subtitle_kind_from_name(path)
        is_chinese = language in CHINESE_SUBTITLE_LANGS
        kind_rank = 0 if kind == "manual" else 1
        lang_rank = CHINESE_SUBTITLE_LANGS.index(language) if is_chinese else len(CHINESE_SUBTITLE_LANGS)
        group_rank = 0 if is_chinese and kind == "manual" else 1 if is_chinese else 2 if kind == "manual" else 3
        candidates.append((group_rank, lang_rank, kind_rank, path.name, SubtitleCandidate(path, language, kind)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:4])
    return candidates[0][4]


def parse_subtitle_time(value: str) -> float:
    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return 0.0
    return 0.0


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        start_raw, end_raw = [part.strip().split()[0] for part in lines[timing_index].split("-->", 1)]
        text = clean_subtitle_text(" ".join(lines[timing_index + 1:]))
        if text:
            segments.append(
                {
                    "index": len(segments) + 1,
                    "start": parse_subtitle_time(start_raw),
                    "end": parse_subtitle_time(end_raw),
                    "text": text,
                }
            )
    return segments


def parse_subtitle_file(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".srt":
        return video_ui.parse_srt(path)
    if path.suffix.lower() == ".vtt":
        return parse_vtt(path)
    return []
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
python3 -m unittest \
  test.test_funasr_web_app.FunasrWebAppTests.test_subtitle_download_command_prefers_subtitle_only_download \
  test.test_funasr_web_app.FunasrWebAppTests.test_parse_vtt_extracts_segments_and_removes_tags \
  test.test_funasr_web_app.FunasrWebAppTests.test_find_downloaded_subtitle_prefers_manual_chinese
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```bash
git add tools/funasr_web_app.py test/test_funasr_web_app.py
git commit -m "feat: add subtitle download parsing helpers"
```

Expected in this workspace: git may not be available; continue without committing if it fails.

## Task 3: Subtitle-First Job Flow

**Files:**
- Modify: `tools/funasr_web_app.py`
- Test: `test/test_funasr_web_app.py`

- [ ] **Step 1: Write failing subtitle import tests**

Add these tests to `test/test_funasr_web_app.py`:

```python
    def test_try_import_subtitles_writes_outputs_when_subtitle_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web_app.build_job_config(
                source="https://www.bilibili.com/video/BV1abc/",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )
            job = web_app.Job(id="job123", output_dir=cfg.output_dir)
            calls = []

            def fake_run(cmd, job_arg):
                calls.append(cmd)
                subtitle_dir = root / "out" / "media" / "subtitles"
                subtitle_dir.mkdir(parents=True, exist_ok=True)
                (subtitle_dir / "source.zh-Hans.vtt").write_text(
                    "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n字幕内容\n",
                    encoding="utf-8",
                )

            imported = web_app.try_import_subtitles(
                cfg,
                cfg.output_dir / "media",
                job,
                cookies_from_browser="",
                cookies_path="",
                command_runner=fake_run,
            )

            stats = json.loads((cfg.output_dir / "stats.json").read_text(encoding="utf-8"))

        self.assertTrue(imported)
        self.assertEqual(job.status, "importing-subtitles")
        self.assertEqual(stats["transcript_source"], "subtitle")
        self.assertEqual(stats["subtitle_language"], "zh-Hans")
        self.assertEqual(stats["subtitle_kind"], "manual")
        self.assertIn("字幕内容", (cfg.output_dir / "transcript.txt").read_text(encoding="utf-8"))
        self.assertIn("--skip-download", calls[0])

    def test_try_import_subtitles_returns_false_when_no_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web_app.build_job_config(
                source="https://www.bilibili.com/video/BV1abc/",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )
            job = web_app.Job(id="job123", output_dir=cfg.output_dir)

            def fake_run(cmd, job_arg):
                (root / "out" / "media" / "subtitles").mkdir(parents=True, exist_ok=True)

            imported = web_app.try_import_subtitles(
                cfg,
                cfg.output_dir / "media",
                job,
                cookies_from_browser="",
                cookies_path="",
                command_runner=fake_run,
            )

        self.assertFalse(imported)
        self.assertFalse((cfg.output_dir / "transcript.json").exists())
        self.assertIn("No usable subtitle", "\n".join(job.log))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest \
  test.test_funasr_web_app.FunasrWebAppTests.test_try_import_subtitles_writes_outputs_when_subtitle_exists \
  test.test_funasr_web_app.FunasrWebAppTests.test_try_import_subtitles_returns_false_when_no_segments
```

Expected: FAIL with missing `try_import_subtitles`.

- [ ] **Step 3: Implement `try_import_subtitles()`**

Add this helper after `download_or_use_source()` in `tools/funasr_web_app.py`:

```python
def try_import_subtitles(
    cfg: video_ui.Config,
    work_dir: Path,
    job: Job,
    cookies_from_browser: str,
    cookies_path: str,
    command_runner=run_command_stream,
) -> bool:
    if not video_ui.is_url(cfg.source):
        return False

    job.status = "checking-subtitles"
    job.append("Checking video subtitles before ASR...")
    subtitle_dir = work_dir / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    try:
        command_runner(
            build_subtitle_download_command(
                cfg.source,
                subtitle_dir,
                cookies_from_browser=cookies_from_browser,
                cookies_path=cookies_path,
            ),
            job,
        )
    except Exception as exc:
        job.append(f"Subtitle check failed; falling back to ASR: {exc}")
        return False

    candidate = find_downloaded_subtitle(subtitle_dir)
    if candidate is None:
        job.append("No usable subtitle file found; falling back to ASR.")
        return False

    segments = parse_subtitle_file(candidate.path)
    if not normalize_segments(segments):
        job.append("No usable subtitle segments found; falling back to ASR.")
        return False

    job.status = "importing-subtitles"
    job.append(
        f"Using video subtitle track: language={candidate.language or '-'} kind={candidate.kind}"
    )
    write_transcript_outputs_from_segments(
        cfg,
        segments,
        elapsed_sec=time.monotonic() - job.started_at,
        source_info={
            "transcript_source": "subtitle",
            "subtitle_language": candidate.language or None,
            "subtitle_kind": candidate.kind,
        },
    )
    return True
```

- [ ] **Step 4: Wire subtitle import into `run_job()`**

In `run_job()`, after `video_ui.write_run_config(cfg)` and before `reusable = reusable_media_for_source(...)`, insert:

```python
        if try_import_subtitles(cfg, work_dir, job, cookies_from_browser, cookies_path):
            job.status = "writing-results"
            txt_path = cfg.output_dir / "transcript.txt"
            job.transcript = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
            if not cfg.keep_media:
                shutil.rmtree(work_dir, ignore_errors=True)
            job.status = "done"
            record_library_item(DEFAULT_LIBRARY_INDEX, source=cfg.source, job=job)
            job.append(f"Done using video subtitles. Output: {cfg.output_dir}")
            return
```

In the ASR path, change:

```python
        video_ui.write_side_outputs(cfg, srt_path, time.monotonic() - started)
```

to:

```python
        video_ui.write_side_outputs(
            cfg,
            srt_path,
            time.monotonic() - started,
            {"transcript_source": "asr", "subtitle_language": None, "subtitle_kind": None},
        )
```

- [ ] **Step 5: Run web app tests**

Run:

```bash
python3 -m unittest test.test_funasr_web_app
```

Expected: PASS.

- [ ] **Step 6: Commit if git is available**

Run:

```bash
git add tools/funasr_web_app.py tools/funasr_video_ui.py test/test_funasr_web_app.py
git commit -m "feat: use subtitles before local asr"
```

Expected in this workspace: git may not be available; continue without committing if it fails.

## Task 4: Extension Subtitle Status Copy

**Files:**
- Modify: `browser-extension/bilibili-funasr-sidebar/content.js`
- Test: `test/test_bilibili_extension.py`

- [ ] **Step 1: Write failing status copy test**

Add this test to `test/test_bilibili_extension.py`:

```python
    def test_content_script_explains_subtitle_first_statuses(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("checking-subtitles", content)
        self.assertIn("正在检查视频自带字幕", content)
        self.assertIn("importing-subtitles", content)
        self.assertIn("正在导入视频自带字幕", content)
        self.assertIn("已使用视频自带字幕快速加载", content)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m unittest test.test_bilibili_extension.BilibiliExtensionTests.test_content_script_explains_subtitle_first_statuses
```

Expected: FAIL because the copy is not in `content.js`.

- [ ] **Step 3: Add status message helper**

In `content.js`, after `setLog()` add:

```javascript
  function messageForJobState(state) {
    if (state.status === "checking-subtitles") {
      return "正在检查视频自带字幕...";
    }
    if (state.status === "importing-subtitles") {
      return "正在导入视频自带字幕...";
    }
    if (state.status === "downloading" && String(state.log || "").includes("No usable subtitle")) {
      return "未找到可用字幕，正在使用本地 ASR。";
    }
    return "";
  }
```

In `pollJob(jobId)`, after `setLog(state.log || "");` add:

```javascript
    const phaseMessage = messageForJobState(state);
    if (phaseMessage) {
      setMessage(phaseMessage);
    }
```

In the `done` branch, before `return;`, add:

```javascript
      if (String(state.log || "").includes("Done using video subtitles")) {
        setMessage("已使用视频自带字幕快速加载。");
      }
```

- [ ] **Step 4: Run extension tests**

Run:

```bash
python3 -m unittest test.test_bilibili_extension
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```bash
git add browser-extension/bilibili-funasr-sidebar/content.js test/test_bilibili_extension.py
git commit -m "feat: show subtitle-first job statuses"
```

Expected in this workspace: git may not be available; continue without committing if it fails.

## Task 5: Mindmap Canvas Markup and State

**Files:**
- Modify: `browser-extension/bilibili-funasr-sidebar/content.js`
- Modify: `browser-extension/bilibili-funasr-sidebar/sidebar.css`
- Test: `test/test_bilibili_extension.py`

- [ ] **Step 1: Write failing canvas marker tests**

Add these tests to `test/test_bilibili_extension.py`:

```python
    def test_mindmap_uses_canvas_viewport_and_toolbar(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn("renderMindmapCanvas", content)
        self.assertIn("funasr-mindmap-toolbar", content)
        self.assertIn("funasr-mindmap-viewport", content)
        self.assertIn("funasr-mindmap-transform", content)
        self.assertIn("funasr-mindmap-svg", content)
        self.assertIn("funasr-mindmap-canvas-node", css)

    def test_mindmap_canvas_supports_pan_zoom_and_jump(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("mindmapViewState", content)
        self.assertIn("setMindmapTransform", content)
        self.assertIn("centerMindmapCanvas", content)
        self.assertIn("wheel", content)
        self.assertIn("pointerdown", content)
        self.assertIn("jumpTo(node.time)", content)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest \
  test.test_bilibili_extension.BilibiliExtensionTests.test_mindmap_uses_canvas_viewport_and_toolbar \
  test.test_bilibili_extension.BilibiliExtensionTests.test_mindmap_canvas_supports_pan_zoom_and_jump
```

Expected: FAIL because the canvas renderer is not present.

- [ ] **Step 3: Add canvas state and layout helpers**

In `content.js`, near `learningState`, add:

```javascript
  let mindmapViewState = {
    scale: 1,
    x: 0,
    y: 0,
    collapsed: new Set(),
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragOriginX: 0,
    dragOriginY: 0,
  };
```

Add these helpers before `renderMindmap()`:

```javascript
  function nodeIdForPath(path) {
    return path.join("-");
  }

  function visibleMindmapNodes(nodes, depth = 0, path = []) {
    const out = [];
    nodes.forEach((node, index) => {
      const nextPath = [...path, index];
      const id = nodeIdForPath(nextPath);
      const children = Array.isArray(node.children) ? node.children : [];
      out.push({ id, node, depth, hasChildren: children.length > 0 });
      if (!mindmapViewState.collapsed.has(id)) {
        out.push(...visibleMindmapNodes(children, depth + 1, nextPath));
      }
    });
    return out;
  }

  function layoutMindmap(nodes) {
    const visible = visibleMindmapNodes(nodes);
    return visible.map((item, index) => ({
      ...item,
      x: 24 + item.depth * 190,
      y: 28 + index * 86,
      width: 158,
      height: 62,
    }));
  }

  function setMindmapTransform(transformEl, svgEl) {
    const value = `translate(${mindmapViewState.x}px, ${mindmapViewState.y}px) scale(${mindmapViewState.scale})`;
    transformEl.style.transform = value;
    svgEl.style.transform = value;
  }

  function centerMindmapCanvas(viewportEl, transformEl, svgEl) {
    mindmapViewState.scale = 1;
    mindmapViewState.x = Math.max(16, Math.floor(viewportEl.clientWidth / 2) - 100);
    mindmapViewState.y = 24;
    setMindmapTransform(transformEl, svgEl);
  }
```

- [ ] **Step 4: Replace mindmap renderer with canvas renderer**

Replace the current `renderMindmap()` and `renderMindmapNode()` block in `content.js` with:

```javascript
  function renderMindmapCanvas(mindmap, options = {}) {
    mindmapEl.textContent = "";
    const title = document.createElement("div");
    title.className = "funasr-mindmap-title";
    title.textContent = mindmap.title || "视频思维导图";
    mindmapEl.appendChild(title);

    const nodes = Array.isArray(mindmap.nodes) ? mindmap.nodes : [];
    if (!nodes.length) {
      const empty = document.createElement("div");
      empty.className = "funasr-chat-empty";
      empty.textContent = "还没有导图。";
      mindmapEl.appendChild(empty);
      return;
    }

    const toolbar = document.createElement("div");
    toolbar.className = "funasr-mindmap-toolbar";
    const viewport = document.createElement("div");
    viewport.className = "funasr-mindmap-viewport";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("funasr-mindmap-svg");
    const transform = document.createElement("div");
    transform.className = "funasr-mindmap-transform";

    const laidOut = layoutMindmap(nodes);
    const byId = new Map(laidOut.map((item) => [item.id, item]));
    svg.setAttribute("width", "1200");
    svg.setAttribute("height", String(Math.max(520, laidOut.length * 86 + 80)));

    for (const item of laidOut) {
      if (!item.id.includes("-")) {
        continue;
      }
      const parentId = item.id.split("-").slice(0, -1).join("-");
      const parent = byId.get(parentId);
      if (!parent) {
        continue;
      }
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${parent.x + parent.width} ${parent.y + 31} C ${parent.x + parent.width + 42} ${parent.y + 31}, ${item.x - 42} ${item.y + 31}, ${item.x} ${item.y + 31}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#b8c7d9");
      path.setAttribute("stroke-width", "2");
      svg.appendChild(path);
    }

    for (const item of laidOut) {
      const node = item.node;
      const card = document.createElement("button");
      card.className = "funasr-mindmap-canvas-node";
      card.type = "button";
      card.style.left = `${item.x}px`;
      card.style.top = `${item.y}px`;
      card.style.width = `${item.width}px`;
      card.setAttribute("aria-label", node.time !== undefined ? `跳转到 ${formatTime(node.time)}：${node.title || "未命名节点"}` : node.title || "未命名节点");

      if (item.hasChildren) {
        const fold = document.createElement("span");
        fold.className = "funasr-mindmap-fold";
        fold.textContent = mindmapViewState.collapsed.has(item.id) ? "+" : "-";
        fold.addEventListener("click", (event) => {
          event.stopPropagation();
          if (mindmapViewState.collapsed.has(item.id)) {
            mindmapViewState.collapsed.delete(item.id);
          } else {
            mindmapViewState.collapsed.add(item.id);
          }
          renderMindmapCanvas(mindmap, { activate: false });
        });
        card.appendChild(fold);
      }

      if (node.time !== undefined) {
        const time = document.createElement("span");
        time.className = "funasr-mindmap-time";
        time.textContent = formatTime(node.time);
        card.appendChild(time);
        card.addEventListener("click", () => jumpTo(node.time));
      }

      const label = document.createElement("span");
      label.className = "funasr-mindmap-node-title";
      label.textContent = node.title || "未命名节点";
      card.appendChild(label);
      if (node.summary) {
        card.title = node.summary;
      }
      transform.appendChild(card);
    }

    function addTool(label, handler) {
      const button = document.createElement("button");
      button.className = "secondary";
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", handler);
      toolbar.appendChild(button);
    }

    addTool("居中", () => centerMindmapCanvas(viewport, transform, svg));
    addTool("+", () => {
      mindmapViewState.scale = Math.min(1.8, mindmapViewState.scale + 0.1);
      setMindmapTransform(transform, svg);
    });
    addTool("-", () => {
      mindmapViewState.scale = Math.max(0.4, mindmapViewState.scale - 0.1);
      setMindmapTransform(transform, svg);
    });
    addTool("展开", () => {
      mindmapViewState.collapsed.clear();
      renderMindmapCanvas(mindmap, { activate: false });
    });
    addTool("收起", () => {
      for (const item of laidOut) {
        if (item.hasChildren) mindmapViewState.collapsed.add(item.id);
      }
      renderMindmapCanvas(mindmap, { activate: false });
    });

    viewport.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".funasr-mindmap-canvas-node")) return;
      mindmapViewState.dragging = true;
      mindmapViewState.dragStartX = event.clientX;
      mindmapViewState.dragStartY = event.clientY;
      mindmapViewState.dragOriginX = mindmapViewState.x;
      mindmapViewState.dragOriginY = mindmapViewState.y;
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!mindmapViewState.dragging) return;
      mindmapViewState.x = mindmapViewState.dragOriginX + event.clientX - mindmapViewState.dragStartX;
      mindmapViewState.y = mindmapViewState.dragOriginY + event.clientY - mindmapViewState.dragStartY;
      setMindmapTransform(transform, svg);
    });
    viewport.addEventListener("pointerup", () => {
      mindmapViewState.dragging = false;
    });
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? -0.08 : 0.08;
      mindmapViewState.scale = Math.max(0.4, Math.min(1.8, mindmapViewState.scale + delta));
      setMindmapTransform(transform, svg);
    }, { passive: false });

    viewport.appendChild(svg);
    viewport.appendChild(transform);
    mindmapEl.appendChild(toolbar);
    mindmapEl.appendChild(viewport);
    requestAnimationFrame(() => centerMindmapCanvas(viewport, transform, svg));
    renderStudyState({ mindmap: true });
    if (options.activate !== false) {
      setActiveTab("mindmap");
    }
  }

  function renderMindmap(mindmap, options = {}) {
    renderMindmapCanvas(mindmap, options);
  }
```

- [ ] **Step 5: Add canvas CSS**

Replace the old mindmap node/card CSS block in `sidebar.css` or append these rules after it:

```css
#funasr-sidebar-host .funasr-mindmap-toolbar {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 6px;
}

#funasr-sidebar-host .funasr-mindmap-toolbar button {
  padding: 7px 5px;
  font-size: 11px;
}

#funasr-sidebar-host .funasr-mindmap-viewport {
  position: relative;
  height: 460px;
  overflow: hidden;
  border: 1px solid #dbe4ee;
  border-radius: 7px;
  background:
    linear-gradient(#eef3f8 1px, transparent 1px),
    linear-gradient(90deg, #eef3f8 1px, transparent 1px),
    #fbfcfe;
  background-size: 24px 24px;
  cursor: grab;
  touch-action: none;
}

#funasr-sidebar-host .funasr-mindmap-viewport:active {
  cursor: grabbing;
}

#funasr-sidebar-host .funasr-mindmap-svg,
#funasr-sidebar-host .funasr-mindmap-transform {
  position: absolute;
  left: 0;
  top: 0;
  width: 1200px;
  min-height: 520px;
  transform-origin: 0 0;
}

#funasr-sidebar-host .funasr-mindmap-svg {
  pointer-events: none;
}

#funasr-sidebar-host .funasr-mindmap-canvas-node {
  position: absolute;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 4px 6px;
  align-items: center;
  min-height: 58px;
  border: 1px solid #dbe4ee;
  border-radius: 7px;
  padding: 7px;
  color: #17202a;
  background: #ffffff;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
  text-align: left;
}

#funasr-sidebar-host .funasr-mindmap-canvas-node:hover {
  border-color: #0f766e;
  background: #f3fbf8;
}

#funasr-sidebar-host .funasr-mindmap-fold {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  color: #ffffff;
  background: #0f766e;
  font-size: 12px;
  font-weight: 900;
}
```

- [ ] **Step 6: Run extension tests**

Run:

```bash
python3 -m unittest test.test_bilibili_extension
```

Expected: PASS.

- [ ] **Step 7: Commit if git is available**

Run:

```bash
git add browser-extension/bilibili-funasr-sidebar/content.js browser-extension/bilibili-funasr-sidebar/sidebar.css test/test_bilibili_extension.py
git commit -m "feat: render mindmap as canvas"
```

Expected in this workspace: git may not be available; continue without committing if it fails.

## Task 6: Full Verification

**Files:**
- Verify: `tools/funasr_web_app.py`
- Verify: `tools/funasr_video_ui.py`
- Verify: `browser-extension/bilibili-funasr-sidebar/content.js`
- Verify: `browser-extension/bilibili-funasr-sidebar/sidebar.css`

- [ ] **Step 1: Run Python and extension unit tests**

Run:

```bash
python3 -m unittest test.test_funasr_web_app test.test_funasr_web_ui test.test_bilibili_extension
```

Expected: PASS.

- [ ] **Step 2: Run codegraph sync**

Run:

```bash
npx @colbymchenry/codegraph sync .
```

Expected: completes without index errors.

- [ ] **Step 3: Manual smoke check with local service**

Run the service:

```bash
python3 tools/funasr_web_app.py
```

Expected: local FastAPI service starts on the configured port. If dependencies are missing, install or report the missing package.

In the browser extension, test a Bilibili video with subtitles:

```text
Click "分析当前视频"
Expected: status reaches checking-subtitles/importing-subtitles/done
Expected: transcript appears without running funasr-cli ASR
Expected: summary/mindmap/chat buttons become available
```

Test a video without subtitles:

```text
Click "分析当前视频"
Expected: status logs subtitle fallback
Expected: existing downloading/extracting/transcribing path runs
Expected: transcript appears after ASR
```

- [ ] **Step 4: Manual smoke check mindmap canvas**

With a completed transcript and generated mindmap:

```text
Open 导图 tab
Expected: toolbar, grid canvas, connected nodes are visible
Drag empty canvas
Expected: map pans
Mouse wheel
Expected: map zooms within clamp
Click time-bearing node
Expected: Bilibili video jumps to that timestamp
Click fold control
Expected: descendants collapse/expand
```

- [ ] **Step 5: Record git limitation if still present**

Run:

```bash
git status --short
```

Expected in a normal repo: shows changed files or clean tree. Expected in this workspace today: `fatal: not a git repository`; mention this in the final status if it persists.
