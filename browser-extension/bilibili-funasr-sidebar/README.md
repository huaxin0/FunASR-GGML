# FunASR-GGML Bilibili Sidebar

First-pass browser extension for the local FunASR-GGML video workflow.

## What It Does

- Injects a sidebar on Bilibili video pages.
- Finds the local service on `http://127.0.0.1:8008`, `:8009`, or `:8010`.
- Polls the local ASR job until it finishes.
- Reads `transcript.json` and renders readable ~10 second subtitle nodes.
- Supports sidebar subtitle search and highlights the node near the current
  playback time.
- Generates DeepSeek-powered takeaways, timestamped highlights, and Markdown
  notes from a completed local transcript.
- Uses separate tabs for summary, transcript, and logs so long videos stay
  readable.
- Adds a video Q&A tab backed by the saved transcript and summary.
- Renders summary notes and chat answers as readable Markdown instead of raw
  text blocks.
- Turns timestamps in notes and answers into clickable video jumps.
- Generates a sidebar-friendly tree mind map with clickable timestamp nodes.
- Clicking a timestamp controls the current Bilibili `<video>` element:

```js
video.currentTime = start;
video.play();
```

## Local Reuse

Completed jobs are recorded by the local service in:

```text
outputs/video_asr_web/index.json
```

On a Bilibili video page, the sidebar looks up the current BV id and page number.
If a previous result exists, it shows "加载已有结果" and loads the saved
`transcript.json` without running ASR again.

When re-analyzing a video that already has local media, the service prefers the
cached 16 kHz wav file. If only the downloaded source media exists, it reuses
that file and only runs ffmpeg + ASR again.

DeepSeek summaries are saved next to the transcript:

```text
summary.json
summary.md
```

Video Q&A is saved as the default local session:

```text
chat_sessions/default.json
```

Mind maps are saved next to the transcript:

```text
mindmap.json
```

Use the "导图" tab to browse the generated tree. Nodes with timestamps jump the
current Bilibili video to that point.

Use the "对话" tab to ask about concepts, timestamps, comparisons, or review
questions. Press Ctrl+Enter in the chat box to send.

Timestamps such as `[3:21-5:14]` or `34:58` in the rendered notes and answers
are clickable and seek the current Bilibili video.

The Markdown note prompt asks for a detailed study-note shape: one-line video
overview, core concepts, timestamped explanations, terminology, pitfalls, and a
review checklist.

Start the local service with an API key, or paste the key into the sidebar field:

```bash
DEEPSEEK_API_KEY=sk-... FUNASR_WEB_APP_PORT=8008 python3 tools/funasr_web_app.py
```

After editing the service or extension:

1. Restart `tools/funasr_web_app.py`.
2. Reload the unpacked extension in `chrome://extensions` or `edge://extensions`.
3. Refresh the Bilibili page.

## Run The Local Service

From the repository root:

```bash
FUNASR_WEB_APP_PORT=8008 python3 tools/funasr_web_app.py
```

If Bilibili needs login state under WSL2, keep a cookies file at:

```text
<repo>/cookies.txt
```

The local service uses this path by default.

## Load In Chrome Or Edge

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable Developer mode.
3. Click "Load unpacked".
4. Select `browser-extension/bilibili-funasr-sidebar`.
5. Open a Bilibili video page and click "分析当前视频".

This is a no-build MV3 MVP. It is intentionally simple so the product loop can
be validated before moving to WXT + React.
