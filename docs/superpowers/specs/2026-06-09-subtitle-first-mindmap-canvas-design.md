# Subtitle-First Offline Learning and Mindmap Canvas Design

## Goal

Improve the browser-extension offline learning experience in two focused ways:

1. Use video-provided subtitles first when they are available, and run local ASR only as a fallback.
2. Replace the current mindmap tree/card rendering with a lightweight draggable, zoomable canvas that feels closer to an XMind-style learning map.

The SDK demo is out of scope. The work targets the Bilibili extension path, the local FastAPI service, and the existing transcript/library outputs.

## Current Behavior

The plugin sends the current video URL to the local FastAPI service. The service downloads media with `yt-dlp`, extracts a 16 kHz mono WAV with `ffmpeg`, runs `funasr-cli`, writes `transcript.srt`, `transcript.txt`, `transcript.json`, and `stats.json`, then records the result in the local library index.

This works well for videos without subtitles, but it is slower than necessary for videos that already provide usable subtitles. The current mindmap UI stores a useful tree-shaped JSON document, but renders it as nested cards/details rather than a real canvas.

## Subtitle-First Design

### Behavior

For URL jobs, the backend first checks for downloadable subtitles before downloading audio/video for ASR.

The job phase flow becomes:

```text
validating
  -> checking-subtitles
  -> importing-subtitles
  -> writing-results
  -> done
```

If subtitle import fails because no subtitle exists, the existing ASR flow continues:

```text
validating
  -> checking-subtitles
  -> downloading
  -> extracting
  -> transcribing
  -> writing-results
  -> done
```

Uploaded local media continues to use ASR, because there is no URL subtitle source to query.

### Subtitle Priority

The backend chooses the first usable subtitle track in this order:

1. Manual Chinese subtitles: `zh-Hans`, `zh-CN`, `zh`, `zh-Hant`, `zh-TW`
2. Automatic Chinese subtitles with the same language preference
3. Manual subtitles in any language
4. Automatic subtitles in any language
5. Local ASR fallback

This keeps the default behavior simple: if a video has subtitles, use them automatically. If the chosen subtitle language is not Chinese, the result still loads because the user explicitly chose subtitle-first behavior.

### Implementation Shape

Add focused helpers in `tools/funasr_web_app.py` or move shared parsing helpers into `tools/funasr_video_ui.py` if reuse is cleaner:

- `build_subtitle_download_command(source, subtitle_dir, cookies_from_browser, cookies_path) -> list[str]`
- `find_downloaded_subtitle(subtitle_dir) -> SubtitleCandidate | None`
- `parse_vtt(path) -> list[dict[str, object]]`
- `segments_to_srt(segments) -> str`
- `write_transcript_outputs_from_segments(cfg, segments, elapsed_sec, source_info) -> None`
- `try_import_subtitles(cfg, work_dir, job, cookies_from_browser, cookies_path) -> bool`

`try_import_subtitles` runs only for URL sources. It creates a subtitle directory under `media/subtitles`, asks `yt-dlp` to write subtitles without downloading full media, parses the chosen file, writes the same transcript outputs as ASR, and returns `True` when at least one non-empty segment is available.

If it returns `False`, the current media-download and ASR path runs unchanged.

### Output Contract

The existing transcript file names remain unchanged:

- `transcript.srt`
- `transcript.txt`
- `transcript.json`
- `stats.json`

`transcript.json` keeps the existing shape:

```json
{
  "source": "https://www.bilibili.com/video/...",
  "segments": [
    {
      "index": 1,
      "start": 0.0,
      "end": 3.2,
      "text": "..."
    }
  ]
}
```

`stats.json` adds source metadata:

```json
{
  "transcript_source": "subtitle",
  "subtitle_language": "zh-Hans",
  "subtitle_kind": "manual"
}
```

For ASR fallback:

```json
{
  "transcript_source": "asr",
  "subtitle_language": null,
  "subtitle_kind": null
}
```

The library index continues to point to the same result files. The extension can load subtitle-derived and ASR-derived transcripts through the same code path.

### User-Facing Status

The extension should surface the shortcut without adding a choice dialog:

- While probing: `正在检查视频自带字幕...`
- When imported: `已使用视频自带字幕快速加载。`
- When falling back: `未找到可用字幕，正在使用本地 ASR。`

The current `status + log` polling model can support this with new backend job statuses. A later progress-bar pass can build on the same phases.

### Error Handling

Subtitle import must be best-effort. The backend falls back to ASR when:

- `yt-dlp` cannot list or download subtitles.
- No subtitle file appears.
- The subtitle file is empty.
- Parsing produces no non-empty text segments.
- The subtitle format is unsupported.

Only validation errors that would also block ASR, such as missing `yt-dlp`, missing model, or missing `funasr-cli`, should fail the job before fallback. Parser failures should be logged, not fatal.

## Mindmap Canvas Design

### Behavior

The saved `mindmap.json` schema stays compatible:

```json
{
  "title": "视频思维导图",
  "nodes": [
    {
      "title": "核心概念",
      "summary": "简要说明",
      "time": 123.0,
      "children": []
    }
  ]
}
```

The extension replaces the nested card renderer with a lightweight canvas renderer:

- A toolbar above the map provides center, zoom in, zoom out, expand all, and collapse all controls.
- A large viewport clips the map and captures drag/wheel interactions.
- A transform layer is translated and scaled for pan/zoom.
- Nodes are positioned with HTML absolute layout.
- Connectors are drawn with SVG lines or paths beneath the nodes.

The first version is a viewer, not an editor. Users can navigate, fold, unfold, and jump to video times, but they cannot edit node text or rearrange the tree manually.

### Layout

The first layout is deterministic and tree-based:

- Root title starts near the horizontal center of the canvas.
- First-level nodes fan out vertically to the right.
- Deeper child nodes continue to the right.
- Subtrees get vertical space based on the number of visible descendants.
- Collapsed nodes hide descendants and recompute the visible layout.

This is simpler than a full graph engine and keeps the plugin dependency-free. The layout can later evolve to left-right balancing if needed.

### Interactions

Pan and zoom:

- Drag empty canvas space to pan.
- Mouse wheel zooms around the cursor.
- Zoom is clamped, for example between `0.4` and `1.8`.
- Center resets pan/zoom to fit the visible map.

Node interaction:

- If a node has `time`, clicking its timestamp jumps the current Bilibili video through the existing `jumpTo(time)` helper.
- Clicking the node main area also jumps when `time` exists.
- A small fold control toggles children when the node has children.
- Nodes without `time` only fold or unfold; they do not jump.

Accessibility:

- Time-bearing nodes get an `aria-label` that includes the formatted timestamp and title.
- Toolbar buttons have clear text labels for the first version.
- Keyboard navigation can remain basic in this pass, but controls must be buttons rather than non-semantic divs.

### Frontend Structure

The existing `renderMindmap(mindmap, options)` remains the public entry point used by cache restore and generation flows. Internally it delegates to the new canvas renderer.

Suggested helpers in `browser-extension/bilibili-funasr-sidebar/content.js`:

- `flattenMindmapVisibleNodes(mindmap, collapsedIds)`
- `layoutMindmap(nodes)`
- `renderMindmapCanvas(mindmap, options)`
- `setMindmapTransform()`
- `centerMindmapCanvas()`
- `toggleMindmapNode(nodeId)`

Suggested CSS additions in `browser-extension/bilibili-funasr-sidebar/sidebar.css`:

- `.funasr-mindmap-toolbar`
- `.funasr-mindmap-viewport`
- `.funasr-mindmap-transform`
- `.funasr-mindmap-svg`
- `.funasr-mindmap-canvas-node`
- `.funasr-mindmap-fold`

The current card class names may stay temporarily if tests or styling reuse them, but the user-facing layout should be canvas-first.

## Data Flow

Subtitle path:

```text
content.js
  -> POST /api/jobs
  -> run_job()
  -> try_import_subtitles()
  -> write transcript outputs
  -> record_library_item()
  -> content.js loads transcript.json
```

ASR fallback path:

```text
content.js
  -> POST /api/jobs
  -> run_job()
  -> subtitle import returns false
  -> existing download/extract/transcribe/write path
```

Mindmap path:

```text
content.js
  -> POST /api/library/mindmap or cached GET
  -> existing mindmap.json shape
  -> renderMindmap()
  -> renderMindmapCanvas()
  -> jumpTo(node.time) on time-bearing nodes
```

## Testing

Backend tests in `test/test_funasr_web_app.py`:

- Subtitle download command includes subtitle options and respects cookie settings.
- VTT parsing produces indexed segments with start, end, and text.
- SRT parsing is still compatible.
- Writing transcript outputs from segments creates `transcript.srt`, `transcript.txt`, `transcript.json`, and `stats.json`.
- Subtitle import success prevents the ASR command from running.
- Subtitle import failure falls back to the existing ASR path.
- `stats.json` records `transcript_source`.

Extension tests in `test/test_bilibili_extension.py`:

- Mindmap panel contains a viewport, transform layer, SVG connector layer, and toolbar.
- Content script includes pan/zoom state and handlers.
- Content script keeps `jumpTo` integration for time-bearing mindmap nodes.
- Cache restore still calls `renderMindmap`.
- Old saved `mindmap.json` data can render without schema migration.

## Out of Scope

- Editing mindmap node text.
- Dragging individual nodes into custom positions.
- Exporting XMind files.
- Translating subtitles.
- Forcing user choice between subtitles and ASR.
- Changing the core ASR pipeline.
- Changing `examples/sdk_ptt_demo.cpp`.

## Rollout

This can ship incrementally:

1. Add subtitle parsing and transcript-output helpers.
2. Add subtitle-first import to `run_job`, with ASR fallback.
3. Update plugin status copy for subtitle import.
4. Replace mindmap rendering with canvas view.
5. Add focused tests for both paths.

The subtitle path is low risk because it preserves the current result file contract. The mindmap canvas is also isolated to extension rendering; the saved mindmap API does not need to change.
