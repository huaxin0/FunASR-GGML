#!/usr/bin/env python3
"""
Native local Web app for FunASR-GGML offline video transcription.

Run:
  python3 tools/funasr_web_app.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.request

try:
    from tools import funasr_video_ui as video_ui
except ModuleNotFoundError:
    import funasr_video_ui as video_ui

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse


ROOT = video_ui.ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "video_asr_web"
DEFAULT_COOKIES_PATH = ROOT / "cookies.txt"
DEFAULT_LIBRARY_INDEX = DEFAULT_OUTPUT_ROOT / "index.json"
RESULT_FILES = {
    "srt": "transcript.srt",
    "txt": "transcript.txt",
    "json": "transcript.json",
    "stats": "stats.json",
    "summary": "summary.json",
    "summary_md": "summary.md",
    "mindmap": "mindmap.json",
    "chat": "chat_sessions/default.json",
    "frames": "frames/frames.json",
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".m4v"}


@dataclass
class Job:
    id: str
    output_dir: Path
    status: str = "queued"
    log: list[str] = field(default_factory=list)
    media_files: dict[str, str] = field(default_factory=dict)
    transcript: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None

    def append(self, line: str) -> None:
        self.log.append(line.rstrip())

    def partial_transcript_state(self) -> tuple[str, list[dict[str, object]]]:
        srt_path = self.output_dir / RESULT_FILES["srt"]
        if not srt_path.exists():
            return self.transcript, []
        try:
            segments = video_ui.parse_srt(srt_path)
        except (OSError, UnicodeError, ValueError):
            return self.transcript, []

        transcript = self.transcript
        if segments and not transcript:
            lines = [str(item.get("text") or "").strip() for item in segments]
            transcript = "\n".join(line for line in lines if line)
            if transcript:
                transcript = f"{transcript}\n"
        return transcript, segments

    def public_state(self) -> dict[str, object]:
        files = {}
        if self.status == "done":
            for key, filename in RESULT_FILES.items():
                path = self.output_dir / filename
                if path.exists():
                    files[key] = f"/api/jobs/{self.id}/files/{key}"
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        transcript, segments = self.partial_transcript_state()
        return {
            "id": self.id,
            "status": self.status,
            "log": "\n".join(self.log),
            "transcript": transcript,
            "segments": segments,
            "error": self.error,
            "elapsed_sec": round(elapsed, 3),
            "files": files,
            "output_dir": str(self.output_dir),
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


def quote_arg(value: str) -> str:
    return video_ui.quote_arg(value)


def default_cookies_path() -> str:
    return str(DEFAULT_COOKIES_PATH)


def library_key_for_source(source: str) -> str:
    parsed = urlparse(source.strip())
    match = re.search(r"/video/([^/?#]+)/?", parsed.path)
    if "bilibili.com" in parsed.netloc and match:
        query = parse_qs(parsed.query)
        page = query.get("p", ["1"])[0] or "1"
        return f"bilibili:{match.group(1)}:p{page}"
    return source.strip()


def _read_library(index_path: Path) -> dict[str, object]:
    if not index_path.exists():
        return {"items": {}}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"items": {}}
    if not isinstance(data, dict):
        return {"items": {}}
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    return data


def _write_library(index_path: Path, data: dict[str, object]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(index_path)


def _result_file_map(output_dir: Path) -> dict[str, str]:
    files = {}
    for key, filename in RESULT_FILES.items():
        path = output_dir / filename
        if path.exists():
            files[key] = str(path)
    media_dir = output_dir / "media"
    if media_dir.exists():
        wavs = sorted(media_dir.glob("*_16k.wav"))
        if wavs:
            files["wav"] = str(wavs[0])
        downloads = sorted((media_dir / "download").glob("source.*"))
        if downloads:
            files["download"] = str(downloads[0])
    return files


def record_library_item(index_path: Path, *, source: str, job: Job) -> dict[str, object]:
    key = library_key_for_source(source)
    item = {
        "key": key,
        "source": source,
        "job_id": job.id,
        "status": job.status,
        "output_dir": str(job.output_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "files": _result_file_map(job.output_dir),
    }
    item["files"].update(job.media_files)
    data = _read_library(index_path)
    items = data.setdefault("items", {})
    assert isinstance(items, dict)
    items[key] = item
    _write_library(index_path, data)
    return item


def lookup_library_item(index_path: Path, source: str) -> dict[str, object] | None:
    key = library_key_for_source(source)
    data = _read_library(index_path)
    items = data.get("items", {})
    if not isinstance(items, dict):
        return None
    item = items.get(key)
    if not isinstance(item, dict):
        return None
    output_dir = Path(str(item.get("output_dir", "")))
    if not (output_dir / "transcript.json").exists():
        return None
    return item


def reusable_media_for_source(index_path: Path, source: str) -> dict[str, str]:
    item = lookup_library_item(index_path, source)
    if item is None:
        return {}
    files = item.get("files", {})
    if not isinstance(files, dict):
        return {}
    reusable = {}
    for key in ("wav", "download"):
        path = Path(str(files.get(key, "")))
        if path.exists():
            reusable[key] = str(path)
    return reusable


def library_item_state(item: dict[str, object]) -> dict[str, object]:
    output_dir = Path(str(item["output_dir"]))
    files = {}
    for key, filename in RESULT_FILES.items():
        if (output_dir / filename).exists():
            files[key] = f"/api/library/files/{item['key']}/{key}"
    return {
        "found": True,
        "key": item["key"],
        "job_id": item.get("job_id", ""),
        "source": item.get("source", ""),
        "status": item.get("status", ""),
        "output_dir": str(output_dir),
        "files": files,
    }


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def transcript_prompt_text(transcript: dict[str, object], max_segments: int = 900) -> str:
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    lines = []
    for raw in segments[:max_segments]:
        if not isinstance(raw, dict):
            continue
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or start)
        text = str(raw.get("text") or "").strip()
        if text:
            lines.append(f"[{format_seconds(start)}-{format_seconds(end)}] {text}")
    if len(segments) > max_segments:
        lines.append(f"... 已截取前 {max_segments} 段字幕用于第一版总结。")
    return "\n".join(lines)


def build_deepseek_summary_payload(
    transcript: dict[str, object],
    model: str = DEFAULT_DEEPSEEK_MODEL,
    custom_prompt: str = "",
) -> dict[str, object]:
    system = (
        "你是一个严谨的视频学习笔记助手，工作方式参考成熟 skill：先理解目标，再提炼结构，"
        "最后输出可复习、可跳转、可保存的学习材料。请根据带时间戳的字幕生成严格 JSON。"
        "JSON 字段必须包含 title、takeaways、highlights、markdown。"
        "takeaways 输出 5-10 条核心要点。highlights 输出 5-10 个重要片段，"
        "每项包含 start、end、title、summary，start/end 使用秒数数字。"
        "markdown 必须是一份详细学习笔记，不要只输出短 bullet。"
        "markdown 必须包含以下小节：视频一句话概览、核心知识点、重点片段讲解、术语解释、"
        "易错点/注意点、复习清单。重点片段讲解中要保留时间戳，方便回看。"
        "如果字幕信息不足，请明确写出不确定性，不要编造视频中没有出现的事实。"
    )
    extra = custom_prompt.strip()
    user = (
        "请总结下面的视频字幕。目标用户是正在学习视频内容的人，"
        "他们需要一份比普通摘要更详细的学习笔记。"
        "请尽量解释概念、串联上下文、指出值得回看的时间段。\n\n"
        f"{'用户额外要求：' + extra + chr(10) + chr(10) if extra else ''}"
        f"{transcript_prompt_text(transcript)}"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.2,
    }


def _normalize_summary(summary: dict[str, object]) -> dict[str, object]:
    title = str(summary.get("title") or "视频学习笔记")
    takeaways = summary.get("takeaways", [])
    if not isinstance(takeaways, list):
        takeaways = []
    highlights = summary.get("highlights", [])
    if not isinstance(highlights, list):
        highlights = []
    normalized_highlights = []
    for item in highlights:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start") or 0)
        end = float(item.get("end") or start)
        normalized_highlights.append(
            {
                "start": start,
                "end": end,
                "title": str(item.get("title") or "重点片段"),
                "summary": str(item.get("summary") or ""),
            }
        )
    markdown = str(summary.get("markdown") or "")
    if not markdown:
        body = "\n".join(f"- {x}" for x in takeaways)
        markdown = f"# {title}\n\n{body}\n"
    return {
        "title": title,
        "takeaways": [str(x) for x in takeaways],
        "highlights": normalized_highlights,
        "markdown": markdown,
    }


def call_deepseek_summary(
    transcript: dict[str, object],
    *,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
    custom_prompt: str = "",
) -> dict[str, object]:
    if not api_key.strip():
        raise ValueError("DeepSeek API key is required.")
    payload = build_deepseek_summary_payload(transcript, model=model, custom_prompt=custom_prompt)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek request failed: {exc.code} {detail}") from exc
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return _normalize_summary(json.loads(content))


def write_summary_outputs(output_dir: Path, summary: dict[str, object]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_summary(summary)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(normalized["markdown"], encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def summarize_library_item(
    item: dict[str, object],
    *,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    custom_prompt: str = "",
) -> dict[str, object]:
    output_dir = Path(str(item["output_dir"]))
    transcript_path = output_dir / "transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    summary = call_deepseek_summary(
        transcript,
        api_key=api_key,
        model=model,
        custom_prompt=custom_prompt,
    )
    write_summary_outputs(output_dir, summary)
    return summary


def read_summary_for_item(item: dict[str, object]) -> dict[str, object] | None:
    path = Path(str(item["output_dir"])) / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_summary(data)


def _normalize_string_list(value: object, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value[:limit] if str(item).strip()]


def _normalize_timestamp_list(value: object, fallback: object = None) -> list[float]:
    raw_values = value if isinstance(value, list) else []
    if not raw_values and fallback is not None:
        raw_values = [fallback]
    timestamps = []
    for raw in raw_values[:8]:
        try:
            timestamps.append(float(raw))
        except (TypeError, ValueError):
            continue
    return timestamps


def _normalize_evidence_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    evidence = []
    for item in value[:6]:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            normalized: dict[str, object] = {"text": text}
            if item.get("time") is not None:
                try:
                    normalized["time"] = float(item.get("time"))
                except (TypeError, ValueError):
                    pass
            evidence.append(normalized)
        else:
            text = str(item).strip()
            if text:
                evidence.append({"text": text})
    return evidence


def _normalize_mindmap_node(node: dict[str, object]) -> dict[str, object]:
    children = node.get("children", [])
    if not isinstance(children, list):
        children = []
    normalized = {
        "title": str(node.get("title") or "未命名节点"),
        "kind": str(node.get("kind") or "concept"),
        "summary": str(node.get("summary") or ""),
        "details": _normalize_string_list(node.get("details"), limit=6),
        "evidence": _normalize_evidence_list(node.get("evidence")),
        "questions": _normalize_string_list(node.get("questions"), limit=4),
        "timestamps": _normalize_timestamp_list(node.get("timestamps"), fallback=node.get("time")),
        "children": [
            _normalize_mindmap_node(child)
            for child in children
            if isinstance(child, dict)
        ],
    }
    if node.get("time") is not None:
        try:
            normalized["time"] = float(node.get("time"))
        except (TypeError, ValueError):
            pass
    return normalized


def _normalize_mindmap(mindmap: dict[str, object]) -> dict[str, object]:
    nodes = mindmap.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    return {
        "title": str(mindmap.get("title") or "视频思维导图"),
        "summary": str(mindmap.get("summary") or ""),
        "nodes": [
            _normalize_mindmap_node(node)
            for node in nodes
            if isinstance(node, dict)
        ],
    }


def build_deepseek_mindmap_payload(
    *,
    transcript: dict[str, object],
    summary: dict[str, object] | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> dict[str, object]:
    normalized_summary = _normalize_summary(summary or {})
    system = (
        "你是一个视频学习思维导图/知识导图助手。请根据视频学习笔记和带时间戳字幕生成严格 JSON。"
        "输出必须是内容学习导图，不是时间轴目录；时间戳只是回看锚点，不能作为主要标题或层级。"
        "输出字段必须包含 title、summary 和 nodes。nodes 是树状数组。"
        "每个节点包含 title、kind、summary、details、evidence、questions、timestamps、time、children。"
        "kind 可用 concept、process、comparison、example、mistake、question、recap。"
        "details 是 2-5 条具体知识点；evidence 是带 time/text 的视频证据；questions 是复习自测问题。"
        "timestamps 是秒数数组，time 是最重要的回看秒数。children 继续使用同样结构。"
        "优先按概念、因果、步骤、对比、例子、误区、复习线索组织，而不是按时间顺序堆节点。"
        "中文解释里可以自然保留英文术语、专有名词和代码词。"
        "不要输出 Markdown，不要输出解释文字，只输出 JSON object。"
    )
    user = (
        f"学习笔记标题：{normalized_summary['title']}\n\n"
        f"核心要点：\n" + "\n".join(f"- {x}" for x in normalized_summary["takeaways"]) + "\n\n"
        f"学习笔记：\n{normalized_summary['markdown']}\n\n"
        f"字幕片段：\n{transcript_prompt_text(transcript, max_segments=700)}"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.2,
    }


def call_deepseek_mindmap(
    *,
    transcript: dict[str, object],
    summary: dict[str, object] | None,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
) -> dict[str, object]:
    if not api_key.strip():
        raise ValueError("DeepSeek API key is required.")
    payload = build_deepseek_mindmap_payload(
        transcript=transcript,
        summary=summary,
        model=model,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek request failed: {exc.code} {detail}") from exc
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return _normalize_mindmap(json.loads(content))


def write_mindmap_output(output_dir: Path, mindmap: dict[str, object]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_mindmap(mindmap)
    path = output_dir / "mindmap.json"
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(path)


def read_mindmap_from_output(output_dir: Path) -> dict[str, object] | None:
    path = output_dir / "mindmap.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_mindmap(data)


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def cached_video_media_for_item(item: dict[str, object]) -> Path | None:
    files = item.get("files", {})
    if isinstance(files, dict):
        download = Path(str(files.get("download", "")))
        if download.exists() and _is_video_file(download):
            return download
    output_dir = Path(str(item["output_dir"]))
    download_dir = output_dir / "media" / "download"
    if download_dir.exists():
        for candidate in sorted(download_dir.glob("source.*")):
            if candidate.exists() and _is_video_file(candidate):
                return candidate
    return None


def frame_filename(index: int, start: float) -> str:
    total = max(0, int(start))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{index:03d}_{h:02d}-{m:02d}-{s:02d}.jpg"


def build_frame_command(video_path: Path, start: float, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, float(start)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]


def read_frames_from_output(output_dir: Path) -> dict[str, object] | None:
    path = output_dir / "frames" / "frames.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        return {"frames": []}
    return data


def extract_frames_for_item(
    item: dict[str, object],
    *,
    run_command=subprocess.run,
    max_frames: int = 8,
) -> dict[str, object]:
    output_dir = Path(str(item["output_dir"]))
    video_path = cached_video_media_for_item(item)
    if video_path is None:
        raise ValueError("No cached video media is available for frame extraction.")
    summary = read_summary_for_item(item)
    if summary is None:
        raise ValueError("Summary highlights are required before extracting key frames.")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, highlight in enumerate(summary["highlights"][:max_frames]):
        start = float(highlight.get("start") or 0)
        image_name = frame_filename(index, start)
        image_path = frames_dir / image_name
        run_command(build_frame_command(video_path, start, image_path), check=True)
        frames.append(
            {
                "start": start,
                "end": float(highlight.get("end") or start),
                "title": str(highlight.get("title") or "重点画面"),
                "summary": str(highlight.get("summary") or ""),
                "image": image_name,
            }
        )
    payload = {"frames": frames}
    (frames_dir / "frames.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def mindmap_library_item(
    item: dict[str, object],
    *,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> dict[str, object]:
    output_dir = Path(str(item["output_dir"]))
    transcript = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
    summary = read_summary_for_item(item)
    mindmap = call_deepseek_mindmap(
        transcript=transcript,
        summary=summary,
        api_key=api_key,
        model=model,
    )
    write_mindmap_output(output_dir, mindmap)
    return mindmap


def relevant_transcript_text(
    question: str,
    transcript: dict[str, object],
    max_segments: int = 80,
) -> str:
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        return ""
    terms = [x.lower() for x in re.findall(r"[\w\u4e00-\u9fff]+", question) if len(x) >= 2]
    scored = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        lower = text.lower()
        score = sum(1 for term in terms if term in lower)
        if score:
            scored.append((score, index, raw))
    if not scored:
        selected = [raw for raw in segments[:max_segments] if isinstance(raw, dict)]
    else:
        selected = [raw for _, _, raw in sorted(scored, key=lambda x: (-x[0], x[1]))[:max_segments]]
        selected.sort(key=lambda raw: float(raw.get("start") or 0))
    lines = []
    for raw in selected:
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or start)
        text = str(raw.get("text") or "").strip()
        if text:
            lines.append(f"[{format_seconds(start)}-{format_seconds(end)}] {text}")
    return "\n".join(lines)


def build_deepseek_chat_payload(
    *,
    question: str,
    transcript: dict[str, object],
    summary: dict[str, object] | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> dict[str, object]:
    normalized_summary = _normalize_summary(summary or {})
    context = relevant_transcript_text(question, transcript)
    system = (
        "你是一个视频学习对话助手。你只能基于给定的视频总结和字幕片段回答，"
        "不要编造视频中没有的信息。回答要适合学习者：先直接回答，再解释原因，"
        "必要时列出步骤、例子或对比。可以引用时间戳，格式如 [1:23]，"
        "这样前端可以帮助用户跳回视频。"
    )
    user = (
        f"视频标题：{normalized_summary['title']}\n\n"
        f"核心要点：\n" + "\n".join(f"- {x}" for x in normalized_summary["takeaways"]) + "\n\n"
        f"已有学习笔记：\n{normalized_summary['markdown']}\n\n"
        f"相关字幕片段：\n{context}\n\n"
        f"用户问题：{question}"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0.2,
    }


def call_deepseek_chat(
    *,
    question: str,
    transcript: dict[str, object],
    summary: dict[str, object] | None,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
) -> str:
    if not api_key.strip():
        raise ValueError("DeepSeek API key is required.")
    payload = build_deepseek_chat_payload(
        question=question,
        transcript=transcript,
        summary=summary,
        model=model,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"DeepSeek request failed: {exc.code} {detail}") from exc
    data = json.loads(raw)
    return str(data["choices"][0]["message"]["content"])


def chat_session_path(output_dir: Path) -> Path:
    return output_dir / "chat_sessions" / "default.json"


def read_chat_session(output_dir: Path) -> dict[str, object]:
    path = chat_session_path(output_dir)
    if not path.exists():
        return {"messages": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return {"messages": []}
    return data


def append_chat_message(output_dir: Path, *, question: str, answer: str) -> dict[str, object]:
    session = read_chat_session(output_dir)
    messages = session.setdefault("messages", [])
    assert isinstance(messages, list)
    messages.append(
        {
            "question": question,
            "answer": answer,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    path = chat_session_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return session


def clear_chat_session(output_dir: Path) -> dict[str, object]:
    session = {"messages": []}
    path = chat_session_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return session


def chat_with_library_item(
    item: dict[str, object],
    *,
    question: str,
    api_key: str,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> dict[str, object]:
    output_dir = Path(str(item["output_dir"]))
    transcript = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
    summary = read_summary_for_item(item)
    answer = call_deepseek_chat(
        question=question,
        transcript=transcript,
        summary=summary,
        api_key=api_key,
        model=model,
    )
    return append_chat_message(output_dir, question=question, answer=answer)


def build_job_config(
    *,
    source: str,
    model: str,
    funasr_cli: str,
    output_dir: Path,
    use_gpu: bool,
    keep_media: bool,
) -> video_ui.Config:
    return video_ui.Config(
        source=source,
        model=Path(model).expanduser(),
        funasr_cli=Path(funasr_cli).expanduser(),
        output_dir=Path(output_dir).expanduser(),
        use_gpu=use_gpu,
        use_vad=False,
        chunk_mode="window",
        chunk_sec="30",
        vad_model=None,
        vad_threshold="0.45",
        vad_min_silence_ms="800",
        vad_speech_pad_ms="120",
        vad_max_speech_sec="30",
        ctx_size="4096",
        max_tokens="220",
        srt_max_chars="28",
        offline_scheduler=True,
        offline_profile=True,
        offline_batch_size="12",
        offline_kv_mode="paged",
        offline_kv_block_size="128",
        keep_media=keep_media,
        offline_preset="long-video",
    )


def build_download_command(
    source: str,
    download_dir: Path,
    cookies_from_browser: str = "",
    cookies_path: str = "",
) -> list[str]:
    template = str(download_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
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


def validate_source(url: str, has_upload: bool) -> None:
    if not url.strip() and not has_upload:
        raise ValueError("Paste a URL or upload a local media file.")


def build_ffmpeg_command(media_path: Path, wav_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]


def build_asr_command(cfg: video_ui.Config, wav_path: Path, srt_path: Path) -> list[str]:
    cmd = [
        str(cfg.funasr_cli),
        "-m",
        str(cfg.model),
        "-f",
        str(wav_path),
        "--max-tokens",
        cfg.max_tokens,
        "--ctx-size",
        cfg.ctx_size,
        "--srt-max-chars",
        cfg.srt_max_chars,
        "-osrt",
        "-o",
        str(srt_path),
    ]
    if cfg.use_gpu:
        cmd.append("--gpu")
    cmd.extend(["--chunk-mode", cfg.chunk_mode])
    if cfg.chunk_mode == "window":
        cmd.extend(["--chunk-sec", cfg.chunk_sec])
    cmd.extend(["--offline-preset", "long-video"])
    return cmd


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    millis = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"


def run_command_stream(cmd: list[str], job: Job) -> None:
    job.append("")
    job.append("$ " + " ".join(quote_arg(x) for x in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        job.append(line)
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def download_or_use_source(
    cfg: video_ui.Config,
    work_dir: Path,
    job: Job,
    cookies_from_browser: str,
    cookies_path: str,
) -> Path:
    if not video_ui.is_url(cfg.source):
        local = Path(cfg.source).expanduser()
        if not local.exists():
            raise FileNotFoundError(f"Local media not found: {local}")
        job.append(f"Using uploaded media: {local}")
        return local

    download_dir = work_dir / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    run_command_stream(
        build_download_command(
            cfg.source,
            download_dir,
            cookies_from_browser=cookies_from_browser,
            cookies_path=cookies_path,
        ),
        job,
    )
    candidates = sorted(p for p in download_dir.iterdir() if p.is_file())
    if not candidates:
        raise RuntimeError("yt-dlp finished but no media file was found.")
    return candidates[0]


def _copy_upload(upload: UploadFile, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(upload.filename or "upload.media").name
    dest = upload_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return dest


def _new_job_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / stamp
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}_{suffix}"
        suffix += 1
    return candidate


def run_job(job: Job, cfg: video_ui.Config, cookies_from_browser: str, cookies_path: str) -> None:
    try:
        job.status = "validating"
        video_ui.validate_config(cfg)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = cfg.output_dir / "media"
        work_dir.mkdir(parents=True, exist_ok=True)
        video_ui.write_run_config(cfg)

        started = time.monotonic()

        reusable = reusable_media_for_source(DEFAULT_LIBRARY_INDEX, cfg.source)
        if video_ui.is_url(cfg.source) and reusable.get("wav"):
            wav_path = Path(reusable["wav"])
            job.media_files.update(reusable)
            job.append(f"Reusing cached 16k wav: {wav_path}")
        else:
            if video_ui.is_url(cfg.source) and reusable.get("download"):
                media_path = Path(reusable["download"])
                job.media_files.update(reusable)
                job.append(f"Reusing cached downloaded media: {media_path}")
            else:
                job.status = "downloading"
                media_path = download_or_use_source(cfg, work_dir, job, cookies_from_browser, cookies_path)

            job.status = "extracting"
            wav_path = work_dir / f"{video_ui.safe_slug(Path(media_path).stem)}_16k.wav"
            run_command_stream(build_ffmpeg_command(media_path, wav_path), job)

        job.status = "transcribing"
        srt_path = cfg.output_dir / "transcript.srt"
        run_command_stream(build_asr_command(cfg, wav_path, srt_path), job)

        job.status = "writing-results"
        video_ui.write_side_outputs(
            cfg,
            srt_path,
            time.monotonic() - started,
            {"transcript_source": "asr", "subtitle_language": None, "subtitle_kind": None},
        )
        txt_path = cfg.output_dir / "transcript.txt"
        job.transcript = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""

        if not cfg.keep_media:
            shutil.rmtree(work_dir, ignore_errors=True)

        job.status = "done"
        record_library_item(DEFAULT_LIBRARY_INDEX, source=cfg.source, job=job)
        job.append(f"Done. Output: {cfg.output_dir}")
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.append(f"Failed: {exc}")
    finally:
        job.finished_at = time.monotonic()


def render_index() -> str:
    return INDEX_HTML


def create_app() -> FastAPI:
    app = FastAPI(title="FunASR-GGML Offline Video")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = JobStore()
    app.state.jobs = store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(render_index())

    @app.post("/api/jobs")
    def create_job(
        background: BackgroundTasks,
        url: str = Form(""),
        model: str = Form(str(video_ui.default_model())),
        funasr_cli: str = Form(str(video_ui.default_funasr_cli())),
        output_root: str = Form(str(DEFAULT_OUTPUT_ROOT)),
        cookies_from_browser: str = Form(""),
        cookies_path: str = Form(default_cookies_path()),
        use_gpu: bool = Form(True),
        keep_media: bool = Form(True),
        upload: UploadFile | None = File(None),
    ) -> dict[str, str]:
        url = url.strip()
        has_upload = upload is not None and bool(upload.filename)
        try:
            validate_source(url, has_upload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_root_path = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser()
        job_id = uuid.uuid4().hex[:12]
        output_dir = _new_job_dir(output_root_path)
        job = Job(id=job_id, output_dir=output_dir)
        store.add(job)

        source = url
        if has_upload and upload is not None:
            source = str(_copy_upload(upload, output_dir / "media" / "upload"))
            job.append("Uploaded file selected; URL input ignored.")

        cfg = build_job_config(
            source=source,
            model=model,
            funasr_cli=funasr_cli,
            output_dir=output_dir,
            use_gpu=use_gpu,
            keep_media=keep_media,
        )
        background.add_task(run_job, job, cfg, cookies_from_browser, cookies_path)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str) -> dict[str, object]:
        return store.get(job_id).public_state()

    @app.get("/api/library/lookup")
    def library_lookup(url: str) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            return {"found": False}
        return library_item_state(item)

    @app.get("/api/library/summary")
    def library_summary(url: str) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        summary = read_summary_for_item(item)
        if summary is None:
            return {"found": False}
        return {"found": True, "summary": summary}

    @app.post("/api/library/summarize")
    def library_summarize(
        url: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(DEFAULT_DEEPSEEK_MODEL),
        custom_prompt: str = Form(""),
    ) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        key = api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        try:
            summary = summarize_library_item(
                item,
                api_key=key,
                model=model,
                custom_prompt=custom_prompt,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"found": True, "summary": summary}

    @app.get("/api/library/chat")
    def library_chat(url: str) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        return {"found": True, "session": read_chat_session(Path(str(item["output_dir"])))}

    @app.post("/api/library/chat")
    def library_chat_post(
        url: str = Form(""),
        question: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(DEFAULT_DEEPSEEK_MODEL),
    ) -> dict[str, object]:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        key = api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        try:
            session = chat_with_library_item(
                item,
                question=question.strip(),
                api_key=key,
                model=model,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"found": True, "session": session}

    @app.post("/api/library/chat/clear")
    def library_chat_clear(url: str = Form("")) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        session = clear_chat_session(Path(str(item["output_dir"])))
        return {"found": True, "session": session}

    @app.get("/api/library/mindmap")
    def library_mindmap(url: str) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        mindmap = read_mindmap_from_output(Path(str(item["output_dir"])))
        if mindmap is None:
            return {"found": False}
        return {"found": True, "mindmap": mindmap}

    @app.post("/api/library/mindmap")
    def library_mindmap_post(
        url: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(DEFAULT_DEEPSEEK_MODEL),
    ) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        key = api_key.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        try:
            mindmap = mindmap_library_item(item, api_key=key, model=model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"found": True, "mindmap": mindmap}

    @app.get("/api/library/frames")
    def library_frames(url: str) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        frames = read_frames_from_output(Path(str(item["output_dir"])))
        if frames is None:
            return {"found": False}
        key = str(item["key"])
        enriched = []
        for frame in frames.get("frames", []):
            if not isinstance(frame, dict):
                continue
            image = str(frame.get("image") or "")
            enriched.append(
                {
                    **frame,
                    "image_url": f"/api/library/frames/{key}/{image}",
                }
            )
        return {"found": True, "frames": {"frames": enriched}}

    @app.post("/api/library/frames")
    def library_frames_post(url: str = Form("")) -> dict[str, object]:
        item = lookup_library_item(DEFAULT_LIBRARY_INDEX, url)
        if item is None:
            raise HTTPException(status_code=404, detail="Library item not found")
        try:
            frames = extract_frames_for_item(item)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        key = str(item["key"])
        enriched = []
        for frame in frames.get("frames", []):
            image = str(frame.get("image") or "")
            enriched.append({**frame, "image_url": f"/api/library/frames/{key}/{image}"})
        return {"found": True, "frames": {"frames": enriched}}

    @app.get("/api/library/frames/{key}/{filename}")
    def library_frame_image(key: str, filename: str) -> FileResponse:
        data = _read_library(DEFAULT_LIBRARY_INDEX)
        items = data.get("items", {})
        if not isinstance(items, dict):
            raise HTTPException(status_code=404, detail="Library item not found")
        item = items.get(key)
        if not isinstance(item, dict):
            raise HTTPException(status_code=404, detail="Library item not found")
        safe_name = Path(filename).name
        path = Path(str(item["output_dir"])) / "frames" / safe_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Frame image not found")
        return FileResponse(path, filename=safe_name)

    @app.get("/api/library/files/{key}/{kind}")
    def library_file(key: str, kind: str) -> FileResponse:
        data = _read_library(DEFAULT_LIBRARY_INDEX)
        items = data.get("items", {})
        if not isinstance(items, dict):
            raise HTTPException(status_code=404, detail="Library item not found")
        item = items.get(key)
        if not isinstance(item, dict):
            raise HTTPException(status_code=404, detail="Library item not found")
        filename = RESULT_FILES.get(kind)
        if filename is None:
            raise HTTPException(status_code=404, detail="Unknown result file")
        path = Path(str(item["output_dir"])) / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Result file not found")
        return FileResponse(path, filename=filename)

    @app.get("/api/jobs/{job_id}/files/{kind}")
    def job_file(job_id: str, kind: str) -> FileResponse:
        job = store.get(job_id)
        filename = RESULT_FILES.get(kind)
        if filename is None:
            raise HTTPException(status_code=404, detail="Unknown result file")
        path = job.output_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Result file not found")
        return FileResponse(path, filename=filename)

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FunASR-GGML Offline Video</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #20242c;
      --muted: #697383;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #b45309;
      --bad: #b91c1c;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 15px;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); }
    .status-pill {
      min-width: 112px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      background: var(--panel);
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 16px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .section-title {
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 700;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input[type="text"], input[type="file"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 284px;
      resize: vertical;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.45;
    }
    details {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
    }
    .checks {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 12px;
    }
    .checks label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      color: var(--text);
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 6px;
      padding: 12px 14px;
      margin-top: 16px;
      background: var(--accent);
      color: white;
      font-weight: 750;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .downloads {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .downloads a {
      display: block;
      text-align: center;
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 8px;
      color: var(--accent-strong);
      font-weight: 700;
      background: #f8fbfb;
    }
    .downloads a.disabled {
      color: #9aa3af;
      pointer-events: none;
      background: #f3f4f6;
    }
    pre {
      height: 382px;
      margin: 0;
      padding: 12px;
      overflow: auto;
      border-radius: 6px;
      border: 1px solid #1f2937;
      background: #111827;
      color: #d1fae5;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfd;
    }
    .metric b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 3px;
    }
    @media (max-width: 860px) {
      main { padding: 18px 12px 28px; }
      header { display: block; }
      .status-pill { margin-top: 12px; }
      .grid { grid-template-columns: 1fr; }
      .downloads { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FunASR-GGML 离线视频转写</h1>
        <div class="subtle">粘贴视频链接或上传本地文件，使用 long-video offline preset 生成字幕和文本。</div>
      </div>
      <div class="status-pill" id="status">idle</div>
    </header>

    <div class="grid">
      <section>
        <h2 class="section-title">输入</h2>
        <form id="jobForm">
          <label for="videoUrl">视频链接</label>
          <input id="videoUrl" name="url" type="text" placeholder="https://www.bilibili.com/video/BV..." />

          <label for="upload">本地视频/音频</label>
          <input id="upload" name="upload" type="file" />

          <details>
            <summary>设置</summary>
            <label for="model">模型路径</label>
            <input id="model" name="model" type="text" value="FunAsr_q8.bin" />

            <label for="funasrCli">funasr-cli 路径</label>
            <input id="funasrCli" name="funasr_cli" type="text" value="build-cuda/funasr-cli" />

            <label for="outputRoot">输出目录</label>
            <input id="outputRoot" name="output_root" type="text" value="outputs/video_asr_web" />

            <label for="cookiesPath">cookies.txt 路径</label>
            <input id="cookiesPath" name="cookies_path" type="text" value="cookies.txt" />

            <label for="cookies">yt-dlp cookies-from-browser</label>
            <input id="cookies" name="cookies_from_browser" type="text" placeholder="chrome / edge / firefox；cookies.txt 路径优先生效" />

            <div class="checks">
              <label><input name="use_gpu" type="checkbox" checked /> GPU</label>
              <label><input name="keep_media" type="checkbox" checked /> 保留媒体文件</label>
            </div>
          </details>

          <button id="submitBtn" type="submit">开始转写</button>
        </form>

        <h2 class="section-title" style="margin-top:18px">结果</h2>
        <div class="metrics">
          <div class="metric"><b>Job</b><span id="jobId">-</span></div>
          <div class="metric"><b>Elapsed</b><span id="elapsed">0s</span></div>
          <div class="metric"><b>Output</b><span id="outputDir">-</span></div>
        </div>
        <div class="downloads">
          <a id="fileSrt" class="disabled" href="#">SRT</a>
          <a id="fileTxt" class="disabled" href="#">TXT</a>
          <a id="fileJson" class="disabled" href="#">JSON</a>
          <a id="fileStats" class="disabled" href="#">Stats</a>
        </div>
        <label for="transcript">文本预览</label>
        <textarea id="transcript" readonly></textarea>
      </section>

      <section>
        <h2 class="section-title">日志</h2>
        <pre id="jobLog"></pre>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("jobForm");
    const button = document.getElementById("submitBtn");
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("jobLog");
    const transcriptEl = document.getElementById("transcript");
    const jobIdEl = document.getElementById("jobId");
    const elapsedEl = document.getElementById("elapsed");
    const outputDirEl = document.getElementById("outputDir");
    const links = {
      srt: document.getElementById("fileSrt"),
      txt: document.getElementById("fileTxt"),
      json: document.getElementById("fileJson"),
      stats: document.getElementById("fileStats"),
    };

    function setDownloads(files) {
      for (const [kind, el] of Object.entries(links)) {
        if (files && files[kind]) {
          el.href = files[kind];
          el.classList.remove("disabled");
        } else {
          el.href = "#";
          el.classList.add("disabled");
        }
      }
    }

    async function poll(jobId) {
      const res = await fetch(`/api/jobs/${jobId}`);
      const state = await res.json();
      statusEl.textContent = state.status;
      logEl.textContent = state.log || "";
      transcriptEl.value = state.transcript || "";
      elapsedEl.textContent = `${state.elapsed_sec}s`;
      outputDirEl.textContent = state.output_dir || "-";
      setDownloads(state.files);
      logEl.scrollTop = logEl.scrollHeight;

      if (state.status === "done" || state.status === "failed") {
        button.disabled = false;
        return;
      }
      setTimeout(() => poll(jobId), 1000);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      statusEl.textContent = "submitting";
      logEl.textContent = "";
      transcriptEl.value = "";
      setDownloads({});

      const data = new FormData(form);
      if (!document.getElementById("upload").files.length) {
        data.delete("upload");
      }
      if (!document.querySelector("input[name='use_gpu']").checked) {
        data.set("use_gpu", "false");
      }
      if (!document.querySelector("input[name='keep_media']").checked) {
        data.set("keep_media", "false");
      }

      const res = await fetch("/api/jobs", { method: "POST", body: data });
      if (!res.ok) {
        const body = await res.json();
        statusEl.textContent = "failed";
        logEl.textContent = body.detail || "Failed to create job.";
        button.disabled = false;
        return;
      }
      const body = await res.json();
      jobIdEl.textContent = body.job_id;
      poll(body.job_id);
    });
  </script>
</body>
</html>
"""


app = create_app()


def main() -> int:
    import uvicorn

    port = int(os.environ.get("FUNASR_WEB_APP_PORT", "8008"))
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
