#!/usr/bin/env python3
"""
Local Gradio Web UI for FunASR-GGML offline video transcription.

The UI is intentionally thin:
  URL or uploaded media -> existing video helper -> funasr-cli long-video preset
"""

from __future__ import annotations

import shutil
import time
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # Tests and non-UI imports should not require Gradio.
    gr = None

try:
    from tools import funasr_video_ui as video_ui
except ModuleNotFoundError:
    import funasr_video_ui as video_ui


ROOT = video_ui.ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "video_asr_web"


@dataclass
class SourceSelection:
    ok: bool
    source: str
    is_upload: bool
    message: str


def select_source(url: str, uploaded_path: str | None, job_dir: Path) -> SourceSelection:
    url = (url or "").strip()
    uploaded_path = (uploaded_path or "").strip()

    if uploaded_path:
        uploaded = Path(uploaded_path)
        if not uploaded.exists():
            return SourceSelection(False, "", False, f"Uploaded file not found: {uploaded}")

        upload_dir = job_dir / "media" / "upload"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / uploaded.name
        shutil.copy2(uploaded, dest)
        return SourceSelection(True, str(dest), True, "Using uploaded media.")

    if url:
        return SourceSelection(True, url, False, "Using URL source.")

    return SourceSelection(False, "", False, "Paste a URL or upload a local media file.")


def build_web_config(
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


def _new_job_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix}"
        suffix += 1
    return candidate


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def transcribe_job(
    url: str,
    uploaded_file: str | None,
    model: str,
    funasr_cli: str,
    output_root: str,
    use_gpu: bool,
    keep_media: bool,
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    job_dir = _new_job_dir(output_root or DEFAULT_OUTPUT_ROOT)
    selection = select_source(url, uploaded_file, job_dir)
    if not selection.ok:
        return selection.message, "", None, None, None, None

    cfg = build_web_config(
        source=selection.source,
        model=model or str(video_ui.default_model()),
        funasr_cli=funasr_cli or str(video_ui.default_funasr_cli()),
        output_dir=job_dir,
        use_gpu=use_gpu,
        keep_media=keep_media,
    )

    try:
        video_ui.validate_config(cfg)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = cfg.output_dir / "media"
        work_dir.mkdir(parents=True, exist_ok=True)
        video_ui.write_run_config(cfg)

        started = time.monotonic()
        media_path = video_ui.download_source(cfg, work_dir)
        wav_path = work_dir / f"{video_ui.safe_slug(Path(media_path).stem)}_16k.wav"
        video_ui.extract_wav(media_path, wav_path)

        srt_path = cfg.output_dir / "transcript.srt"
        video_ui.run_asr(cfg, wav_path, srt_path)
        elapsed = time.monotonic() - started
        video_ui.write_side_outputs(cfg, srt_path, elapsed)

        if not cfg.keep_media:
            shutil.rmtree(work_dir, ignore_errors=True)

        txt_path = cfg.output_dir / "transcript.txt"
        json_path = cfg.output_dir / "transcript.json"
        stats_path = cfg.output_dir / "stats.json"
        status = f"Done. Output: {cfg.output_dir}"
        return status, _read_text(txt_path), str(srt_path), str(txt_path), str(json_path), str(stats_path)
    except Exception as exc:
        return f"Failed: {exc}", "", None, None, None, None


def build_app():
    if gr is None:
        raise RuntimeError("Gradio is not installed. Run: pip install gradio")

    with gr.Blocks(title="FunASR-GGML Offline Video") as app:
        gr.Markdown("# FunASR-GGML Offline Video")
        gr.Markdown("Paste a video URL or upload local media, then run the long-video offline preset.")

        with gr.Row():
            url = gr.Textbox(label="Video URL", placeholder="Bilibili / YouTube / Douyin URL")
            upload = gr.File(label="Local video/audio", type="filepath")

        with gr.Accordion("Settings", open=False):
            model = gr.Textbox(label="Model path", value=str(video_ui.default_model()))
            funasr_cli = gr.Textbox(label="funasr-cli path", value=str(video_ui.default_funasr_cli()))
            output_root = gr.Textbox(label="Output root", value=str(DEFAULT_OUTPUT_ROOT))
            use_gpu = gr.Checkbox(label="Use GPU", value=True)
            keep_media = gr.Checkbox(label="Keep downloaded/extracted media", value=True)

        run = gr.Button("Transcribe", variant="primary")
        status = gr.Textbox(label="Status", lines=3)
        transcript = gr.Textbox(label="Transcript preview", lines=16)

        with gr.Row():
            srt_file = gr.File(label="SRT")
            txt_file = gr.File(label="TXT")
            json_file = gr.File(label="JSON")
            stats_file = gr.File(label="Stats")

        run.click(
            fn=transcribe_job,
            inputs=[url, upload, model, funasr_cli, output_root, use_gpu, keep_media],
            outputs=[status, transcript, srt_file, txt_file, json_file, stats_file],
        )

    return app


def main() -> int:
    app = build_app()
    port_value = os.environ.get("FUNASR_WEB_UI_PORT", "").strip()
    server_port = int(port_value) if port_value else None
    app.launch(server_name="127.0.0.1", server_port=server_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
