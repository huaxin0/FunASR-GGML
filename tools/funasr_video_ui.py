#!/usr/bin/env python3
"""
Small terminal UI for transcribing video URLs with FunASR-GGML.

Pipeline:
  URL or local media -> yt-dlp/ffmpeg -> 16 kHz mono wav -> funasr-cli -> srt/txt/json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    source: str
    model: Path
    funasr_cli: Path
    output_dir: Path
    use_gpu: bool
    use_vad: bool
    chunk_mode: str
    chunk_sec: str
    vad_model: Path | None
    vad_threshold: str
    vad_min_silence_ms: str
    vad_speech_pad_ms: str
    vad_max_speech_sec: str
    ctx_size: str
    max_tokens: str
    srt_max_chars: str
    offline_scheduler: bool
    offline_profile: bool
    offline_batch_size: str
    offline_kv_mode: str
    offline_kv_block_size: str
    keep_media: bool
    offline_preset: str = ""


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    chunk_mode: str
    chunk_sec: str
    vad_min_silence_ms: str
    vad_speech_pad_ms: str
    vad_max_speech_sec: str
    ctx_size: str
    max_tokens: str
    srt_max_chars: str
    offline_scheduler: bool
    offline_profile: bool
    offline_batch_size: str
    offline_kv_mode: str
    offline_kv_block_size: str


PRESETS = {
    "1": Preset(
        name="long",
        description="fixed 30s windows, ctx 4096, offline paged batch",
        chunk_mode="window",
        chunk_sec="30",
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
    ),
    "2": Preset(
        name="balanced",
        description="fixed 20s windows, ctx 4096",
        chunk_mode="window",
        chunk_sec="20",
        vad_min_silence_ms="600",
        vad_speech_pad_ms="80",
        vad_max_speech_sec="15",
        ctx_size="4096",
        max_tokens="160",
        srt_max_chars="24",
        offline_scheduler=False,
        offline_profile=False,
        offline_batch_size="12",
        offline_kv_mode="continuous",
        offline_kv_block_size="64",
    ),
    "3": Preset(
        name="vad",
        description="Silero VAD chunks, skips silence",
        chunk_mode="vad",
        chunk_sec="15",
        vad_min_silence_ms="500",
        vad_speech_pad_ms="80",
        vad_max_speech_sec="10",
        ctx_size="2048",
        max_tokens="120",
        srt_max_chars="20",
        offline_scheduler=False,
        offline_profile=False,
        offline_batch_size="12",
        offline_kv_mode="continuous",
        offline_kv_block_size="64",
    ),
}


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_header() -> None:
    print("=" * 66)
    print(" FunASR-GGML Video Transcriber")
    print("=" * 66)
    print("Paste a Bilibili / YouTube / Douyin URL, or a local media path.")
    print("This tool only prepares media and calls funasr-cli; core ASR code is untouched.")
    print()


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [Enter = {default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    if not value:
        return default or ""
    if default and value.lower() in {"y", "yes"}:
        return default
    return value


def prompt_bool(text: str, default: bool) -> bool:
    label = "Y/n" if default else "y/N"
    while True:
        value = input(f"{text} [{label}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "1", "true"}:
            return True
        if value in {"n", "no", "0", "false"}:
            return False
        print("Please enter y or n.")


def prompt_preset() -> Preset:
    print()
    print("Preset:")
    for key, preset in PRESETS.items():
        print(f"  {key}. {preset.name:<8} - {preset.description}")
    while True:
        value = input("Choose preset [Enter = 1]: ").strip()
        if not value:
            value = "1"
        preset = PRESETS.get(value)
        if preset:
            return preset
        print("Please enter 1, 2, or 3.")


def prompt_chunk_mode(default: str) -> str:
    while True:
        value = prompt("Chunk mode (none/window/vad)", default).lower()
        if value in {"none", "window", "vad"}:
            return value
        print("Please enter none, window, or vad.")


def prompt_offline_kv_mode(default: str) -> str:
    while True:
        value = prompt("Offline KV mode (continuous/paged)", default).lower()
        if value in {"continuous", "paged"}:
            return value
        print("Please enter continuous or paged.")


def prompt_positive_int_text(text: str, default: str) -> str:
    while True:
        value = prompt(text, default)
        if value.isdigit() and int(value) > 0:
            return value
        print("Please enter a positive integer.")


def find_first(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def default_funasr_cli() -> Path:
    found = find_first(
        [
            ROOT / "build-cuda" / "funasr-cli",
            ROOT / "build-cpu" / "funasr-cli",
            ROOT / "build" / "funasr-cli",
        ]
    )
    return found or (ROOT / "build-cuda" / "funasr-cli")


def default_model() -> Path:
    return ROOT / "FunAsr_q8.bin"


def default_vad_model() -> Path | None:
    found = find_first(
        [
            ROOT / "ggml-silero-v6.2.0.bin",
            ROOT / "models" / "ggml-silero-v6.2.0.bin",
        ]
    )
    return found


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print()
    print("$ " + " ".join(quote_arg(x) for x in cmd))
    print(flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def quote_arg(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@%+-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def safe_slug(value: str, fallback: str = "video") -> str:
    value = value.strip()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    if not value:
        return fallback
    return value[:80]


def collect_config() -> Config:
    clear_screen()
    print_header()

    source = prompt("Video URL or local media path")
    if not source:
        raise SystemExit("No source provided.")

    model = Path(prompt("Model path", str(default_model()))).expanduser()
    funasr_cli = Path(prompt("funasr-cli path", str(default_funasr_cli()))).expanduser()
    default_out = ROOT / "outputs" / "video_asr" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(prompt("Output directory", str(default_out))).expanduser()

    use_gpu = prompt_bool("Use GPU", True)
    preset = prompt_preset()
    chunk_mode = prompt_chunk_mode(preset.chunk_mode)
    chunk_sec = preset.chunk_sec
    if chunk_mode == "window":
        chunk_sec = prompt("Window chunk seconds", chunk_sec)
    use_vad = chunk_mode == "vad"

    vad_model: Path | None = None
    vad_threshold = "0.45"
    vad_min_silence_ms = preset.vad_min_silence_ms
    vad_speech_pad_ms = preset.vad_speech_pad_ms
    vad_max_speech_sec = preset.vad_max_speech_sec
    if use_vad:
        vad_default = default_vad_model()
        vad_value = prompt(
            "Silero VAD model path (empty = energy VAD)",
            str(vad_default) if vad_default else "",
        )
        vad_model = Path(vad_value).expanduser() if vad_value else None
        vad_threshold = prompt("VAD threshold", vad_threshold)
        vad_min_silence_ms = prompt("VAD min silence ms", vad_min_silence_ms)
        vad_speech_pad_ms = prompt("VAD speech pad ms", vad_speech_pad_ms)
        vad_max_speech_sec = prompt("VAD max speech sec", vad_max_speech_sec)

    offline_scheduler = prompt_bool("Use offline scheduler", preset.offline_scheduler)
    offline_profile = False
    offline_batch_size = preset.offline_batch_size
    offline_kv_mode = preset.offline_kv_mode
    offline_kv_block_size = preset.offline_kv_block_size
    if offline_scheduler:
        offline_profile = prompt_bool("Print offline profile", preset.offline_profile)
        offline_batch_size = prompt_positive_int_text("Offline batch size", offline_batch_size)
        offline_kv_mode = prompt_offline_kv_mode(offline_kv_mode)
        offline_kv_block_size = prompt_positive_int_text(
            "Offline KV block size",
            offline_kv_block_size,
        )

    return Config(
        source=source,
        model=model,
        funasr_cli=funasr_cli,
        output_dir=output_dir,
        use_gpu=use_gpu,
        use_vad=use_vad,
        chunk_mode=chunk_mode,
        chunk_sec=chunk_sec,
        vad_model=vad_model,
        vad_threshold=vad_threshold,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        vad_max_speech_sec=vad_max_speech_sec,
        ctx_size=prompt("LLM ctx size", preset.ctx_size),
        max_tokens=prompt("Max generated tokens per chunk", preset.max_tokens),
        srt_max_chars=prompt("SRT max UTF-8 chars", preset.srt_max_chars),
        offline_scheduler=offline_scheduler,
        offline_profile=offline_profile,
        offline_batch_size=offline_batch_size,
        offline_kv_mode=offline_kv_mode,
        offline_kv_block_size=offline_kv_block_size,
        keep_media=prompt_bool("Keep downloaded/extracted media", True),
        offline_preset="",
    )


def validate_config(cfg: Config) -> None:
    missing = []
    if not cfg.model.exists():
        missing.append(f"model not found: {cfg.model}")
    if not cfg.funasr_cli.exists():
        missing.append(f"funasr-cli not found: {cfg.funasr_cli}")
    if cfg.vad_model is not None and not cfg.vad_model.exists():
        missing.append(f"VAD model not found: {cfg.vad_model}")
    if not command_exists("ffmpeg"):
        missing.append("ffmpeg not found in PATH")
    if is_url(cfg.source) and not command_exists("yt-dlp"):
        missing.append("yt-dlp not found in PATH")

    if missing:
        print()
        print("Cannot start:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def download_source(cfg: Config, work_dir: Path) -> Path:
    if not is_url(cfg.source):
        local = Path(cfg.source).expanduser()
        if not local.exists():
            raise SystemExit(f"Local media not found: {local}")
        return local

    download_dir = work_dir / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    template = str(download_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "-o",
        template,
        cfg.source,
    ]
    run_command(cmd)

    candidates = sorted(p for p in download_dir.iterdir() if p.is_file())
    if not candidates:
        raise SystemExit("yt-dlp finished but no media file was found.")
    return candidates[0]


def extract_wav(media_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    run_command(cmd)


def run_asr(cfg: Config, wav_path: Path, srt_path: Path) -> None:
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
    if cfg.chunk_mode == "vad":
        if cfg.vad_model is not None:
            cmd.extend(
                [
                    "--vad-model",
                    str(cfg.vad_model),
                    "--vad-threshold",
                    cfg.vad_threshold,
                    "--vad-min-silence-ms",
                    cfg.vad_min_silence_ms,
                    "--vad-speech-pad-ms",
                    cfg.vad_speech_pad_ms,
                    "--vad-max-speech-sec",
                    cfg.vad_max_speech_sec,
                ]
            )
    if cfg.offline_preset:
        cmd.extend(["--offline-preset", cfg.offline_preset])
    elif cfg.offline_scheduler:
        cmd.extend(
            [
                "--offline-scheduler",
                "--batch-size",
                cfg.offline_batch_size,
                "--kv-mode",
                cfg.offline_kv_mode,
                "--kv-block-size",
                cfg.offline_kv_block_size,
            ]
        )
        if cfg.offline_profile:
            cmd.append("--offline-profile")
    run_command(cmd)


def parse_srt_time(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = [int(x) for x in match.groups()]
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_srt(srt_path: Path) -> list[dict[str, object]]:
    if not srt_path.exists():
        return []

    blocks = re.split(r"\n\s*\n", srt_path.read_text(encoding="utf-8", errors="ignore").strip())
    items: list[dict[str, object]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
        text = " ".join(lines[2:]).strip()
        items.append(
            {
                "index": len(items) + 1,
                "start": parse_srt_time(start_raw),
                "end": parse_srt_time(end_raw),
                "text": text,
            }
        )
    return items


def write_side_outputs(
    cfg: Config,
    srt_path: Path,
    elapsed_sec: float,
    source_info: dict[str, object] | None = None,
) -> None:
    source_info = source_info or {"transcript_source": "asr"}
    segments = parse_srt(srt_path)
    txt_path = cfg.output_dir / "transcript.txt"
    json_path = cfg.output_dir / "transcript.json"
    stats_path = cfg.output_dir / "stats.json"

    text = "\n".join(str(item["text"]) for item in segments)
    txt_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "source": cfg.source,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    duration = max((float(item["end"]) for item in segments), default=0.0)
    stats_path.write_text(
        json.dumps(
            {
                "source": cfg.source,
                "elapsed_sec": round(elapsed_sec, 3),
                "audio_duration_sec": round(duration, 3),
                "rtf": round(elapsed_sec / duration, 4) if duration > 0 else None,
                "segments": len(segments),
                "gpu": cfg.use_gpu,
                "chunk_mode": cfg.chunk_mode,
                "chunk_sec": cfg.chunk_sec if cfg.chunk_mode == "window" else None,
                "vad": cfg.use_vad,
                "vad_model": str(cfg.vad_model) if cfg.vad_model else None,
                "offline_scheduler": cfg.offline_scheduler,
                "offline_profile": cfg.offline_profile,
                "offline_batch_size": int(cfg.offline_batch_size)
                if cfg.offline_scheduler
                else None,
                "offline_kv_mode": cfg.offline_kv_mode if cfg.offline_scheduler else None,
                "offline_kv_block_size": int(cfg.offline_kv_block_size)
                if cfg.offline_scheduler
                else None,
                "transcript_source": source_info.get("transcript_source", "asr"),
                "subtitle_language": source_info.get("subtitle_language"),
                "subtitle_kind": source_info.get("subtitle_kind"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_run_config(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "source": cfg.source,
        "model": str(cfg.model),
        "funasr_cli": str(cfg.funasr_cli),
        "use_gpu": cfg.use_gpu,
        "chunk_mode": cfg.chunk_mode,
        "chunk_sec": cfg.chunk_sec,
        "use_vad": cfg.use_vad,
        "vad_model": str(cfg.vad_model) if cfg.vad_model else None,
        "vad_threshold": cfg.vad_threshold,
        "vad_min_silence_ms": cfg.vad_min_silence_ms,
        "vad_speech_pad_ms": cfg.vad_speech_pad_ms,
        "vad_max_speech_sec": cfg.vad_max_speech_sec,
        "ctx_size": cfg.ctx_size,
        "max_tokens": cfg.max_tokens,
        "srt_max_chars": cfg.srt_max_chars,
        "offline_scheduler": cfg.offline_scheduler,
        "offline_profile": cfg.offline_profile,
        "offline_batch_size": cfg.offline_batch_size,
        "offline_kv_mode": cfg.offline_kv_mode,
        "offline_kv_block_size": cfg.offline_kv_block_size,
        "offline_preset": cfg.offline_preset,
        "keep_media": cfg.keep_media,
    }
    (cfg.output_dir / "run_config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (cfg.output_dir / "source.txt").write_text(cfg.source + "\n", encoding="utf-8")


def main() -> int:
    cfg = collect_config()
    validate_config(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = cfg.output_dir / "media"
    work_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(cfg)

    print()
    print("=" * 66)
    print("Starting")
    print("=" * 66)
    print(f"Output: {cfg.output_dir}", flush=True)

    started = time.monotonic()
    media_path = download_source(cfg, work_dir)
    wav_path = work_dir / f"{safe_slug(Path(media_path).stem)}_16k.wav"
    extract_wav(media_path, wav_path)

    srt_path = cfg.output_dir / "transcript.srt"
    run_asr(cfg, wav_path, srt_path)
    elapsed = time.monotonic() - started
    write_side_outputs(cfg, srt_path, elapsed)

    if not cfg.keep_media:
        shutil.rmtree(work_dir, ignore_errors=True)

    print()
    print("=" * 66)
    print("Done")
    print("=" * 66)
    print(f"SRT:   {cfg.output_dir / 'transcript.srt'}")
    print(f"Text:  {cfg.output_dir / 'transcript.txt'}")
    print(f"JSON:  {cfg.output_dir / 'transcript.json'}")
    print(f"Stats: {cfg.output_dir / 'stats.json'}")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
