#!/usr/bin/env python3
"""Shared result parsing and reporting helpers for ASR benchmarks."""

from __future__ import annotations

import csv
import json
import re
import statistics
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_FIELD_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_./-]+)=([^\s]+)")


def _fields(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _FIELD_RE.finditer(line)}


def _number(value: str) -> float:
    return float(value.rstrip("ms%"))


def _ratio(value: str) -> tuple[int, int]:
    left, right = value.split("/", 1)
    return int(left), int(right)


def parse_offline_log(text: str) -> dict[str, Any]:
    """Parse one completed ``test_offline_batching`` log.

    Raises ``ValueError`` when the run is incomplete or reports failed chunks.
    """

    result: dict[str, Any] = {"status": "failed"}
    saw_wall = False
    saw_throughput = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        fields = _fields(line)
        if line.startswith("[OfflineTest] ok="):
            success, total = _ratio(fields["ok"])
            result.update(success=success, total=total)
        elif line.startswith("[OfflineTest] wall total="):
            result["wall_sec"] = _number(fields["total"]) / 1000.0
            result["rtf"] = _number(fields["rtf"])
            saw_wall = True
        elif line.startswith("[OfflineTest] scheduler:"):
            result["avg_active"] = _number(fields["avg_active"])
            result["fallback_calls"] = int(fields["fallback_calls"])
            peak, capacity = _ratio(fields["blocks_peak"])
            result["peak_blocks"] = peak
            result["block_capacity"] = capacity
        elif line.startswith("[OfflineTest] paged_kv_runtime:"):
            result["ownership_errors"] = int(fields["ownership_errors"])
        elif line.startswith("[OfflineTest] scheduler_profile:"):
            result["prefill_sec"] = _number(fields["prefill_wall"]) / 1000.0
            result["decode_sec"] = _number(fields["decode_dispatch"]) / 1000.0
        elif line.startswith("[OfflineTest] throughput:"):
            result["audio_sec"] = _number(fields["audio_sec"])
            result["rtfx"] = _number(fields["audio_sec/s"])
            result["tokens_s"] = _number(fields["tokens/s"])
            saw_throughput = True
        elif line.startswith("[OfflineTest] paged_graph_cache_probe:"):
            result["graph_hit_rate"] = _number(fields["cache_hit_rate"])

    if "success" not in result:
        raise ValueError("offline benchmark log has no completion line")
    if result["success"] != result["total"]:
        raise ValueError(
            f"offline benchmark completed only {result['success']}/{result['total']} chunks"
        )
    if not saw_wall or not saw_throughput:
        raise ValueError("offline benchmark log is missing wall or throughput metrics")

    result["status"] = "ok"
    return result


def split_wav_fixed(
    source: str | Path,
    output_dir: str | Path,
    chunk_sec: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Split a PCM WAV into exact fixed-size windows and return a manifest."""

    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    source_path = Path(source)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        if params.comptype != "NONE":
            raise ValueError(f"compressed WAV is not supported: {params.comptype}")
        frames_per_chunk = max(1, int(round(params.framerate * chunk_sec)))
        frame_offset = 0
        chunk_index = 0
        while limit is None or chunk_index < limit:
            payload = src.readframes(frames_per_chunk)
            frame_count = len(payload) // (params.nchannels * params.sampwidth)
            if frame_count == 0:
                break

            chunk_id = f"chunk_{chunk_index:04d}"
            chunk_path = target_dir / f"{chunk_id}.wav"
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(payload)

            start_sec = frame_offset / params.framerate
            duration_sec = frame_count / params.framerate
            chunks.append(
                {
                    "id": chunk_id,
                    "path": str(chunk_path),
                    "frames": frame_count,
                    "start_sec": start_sec,
                    "end_sec": start_sec + duration_sec,
                    "duration_sec": duration_sec,
                    "sample_rate": params.framerate,
                    "channels": params.nchannels,
                    "sample_width": params.sampwidth,
                }
            )
            frame_offset += frame_count
            chunk_index += 1

    return chunks


_MEDIAN_FIELDS = (
    "wall_sec",
    "rtf",
    "rtfx",
    "tokens_s",
    "peak_vram_mib",
    "prefill_sec",
    "decode_sec",
    "cer",
    "avg_active",
    "fallback_calls",
    "peak_blocks",
    "graph_hit_rate",
)


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured records by engine while retaining skipped engines."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["engine"])].append(record)

    summaries: list[dict[str, Any]] = []
    for engine in sorted(grouped):
        engine_records = grouped[engine]
        successful = [record for record in engine_records if record.get("status") == "ok"]
        if not successful:
            first = engine_records[0]
            summaries.append(
                {
                    "engine": engine,
                    "status": first.get("status", "failed"),
                    "reason": first.get("reason", "no successful measurements"),
                    "measured_repeats": 0,
                    "model": first.get("model", ""),
                    "precision": first.get("precision", ""),
                }
            )
            continue

        first = successful[0]
        summary: dict[str, Any] = {
            "engine": engine,
            "status": "ok",
            "reason": "",
            "measured_repeats": len(successful),
            "model": first.get("model", ""),
            "precision": first.get("precision", ""),
            "success": first.get("success", ""),
            "total": first.get("total", ""),
        }
        for field in _MEDIAN_FIELDS:
            values = [
                float(record[field])
                for record in successful
                if record.get(field) not in (None, "")
            ]
            if values:
                summary[f"median_{field}"] = statistics.median(values)
                if field == "wall_sec":
                    summary["min_wall_sec"] = min(values)
                    summary["max_wall_sec"] = max(values)
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "engine", "repeat", "status", "reason", "model", "precision",
        "wall_sec", "rtf", "rtfx", "cer", "peak_vram_mib", "success", "total",
    ]
    all_fields = {key for row in rows for key in row}
    fields = [field for field in preferred if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR Benchmark Summary",
        "",
        "| Engine | Status | Model | Precision | Repeats | Wall median (s) | Wall range (s) | RTFx | RTF | CER (%) | Peak VRAM (MiB) | Success | Reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        wall_range = "-"
        if item.get("min_wall_sec") is not None:
            wall_range = f"{_fmt(item['min_wall_sec'])}-{_fmt(item['max_wall_sec'])}"
        success = "-"
        if item.get("success") not in (None, "") and item.get("total") not in (None, ""):
            success = f"{item['success']}/{item['total']}"
        lines.append(
            "| {engine} | {status} | {model} | {precision} | {repeats} | {wall} | "
            "{wall_range} | {rtfx} | {rtf} | {cer} | {vram} | {success} | {reason} |".format(
                engine=item["engine"],
                status=item["status"],
                model=item.get("model", "-") or "-",
                precision=item.get("precision", "-") or "-",
                repeats=item.get("measured_repeats", 0),
                wall=_fmt(item.get("median_wall_sec"), 3),
                wall_range=wall_range,
                rtfx=_fmt(item.get("median_rtfx")),
                rtf=_fmt(item.get("median_rtf"), 4),
                cer=_fmt(item.get("median_cer")),
                vram=_fmt(item.get("median_peak_vram_mib"), 0),
                success=success,
                reason=str(item.get("reason", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "RTFx is audio seconds divided by measured inference wall seconds. Model loading is reported separately and is not included unless explicitly stated in metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write raw records, aggregate summary, Markdown, and run metadata."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_records(records)
    records_path = target / "records.csv"
    summary_path = target / "summary.csv"
    markdown_path = target / "summary.md"
    metadata_path = target / "metadata.json"

    _write_csv(records_path, records)
    _write_csv(summary_path, summaries)
    markdown_path.write_text(_summary_markdown(summaries), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "records": str(records_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "metadata": str(metadata_path),
    }
