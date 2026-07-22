#!/usr/bin/env python3
"""Run reproducible local and optional external Fun-ASR benchmark engines."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_common import parse_offline_log, split_wav_fixed, write_reports


CPP_ENGINES = {"cpp-b1", "cpp-b12"}
PYTHON_ENGINES = {"funasr-pytorch", "funasr-vllm", "sherpa-onnx"}
SUPPORTED_ENGINES = CPP_ENGINES | PYTHON_ENGINES


def build_cpp_command(
    engine: str, args: argparse.Namespace
) -> tuple[dict[str, str], list[str]]:
    if engine not in CPP_ENGINES:
        raise ValueError(f"unsupported C++ engine: {engine}")

    env = {
        "FUNASR_PAGED_KV_WRITE_OP": "1",
        "FUNASR_PAGED_DECODE_BUCKET_MAX_KV": "1",
        "FUNASR_PAGED_DECODE_GRAPH_CACHE": "1",
    }
    batch_size = 1 if engine == "cpp-b1" else 12
    kv_mode = "continuous" if engine == "cpp-b1" else "paged"
    dynamic_blocks = "off" if engine == "cpp-b1" else "on"
    command = [
        str(args.binary),
        str(args.model),
        str(args.audio),
        "--gpu",
        "--kv-mode", kv_mode,
        "--batch-size", str(batch_size),
        "--ctx-size", str(args.ctx_size),
        "--kv-block-size", str(args.block_size),
        "--chunk-mode", "window",
        "--chunk-sec", str(args.chunk_sec),
        "--max-tokens", str(args.max_tokens),
        "--threads", str(args.threads),
        "--prefix-kv-cache", "off",
        "--dynamic-kv-blocks", dynamic_blocks,
    ]
    if args.gpu_id > 0:
        command.extend(["--gpu-id", str(args.gpu_id)])
    if args.max_chunks and args.max_chunks > 0:
        command.extend(["--max-chunks", str(args.max_chunks)])
    return env, command


def build_python_engine_command(
    engine: str,
    args: argparse.Namespace,
    wav_dir: Path,
    result_json: Path,
    text_output_dir: Path,
) -> list[str]:
    if engine not in PYTHON_ENGINES:
        raise ValueError(f"unsupported Python engine: {engine}")
    python_by_engine = {
        "funasr-pytorch": args.pytorch_python,
        "funasr-vllm": args.vllm_python,
        "sherpa-onnx": args.sherpa_python,
    }
    command = [
        str(python_by_engine[engine]),
        str(ROOT / "tools/benchmark_python_engines.py"),
        "--engine", engine,
        "--wav-dir", str(wav_dir),
        "--result-json", str(result_json),
        "--text-output-dir", str(text_output_dir),
        "--model", args.official_model,
        "--hub", args.official_hub,
        "--repeat", str(args.repeat),
        "--warmup", str(args.warmup),
        "--max-tokens", str(args.max_tokens),
        "--gpu-id", str(args.gpu_id),
        "--threads", str(args.threads),
        "--batch-size-s", str(args.official_batch_size_s),
        "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--sherpa-provider", args.sherpa_provider,
        "--sherpa-batch-size", str(args.sherpa_batch_size),
    ]
    if args.sherpa_model_dir is not None:
        command.extend(["--sherpa-model-dir", str(args.sherpa_model_dir)])
    return command


class GpuMemorySampler:
    """Poll total device memory use while a subprocess is running."""

    def __init__(self, gpu_id: int, interval_sec: float = 0.1):
        self.gpu_id = gpu_id
        self.interval_sec = interval_sec
        self.peak_mib: float | None = None
        self.baseline_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> float | None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return float(output.strip().splitlines()[0])
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_sec):
            value = self._read()
            if value is not None and (self.peak_mib is None or value > self.peak_mib):
                self.peak_mib = value

    def start(self) -> None:
        self.baseline_mib = self._read()
        self.peak_mib = self.baseline_mib
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def run_logged(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    gpu_id: int,
    quiet: bool,
) -> tuple[int, float | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = GpuMemorySampler(gpu_id)
    sampler.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if not quiet:
                    print(line, end="", flush=True)
            return_code = process.wait()
    finally:
        sampler.stop()
    return return_code, sampler.peak_mib


def run_cpp_engine(
    engine: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    env, command = build_cpp_command(engine, args)
    records: list[dict[str, Any]] = []
    total_runs = args.warmup + args.repeat
    for run_index in range(total_runs):
        is_warmup = run_index < args.warmup
        repeat_id = run_index - args.warmup + 1
        label = f"warmup{run_index + 1}" if is_warmup else f"r{repeat_id}"
        log_path = output_dir / "logs" / f"{engine}_{label}.log"
        print(f"\n=== {engine} {label} ===")
        print("$ " + " ".join(shlex.quote(part) for part in command))
        return_code, peak_vram = run_logged(
            command, env, log_path, args.gpu_id, args.quiet
        )
        if is_warmup:
            if return_code != 0:
                raise RuntimeError(f"{engine} warmup failed; see {log_path}")
            continue

        record: dict[str, Any] = {
            "engine": engine,
            "repeat": repeat_id,
            "status": "failed",
            "model": "Fun-ASR-Nano-2512",
            "precision": "Q8_0",
            "peak_vram_mib": peak_vram,
            "log": str(log_path),
        }
        if return_code != 0:
            record["reason"] = f"process exited with code {return_code}"
        else:
            try:
                record.update(parse_offline_log(log_path.read_text(encoding="utf-8")))
            except ValueError as exc:
                record["reason"] = str(exc)
        records.append(record)
    return records


def run_python_engine(
    engine: str,
    args: argparse.Namespace,
    output_dir: Path,
    wav_dir: Path,
) -> list[dict[str, Any]]:
    result_json = output_dir / "engine_results" / f"{engine}.json"
    text_output_dir = output_dir / "texts"
    log_path = output_dir / "logs" / f"{engine}.log"
    command = build_python_engine_command(
        engine, args, wav_dir, result_json, text_output_dir
    )
    print(f"\n=== {engine} ===")
    print("$ " + " ".join(shlex.quote(part) for part in command))
    try:
        return_code, peak_vram = run_logged(
            command, {}, log_path, args.gpu_id, args.quiet
        )
    except OSError as exc:
        return [
            {
                "engine": engine,
                "repeat": 0,
                "status": "skipped",
                "model": "Fun-ASR-Nano-2512",
                "precision": "INT8 ONNX" if engine == "sherpa-onnx" else "framework-default",
                "reason": f"optional Python executable unavailable: {exc}",
                "log": str(log_path),
            }
        ]
    if not result_json.is_file():
        return [
            {
                "engine": engine,
                "repeat": 0,
                "status": "failed",
                "model": "Fun-ASR-Nano-2512",
                "precision": "framework-default",
                "reason": f"runner exited with code {return_code} without result JSON",
                "peak_vram_mib": peak_vram,
                "log": str(log_path),
            }
        ]

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    payload_records = payload.get("records", [])
    if not payload_records:
        return [
            {
                "engine": engine,
                "repeat": 0,
                "status": payload.get("status", "failed"),
                "model": "Fun-ASR-Nano-2512",
                "precision": "INT8 ONNX" if engine == "sherpa-onnx" else "framework-default",
                "reason": payload.get("reason", f"runner exited with code {return_code}"),
                "peak_vram_mib": peak_vram,
                "log": str(log_path),
            }
        ]

    records: list[dict[str, Any]] = []
    for item in payload_records:
        record = dict(item)
        record["peak_vram_mib"] = peak_vram
        record["log"] = str(log_path)
        records.append(record)
    return records


def gpu_metadata(gpu_id: int) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        name, memory, driver = [item.strip() for item in output.split(",", 2)]
        return {"name": name, "memory_total_mib": float(memory), "driver": driver}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {"name": "unknown"}


def git_metadata() -> dict[str, Any]:
    def run(*parts: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *parts], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.SubprocessError:
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def parse_engine_list(value: str) -> list[str]:
    engines = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(engines) - SUPPORTED_ENGINES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown engines: {', '.join(unknown)}")
    if not engines:
        raise argparse.ArgumentTypeError("at least one engine is required")
    return engines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local and optional Fun-ASR engines with one result schema."
    )
    parser.add_argument("--model", type=Path, required=True, help="Local Q8 GGUF model")
    parser.add_argument("--audio", type=Path, required=True, help="16 kHz PCM WAV workload")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build-cuda/test_offline_batching"
    )
    parser.add_argument(
        "--engines", type=parse_engine_list, default=["cpp-b1", "cpp-b12"],
        help="Comma-separated: cpp-b1,cpp-b12,funasr-pytorch,funasr-vllm,sherpa-onnx",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--chunk-sec", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/benchmark_suite")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pytorch-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vllm-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sherpa-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--official-model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--official-hub", choices=["hf", "ms"], default="hf")
    parser.add_argument("--official-batch-size-s", type=float, default=300.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--sherpa-model-dir", type=Path)
    parser.add_argument("--sherpa-provider", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sherpa-batch-size", type=int, default=12)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup must be >= 0")
    return args


def validate_local_inputs(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")
    if not args.audio.is_file():
        raise FileNotFoundError(f"audio not found: {args.audio}")
    if any(engine in CPP_ENGINES for engine in args.engines) and not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {args.binary}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build:
        subprocess.run(
            ["cmake", "--build", str(ROOT / "build-cuda"), "--target", "test_offline_batching", "-j2"],
            cwd=ROOT,
            check=True,
        )
    validate_local_inputs(args)

    if args.dry_run:
        for engine in args.engines:
            if engine in CPP_ENGINES:
                env, command = build_cpp_command(engine, args)
                prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
                print(f"[{engine}] {prefix} " + " ".join(shlex.quote(part) for part in command))
            else:
                command = build_python_engine_command(
                    engine,
                    args,
                    Path("<fixed-window-wavs>"),
                    Path(f"<output>/{engine}.json"),
                    Path("<output>/texts"),
                )
                print(f"[{engine}] " + " ".join(shlex.quote(part) for part in command))
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    wav_dir: Path | None = None
    workload_manifest: list[dict[str, Any]] = []
    if any(engine in PYTHON_ENGINES for engine in args.engines):
        wav_dir = output_dir / "workload" / "fixed_windows"
        limit = args.max_chunks if args.max_chunks > 0 else None
        workload_manifest = split_wav_fixed(
            args.audio, wav_dir, args.chunk_sec, limit=limit
        )
        manifest_path = output_dir / "workload" / "manifest.json"
        manifest_path.write_text(
            json.dumps(workload_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    records: list[dict[str, Any]] = []
    for engine in args.engines:
        if engine in CPP_ENGINES:
            records.extend(run_cpp_engine(engine, args, output_dir))
        else:
            assert wav_dir is not None
            records.extend(run_python_engine(engine, args, output_dir, wav_dir))

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(ROOT),
        "model": str(args.model.resolve()),
        "audio": str(args.audio.resolve()),
        "engines": args.engines,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "chunk_sec": args.chunk_sec,
        "max_chunks": args.max_chunks,
        "timing_scope": "model load excluded; inference pipeline included",
        "workload": {
            "segmentation": "fixed windows",
            "prepared_chunks": len(workload_manifest) if workload_manifest else None,
            "prepared_audio_sec": sum(
                item["duration_sec"] for item in workload_manifest
            ) if workload_manifest else None,
        },
        "gpu": gpu_metadata(args.gpu_id),
        "git": git_metadata(),
    }
    paths = write_reports(output_dir, records, metadata)
    print("\n=== Summary ===")
    print(Path(paths["markdown"]).read_text(encoding="utf-8"))
    print(f"Artifacts: {output_dir}")
    return 0 if all(record["status"] in {"ok", "skipped"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
