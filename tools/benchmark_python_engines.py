#!/usr/bin/env python3
"""Optional Python framework adapters for the unified ASR benchmark suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Callable


def validate_sherpa_model_dir(model_dir: str | Path) -> dict[str, Path]:
    root = Path(model_dir)
    paths = {
        "encoder_adaptor": root / "encoder_adaptor.int8.onnx",
        "llm": root / "llm.int8.onnx",
        "embedding": root / "embedding.int8.onnx",
        "tokenizer": root / "Qwen3-0.6 B",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
    tokenizer_json = paths["tokenizer"] / "tokenizer.json"
    if paths["tokenizer"].is_dir() and not tokenizer_json.is_file():
        missing.append(str(tokenizer_json.relative_to(root)))
    if missing:
        raise ValueError("missing sherpa model artifacts: " + ", ".join(missing))
    return paths


def dependency_status(engine: str, sherpa_model_dir: Path | None) -> tuple[bool, str]:
    modules = {
        "funasr-pytorch": ("funasr",),
        "funasr-vllm": ("funasr", "vllm"),
        "sherpa-onnx": ("sherpa_onnx",),
    }[engine]
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        return False, "missing Python packages: " + ", ".join(missing)
    if engine == "sherpa-onnx":
        if sherpa_model_dir is None:
            return False, "--sherpa-model-dir is required"
        try:
            validate_sherpa_model_dir(sherpa_model_dir)
        except ValueError as exc:
            return False, str(exc)
    return True, ""


def collect_wavs(wav_dir: Path) -> list[Path]:
    return sorted(path for path in wav_dir.glob("*.wav") if path.is_file())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def extract_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "sentence", "value"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return ""
    if isinstance(result, (list, tuple)):
        for item in result:
            text = extract_text(item)
            if text:
                return text
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else ""


def _ordered_texts(results: Any, wavs: list[Path]) -> list[str]:
    if not isinstance(results, (list, tuple)):
        results = [results]
    by_key: dict[str, str] = {}
    ordered: list[str] = []
    for result in results:
        text = extract_text(result)
        ordered.append(text)
        if isinstance(result, dict):
            key = result.get("key")
            if isinstance(key, str):
                by_key[Path(key).stem] = text
    if by_key:
        return [by_key.get(path.stem, "") for path in wavs]
    ordered.extend([""] * max(0, len(wavs) - len(ordered)))
    return ordered[:len(wavs)]


def load_funasr_pytorch(args: argparse.Namespace) -> tuple[Callable[[list[Path]], list[str]], str]:
    from funasr import AutoModel

    model = AutoModel(
        model=args.model,
        device=f"cuda:{args.gpu_id}",
        hub=args.hub,
        trust_remote_code=True,
        disable_update=True,
        ncpu=args.threads,
    )

    def infer(wavs: list[Path]) -> list[str]:
        results = model.generate(
            input=[str(path) for path in wavs],
            cache={},
            batch_size_s=args.batch_size_s,
        )
        return _ordered_texts(results, wavs)

    return infer, "framework-default"


def load_funasr_vllm(args: argparse.Namespace) -> tuple[Callable[[list[Path]], list[str]], str]:
    from funasr.auto.auto_model_vllm import AutoModelVLLM

    model = AutoModelVLLM(
        model=args.model,
        hub=args.hub,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    def infer(wavs: list[Path]) -> list[str]:
        results = model.generate(
            [str(path) for path in wavs],
            language="auto",
            temperature=0.0,
            repetition_penalty=1.0,
            max_new_tokens=args.max_tokens,
        )
        return _ordered_texts(results, wavs)

    return infer, "framework-default"


def load_sherpa(args: argparse.Namespace) -> tuple[Callable[[list[Path]], list[str]], str]:
    import numpy as np
    import soundfile as sf
    import sherpa_onnx

    paths = validate_sherpa_model_dir(args.sherpa_model_dir)
    recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=str(paths["encoder_adaptor"]),
        llm=str(paths["llm"]),
        embedding=str(paths["embedding"]),
        tokenizer=str(paths["tokenizer"]),
        num_threads=args.threads,
        provider=args.sherpa_provider,
        max_new_tokens=args.max_tokens,
        temperature=1e-6,
        top_p=0.8,
    )

    def infer(wavs: list[Path]) -> list[str]:
        texts: list[str] = []
        for offset in range(0, len(wavs), args.sherpa_batch_size):
            group = wavs[offset:offset + args.sherpa_batch_size]
            streams = []
            for path in group:
                samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
                if samples.ndim > 1:
                    samples = np.mean(samples, axis=1, dtype=np.float32)
                stream = recognizer.create_stream()
                stream.accept_waveform(sample_rate, np.asarray(samples, dtype=np.float32))
                streams.append(stream)
            recognizer.decode_streams(streams)
            texts.extend(extract_text(stream.result) for stream in streams)
        return texts

    return infer, "INT8 ONNX"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one optional ASR Python engine.")
    parser.add_argument("--engine", required=True, choices=["funasr-pytorch", "funasr-vllm", "sherpa-onnx"])
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--text-output-dir", type=Path)
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", choices=["hf", "ms"], default="hf")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size-s", type=float, default=300.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--sherpa-model-dir", type=Path)
    parser.add_argument("--sherpa-provider", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sherpa-batch-size", type=int, default=12)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    available, reason = dependency_status(args.engine, args.sherpa_model_dir)
    if args.check_only or not available:
        payload = {
            "engine": args.engine,
            "status": "available" if available else "skipped",
            "reason": reason,
            "records": [],
        }
        write_result(args.result_json, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.wav_dir is None or not args.wav_dir.is_dir():
        write_result(args.result_json, {
            "engine": args.engine,
            "status": "failed",
            "reason": f"wav directory not found: {args.wav_dir}",
            "records": [],
        })
        return 2

    wavs = collect_wavs(args.wav_dir)
    if not wavs:
        write_result(args.result_json, {
            "engine": args.engine,
            "status": "failed",
            "reason": f"no WAV files found in {args.wav_dir}",
            "records": [],
        })
        return 2

    audio_sec = sum(wav_duration(path) for path in wavs)
    loaders = {
        "funasr-pytorch": load_funasr_pytorch,
        "funasr-vllm": load_funasr_vllm,
        "sherpa-onnx": load_sherpa,
    }
    try:
        load_start = time.perf_counter()
        infer, precision = loaders[args.engine](args)
        model_load_sec = time.perf_counter() - load_start

        warmup_wavs = wavs[:min(len(wavs), max(1, args.sherpa_batch_size))]
        for warmup_id in range(args.warmup):
            print(f"[{args.engine}] warmup {warmup_id + 1}/{args.warmup}", flush=True)
            infer(warmup_wavs)

        records: list[dict[str, Any]] = []
        output_dir = args.text_output_dir or args.result_json.parent / "texts"
        output_dir.mkdir(parents=True, exist_ok=True)
        for repeat_id in range(1, args.repeat + 1):
            print(f"[{args.engine}] repeat {repeat_id}/{args.repeat}", flush=True)
            start = time.perf_counter()
            texts = infer(wavs)
            wall_sec = time.perf_counter() - start
            text_path = output_dir / f"{args.engine}_r{repeat_id}.txt"
            with text_path.open("w", encoding="utf-8") as output:
                for wav_path, text in zip(wavs, texts):
                    output.write(f"{wav_path.stem}\t{text or '[EMPTY]'}\n")
            success = sum(bool(text) for text in texts)
            records.append(
                {
                    "engine": args.engine,
                    "repeat": repeat_id,
                    "status": "ok" if success == len(wavs) else "failed",
                    "reason": "" if success == len(wavs) else f"empty outputs: {len(wavs) - success}",
                    "model": "Fun-ASR-Nano-2512",
                    "precision": precision,
                    "wall_sec": wall_sec,
                    "audio_sec": audio_sec,
                    "rtf": wall_sec / audio_sec,
                    "rtfx": audio_sec / wall_sec,
                    "success": success,
                    "total": len(wavs),
                    "model_load_sec": model_load_sec,
                    "text_output": str(text_path),
                }
            )
        write_result(args.result_json, {
            "engine": args.engine,
            "status": "ok",
            "reason": "",
            "model_load_sec": model_load_sec,
            "records": records,
        })
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_result(args.result_json, {
            "engine": args.engine,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "records": [],
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
