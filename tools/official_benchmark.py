#!/usr/bin/env python3
"""
Benchmark official FunASR Python inference on AISHELL-1 style wav directories.

Example:
  python3 tools/official_benchmark.py \
      --model FunAudioLLM/Fun-ASR-Nano-2512 \
      --wav-dir /home/hua/data_aishell/wav/test \
      --output results_official_gpu.txt \
      --device cuda:0 \
      --vad-model fsmn-vad \
      --transcript /home/hua/data_aishell/transcript/aishell_transcript_v0.8.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import soundfile as sf
import torch
from funasr import AutoModel


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_cer import load_results, load_transcript, normalize_text, edit_distance


def collect_wav_files(wav_dir: str) -> List[Path]:
    files = sorted(p for p in Path(wav_dir).rglob("*.wav") if p.is_file())
    return files


def get_utterance_id(path: Path) -> str:
    return path.stem


def get_audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def extract_text(result) -> str:
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
    if isinstance(result, list):
        for item in result:
            text = extract_text(item)
            if text:
                return text
        return ""
    return ""


def run_cer(results_path: str, transcript_path: str) -> float:
    refs = load_transcript(transcript_path)
    hyps = load_results(results_path)
    common_ids = sorted(set(refs.keys()) & set(hyps.keys()))
    if not common_ids:
        raise RuntimeError("No common utterances found between results and transcript.")

    total_s = 0
    total_d = 0
    total_i = 0
    total_n = 0

    for utt_id in common_ids:
        ref = normalize_text(refs[utt_id])
        hyp = normalize_text(hyps[utt_id])
        if not ref:
            continue
        if not hyp:
            total_d += len(ref)
            total_n += len(ref)
            continue
        _, s, d, i = edit_distance(list(ref), list(hyp))
        total_s += s
        total_d += d
        total_i += i
        total_n += len(ref)

    if total_n == 0:
        return 0.0
    return (total_s + total_d + total_i) * 100.0 / total_n


def print_config(args: argparse.Namespace) -> None:
    print("[2] Loading official FunASR model")
    print(f"    Model:   {args.model}")
    print(f"    Device:  {args.device}")
    print(f"    Hub:     {args.hub}")
    print(f"    CUDA:    {'available' if torch.cuda.is_available() else 'not available'}")
    if args.vad_model:
        print(f"    VAD:     {args.vad_model}")
        print(f"    Max seg: {args.max_single_segment_time} ms")
    else:
        print("    VAD:     disabled")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark official FunASR Python inference.")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512",
                        help="Official model name or local model path.")
    parser.add_argument("--wav-dir", required=True,
                        help="Directory containing wav files, e.g. AISHELL-1 test wav dir.")
    parser.add_argument("--output", required=True,
                        help="Output text file, format: utterance_id<TAB>text")
    parser.add_argument("--device", default="cuda:0",
                        help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--hub", default="hf", choices=["hf", "ms"],
                        help="Model hub for remote model names.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of wav files for quick tests.")
    parser.add_argument("--transcript", default="",
                        help="Optional AISHELL transcript path for CER evaluation.")
    parser.add_argument("--vad-model", default="fsmn-vad",
                        help="Official VAD model name. Use empty string to disable.")
    parser.add_argument("--max-single-segment-time", type=int, default=30000,
                        help="VAD max_single_segment_time in ms.")
    parser.add_argument("--batch-size-s", type=float, default=0.0,
                        help="Pass-through to generate(batch_size_s=...). 0 disables dynamic batching.")
    parser.add_argument("--ncpu", type=int, default=4,
                        help="CPU threads for official FunASR preprocessing/runtime.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[1] Scanning WAV files in: {args.wav_dir}")
    wav_files = collect_wav_files(args.wav_dir)
    if args.limit > 0:
        wav_files = wav_files[:args.limit]
    print(f"    Found {len(wav_files)} files")
    print()

    if not wav_files:
        print("No WAV files found.")
        return 1

    print_config(args)

    model_kwargs = {
        "model": args.model,
        "device": args.device,
        "hub": args.hub,
        "disable_update": True,
        "ncpu": args.ncpu,
    }
    if args.vad_model:
        model_kwargs["vad_model"] = args.vad_model
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": args.max_single_segment_time}

    print("[2.1] Building AutoModel... this may take a while on first load")
    t_model_start = time.perf_counter()
    model = AutoModel(**model_kwargs)
    t_model_end = time.perf_counter()
    print(f"[2.2] AutoModel ready in {t_model_end - t_model_start:.1f} sec")
    print()

    total = len(wav_files)
    success = 0
    failed = 0
    total_inference_ms = 0.0
    total_inference_ms_excl_first = 0.0
    first_file_ms = 0.0
    total_audio_sec = 0.0

    t_all_start = time.perf_counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3] Running official inference on {total} files...")
    with output_path.open("w", encoding="utf-8") as out:
        for i, wav_path in enumerate(wav_files):
            utt_id = get_utterance_id(wav_path)
            audio_sec = get_audio_duration_sec(wav_path)

            t_start = time.perf_counter()
            res = model.generate(
                input=[str(wav_path)],
                cache={},
                batch_size_s=args.batch_size_s,
            )
            t_end = time.perf_counter()

            ms = (t_end - t_start) * 1000.0
            text = extract_text(res)

            if text:
                out.write(f"{utt_id}\t{text}\n")
                success += 1
                total_inference_ms += ms
                total_audio_sec += audio_sec
                if i == 0:
                    first_file_ms = ms
                else:
                    total_inference_ms_excl_first += ms
            else:
                out.write(f"{utt_id}\t[EMPTY]\n")
                failed += 1

            if (i + 1) % 100 == 0 or i == total - 1:
                avg_ms = total_inference_ms / success if success > 0 else 0.0
                progress = (i + 1) * 100.0 / total
                print(
                    f"\r    [{i + 1}/{total}] {progress:.1f}% | "
                    f"OK: {success}, Failed: {failed} | Avg: {avg_ms:.0f} ms/file",
                    end="",
                    flush=True,
                )

    t_all_end = time.perf_counter()
    total_sec = t_all_end - t_all_start

    print("\n\n========================================")
    print("  Official FunASR Benchmark Results")
    print("========================================")
    print(f"  Total files:    {total}")
    print(f"  Success:        {success}")
    print(f"  Failed:         {failed}")
    print(f"  Total time:     {total_sec:.1f} sec")
    print(f"  Avg per file:   {total_inference_ms / success:.0f} ms" if success else "  Avg per file:   0 ms")
    if success > 0:
        print(f"  First file:     {first_file_ms:.0f} ms")
    if success > 1:
        print(f"  Avg excl first: {total_inference_ms_excl_first / (success - 1):.0f} ms")
    throughput = total / total_sec if total_sec > 0 else 0.0
    print(f"  Throughput:     {throughput:.1f} files/sec")
    if total_audio_sec > 0 and total_inference_ms > 0:
        rtf = (total_inference_ms / 1000.0) / total_audio_sec
        print(f"  RTF:            {rtf:.3f}")
    print(f"  Output:         {output_path}")

    if args.transcript:
        cer = run_cer(str(output_path), args.transcript)
        print(f"  CER:            {cer:.2f}%")

    print("========================================")

    if args.transcript:
        print("\nFor detailed CER breakdown:")
        print(f"  python3 tools/eval_cer.py {output_path} {args.transcript}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
