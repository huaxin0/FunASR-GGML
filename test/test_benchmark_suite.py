import sys
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_common import (
    aggregate_records,
    parse_offline_log,
    split_wav_fixed,
    write_reports,
)
from tools.benchmark_python_engines import (
    load_funasr_pytorch,
    validate_sherpa_model_dir,
)
from tools.benchmark_suite import (
    build_cpp_command,
    build_python_engine_command,
    run_python_engine,
)


OFFLINE_LOG = """
[OfflineTest] ok=204/204
[OfflineTest] wall total=57679ms rtf=0.0094
[OfflineTest] scheduler: gpu=1 kv=paged batch=12 chunks=204 decode_steps=1672 grouped_calls=1670 fallback_calls=2 avg_active=11.47 blocks_peak=61/384 block_size=128
[OfflineTest] fallback_reasons: single=2 token_id=0 host_embed=0 serial_env=0 invalid=0
[OfflineTest] paged_kv_runtime: prefix_builds=0 prefix_hits=0 cow_copies=0 prefill_appends=1019 decode_appends=5 ownership_errors=0 final_free=384/384
[OfflineTest] scheduler_profile: admit=204/204 no_kv=0 admit_rounds=181 avg_admit_round=1.13 max_admit_round=12 prefill_wall=30060ms avg_prefill=147.35ms decode_dispatch=27570ms avg_decode_step=16.49ms idle_steps=1
[OfflineTest] throughput: audio_sec=6119.29 wall_sec=57.68 audio_sec/s=106.09 rtf=0.0094 tokens/s=332.6
[OfflineTest] paged_profile: calls=1672 build=0.200ms alloc=0.020ms set=0.170ms compute=15.980ms get=0.100ms total=16.470ms
[OfflineTest] paged_graph_cache_probe: calls=1672 shape_hits=1346 shape_hit_rate=80.50% param_hits=1346 param_hit_rate=80.50% full_hits=1346 full_hit_rate=80.50% cache_hits=1346 cache_misses=326 cache_hit_rate=80.50%
"""


class BenchmarkCommonTests(unittest.TestCase):
    def test_parse_offline_log(self):
        result = parse_offline_log(OFFLINE_LOG)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["success"], 204)
        self.assertEqual(result["total"], 204)
        self.assertAlmostEqual(result["wall_sec"], 57.679)
        self.assertAlmostEqual(result["rtf"], 0.0094)
        self.assertAlmostEqual(result["rtfx"], 106.09)
        self.assertAlmostEqual(result["tokens_s"], 332.6)
        self.assertAlmostEqual(result["prefill_sec"], 30.060)
        self.assertAlmostEqual(result["decode_sec"], 27.570)
        self.assertAlmostEqual(result["avg_active"], 11.47)
        self.assertEqual(result["fallback_calls"], 2)
        self.assertEqual(result["peak_blocks"], 61)
        self.assertEqual(result["block_capacity"], 384)
        self.assertEqual(result["ownership_errors"], 0)
        self.assertAlmostEqual(result["graph_hit_rate"], 80.50)

    def test_split_wav_fixed_preserves_frames_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(10)
                wav.writeframes(b"\x01\x00" * 10)

            chunks = split_wav_fixed(source, root / "chunks", chunk_sec=0.4)

            self.assertEqual([item["id"] for item in chunks], [
                "chunk_0000", "chunk_0001", "chunk_0002"
            ])
            self.assertEqual([item["frames"] for item in chunks], [4, 4, 2])
            self.assertEqual([item["start_sec"] for item in chunks], [0.0, 0.4, 0.8])
            self.assertEqual([item["duration_sec"] for item in chunks], [0.4, 0.4, 0.2])
            with wave.open(chunks[0]["path"], "rb") as first:
                self.assertEqual(first.getnchannels(), 1)
                self.assertEqual(first.getsampwidth(), 2)
                self.assertEqual(first.getframerate(), 10)
                self.assertEqual(first.getnframes(), 4)

    def test_aggregate_records_uses_medians_and_preserves_failures(self):
        records = [
            {"engine": "cpp-b12", "repeat": 1, "status": "ok", "wall_sec": 60.0, "rtfx": 100.0},
            {"engine": "cpp-b12", "repeat": 2, "status": "ok", "wall_sec": 50.0, "rtfx": 120.0},
            {"engine": "cpp-b12", "repeat": 3, "status": "ok", "wall_sec": 55.0, "rtfx": 110.0},
            {"engine": "funasr-vllm", "repeat": 0, "status": "skipped", "reason": "missing vllm"},
        ]

        summary = {item["engine"]: item for item in aggregate_records(records)}

        self.assertEqual(summary["cpp-b12"]["status"], "ok")
        self.assertEqual(summary["cpp-b12"]["measured_repeats"], 3)
        self.assertEqual(summary["cpp-b12"]["median_wall_sec"], 55.0)
        self.assertEqual(summary["cpp-b12"]["min_wall_sec"], 50.0)
        self.assertEqual(summary["cpp-b12"]["max_wall_sec"], 60.0)
        self.assertEqual(summary["cpp-b12"]["median_rtfx"], 110.0)
        self.assertEqual(summary["funasr-vllm"]["status"], "skipped")
        self.assertEqual(summary["funasr-vllm"]["reason"], "missing vllm")

    def test_write_reports_emits_csv_json_and_markdown(self):
        records = [
            {
                "engine": "cpp-b12",
                "repeat": 1,
                "status": "ok",
                "model": "Fun-ASR-Nano-2512",
                "precision": "Q8_0",
                "wall_sec": 57.679,
                "rtfx": 106.09,
                "rtf": 0.0094,
                "success": 204,
                "total": 204,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reports(Path(tmp), records, {"gpu": "RTX 4070 Laptop"})

            self.assertEqual(set(paths), {"records", "summary", "markdown", "metadata"})
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("cpp-b12", markdown)
            self.assertIn("106.09", markdown)
            self.assertIn("Q8_0", markdown)

    def test_write_reports_handles_failed_record_without_counts(self):
        records = [
            {
                "engine": "cpp-b12",
                "repeat": 1,
                "status": "failed",
                "model": "Fun-ASR-Nano-2512",
                "precision": "Q8_0",
                "reason": "process exited with code 1",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reports(Path(tmp), records, {})

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("failed", markdown)
            self.assertIn("process exited with code 1", markdown)


class BenchmarkSuiteTests(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            binary=Path("build-cuda/test_offline_batching"),
            model=Path("FunAsr_q8.bin"),
            audio=Path("source.wav"),
            ctx_size=4096,
            block_size=128,
            chunk_sec=30,
            max_tokens=220,
            max_chunks=2,
            threads=4,
            gpu_id=0,
            repeat=3,
            warmup=1,
            pytorch_python=Path("/envs/funasr/bin/python"),
            vllm_python=Path("/envs/vllm/bin/python"),
            sherpa_python=Path("/envs/sherpa/bin/python"),
            official_model="FunAudioLLM/Fun-ASR-Nano-2512",
            official_hub="hf",
            official_batch_size_s=300.0,
            vllm_gpu_memory_utilization=0.8,
            sherpa_model_dir=Path("/models/sherpa-funasr-nano"),
            sherpa_provider="cuda",
            sherpa_batch_size=12,
            quiet=True,
        )

    def test_cpp_b1_uses_single_request_continuous_kv(self):
        env, command = build_cpp_command("cpp-b1", self.args)

        joined = " ".join(command)
        self.assertIn("--kv-mode continuous", joined)
        self.assertIn("--batch-size 1", joined)
        self.assertIn("--max-chunks 2", joined)
        self.assertEqual(env["FUNASR_PAGED_DECODE_GRAPH_CACHE"], "1")

    def test_cpp_command_omits_default_zero_gpu_id(self):
        _, command = build_cpp_command("cpp-b12", self.args)

        self.assertNotIn("--gpu-id", command)

    def test_cpp_command_passes_nonzero_gpu_id(self):
        self.args.gpu_id = 1

        _, command = build_cpp_command("cpp-b12", self.args)

        index = command.index("--gpu-id")
        self.assertEqual(command[index + 1], "1")

    def test_vllm_command_uses_its_isolated_python_and_common_workload(self):
        command = build_python_engine_command(
            "funasr-vllm",
            self.args,
            Path("chunks"),
            Path("result.json"),
            Path("texts"),
        )

        joined = " ".join(command)
        self.assertEqual(command[0], "/envs/vllm/bin/python")
        self.assertIn("--engine funasr-vllm", joined)
        self.assertIn("--wav-dir chunks", joined)
        self.assertIn("--repeat 3", joined)
        self.assertIn("--warmup 1", joined)
        self.assertIn("--gpu-memory-utilization 0.8", joined)
        self.assertIn("--result-json result.json", joined)

    def test_sherpa_command_passes_model_and_batch_configuration(self):
        command = build_python_engine_command(
            "sherpa-onnx",
            self.args,
            Path("chunks"),
            Path("result.json"),
            Path("texts"),
        )

        joined = " ".join(command)
        self.assertEqual(command[0], "/envs/sherpa/bin/python")
        self.assertIn("--sherpa-model-dir /models/sherpa-funasr-nano", joined)
        self.assertIn("--sherpa-provider cuda", joined)
        self.assertIn("--sherpa-batch-size 12", joined)

    def test_cpp_b12_uses_current_paged_dynamic_fast_path(self):
        env, command = build_cpp_command("cpp-b12", self.args)

        joined = " ".join(command)
        self.assertIn("--kv-mode paged", joined)
        self.assertIn("--batch-size 12", joined)
        self.assertIn("--prefix-kv-cache off", joined)
        self.assertIn("--dynamic-kv-blocks on", joined)
        self.assertEqual(env["FUNASR_PAGED_KV_WRITE_OP"], "1")
        self.assertEqual(env["FUNASR_PAGED_DECODE_BUCKET_MAX_KV"], "1")
        self.assertEqual(env["FUNASR_PAGED_DECODE_GRAPH_CACHE"], "1")

    def test_missing_optional_python_is_recorded_without_aborting_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "tools.benchmark_suite.run_logged",
                side_effect=FileNotFoundError("missing interpreter"),
            ):
                records = run_python_engine(
                    "funasr-vllm",
                    self.args,
                    Path(tmp),
                    Path(tmp) / "wavs",
                )

        self.assertEqual(records[0]["status"], "skipped")
        self.assertIn("missing interpreter", records[0]["reason"])


class PythonEngineTests(unittest.TestCase):
    def test_funasr_pytorch_enables_required_remote_model_code(self):
        captured = {}
        fake_funasr = ModuleType("funasr")

        class FakeAutoModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_funasr.AutoModel = FakeAutoModel
        args = Namespace(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            gpu_id=0,
            hub="hf",
            threads=4,
            batch_size_s=300.0,
        )

        with patch.dict(sys.modules, {"funasr": fake_funasr}):
            load_funasr_pytorch(args)

        self.assertIs(captured["trust_remote_code"], True)

    def test_validate_sherpa_model_dir_returns_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            tokenizer = model_dir / "Qwen3-0.6 B"
            tokenizer.mkdir()
            for name in (
                "encoder_adaptor.int8.onnx",
                "llm.int8.onnx",
                "embedding.int8.onnx",
            ):
                (model_dir / name).touch()
            (tokenizer / "tokenizer.json").touch()

            paths = validate_sherpa_model_dir(model_dir)

            self.assertEqual(paths["encoder_adaptor"], model_dir / "encoder_adaptor.int8.onnx")
            self.assertEqual(paths["llm"], model_dir / "llm.int8.onnx")
            self.assertEqual(paths["embedding"], model_dir / "embedding.int8.onnx")
            self.assertEqual(paths["tokenizer"], tokenizer)

    def test_validate_sherpa_model_dir_lists_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "encoder_adaptor.int8.onnx"):
                validate_sherpa_model_dir(Path(tmp))

if __name__ == "__main__":
    unittest.main()
