import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.funasr_web_ui as web


class FunasrWebUiTests(unittest.TestCase):
    def test_select_source_requires_url_or_upload(self):
        result = web.select_source("", None, Path("/tmp/job"))
        self.assertFalse(result.ok)
        self.assertIn("Paste a URL or upload", result.message)

    def test_select_source_uses_url_when_no_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            result = web.select_source("https://example.com/video", None, job_dir)
            self.assertTrue(result.ok)
            self.assertEqual(result.source, "https://example.com/video")
            self.assertFalse(result.is_upload)

    def test_select_source_prefers_uploaded_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploaded = root / "clip.mp4"
            uploaded.write_bytes(b"fake")
            job_dir = root / "job"

            result = web.select_source("https://example.com/video", str(uploaded), job_dir)

            copied = job_dir / "media" / "upload" / "clip.mp4"
            self.assertTrue(result.ok)
            self.assertEqual(result.source, str(copied))
            self.assertTrue(result.is_upload)
            self.assertEqual(copied.read_bytes(), b"fake")

    def test_build_web_config_uses_long_video_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web.build_web_config(
                source="https://example.com/video",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )

            self.assertEqual(cfg.chunk_mode, "window")
            self.assertEqual(cfg.chunk_sec, "30")
            self.assertEqual(cfg.ctx_size, "4096")
            self.assertEqual(cfg.max_tokens, "220")
            self.assertEqual(cfg.srt_max_chars, "28")
            self.assertTrue(cfg.offline_scheduler)
            self.assertTrue(cfg.offline_profile)
            self.assertEqual(cfg.offline_batch_size, "12")
            self.assertEqual(cfg.offline_kv_mode, "paged")
            self.assertEqual(cfg.offline_kv_block_size, "128")

    def test_run_asr_uses_long_video_preset(self):
        calls = []

        def fake_run_command(cmd, cwd=None):
            calls.append(cmd)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web.build_web_config(
                source="https://example.com/video",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )

            with mock.patch.object(web.video_ui, "run_command", fake_run_command):
                web.video_ui.run_asr(cfg, root / "source.wav", root / "out.srt")

        self.assertTrue(calls)
        cmd = calls[0]
        self.assertIn("--offline-preset", cmd)
        preset_index = cmd.index("--offline-preset")
        self.assertEqual(cmd[preset_index + 1], "long-video")


if __name__ == "__main__":
    unittest.main()
