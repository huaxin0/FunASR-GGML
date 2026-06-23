import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.funasr_web_app as web_app


class FunasrWebAppTests(unittest.TestCase):
    def test_index_serves_native_page(self):
        html = web_app.render_index()
        self.assertIn("FunASR-GGML", html)
        self.assertIn("videoUrl", html)
        self.assertIn("jobLog", html)

    def test_app_exposes_health_for_extension(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)

    def test_job_state_exposes_partial_srt_segments_before_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "transcript.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n实时字幕\n",
                encoding="utf-8",
            )
            job = web_app.Job(id="job123", output_dir=output, status="transcribing")

            state = job.public_state()

        self.assertEqual(state["segments"][0]["text"], "实时字幕")
        self.assertEqual(state["transcript"], "实时字幕\n")

    def test_default_cookies_path_uses_project_cookie_file(self):
        self.assertEqual(
            web_app.default_cookies_path(),
            str(web_app.ROOT / "cookies.txt"),
        )

    def test_validate_source_requires_url_or_upload(self):
        with self.assertRaisesRegex(ValueError, "Paste a URL or upload"):
            web_app.validate_source("", False)

    def test_download_command_can_use_browser_cookies(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = web_app.build_download_command(
                "https://www.bilibili.com/video/BV123/",
                Path(tmp),
                cookies_from_browser="chrome",
            )
        self.assertIn("--cookies-from-browser", cmd)
        index = cmd.index("--cookies-from-browser")
        self.assertEqual(cmd[index + 1], "chrome")
        self.assertEqual(cmd[-1], "https://www.bilibili.com/video/BV123/")

    def test_download_command_can_use_cookies_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cookies = root / "cookies.txt"
            cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            cmd = web_app.build_download_command(
                "https://www.bilibili.com/video/BV123/",
                root,
                cookies_path=str(cookies),
            )
        self.assertIn("--cookies", cmd)
        index = cmd.index("--cookies")
        self.assertEqual(cmd[index + 1], str(cookies))
        self.assertNotIn("--cookies-from-browser", cmd)

    def test_asr_command_uses_long_video_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = web_app.build_job_config(
                source="https://example.com/video",
                model=str(root / "model.bin"),
                funasr_cli=str(root / "funasr-cli"),
                output_dir=root / "out",
                use_gpu=True,
                keep_media=True,
            )
            cmd = web_app.build_asr_command(cfg, root / "source.wav", root / "out.srt")

        self.assertIn("--offline-preset", cmd)
        index = cmd.index("--offline-preset")
        self.assertEqual(cmd[index + 1], "long-video")
        self.assertIn("--gpu", cmd)

    def test_library_key_normalizes_bilibili_noise_params(self):
        first = web_app.library_key_for_source(
            "https://www.bilibili.com/video/BV1abc/?spm_id_from=333&vd_source=x"
        )
        second = web_app.library_key_for_source("https://www.bilibili.com/video/BV1abc/")
        self.assertEqual(first, second)
        self.assertEqual(first, "bilibili:BV1abc:p1")

    def test_library_records_and_looks_up_completed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "job"
            output.mkdir()
            (output / "transcript.json").write_text(
                '{"source":"https://www.bilibili.com/video/BV1abc/","segments":[]}\n',
                encoding="utf-8",
            )
            job = web_app.Job(id="job123", output_dir=output, status="done")
            index_path = root / "index.json"

            web_app.record_library_item(
                index_path,
                source="https://www.bilibili.com/video/BV1abc/?vd_source=x",
                job=job,
            )
            item = web_app.lookup_library_item(
                index_path,
                "https://www.bilibili.com/video/BV1abc/",
            )

        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["job_id"], "job123")
        self.assertEqual(item["key"], "bilibili:BV1abc:p1")
        self.assertTrue(item["files"]["json"].endswith("transcript.json"))

    def test_app_exposes_library_lookup_route(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/lookup", paths)

    def test_reusable_media_prefers_existing_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "job"
            media = output / "media"
            media.mkdir(parents=True)
            wav = media / "source_16k.wav"
            wav.write_bytes(b"wav")
            download = media / "download" / "source.m4a"
            download.parent.mkdir()
            download.write_bytes(b"m4a")
            (output / "transcript.json").write_text('{"segments":[]}\n', encoding="utf-8")
            job = web_app.Job(id="job123", output_dir=output, status="done")
            index_path = root / "index.json"

            web_app.record_library_item(
                index_path,
                source="https://www.bilibili.com/video/BV1abc/",
                job=job,
            )
            reusable = web_app.reusable_media_for_source(
                index_path,
                "https://www.bilibili.com/video/BV1abc/?spm_id_from=333",
            )

        self.assertEqual(reusable["wav"], str(wav))
        self.assertEqual(reusable["download"], str(download))

    def test_build_deepseek_summary_payload_requests_json(self):
        transcript = {
            "segments": [
                {"start": 1.0, "end": 3.0, "text": "这里介绍 transformer。"},
                {"start": 12.0, "end": 15.0, "text": "这里解释 attention。"},
            ]
        }
        payload = web_app.build_deepseek_summary_payload(transcript, model="deepseek-v4-flash")

        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        self.assertIn("5-10", payload["messages"][0]["content"])
        self.assertIn("[0:01-0:03]", payload["messages"][1]["content"])

    def test_summary_prompt_requests_detailed_learning_note_sections(self):
        payload = web_app.build_deepseek_summary_payload({"segments": []})
        combined = "\n".join(message["content"] for message in payload["messages"])

        self.assertIn("视频一句话概览", combined)
        self.assertIn("核心知识点", combined)
        self.assertIn("重点片段讲解", combined)
        self.assertIn("术语解释", combined)
        self.assertIn("易错点", combined)
        self.assertIn("复习清单", combined)
        self.assertIn("不要只输出短 bullet", combined)

    def test_summary_prompt_includes_user_custom_instruction(self):
        payload = web_app.build_deepseek_summary_payload(
            {"segments": []},
            custom_prompt="请整理成考试复习版，重点保留易错点。",
        )
        combined = "\n".join(message["content"] for message in payload["messages"])

        self.assertIn("用户额外要求", combined)
        self.assertIn("考试复习版", combined)

    def test_write_summary_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = {
                "title": "Transformer 学习笔记",
                "takeaways": ["attention 是核心"],
                "highlights": [
                    {"start": 12, "end": 15, "title": "Attention", "summary": "解释注意力。"}
                ],
                "markdown": "# Transformer 学习笔记\n\n- attention 是核心\n",
            }

            files = web_app.write_summary_outputs(output, summary)

        self.assertTrue(files["json"].endswith("summary.json"))
        self.assertTrue(files["markdown"].endswith("summary.md"))
        self.assertIn("summary", web_app.RESULT_FILES)
        self.assertIn("summary_md", web_app.RESULT_FILES)

    def test_app_exposes_summary_routes(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/summarize", paths)
        self.assertIn("/api/library/summary", paths)

    def test_build_deepseek_chat_payload_uses_summary_and_relevant_segments(self):
        transcript = {
            "segments": [
                {"start": 10, "end": 12, "text": "这里介绍 attention 机制。"},
                {"start": 40, "end": 42, "text": "这里讲无关内容。"},
            ]
        }
        summary = {
            "title": "Transformer 学习笔记",
            "takeaways": ["attention 是核心"],
            "highlights": [],
            "markdown": "# Transformer\n",
        }

        payload = web_app.build_deepseek_chat_payload(
            question="attention 是什么意思？",
            transcript=transcript,
            summary=summary,
        )

        combined = "\n".join(message["content"] for message in payload["messages"])
        self.assertIn("视频学习对话助手", combined)
        self.assertIn("Transformer 学习笔记", combined)
        self.assertIn("[0:10-0:12]", combined)
        self.assertIn("attention 是什么意思", combined)
        self.assertIn("可以引用时间戳", combined)

    def test_chat_session_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            session = web_app.append_chat_message(
                output,
                question="这里是什么意思？",
                answer="这里在解释 attention。",
            )
            loaded = web_app.read_chat_session(output)

        self.assertEqual(session["messages"][0]["question"], "这里是什么意思？")
        self.assertEqual(loaded["messages"][0]["answer"], "这里在解释 attention。")

    def test_clear_chat_session_removes_saved_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            web_app.append_chat_message(
                output,
                question="这里是什么意思？",
                answer="这里在解释 attention。",
            )

            cleared = web_app.clear_chat_session(output)
            loaded = web_app.read_chat_session(output)

        self.assertEqual(cleared, {"messages": []})
        self.assertEqual(loaded, {"messages": []})
        self.assertIn("chat", web_app.RESULT_FILES)

    def test_app_exposes_chat_routes(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/chat", paths)
        self.assertIn("/api/library/chat/clear", paths)

    def test_build_deepseek_mindmap_payload_requests_tree_json(self):
        transcript = {
            "segments": [
                {"start": 10, "end": 12, "text": "这里介绍 attention 机制。"},
            ]
        }
        summary = {
            "title": "Transformer 学习笔记",
            "takeaways": ["attention 是核心"],
            "highlights": [{"start": 10, "end": 12, "title": "Attention", "summary": "解释注意力"}],
            "markdown": "# Transformer\n",
        }

        payload = web_app.build_deepseek_mindmap_payload(transcript=transcript, summary=summary)
        combined = "\n".join(message["content"] for message in payload["messages"])

        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("思维导图", combined)
        self.assertIn("nodes", combined)
        self.assertIn("children", combined)
        self.assertIn("time", combined)
        self.assertIn("details", combined)
        self.assertIn("evidence", combined)
        self.assertIn("questions", combined)
        self.assertIn("timestamps", combined)
        self.assertIn("不是时间轴", combined)
        self.assertIn("[0:10-0:12]", combined)

    def test_write_mindmap_output_normalizes_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            mindmap = {
                "title": "Self-Attention",
                "nodes": [
                    {
                        "title": "QKV",
                        "kind": "concept",
                        "summary": "解释 Query、Key、Value 的作用。",
                        "details": ["Query 用来提出匹配请求", "Key 用来被匹配"],
                        "evidence": [{"time": 120, "text": "视频用检索类比解释 QKV。"}],
                        "questions": ["为什么 Q 和 K 要做相似度计算？"],
                        "timestamps": [120, 130],
                        "time": 120,
                        "children": [{"title": "Query", "time": 130}],
                    }
                ],
            }
            path = web_app.write_mindmap_output(output, mindmap)
            loaded = web_app.read_mindmap_from_output(output)

        self.assertTrue(path.endswith("mindmap.json"))
        self.assertEqual(loaded["nodes"][0]["children"][0]["title"], "Query")
        self.assertEqual(loaded["nodes"][0]["kind"], "concept")
        self.assertEqual(loaded["nodes"][0]["details"][0], "Query 用来提出匹配请求")
        self.assertEqual(loaded["nodes"][0]["evidence"][0]["time"], 120.0)
        self.assertEqual(loaded["nodes"][0]["questions"][0], "为什么 Q 和 K 要做相似度计算？")
        self.assertEqual(loaded["nodes"][0]["timestamps"], [120.0, 130.0])
        self.assertIn("mindmap", web_app.RESULT_FILES)

    def test_app_exposes_mindmap_routes(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/mindmap", paths)

    def test_frame_extraction_uses_summary_highlights_and_video_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            media = output / "media" / "download"
            media.mkdir(parents=True)
            video = media / "source.mp4"
            video.write_bytes(b"video")
            web_app.write_summary_outputs(
                output,
                {
                    "title": "课程",
                    "takeaways": [],
                    "highlights": [
                        {"start": 12, "end": 18, "title": "Attention", "summary": "解释注意力。"}
                    ],
                    "markdown": "# 课程\n",
                },
            )
            item = {"key": "bilibili:BV1:p1", "output_dir": str(output), "files": {"download": str(video)}}
            calls = []

            def fake_run(cmd, check):
                calls.append(cmd)
                Path(cmd[-1]).write_bytes(b"jpg")

            frames = web_app.extract_frames_for_item(item, run_command=fake_run)
            loaded = web_app.read_frames_from_output(output)

        self.assertEqual(len(frames["frames"]), 1)
        self.assertEqual(frames["frames"][0]["title"], "Attention")
        self.assertTrue(frames["frames"][0]["image"].endswith(".jpg"))
        self.assertEqual(loaded["frames"][0]["start"], 12.0)
        self.assertIn("-ss", calls[0])
        self.assertIn(str(video), calls[0])

    def test_frame_extraction_requires_cached_video_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            audio = output / "media" / "download" / "source.m4a"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"audio")
            item = {"key": "bilibili:BV1:p1", "output_dir": str(output), "files": {"download": str(audio)}}

            with self.assertRaisesRegex(ValueError, "No cached video media"):
                web_app.extract_frames_for_item(item, run_command=lambda cmd, check: None)

    def test_app_exposes_frame_routes(self):
        app = web_app.create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/api/library/frames", paths)
        self.assertIn("/api/library/frames/{key}/{filename}", paths)


if __name__ == "__main__":
    unittest.main()
