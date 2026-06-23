import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXT_ROOT = ROOT / "browser-extension" / "bilibili-funasr-sidebar"


class BilibiliExtensionTests(unittest.TestCase):
    def test_manifest_targets_bilibili_and_local_api(self):
        manifest = json.loads((EXT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertIn("https://www.bilibili.com/*", manifest["host_permissions"])
        self.assertIn("http://127.0.0.1:*/*", manifest["host_permissions"])
        scripts = manifest["content_scripts"][0]["js"]
        self.assertIn("content.js", scripts)

    def test_content_script_creates_sidebar_and_controls_video(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("funasr-sidebar", content)
        self.assertIn("/api/jobs", content)
        self.assertIn("document.querySelector(\"video\")", content)
        self.assertIn("video.currentTime", content)

    def test_content_script_probes_common_local_ports(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("API_CANDIDATES", content)
        self.assertIn("http://127.0.0.1:8008", content)
        self.assertIn("http://127.0.0.1:8009", content)
        self.assertIn("http://127.0.0.1:8010", content)
        self.assertIn("detectApiBase", content)

    def test_content_script_does_not_truncate_long_transcripts(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertNotIn("segments.slice(0, 300)", content)
        self.assertIn("for (const item of segments)", content)

    def test_content_script_renders_partial_job_segments_while_polling(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("Array.isArray(state.segments)", content)
        self.assertIn("renderSegments(state.segments)", content)
        self.assertIn("正在显示实时字幕", content)

    def test_content_script_can_lookup_and_load_cached_result(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("/api/library/lookup", content)
        self.assertIn("loadCachedResult", content)
        self.assertIn("加载已有结果", content)

    def test_content_script_groups_short_segments_for_readability(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("GROUP_SECONDS", content)
        self.assertIn("groupSegments", content)
        self.assertIn("const GROUP_SECONDS = 10", content)

    def test_content_script_supports_search_and_playback_highlight(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn('data-role="search"', content)
        self.assertIn("filterSegments", content)
        self.assertIn("highlightCurrentSegment", content)
        self.assertIn("timeupdate", content)
        self.assertIn("funasr-active", content)

    def test_playback_highlight_does_not_force_scroll(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertNotIn("scrollIntoView", content)

    def test_content_script_supports_deepseek_summary(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("生成总结", content)
        self.assertIn("/api/library/summarize", content)
        self.assertIn("/api/library/summary", content)
        self.assertIn("renderSummary", content)
        self.assertIn("summary.highlights", content)
        self.assertIn("summary.markdown", content)

    def test_summary_regeneration_can_use_custom_prompt(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn('data-role="summary-prompt-panel"', content)
        self.assertIn('data-role="summary-prompt"', content)
        self.assertIn('data-role="summary-confirm"', content)
        self.assertIn("showSummaryPromptPanel", content)
        self.assertIn('form.append("custom_prompt"', content)
        self.assertIn("funasr-summary-prompt-panel", css)

    def test_content_script_uses_tabs_for_summary_transcript_and_logs(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("funasr-tabs", content)
        self.assertIn('data-tab="summary"', content)
        self.assertIn('data-tab="mindmap"', content)
        self.assertIn('data-tab="chat"', content)
        self.assertIn('data-tab="transcript"', content)
        self.assertIn('data-tab="log"', content)
        self.assertIn('data-panel="summary"', content)
        self.assertIn('data-panel="mindmap"', content)
        self.assertIn('data-panel="chat"', content)
        self.assertIn('data-panel="transcript"', content)
        self.assertIn('data-panel="log"', content)
        self.assertIn("setActiveTab", content)

    def test_content_script_supports_video_chat(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("视频问答", content)
        self.assertIn('data-role="chat-input"', content)
        self.assertIn('data-role="chat-send"', content)
        self.assertIn('data-role="chat-list"', content)
        self.assertIn("/api/library/chat", content)
        self.assertIn("sendChatQuestion", content)
        self.assertIn("renderChatSession", content)

    def test_content_script_renders_markdown_and_clickable_timestamps(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("renderMarkdown", content)
        self.assertIn("appendRichTextWithTimestamps", content)
        self.assertIn("parseTimestampToSeconds", content)
        self.assertIn("funasr-timestamp-link", content)
        self.assertNotIn('className = "funasr-markdown"', content)
        self.assertNotIn("markdown.textContent = summary.markdown", content)

    def test_content_script_supports_mindmap(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("生成导图", content)
        self.assertIn("/api/library/mindmap", content)
        self.assertIn("renderMindmap", content)
        self.assertIn("renderMindmapCanvas", content)
        self.assertIn("layoutMindmap", content)
        self.assertIn("funasr-mindmap-canvas-node", content)

    def test_content_script_supports_key_frames(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn("生成关键图", content)
        self.assertIn('data-tab="frames"', content)
        self.assertIn('data-panel="frames"', content)
        self.assertIn('data-role="frames-view"', content)
        self.assertIn("/api/library/frames", content)
        self.assertIn("renderFrames", content)
        self.assertIn("generateFrames", content)
        self.assertIn("funasr-frame-card", css)

    def test_mindmap_uses_canvas_structure(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn("funasr-mindmap-toolbar", content)
        self.assertIn("funasr-mindmap-viewport", content)
        self.assertIn("funasr-mindmap-transform", content)
        self.assertIn("funasr-mindmap-time", content)
        self.assertIn("funasr-mindmap-node-title", content)
        self.assertIn("funasr-mindmap-kind", content)
        self.assertIn("funasr-mindmap-detail", content)
        self.assertIn("funasr-mindmap-question", content)
        self.assertIn("funasr-mindmap-evidence", content)
        self.assertIn("node.timestamps", content)
        self.assertIn("funasr-mindmap-svg", content)
        self.assertIn("funasr-mindmap-canvas-node", css)
        self.assertIn("funasr-mindmap-detail", css)
        self.assertIn("funasr-mindmap-question", css)
        self.assertIn("funasr-mindmap-evidence", css)
        self.assertIn("linear-gradient", css)
        self.assertIn("cursor: grab", css)

    def test_content_script_restores_cached_learning_state(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn('data-role="study-state"', content)
        self.assertIn("renderStudyState", content)
        self.assertIn("autoRestoreCachedResult", content)
        self.assertIn("chatTurns", content)
        self.assertIn("已恢复历史学习记录", content)

    def test_content_script_shows_project_files_and_can_clear_chat(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn('data-role="project-files"', content)
        self.assertIn('data-role="output-dir"', content)
        self.assertIn('data-role="file-summary-md"', content)
        self.assertIn('data-role="file-mindmap"', content)
        self.assertIn('data-role="file-chat"', content)
        self.assertIn('data-role="clear-chat"', content)
        self.assertIn("renderProjectFiles", content)
        self.assertIn("clearChatSession", content)
        self.assertIn("/api/library/chat/clear", content)
        self.assertIn("funasr-project-files", css)

    def test_sidebar_styles_learning_state_badges(self):
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn("funasr-study-state", css)
        self.assertIn("funasr-state-badge", css)
        self.assertIn("funasr-state-badge.ready", css)

    def test_content_script_does_not_advertise_subtitle_first_statuses(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertNotIn("checking-subtitles", content)
        self.assertNotIn("正在检查视频自带字幕", content)
        self.assertNotIn("importing-subtitles", content)
        self.assertNotIn("正在导入视频自带字幕", content)
        self.assertNotIn("已使用视频自带字幕快速加载", content)

    def test_mindmap_uses_canvas_viewport_and_toolbar(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        css = (EXT_ROOT / "sidebar.css").read_text(encoding="utf-8")
        self.assertIn("renderMindmapCanvas", content)
        self.assertIn("funasr-mindmap-toolbar", content)
        self.assertIn("funasr-mindmap-viewport", content)
        self.assertIn("funasr-mindmap-transform", content)
        self.assertIn("funasr-mindmap-svg", content)
        self.assertIn("funasr-mindmap-canvas-node", css)

    def test_mindmap_canvas_supports_pan_zoom_and_jump(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("mindmapViewState", content)
        self.assertIn("setMindmapTransform", content)
        self.assertIn("centerMindmapCanvas", content)
        self.assertIn("wheel", content)
        self.assertIn("pointerdown", content)
        self.assertIn("jumpTo(seconds)", content)
        self.assertIn("appendMindmapTimestampChips", content)

    def test_mindmap_layout_accounts_for_rich_node_height(self):
        content = (EXT_ROOT / "content.js").read_text(encoding="utf-8")
        self.assertIn("layoutMindmapBranches", content)
        self.assertIn("funasr-mindmap-root-node", content)
        self.assertIn("funasr-mindmap-inspector", content)
        self.assertIn("selectedMindmapNode", content)
        self.assertIn("side:", content)
        self.assertIn("Math.max(...laidOut.map((item) => item.y + item.height))", content)
        self.assertNotIn("y: 28 + index * 146", content)
        self.assertNotIn("laidOut.length * 86", content)


if __name__ == "__main__":
    unittest.main()
