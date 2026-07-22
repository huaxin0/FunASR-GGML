import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_cer import (
    KeyedText,
    edit_distance,
    evaluate_corpus,
    evaluate_gate,
    load_hypothesis_records,
    main,
    normalize_text,
)


class AccuracyMetricTests(unittest.TestCase):
    def test_normalization_preserves_legacy_aishell_behavior(self):
        self.assertEqual(normalize_text("二〇二六年 7月15日，你好！"),
                         "二〇二六年七月十五日你好")

    def test_edit_distance_reports_substitution_deletion_insertion(self):
        errors, substitutions, deletions, insertions = edit_distance(
            list("abcd"), list("axce"))
        self.assertEqual(errors, 2)
        self.assertEqual(substitutions + deletions + insertions, 2)

    def test_missing_hypothesis_is_scored_as_full_deletion(self):
        refs = KeyedText({"a": "你 好", "b": "世 界"})
        hyps = KeyedText({"a": "你好"})

        report = evaluate_corpus(refs, hyps, ["a", "b"])

        self.assertEqual(report["corpus"]["missing_hypotheses"], 1)
        self.assertEqual(report["corpus"]["coverage_percent"], 50.0)
        self.assertEqual(report["metrics"]["cer"]["deletions"], 2)
        self.assertEqual(report["metrics"]["cer"]["percent"], 50.0)

    def test_duplicate_hypothesis_is_visible_and_first_value_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hyp.tsv"
            path.write_text("a\tfirst\na\tsecond\n", encoding="utf-8")
            records = load_hypothesis_records(path)

        self.assertEqual(records.texts["a"], "first")
        self.assertEqual(records.duplicates, ["a"])

    def test_strict_gate_checks_coverage_empty_and_duplicates(self):
        refs = KeyedText({"a": "hello world", "b": "good day"})
        hyps = KeyedText({"a": "hello world"}, duplicates=["a"])
        report = evaluate_corpus(refs, hyps, ["a", "b"], ("wer",))

        gate = evaluate_gate(report, enabled=True, max_wer=10.0)

        self.assertFalse(gate["passed"])
        failed = {item["name"] for item in gate["checks"] if not item["passed"]}
        self.assertIn("min_coverage", failed)
        self.assertIn("max_empty_rate", failed)
        self.assertIn("max_duplicates", failed)

    def test_whitespace_wer_uses_word_units(self):
        refs = KeyedText({"a": "hello brave world"})
        hyps = KeyedText({"a": "hello world"})

        report = evaluate_corpus(refs, hyps, ["a"], ("wer",))

        self.assertEqual(report["metrics"]["wer"]["deletions"], 1)
        self.assertAlmostEqual(report["metrics"]["wer"]["percent"], 100.0 / 3.0)

    def test_cli_writes_reports_and_returns_nonzero_on_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.txt"
            hyp = root / "hyp.tsv"
            ids = root / "ids.txt"
            json_out = root / "report.json"
            md_out = root / "report.md"
            ref.write_text("a 你 好\n", encoding="utf-8")
            hyp.write_text("a\t你错\n", encoding="utf-8")
            ids.write_text("a\n", encoding="utf-8")

            return_code = main([
                str(hyp), str(ref), "--expected-ids", str(ids),
                "--gate", "--max-cer", "10", "--json-out", str(json_out),
                "--markdown-out", str(md_out),
            ])

            self.assertEqual(return_code, 1)
            self.assertFalse(json.loads(json_out.read_text(encoding="utf-8"))["gate"]["passed"])
            self.assertIn("Gate: FAIL", md_out.read_text(encoding="utf-8"))

    def test_legacy_cli_without_manifest_evaluates_common_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.txt"
            hyp = root / "hyp.tsv"
            json_out = root / "report.json"
            ref.write_text("a 你 好\nb 世 界\n", encoding="utf-8")
            hyp.write_text("a\t你好\n", encoding="utf-8")

            return_code = main([
                str(hyp), str(ref), "--json-out", str(json_out),
                "--worst-cases", "0",
            ])

            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(return_code, 0)
            self.assertEqual(report["corpus"]["expected_utterances"], 1)
            self.assertEqual(report["metrics"]["cer"]["percent"], 0.0)

    def test_gate_requires_explicit_expected_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.txt"
            hyp = root / "hyp.tsv"
            ref.write_text("a 你 好\n", encoding="utf-8")
            hyp.write_text("a\t你好\n", encoding="utf-8")

            return_code = main([str(hyp), str(ref), "--gate", "--max-cer", "3"])

            self.assertEqual(return_code, 2)

    def test_regression_threshold_requires_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.txt"
            hyp = root / "hyp.tsv"
            ids = root / "ids.txt"
            ref.write_text("a 你 好\n", encoding="utf-8")
            hyp.write_text("a\t你好\n", encoding="utf-8")
            ids.write_text("a\n", encoding="utf-8")

            return_code = main([
                str(hyp), str(ref), "--expected-ids", str(ids),
                "--max-cer-regression", "0.1",
            ])

            self.assertEqual(return_code, 2)


if __name__ == "__main__":
    unittest.main()
