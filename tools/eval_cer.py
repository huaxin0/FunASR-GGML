#!/usr/bin/env python3
"""Corpus-level CER/WER evaluation and accuracy gates.

The legacy positional invocation remains supported::

    python3 tools/eval_cer.py hypotheses.tsv transcript.txt

For a strict gate, provide the complete expected ID list. Missing hypotheses
are then scored as empty outputs instead of silently disappearing from the
denominator::

    python3 tools/eval_cer.py hypotheses.tsv transcript.txt \
        --expected-ids expected_ids.txt --metrics cer --gate --max-cer 3.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EMPTY_MARKERS = {"[EMPTY]", "<EMPTY>"}

CHINESE_DIGITS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}

PUNCT_TRANSLATION = str.maketrans("", "", (
    "，。！？、：；“”‘’《》【】（）()[]{}"
    ",.!?;:\"'<>"
))


@dataclass
class KeyedText:
    texts: Dict[str, str] = field(default_factory=dict)
    duplicates: List[str] = field(default_factory=list)
    malformed_lines: List[int] = field(default_factory=list)


def digits_to_chinese(text: str) -> str:
    return "".join(CHINESE_DIGITS.get(ch, ch) for ch in text)


def int_to_chinese(num_text: str) -> str:
    value = int(num_text)
    if value < 10:
        return CHINESE_DIGITS[str(value)]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + CHINESE_DIGITS[str(value % 10)]
    tens, ones = divmod(value, 10)
    suffix = "" if ones == 0 else CHINESE_DIGITS[str(ones)]
    return CHINESE_DIGITS[str(tens)] + "十" + suffix


def normalize_numbers(text: str) -> str:
    def repl_ymd(match: re.Match[str]) -> str:
        suffix = match.group(4) or ""
        return (
            f"{digits_to_chinese(match.group(1))}年"
            f"{int_to_chinese(match.group(2))}月"
            f"{int_to_chinese(match.group(3))}{suffix}"
        )

    def repl_md(match: re.Match[str]) -> str:
        suffix = match.group(3) or ""
        return (
            f"{int_to_chinese(match.group(1))}月"
            f"{int_to_chinese(match.group(2))}{suffix}"
        )

    text = re.sub(r"(\d{2,4})年(\d{1,2})月(\d{1,2})(日|号)?", repl_ymd, text)
    text = re.sub(r"(\d{1,2})月(\d{1,2})(日|号)?", repl_md, text)
    return re.sub(r"\d+", lambda match: digits_to_chinese(match.group(0)), text)


def normalize_text(text: str) -> str:
    """AISHELL-compatible character normalization used by the legacy tool."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = normalize_numbers(text)
    text = text.translate(PUNCT_TRANSLATION)
    return "".join(text.split())


def normalize_words(text: str) -> List[str]:
    """Normalize text and return whitespace-delimited words.

    This is a real word-boundary metric only when references and hypotheses
    both contain word boundaries. Chinese AISHELL output normally does not, so
    CER should be the release gate for that corpus.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = normalize_numbers(text)
    chars = [" " if unicodedata.category(ch).startswith(("P", "S")) else ch
             for ch in text]
    return "".join(chars).split()


def load_reference_records(path: str | Path) -> KeyedText:
    records = KeyedText()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[0]:
                records.malformed_lines.append(line_no)
                continue
            utt_id, text = parts
            if utt_id in records.texts:
                records.duplicates.append(utt_id)
                continue
            records.texts[utt_id] = text
    return records


def load_hypothesis_records(path: str | Path) -> KeyedText:
    records = KeyedText()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0]:
                records.malformed_lines.append(line_no)
                continue
            utt_id, text = parts
            if utt_id in records.texts:
                records.duplicates.append(utt_id)
                continue
            records.texts[utt_id] = "" if text.strip() in EMPTY_MARKERS else text
    return records


def load_transcript(path: str | Path) -> Dict[str, str]:
    """Compatibility API used by ``official_benchmark.py``."""
    return {
        utt_id: "".join(text.split())
        for utt_id, text in load_reference_records(path).texts.items()
    }


def load_results(path: str | Path) -> Dict[str, str]:
    """Compatibility API used by ``official_benchmark.py``."""
    return load_hypothesis_records(path).texts


def load_expected_ids(path: str | Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seen = set()
    duplicates: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            utt_id = line.split("\t", 1)[0].split(None, 1)[0]
            if utt_id in seen:
                duplicates.append(utt_id)
                continue
            seen.add(utt_id)
            ids.append(utt_id)
    return ids, duplicates


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    """Return ``(errors, substitutions, deletions, insertions)``.

    Only two dynamic-programming rows are retained, avoiding the quadratic
    tuple matrix used by the old evaluator.
    """
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_item in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_item in enumerate(hyp, 1):
            if ref_item == hyp_item:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            choices = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            current.append(min(choices, key=lambda item: item[0]))
        previous = current
    return previous[-1]


def _metric_result(
    refs: Mapping[str, str],
    hyps: Mapping[str, str],
    expected_ids: Sequence[str],
    metric: str,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    substitutions = deletions = insertions = units = perfect = 0
    utterance_rates: Dict[str, float] = {}

    for utt_id in expected_ids:
        raw_ref = refs[utt_id]
        raw_hyp = hyps.get(utt_id, "")
        if metric == "cer":
            ref_units: Sequence[str] = list(normalize_text(raw_ref))
            hyp_units: Sequence[str] = list(normalize_text(raw_hyp))
        else:
            ref_units = normalize_words(raw_ref)
            hyp_units = normalize_words(raw_hyp)
        if not ref_units:
            continue
        _, sub, delete, insert = edit_distance(ref_units, hyp_units)
        substitutions += sub
        deletions += delete
        insertions += insert
        units += len(ref_units)
        errors = sub + delete + insert
        utterance_rates[utt_id] = errors / len(ref_units)
        if errors == 0:
            perfect += 1

    errors = substitutions + deletions + insertions
    rate = errors / units if units else None
    result: Dict[str, object] = {
        "rate": rate,
        "percent": rate * 100.0 if rate is not None else None,
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": units,
        "perfect_utterances": perfect,
    }
    return result, utterance_rates


def evaluate_corpus(
    references: KeyedText,
    hypotheses: KeyedText,
    expected_ids: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("cer",),
    worst_case_count: int = 10,
) -> Dict[str, object]:
    expected = list(expected_ids) if expected_ids is not None else sorted(references.texts)
    expected_set = set(expected)
    missing_references = [utt_id for utt_id in expected if utt_id not in references.texts]
    if missing_references:
        raise ValueError(
            f"{len(missing_references)} expected IDs have no reference; "
            f"first={missing_references[0]}"
        )

    missing = [utt_id for utt_id in expected if utt_id not in hypotheses.texts]
    present = [utt_id for utt_id in expected if utt_id in hypotheses.texts]
    empty_present = [utt_id for utt_id in present
                     if not normalize_text(hypotheses.texts[utt_id])]
    extras = sorted(set(hypotheses.texts) - expected_set)
    empty_references = [utt_id for utt_id in expected
                        if not normalize_text(references.texts[utt_id])]
    expected_count = len(expected)
    coverage = len(present) / expected_count if expected_count else 0.0
    effective_empty = len(missing) + len(empty_present)
    empty_rate = effective_empty / expected_count if expected_count else 0.0

    metric_results: Dict[str, object] = {}
    per_utterance: Dict[str, Dict[str, float]] = {utt_id: {} for utt_id in expected}
    for metric in metrics:
        result, rates = _metric_result(
            references.texts, hypotheses.texts, expected, metric)
        metric_results[metric] = result
        for utt_id, rate in rates.items():
            per_utterance[utt_id][metric] = rate

    ranking_metric = "cer" if "cer" in metrics else metrics[0]
    ranked = sorted(
        ((values.get(ranking_metric, 0.0), utt_id)
         for utt_id, values in per_utterance.items()),
        reverse=True,
    )
    worst_cases = []
    for rate, utt_id in ranked[:max(0, worst_case_count)]:
        worst_cases.append({
            "id": utt_id,
            "metric": ranking_metric,
            "rate": rate,
            "percent": rate * 100.0,
            "reference": normalize_text(references.texts[utt_id]),
            "hypothesis": normalize_text(hypotheses.texts.get(utt_id, "")),
        })

    return {
        "schema_version": 1,
        "corpus": {
            "expected_utterances": expected_count,
            "present_hypotheses": len(present),
            "coverage": coverage,
            "coverage_percent": coverage * 100.0,
            "missing_hypotheses": len(missing),
            "empty_hypotheses": len(empty_present),
            "effective_empty_outputs": effective_empty,
            "empty_rate": empty_rate,
            "empty_rate_percent": empty_rate * 100.0,
            "extra_hypotheses": len(extras),
            "duplicate_hypotheses": len(hypotheses.duplicates),
            "malformed_hypothesis_lines": len(hypotheses.malformed_lines),
            "duplicate_references": len(references.duplicates),
            "malformed_reference_lines": len(references.malformed_lines),
            "empty_references": len(empty_references),
        },
        "metrics": metric_results,
        "issues": {
            "missing_ids": missing,
            "empty_ids": empty_present,
            "extra_ids": extras,
            "duplicate_hypothesis_ids": hypotheses.duplicates,
            "duplicate_reference_ids": references.duplicates,
            "empty_reference_ids": empty_references,
        },
        "worst_cases": worst_cases,
    }


def evaluate_gate(
    report: Dict[str, object],
    *,
    enabled: bool,
    max_cer: Optional[float] = None,
    max_wer: Optional[float] = None,
    min_coverage: Optional[float] = None,
    max_empty_rate: Optional[float] = None,
    max_missing: Optional[int] = None,
    max_extra: Optional[int] = None,
    max_duplicates: Optional[int] = None,
    max_malformed: Optional[int] = None,
    baseline_report: Optional[Dict[str, object]] = None,
    max_cer_regression: Optional[float] = None,
    max_wer_regression: Optional[float] = None,
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def add(name: str, actual: float, limit: float, passed: bool, unit: str) -> None:
        checks.append({
            "name": name,
            "actual": actual,
            "limit": limit,
            "unit": unit,
            "passed": passed,
        })

    corpus = report["corpus"]
    metrics = report["metrics"]
    if enabled:
        min_coverage = 100.0 if min_coverage is None else min_coverage
        max_empty_rate = 0.0 if max_empty_rate is None else max_empty_rate
        max_missing = 0 if max_missing is None else max_missing
        max_extra = 0 if max_extra is None else max_extra
        max_duplicates = 0 if max_duplicates is None else max_duplicates
        max_malformed = 0 if max_malformed is None else max_malformed

    if max_cer is not None:
        actual = metrics.get("cer", {}).get("percent")
        add("max_cer", float("inf") if actual is None else actual,
            max_cer, actual is not None and actual <= max_cer, "percent")
    if max_wer is not None:
        actual = metrics.get("wer", {}).get("percent")
        add("max_wer", float("inf") if actual is None else actual,
            max_wer, actual is not None and actual <= max_wer, "percent")
    if min_coverage is not None:
        actual = corpus["coverage_percent"]
        add("min_coverage", actual, min_coverage, actual >= min_coverage, "percent")
    if max_empty_rate is not None:
        actual = corpus["empty_rate_percent"]
        add("max_empty_rate", actual, max_empty_rate, actual <= max_empty_rate, "percent")
    if max_missing is not None:
        actual = corpus["missing_hypotheses"]
        add("max_missing", actual, max_missing, actual <= max_missing, "utterances")
    if max_extra is not None:
        actual = corpus["extra_hypotheses"]
        add("max_extra", actual, max_extra, actual <= max_extra, "utterances")
    if max_duplicates is not None:
        actual = corpus["duplicate_hypotheses"] + corpus["duplicate_references"]
        add("max_duplicates", actual, max_duplicates,
            actual <= max_duplicates, "utterances")
    if max_malformed is not None:
        actual = (corpus["malformed_hypothesis_lines"] +
                  corpus["malformed_reference_lines"])
        add("max_malformed", actual, max_malformed,
            actual <= max_malformed, "lines")

    if baseline_report is not None:
        for metric, limit in (("cer", max_cer_regression), ("wer", max_wer_regression)):
            if limit is None:
                continue
            current = metrics.get(metric, {}).get("percent")
            baseline = baseline_report["metrics"].get(metric, {}).get("percent")
            delta = float("inf") if current is None or baseline is None else current - baseline
            add(f"max_{metric}_regression", delta, limit, delta <= limit,
                "percentage_points")

    configured = enabled or bool(checks)
    return {
        "configured": configured,
        "passed": all(check["passed"] for check in checks) if configured else None,
        "checks": checks,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    corpus = report["corpus"]
    metrics = report["metrics"]
    gate = report.get("gate", {})
    lines = [
        "# ASR Accuracy Report",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Expected utterances | {corpus['expected_utterances']} |",
        f"| Coverage | {corpus['coverage_percent']:.4f}% |",
        f"| Missing hypotheses | {corpus['missing_hypotheses']} |",
        f"| Empty hypotheses | {corpus['empty_hypotheses']} |",
        f"| Extra hypotheses | {corpus['extra_hypotheses']} |",
    ]
    for metric in ("cer", "wer"):
        if metric in metrics:
            value = metrics[metric]["percent"]
            lines.append(f"| {metric.upper()} | {value:.4f}% |")
    if gate.get("configured"):
        lines.extend(["", f"**Gate: {'PASS' if gate['passed'] else 'FAIL'}**", ""])
        lines.extend(["| Check | Actual | Limit | Result |", "|---|---:|---:|---|"])
        for check in gate["checks"]:
            lines.append(
                f"| {check['name']} | {check['actual']:.4f} | "
                f"{check['limit']:.4f} | {'PASS' if check['passed'] else 'FAIL'} |"
            )
    lines.extend(["", "## Worst Cases", ""])
    for item in report["worst_cases"]:
        lines.extend([
            f"### {item['id']} ({item['metric'].upper()} {item['percent']:.2f}%)",
            "",
            f"- REF: {item['reference']}",
            f"- HYP: {item['hypothesis'] or '[EMPTY]'}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def print_report(report: Mapping[str, object]) -> None:
    corpus = report["corpus"]
    print("=" * 58)
    print("  ASR Accuracy Evaluation")
    print("=" * 58)
    print(f"  Expected utterances:  {corpus['expected_utterances']}")
    print(f"  Coverage:             {corpus['coverage_percent']:.4f}%")
    print(f"  Missing hypotheses:   {corpus['missing_hypotheses']}")
    print(f"  Empty hypotheses:     {corpus['empty_hypotheses']}")
    print(f"  Extra hypotheses:     {corpus['extra_hypotheses']}")
    for name, metric in report["metrics"].items():
        print(
            f"  {name.upper()}:                  {metric['percent']:.4f}% "
            f"(S={metric['substitutions']} D={metric['deletions']} "
            f"I={metric['insertions']} N={metric['reference_units']})"
        )
    gate = report.get("gate", {})
    if gate.get("configured"):
        print("-" * 58)
        for check in gate["checks"]:
            verdict = "PASS" if check["passed"] else "FAIL"
            print(
                f"  [{verdict}] {check['name']}: "
                f"actual={check['actual']:.4f} limit={check['limit']:.4f}"
            )
        print(f"  GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    print("=" * 58)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and gate ASR CER/WER.")
    parser.add_argument("hypothesis", help="TSV: utterance_id<TAB>recognized text")
    parser.add_argument("reference", help="AISHELL-style: utterance_id reference text")
    parser.add_argument("--expected-ids",
                        help="Complete expected ID list; first column is used.")
    parser.add_argument("--metrics", choices=("cer", "wer", "both"), default="cer")
    parser.add_argument("--baseline", help="Optional baseline hypothesis TSV.")
    parser.add_argument("--gate", action="store_true",
                        help="Enable strict coverage/empty/format checks.")
    parser.add_argument("--max-cer", type=float, help="Maximum corpus CER in percent.")
    parser.add_argument("--max-wer", type=float, help="Maximum corpus WER in percent.")
    parser.add_argument("--min-coverage", type=float, help="Minimum coverage in percent.")
    parser.add_argument("--max-empty-rate", type=float,
                        help="Maximum missing+empty rate in percent.")
    parser.add_argument("--max-missing", type=int)
    parser.add_argument("--max-extra", type=int)
    parser.add_argument("--max-duplicates", type=int)
    parser.add_argument("--max-malformed", type=int)
    parser.add_argument("--max-cer-regression", type=float,
                        help="Maximum CER increase versus baseline, percentage points.")
    parser.add_argument("--max-wer-regression", type=float,
                        help="Maximum WER increase versus baseline, percentage points.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--worst-cases", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metric_names = ("cer", "wer") if args.metrics == "both" else (args.metrics,)
    if (args.max_cer is not None or args.max_cer_regression is not None) and \
            "cer" not in metric_names:
        print("ERROR: CER thresholds require --metrics cer or both", file=sys.stderr)
        return 2
    if (args.max_wer is not None or args.max_wer_regression is not None) and \
            "wer" not in metric_names:
        print("ERROR: WER thresholds require --metrics wer or both", file=sys.stderr)
        return 2
    if (args.max_cer_regression is not None or
            args.max_wer_regression is not None) and not args.baseline:
        print("ERROR: regression thresholds require --baseline", file=sys.stderr)
        return 2
    for name, value in (("max_cer", args.max_cer), ("max_wer", args.max_wer),
                        ("max_empty_rate", args.max_empty_rate),
                        ("min_coverage", args.min_coverage)):
        if value is not None and not 0.0 <= value <= 100.0:
            print(f"ERROR: --{name.replace('_', '-')} must be in [0, 100]",
                  file=sys.stderr)
            return 2
    references = load_reference_records(args.reference)
    hypotheses = load_hypothesis_records(args.hypothesis)
    thresholds_present = any(value is not None for value in (
        args.max_cer, args.max_wer, args.min_coverage, args.max_empty_rate,
        args.max_missing, args.max_extra, args.max_duplicates,
        args.max_malformed, args.max_cer_regression, args.max_wer_regression,
    ))
    expected_ids = None
    expected_duplicates: List[str] = []
    if args.expected_ids:
        expected_ids, expected_duplicates = load_expected_ids(args.expected_ids)
        if expected_duplicates:
            print(
                f"ERROR: expected ID list contains {len(expected_duplicates)} duplicates",
                file=sys.stderr,
            )
            return 2
    elif args.gate or thresholds_present:
        print(
            "ERROR: strict accuracy gates require --expected-ids so missing "
            "hypotheses cannot disappear from the denominator",
            file=sys.stderr,
        )
        return 2
    else:
        # Preserve the legacy evaluator's exploratory behavior. Release gates
        # never use this path; they require an explicit workload manifest.
        expected_ids = sorted(set(references.texts) & set(hypotheses.texts))

    try:
        report = evaluate_corpus(
            references, hypotheses, expected_ids, metric_names, args.worst_cases)
        baseline_report = None
        if args.baseline:
            baseline_report = evaluate_corpus(
                references,
                load_hypothesis_records(args.baseline),
                expected_ids,
                metric_names,
                0,
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report["gate"] = evaluate_gate(
        report,
        enabled=args.gate,
        max_cer=args.max_cer,
        max_wer=args.max_wer,
        min_coverage=args.min_coverage,
        max_empty_rate=args.max_empty_rate,
        max_missing=args.max_missing,
        max_extra=args.max_extra,
        max_duplicates=args.max_duplicates,
        max_malformed=args.max_malformed,
        baseline_report=baseline_report,
        max_cer_regression=args.max_cer_regression,
        max_wer_regression=args.max_wer_regression,
    )
    if thresholds_present and not report["gate"]["configured"]:
        raise AssertionError("thresholds must configure the gate")

    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")

    gate = report["gate"]
    return 1 if gate["configured"] and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
