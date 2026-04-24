#!/usr/bin/env python3
"""
tools/eval_cer_v2.py
带数字归一化的 CER 评测

用法:
  python3 tools/eval_cer_v2.py results.txt transcript.txt
"""

import sys
import re
from collections import defaultdict

DIGIT_MAP = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}

def normalize_digits(text):
    """把阿拉伯数字转成中文数字"""
    result = []
    for ch in text:
        if ch in DIGIT_MAP:
            result.append(DIGIT_MAP[ch])
        else:
            result.append(ch)
    return ''.join(result)

def load_transcript(path):
    refs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[1].replace(' ', '')
            refs[utt_id] = text
    return refs

def load_results(path):
    hyps = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[1]
            if text == '[EMPTY]':
                text = ''
            hyps[utt_id] = text
    return hyps

def normalize_text(text):
    text = re.sub(r'[，。！？、：；""\'\'《》【】（）()\[\]{}<>,.!?;:\"]', '', text)
    text = text.replace(' ', '')
    text = text.lower()
    text = normalize_digits(text)
    return text

def edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                dele = dp[i - 1][j]
                del_cost = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
                dp[i][j] = min(sub_cost, del_cost, ins_cost, key=lambda x: x[0])
    return dp[n][m]

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 eval_cer_v2.py <results.txt> <transcript.txt>")
        sys.exit(1)

    results_path = sys.argv[1]
    transcript_path = sys.argv[2]

    print("=" * 50)
    print("  AISHELL-1 CER Evaluation (with digit normalization)")
    print("=" * 50)
    print()

    refs = load_transcript(transcript_path)
    print(f"Transcript: {len(refs)} utterances")
    hyps = load_results(results_path)
    print(f"Results: {len(hyps)} utterances")

    common_ids = sorted(set(refs.keys()) & set(hyps.keys()))
    print(f"Common: {len(common_ids)}")

    if not common_ids:
        print("ERROR: No common utterances!")
        sys.exit(1)

    total_S = total_D = total_I = total_N = 0
    n_perfect = 0
    n_empty = 0
    errors = []

    for utt_id in common_ids:
        ref = normalize_text(refs[utt_id])
        hyp = normalize_text(hyps[utt_id])
        if not ref:
            continue
        if not hyp:
            n_empty += 1
            total_D += len(ref)
            total_N += len(ref)
            continue
        _, S, D, I = edit_distance(list(ref), list(hyp))
        total_S += S
        total_D += D
        total_I += I
        total_N += len(ref)
        cer = (S + D + I) / len(ref)
        if cer == 0:
            n_perfect += 1
        if cer > 0.3:
            errors.append((utt_id, cer, ref, hyp))

    total_errors = total_S + total_D + total_I
    cer = total_errors / total_N if total_N > 0 else 0

    print()
    print("=" * 50)
    print("  Results (digits normalized: 2013 → 二零一三)")
    print("=" * 50)
    print(f"  Utterances:     {len(common_ids)}")
    print(f"  Characters:     {total_N}")
    print(f"  Substitutions:  {total_S}")
    print(f"  Deletions:      {total_D}")
    print(f"  Insertions:     {total_I}")
    print(f"  Total errors:   {total_errors}")
    print()
    print(f"  *** CER = {cer * 100:.2f}% ***")
    print()
    print(f"  Perfect:        {n_perfect} ({n_perfect * 100.0 / len(common_ids):.1f}%)")
    print(f"  Empty:          {n_empty}")
    print("=" * 50)

    if errors:
        errors.sort(key=lambda x: -x[1])
        print()
        print("Worst cases (CER > 30%):")
        for utt_id, cer_val, ref, hyp in errors[:10]:
            print(f"  {utt_id}: CER={cer_val * 100:.1f}%")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")
            print()

if __name__ == '__main__':
    main()