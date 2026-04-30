#!/usr/bin/env python3
"""
tools/eval_cer.py
AISHELL-1 CER 评测脚本

用法:
  python3 tools/eval_cer.py results.txt ~/data_aishell/transcript/aishell_transcript_v0.8.txt

输入:
  results.txt:  C++ 推理输出，格式为 "utterance_id\t识别文本"
  transcript:   AISHELL-1 标注文件，格式为 "utterance_id 字1 字2 字3 ..."

输出:
  总体 CER，以及错误分析
"""

import sys
import re
import unicodedata


def load_transcript(path):
    """加载 AISHELL-1 标注文件"""
    refs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # 按第一个空格分割
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            # 标注格式: "字 字 字 字"，去掉空格得到连续文本
            text = parts[1].replace(' ', '')
            refs[utt_id] = text
    return refs


def load_results(path):
    """加载 C++ 推理结果"""
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


def digits_to_chinese(text):
    """把数字串逐位转成中文数字，适合年份/编号。"""
    return "".join(CHINESE_DIGITS.get(ch, ch) for ch in text)


def int_to_chinese(num_text):
    """把 0-99 的整数转成常见中文表达，适合月/日。"""
    value = int(num_text)
    if value < 10:
        return CHINESE_DIGITS[str(value)]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + CHINESE_DIGITS[str(value % 10)]

    tens = value // 10
    ones = value % 10
    if ones == 0:
        return CHINESE_DIGITS[str(tens)] + "十"
    return CHINESE_DIGITS[str(tens)] + "十" + CHINESE_DIGITS[str(ones)]


def normalize_numbers(text):
    """数字归一化，尽量贴近中文 ASR 常见标注。"""

    def repl_ymd(match):
        year = digits_to_chinese(match.group(1))
        month = int_to_chinese(match.group(2))
        day = int_to_chinese(match.group(3))
        suffix = match.group(4) or ""
        return f"{year}年{month}月{day}{suffix}"

    def repl_md(match):
        month = int_to_chinese(match.group(1))
        day = int_to_chinese(match.group(2))
        suffix = match.group(3) or ""
        return f"{month}月{day}{suffix}"

    text = re.sub(r"(\d{2,4})年(\d{1,2})月(\d{1,2})(日|号)?", repl_ymd, text)
    text = re.sub(r"(\d{1,2})月(\d{1,2})(日|号)?", repl_md, text)
    text = re.sub(r"\d+", lambda m: digits_to_chinese(m.group(0)), text)
    return text


def normalize_text(text):
    """文本标准化: 统一全角半角、数字格式、去除标点和空格。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = normalize_numbers(text)
    text = text.translate(PUNCT_TRANSLATION)
    text = "".join(text.split())
    return text


def edit_distance(ref, hyp):
    """计算编辑距离（Levenshtein distance），返回 (S, D, I)"""
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = (cost, S, D, I)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # 全部删除
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # 全部插入

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 替换
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                # 删除（ref 有，hyp 没有）
                dele = dp[i - 1][j]
                del_cost = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])

                # 插入（hyp 多了）
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)

                dp[i][j] = min(sub_cost, del_cost, ins_cost, key=lambda x: x[0])

    return dp[n][m]  # (total_errors, S, D, I)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 eval_cer.py <results.txt> <transcript.txt>")
        sys.exit(1)

    results_path = sys.argv[1]
    transcript_path = sys.argv[2]

    print("=" * 50)
    print("  AISHELL-1 CER Evaluation")
    print("=" * 50)
    print()

    # 加载数据
    print(f"Loading transcript: {transcript_path}")
    refs = load_transcript(transcript_path)
    print(f"  {len(refs)} utterances")

    print(f"Loading results: {results_path}")
    hyps = load_results(results_path)
    print(f"  {len(hyps)} utterances")
    print()

    # 找到共有的 utterance
    common_ids = sorted(set(refs.keys()) & set(hyps.keys()))
    print(f"Common utterances: {len(common_ids)}")

    if not common_ids:
        print("ERROR: No common utterances found!")
        print("  Check that utterance IDs match between results and transcript.")
        print(f"  Sample result IDs: {list(hyps.keys())[:5]}")
        print(f"  Sample transcript IDs: {list(refs.keys())[:5]}")
        sys.exit(1)

    # 计算 CER
    total_S = 0
    total_D = 0
    total_I = 0
    total_N = 0
    n_empty = 0
    n_perfect = 0

    errors_by_utt = []

    for utt_id in common_ids:
        ref = normalize_text(refs[utt_id])
        hyp = normalize_text(hyps[utt_id])

        if not ref:
            continue

        if not hyp:
            n_empty += 1
            total_D += len(ref)
            total_N += len(ref)
            errors_by_utt.append((utt_id, 1.0, ref, hyp))
            continue

        _, S, D, I = edit_distance(list(ref), list(hyp))

        total_S += S
        total_D += D
        total_I += I
        total_N += len(ref)

        cer = (S + D + I) / len(ref)
        if cer == 0:
            n_perfect += 1

        if cer > 0.5:  # 记录高错误率的案例
            errors_by_utt.append((utt_id, cer, ref, hyp))

    # 输出结果
    total_errors = total_S + total_D + total_I
    cer = total_errors / total_N if total_N > 0 else 0

    print()
    print("=" * 50)
    print("  Results")
    print("=" * 50)
    print(f"  Utterances evaluated: {len(common_ids)}")
    print(f"  Total characters:     {total_N}")
    print(f"  Substitutions (S):    {total_S}")
    print(f"  Deletions (D):        {total_D}")
    print(f"  Insertions (I):       {total_I}")
    print(f"  Total errors:         {total_errors}")
    print()
    print(f"  *** CER = {cer * 100:.2f}% ***")
    print()
    print(f"  Perfect matches:      {n_perfect} ({n_perfect * 100.0 / len(common_ids):.1f}%)")
    print(f"  Empty outputs:        {n_empty}")
    print("=" * 50)

    # 输出最差的几个案例（用于排查问题）
    if errors_by_utt:
        errors_by_utt.sort(key=lambda x: -x[1])
        print()
        print("Worst cases (CER > 50%):")
        for utt_id, cer_val, ref, hyp in errors_by_utt[:10]:
            print(f"  {utt_id}: CER={cer_val * 100:.1f}%")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")
            print()


if __name__ == '__main__':
    main()
