import os
import json
import re
import string
from collections import defaultdict


# ----------------------------
# Basic text normalization
# ----------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_free_text(s: str) -> str:
    s = clean_text(s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Extract final answer content
# Supports:
#   \boxed{B}
#   \boxed{Yes}
#   Answer: A
#   Answer: Yes
#   tail yes/no/true/false
#   tail single option letter
# ----------------------------
def extract_answer_text(pred: str) -> str:
    """
    Priority:
    1) last \\boxed{...} / boxed{...}
    2) last Answer: ...
    3) tail yes/no/true/false
    4) tail single option letter
    5) fallback: full text
    """
    if pred is None:
        return ""

    s = str(pred).strip()

    # 1) boxed answer, e.g. \boxed{B}, \boxed{Yes}
    boxed_matches = re.findall(
        r"(?:\\boxed|boxed)\s*\{\s*([^{}]+?)\s*\}",
        s,
        flags=re.IGNORECASE,
    )
    if boxed_matches:
        return clean_text(boxed_matches[-1])

    # 2) explicit Answer: ...
    answer_matches = re.findall(
        r"answer\s*[:\-]\s*(.+)",
        s,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        return clean_text(answer_matches[-1])

    # 3) near the end, look for yes/no/true/false
    tail = s[-120:]
    yn_matches = re.findall(
        r"\b(yes|no|true|false)\b",
        tail,
        flags=re.IGNORECASE,
    )
    if yn_matches:
        return clean_text(yn_matches[-1])

    # 4) near the end, look for single letter option
    letter_matches = re.findall(r"\b([A-Da-d])\b", tail)
    if letter_matches:
        return clean_text(letter_matches[-1])

    return clean_text(s)


# ----------------------------
# yes/no or true/false normalization
# map true -> yes, false -> no
# ----------------------------
def normalize_binary_label(s: str):
    ans = extract_answer_text(s)
    ans_norm = normalize_free_text(ans)

    mapping = {
        "yes": "yes",
        "true": "yes",
        "no": "no",
        "false": "no",
    }

    if ans_norm in mapping:
        return mapping[ans_norm]

    # e.g. "yes, the vehicle should stop"
    m = re.match(r"^(yes|no|true|false)\b", ans_norm, flags=re.IGNORECASE)
    if m:
        return mapping[m.group(1).lower()]

    # fallback: last binary token in text
    tokens = re.findall(r"\b(yes|no|true|false)\b", ans_norm, flags=re.IGNORECASE)
    if tokens:
        return mapping[tokens[-1].lower()]

    return None


# ----------------------------
# multiple choice parsing
# Supports:
#   "A"
#   "Answer: A"
#   "Answer: B. Right"
#   "\\boxed{B}"
#   "\\boxed{Drive on the right side}"
#   direct option text
# ----------------------------
def parse_choice_prediction(pred: str, options):
    """
    Return predicted option text if can parse, else normalized raw answer text.
    """
    ans = extract_answer_text(pred)
    if not ans:
        return ""

    normalized_options = [normalize_free_text(x) for x in options]

    # 1) single-letter option
    # match A / B / C / D / A. xxx / B) xxx / (C)
    m = re.match(
        r"^\(?([a-z])\)?(?:[\.\):\-]\s*(.*))?$",
        ans,
        flags=re.IGNORECASE,
    )
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return normalized_options[idx]

    # 2) patterns like "answer is B" / "option B"
    letter_mentions = re.findall(
        r"(?:answer\s*(?:is|:)?|option)\s*\(?([A-Da-d])\)?\b",
        ans,
        flags=re.IGNORECASE,
    )
    if letter_mentions:
        idx = ord(letter_mentions[-1].upper()) - ord("A")
        if 0 <= idx < len(options):
            return normalized_options[idx]

    # 3) exact text match against one option
    ans_norm = normalize_free_text(ans)
    for opt in normalized_options:
        if ans_norm == opt:
            return opt

    # 4) substring match against option text
    for opt in normalized_options:
        if opt and opt in ans_norm:
            return opt

    return ans_norm


# ----------------------------
# token F1 for explanation
# ----------------------------
def token_f1(pred: str, gt: str) -> float:
    pred = extract_answer_text(pred)
    pred_tokens = normalize_free_text(pred).split()
    gt_tokens = normalize_free_text(gt).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_count = {}
    gt_count = {}

    for t in pred_tokens:
        pred_count[t] = pred_count.get(t, 0) + 1
    for t in gt_tokens:
        gt_count[t] = gt_count.get(t, 0) + 1

    common = 0
    for t in pred_count:
        if t in gt_count:
            common += min(pred_count[t], gt_count[t])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ----------------------------
# single-example scoring
# ----------------------------
def score_item(item, task=None, country=None):
    qtype = item.get("type", "").strip().lower()
    gt = item.get("gt", "")
    pred = item.get("pred", "")
    options = item.get("options", []) or []

    # Keep your original country-based override logic
    if country == "cn" or country == "us":
        if "left" in gt.lower():
            gt = "right"
    else:
        if "right" in gt.lower():
            gt = "left"

    gt_norm = normalize_free_text(gt)

    # yes/no
    if qtype == "yes/no":
        gt_bin = normalize_binary_label(gt)
        pred_bin = normalize_binary_label(pred)
        return 1.0 if (gt_bin is not None and pred_bin == gt_bin) else 0.0

    # true/false
    if qtype == "true/false":
        gt_bin = normalize_binary_label(gt)
        pred_bin = normalize_binary_label(pred)
        return 1.0 if (gt_bin is not None and pred_bin == gt_bin) else 0.0

    # multiple choice
    if qtype == "multiple_choice":
        # if options exist, do real MC parsing
        if len(options) > 0:
            pred_norm = parse_choice_prediction(pred, options)
            return 1.0 if pred_norm == gt_norm else 0.0

        # fallback: if options are empty but GT is binary, treat as binary QA
        gt_bin = normalize_binary_label(gt)
        pred_bin = normalize_binary_label(pred)
        if gt_bin is not None:
            return 1.0 if pred_bin == gt_bin else 0.0

        # otherwise fallback to normalized exact match
        pred_norm = normalize_free_text(extract_answer_text(pred))
        return 1.0 if pred_norm == gt_norm else 0.0

    # explanation
    if qtype == "one-sentence explanation":
        return token_f1(pred, gt)

    # generic binary fallback for empty-option cases
    if len(options) == 0:
        gt_bin = normalize_binary_label(gt)
        pred_bin = normalize_binary_label(pred)
        if gt_bin is not None:
            return 1.0 if pred_bin == gt_bin else 0.0

    # fallback: exact normalized match
    return 1.0 if normalize_free_text(extract_answer_text(pred)) == gt_norm else 0.0


# ----------------------------
# main scoring function
# ----------------------------
def compute_scores(results: dict):
    """
    Return:
    {
        country: {
            task: avg_score
        }
    }
    """
    score_sum = defaultdict(lambda: defaultdict(float))
    score_cnt = defaultdict(lambda: defaultdict(int))

    for country, samples in results.items():
        for item in samples:
            task = item.get("task", "unknown")
            s = score_item(item, task, country)
            score_sum[country][task] += s
            score_cnt[country][task] += 1

    out = {}
    all_tasks = ["perception", "prediction", "planning", "regional"]

    for country in results:
        out[country] = {}
        for task in all_tasks:
            cnt = score_cnt[country][task]
            out[country][task] = score_sum[country][task] / cnt if cnt > 0 else 0.0

    return out


# ----------------------------
# optional detailed report
# ----------------------------
def compute_scores_with_details(results: dict):
    score_sum = defaultdict(lambda: defaultdict(float))
    score_cnt = defaultdict(lambda: defaultdict(int))
    details = defaultdict(lambda: defaultdict(list))

    for country, samples in results.items():
        for item in samples:
            task = item.get("task", "unknown")
            s = score_item(item, task, country)

            score_sum[country][task] += s
            score_cnt[country][task] += 1

            details[country][task].append({
                "question": item.get("question", ""),
                "type": item.get("type", ""),
                "gt": item.get("gt", ""),
                "pred": item.get("pred", ""),
                "score": s,
            })

    out = {}
    all_tasks = ["perception", "prediction", "planning", "regional"]

    for country in results:
        out[country] = {}
        for task in all_tasks:
            cnt = score_cnt[country][task]
            out[country][task] = score_sum[country][task] / cnt if cnt > 0 else 0.0

    return out, details


if __name__ == "__main__":
    file_path = "./results/qwen2.5vl_r_output.json"

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for country in results:
        print(country, len(results[country]))

    scores, details = compute_scores_with_details(results)

    print("=== Scores ===")
    print("results from file:", file_path)
    print(json.dumps(scores, indent=2, ensure_ascii=False))

    # optional save
    with open("scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    with open("score_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)