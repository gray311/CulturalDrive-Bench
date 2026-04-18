"""
Validate eval set using Qwen3-VL-235B via Bedrock API.

For each example, send image + question + GT answer + rule to VLM,
ask it to judge if the answer is correct given the image.

Usage:
    python validate_eval.py [--max_items N] [--resume]
"""

import os
import sys
import json
import ast
import base64
import argparse
import mimetypes
from pathlib import Path
from typing import List
from tqdm import tqdm

# Credentials must be supplied externally — never committed.
# Set before running:
#   export AWS_BEARER_TOKEN_BEDROCK=...
#   export AWS_REGION=us-east-2
for _v in ('AWS_BEARER_TOKEN_BEDROCK', 'AWS_REGION'):
    if not os.environ.get(_v):
        raise RuntimeError(f'Missing required environment variable: {_v}')

import boto3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH   = str(_REPO_ROOT / 'culturebenchmark_eval.json')
OUTPUT_PATH = str(_REPO_ROOT / 'eval_validation_results.json')
MODEL_ID = "qwen.qwen3-vl-235b-a22b"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_handbook import (
    china_traffic_handbook, india_traffic_handbook, japan_traffic_handbook,
    singapore_traffic_handbook, uk_traffic_handbook, us_traffic_handbook,
)

HANDBOOKS = {
    "cn": china_traffic_handbook, "ind": india_traffic_handbook,
    "jp": japan_traffic_handbook, "sg": singapore_traffic_handbook,
    "uk": uk_traffic_handbook, "us": us_traffic_handbook,
}

COUNTRY_NAMES = {
    "cn": "China", "jp": "Japan", "uk": "United Kingdom",
    "sg": "Singapore", "ind": "India", "us": "United States",
}

# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

# ---------------------------------------------------------------------------
# Validation prompt
# ---------------------------------------------------------------------------
VALIDATE_PROMPT = """You are a traffic scene expert validating a driving benchmark dataset. You must carefully examine the provided image(s) and judge whether the given ground-truth (GT) answer is correct.

You are given:
- One or more driving scene images from {country}
- A multiple-choice question (category: {category})
- The ground-truth (GT) answer
- The relevant traffic rule for this country (if available)

Your task: Judge whether the GT answer is CORRECT by checking ALL of the following:

**1. Visual Consistency** — Does the GT answer match what is ACTUALLY VISIBLE in the image?
   - Colors, shapes, positions of objects (vehicles, signs, signals, plates, markings)
   - If the question asks about a specific object, verify the object exists and matches the described attributes
   - Do NOT assume — look at the image carefully

**2. Factual Correctness** — Is the GT answer factually accurate for this country?
   - Traffic rules, sign meanings, driving conventions, legal requirements
   - Country-specific details (e.g., plate colors, sign shapes, road markings vary by country)

**3. Question-Answer Alignment** — Does the GT answer actually address what the question is asking?
   - For PERCEPTION: Does the answer correctly describe what is visible (colors, shapes, positions)?
   - For PREDICTION: Is the predicted behavior reasonable given the scene context and local driving norms?
   - For PLANNING: Is the suggested action legally correct and safe given the traffic situation in the image?
   - For REGION: Is the cultural/regional traffic knowledge accurate for this specific country?

**4. Option Quality** — Are the distractors (wrong options) clearly distinguishable from the GT?
   - If multiple options could be correct, mark as AMBIGUOUS
   - If the GT is wrong but another option is correct, mark as INCORRECT

Respond in this exact JSON format:
{{
  "verdict": "CORRECT" or "INCORRECT" or "AMBIGUOUS",
  "confidence": 0.0 to 1.0,
  "reason": "Brief explanation",
  "suggested_fix": "If INCORRECT, state which option (e.g., 'B. ...') should be the correct answer and why. Otherwise null."
}}

Country: {country}
Category: {category}
Question: {question}
Options:
{options}
GT Answer: {gt_answer}
Traffic Rule: {rule}

Now examine the image(s) carefully and validate."""


def encode_image(path: str) -> dict:
    """Encode image for Bedrock API."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    fmt = mime.split("/")[1]
    if fmt == "jpg":
        fmt = "jpeg"
    with open(path, "rb") as f:
        data = f.read()
    return {
        "image": {
            "format": fmt,
            "source": {"bytes": data}
        }
    }


def validate_item(item: dict) -> dict:
    """Send one item to Qwen3-VL for validation."""
    country = item.get("country", "")
    question = item.get("question", "")
    options = item.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options)
    answer = item.get("answer", "").strip().upper()
    gt_idx = ord(answer) - ord("A") if answer else -1
    gt_text = options[gt_idx] if 0 <= gt_idx < len(options) else answer
    rule_refs = item.get("rule_reference", [])
    if isinstance(rule_refs, str):
        try:
            rule_refs = ast.literal_eval(rule_refs)
        except:
            rule_refs = []
    explanation = item.get("explanation", "")

    # Format options text
    options_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))

    # Get rule text
    handbook = HANDBOOKS.get(country, "")
    rule_text = ""
    if rule_refs and handbook:
        for ref in rule_refs:
            for line in handbook.strip().split("\n"):
                if line.startswith(ref.strip()):
                    rule_text += line + "\n"
    if not rule_text:
        rule_text = "No specific rule referenced."

    # Build prompt
    category = item.get("question_category", "unknown")
    prompt = VALIDATE_PROMPT.format(
        country=COUNTRY_NAMES.get(country, country),
        category=category,
        question=question,
        options=options_text,
        gt_answer=f"{answer}. {gt_text}",
        rule=rule_text,
    )

    # Build message content with images
    content = []

    # Add images (last 3)
    image_paths = item.get("image_path", [])
    if isinstance(image_paths, str):
        image_paths = ast.literal_eval(image_paths)
    for p in image_paths[-3:]:
        try:
            content.append(encode_image(p))
        except Exception as e:
            pass

    content.append({"text": prompt})

    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        output = response["output"]["message"]["content"][0]["text"]

        # Parse JSON from response
        # Try to find JSON block
        import re
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"verdict": "PARSE_ERROR", "confidence": 0, "reason": output, "suggested_fix": None}

    except Exception as e:
        result = {"verdict": "API_ERROR", "confidence": 0, "reason": str(e), "suggested_fix": None}

    result["id"] = item.get("id")
    result["country"] = country
    result["category"] = item.get("question_category", "")
    result["question"] = question
    result["gt_answer"] = f"{answer}. {gt_text}"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data = json.load(open(EVAL_PATH))
    print(f"Loaded {len(data)} eval items")

    # Resume
    results = []
    done_ids = set()
    if args.resume and os.path.exists(OUTPUT_PATH):
        results = json.load(open(OUTPUT_PATH))
        done_ids = {r["id"] for r in results}
        print(f"Resuming from {len(done_ids)} done items")

    items = [d for d in data if d["id"] not in done_ids]
    if args.max_items:
        items = items[:args.max_items]

    print(f"Validating {len(items)} items...")

    for item in tqdm(items, desc="Validating"):
        result = validate_item(item)
        results.append(result)

        # Save every 50 items
        if len(results) % 50 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in results)
    print(f"\n=== Validation Summary ===")
    print(f"Total: {len(results)}")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c} ({c/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
