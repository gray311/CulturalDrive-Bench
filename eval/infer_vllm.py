"""
Unified vLLM inference script for CulturalDrive benchmark.
All models use vLLM for fast inference.

Supported models (--model):
  llava       → llava-hf/llava-v1.6-mistral-7b-hf
  qwen25vl    → Qwen/Qwen2.5-VL-7B-Instruct
  qwen3vl     → Qwen/Qwen3-VL-8B-Instruct
  internvl3   → OpenGVLab/InternVL3-8B
  internvl35  → OpenGVLab/InternVL3_5-8B
  gemma3      → google/gemma-3-12b-it
  llama32     → meta-llama/Llama-3.2-11B-Vision-Instruct

Three evaluation settings (--setting):
  direct      - output the answer letter only
  reasoning   - CoT: infer country, cite cultural traffic rules, visual reasoning
  rule_given  - provide the relevant handbook rule(s), then reason step-by-step

Usage:
  python infer_vllm.py --model qwen3vl --setting direct
  python infer_vllm.py --model gemma3 --setting reasoning
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Handbook
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_handbook import (
    china_traffic_handbook, india_traffic_handbook, japan_traffic_handbook,
    singapore_traffic_handbook, uk_traffic_handbook, us_traffic_handbook,
)

HANDBOOKS: Dict[str, str] = {
    "cn": china_traffic_handbook, "ind": india_traffic_handbook,
    "jp": japan_traffic_handbook, "sg": singapore_traffic_handbook,
    "uk": uk_traffic_handbook,   "us": us_traffic_handbook,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # vLLM-compatible models
    "llava":      {"path": "llava-hf/llava-v1.6-mistral-7b-hf",       "max_images": 1, "engine": "vllm"},
    "qwen25vl":   {"path": "Qwen/Qwen2.5-VL-7B-Instruct",             "max_images": 6, "engine": "vllm"},
    "qwen3vl":    {"path": "Qwen/Qwen3-VL-8B-Instruct",               "max_images": 6, "engine": "vllm", "enable_thinking": False},
    "internvl3":  {"path": "OpenGVLab/InternVL3-8B",                   "max_images": 6, "engine": "vllm"},
    "gemma3":     {"path": "google/gemma-3-12b-it",                    "max_images": 6, "engine": "vllm"},
    # HF/lmdeploy models (not vLLM compatible)
    "llama32":    {"path": "meta-llama/Llama-3.2-11B-Vision-Instruct", "max_images": 1, "engine": "hf"},
    "internvl35": {"path": "OpenGVLab/InternVL3_5-8B",                 "max_images": 6, "engine": "lmdeploy", "chat_template": "internvl2_5"},
}

# ---------------------------------------------------------------------------
# Handbook utilities
# ---------------------------------------------------------------------------
def extract_rules(handbook_text: str, rule_refs: List[str]) -> str:
    if not rule_refs:
        return ""
    extracted = []
    for ref in rule_refs:
        pattern = rf"({re.escape(ref)}\s.+?)(?=\nS\d+\s|\Z)"
        match = re.search(pattern, handbook_text, re.DOTALL)
        if match:
            extracted.append(match.group(1).strip())
        else:
            for line in handbook_text.splitlines():
                if line.startswith(ref + " ") or line.startswith(ref + ":"):
                    extracted.append(line.strip())
                    break
    return "\n\n".join(extracted)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_prompt_direct(question: str, options: List[str]) -> str:
    option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    return (
        "Answer the question using only the image. Choose exactly one option.\n"
        f"Question: {question}\n"
        f"Options:\n{option_text}\n"
        "Output only the answer letter (A/B/C/D).\n"
    )

def build_prompt_reasoning(question: str, options: List[str]) -> str:
    option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    return (
        "Step 1 — Geographic inference: Based on road infrastructure, traffic signs, "
        "lane markings, and environmental cues in the image, identify the most likely "
        "country or region where this scene was captured.\n"
        "Step 2 — Cultural traffic rule: State the relevant traffic rule(s) for that "
        "country that apply to this question.\n"
        "Step 3 — Visual reasoning: Describe what you observe in the image that is "
        "relevant to answering the question.\n"
        "Step 4 — Answer: Choose exactly one option.\n"
        f"Question: {question}\n"
        f"Options:\n{option_text}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    )

def build_prompt_rule_given(question: str, options: List[str], rule_text: str) -> str:
    option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    rule_section = f"Relevant traffic rule(s):\n{rule_text}\n\n" if rule_text else ""
    return (
        f"{rule_section}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_text}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    )

def build_prompt(item: Dict, setting: str) -> str:
    question = item["question"]
    options  = item["options"]
    country  = item["country"]
    if setting == "direct":
        return build_prompt_direct(question, options)
    if setting == "reasoning":
        return build_prompt_reasoning(question, options)
    if setting == "rule_given":
        rule_refs = item.get("rule_reference", [])
        handbook  = HANDBOOKS.get(country, "")
        rule_text = extract_rules(handbook, rule_refs) if rule_refs else ""
        return build_prompt_rule_given(question, options, rule_text)
    raise ValueError(f"Unknown setting: {setting}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def open_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

# ---------------------------------------------------------------------------
# vLLM unified loader & inference
# ---------------------------------------------------------------------------
def load_model(model_name: str, model_path: str):
    """Load any VLM via vLLM."""
    from vllm import LLM
    from transformers import AutoProcessor

    cfg = MODEL_REGISTRY[model_name]
    max_images = cfg.get("max_images", 6)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=16384,
        limit_mm_per_prompt={"image": max_images},
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=64,
        disable_mm_preprocessor_cache=True,
        mm_processor_cache_gb=0,
        enforce_eager=True,
    )

    return processor, llm


def build_chat_prompt(processor, model_name: str, images: List[Image.Image], prompt: str) -> str:
    """Build chat-formatted prompt using model's processor."""
    cfg = MODEL_REGISTRY[model_name]

    image_content = [{"type": "image"} for _ in images]
    messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}]

    kwargs = {}
    if cfg.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return text


def infer_vllm_batch(processor, llm, model_name: str,
                     batch_image_paths: List[List[str]],
                     batch_prompts: List[str],
                     max_new_tokens: int) -> List[str]:
    """Batched vLLM inference — lets vLLM do continuous batching for high GPU util."""
    from vllm import SamplingParams

    cfg = MODEL_REGISTRY[model_name]
    max_img = cfg.get("max_images", 6)

    requests = []
    for img_paths, prompt in zip(batch_image_paths, batch_prompts):
        if len(img_paths) > max_img:
            img_paths = img_paths[-max_img:]
        images = [open_image(p) for p in img_paths]
        text = build_chat_prompt(processor, model_name, images, prompt)
        requests.append({"prompt": text, "multi_modal_data": {"image": images}})

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    outputs = llm.generate(requests, sampling_params=sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def infer_vllm(processor, llm, model_name: str, image_paths: List[str],
               prompt: str, max_new_tokens: int) -> str:
    """Run inference via vLLM for any model."""
    from vllm import SamplingParams

    cfg = MODEL_REGISTRY[model_name]
    max_img = cfg.get("max_images", 6)

    # Limit images
    if len(image_paths) > max_img:
        image_paths = image_paths[-max_img:]

    images = [open_image(p) for p in image_paths]

    text = build_chat_prompt(processor, model_name, images, prompt)

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    mm_data = {"image": images}

    outputs = llm.generate(
        [{"prompt": text, "multi_modal_data": mm_data}],
        sampling_params=sampling_params,
    )
    return outputs[0].outputs[0].text.strip()

# ---------------------------------------------------------------------------
# HF fallback (llama32)
# ---------------------------------------------------------------------------
def load_hf(model_name: str, model_path: str):
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return processor, model


def infer_hf(processor, model, model_name: str, image_paths: List[str],
             prompt: str, max_new_tokens: int) -> str:
    cfg = MODEL_REGISTRY[model_name]
    max_img = cfg.get("max_images", 1)
    if len(image_paths) > max_img:
        image_paths = image_paths[-max_img:]

    images = [open_image(p) for p in image_paths]
    image_content = [{"type": "image"} for _ in images]
    messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# lmdeploy fallback (internvl35)
# ---------------------------------------------------------------------------
def load_lmdeploy(model_name: str, model_path: str):
    from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
    cfg = MODEL_REGISTRY[model_name]
    pipe = pipeline(
        model_path,
        backend_config=TurbomindEngineConfig(session_len=16384, tp=1, device="cuda"),
        chat_template_config=ChatTemplateConfig(model_name=cfg.get("chat_template", "internvl2_5")),
    )
    return None, pipe


def infer_lmdeploy(processor, pipe, model_name: str, image_paths: List[str],
                   prompt: str, max_new_tokens: int) -> str:
    from lmdeploy.vl import load_image as lmd_load
    cfg = MODEL_REGISTRY[model_name]
    max_img = cfg.get("max_images", 6)
    if len(image_paths) > max_img:
        image_paths = image_paths[-max_img:]

    if len(image_paths) > 1:
        images = [lmd_load(p) for p in image_paths]
        prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(len(images)))
        response = pipe((prefix + prompt, images))
    else:
        response = pipe((prompt, lmd_load(image_paths[0])))
    return response.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    cfg = MODEL_REGISTRY[args.model]
    model_path = args.model_path or cfg["path"]
    engine = cfg.get("engine", "vllm")

    print(f"Loading {args.model} [{model_path}] via {engine} ...")
    if engine == "vllm":
        processor, model = load_model(args.model, model_path)
        infer_fn = infer_vllm
    elif engine == "hf":
        processor, model = load_hf(args.model, model_path)
        infer_fn = infer_hf
    elif engine == "lmdeploy":
        processor, model = load_lmdeploy(args.model, model_path)
        infer_fn = infer_lmdeploy
    else:
        raise ValueError(f"Unknown engine: {engine}")

    print(f"Loading benchmark from {args.benchmark} ...")
    data: List[Dict] = load_json(args.benchmark)

    if args.country:
        data = [d for d in data if d["country"] in args.country]
    if args.category:
        data = [d for d in data if d["question_category"] in args.category]

    # Resume — composite key (id, image_path tuple, question) to handle duplicate ids
    def _key(r):
        img = r.get("image_path") or []
        return (r.get("id"), tuple(img) if isinstance(img, list) else (img,), r.get("question", ""))

    done_keys: set = set()
    results: List[Dict] = []
    if os.path.exists(args.output) and not args.overwrite:
        results = load_json(args.output)
        # Drop any prior ERROR entries so resume retries them
        pre = len(results)
        results = [r for r in results if not str(r.get("pred","")).startswith("ERROR")]
        if len(results) != pre:
            print(f"Dropped {pre-len(results)} prior ERROR entries for retry.")
        done_keys = {_key(r) for r in results}
        print(f"Resuming — {len(done_keys)} items already done.")

    n_tokens_direct = args.max_new_tokens
    n_tokens_long = args.max_new_tokens_long
    n_tokens = n_tokens_direct if args.setting == "direct" else n_tokens_long

    # Filter out already-done items
    pending = [item for item in data if _key(item) not in done_keys]
    print(f"Pending items: {len(pending)} (done: {len(done_keys)})")

    if engine == "vllm":
        # Batched inference — vLLM continuous batching for high GPU util
        BATCH = args.batch_size
        for i in tqdm(range(0, len(pending), BATCH), desc=f"{args.model}/{args.setting}"):
            chunk = pending[i:i + BATCH]
            batch_prompts = [build_prompt(item, args.setting) for item in chunk]
            batch_imgs: List[List[str]] = []
            for item in chunk:
                img_paths = item["image_path"]
                if args.max_images and len(img_paths) > args.max_images:
                    img_paths = img_paths[-args.max_images:]
                batch_imgs.append(img_paths)

            try:
                preds = infer_vllm_batch(processor, model, args.model,
                                         batch_imgs, batch_prompts, n_tokens)
            except Exception as e:
                # Fall back to per-item so one bad sample doesn't kill the whole chunk
                preds = []
                for imgs, prm in zip(batch_imgs, batch_prompts):
                    try:
                        preds.append(infer_vllm(processor, model, args.model, imgs, prm, n_tokens))
                    except Exception as ee:
                        preds.append(f"ERROR: {ee}")

            for item, pred in zip(chunk, preds):
                results.append({
                    "id":                item["id"],
                    "country":           item["country"],
                    "question_category": item["question_category"],
                    "question_type":     item.get("question_type", "multiple_choice"),
                    "image_path":        item["image_path"],
                    "question":          item["question"],
                    "options":           item["options"],
                    "gt":                item["answer"],
                    "pred":              pred,
                    "rule_reference":    item.get("rule_reference", []),
                    "setting":           args.setting,
                })
            save_json(results, args.output)
    else:
        for item in tqdm(pending, desc=f"{args.model}/{args.setting}"):
            prompt = build_prompt(item, args.setting)
            img_paths = item["image_path"]
            if args.max_images and len(img_paths) > args.max_images:
                img_paths = img_paths[-args.max_images:]
            try:
                pred = infer_fn(processor, model, args.model, img_paths, prompt, n_tokens)
            except Exception as e:
                pred = f"ERROR: {e}"
            results.append({
                "id":                item["id"],
                "country":           item["country"],
                "question_category": item["question_category"],
                "question_type":     item.get("question_type", "multiple_choice"),
                "image_path":        item["image_path"],
                "question":          item["question"],
                "options":           item["options"],
                "gt":                item["answer"],
                "pred":              pred,
                "rule_reference":    item.get("rule_reference", []),
                "setting":           args.setting,
            })
            if len(results) % 50 == 0:
                save_json(results, args.output)

    save_json(results, args.output)
    print(f"Saved {len(results)} results → {args.output}")


def main():
    parser = argparse.ArgumentParser(description="CulturalDrive VLM inference (vLLM)")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--setting", required=True,
                        choices=["direct", "reasoning", "rule_given"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--benchmark", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "culturebenchmark_eval.json")))
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens_long", type=int, default=512)
    parser.add_argument("--country", nargs="+", default=None)
    parser.add_argument("--category", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="vLLM batch size — larger = higher GPU util, needs more VRAM")

    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "results",
            f"{args.model}_{args.setting}_results.json"
        )
    run(args)

if __name__ == "__main__":
    main()
