"""Multi-backend inference for CultureBenchmark local VLMs.

Three backends share one loop / one registry:
    vllm      — vLLM continuous batching   (llava, qwen25vl, qwen3vl, internvl3, gemma3)
    hf        — HuggingFace transformers   (llama32)
    lmdeploy  — lmdeploy TurboMind         (internvl35)

Example
    python infer_vllm.py --model qwen3vl --setting direct
    python infer_vllm.py --model internvl35 --setting rule_given
"""

import argparse, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from _utils import composite_key, load_json, save_json, load_with_resume

# Handbooks (from repo-root `traffic_handbook/`) drive the rule_given prompt.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_handbook import (
    china_traffic_handbook, india_traffic_handbook, japan_traffic_handbook,
    singapore_traffic_handbook, uk_traffic_handbook, us_traffic_handbook,
)
HANDBOOKS = {
    "cn": china_traffic_handbook, "ind": india_traffic_handbook,
    "jp": japan_traffic_handbook, "sg": singapore_traffic_handbook,
    "uk": uk_traffic_handbook,    "us": us_traffic_handbook,
}

MODEL_REGISTRY: Dict[str, Dict] = {
    "llava":      {"path": "llava-hf/llava-v1.6-mistral-7b-hf",       "engine": "vllm",     "max_images": 1},
    "qwen25vl":   {"path": "Qwen/Qwen2.5-VL-7B-Instruct",             "engine": "vllm",     "max_images": 6},
    "qwen3vl":    {"path": "Qwen/Qwen3-VL-8B-Instruct",               "engine": "vllm",     "max_images": 6, "enable_thinking": False},
    "internvl3":  {"path": "OpenGVLab/InternVL3-8B",                  "engine": "vllm",     "max_images": 6},
    "gemma3":     {"path": "google/gemma-3-12b-it",                   "engine": "vllm",     "max_images": 6},
    "llama32":    {"path": "meta-llama/Llama-3.2-11B-Vision-Instruct","engine": "hf",       "max_images": 1},
    "internvl35": {"path": "OpenGVLab/InternVL3_5-8B",                "engine": "lmdeploy", "max_images": 6, "chat_template": "internvl2_5"},
}

# --------------------------------------------------------------- prompts ----
def extract_rules(handbook: str, refs: List[str]) -> str:
    out = []
    for ref in refs or []:
        m = re.search(rf"({re.escape(ref)}\s.+?)(?=\nS\d+\s|\Z)", handbook, re.DOTALL)
        if m:
            out.append(m.group(1).strip()); continue
        for line in handbook.splitlines():
            if line.startswith(ref + " ") or line.startswith(ref + ":"):
                out.append(line.strip()); break
    return "\n\n".join(out)

_PROMPTS = {
    "direct": (
        "Answer the question using only the image. Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output only the answer letter (A/B/C/D).\n"
    ),
    "reasoning": (
        "Step 1 — Geographic inference: Based on road infrastructure, traffic signs, "
        "lane markings, and environmental cues in the image, identify the most likely "
        "country or region where this scene was captured.\n"
        "Step 2 — Cultural traffic rule: State the relevant traffic rule(s) for that "
        "country that apply to this question.\n"
        "Step 3 — Visual reasoning: Describe what you observe in the image that is "
        "relevant to answering the question.\n"
        "Step 4 — Answer: Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "rule_given": (
        "{rule_block}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
}

def build_prompt(item: Dict, setting: str) -> str:
    q, options = item["question"], item["options"]
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    if setting == "rule_given":
        rule = extract_rules(HANDBOOKS.get(item["country"], ""), item.get("rule_reference", []))
        rule_block = f"Relevant traffic rule(s):\n{rule}\n\n" if rule else ""
        return _PROMPTS[setting].format(q=q, opts=opts, rule_block=rule_block)
    return _PROMPTS[setting].format(q=q, opts=opts)

# ---------------------------------------------------------------- image ----
def _img(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

# ---------------------------------------------------------------- vLLM  ----
def _load_vllm(cfg: Dict):
    from vllm import LLM
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg["path"], trust_remote_code=True)
    llm = LLM(
        model=cfg["path"], dtype="bfloat16", max_model_len=16384,
        limit_mm_per_prompt={"image": cfg["max_images"]},
        trust_remote_code=True, gpu_memory_utilization=0.90,
        max_num_seqs=64, disable_mm_preprocessor_cache=True,
        mm_processor_cache_gb=0, enforce_eager=True,
    )
    return proc, llm

def _call_vllm(proc, llm, cfg, batch: List[Tuple[List[str], str]], max_tokens: int):
    from vllm import SamplingParams
    reqs = []
    for imgs, prompt in batch:
        images = [_img(p) for p in imgs]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        kwargs = {"enable_thinking": False} if cfg.get("enable_thinking") is False else {}
        text = proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, **kwargs,
        )
        reqs.append({"prompt": text, "multi_modal_data": {"image": images}})
    out = llm.generate(reqs, sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
    return [o.outputs[0].text.strip() for o in out]

# ------------------------------------------------------------- HuggingFace -
def _load_hf(cfg: Dict):
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(cfg["path"])
    mdl = MllamaForConditionalGeneration.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    return proc, mdl

def _call_hf(proc, mdl, cfg, batch, max_tokens):
    preds = []
    for imgs, prompt in batch:
        images = [_img(p) for p in imgs]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        ins = proc(text=text, images=images, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**ins, max_new_tokens=max_tokens, do_sample=False)
        preds.append(proc.decode(gen[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return preds

# ----------------------------------------------------------------- lmdeploy
def _load_lmdeploy(cfg: Dict):
    from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
    pipe = pipeline(
        cfg["path"],
        backend_config=TurbomindEngineConfig(session_len=16384, tp=1, device="cuda"),
        chat_template_config=ChatTemplateConfig(
            model_name=cfg.get("chat_template", "internvl2_5")),
    )
    return None, pipe

def _call_lmdeploy(_proc, pipe, cfg, batch, max_tokens):
    from lmdeploy.vl import load_image as lmd_load
    preds = []
    for imgs, prompt in batch:
        if len(imgs) > 1:
            images = [lmd_load(p) for p in imgs]
            pfx = "".join(f"Frame{i+1}: <image>\n" for i in range(len(images)))
            resp = pipe((pfx + prompt, images))
        else:
            resp = pipe((prompt, lmd_load(imgs[0])))
        preds.append(resp.text)
    return preds

ENGINES = {
    "vllm":     (_load_vllm,     _call_vllm),
    "hf":       (_load_hf,       _call_hf),
    "lmdeploy": (_load_lmdeploy, _call_lmdeploy),
}

# ---------------------------------------------------------------- runner ---
def run(args):
    cfg = dict(MODEL_REGISTRY[args.model])
    if args.model_path:
        cfg["path"] = args.model_path
    print(f"Loading {args.model} [{cfg['path']}] via {cfg['engine']} ...")
    load, call = ENGINES[cfg["engine"]]
    proc, mdl = load(cfg)

    data = load_json(args.benchmark)
    if args.country:  data = [d for d in data if d["country"] in args.country]
    if args.category: data = [d for d in data if d["question_category"] in args.category]

    results, done_keys = load_with_resume(args.output, args.overwrite)
    pending = [it for it in data if composite_key(it) not in done_keys]
    print(f"Pending: {len(pending)} / {len(data)}")
    if not pending:
        save_json(results, args.output); return

    n_tokens = args.max_new_tokens if args.setting == "direct" else args.max_new_tokens_long
    max_img = cfg["max_images"]
    batched = cfg["engine"] == "vllm"
    BATCH = args.batch_size if batched else 1

    def _pack(items):
        return [((it["image_path"][-max_img:] if len(it["image_path"]) > max_img
                  else it["image_path"]), build_prompt(it, args.setting))
                for it in items]

    def _save_row(it, pred):
        results.append({
            "id": it["id"], "country": it["country"],
            "question_category": it["question_category"],
            "question_type": it.get("question_type", "multiple_choice"),
            "image_path": it["image_path"], "question": it["question"],
            "options": it["options"], "gt": it["answer"], "pred": pred,
            "rule_reference": it.get("rule_reference", []),
            "setting": args.setting,
        })

    save_every = max(50, BATCH)  # rows between disk syncs
    since_save = 0
    for i in tqdm(range(0, len(pending), BATCH), desc=f"{args.model}/{args.setting}"):
        chunk = pending[i:i + BATCH]
        reqs = _pack(chunk)
        try:
            preds = call(proc, mdl, cfg, reqs, n_tokens)
        except Exception:
            # Engine crashed: retry per-item so one bad sample doesn't kill the chunk.
            preds = []
            for r in reqs:
                try:
                    preds.extend(call(proc, mdl, cfg, [r], n_tokens))
                except Exception as ee:
                    preds.append(f"ERROR: {ee}")
        for it, pred in zip(chunk, preds):
            _save_row(it, pred)
        since_save += len(chunk)
        if since_save >= save_every:
            save_json(results, args.output); since_save = 0

    save_json(results, args.output)
    print(f"Saved {len(results)} results → {args.output}")


def main():
    ROOT = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="CultureBenchmark VLM inference")
    p.add_argument("--model",    required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--setting",  required=True, choices=["direct", "reasoning", "rule_given"])
    p.add_argument("--model_path", default=None)
    p.add_argument("--benchmark",  default=str(ROOT / "culturebenchmark_eval.json"))
    p.add_argument("--output",     default=None)
    p.add_argument("--max_new_tokens",      type=int, default=32)
    p.add_argument("--max_new_tokens_long", type=int, default=512)
    p.add_argument("--batch_size",          type=int, default=64)
    p.add_argument("--country",  nargs="+", default=None)
    p.add_argument("--category", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.output is None:
        a.output = str(Path(__file__).resolve().parent / "results" /
                       f"{a.model}_{a.setting}_results.json")
    run(a)


if __name__ == "__main__":
    main()
