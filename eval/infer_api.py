"""Unified API-based inference for CultureBenchmark.

Supports two remote backends with identical I/O / resume semantics:
    --backend openai   → OpenAI chat.completions (GPT-5, GPT-5.4 …)
    --backend bedrock  → AWS Bedrock converse   (Qwen3-VL-235B …)

Both backends share the same concurrent dispatcher, composite-key resume,
ERROR retry, and output schema as infer_vllm.py.

Examples
--------
# GPT-5.4, all 3 settings
export OPENAI_API_KEY='sk-...'
python infer_api.py --backend openai  --model gpt-5.4 --setting direct

# Qwen3-VL-235B (Bedrock) — pass creds inline, never `export`
AWS_BEARER_TOKEN_BEDROCK='ABSK...' AWS_REGION=us-east-2 \
    python infer_api.py --backend bedrock --model qwen.qwen3-vl-235b-a22b \
                        --setting direct --workers 8
"""

import argparse
import base64
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _utils import (
    composite_key, load_json, save_json, load_with_resume,
)
from infer_vllm import build_prompt  # reuse prompt builders

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BENCHMARK = str(_REPO_ROOT / 'culturebenchmark_eval.json')

# ============================================================ OpenAI =====
_openai_client = None

def _openai_key() -> str:
    k = os.environ.get('OPENAI_API_KEY')
    if k:
        return k.strip()
    raise RuntimeError('OPENAI_API_KEY is not set (export it before running)')

def _openai_client_get():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=_openai_key())
    return _openai_client

def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or 'image/jpeg'
    b64 = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    return f'data:{mime};base64,{b64}'

def openai_call(model: str, prompt: str, image_paths: List[str],
                max_tokens: int, reasoning_effort: str = 'none',
                max_retries: int = 4, timeout: int = 120) -> str:
    content = [{'type': 'image_url', 'image_url': {'url': _data_uri(p)}}
               for p in image_paths]
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    kwargs = dict(model=model, messages=messages, timeout=timeout)

    backoff = 2.0
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            cli = _openai_client_get()
            try:
                resp = cli.chat.completions.create(
                    max_completion_tokens=max_tokens,
                    reasoning_effort=reasoning_effort, **kwargs)
            except TypeError:
                try:
                    resp = cli.chat.completions.create(
                        max_completion_tokens=max_tokens, **kwargs)
                except TypeError:
                    resp = cli.chat.completions.create(
                        max_tokens=max_tokens, **kwargs)
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if 'invalid' in msg and ('model' in msg or 'parameter' in msg):
                break
            time.sleep(backoff); backoff = min(backoff * 2, 30.0)
    return f'ERROR: {last_err}'

# =========================================================== Bedrock =====
_bedrock_local = threading.local()

def _bedrock_client(region: str):
    cli = getattr(_bedrock_local, 'client', None)
    if cli is None:
        import boto3
        cli = boto3.client('bedrock-runtime', region_name=region)
        _bedrock_local.client = cli
    return cli

def _img_format(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    if mt:
        sub = mt.split('/')[-1].lower()
        if sub in {'jpeg', 'jpg'}: return 'jpeg'
        if sub in {'png', 'gif', 'webp'}: return sub
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return {'jpg': 'jpeg'}.get(ext, ext or 'jpeg')

def bedrock_call(model: str, region: str, prompt: str, image_paths: List[str],
                 max_tokens: int, max_retries: int = 4) -> str:
    content = []
    for p in image_paths:
        if not os.path.isfile(p):
            return f'ERROR: image not found: {p}'
        content.append({'image': {'format': _img_format(p),
                                  'source': {'bytes': open(p, 'rb').read()}}})
    content.append({'text': prompt})

    backoff = 2.0
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            cli = _bedrock_client(region)
            resp = cli.converse(
                modelId=model,
                messages=[{'role': 'user', 'content': content}],
                inferenceConfig={'maxTokens': max_tokens, 'temperature': 0.0})
            blocks = resp['output']['message']['content']
            return ''.join(b.get('text', '') for b in blocks).strip()
        except Exception as e:
            last_err = e
            if 'validation' in str(e).lower() and 'image' in str(e).lower():
                break
            time.sleep(backoff); backoff = min(backoff * 2, 30.0)
    return f'ERROR: {last_err}'

# ============================================================ shared =====
def _limit_imgs(paths: List[str], mx: int) -> List[str]:
    return paths[-mx:] if mx and len(paths) > mx else paths

def _call_dispatch(args, prompt: str, image_paths: List[str],
                   max_tokens: int) -> str:
    imgs = _limit_imgs(image_paths, args.max_images)
    if args.backend == 'openai':
        return openai_call(args.model, prompt, imgs, max_tokens,
                           reasoning_effort=args.reasoning_effort)
    if args.backend == 'bedrock':
        return bedrock_call(args.model, args.region, prompt, imgs, max_tokens)
    raise ValueError(f'unknown backend: {args.backend}')

def run(args) -> None:
    print(f'[{args.backend}] model={args.model}  setting={args.setting}  '
          f'workers={args.workers}')
    data: List[Dict] = load_json(args.benchmark)
    if args.country:
        data = [d for d in data if d['country'] in args.country]
    if args.category:
        data = [d for d in data if d['question_category'] in args.category]

    results, done_keys = load_with_resume(args.output, args.overwrite)
    pending = [it for it in data if composite_key(it) not in done_keys]
    print(f'Pending items: {len(pending)} (total: {len(data)})')
    if not pending:
        save_json(results, args.output); return

    n_tokens = (args.max_new_tokens if args.setting == 'direct'
                else args.max_new_tokens_long)

    work = [(it, build_prompt(it, args.setting),
             _limit_imgs(it['image_path'], args.max_images))
            for it in pending]

    save_every = max(args.save_every, args.workers)
    pbar = tqdm(total=len(work), desc=f'{args.backend}/{args.setting}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_call_dispatch, args, prm, imgs, n_tokens): it
            for (it, prm, imgs) in work
        }
        since_save = 0
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                pred = fut.result()
            except Exception as e:
                pred = f'ERROR: {e}'
            results.append({
                'id':                it['id'],
                'country':           it['country'],
                'question_category': it['question_category'],
                'question_type':     it.get('question_type', 'multiple_choice'),
                'image_path':        it['image_path'],
                'question':          it['question'],
                'options':           it['options'],
                'gt':                it['answer'],
                'pred':              pred,
                'rule_reference':    it.get('rule_reference', []),
                'setting':           args.setting,
            })
            since_save += 1
            pbar.update(1)
            if since_save >= save_every:
                save_json(results, args.output); since_save = 0
    pbar.close()
    save_json(results, args.output)
    print(f'Saved {len(results)} results → {args.output}')


def main():
    p = argparse.ArgumentParser(description='CultureBenchmark API inference')
    p.add_argument('--backend', required=True, choices=['openai', 'bedrock'])
    p.add_argument('--model', required=True,
                   help='openai: e.g. gpt-5, gpt-5.4, gpt-5-mini   '
                        'bedrock: e.g. qwen.qwen3-vl-235b-a22b')
    p.add_argument('--setting', required=True,
                   choices=['direct', 'reasoning', 'rule_given'])
    p.add_argument('--benchmark', default=_DEFAULT_BENCHMARK)
    p.add_argument('--output', default=None,
                   help='results/<model>_<setting>_results.json by default')
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--max_new_tokens_long', type=int, default=1024)
    p.add_argument('--country', nargs='+', default=None)
    p.add_argument('--category', nargs='+', default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--max_images', type=int, default=3)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save_every', type=int, default=20)
    # backend-specific
    p.add_argument('--reasoning_effort', default='none',
                   choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                   help='openai only. gpt-5→minimal, gpt-5.4→none recommended.')
    p.add_argument('--region', default=os.environ.get('AWS_REGION', 'us-east-2'),
                   help='bedrock only.')
    args = p.parse_args()

    if args.output is None:
        tag = args.model.replace('/', '_').replace('.', '.')
        # map long bedrock id to short filename tag, keep openai id as-is
        if args.backend == 'bedrock' and 'qwen3-vl-235b' in args.model:
            tag = 'qwen3vl235b'
        out_dir = os.path.join(os.path.dirname(__file__), 'results')
        args.output = os.path.join(out_dir, f'{tag}_{args.setting}_results.json')
    run(args)


if __name__ == '__main__':
    main()
