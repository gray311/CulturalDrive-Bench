"""Shared utilities across inference and analysis scripts."""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

# --- benchmark constants ---------------------------------------------------
COUNTRIES = ['cn', 'us', 'uk', 'jp', 'sg', 'ind']
COUNTRY_LABELS = {'cn': 'CN', 'us': 'US', 'uk': 'UK',
                  'jp': 'JP', 'sg': 'SG', 'ind': 'IND'}
CATEGORIES = ['perception', 'prediction', 'planning', 'region']
CATEGORY_LABELS = {'perception': 'Perc.', 'prediction': 'Pred.',
                   'planning': 'Plan.',  'region': 'Reg.'}
SETTINGS = ['direct', 'reasoning', 'rule_given']
SETTING_LABELS = {'direct': 'Direct', 'reasoning': 'Reasoning',
                  'rule_given': 'Rule-Given'}

# Country palette used by all plots (user-specified).
# Index order must match COUNTRIES above.
PALETTE = {
    'cn':  '#E8706F',  # salmon red
    'us':  '#F0D258',  # mustard yellow
    'uk':  '#678CB5',  # steel blue
    'jp':  '#F0973B',  # orange
    'sg':  '#87C1BD',  # teal
    'ind': '#6DB066',  # soft green
}

# --- JSON helpers ----------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- composite key for de-dup / resume ------------------------------------
def composite_key(r: Dict) -> Tuple:
    """Key that distinguishes counterfactual variants: (id, image_path, question)."""
    img = r.get('image_path') or []
    return (r.get('id'),
            tuple(img) if isinstance(img, list) else (img,),
            r.get('question', ''))

# --- answer extraction from model output ----------------------------------
def extract_ans(pred: str):
    """Pull a single letter (A-D) from a model output — robust to verbose CoT."""
    s = (pred or '').strip()
    if not s:
        return None
    m = re.search(r'(?:final\s*answer|answer)\s*[:：]\s*([A-D])', s, re.I)
    if m:
        return m.group(1).upper()
    tail = s[-60:]
    ms = re.findall(r'\b([A-D])\b', tail)
    if ms:
        return ms[-1]
    ms = re.findall(r'\b([A-D])\b', s)
    return ms[-1] if ms else None

# --- resume loader (shared across inference scripts) ----------------------
def load_with_resume(output_path: str, overwrite: bool = False) -> Tuple[List[Dict], set]:
    """Load existing result JSON, drop ERROR entries so they retry, return
    (results, done_keys)."""
    if not os.path.exists(output_path) or overwrite:
        return [], set()
    results = load_json(output_path)
    pre = len(results)
    results = [r for r in results if not str(r.get('pred', '')).startswith('ERROR')]
    dropped = pre - len(results)
    if dropped:
        print(f'Dropped {dropped} prior ERROR entries for retry.')
    done_keys = {composite_key(r) for r in results}
    print(f'Resuming — {len(done_keys)} items already done.')
    return results, done_keys
