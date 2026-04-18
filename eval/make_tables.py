"""Generate the main LaTeX results table and a long-format CSV.

Outputs
    tables/results_full_all.tex   — the paper table (model × setting,
                                     columns = 4 categories × 6 countries + Overall)
    tables/results_full.csv       — long-format: setting,model,country,category,accuracy
"""

import json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (COUNTRIES, COUNTRY_LABELS, CATEGORIES,
                    SETTINGS, SETTING_LABELS, composite_key, extract_ans)

_EVAL_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
RESULT_DIR = _EVAL_DIR / "results"
BENCHMARK  = _REPO_ROOT / "culturebenchmark_eval.json"
OUT_DIR    = _EVAL_DIR / "tables"
OUT_DIR.mkdir(exist_ok=True)

MODELS = ['llava', 'gemma3', 'internvl3', 'internvl35',
          'qwen25vl', 'qwen3vl', 'llama32', 'gpt-5.4', 'qwen3vl235b']
MODEL_LABELS = {
    'llava':       'LLaVA-1.6-7B',
    'gemma3':      'Gemma3-12B',
    'internvl3':   'InternVL3-8B',
    'internvl35':  'InternVL3.5-8B',
    'qwen25vl':    'Qwen2.5-VL-7B',
    'qwen3vl':     'Qwen3-VL-8B',
    'llama32':     'Llama-3.2-11B-V',
    'gpt-5.4':     'GPT-5.4',
    'qwen3vl235b': 'Qwen3-VL-235B',
}

# ---- gather accuracy[setting][model][country][category] = (correct, total)
bench_info = {composite_key(x): (x['country'], x['question_category'])
              for x in json.load(open(BENCHMARK))}
acc = {s: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
       for s in SETTINGS}

for s in SETTINGS:
    for m in MODELS:
        path = RESULT_DIR / f'{m}_{s}_results.json'
        if not path.exists():
            continue
        uniq = {composite_key(r): r for r in json.load(open(path))}
        for k, r in uniq.items():
            info = bench_info.get(k)
            if info is None:
                continue
            country, cat = info
            acc[s][m][country][cat][1] += 1
            if extract_ans(r.get('pred', '')) == r.get('gt'):
                acc[s][m][country][cat][0] += 1

def _acc(s, m, country, cat):
    a, n = acc[s][m][country][cat]
    return (a / n) if n else 0.0

def _overall(s, m):
    a, n = 0, 0
    for c in COUNTRIES:
        for k in CATEGORIES:
            ai, ni = acc[s][m][c][k]; a += ai; n += ni
    return (a / n) if n else 0.0

# ---- long-format CSV
with open(OUT_DIR / 'results_full.csv', 'w') as f:
    f.write('setting,model,country,category,correct,total,accuracy\n')
    for s in SETTINGS:
        for m in MODELS:
            for c in COUNTRIES:
                for k in CATEGORIES:
                    a, n = acc[s][m][c][k]
                    f.write(f'{s},{m},{c},{k},{a},{n},{(a/n if n else 0):.4f}\n')

# ---- main LaTeX table: rows grouped by model (3 settings each),
#      columns = 4 categories × 6 countries + Overall; bold best per column per setting.
def _best_row_per_col(setting: str) -> Dict[int, int]:
    cols: Dict[int, List[float]] = {}
    for mi, m in enumerate(MODELS):
        j = 1
        for k in CATEGORIES:
            for c in COUNTRIES:
                cols.setdefault(j, []).append(_acc(setting, m, c, k)); j += 1
        cols.setdefault(j, []).append(_overall(setting, m))
    return {j: max(range(len(MODELS)), key=lambda i: v[i]) for j, v in cols.items()}

best = {s: _best_row_per_col(s) for s in SETTINGS}

col_spec = 'll' + ('cccccc|') * len(CATEGORIES) + 'c'
header1 = ([r'\multirow{2}{*}{Model}', r'\multirow{2}{*}{Setting}']
           + [r'\multicolumn{6}{c|}{' + k.capitalize() + '}' for k in CATEGORIES]
           + [r'\multirow{2}{*}{Overall}'])
cmid = ' '.join(r'\cmidrule(lr){' + f'{3+6*i}-{8+6*i}' + '}' for i in range(len(CATEGORIES)))
header2 = ['', ''] + [COUNTRY_LABELS[c] for _ in CATEGORIES for c in COUNTRIES] + ['']

def _fmt(val: float, is_best: bool) -> str:
    s = f'{val*100:.1f}'
    return r'\textbf{' + s + '}' if is_best else s

rows = []
for mi, m in enumerate(MODELS):
    for si, s in enumerate(SETTINGS):
        row = [r'\multirow{3}{*}{' + MODEL_LABELS[m] + '}' if si == 0 else '',
               SETTING_LABELS[s]]
        j = 1
        for k in CATEGORIES:
            for c in COUNTRIES:
                row.append(_fmt(_acc(s, m, c, k), mi == best[s][j])); j += 1
        row.append(_fmt(_overall(s, m), mi == best[s][j]))
        rows.append(' & '.join(row) + r' \\')
    if mi < len(MODELS) - 1:
        rows.append(r'\midrule')

tex = '\n'.join([
    r'% Combined full accuracy grid (all settings) — grouped by model',
    r'\begin{table*}[t]', r'\centering', r'\scriptsize',
    r'\setlength{\tabcolsep}{3pt}',
    r'\caption{CultureBenchmark accuracy (\%) across three settings per model. '
    r'Each task category is split by 6 countries; Overall is the mean across all '
    r'cells. For each column, the best score across models \emph{within the same '
    r'setting} is in \textbf{bold}.}',
    r'\label{tab:results_full_all}',
    r'\begin{tabular}{' + col_spec + '}',
    r'\toprule',
    ' & '.join(header1) + r' \\',
    cmid,
    ' & '.join(header2) + r' \\',
    r'\midrule',
    *rows,
    r'\bottomrule', r'\end{tabular}', r'\end{table*}',
])
(OUT_DIR / 'results_full_all.tex').write_text(tex)

print('saved:')
print(' ', OUT_DIR / 'results_full_all.tex')
print(' ', OUT_DIR / 'results_full.csv')
