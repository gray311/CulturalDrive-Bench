"""Generate LaTeX tables of accuracy per (model, country, category, setting)
from the 21 result JSONs. Output:
  tables/results_by_category.tex  — model rows, 4 categories + Avg columns
  tables/results_by_country.tex   — model rows, 6 country + Avg columns
  tables/results_full.csv         — long-format CSV
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (COUNTRIES, COUNTRY_LABELS, CATEGORIES, CATEGORY_LABELS,
                    SETTINGS, SETTING_LABELS, composite_key, extract_ans)

_EVAL_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
RESULT_DIR = str(_EVAL_DIR / 'results')
BENCHMARK  = str(_REPO_ROOT / 'culturebenchmark_eval.json')
OUT_DIR    = str(_EVAL_DIR / 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['llava', 'gemma3', 'internvl3', 'internvl35', 'qwen25vl', 'qwen3vl', 'llama32',
          'gpt-5.4', 'qwen3vl235b']
MODEL_LABELS = {
    'llava':       'LLaVA-1.6-7B',
    'gemma3':      'Gemma3-12B',
    'internvl3':   'InternVL3-8B',
    'internvl35':  'InternVL3.5-8B',
    'qwen25vl':    'Qwen2.5-VL-7B',
    'qwen3vl':     'Qwen3-VL-8B',
    'llama32':     'Llama-3.2-11B-V',
    'gpt-5.4':    'GPT-5.4',
    'qwen3vl235b': 'Qwen3-VL-235B',
}
bench = json.load(open(BENCHMARK))
bench_info = {composite_key(x): (x['country'], x['question_category']) for x in bench}

# accuracy[setting][model][country][category] = (correct, total)
accuracy = {s: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0,0]))) for s in SETTINGS}

for s in SETTINGS:
    for m in MODELS:
        p = os.path.join(RESULT_DIR, f'{m}_{s}_results.json')
        if not os.path.exists(p): continue
        data = json.load(open(p))
        uniq = {composite_key(r): r for r in data}
        for k, r in uniq.items():
            info = bench_info.get(k)
            if info is None: continue
            country, cat = info
            pred_ans = extract_ans(r.get('pred', ''))
            gt = r.get('gt')
            accuracy[s][m][country][cat][1] += 1
            if pred_ans == gt:
                accuracy[s][m][country][cat][0] += 1

def mean_by_cat(s, m, cat):
    c, n = 0, 0
    for country in COUNTRIES:
        a, b = accuracy[s][m][country][cat]
        c += a; n += b
    return (c / n) if n else 0.0

def mean_by_country(s, m, country):
    c, n = 0, 0
    for cat in CATEGORIES:
        a, b = accuracy[s][m][country][cat]
        c += a; n += b
    return (c / n) if n else 0.0

def overall(s, m):
    c, n = 0, 0
    for country in COUNTRIES:
        for cat in CATEGORIES:
            a, b = accuracy[s][m][country][cat]
            c += a; n += b
    return (c / n) if n else 0.0

# ---------- Table 1: model × category, one block per setting ----------
def tex_escape(s):
    return s.replace('%', r'\%').replace('_', r'\_')

lines = []
lines.append(r'% Accuracy by task category (mean across 6 countries)')
lines.append(r'\begin{table*}[t]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\caption{CultureBenchmark accuracy by task category. Mean across 6 countries. Best per column in each setting in \textbf{bold}.}')
lines.append(r'\label{tab:results_by_category}')
lines.append(r'\begin{tabular}{l' + 'c' * (len(CATEGORIES) + 1) + '}')
lines.append(r'\toprule')
header = ['Model'] + [CATEGORY_LABELS[c] for c in CATEGORIES] + ['Avg']
lines.append(' & '.join(header) + r' \\')
lines.append(r'\midrule')

for s in SETTINGS:
    lines.append(r'\multicolumn{' + str(len(CATEGORIES)+2) + r'}{l}{\textit{' + SETTING_LABELS[s] + r'}} \\')
    # find best per column
    col_vals = {cat: [mean_by_cat(s, m, cat) for m in MODELS] for cat in CATEGORIES}
    col_vals['__avg__'] = [overall(s, m) for m in MODELS]
    best = {k: max(range(len(MODELS)), key=lambda i: v[i]) for k, v in col_vals.items()}
    for mi, m in enumerate(MODELS):
        row = [MODEL_LABELS[m]]
        for cat in CATEGORIES:
            v = col_vals[cat][mi]
            fmt = f'{v*100:.1f}'
            if mi == best[cat]: fmt = r'\textbf{' + fmt + '}'
            row.append(fmt)
        v = col_vals['__avg__'][mi]
        fmt = f'{v*100:.1f}'
        if mi == best['__avg__']: fmt = r'\textbf{' + fmt + '}'
        row.append(fmt)
        lines.append(' & '.join(row) + r' \\')
    lines.append(r'\midrule' if s != SETTINGS[-1] else r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')

open(os.path.join(OUT_DIR, 'results_by_category.tex'), 'w').write('\n'.join(lines))

# ---------- Table 2: model × country, one block per setting ----------
lines = []
lines.append(r'% Accuracy by country (mean across 4 task categories)')
lines.append(r'\begin{table*}[t]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\caption{CultureBenchmark accuracy by country. Mean across 4 task categories.}')
lines.append(r'\label{tab:results_by_country}')
lines.append(r'\begin{tabular}{l' + 'c' * (len(COUNTRIES) + 1) + '}')
lines.append(r'\toprule')
header = ['Model'] + [COUNTRY_LABELS[c] for c in COUNTRIES] + ['Avg']
lines.append(' & '.join(header) + r' \\')
lines.append(r'\midrule')

for s in SETTINGS:
    lines.append(r'\multicolumn{' + str(len(COUNTRIES)+2) + r'}{l}{\textit{' + SETTING_LABELS[s] + r'}} \\')
    col_vals = {c: [mean_by_country(s, m, c) for m in MODELS] for c in COUNTRIES}
    col_vals['__avg__'] = [overall(s, m) for m in MODELS]
    best = {k: max(range(len(MODELS)), key=lambda i: v[i]) for k, v in col_vals.items()}
    for mi, m in enumerate(MODELS):
        row = [MODEL_LABELS[m]]
        for c in COUNTRIES:
            v = col_vals[c][mi]
            fmt = f'{v*100:.1f}'
            if mi == best[c]: fmt = r'\textbf{' + fmt + '}'
            row.append(fmt)
        v = col_vals['__avg__'][mi]
        fmt = f'{v*100:.1f}'
        if mi == best['__avg__']: fmt = r'\textbf{' + fmt + '}'
        row.append(fmt)
        lines.append(' & '.join(row) + r' \\')
    lines.append(r'\midrule' if s != SETTINGS[-1] else r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')

open(os.path.join(OUT_DIR, 'results_by_country.tex'), 'w').write('\n'.join(lines))

# ---------- CSV long format ----------
with open(os.path.join(OUT_DIR, 'results_full.csv'), 'w') as f:
    f.write('setting,model,country,category,correct,total,accuracy\n')
    for s in SETTINGS:
        for m in MODELS:
            for c in COUNTRIES:
                for cat in CATEGORIES:
                    a, b = accuracy[s][m][c][cat]
                    acc = (a/b) if b else 0.0
                    f.write(f'{s},{m},{c},{cat},{a},{b},{acc:.4f}\n')

# ---------- Table 3: full grid (category × country) per setting ----------
def acc_cell(s, m, country, cat):
    a, n = accuracy[s][m][country][cat]
    return (a / n) if n else 0.0

for s in SETTINGS:
    lines = []
    lines.append(r'% ' + SETTING_LABELS[s] + ' — full accuracy grid (category x country)')
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{3.2pt}')
    lines.append(r'\caption{CultureBenchmark accuracy (\%) — setting: ' + SETTING_LABELS[s] +
                 r'. Each task category split by 6 countries. Best per column in \textbf{bold}.}')
    lines.append(r'\label{tab:results_full_' + s + '}')
    # 1 (Model) + 4 categories * (6 countries + 1 avg) + 1 overall avg = 1 + 28 + 1 = 30 cols
    col_spec = 'l' + ('c' * 6 + '|c') * len(CATEGORIES) + '|c'
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    top = [r'\multirow{2}{*}{Model}']
    for cat in CATEGORIES:
        top.append(r'\multicolumn{7}{c|}{' + cat.capitalize() + '}')
    top.append(r'\multirow{2}{*}{Overall}')
    lines.append(' & '.join(top) + r' \\')
    # cmidrules under each category
    cmid = ''
    start = 2
    for _ in CATEGORIES:
        end = start + 6
        cmid += r'\cmidrule(lr){' + str(start) + '-' + str(end) + '} '
        start = end + 1
    lines.append(cmid.strip())
    # second header row: country labels + Avg
    sub = ['']
    for cat in CATEGORIES:
        for c in COUNTRIES:
            sub.append(COUNTRY_LABELS[c])
        sub.append('Avg')
    sub.append('')
    lines.append(' & '.join(sub) + r' \\')
    lines.append(r'\midrule')

    # collect per-column values to bold best
    ncol = 1 + len(CATEGORIES) * 7 + 1
    col_vals = [[] for _ in range(ncol)]
    for m in MODELS:
        row = [MODEL_LABELS[m]]
        j = 1
        for cat in CATEGORIES:
            for c in COUNTRIES:
                v = acc_cell(s, m, c, cat)
                row.append(v)
                col_vals[j].append(v); j += 1
            v = mean_by_cat(s, m, cat)
            row.append(v); col_vals[j].append(v); j += 1
        v = overall(s, m)
        row.append(v); col_vals[j].append(v)
        # defer formatting — need best indices
        # we'll format after
    # compute best row per col
    best_row = {j: max(range(len(MODELS)), key=lambda i: col_vals[j][i])
                for j in range(1, ncol)}
    for mi, m in enumerate(MODELS):
        row = [MODEL_LABELS[m]]
        j = 1
        for cat in CATEGORIES:
            for c in COUNTRIES:
                v = col_vals[j][mi]
                fmt = f'{v*100:.1f}'
                if mi == best_row[j]: fmt = r'\textbf{' + fmt + '}'
                row.append(fmt); j += 1
            v = col_vals[j][mi]
            fmt = f'{v*100:.1f}'
            if mi == best_row[j]: fmt = r'\textbf{' + fmt + '}'
            row.append(fmt); j += 1
        v = col_vals[j][mi]
        fmt = f'{v*100:.1f}'
        if mi == best_row[j]: fmt = r'\textbf{' + fmt + '}'
        row.append(fmt)
        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    open(os.path.join(OUT_DIR, f'results_full_{s}.tex'), 'w').write('\n'.join(lines))

# ---------- Table 4: merged full grid (all settings in one table)  ----------
# Rows grouped by model. Each category shows 6 country columns (no per-category Avg).
lines = []
lines.append(r'% Combined full accuracy grid (all settings) — grouped by model')
lines.append(r'\begin{table*}[t]')
lines.append(r'\centering')
lines.append(r'\scriptsize')
lines.append(r'\setlength{\tabcolsep}{3pt}')
lines.append(r'\caption{CultureBenchmark accuracy (\%) across three settings per model. Each task category is split by 6 countries; Overall is the mean across all cells. For each column, the best score across models \emph{within the same setting} is in \textbf{bold}.}')
lines.append(r'\label{tab:results_full_all}')
# columns: Model | Setting | 4x6 countries | Overall
col_spec = 'll' + ('cccccc|') * len(CATEGORIES) + 'c'
lines.append(r'\begin{tabular}{' + col_spec + '}')
lines.append(r'\toprule')
top = [r'\multirow{2}{*}{Model}', r'\multirow{2}{*}{Setting}']
for cat in CATEGORIES:
    top.append(r'\multicolumn{6}{c|}{' + cat.capitalize() + '}')
top.append(r'\multirow{2}{*}{Overall}')
lines.append(' & '.join(top) + r' \\')
cmid = ''
start = 3  # first category starts at col 3 (Model=1, Setting=2)
for _ in CATEGORIES:
    end = start + 5
    cmid += r'\cmidrule(lr){' + str(start) + '-' + str(end) + '} '
    start = end + 1
lines.append(cmid.strip())
sub = ['', '']
for cat in CATEGORIES:
    for c in COUNTRIES:
        sub.append(COUNTRY_LABELS[c])
sub.append('')
lines.append(' & '.join(sub) + r' \\')
lines.append(r'\midrule')

# Precompute per-setting best-model index for each column
ncol_data = 1 + len(CATEGORIES) * 6 + 1   # excluding Setting col (col 2)
best_by_setting = {}
for s in SETTINGS:
    col_vals = [[] for _ in range(ncol_data)]
    for m in MODELS:
        j = 1
        for cat in CATEGORIES:
            for c in COUNTRIES:
                col_vals[j].append(acc_cell(s, m, c, cat)); j += 1
        col_vals[j].append(overall(s, m))
    best_by_setting[s] = ({j: max(range(len(MODELS)), key=lambda i: col_vals[j][i])
                           for j in range(1, ncol_data)}, col_vals)

for mi, m in enumerate(MODELS):
    for si, s in enumerate(SETTINGS):
        best, col_vals = best_by_setting[s]
        row = []
        if si == 0:
            row.append(r'\multirow{3}{*}{' + MODEL_LABELS[m] + '}')
        else:
            row.append('')
        row.append(SETTING_LABELS[s])
        j = 1
        for cat in CATEGORIES:
            for c in COUNTRIES:
                v = col_vals[j][mi]
                fmt = f'{v*100:.1f}'
                if mi == best[j]: fmt = r'\textbf{' + fmt + '}'
                row.append(fmt); j += 1
        v = col_vals[j][mi]
        fmt = f'{v*100:.1f}'
        if mi == best[j]: fmt = r'\textbf{' + fmt + '}'
        row.append(fmt)
        lines.append(' & '.join(row) + r' \\')
    if mi < len(MODELS) - 1:
        lines.append(r'\midrule')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')
open(os.path.join(OUT_DIR, 'results_full_all.tex'), 'w').write('\n'.join(lines))

print('saved:')
for f in ['results_by_category.tex', 'results_by_country.tex', 'results_full.csv',
          'results_full_direct.tex', 'results_full_reasoning.tex', 'results_full_rule_given.tex',
          'results_full_all.tex']:
    print(' ', os.path.join(OUT_DIR, f))
