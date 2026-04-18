"""Clean publication-style 1x4 grouped-bar plots per setting."""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (COUNTRIES, CATEGORIES, SETTINGS, PALETTE as _PAL,
                    composite_key, extract_ans)

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

_EVAL_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
RESULT_DIR = str(_EVAL_DIR / 'results')
BENCHMARK  = str(_REPO_ROOT / 'culturebenchmark_eval.json')
OUT_DIR    = str(_EVAL_DIR / 'plots')

MODELS = ['llava', 'gemma3', 'internvl3', 'internvl35', 'qwen25vl', 'qwen3vl', 'llama32']
MODEL_LABELS = ['LLaVA-1.6', 'Gemma3-12B', 'InternVL3-8B', 'InternVL3.5-8B',
                'Qwen2.5VL-7B', 'Qwen3VL-8B', 'Llama-3.2-11B']
COUNTRY_LABELS = ['CN', 'US', 'UK', 'JP', 'SG', 'IND']
PALETTE = [_PAL[c] for c in COUNTRIES]   # list in COUNTRIES order

bench = json.load(open(BENCHMARK))
bench_info = {composite_key(x): (x['country'], x['question_category']) for x in bench}

def compute_accuracy(setting):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for m in MODELS:
        path = os.path.join(RESULT_DIR, f'{m}_{setting}_results.json')
        if not os.path.exists(path): continue
        data = json.load(open(path))
        uniq = {composite_key(r): r for r in data}
        for k, r in uniq.items():
            info = bench_info.get(k)
            if info is None: continue
            country, cat = info
            pred_ans = extract_ans(r.get('pred', ''))
            gt = r.get('gt')
            acc[m][country][cat][1] += 1
            if pred_ans == gt:
                acc[m][country][cat][0] += 1
    return acc

def plot_setting(setting):
    acc = compute_accuracy(setting)
    fig, axes = plt.subplots(1, 4, figsize=(36, 6), sharey=True)
    fig.patch.set_facecolor('white')

    n_c = len(COUNTRIES)
    n_m = len(MODELS)
    model_gap = 2.8          # unit spacing between model groups
    group_w = 2.0            # bar group width
    bar_w = group_w / n_c
    x = np.arange(n_m) * model_gap

    for ax_idx, (ax, cat) in enumerate(zip(axes, CATEGORIES)):
        ax.set_facecolor('white')
        # grouped bars: countries as hue
        for i, c in enumerate(COUNTRIES):
            vals = []
            for m in MODELS:
                corr, tot = acc[m][c][cat]
                vals.append(corr / tot if tot else 0.0)
            offsets = x - group_w/2 + (i + 0.5) * bar_w
            ax.bar(offsets, vals, bar_w * 0.98,
                   color=PALETTE[i], label=COUNTRY_LABELS[i],
                   edgecolor='white', linewidth=0.4)

        # per-model mean line
        model_means = []
        for m in MODELS:
            tot_c, tot_n = 0, 0
            for c in COUNTRIES:
                a, n = acc[m][c][cat]
                tot_c += a; tot_n += n
            model_means.append(tot_c / tot_n if tot_n else 0)
        ax.plot(x, model_means, '-', color='#222222', linewidth=1.6,
                zorder=5, alpha=0.85)
        ax.scatter(x, model_means, s=55, color='white',
                   edgecolors='#222222', linewidths=1.6, zorder=6)
        for xi, v in zip(x, model_means):
            ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=9,
                    color='#222222', zorder=7)

        # overall avg dashed
        overall = sum(model_means) / len(model_means) if model_means else 0
        ax.axhline(overall, linestyle=(0, (4, 3)), linewidth=1.3,
                   color='#888888', zorder=2)
        ax.text(n_m - 0.4, overall + 0.008, f'avg={overall:.2f}',
                ha='right', fontsize=9, color='#555555', zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, rotation=22, ha='right', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(cat.capitalize(), fontsize=13, fontweight='600', pad=8)
        ax.yaxis.grid(True, alpha=0.35, linestyle='-', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=12)

    # single shared legend at top
    handles = [plt.Rectangle((0,0),1,1, color=PALETTE[i]) for i in range(n_c)]
    fig.legend(handles, COUNTRY_LABELS, loc='upper center',
               ncol=n_c, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.96))

    fig.suptitle(f'CultureBenchmark Accuracy — setting: {setting}',
                 fontsize=15, fontweight='600', y=1.02)
    fig.subplots_adjust(top=0.84, wspace=0.08)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = os.path.join(OUT_DIR, f'accuracy_by_category_{setting}.png')
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out)

    summary = {m: {c: {cat: {'acc': round(acc[m][c][cat][0]/acc[m][c][cat][1],4)
                              if acc[m][c][cat][1] else None,
                              'n': acc[m][c][cat][1]}
                        for cat in CATEGORIES}
                    for c in COUNTRIES}
                for m in MODELS}
    json.dump(summary, open(out.replace('.png', '.json'), 'w'),
              ensure_ascii=False, indent=2)

for s in SETTINGS:
    plot_setting(s)
