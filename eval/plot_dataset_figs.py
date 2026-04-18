"""Regenerate dataset distribution figures (Fig_*) for the updated eval set
(culturebenchmark_eval.json, 5047 items / 6 countries / 4 task categories)."""
import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_EVAL_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
BENCHMARK = str(_REPO_ROOT / 'culturebenchmark_eval.json')
OUT_DIR   = str(_EVAL_DIR / 'plots')
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300

PALETTE_MAIN = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                '#59A14F', '#EDC948', '#B07AA1', '#9C755F', '#BAB0AC']

# Country palette — same mapping as model plot
COUNTRY_COLOR = {
    'cn':  '#E15759',  # red
    'us':  '#4E79A7',  # blue
    'uk':  '#EDC948',  # yellow
    'jp':  '#B07AA1',  # purple
    'sg':  '#59A14F',  # green
    'ind': '#9C755F',  # brown
}
COUNTRY_LABEL = {'cn': 'China', 'us': 'US', 'uk': 'UK', 'jp': 'Japan',
                 'sg': 'Singapore', 'ind': 'India'}

bench = json.load(open(BENCHMARK))
print(f'loaded {len(bench)} items')

# ---------- Helpers ----------
def src_of(path: str) -> str:
    s = path.lower()
    if 'once'     in s: return 'ONCE'
    if 'covla'    in s: return 'CoVLA'
    if 'lingoqa' in s or '/lingo' in s: return 'LingoQA'
    if 'nuscenes' in s: return 'nuScenes'
    if 'idd'      in s: return 'IDD'
    if 'waymo'    in s: return 'Waymo'
    return 'unknown'

# keyword set per topic
TOPIC_KWS = {
    'Legality':       [r'\bis\s+it\s+legal', r'\billegal', r'\bpermitted', r'\ballowed', r'may\s+.*\s+turn'],
    'Turning Rule':   [r'\bturn(ing)?\b', r'right[- ]?turn', r'left[- ]?turn', r'u[- ]?turn'],
    'Lane Rule':      [r'\blane\b', r'merge', r'change\s+lane'],
    'Traffic Signal': [r'traffic\s+light', r'signal', r'red\s+light', r'green\s+light', r'yellow\s+light'],
    'License Plate':  [r'license\s+plate', r'plate\b'],
    'Road Sign':      [r'\bsign\b', r'road\s+sign'],
    'Pedestrian':     [r'pedestrian'],
    'Crossing':       [r'crossing|crosswalk|zebra'],
    'Stopping':       [r'\bstop\b|stopping|parking\s+stop'],
    'Shape':          [r'\bshape\b|octagon|triangle|diamond'],
    'Overtaking':     [r'overtak|pass(ing)?'],
    'Parking':        [r'\bpark(ing)?\b'],
    'Speed':          [r'speed\s+limit|speed'],
}

def topic_of(text: str):
    t = (text or '').lower()
    matches = []
    for topic, kws in TOPIC_KWS.items():
        if any(re.search(kw, t) for kw in kws):
            matches.append(topic)
    # return first match (priority order)
    return matches[0] if matches else 'Other'

# Scenario-id → road-type mapping (from scenarios.json titles)
SID_TO_ROAD = {
    'S1':  'Signalized Intersection',
    'S2':  'Unsignalized Intersection',
    'S3':  'Multi-lane Road',
    'S4':  'Unsignalized Intersection',
    'S5':  'Signalized Intersection',
    'S6':  'Signalized Intersection',
    'S7':  'Residential / Single Lane',
    'S8':  'Multi-lane Road',
    'S9':  'Other',
    'S10': 'Multi-lane Road',
    'S11': 'Multi-lane Road',
    'S12': 'Multi-lane Road',
    'S13': 'Multi-lane Road',
}

def road_scenario_of(item):
    refs = item.get('rule_reference') or []
    if refs:
        # use first ref's mapped category
        for r in refs:
            road = SID_TO_ROAD.get(r)
            if road: return road
    t = (item.get('question') or '').lower()
    if re.search(r'signalized\s+intersection|traffic\s+light|signal(?!\s+before)', t):
        return 'Signalized Intersection'
    if re.search(r'unsignalized|roundabout|uncontrolled', t):
        return 'Unsignalized Intersection'
    if re.search(r'\blane\b|highway|expressway|freeway|motorway|multi[- ]?lane', t):
        return 'Multi-lane Road'
    if re.search(r'intersection|junction|crossroad|crosswalk|pedestrian', t):
        return 'Signalized Intersection'
    if re.search(r'residential|single\s+lane|single-lane|one\s+lane|school', t):
        return 'Residential / Single Lane'
    return 'Other'

# =============== Fig 1: category pie ===============
cat_counts = Counter(x['question_category'] for x in bench)
order = ['perception', 'prediction', 'planning', 'region']
sizes = [cat_counts[c] for c in order]
labels = [c.capitalize() for c in order]
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=PALETTE_MAIN[:4], startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=13))
for t in autotexts:
    t.set_color('white'); t.set_fontweight('bold')
ax.set_title('Task Category Distribution', fontsize=16, fontweight='bold', pad=14)
fig.savefig(f'{OUT_DIR}/fig_category.png', dpi=DPI, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_category.pdf', bbox_inches='tight')
plt.close(fig)
print('saved fig_category')

# =============== Fig 2: country pie ===============
country_counts = Counter(x['country'] for x in bench)
order_c = ['cn', 'jp', 'sg', 'uk', 'ind', 'us']
sizes = [country_counts[c] for c in order_c]
labels = [COUNTRY_LABEL[c] for c in order_c]
colors = [COUNTRY_COLOR[c] for c in order_c]
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=colors, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=13))
for t in autotexts:
    t.set_color('white'); t.set_fontweight('bold')
ax.set_title('Country Distribution', fontsize=16, fontweight='bold', pad=14)
fig.savefig(f'{OUT_DIR}/fig_country.png', dpi=DPI, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_country.pdf', bbox_inches='tight')
plt.close(fig)
print('saved fig_country')

# =============== Fig 3: source dataset donut ===============
src_counts = Counter()
for x in bench:
    paths = x.get('image_path') or []
    if paths:
        src_counts[src_of(paths[0])] += 1
order_s = ['ONCE', 'CoVLA', 'LingoQA', 'nuScenes', 'IDD', 'Waymo']
sizes = [src_counts[s] for s in order_s]
labels = [f'{s}\n{src_counts[s]} ({src_counts[s]/len(bench)*100:.1f}%)' for s in order_s]
# Match source → country color
SRC_TO_COUNTRY = {'ONCE': 'cn', 'CoVLA': 'jp', 'LingoQA': 'uk',
                  'nuScenes': 'sg', 'IDD': 'ind', 'Waymo': 'us'}
colors = [COUNTRY_COLOR[SRC_TO_COUNTRY[s]] for s in order_s]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
wedges, texts = ax.pie(
    sizes, labels=labels, colors=colors, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=3, width=0.38),
    textprops=dict(fontsize=11))
ax.set_title('Source Dataset Distribution', fontsize=16, fontweight='bold', pad=14)
fig.savefig(f'{OUT_DIR}/fig_source_dataset.png', dpi=DPI, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_source_dataset.pdf', bbox_inches='tight')
plt.close(fig)
print('saved fig_source_dataset')

# =============== Fig 4: road scenario pie ===============
rs_counts = Counter(road_scenario_of(x) for x in bench)
order_r = ['Signalized Intersection', 'Multi-lane Road',
           'Unsignalized Intersection', 'Residential / Single Lane', 'Other']
sizes = [rs_counts[s] for s in order_r]
labels = [s for s in order_r]
colors = ['#4E79A7', '#F28E2B', '#E15759', '#BAB0AC', '#C9C9C9']
fig, ax = plt.subplots(figsize=(7.5, 7.5))
# explode smaller slices for readability
explode = [0, 0, 0.02, 0.04, 0.06]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=colors, startangle=90, explode=explode,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=11))
for t in autotexts:
    t.set_color('white'); t.set_fontweight('bold')
ax.set_title('Road Scenario Distribution', fontsize=16, fontweight='bold', pad=14)
fig.savefig(f'{OUT_DIR}/fig_road_scenario.png', dpi=DPI, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_road_scenario.pdf', bbox_inches='tight')
plt.close(fig)
print('saved fig_road_scenario')

# =============== Fig 5: keywords_by_country (6 panels) ===============
country_topics = defaultdict(Counter)
for x in bench:
    country_topics[x['country']][topic_of(x['question'])] += 1

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
order_panel = ['cn', 'jp', 'sg', 'uk', 'ind', 'us']

for ax, c in zip(axes.flat, order_panel):
    topics = country_topics[c]
    # drop 'Other', take top 8
    items = [(t, n) for t, n in topics.most_common() if t != 'Other'][:8]
    topics_sorted = sorted(items, key=lambda kv: kv[1])  # ascending for horizontal bar
    names = [t for t, _ in topics_sorted]
    counts = [n for _, n in topics_sorted]
    color = COUNTRY_COLOR[c]
    bars = ax.barh(names, counts, color=color, edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, counts):
        ax.text(v + max(counts) * 0.012, bar.get_y() + bar.get_height() / 2,
                str(v), va='center', fontsize=10, fontweight='bold')
    ax.set_title(COUNTRY_LABEL[c], fontsize=13, fontweight='bold')
    ax.set_xlabel('Count', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle('Question Topic Distribution by Country', fontsize=16, fontweight='bold', y=1.00)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_keywords_by_country.png', dpi=DPI, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_keywords_by_country.pdf', bbox_inches='tight')
plt.close(fig)
print('saved fig_keywords_by_country')

print('done.')
