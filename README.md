# CulturalDrive-Bench

A cross-cultural VLM benchmark for autonomous-driving perception, prediction, planning, and regional traffic-rule knowledge — **5,047 multiple-choice items across 6 countries** (CN / US / UK / JP / SG / IND) drawn from real driving-scene datasets (ONCE, Waymo, LingoQA, CoVLA, nuScenes, IDD) and the countries' official traffic handbooks.

## Headline results

9 VLMs evaluated under 3 prompting protocols. Accuracy is % correct on 5,047 items (equal weight across countries and categories).

| Model | Direct | Reasoning | Rule-Given |
|---|---:|---:|---:|
| LLaVA-1.6-7B | 41.7 | 47.3 | 62.8 |
| Gemma3-12B | 64.5 | 65.3 | 76.0 |
| InternVL3-8B | 70.0 | 66.6 | 81.3 |
| InternVL3.5-8B | 38.0 | 37.7 | 53.8 |
| Qwen2.5-VL-7B | 65.3 | 56.0 | 73.1 |
| Qwen3-VL-8B | 64.8 | 44.4 | 80.2 |
| Llama-3.2-11B-Vision | 51.2 | 52.8 | 68.9 |
| **Qwen3-VL-235B-A22B** | **80.9** | 82.7 | 90.8 |
| **GPT-5.4** | **82.5** | **88.0** | **91.9** |

Full breakdown (task category × country × setting): `eval/tables/results_full_all.tex`.

## The benchmark

- **5,047 multiple-choice questions**, 4 task categories:
  | Category | Count | What it tests |
  |---|---:|---|
  | Perception | 1,527 | What is visible in the scene (plate color, sign identity, signal state) |
  | Prediction | 1,088 | What will likely happen next |
  | Planning | 1,179 | What the ego vehicle should do |
  | Region | 1,253 | Country-specific traffic-law knowledge (no image required) |
- Two failure modes are intentionally preserved as **counterfactual variants** to probe cultural bias:
  - 54 question IDs have the same question text but different images from different countries
  - 70 IDs total are reused — see the dedup discussion in `eval/REPORT.md`
- 826 candidate items were **dropped by an LLM-as-judge filter** (5,873 → 5,047). See `eval/REPORT.md` for the strategy, verdict distribution, and failure-mode breakdown.

## Three prompting protocols

| Setting | Prompt style |
|---|---|
| `direct` | Answer with one letter (A/B/C/D). |
| `reasoning` | 4-step chain-of-thought: infer country → cite cultural rule → visual reasoning → answer. |
| `rule_given` | Relevant traffic-handbook rule is provided in the prompt; model then reasons step-by-step. |

Rule text per country/scenario lives in `traffic_handbook/` — six Python modules (`cn.py`, `us.py`, `uk.py`, `jp.py`, `sg.py`, `ind.py`) that compile into plain-text handbooks used by the `rule_given` protocol.

## Repository layout

```
CulturalDrive-Bench/
├── culturebenchmark_eval.json      # the 5,047-item eval set (the benchmark)
├── eval_validation_results.json    # LLM-as-judge audit over 5,873 pre-filter candidates
├── annotations_human.json          # human-verified labels on a subset
├── traffic_handbook/               # country handbook generators (6 .py files)
├── dataset/                        # QA generation pipeline (annotation, QA gen, scenario detection)
├── fliter.py                       # question-rewriting utility (AWS Bedrock)
└── eval/
    ├── README.md                   # how to reproduce inference + analysis
    ├── REPORT.md                   # filtering strategy (5,873 → 5,047)
    ├── _utils.py                   # shared constants / extract_ans / composite_key
    ├── infer_vllm.py               # 7 open-source VLMs via vLLM / lmdeploy / HF
    ├── infer_api.py                # unified OpenAI + AWS Bedrock remote inference
    ├── run_fills_queue.sh          # generic per-GPU queue with retry + ERROR-drop
    ├── make_tables.py              # LaTeX tables from result JSONs
    ├── plot_by_category.py         # 1×4 grouped-bar plot per setting
    ├── plot_dataset_figs.py        # dataset distribution figures
    ├── validate_eval.py            # LLM-as-judge filter run
    ├── tables/                     # generated LaTeX tables + CSV
    └── plots/                      # generated PNG/PDF figures
```

Model prediction JSONs (~240 MB total) are too large for git and are distributed separately via a GitHub Release.

## Quick start

### Install

```bash
git clone https://github.com/gray311/CulturalDrive-Bench.git
cd CulturalDrive-Bench
pip install vllm lmdeploy transformers openai boto3 tqdm matplotlib pillow
```

### Reproduce a single model × setting

Open-source VLMs (single GPU):

```bash
cd eval
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py --model qwen3vl --setting direct
```

Supported `--model` values (see `eval/infer_vllm.py` for full registry):
`llava, gemma3, internvl3, internvl35, qwen25vl, qwen3vl, llama32`.

### Remote APIs

```bash
# GPT-5 / GPT-5.4 — set your key FIRST
export OPENAI_API_KEY='sk-...'
python eval/infer_api.py --backend openai --model gpt-5.4 --setting direct --workers 12

# Qwen3-VL-235B on AWS Bedrock — pass creds INLINE, never export
AWS_BEARER_TOKEN_BEDROCK='ABSK...' AWS_REGION=us-east-2 \
    python eval/infer_api.py --backend bedrock --model qwen.qwen3-vl-235b-a22b \
                             --setting direct --workers 8
```

All inference scripts resume via composite key `(id, image_path, question)`; any entry whose `pred` starts with `ERROR` is retried on the next run.

### Batch a whole model on one GPU

```bash
bash eval/run_fills_queue.sh 0 qwen3vl direct qwen3vl reasoning qwen3vl rule_given
```

### Regenerate tables & plots

```bash
python eval/make_tables.py        # → eval/tables/*.tex + results_full.csv
python eval/plot_by_category.py   # → eval/plots/accuracy_by_category_*.png
python eval/plot_dataset_figs.py  # → eval/plots/fig_*.png
```

## Data distribution & dedup

The benchmark contains **3,848 unique question IDs** across 5,047 rows. 70 IDs repeat:
- 54 reuse the same question with a different image (counterfactual perception probes)
- 16 IDs are re-used for completely different questions — an artifact of the generator pipeline

All evaluation code de-duplicates on the composite key `(id, image_path, question)` so every logical item is scored exactly once.

See `eval/REPORT.md` for the upstream filtering strategy and the 70 dropped-reason breakdown.

## Environment variables

Never commit these. Set them in your shell when you need them:

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | for `infer_api.py --backend openai` |
| `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` | for `infer_api.py --backend bedrock` and `eval/validate_eval.py` |

## License

Code: MIT. Benchmark images inherit from their upstream dataset licenses (ONCE, Waymo, LingoQA, CoVLA, nuScenes, IDD) — see those projects for terms.

## Citation

TBD.
