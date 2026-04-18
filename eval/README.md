# CultureBenchmark Evaluation

Eval harness for 9 VLMs on CultureBenchmark (5,047 items · 6 countries · 4 task categories).

## Files

```
eval/
├── _utils.py               # Shared constants (COUNTRIES / CATEGORIES / PALETTE),
│                           # composite_key, extract_ans, load/save/resume helpers
├── infer_vllm.py           # 7 open-source VLMs via vLLM / lmdeploy / HF
├── infer_api.py            # Unified OpenAI + AWS Bedrock remote inference
├── run_fills_queue.sh      # Generic per-GPU queue with retry + ERROR clean
├── make_tables.py          # Generate LaTeX tables from result JSONs
├── plot_by_category.py     # 1×4 grouped-bar plot per setting
├── plot_dataset_figs.py    # Dataset distribution figures
├── validate_eval.py        # LLM-as-judge filter audit
├── REPORT.md               # Filtering strategy (5,873 → 5,047)
├── results/                # 27 final result JSONs (9 models × 3 settings)
├── tables/                 # LaTeX tables + CSV
├── plots/                  # PNG/PDF figures
├── logs/                   # Run logs
└── archive/                # Old scripts / docs / results
```

## Models covered

| Model | Backend | Script |
|---|---|---|
| LLaVA-1.6-7B | vLLM | `infer_vllm.py --model llava` |
| Gemma3-12B | vLLM | `infer_vllm.py --model gemma3` |
| InternVL3-8B | vLLM | `infer_vllm.py --model internvl3` |
| InternVL3.5-8B | lmdeploy | `infer_vllm.py --model internvl35` |
| Qwen2.5-VL-7B | vLLM | `infer_vllm.py --model qwen25vl` |
| Qwen3-VL-8B | vLLM | `infer_vllm.py --model qwen3vl` |
| Llama-3.2-11B-Vision | HF | `infer_vllm.py --model llama32` |
| GPT-5.4 | OpenAI API | `infer_api.py --backend openai --model gpt-5.4` |
| Qwen3-VL-235B-A22B | AWS Bedrock | `infer_api.py --backend bedrock --model qwen.qwen3-vl-235b-a22b` |

## Settings (prompting strategy)

| Setting | Prompt style |
|---|---|
| `direct` | Output only the answer letter (A/B/C/D) |
| `reasoning` | 4-step CoT: infer country → cultural rule → visual reasoning → answer |
| `rule_given` | Handbook rule text provided; 4-step CoT conditioned on the rule |

## Running inference

Each script resumes from existing `results/<model>_<setting>_results.json` using composite key `(id, image_path, question)` and re-runs any entries whose `pred` starts with `ERROR`.

### Local (single GPU, vLLM / lmdeploy / HF)

```bash
cd eval
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py --model qwen3vl --setting direct
```

### Queue multiple (model, setting) pairs on one GPU

```bash
# Run qwen3vl then gemma3 on GPU 1, 3 settings each
bash run_fills_queue.sh 1 qwen3vl direct qwen3vl reasoning qwen3vl rule_given \
                          gemma3 direct gemma3 reasoning gemma3 rule_given
```

### GPT-5 / GPT-5.4 (OpenAI)

```bash
export OPENAI_API_KEY='sk-...'
python infer_api.py --backend openai --model gpt-5.4 --setting direct     --workers 12
python infer_api.py --backend openai --model gpt-5.4 --setting reasoning  --workers 12
python infer_api.py --backend openai --model gpt-5.4 --setting rule_given --workers 12
```

### Qwen3-VL-235B (AWS Bedrock)

**Pass credentials inline — never `export` them (they affect other AWS tooling).**

```bash
AWS_BEARER_TOKEN_BEDROCK='ABSK...' AWS_REGION=us-east-2 \
    python infer_api.py --backend bedrock --model qwen.qwen3-vl-235b-a22b \
                        --setting direct --workers 8
```

## Analysis

```bash
python make_tables.py         # → tables/results_full_all.tex (the paper table)
python plot_by_category.py    # → plots/accuracy_by_category_<setting>.png
python plot_dataset_figs.py   # → plots/fig_*.png
```

## Data

Paths are relative to the repo root:

- `culturebenchmark_eval.json` — 5,047 items · 3,848 unique IDs · 1,199 counterfactual duplicates
- `eval_validation_results.json` — 5,873 pre-filter items with LLM-as-judge verdicts (see `REPORT.md`)
