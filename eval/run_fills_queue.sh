#!/bin/bash
# Usage: run_fills_queue.sh <gpu> <model1> <setting1> <model2> <setting2> ...
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dllm
export TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

# Resolve paths relative to this script.
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$EVAL_DIR")"
BENCHMARK="$REPO_ROOT/culturebenchmark_eval.json"
OUTDIR="$EVAL_DIR/results"
LOGDIR="$EVAL_DIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
cd "$EVAL_DIR"

gpu=$1; shift
export CUDA_VISIBLE_DEVICES=$gpu

while [ $# -ge 2 ]; do
  m=$1; s=$2; shift 2
  for try in 1 2 3; do
    echo "[GPU$gpu] FILL $m/$s try=$try START $(date +%H:%M:%S)"
    python infer_vllm.py --model "$m" --setting $s \
      --output "$OUTDIR/${m}_${s}_results.json" \
      --max_images 3 --max_new_tokens_long 512 --batch_size 64 \
      --benchmark "$BENCHMARK" \
      > "$LOGDIR/fill_${m}_${s}.log" 2>&1
    rc=$?
    python3 -c "
import json; p='$OUTDIR/${m}_${s}_results.json'
d=json.load(open(p)); c=[r for r in d if not str(r.get('pred','')).startswith('ERROR')]
json.dump(c,open(p,'w'),ensure_ascii=False,indent=2)" 2>/dev/null
    n=$(python3 -c "import json; d=json.load(open('$OUTDIR/${m}_${s}_results.json')); print(len(d))")
    echo "[GPU$gpu] $m/$s try=$try END rc=$rc total=$n"
    [ "$n" -ge 5000 ] && break
  done
done
echo "[GPU$gpu] QUEUE DONE $(date)"
