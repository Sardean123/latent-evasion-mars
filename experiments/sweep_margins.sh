#!/usr/bin/env bash
# Small margin sweep for eyeballing steering effects.
#
# Fixed layer window (11-18, end-exclusive) from the paper's Optuna result on Llama-3-8B.
# Runs on --limit N prompts from harmbench_test. Includes a hookless baseline.
#
# Usage:
#   bash experiments/sweep_margins.sh                    # defaults
#   MARGINS="1.0 3.0 6.0" LIMIT=1 bash experiments/sweep_margins.sh
#
# All outputs land under completions/{MODEL}/{baseline,projection,pipeline}/.
# Use experiments/view_completions.py to eyeball them side-by-side.

set -euo pipefail

MODEL="${MODEL:-llama3-8b}"
DATASET="${DATASET:-harmbench_test}"
LAYERS="${LAYERS:-11-18}"
LIMIT="${LIMIT:-1}"
MAX_NEW="${MAX_NEW:-256}"
SEED="${SEED:-0}"
MARGINS="${MARGINS:-1.0 3.0 6.0}"

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Model:    $MODEL"
echo "Dataset:  $DATASET  (limit=$LIMIT)"
echo "Layers:   $LAYERS"
echo "Margins:  $MARGINS"
echo "Seed:     $SEED"
echo "=========================================="

echo
echo "### Baseline (no steering) ###"
python experiments/baseline_generate.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW" \
    --seed "$SEED"

for M in $MARGINS; do
    echo
    echo "### CLE-P  layers=$LAYERS  margin=$M ###"
    python cle-p.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --limit "$LIMIT" \
        --layers "$LAYERS" \
        --margin "$M" \
        --max_new_tokens "$MAX_NEW" \
        --seed "$SEED"

    echo
    echo "### CLE-A  layers=$LAYERS  margin=$M ###"
    python cle-a.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --limit "$LIMIT" \
        --layers "$LAYERS" \
        --margin "$M" \
        --max_new_tokens "$MAX_NEW" \
        --seed "$SEED"
done

echo
echo "Sweep complete."
echo "View results with:"
echo "  python experiments/view_completions.py --model_name $MODEL"
