#!/usr/bin/env bash
# Per-layer margin sweep, the {m_l} counterpart to sweep_margins.sh.
#
# Same fixed window (11-18, end-exclusive) as the paper's Optuna result on
# Llama-3-8B, but varying the margin SCHEDULE across layers instead of one shared
# scalar. Schedules are mean-matched (~1.5) so the comparison isolates the shape
# of {m_l}, not its average size.
#
#   shared     paper Table 3 CLE-A value, flat
#   paper      read off Fig. 7(b) of arXiv:2605.21706
#   ramp_up    monotone increasing, same mean as shared
#   ramp_down  monotone decreasing, same mean as shared
#
# Per-layer runs are tagged marginvec<sha1> rather than margin<float>, so this
# script prints a label -> run_tag manifest; without it the output filenames are
# not attributable to a schedule.
#
# Usage:
#   bash experiments/sweep_layer_margins.sh
#   LIMIT=10 MAX_NEW=512 bash experiments/sweep_layer_margins.sh

set -euo pipefail

MODEL="${MODEL:-llama3-8b}"
DATASET="${DATASET:-harmbench_test}"
LAYERS="${LAYERS:-11-18}"
LIMIT="${LIMIT:-5}"
MAX_NEW="${MAX_NEW:-256}"
SEED="${SEED:-0}"
TARGET="${TARGET:-cle-a.py}"

cd "$(dirname "$0")/.."

LABELS=(shared paper ramp_up ramp_down)
SCHEDULES=(
    "1.5,1.5,1.5,1.5,1.5,1.5,1.5"
    "1.2,2.0,1.8,1.8,2.0,0.9,1.2"
    "0.9,1.1,1.3,1.5,1.7,1.9,2.1"
    "2.1,1.9,1.7,1.5,1.3,1.1,0.9"
)

echo "=========================================="
echo "Model:   $MODEL   Target: $TARGET"
echo "Dataset: $DATASET  (limit=$LIMIT)"
echo "Layers:  $LAYERS"
echo "Seed:    $SEED"
echo "=========================================="

echo
echo "### Baseline (no steering) ###"
python experiments/baseline_generate.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW" \
    --seed "$SEED"

MANIFEST="completions/${MODEL}/layer_margin_manifest.txt"
mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    VEC="${SCHEDULES[$i]}"
    TAG=$(python utils/margin_utils.py --layers "$LAYERS" --layer_margin "$VEC" \
          | sed -n 's/^tag=//p')

    echo
    echo "### $LABEL  m_l=[$VEC]  ->  $TAG ###"
    echo -e "${LABEL}\t${VEC}\t${TAG}" >> "$MANIFEST"

    python "$TARGET" \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --limit "$LIMIT" \
        --layers "$LAYERS" \
        --layer_margins "$VEC" \
        --max_new_tokens "$MAX_NEW" \
        --seed "$SEED"
done

echo
echo "Sweep complete. Label -> run_tag manifest ($MANIFEST):"
cat "$MANIFEST"
