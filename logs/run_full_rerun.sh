#!/bin/bash
# Full re-run on HarmBench standard (n=200): {CLE-A, CLE-P} x {bo-external, hlmean}.
#
# Every run is judged TWICE:
#   * harmbench    -- the standard test-set protocol (cais/HarmBench-Llama-2-13b-cls), binary
#                     is_jailbreak_harmbench + ASR. Comparable with RESULTS_SUMMARY.md.
#   * strongreject -- fine-tuned evaluator (LoRA on gemma-2b), continuous strongreject_score in
#                     [0,1] + mean. No threshold, no ASR -- StrongREJECT does not define one.
# Both keys land on the same completions, so neither judge displaces the other.
#
# Margins come from config/margins.json by name; the schedule also supplies the layer window.
# NOTE bo-external has optimized_for="unknown", so one of the two CLE variants below is
# method-mismatched and the runs will say so. That is the accepted decision, not an oversight.
#
# Phases are separate so a failure in generation does not leave half-judged artifacts, and so
# judging can be re-run alone. Strictly serial: one GPU.
set -e
cd /workspace/lem-durable/repo

export HF_HOME=/workspace/.cache/huggingface/
export HF_HUB_ENABLE_HF_TRANSFER=1

SCHEDULES="bo-external hlmean"
COMMON="--model_name llama3-8b --device cuda:0 --beta 1.0 \
  --dataset harmbench_standard --max_new_tokens 512 --batch_size 16"

out_dir_for() {
  case "$1" in
    cle-a) echo ./completions/llama3-8b/cle-a ;;
    cle-p) echo ./completions/llama3-8b/projection ;;
    *) echo "unknown method $1" >&2; exit 1 ;;
  esac
}

echo "############ PHASE 1/2: generation (4 runs) ############"
for M in cle-a cle-p; do
  OUT=$(out_dir_for $M)
  for S in $SCHEDULES; do
    echo "=== [gen] $M / margin_schedule=$S ==="
    python $M.py $COMMON --margin_schedule "$S" --out_dir "$OUT"
  done
done
echo "############ GENERATION DONE ############"

echo "############ PHASE 2/2: judging (4 runs x 2 judges) ############"
for M in cle-a cle-p; do
  OUT=$(out_dir_for $M)
  for S in $SCHEDULES; do
    F=completions_harmbench_standard_FULL_layers11to18_beta1.0_margin${S}_seed0.json
    echo "=== [judge] $M / $S ==="
    python utils/eval_jailbreaks.py \
      --completions_path "$OUT/$F" \
      --methodologies harmbench strongreject \
      --evaluation_path "$OUT/evaluation/evaluation_${F#completions_}"
  done
done
echo "############ ALL DONE ############"
