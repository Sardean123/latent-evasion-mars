#!/bin/bash
# The StrongREJECT BENCHMARK: {CLE-A, CLE-P} x {bo-external, hlmean} on the native 313-prompt
# StrongREJECT set, judged by the StrongREJECT fine-tuned evaluator.
#
# This is the matched configuration. `run_full_rerun.sh` scores the StrongREJECT JUDGE over
# HARMBENCH prompts (n=200, zero overlap with these 313) -- a valid continuous-harmfulness
# measurement, but not the StrongREJECT benchmark, and the two must not be conflated or
# compared cell-to-cell. Different prompts, different denominator.
#
# dataset/processed/strong_reject.json ships with upstream pralab/latent-evasion (initial
# commit) and is registered in load_dataset.py, but nothing upstream ever selects it and the
# paper never mentions StrongREJECT. So this is the first time the file is used for anything.
#
# Judged TWICE, as in run_full_rerun.sh:
#   * strongreject -- the headline. Continuous [0,1], no threshold, no ASR.
#   * harmbench    -- CAVEAT: cais/HarmBench-Llama-2-13b-cls expects HarmBench *behavior*
#                     strings. These are StrongREJECT instructions, so it runs off-distribution.
#                     Kept because an ASR column is still useful for reading the gap against
#                     the continuous score, but it is NOT the standard HarmBench protocol here
#                     and should be labelled as off-distribution wherever it is reported.
set -e
cd /workspace/lem-durable/repo

export HF_HOME=/workspace/.cache/huggingface/
export HF_HUB_ENABLE_HF_TRANSFER=1

SCHEDULES="bo-external hlmean"
DATASET=strong_reject
COMMON="--model_name llama3-8b --device cuda:0 --beta 1.0 \
  --dataset $DATASET --max_new_tokens 512 --batch_size 16"

out_dir_for() {
  case "$1" in
    cle-a) echo ./completions/llama3-8b/cle-a ;;
    cle-p) echo ./completions/llama3-8b/projection ;;
    *) echo "unknown method $1" >&2; exit 1 ;;
  esac
}

echo "############ PHASE 1/2: generation (4 runs x 313 prompts) ############"
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
    F=completions_${DATASET}_FULL_layers11to18_beta1.0_margin${S}_seed0.json
    echo "=== [judge] $M / $S ==="
    python utils/eval_jailbreaks.py \
      --completions_path "$OUT/$F" \
      --methodologies strongreject harmbench \
      --evaluation_path "$OUT/evaluation/evaluation_${F#completions_}"
  done
done
echo "############ ALL DONE ############"
