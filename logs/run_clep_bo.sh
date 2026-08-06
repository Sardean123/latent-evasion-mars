#!/bin/bash
# CLE-P with 'bo-external-clep' -- the first CLE-P run against a CLE-P-TUNED BO schedule.
# Every prior CLE-P-vs-BO comparison used a CLE-A-tuned schedule, so this is the cell that
# actually tests "does hlmean match BO under CLE-P".
#
# Three measurements, matching what every other cell in RESULTS_SUMMARY.md carries:
#   1. HarmBench standard n=200  -> ASR (standard protocol) + StrongREJECT judge
#   2. StrongREJECT native n=313 -> SR-313 benchmark + off-distribution ASR
#   3. TruthfulQA MC1 n=790      -> coherence
set -e
cd /workspace/lem-durable/repo
export HF_HOME=/workspace/.cache/huggingface/
export HF_HUB_ENABLE_HF_TRANSFER=1

S=bo-external-clep
OUT=./completions/llama3-8b/projection
COMMON="--model_name llama3-8b --device cuda:0 --beta 1.0 --max_new_tokens 512 --batch_size 16"

for DS in harmbench_standard strong_reject; do
  echo "############ [gen] cle-p / $S / $DS ############"
  python cle-p.py $COMMON --dataset $DS --margin_schedule "$S" --out_dir "$OUT"
done

for DS in harmbench_standard strong_reject; do
  F=completions_${DS}_FULL_layers11to18_beta1.0_margin${S}_seed0.json
  echo "############ [judge] $DS ############"
  python utils/eval_jailbreaks.py --completions_path "$OUT/$F" \
    --methodologies strongreject harmbench \
    --evaluation_path "$OUT/evaluation/evaluation_${F#completions_}"
done

echo "############ [coherence] TruthfulQA MC1 ############"
python experiments/truthfulqa_mc.py --model_name llama3-8b --device cuda:0 \
  --layers 11-18 --method clep --schedules "$S"

echo "############ ALL DONE ############"
