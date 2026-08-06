#!/bin/bash
# CLE-P* gate sweep, part 2: the region between -m and 0, where ASR falls 88.5% -> 57.0%.
# Endpoints already measured: -1m/relu = 88.5% ASR / -7.5 MC2 ; 0 = 57.0% / -1.3.
set -e
cd /workspace/lem-durable/repo
while pgrep -f "run_answer_quality.sh" > /dev/null; do sleep 30; done
echo "=== GPU free, starting CLE-P* sweep 2 ==="

SCHEDULE=hlmean          # resolved from config/margins.json; also supplies the layer window
STAR=./completions/llama3-8b/cle-p-star
GATES="-0.75m -0.5m -0.25m"

for C in $GATES; do
  echo "=== [gen] CLE-P* gate_c=$C ==="
  python cle-p-star.py --model_name llama3-8b --device cuda:0 \
    --beta 1.0 --margin_schedule $SCHEDULE --gate_c="$C" \
    --dataset harmbench_standard --max_new_tokens 512 --batch_size 16 --out_dir $STAR
done
echo "=== SWEEP2 GENERATION DONE ==="

for C in $GATES; do
  TAG="gate$(echo $C | sed 's/-/neg/')"
  F=completions_harmbench_standard_FULL_layers11to18_beta1.0_margin${SCHEDULE}_seed0_${TAG}.json
  echo "=== [judge] CLE-P* gate_c=$C  ($F) ==="
  python utils/eval_jailbreaks.py --completions_path $STAR/$F --methodologies harmbench \
    --evaluation_path $STAR/evaluation/evaluation_${F#completions_}
done
echo "=== SWEEP2 JUDGE DONE ==="

for C in $GATES; do
  echo "=== [mc] CLE-P* gate_c=$C ==="
  python experiments/truthfulqa_mc.py --model_name llama3-8b --device cuda:0 \
    --layers 11-18 --method clepstar --gate_c="$C"
done
echo "=== SWEEP2 DONE ==="
