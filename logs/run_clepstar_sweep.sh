#!/bin/bash
# CLE-P* gate sweep: hlmean margins, three gate settings, on one GPU (strictly serial).
#   c = relu (-m_l)  most inclusive, never back-steers
#   c = 0            the probe decision boundary
#   c = 0.61         mean margin 1.11 - 0.5, most selective
# 1. HarmBench standard generations (n=200) for each gate
# 2. HarmBench judge for each gate
# 3. TruthfulQA MC under CLE-P* for each gate (reports gate rate alongside MC1/MC2)
# Free-form TruthfulQA + fluency judge are deliberately NOT here -- they cost ~2.5h per gate,
# so run them only for whichever gate the ASR/MC results single out.
set -e
cd /workspace/lem-durable/repo

# wait for the CLE-A stage-2 chain to release the GPU
while pgrep -f "run_clea_stage2.sh" > /dev/null; do sleep 30; done
echo "=== GPU free, starting CLE-P* sweep ==="

SCHEDULE=hlmean          # resolved from config/margins.json; also supplies the layer window
STAR=./completions/llama3-8b/cle-p-star
GATES="relu 0 0.61"

for C in $GATES; do
  echo "=== [gen] CLE-P* gate_c=$C ==="
  python cle-p-star.py --model_name llama3-8b --device cuda:0 \
    --beta 1.0 --margin_schedule $SCHEDULE --gate_c "$C" \
    --dataset harmbench_standard --max_new_tokens 512 --batch_size 16 \
    --out_dir $STAR
done
echo "=== CLEPSTAR GENERATION DONE ==="

for C in $GATES; do
  case "$C" in relu) TAG=relu ;; *) TAG=$C ;; esac
  F=completions_harmbench_standard_FULL_layers11to18_beta1.0_margin${SCHEDULE}_seed0_gate${TAG}.json
  echo "=== [judge] CLE-P* gate_c=$C ==="
  python utils/eval_jailbreaks.py \
    --completions_path $STAR/$F \
    --methodologies harmbench \
    --evaluation_path $STAR/evaluation/evaluation_${F#completions_}
done
echo "=== CLEPSTAR JUDGE DONE ==="

for C in $GATES; do
  echo "=== [mc] CLE-P* gate_c=$C ==="
  python experiments/truthfulqa_mc.py --model_name llama3-8b --device cuda:0 \
    --layers 11-18 --method clepstar --gate_c "$C"
done
echo "=== CLEPSTAR SWEEP DONE ==="
