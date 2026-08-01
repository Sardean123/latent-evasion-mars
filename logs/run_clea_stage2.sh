#!/bin/bash
# Stage 2 of the CLE-A replication of the CLE-P experiments (single GPU, strictly serial).
#   1. HarmBench judge on the two steered completion sets
#   2. TruthfulQA MC1/MC2 under the CLE-A intervention
#   3. TruthfulQA free-form generations (paper + hlmean)
#   4. AxBench-style fluency judge on those generations
set -e
cd /workspace/lem-durable/repo

CLEA=./completions/llama3-8b/cle-a
PAPER=completions_harmbench_standard_FULL_layers11to18_beta1.0_marginvecb9333bab29ae_seed0.json
HLMEAN=completions_harmbench_standard_FULL_layers11to18_beta1.0_marginvece7a04fe44003_seed0.json

echo "=== [1/4] HarmBench judge: paper (Fig 7b) ==="
python utils/eval_jailbreaks.py \
  --completions_path $CLEA/$PAPER \
  --methodologies harmbench \
  --evaluation_path $CLEA/evaluation/evaluation_${PAPER#completions_}

echo "=== [1/4] HarmBench judge: hlmean ==="
python utils/eval_jailbreaks.py \
  --completions_path $CLEA/$HLMEAN \
  --methodologies harmbench \
  --evaluation_path $CLEA/evaluation/evaluation_${HLMEAN#completions_}
echo "=== JUDGE DONE ==="

echo "=== [2/4] TruthfulQA MC under CLE-A ==="
python experiments/truthfulqa_mc.py --model_name llama3-8b --device cuda:0 \
  --layers 11-18 --method clea
echo "=== MC DONE ==="

echo "=== [3/4] TruthfulQA free-form generations ==="
COMMON="--model_name llama3-8b --device cuda:0 --layers 11-18 --beta 1.0 \
  --dataset truthfulqa --max_new_tokens 512 --batch_size 16 --out_dir $CLEA"
python cle-a.py $COMMON --layer_margins 1.2,2.0,1.8,1.8,2.0,0.9,1.2
python cle-a.py $COMMON --layer_margins 1.08,1.08,1.1,1.11,1.12,1.14,1.14
echo "=== TQA GENERATION DONE ==="

echo "=== [4/4] Fluency judge ==="
python experiments/truthfulqa_fluency.py --method clea
echo "=== STAGE 2 DONE ==="
