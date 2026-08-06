#!/bin/bash
# Stage 2 of the CLE-A replication of the CLE-P experiments (single GPU, strictly serial).
#   1. HarmBench judge on the two steered completion sets
#   2. TruthfulQA MC1/MC2 under the CLE-A intervention
#   3. TruthfulQA free-form generations (paper + hlmean)
#   4. AxBench-style fluency judge on those generations
set -e
cd /workspace/lem-durable/repo

CLEA=./completions/llama3-8b/cle-a
# Filenames follow the run tag produced by --margin_schedule (name, not SHA1 digest). The
# pre-registry artifacts on disk use marginvecb9333bab29ae / marginvece7a04fe44003 instead, so
# this script now regenerates rather than reuses them.
PAPER=completions_harmbench_standard_FULL_layers11to18_beta1.0_marginpaper-fig7b_seed0.json
HLMEAN=completions_harmbench_standard_FULL_layers11to18_beta1.0_marginhlmean_seed0.json

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
COMMON="--model_name llama3-8b --device cuda:0 --beta 1.0 \
  --dataset truthfulqa --max_new_tokens 512 --batch_size 16 --out_dir $CLEA"
python cle-a.py $COMMON --margin_schedule paper-fig7b
python cle-a.py $COMMON --margin_schedule hlmean
echo "=== TQA GENERATION DONE ==="

echo "=== [4/4] Fluency judge ==="
python experiments/truthfulqa_fluency.py --method clea
echo "=== STAGE 2 DONE ==="
