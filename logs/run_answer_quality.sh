#!/bin/bash
set -e
cd /workspace/lem-durable/repo
P=completions/llama3-8b/projection/evaluation
A=completions/llama3-8b/cle-a/evaluation
S=completions/llama3-8b/cle-p-star/evaluation
HB=harmbench_standard_FULL_layers11to18_beta1.0
python experiments/answer_quality.py --batch_size 6 --out harmbench_answer_quality.json --runs \
  "CLE-P paper=$P/evaluation_${HB}_marginvecb9333bab29ae_seed0.json" \
  "CLE-P hlmean=$P/evaluation_${HB}_marginvece7a04fe44003_seed0.json" \
  "CLE-A paper=$A/evaluation_${HB}_marginvecb9333bab29ae_seed0.json" \
  "CLE-A hlmean=$A/evaluation_${HB}_marginvece7a04fe44003_seed0.json" \
  "CLE-P* relu=$S/evaluation_${HB}_marginvece7a04fe44003_seed0_gaterelu.json" \
  "CLE-P* c0=$S/evaluation_${HB}_marginvece7a04fe44003_seed0_gate0.json" \
  "CLE-P* c0.61=$S/evaluation_${HB}_marginvece7a04fe44003_seed0_gate0.61.json"
echo "=== ANSWER QUALITY DONE ==="
