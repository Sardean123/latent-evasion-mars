#!/bin/bash
set -e
cd /workspace/lem-durable/repo
S=completions/llama3-8b/cle-p-star/evaluation
HB=harmbench_standard_FULL_layers11to18_beta1.0_marginvece7a04fe44003_seed0
python experiments/answer_quality.py --batch_size 6 --out clepstar_gate_sweep_answer_quality.json --runs \
  "CLE-P* -0.75m=$S/evaluation_${HB}_gateneg0.75m.json" \
  "CLE-P* -0.5m=$S/evaluation_${HB}_gateneg0.5m.json" \
  "CLE-P* -0.25m=$S/evaluation_${HB}_gateneg0.25m.json"
echo "=== AQ SWEEP2 DONE ==="
