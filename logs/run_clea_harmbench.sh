#!/bin/bash
set -e
cd /workspace/lem-durable/repo
# Margins come from config/margins.json by name; the window ships with the schedule.
COMMON="--model_name llama3-8b --device cuda:0 --beta 1.0 \
  --dataset harmbench_standard --max_new_tokens 512 --batch_size 16 \
  --out_dir ./completions/llama3-8b/cle-a"
echo "=== [1/2] CLE-A paper (Fig 7b) margins ==="
python cle-a.py $COMMON --margin_schedule paper-fig7b
echo "=== [2/2] CLE-A hlmean margins ==="
python cle-a.py $COMMON --margin_schedule hlmean
echo "=== HARMBENCH GENERATION DONE ==="
