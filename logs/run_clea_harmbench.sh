#!/bin/bash
set -e
cd /workspace/lem-durable/repo
COMMON="--model_name llama3-8b --device cuda:0 --layers 11-18 --beta 1.0 \
  --dataset harmbench_standard --max_new_tokens 512 --batch_size 16 \
  --out_dir ./completions/llama3-8b/cle-a"
echo "=== [1/2] CLE-A paper (Fig 7b) margins ==="
python cle-a.py $COMMON --layer_margins 1.2,2.0,1.8,1.8,2.0,0.9,1.2
echo "=== [2/2] CLE-A hlmean margins ==="
python cle-a.py $COMMON --layer_margins 1.08,1.08,1.1,1.11,1.12,1.14,1.14
echo "=== HARMBENCH GENERATION DONE ==="
