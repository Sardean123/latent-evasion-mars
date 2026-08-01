# Resume notes — CLE-A replication, CLE-P*, and the judging harness

Written 2026-07-28, at the point where the pod was stopped. Everything below is reconstructable
from the repo, but this file exists so it does not have to be.

## 1. Rebuilding the environment (this is the ONLY thing a pod stop destroys)

`/workspace` is the RunPod network volume and survives. `/` is the container overlay and does
not — which means **the Python environment is gone on restart** and nothing else is.

```bash
export PIP_CACHE_DIR=/workspace/.cache/pip
pip install --root-user-action=ignore \
  torch==2.10.0 transformers==4.57.6 tokenizers==0.22.2 accelerate==1.2.0 \
  huggingface_hub==0.36.2 safetensors==0.7.0 numpy==2.2.6 scikit-learn==1.8.0 \
  scipy==1.17.1 matplotlib==3.10.8 tqdm==4.67.3 sentencepiece==0.2.1 protobuf==6.33.6 \
  hf_transfer
# REQUIRED: the base image ships torchvision/torchaudio built for torch 2.8. They are not
# importable against torch 2.10 and transformers imports torchvision, so `from
# transformers.models.llama import modeling_llama` dies with a misleading
# "Could not find LlamaForCausalLM" until these are removed. Nothing here uses them.
pip uninstall -y torchvision torchaudio
```

`requirements.txt` additionally pins vllm and optuna; neither is imported by anything used in
these experiments (only `optuna_search.py` needs optuna), so both are skipped above.

Environment variables the runs rely on (confirm they survive the restart; re-export if not):

```bash
export HF_HOME=/workspace/.cache/huggingface/     # model cache lives on the volume
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Models already cached on the volume (~40 GB, no re-download needed): `meta-llama/Meta-Llama-3-8B-Instruct`,
`cais/HarmBench-Llama-2-13b-cls`. `HF_TOKEN` is NOT set and is not needed while the cache is warm.

Sanity check after install:

```bash
python -c "import torch,transformers; from transformers.models.llama import modeling_llama; \
  print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

## 2. Where the data lives

| what | path | in git? |
| --- | --- | --- |
| code + result JSONs | `experiments/results/**` | yes |
| raw completions + judged evaluations | `completions/` -> `/workspace/lem-durable/repo-backup/completions/` | **no** (`.gitignore`), volume only |
| plain-text harmful completion dumps | `experiments/results/model_outputs/` | **no** (`.gitignore`), volume only |
| run scripts | `logs/*.sh` | yes |
| run stdout logs | `logs/*.log` | **no** (`.gitignore`) |

The two git-excluded directories are deliberate (harmful text stays out of git) and are
regenerable: completions from the `logs/run_*.sh` scripts, dumps from
`experiments/compare_completions.py`. Everything numeric is committed.

## 3. State of the experiments

All headline numbers are in `experiments/results/RESULTS_SUMMARY.md`. Model llama3-8b, window
layers 11-18, beta=1, HarmBench standard n=200, TruthfulQA MC n=790.

Established this session:

- **CLE-A replication.** hlmean margins do NOT substitute for BO margins under CLE-A: 76.5% vs
  89.5% ASR, paired McNemar p=2e-5 (32 vs 6 discordant). Under CLE-P they are indistinguishable
  (88.5 vs 88.0, 12 vs 13 discordant). Mechanism: CLE-P re-projects every position every decode
  step so a smaller margin costs nothing; CLE-A applies one prompt-derived delta whose magnitude
  is proportional to `(score + m)`, and nothing re-asserts it during decoding.
- **Figure 7 of the paper is the CLE-A schedule, not CLE-P.** Panel (a) spans 2.5-4.0 and
  Table 3 gives LLaMA2-7B CLE-A m=3.1 vs CLE-P m=1.6; a +/-0.1 walk seeded at 1.6 cannot reach
  that range. So the CLE-P "paper" baseline used earlier is a CLE-A-derived schedule. An
  untested follow-up is CLE-P at the Table 3 shared m=1.3.
- **CLE-P\* (new, `cle-p-star.py`).** Gated projection, only steer where `w·h + b > c`. The
  back-steering CLE-P does is real but rare: `c=-1m` (never steer backwards) fires on 93-100% of
  positions and reproduces ungated CLE-P exactly. Raising c mostly switches off DOWNSTREAM
  layers (L17 fire rate 98.8% -> 12.0% -> 1.2%), so c behaves as a soft, activation-dependent
  replacement for the paper's binary layer window lambda_l rather than as a strength knob.
- **Soft breakage is 8-18% of nominal ASR** (`experiments/answer_quality.py`). HarmBench scores
  *attempt* by its own rules; the responsiveness judge scores *delivery*. This reorders the top
  of the table (CLE-A paper 89.5% ASR -> 78.0% effective) though the top four are not
  significantly separated.
- **The old fluency numbers are weakly grounded.** The self-judge ignored the 0/1/2 rubric,
  emitting 1.5 for 86% of items and never 0 or 2, so the published "% fluent (2) = 92.8%" and
  "% degenerate = 0.8%" columns are `round(1.5)->2` / `round(0.5)->0` artifacts. Means are
  valid; only CLE-P paper's -0.051 is significant (p<1e-4). Both judges now use constrained
  scoring (`utils/llm_judge.py`) which makes off-rubric scores impossible.

## 4. Immediate next steps (in the order I would do them)

1. **Re-run fluency under constrained scoring** so those numbers are trustworthy, and fix the
   distribution columns in `experiments/results/truthfulqa_fluency/README.md`:
   ```bash
   python experiments/truthfulqa_fluency.py --method clep    # ~1.5h
   python experiments/truthfulqa_fluency.py --method clea
   ```
2. **Free-form TruthfulQA for the chosen CLE-P\* gate** (generation ~30 min + judge ~1.5h each,
   so pick one gate rather than sweeping):
   ```bash
   python cle-p-star.py --model_name llama3-8b --device cuda:0 --layers 11-18 --beta 1.0 \
     --layer_margins 1.08,1.08,1.1,1.11,1.12,1.14,1.14 --gate_c=-0.5m \
     --dataset truthfulqa --max_new_tokens 512 --batch_size 16 \
     --out_dir ./completions/llama3-8b/cle-p-star
   python experiments/truthfulqa_fluency.py --method clepstar --gate_c=-0.5m
   ```
3. **The two open threads from the very first review**, neither started:
   - *Cross-model ablation.* Only llama3-8b has trained probes; `dataset/splits/` has 15 models.
     `|mean_HL|` pins to ~1.1 at every separable layer because that is the LinearSVC margin unit,
     so hlmean will emit ~1.1 for any model, while Table 3's BO margins span 0.6-9.8. Testing
     the extremes (qwen35-9b BO 5.5, gpt-oss-20b BO 9.8) either shows BO massively overshoots or
     bounds the method. Probe training is minutes; models <=15B fit the A40.
   - *The layer window is still BO-derived.* 11-18 comes from the paper's Table 3, so "no BO" is
     not yet true end to end. CLE-P\*'s gate is a candidate replacement — worth testing whether
     a wide window plus a gate matches the tuned window.

## 5. Gotchas worth not rediscovering

- `--gate_c=-0.5m` needs the `=`; argparse reads a bare `-0.5m` as an option name.
- `--runs "label=path"` in `answer_quality.py` splits on the LAST `=`, because labels contain
  `=` (e.g. `CLE-P* c=0`).
- Constrained scoring must use single-token label encodings. In the Llama-3 vocabulary `" 0"` is
  `[220, 15]` and that leading space token is shared by every label, so scoring on it silently
  makes all labels identical. `utils/llm_judge.py` reports `captured_mass` to catch this class
  of bug; it should sit at ~0.99-1.00.
- Never read steered activations off `output_hidden_states` — in this transformers version
  `hidden_states[l+1]` is layer l's output *before* layer l's own forward hook. Record through
  hooks. (Pre-existing note, still true.)
