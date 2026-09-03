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

Alternative that also works and is faster (used on the 2026-08-06 restart): **keep the base
image's torch 2.8.0+cu128** and install everything else against it, which avoids the ~3 GB torch
download and the torchvision/torchaudio uninstall entirely, since the shipped torchvision matches
2.8. Add `peft` and `datasets` if you are running the StrongREJECT judge, and `matplotlib` for
the plot scripts:

```bash
pip install --no-cache-dir transformers==4.57.6 tokenizers==0.22.2 accelerate==1.2.0 \
  peft==0.18.1 safetensors==0.7.0 sentencepiece==0.2.1 scikit-learn==1.8.0 \
  huggingface_hub==0.36.2 datasets==4.8.4 tqdm matplotlib openai scipy
```

`openai` is needed only for the `strongreject_api` rubric judge (`utils/eval_jailbreaks.py`,
`experiments/strongreject_api_batch.py`); `scipy` for the paired significance tests.

Environment variables the runs rely on (confirm they survive the restart; re-export if not):

```bash
export HF_HOME=/workspace/.cache/huggingface/     # model cache lives on the volume
export HF_HUB_ENABLE_HF_TRANSFER=1
```

The OpenAI API judge reads `OPENAI_API_KEY` from the repo-root `.env` (gitignored, loaded by
`_load_dotenv()`). That file lives on the volume, so it survives a pod stop — but it is NOT in
git, so if the new pod does not re-attach this exact network volume it is gone and must be
re-added: `echo 'OPENAI_API_KEY=sk-...' >> /workspace/lem-durable/repo/.env`.

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

Established 2026-08-06:

- **Margins are now named, not pasted.** `config/margins.json` is the registry; scripts take
  `--margin_schedule <name>` and tag files `margin<name>` instead of a SHA1 digest. Each entry
  carries provenance and `optimized_for`. `legacy_digest` maps a schedule to its old digest so
  pre-registry filenames still resolve. Tests: `tests/test_registry.py` (CPU-only, no GPU stack).
- **StrongREJECT fine-tuned evaluator wired in** as a `strongreject` methodology in
  `utils/eval_jailbreaks.py`. Continuous 0-1, one forward pass, expected value of the 1..5 token
  distribution. Writes `strongreject_score`, never `is_jailbreak_*` and no ASR (the benchmark
  defines no threshold), so it composes with the HarmBench judges over the same completions.
  Tests: `tests/test_strongreject_integration.py` (needs GPU), `tests/smoke_strongreject.py`.
- **`bo-external` was optimized for CLE-A — confirmed by the provider on 2026-08-06.** The
  registry's `optimized_for` is now `cle-a`, not `unknown`. The empirical signal had already
  pointed there: it helps CLE-A (91.0% vs 76.5% ASR) and hurts CLE-P (84.5 vs 88.0 ASR, 0.586 vs
  0.678 StrongREJECT), with both judges agreeing on the sign in both cases — an interaction, not
  a level shift. **Consequence: the CLE-P/bo-external row is method-mismatched** and is not a
  fair CLE-P result; the clean CLE-P comparison is paper-fig7b (also CLE-A-tuned, so also
  mismatched) vs hlmean. Single seed, no intervals; the CLE-P ASR gap alone is inside n=200 noise.
- **The two judges reorder the table.** ASR ranks CLE-A/bo-external first, StrongREJECT ranks
  CLE-P/hlmean first, and CLE-A/hlmean beats CLE-P/bo-external on StrongREJECT despite 8 points
  less ASR. Same soft-breakage effect `answer_quality.py` found, now confirmed by an independent
  published judge rather than a self-judge.
- **MC1 is the paper-comparable TruthfulQA metric, not MC2.** The paper (Tab. 4/5, App. H)
  reports a single TruthfulQA accuracy; its LLaMA3-8B baseline of 37.08 matches MC1 here (~38%)
  and not MC2 (~59%). `truthfulqa_mc.py` now headlines MC1 as unsteered-vs-steered; MC2 is still
  computed and saved. Note the paper also evaluates MMLU and ARC, which this repo does not, so
  the coherence axis here is narrower than the paper's.

Established 2026-09-03:

- **Cross-judge disagreement, prompts held constant** (`experiments/cross_judge_disagreement.py`,
  results in `experiments/results/judge_disagreement/`). Both prompt sets scored by their native
  AND foreign judge over the same completions. Agreement is only fair-to-moderate (Cohen kappa
  0.32-0.45). Two mechanisms on different harm types: HarmBench OVER-counts broken attack code
  (`cybercrime_intrusion`, Quadrant A), the StrongREJECT rubric OVER-counts fluent prose and
  UNDER-credits broken code (Quadrant B, prose-heavy, dominates the 313). `.txt` views:
  `dump_txt_views.py` -> `model_outputs/strongreject_*.txt` (gitignored) and
  `judge_disagreement/judge_disagreement.txt`.
- **Refusal-gradient vs probe-direction alignment** (`experiments/refusal_gradient_alignment.py`,
  `refusal_gradient_walkthrough.py`, `plot_refusal_alignment.py`; results in
  `experiments/results/refusal_gradient/`). The probe weight `w` CLE steers along tracks
  difference-in-means (~0.85) but is near-ORTHOGONAL to the gradient of log P(refusal) w.r.t. the
  block-l residual stream at the last prompt token (cos ~0.01-0.03, n=159 harmful). Gradient taken
  w.r.t. a non-leaf activation; the loaded model is frozen so a forward builds no graph -- an
  input-embedding grad-enable hook fixes it without unfreezing weights. A finite-difference check
  shows moving along `-w` (CLE's jailbreak direction) drops refusal ~15-40x less per unit step than
  moving along `-grad`; CLE compensates with large BO margins x 7 layers x every decode token. The
  margin->step conversion: step = (raw_score + m)/||w|| (Euclidean move distance). Harmful vs
  harmless (log-odds target grad[logP("I cannot") - logP("Sure, here")]): the refusal-inducing
  direction aligns with the harm axis on HARMLESS prompts (+0.06..+0.11) but not on harmful
  (saturated regime, ~0 to -0.10) -- w is the refusal lever near the boundary, not once harm is
  detected. Section in RESULTS_SUMMARY.md ("Cross-judge disagreement"); the refusal-gradient
  section is not yet written into RESULTS_SUMMARY.

## 4. Immediate next steps (in the order I would do them)

0. **Paired McNemar over the shared 200 prompts** for the bo-external-vs-hlmean contrasts, to
   put an interval on the interaction claim above. The machinery already exists
   (`## Significance` in RESULTS_SUMMARY.md) and single-seed point estimates are all we have.
   *(Done: `truthfulqa_mc.py --schedules <registry names>` now takes any schedule, and the
   CLE-A/bo-external coherence cell is filled. CLE-P/bo-external is still blank, but that row is
   now known-method-mismatched, so it is a curiosity rather than a gap.)*
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

- **The volume has a ~47 GB quota and `df` will not show it.** `df` reports the underlying
  MooseFS cluster (240 TB free), so a write failing with `OSError: [Errno 122] Disk quota
  exceeded` looks impossible. `du -sh /workspace` is the number that matters. The HF cache is
  ~44 GB of it (HarmBench-13B 25 GB, Llama-3-8B 15 GB, gemma-2b 4.7 GB) and none of it is
  duplicated — blob sums match the shard counts exactly, so there is nothing to reclaim there.
- **Results do not travel with `git push`.** `completions/` is a symlink to
  `repo-backup/completions`, outside the repo, and git tracks zero entries under it. Completions
  and evaluation JSONs live only on the volume. `experiments/results/` IS tracked and does push.
- **`experiments/harmless_mean_schedule.py` is broken on its defaults.** `--reps_dir` falls back
  to `dataset/representations/llama3-8b/train_svm`, which holds the `svm_layer*.pt` probes but no
  `HFx_train.pt`/`HLx_train.pt`. Those tensors only exist at the doubly-nested
  `dataset/representations/representations/...` path and in `repo-backup/`. Regenerating hlmean
  needs an explicit `--reps_dir`; the registry's `generator` field for hlmean records the command
  as documented, not as runnable.
- **StrongREJECT's prompt template ends in a trailing space that is load-bearing.** `"### Answer: "`
  — the space is its own token and the score is read at the position after it. Trim it and every
  score silently becomes meaningless. `_load_strongreject_template()` asserts on this, and
  `captured_mass` (digit probability over the FULL vocabulary, ~0.998 in practice) is the runtime
  tripwire. Upstream cannot catch this: it softmaxes over only the five digit logits, so its
  probabilities sum to 1 by construction.
- **StrongREJECT upstream injects a literal `<bos>` into every response.** It truncates by
  tokenise-then-decode without `skip_special_tokens`, so the judge prompt reads
  `AI model response: <bos>...`. Reproduced by default (`reference_bos_quirk=True`) so numbers
  stay comparable with published ones. On four hand-built cases it barely moved near-zero scores
  but pushed a compliant response 0.72 -> 0.52, i.e. it may systematically deflate exactly the
  successful jailbreaks the benchmark exists to measure. Unmeasured at scale — one extra judge
  pass over existing completions would settle it.
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
