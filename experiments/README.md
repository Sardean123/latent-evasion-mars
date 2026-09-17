# Experiments

Scripts are grouped by sub-experiment. Run each from the **repo root**
(`python experiments/<folder>/<script>.py`); they resolve data/results paths relative to it.

## `refusal_direction/` — is the probe the direction the model uses to refuse?
Gradient of `logP(refusal)` w.r.t. activations vs. the probe weight `w`, and multi-step
activation descent. Headline finding: the probe is a *detection* axis (≈ diff-in-means, ~0.85),
not the *control* axis (`cos(∇, w) ≈ 0` from single gradient through a 100-step descent).
- `refusal_gradient_alignment.py` — per-layer `cos(∇logP(refusal), w)` vs `cos(diff-in-means, w)`; saves gradient vectors.
- `pool_consensus.py` — pool the saved vectors → `cos(mean gradient, w)`.
- `refusal_gradient_walkthrough.py` — print-everything single-prompt walkthrough + finite-difference check.
- `refusal_activation_descent.py` — optimise a per-layer δ to flip the refuse-vs-comply log-odds.
- `descent_step_sweep.py` — how the δ direction evolves at 20/50/100 steps.
- `descent_generate.py` — generate with the optimised δ, to read the steered outputs.
- `plot_refusal_alignment.py`, `plot_refusal_gradient_lines.py` — figures.

## `judges/` — StrongREJECT vs. HarmBench judge analysis (the reward-hacking thread)
- `cross_judge_disagreement.py` — where the two judges disagree, prompts held constant.
- `low_score_composition.py` — what *low* StrongREJECT scores are made of (refusal / broken / subverted).
- `score_composition.py` — full 5-way agreement/disagreement per method (AGREE_pos/neg, HB>SR, SR>HB).
- `strongreject_api_batch.py` — grade the native 313 with the rubric judge via the OpenAI Batch API.
- `plot_judge_composition.py` — grouped bar charts. `dump_txt_views.py` — readable `.txt` dumps.

## `coherence/` — capability retention under steering
- `truthfulqa_mc.py` (MC1, the paper-comparable metric), `truthfulqa_fluency.py`.

## `probe_geometry/` — probe geometry and steering internals
- `probe_score_distributions.py`, `probe_score_over_generation.py`, `genpos_training_bands.py`,
  `plot_over_generation_compare.py`, `plot_over_generation_perpos.py`, `steering_magnitude.py`,
  `gradient_check.py` (CLE *projection* gradients — which (param, loss) pairs carry signal).

## `cle_core/` — CLE runs, completions I/O, headline plots, plumbing
- `baseline_generate.py`, `dump_completions.py`, `compare_completions.py`, `view_completions.py`,
  `answer_quality.py`, `consolidate_results.py`, `harmless_mean_schedule.py`,
  `plot_harmbench_asr.py`, `plot_clep_tradeoff.py`.

## `agentharm/` — the AgentHarm / Inspect track (M0–M4)
A separate track: does the single-turn refusal probe hold when the model runs as a tool-using
agent? See `agentharm/README.md`. Scripts: `m0a_hidden_states.py`, `m0b_dataset.py`, `mini_run.py`,
`diagnose_separation.py`, `multistep_dissociation.py`, `full_run_graded.py`, `check_logs.py`.

## `results/`
Committed numeric outputs and figures. Harmful text dumps under `results/model_outputs/` are
gitignored (regenerable). Cross-cutting notes live at the `experiments/` root
(`agentharm-refusal-tracker.md`, `dgx-setup.md`, `optuna-stage2-notes.md`).
