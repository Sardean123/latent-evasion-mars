# TruthfulQA MC under CLE-P steering — coherence / capability retention

**Question:** does CLE-P steering damage the model on *benign* inputs, and does the BO-free
`hlmean` margin schedule preserve more capability than the paper's BO-optimized margins?

**Metric:** TruthfulQA MC1/MC2 (answer log-probs, judge-free) computed with the CLE-P
`projection_hook` active during scoring, so the *steered* model is measured. MC1 = best-correct
answer gets the highest log-prob; MC2 = normalized probability mass on all correct answers.
Raw summed token log-probs (matches lm-eval `truthfulqa_mc`). n = 790, window layers 11–18,
probes `train_svm`. Computed by `experiments/truthfulqa_mc.py`.

If steering makes the model incoherent, its ability to prefer true over false answers falls
below the unsteered baseline. This is a coherence/capability-retention proxy — a separate
free-form generation + fluency-judge (AxBench-style) is planned on `--dataset truthfulqa`.

## Result

| condition | margin mean | MC1 | ΔMC1 | MC2 | ΔMC2 |
| --- | --- | --- | --- | --- | --- |
| baseline (unsteered) | — | 38.0% | — | 58.7% | — |
| paper (BO) | 1.56 | 29.6% | −8.4 | 46.3% | −12.4 |
| **hlmean** | 1.11 | 33.3% | **−4.7** | 51.3% | **−7.4** |

Both steering schedules cost accuracy, but `hlmean` loses ~40–45% less than `paper` on both
metrics — it retains roughly half of what the BO margins give up.

## Combined with the HarmBench ASR result

| margins | HarmBench ASR (↑ better) | TruthfulQA MC2 Δ vs baseline (↑ better) |
| --- | --- | --- |
| paper (BO, mean 1.56) | 88.0% | −12.4 |
| **hlmean (mean 1.11)** | **88.5%** | **−7.4** |

`hlmean` matches BO on jailbreak effectiveness while preserving substantially more coherence —
the payoff of steering ~30% less. See `../harmbench_clep_margins/` for the ASR side.

Data: `truthfulqa_mc_llama3-8b_layers11to18.json` (overall + per-category MC1/MC2 per condition).
