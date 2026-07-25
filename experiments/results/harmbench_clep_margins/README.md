# CLE-P: BO-optimized vs harmless-mean margins on HarmBench standard

**Question:** does the BO-free harmless-mean margin schedule (`hlmean`) match the CLE paper's
Bayesian-optimized per-layer margins (`paper`, Fig 7b) for jailbreaking?

**Setup (all held constant except the margins):** CLE-P, model `llama3-8b`, probes `train_svm`,
window layers 11–18, β=1, dataset `harmbench_standard` (n=200), greedy `max_new_tokens=512`,
judged by `cais/HarmBench-Llama-2-13b-cls` (HarmBench's official standard classifier).

**Contamination check:** the probes were trained on `harmful_train_filtered` / `harmless_train_filtered`
(Arditi split), which has **zero** overlap with `harmbench_standard`, so evaluating on the standard
set is not evaluating on training points. `hlmean` margins come from the training-harmless
distribution; `paper` margins are fixed literals from the CLE paper. No leakage.

## Result

| run | per-layer margins (L11..L17) | mean | HarmBench ASR |
| --- | --- | --- | --- |
| baseline (unsteered, β=0) | — | — | 0.5% (1/200) |
| **paper (BO)** | `1.2, 2.0, 1.8, 1.8, 2.0, 0.9, 1.2` | 1.56 | **88.0%** (176/200) |
| **hlmean** | `1.08, 1.08, 1.1, 1.11, 1.12, 1.14, 1.14` | 1.11 | **88.5%** (177/200) |

The BO-free harmless-mean schedule is statistically indistinguishable from the BO-optimized
schedule (one-prompt difference) while steering ~30% less (mean margin 1.11 vs 1.56).

- `asr_summary.json` — overall + per-category ASR for all three runs.
- `harmbench_asr_clep_margins.png` — bar chart (overall + per-category), from `experiments/plot_harmbench_asr.py`.
- Raw completions + per-run evaluation JSONs: `completions/llama3-8b/projection/` (git-ignored; see its `RUN_MANIFEST.md`).

Coherence/4-gram-repetition comparison: TODO (lighter hlmean steering is expected to degrade fluency less).
