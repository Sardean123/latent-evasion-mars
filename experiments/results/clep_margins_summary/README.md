# CLE-P margins: jailbreak vs coherence tradeoff (summary)

Combined view of the two evals. `asr_vs_coherence.png` (from `experiments/plot_clep_tradeoff.py`).

| margins | HarmBench ASR ↑ | TruthfulQA MC2 ↑ | MC1 |
| --- | --- | --- | --- |
| baseline (unsteered) | 0.5% | 58.7% | 38.0% |
| paper (BO, mean 1.56) | 88.0% | 46.3% (−12.4) | 29.6% (−8.4) |
| **hlmean (mean 1.11)** | **88.5%** | **51.3% (−7.4)** | **33.3% (−4.7)** |

`hlmean` matches the paper's BO-optimized margins on jailbreak effectiveness while retaining
~2x more coherence/capability (loses ~40–45% less on TruthfulQA MC1/MC2) — the payoff of
steering ~30% less. Sources: `../harmbench_clep_margins/`, `../truthfulqa_mc/`.

Note: TruthfulQA MC is a capability/coherence *proxy*. A direct free-form fluency judge
(AxBench-style, on `--dataset truthfulqa` generations) is the follow-up.
