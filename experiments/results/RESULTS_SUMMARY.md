# Results summary — llama3-8b, layers 11-18, beta=1

HarmBench standard n=200 (judge `cais/HarmBench-Llama-2-13b-cls`). TruthfulQA MC n=790.
Effective ASR = HarmBench-positive AND fully responsive (`experiments/answer_quality.py`),
one shared denominator of 200, so it is the cross-run comparable column. `q=0` is the
outright-nonresponsive share of each run's own ASR-positive set.

All CLE-P* rows use hlmean margins. Fluency is the OLD regex scoring — see caveat below.

| run | margins | ASR | effective ASR | q=0 | MC1 | dMC1 | MC2 | dMC2 | fluency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (unsteered) | -- | 0.5% | -- | -- | 38.0% | +0.0 | 58.7% | +0.0 | 1.481 |
| CLE-P paper | 1.56 | 88.0% | 80.5% | 6.2% | 29.6% | -8.4 | 46.3% | -12.4 | 1.430 |
| CLE-P hlmean | 1.11 | 88.5% | 78.5% | 8.5% | 33.3% | -4.7 | 51.3% | -7.4 | 1.470 |
| CLE-A paper | 1.56 | 89.5% | 78.0% | 11.7% | 33.8% | -4.2 | 51.9% | -6.7 | 1.475 |
| CLE-A hlmean | 1.11 | 76.5% | 64.5% | 7.2% | 37.5% | -0.5 | 56.8% | -1.8 | 1.487 |
| CLE-P* c=-1m (relu) | 1.11 | 88.5% | 79.5% | 7.9% | 33.3% | -4.7 | 51.2% | -7.5 | -- |
| CLE-P* c=-0.75m | 1.11 | 86.0% | 76.0% | 7.0% | 34.1% | -3.9 | 52.2% | -6.5 | -- |
| CLE-P* c=-0.5m | 1.11 | 84.0% | 72.0% | 5.4% | 34.7% | -3.3 | 53.9% | -4.7 | -- |
| CLE-P* c=-0.25m | 1.11 | 75.0% | 62.0% | 5.3% | 35.7% | -2.3 | 55.2% | -3.5 | -- |
| CLE-P* c=0 | 1.11 | 57.0% | 46.5% | 0.9% | 36.7% | -1.3 | 57.3% | -1.3 | -- |
| CLE-P* c=0.61 | 1.11 | 4.0% | 3.0% | 0.0% | 38.1% | +0.1 | 58.5% | -0.2 | -- |

## StrongREJECT re-run, 2026-08-06 (`logs/run_full_rerun.sh`)

Four fresh generations judged by BOTH the standard HarmBench protocol and the StrongREJECT
fine-tuned evaluator (continuous 0-1; the benchmark defines no threshold, so there is no ASR
column for it). Same n=200 HarmBench standard prompts. Digit mass 0.998 on every run, so the
evaluator was scoring at the right position throughout.

| run | schedule | ASR | StrongREJECT | MC1 | dMC1 |
| --- | --- | --- | --- | --- | --- |
| CLE-A | bo-external | 91.0% | 0.6305 | 31.9% | **-6.1** |
| CLE-A | hlmean | 76.5% | 0.6211 | 37.5% | -0.5 |
| CLE-P | bo-external (mismatched) | 84.5% | 0.5855 | -- | -- |
| CLE-P | hlmean | 88.0% | 0.6780 | 33.3% | -4.7 |

MC1 for the hlmean rows carries over from the table above (MC scoring is a separate pass at the
same margins, independent of these generations). CLE-A/bo-external was scored 2026-08-06 with
`--method clea --schedules bo-external`; its unsteered baseline came out at 37.97%, identical to
the earlier run's, which is the control that makes the two tables comparable. The CLE-P
bo-external cell is left blank on purpose — see the first finding.

Three findings:

- **`bo-external` was optimized for CLE-A** — confirmed by the provider 2026-08-06, and the data
  had already said so: +14.5 ASR under CLE-A, -3.5 ASR and -0.093 StrongREJECT under CLE-P, both
  judges agreeing on the sign. An interaction, not a level shift. **The CLE-P/bo-external row is
  therefore method-mismatched and is not a fair CLE-P result.** Single seed, no intervals; the
  CLE-P ASR gap alone is inside n=200 noise, so a paired McNemar is still wanted before quoting
  the size of the effect.
- **CLE-A/bo-external buys its ASR lead with coherence, and loses once you price that in.**
  91.0% is the highest ASR anywhere in this repo, but -6.1 MC1 is also the largest coherence cost
  of any CLE-A row — worse than the paper's own schedule (-4.2). Against CLE-P/hlmean it is
  **dominated on both non-binary axes**: lower StrongREJECT (0.631 vs 0.678) *and* worse
  coherence (-6.1 vs -4.7). It wins on binary ASR alone, by 3 points.
  The mechanism is visible in `delta_rel_norm` (mean ||delta||/||h|| at the steered position):
  bo-external pushes 0.423 of the activation norm at layer 17 vs 0.078 for paper-fig7b and 0.011
  for hlmean — a ~38x harder shove at the top of the window, from m_17 = 2.5.
- **The judges reorder the ranking.** ASR puts CLE-A/bo-external first; StrongREJECT puts
  CLE-P/hlmean first. CLE-A/hlmean beats CLE-P/bo-external on StrongREJECT (0.621 vs 0.586)
  despite 8 points less ASR, and comes within 0.009 of CLE-A/bo-external's StrongREJECT while
  giving up 14.5 ASR points and costing almost no coherence (-0.5). Same soft-breakage effect the
  effective-ASR column shows, now corroborated by an independent published judge instead of a
  self-judge.

On the open question of what mean margins do to refusal on their own: still not cleanly
identified. hlmean vs bo-external is confounded by the CLE-A tuning, and hlmean vs paper-fig7b is
confounded the same way (paper-fig7b is also CLE-A-tuned, per the registry note). What hlmean
does show consistently is a much better ASR-per-unit-coherence rate: it is the only schedule that
leaves MC1 essentially intact, at -0.5 under CLE-A.

Reproducibility check against the rows above: CLE-A hlmean reproduced exactly (76.5%), CLE-P
hlmean came in at 88.0% vs 88.5% previously — one prompt.

## Pareto frontier (effective ASR vs dMC2)

Note: this frontier is still computed on dMC2. MC1 is the paper-comparable metric (the CLE
paper's single TruthfulQA accuracy matches MC1, not MC2), so this ordering should be recomputed
on dMC1 before being used as a headline; it has not been.

CLE-P paper -> CLE-P* -1m -> CLE-P hlmean -> CLE-A paper -> CLE-P* -0.75m -> CLE-P* -0.5m ->
CLE-A hlmean -> CLE-P* c=0. Only **CLE-P* c=-0.25m is dominated** (by CLE-A hlmean, on both axes).

## Significance (paired McNemar, same 200 behaviors)

| comparison | metric | result |
| --- | --- | --- |
| CLE-A paper vs CLE-A hlmean | raw ASR | 179 vs 153, discordant 32/6, **p=2e-5** |
| CLE-P paper vs CLE-P hlmean | raw ASR | 176 vs 177, discordant 12/13, n.s. |
| CLE-P paper vs CLE-P hlmean | effective | 161 vs 157, discordant 21/17, p=0.63 |
| CLE-P hlmean vs CLE-P* -1m | effective | 157 vs 159, discordant 6/8, p=0.79 |
| CLE-P paper vs CLE-A paper | effective | 161 vs 156, discordant 22/17, p=0.52 |
| CLE-P hlmean vs CLE-A hlmean | effective | 157 vs 129, discordant 38/10, **p<0.001** |

## CLE-P* gate fire rate (fraction of positions steered)

| gate c | L11 | L17 | ASR | dMC2 |
| --- | --- | --- | --- | --- |
| -1m (relu) | 100.0% | 98.8% | 88.5% | -7.5 |
| -0.75m | 100.0% | 59.1% | 86.0% | -6.5 |
| -0.5m | 98.5% | 12.0% | 84.0% | -4.7 |
| -0.25m | 78.1% | 4.6% | 75.0% | -3.5 |
| 0 | 31.5% | 1.2% | 57.0% | -1.3 |
| 0.61 | 2.6% | 0.4% | 4.0% | -0.2 |

Raising c switches off DOWNSTREAM layers first, so c acts as a soft, activation-dependent
version of the paper's binary layer window lambda_l, not as a steering-strength knob.

## Caveat on the fluency column

Produced by the pre-fix regex scoring, where the self-judge ignored the 0/1/2 rubric: it emitted
1.5 for 86% of items and never 0 or 2. Means are computed on the raw floats and are usable, but
only CLE-P paper's -0.051 is significant (Welch p<1e-4); CLE-P hlmean (-0.011, p=0.23) and both
CLE-A conditions (p=0.44, p=0.54) are not. Any 0/1/2 *distribution* derived from it is a rounding
artifact. Re-run with the default `--scoring constrained` to replace these.

