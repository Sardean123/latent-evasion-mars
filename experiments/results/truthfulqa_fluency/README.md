# TruthfulQA free-form fluency (AxBench-style, self-judge) — CLE-P margins

**Question:** does heavier steering (paper BO, mean 1.56) hurt free-form *fluency* more than the
lighter harmless-mean margins (hlmean, mean 1.11)? Complements the judge-free TruthfulQA MC
coherence proxy (`../truthfulqa_mc/`) with a direct text-quality read.

**Setup:** free-form CLE-P generations on TruthfulQA (790, 512 tokens) for baseline / paper / hlmean,
scored 0-2 for fluency by the AxBench fluency prompt (edited: "response" not "sentence fragment").
Judge = **Meta-Llama-3-8B-Instruct (self-judge first pass)**, no system prompt. `truthfulqa_fluency.py`.

## Result

| condition | mean fluency | % fluent (2) | % degenerate (0) | Δ vs baseline |
| --- | --- | --- | --- | --- |
| baseline | 1.481 | 92.8% | 0.8% | — |
| paper (BO, 1.56) | 1.430 | 86.1% | 1.8% | −0.051 |
| **hlmean (1.11)** | 1.470 | 91.6% | 0.8% | **−0.011** |

hlmean's fluency is nearly identical to unsteered; paper degrades ~5x more (−0.051 vs −0.011),
dropping ~7 pts of fully-fluent responses and doubling the degenerate rate. Same direction as
the MC2 coherence result — the lighter margins preserve text quality better.

## Caveats (rough first pass)

- **Self-judge**: the judge is the same base model that produced the text. Fine for a surface
  property like fluency, but it's miscalibrated in *absolute* terms (harsh on terse text,
  penalizes for engagement/informativeness despite the rubric) — trust only the *relative*
  ordering, not the absolute scores. An independent/stronger judge (e.g. gpt-4o-mini per AxBench)
  would firm up the numbers.
- **~6-7% parse failures/condition** (~50/790): the judge sometimes ran past the 256-token
  budget without a parseable `Rating:`. Excluded from the mean; adds some noise.
- **Small gaps**: at the tuned margins neither method degenerates much (consistent with the low
  4-gram repetition in `../RESULTS.md`). The fluency difference is real but modest.

`truthfulqa_fluency_summary.json` (means + 0/1/2 distributions), `truthfulqa_fluency_graded.json`
(per-item ratings + raw judge text). Raw generations: `completions/llama3-8b/projection/` (git-ignored).
