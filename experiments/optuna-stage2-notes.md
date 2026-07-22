# Optuna two-stage margin search: paper vs. code, and per-layer margin experiments

Session notes, 2026-07-22. Model under study: llama3-8b, window 11-18 (paper's Table 3
result). Everything here is reconstructable from the paper (arXiv:2605.21706) plus the
code; this file exists so we don't have to reconstruct it.

## 1. The gap: code implements stage 1 only

`optuna_search.py` optimizes a contiguous layer window `(s, e)` and **one shared scalar
margin `m`** applied to every selected layer. It runs a single `study.optimize(...,
n_trials=args.trials)` (optuna_search.py:563) with one `suggest_float("margin", ...)`
per trial (optuna_search.py:420) and passes `--margin <scalar>` (optuna_search.py:363).
Its own banner says so: "Search mode: layer window and shared margin optimized"
(optuna_search.py:555). No trial-budget split, no per-layer margins, no `--layer_margin`.

The paper's Algorithm 2 (Appendix D) is **two-stage**. The code is stage 1 for 100% of
the budget. This is a faithful-reproduction gap, not a misreading — the previous session
read it correctly.

## 2. What the paper actually specifies (Algorithm 2, App. D)

- `T_C = floor(0.7 T)`. Stage 1 = first 70% of trials: TPE-guided BO over `(s, e, m)`,
  with `lambda_l = 1{s <= l < e}` and `m_l = lambda_l * m`. Evaluate ASR on Dval with
  the validation judge. Keep the best `(lambda*, m*)`.
- Stage 2 = remaining 30%: **window frozen** at `lambda*`. For each active layer,
  `m_l <- Sample(M intersect {m* - eta, m*, m* + eta})` where `M = {10^-1 * k : k=1..50}`
  = 0.1 .. 5.0 and `eta = 0.1` (one grid step). Note it is **`Sample`, not `BO.suggest`**
  — stage 2 is random local search, no `BO.observe`. Keep-if-better.
- `ASR*` carries across both stages; return the best `(lambda*, {m*_l})` overall.
- Budget: `T = 500` for most models, `700` for >32 layers.
- These map to the README's `--margin_low 0.1 --margin_high 5.0 --margin_step 0.1`.

## 3. Unresolved discrepancy — DO NOT implement stage 2 literally without resolving this

Algorithm 2 line 13 allows only `{m* - eta, m*, m* + eta}` = a +/-0.1 band around the
single stage-1 scalar. For llama3-8b that scalar is `m* = 1.5` (Table 3, CLE-A), so a
literal reading permits only `{1.4, 1.5, 1.6}` at every layer.

But Figure 7(b) — same model, same 11-18 window — shows per-layer margins spanning
**~0.9 to 2.0** (`m* +/- 0.6`, six grid steps), read off the plot by eye (+/-0.05):

    L11=1.2  L12=2.0  L13=1.8  L14=1.8  L15=2.0  L16=0.9  L17=1.2   (mean 1.56)

Algorithm 2 as written **cannot produce Figure 7**. Most likely reconciliation: stage 2
re-centers on the current best *per-layer* margins, `Sample(M intersect {m*_l - eta,
m*_l, m*_l + eta})`, i.e. a coordinate-wise random walk seeded at `m*` that drifts over
~150 trials. Fits: Fig 7(b) values average to 1.56 ~ the shared m*. But the paper writes
`m*` (scalar), not `m*_l`. This is a genuine ambiguity — resolve before trusting any
stage-2 port. Also unstated in the paper: `{m*_l}` initialization (seed to `m* *
lambda*_l`) and dedup of repeated samples.

## 4. Table 3 (selected windows + shared margins), for reference

    Model          #L  window   CLE-A m  CLE-P m
    LLaMA2-7B       32  5-25     3.1      1.6
    LLaMA3-8B       32  11-18    1.5      1.3
    Mistral-7B-RR   32  5-17     1.4      0.6
    LLaMA3.2-3B     28  11-23    1.2      1.4
    Mistral-7Bv0.3  32  12-15    1.2      1.2
    Phi3.5-mini     32  5-20     2.2      2.1
    Olmo3-7B        32  11-29    1.6      0.8
    Qwen2.5-32B     64  3-47     3.0      2.0
    Mixtral-8x7B    32  10-28    1.1      0.6
    GPT-OSS-20B     24  9-17     9.8      9.8
    DeepSeek-R1-8B  32  13-20    1.8      1.4
    Qwen3.5-9B      32  12-18    6.2      5.5
    Phi-4-15B       40  9-30     2.6      2.0
    Gemma3-12B      48  18-29    1.0      1.0
    Ministral3-14B  40  13-33    2.0      0.8

Full extracted paper text + PDF live on the persistent volume:
`/workspace/lem-durable/paper_2605.21706_extracted.txt` and `.pdf`.

## 5. Per-layer margin experiments run this session (llama3-8b, layers 11-18)

Tooling added: `experiments/steering_magnitude.py` gained `--schedules NAME=v1,v2,...`
and `--out_tag`, plus `m`/`dm`/`score` columns and a margin-schedule plot panel.
`experiments/sweep_layer_margins.sh` runs mean-matched schedules through cle-a.py using
its existing `--layer_margins` support (tags outputs `marginvec<sha1>`, writes a
label->tag manifest under `completions/<model>/`).

### Mechanism findings (n=1 prompt, prefill, single seed — mechanism only, not effect size)

Shared margin is self-limiting. With flat `m=1.5`, L11 lands the activation at probe
score exactly -1.5; the model's own computation drifts it back only ~+0.2 per layer, so
downstream layers see score ~ -1.3 and their own `score + m ~ 0` — they barely steer.
L16 was a near no-op (0.1% of clean norm moved). L11 does ~56% of all steering.

The driver at each layer is the **increment `dm = m_l - m_{l-1}`**, not `m_l`. A schedule
only buys downstream work where `dm` exceeds the ~0.2 natural drift.

Summed `||dh||/||h_clean||` across the window, mean-matched schedules (~1.5):

    shared      1.5x7                       49.9%   (L11 does 56%)
    ramp_down   2.1->0.9                     69.2%   (L11 does 58%, just frontloads harder)
    ramp_up     0.9->2.1                     61.9%   (L11 does 27%)
    paper       Fig 7(b) above (mean 1.56)  101.6%  (L11 does 22%)

The paper's own schedule does ~2x the total steering work of a flat margin at nearly the
same mean.

Two unexpected structural facts:
- `||w||` decays monotonically with depth (0.905 @ L11 -> 0.352 @ L17, 2.6x). Since
  `||dh|| = |score|/||w||`, an identical margin increment buys 2.6x more displacement at
  L17 than L11. A flat `m` is therefore NOT a flat intervention — independent reason the
  shared-margin parameterization is ill-posed, and a reason to expect optimal `{m_l}` to
  rise with depth (which the paper's schedule broadly does until L16).
- The paper's largest single move is *backwards*: Fig 7(b) L15->L16 is `dm = -1.10`,
  pushing activations back toward refusal (largest ||dh|| in the run, 27%).

### Behavioral (n=5 harmbench_test, cle-a.py)

Baseline 5/5 refuse. All four schedules 0/5 refuse (all evade). Completions stay fluent
and on-task but content fidelity is low — e.g. the parathion "reaction equation" doesn't
balance and misassigns formulas; the KRACK "exploit" is a Scapy packet-print stub. CLE
flips the refusal *decision* cleanly but does not confer capability the model lacks. The
HarmBench judge scores attempt, not correctness, so these read as ASR successes anyway.

At mean ~1.5 every schedule saturates to 5/5, so completions can't rank them.

## 6. Next experiment (agreed, not yet run)

Threshold sweep: scale each schedule's mean margin down (e.g. 1.5 -> 1.2 -> 0.9 -> 0.6 ->
0.3) until evasion breaks. If `paper` (2x work per unit mean margin) still evades where
flat `shared` has snapped back to refusing, that's the quantitative case that per-layer
margins beat a shared scalar — and the measurement to trust before any stage-2 port.
Reuses `--layer_margins`, no new search code, ~30 min at n=5.
