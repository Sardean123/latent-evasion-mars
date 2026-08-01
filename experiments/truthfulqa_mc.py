"""TruthfulQA MC1/MC2 accuracy under CLE steering -- a coherence/capability-retention proxy.

TruthfulQA MC scores answers by LOG-PROBABILITY (no generation): for each question the model
assigns a likelihood to every reference answer.
  * MC1: the single best-correct answer must get the highest log-prob (1/0 per question).
  * MC2: normalized probability mass placed on ALL correct answers (0-1 per question).
If steering makes the model incoherent, its ability to prefer true over false answers drops,
so MC1/MC2 fall below the unsteered baseline.

We run three conditions in one model load, applying the same intervention the chosen CLE
variant uses at inference during the answer-scoring forwards (so the steered model is scored,
not the base model):
  * baseline : no steering
  * paper    : CLE paper BO-optimized per-layer margins (Fig 7b)
  * hlmean   : BO-free harmless-mean margins

--method selects the variant, mirroring what cle-p.py / cle-a.py do at generation time:
  * clep : projection_hook active at the window layers for every scored position (cle-p.py
           keeps the projection on throughout prefill and decoding).
  * clea : per-question additive delta. A prompt-only forward with the projection hook
           captures the last-prompt-token delta per layer (cle-a.py's compute_additive_deltas),
           and that fixed delta is then added at every position of the answer-scoring forward
           (cle-a.py's add_hook during generation). Deltas are recomputed per question because
           they depend on the prompt.

Answers are scored batched per question (context is shared, answers right-padded); raw summed
token log-probs, matching lm-eval's truthfulqa_mc. Reports MC1/MC2 overall, per category, and
the delta vs baseline.

Usage:
    python experiments/truthfulqa_mc.py --model_name llama3-8b --layers 11-18
    python experiments/truthfulqa_mc.py --method clea
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.args import gate_tag, parse_layers_arg, resolve_gate_thresholds
from utils.hooks import add_hook, gated_projection_hook, pipeline_delta_hook, projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes
from utils.runtime import load_model, set_seed

PAPER = [1.2, 2.0, 1.8, 1.8, 2.0, 0.9, 1.2]
HLMEAN = [1.08, 1.08, 1.1, 1.11, 1.12, 1.14, 1.14]


def answer_loglikes(model, context_ids, answers, max_batch=16):
    """Sum of token log-probs of each answer string appended after context_ids (1, Lc).
    Shared context => answers start at the same position Lc; right-pad and read answer spans."""
    tok = model.tokenizer
    Lc = context_ids.shape[1]
    enc = [tok(a, add_special_tokens=False, return_tensors="pt").input_ids[0] for a in answers]
    out = [0.0] * len(answers)
    for s in range(0, len(answers), max_batch):
        chunk = list(range(s, min(s + max_batch, len(answers))))
        La = [enc[i].shape[0] for i in chunk]
        Lmax = Lc + max(La)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        seqs = torch.full((len(chunk), Lmax), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), Lmax), dtype=torch.long)
        for r, i in enumerate(chunk):
            full = torch.cat([context_ids[0].cpu(), enc[i]])
            seqs[r, :full.shape[0]] = full
            mask[r, :full.shape[0]] = 1
        seqs = seqs.to(model.device); mask = mask.to(model.device)
        with torch.no_grad():
            logits = model.model(input_ids=seqs, attention_mask=mask).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for r, i in enumerate(chunk):
            n = La[r]
            if n == 0:
                out[i] = -1e30
                continue
            # answer tokens occupy [Lc, Lc+n); each predicted by the logits at the previous pos
            idx = torch.arange(Lc, Lc + n)
            tgt = seqs[r, idx]
            out[i] = logp[r, idx - 1, tgt].sum().item()
    return out


def clea_deltas(model, layers, window, probes, margins, context_ids):
    """CLE-A's compute_additive_deltas: prompt-only forward with the projection hook, keeping
    the per-layer delta at the last prompt token. Returned as 1-D tensors so add_hook
    broadcasts them across the answer batch and all positions.

    Also returns the clean (pre-steering) activation norm at that position per layer, so the
    caller can report ||delta|| as a fraction of it."""
    deltas, clean_norms = {}, {}

    def norm_probe(layer_idx):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            clean_norms[layer_idx] = float(h[:, -1, :].float().norm())
        return hook

    handles = []
    for l in window:
        # the norm probe must run BEFORE the steering hook to see the clean activation
        handles.append(layers[l].register_forward_hook(norm_probe(l)))
        handles.append(layers[l].register_forward_hook(
            pipeline_delta_hook(probes[l]["w"], probes[l]["b"], 1.0, margins[l], l, deltas)))
    try:
        with torch.no_grad():
            model.model(input_ids=context_ids)
    finally:
        remove_hooks(handles)
    missing = [l for l in window if l not in deltas]
    if missing:
        raise RuntimeError(f"Failed to capture CLE-A deltas for layers: {missing}")
    return {l: deltas[l][0] for l in window}, clean_norms  # (1, D) -> (D,)


def score_condition(model, layers, window, probes, margins, questions, cats, method="clep", gate_c="0"):
    handles = []
    gate_stats = {}
    if margins is not None and method == "clep":
        handles = [layers[l].register_forward_hook(
            projection_hook(probes[l]["w"], probes[l]["b"], 1.0, margins[l])) for l in window]
    elif margins is not None and method == "clepstar":
        gate_map = resolve_gate_thresholds(gate_c, margins)
        handles = [layers[l].register_forward_hook(
            gated_projection_hook(probes[l]["w"], probes[l]["b"], 1.0, margins[l], gate_map[l],
                                  gate_stats.setdefault(l, {}))) for l in window]
    mc1_hits, mc2_vals = [], []
    per_cat = {}
    # CLE-A only: how hard does the intervention actually push on these (benign) prompts?
    # ||delta|| relative to the clean activation norm at the same position, per window layer.
    delta_rel = {l: [] for l in window}
    try:
        for q in tqdm(questions, desc="scoring", leave=False):
            ctx = model.prepare_inputs(q["question"]).input_ids
            mc1, mc2 = q["mc1_targets"], q["mc2_targets"]
            answers = list(dict.fromkeys(list(mc1) + list(mc2)))  # unique, score once
            per_q = []
            if margins is not None and method == "clea":
                deltas, clean_norms = clea_deltas(model, layers, window, probes, margins, ctx)
                per_q = [layers[l].register_forward_hook(add_hook(deltas[l])) for l in window]
                for l in window:
                    delta_rel[l].append(float(deltas[l].norm() / clean_norms[l]))
            try:
                ll = dict(zip(answers, answer_loglikes(model, ctx, answers)))
            finally:
                remove_hooks(per_q)
            # MC1: argmax over mc1 answers must be the (single) correct one
            best = max(mc1, key=lambda a: ll[a])
            mc1_hit = int(mc1[best] == 1)
            # MC2: normalized prob mass on correct answers within mc2 set
            keys = list(mc2)
            lls = torch.tensor([ll[a] for a in keys])
            probs = torch.softmax(lls, dim=0)
            mc2_val = float(sum(probs[j] for j, a in enumerate(keys) if mc2[a] == 1))
            mc1_hits.append(mc1_hit); mc2_vals.append(mc2_val)
            c = cats.get(q["question"], "unknown")
            per_cat.setdefault(c, {"mc1": [], "mc2": []})
            per_cat[c]["mc1"].append(mc1_hit); per_cat[c]["mc2"].append(mc2_val)
    finally:
        remove_hooks(handles)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    out = {
        "mc1": mean(mc1_hits), "mc2": mean(mc2_vals), "n": len(mc1_hits),
        "per_category": {c: {"mc1": mean(v["mc1"]), "mc2": mean(v["mc2"]), "n": len(v["mc1"])}
                         for c, v in sorted(per_cat.items())},
    }
    if any(delta_rel[l] for l in window):
        # mean ||delta|| / ||h|| at the steered position, per layer (h = unsteered at that layer)
        out["delta_rel_norm"] = {str(l): mean(delta_rel[l]) for l in window}
    if gate_stats:
        # CLE-P* only: fraction of scored positions the gate let through, per layer
        out["gate_rate"] = {str(l): float(s["fired"] / s["total"])
                            for l, s in sorted(gate_stats.items()) if float(s.get("total", 0)) > 0}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--method", default="clep", choices=["clep", "clea", "clepstar"],
                    help="Which CLE variant's intervention to apply while scoring (see module docstring).")
    ap.add_argument("--gate_c", default="0",
                    help="CLE-P* only: gate threshold, absolute ('0') or as a fraction of the "
                         "layer margin ('-0.5m' = halfway between -m_l and the boundary; "
                         "'-1m'/'relu' = never steer backwards).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "experiments/results/truthfulqa_mc"))
    args = ap.parse_args()

    set_seed(args.seed)
    model = load_model(args)
    layers = get_transformer_layers(model)
    window = parse_layers_arg(args.layers, len(layers))
    if len(window) != len(PAPER):
        raise ValueError(f"window {window} has {len(window)} layers but margin schedules have {len(PAPER)}")
    probes = load_probes(probe_type="svm",
                         svm_dir=os.path.join(ROOT, "dataset/representations", args.model_name, "train_svm"),
                         layer_indices=window, device=torch.device("cpu"))

    mc = json.load(open(os.path.join(ROOT, "dataset/processed/truthfulqa_mc.json")))
    cats = {d["instruction"]: d["category"]
            for d in json.load(open(os.path.join(ROOT, "dataset/processed/truthfulqa.json")))}
    if args.limit:
        mc = mc[: args.limit]

    conds = {"baseline": None,
             "paper": {l: m for l, m in zip(window, PAPER)},
             "hlmean": {l: m for l, m in zip(window, HLMEAN)}}
    results = {}
    for name, margins in conds.items():
        print(f"\n=== condition: {name}  method={args.method}  margins={margins} ===")
        results[name] = score_condition(model, layers, window, probes, margins, mc, cats,
                                        method=args.method, gate_c=args.gate_c)
        r = results[name]
        print(f"  MC1={r['mc1']*100:.1f}%  MC2={r['mc2']*100:.1f}%  (n={r['n']})")

    os.makedirs(args.out_dir, exist_ok=True)
    # The original CLE-P result file has no method suffix; keep that name stable.
    suffix = "" if args.method == "clep" else f"_{args.method}"
    if args.method == "clepstar":
        suffix += gate_tag(args.gate_c)
    out = os.path.join(args.out_dir,
                       f"truthfulqa_mc_{args.model_name}_layers{args.layers.replace('-','to')}{suffix}.json")
    json.dump({"model": args.model_name, "method": args.method, "gate_c": args.gate_c,
               "window": window, "conditions": results,
               "margins": {"paper": PAPER, "hlmean": HLMEAN}}, open(out, "w"), indent=2)

    print(f"\n===== TruthfulQA MC ({args.method}, relative to baseline) =====")
    b = results["baseline"]
    print(f"{'condition':>10} {'MC1':>8} {'ΔMC1':>8} {'MC2':>8} {'ΔMC2':>8}")
    for name in ("baseline", "paper", "hlmean"):
        r = results[name]
        d1 = (r["mc1"] - b["mc1"]) * 100
        d2 = (r["mc2"] - b["mc2"]) * 100
        print(f"{name:>10} {r['mc1']*100:>7.1f}% {d1:>+7.1f} {r['mc2']*100:>7.1f}% {d2:>+7.1f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
