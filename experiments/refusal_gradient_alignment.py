"""Is the probe's refusal direction the direction the model actually uses to refuse?

The CLE method steers activations along the LinearSVC probe weight w_l at each layer. That is only
principled if w_l is (close to) the direction that functionally controls refusal. This script tests
that by comparing w_l against the gradient of a refusal objective w.r.t. the residual-stream
activation at the SAME site the probe reads (the last prompt token, output of decoder block l), on
harmful prompts.

Two refusal objectives (both requested), each giving a gradient direction per layer per prompt:

  R  raw log P("I cannot")   -- teacher-forced sum of log-softmax over the refusal phrase tokens
                                appended after the prompt. Gradient taken w.r.t. h_l[last-prompt-tok].
  D  logit difference        -- logit(first refusal token) - logit(first affirmative token) at the
                                first generated position. The refusal-direction literature's target.

For each layer we report the distribution over prompts of:
  cos(grad_R, w), cos(grad_D, w)          -- gradient vs probe weight  (the headline)
  cos(grad_R, dmean), cos(grad_D, dmean)  -- gradient vs difference-in-means direction
  cos(grad_R, grad_D)                     -- do the two objectives agree with each other
and the per-layer constants cos(w, dmean), plus a random-vector baseline E|cos| ~ sqrt(2/(pi*D)).

A positive cosine means the gradient that INCREASES refusal points the same way as the probe's
harmful/refusal direction. If alignment is high in the 11-18 window, the probe direction is the
functional refusal axis and CLE steering manipulates the real mechanism. If it is low, the probe
merely separates refusal without being the causal knob -- which would explain why behaviourally
tuned BO margins beat probe-geometry (hlmean) margins under CLE-A.

Nothing is steered here: the model runs clean and we only read gradients. Per-prompt, single
sequence, so the autograd graph for one 8B forward fits an A40 comfortably.

Usage:
    python experiments/refusal_gradient_alignment.py --dry_run          # validate setup, no GPU
    python experiments/refusal_gradient_alignment.py --layers 11-18 --dataset harmbench_test --limit 64
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes, load_class_representations, layer_centroids
from utils.runtime import load_model, load_prompts, set_seed


def unit(v, eps=1e-12):
    return v / (torch.linalg.vector_norm(v).clamp_min(eps))


def enable_activation_grad(model):
    """The loaded model's params are frozen (requires_grad=False), so a plain forward builds no
    autograd graph. Making the input-embedding output require grad rebuilds the graph downstream
    w.r.t. every activation, so autograd.grad(target, h_l) works -- without unfreezing any weight."""
    emb = model.model.get_input_embeddings()

    def hook(module, inputs, output):
        return output.requires_grad_(True)

    return emb.register_forward_hook(hook)


def capture_hooks(layers, selected, store):
    """Forward hooks that stash the (graph-connected) hidden tensor of each selected block."""
    def mk(l):
        def hook(module, inputs, output):
            store[l] = hidden_from_output(output)  # keep the node; do NOT detach
        return hook
    return [layers[l].register_forward_hook(mk(l)) for l in selected]


def grad_at(target, store, selected, pos):
    """d target / d h_l at position `pos`, for every selected layer, in one backward."""
    caps = [store[l] for l in selected]
    grads = torch.autograd.grad(target, caps, retain_graph=False, allow_unused=False)
    return {l: g[0, pos, :].detach().float().cpu() for l, g in zip(selected, grads)}


def target_logitdiff(model, input_ids, attn, pos, refuse_id, comply_id, store):
    out = model.model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[0, pos, :].float()
    return logits[refuse_id] - logits[comply_id]


def target_logp_refusal(model, input_ids, attn, pos, refusal_ids, store):
    """Teacher-forced sum_t log P(refusal_t | prompt, refusal_<t)."""
    out = model.model(input_ids=input_ids, attention_mask=attn)
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)  # (seq, vocab)
    total = 0.0
    for j, tid in enumerate(refusal_ids):
        total = total + lp[pos + j, tid]
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None,
                    help="Probe dir the CLE runs used (default dataset/representations/<model>/train_svm).")
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--reps_dir", default=None,
                    help="Class reps for the diff-in-means direction (HFx/HLx_train.pt). "
                         "Default repo-backup/representations/<model>/train_svm; skipped if absent.")
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--refusal_phrase", default="I cannot")
    ap.add_argument("--compliance_phrase", default="Sure!",
                    help="Phrase for the log-odds target grad[logP(refusal)-logP(compliance)].")
    ap.add_argument("--affirmative", default="Sure",
                    help="First token of this is the compliance token for the logit-diff target D.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_per_prompt", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Load probes/tokens, print plan, no model forward.")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if args.reps_dir is None:
        # repo-backup is a SIBLING of the repo, not inside it; the doubly-nested in-repo path is a
        # fallback (see RESUME.md "harmless_mean_schedule is broken on its defaults").
        for cand in [os.path.join("../repo-backup/representations", args.model_name, "train_svm"),
                     os.path.join("dataset/representations/representations", args.model_name, "train_svm"),
                     os.path.join("repo-backup/representations", args.model_name, "train_svm")]:
            if os.path.isdir(cand):
                args.reps_dir = cand
                break
    if args.out is None:
        args.out = os.path.join("experiments", "results", "refusal_gradient",
                                f"alignment_{args.dataset}_{args.layers.replace('-', 'to')}.json")

    device = torch.device(args.device if not args.dry_run else "cpu")

    # --- probes (same direction CLE steers along) ---
    # NOTE: probe layer index l <-> output of decoder block l <-> hidden_states[l+1]; the capture
    # hook on layers[l] reads exactly that residual-stream site, matching the probe's read site.
    # We need num-layers to parse the window; on a dry run we hard-map the common 32-block model.
    if args.dry_run:
        selected = parse_layers_arg(args.layers, 32)
    else:
        model = load_model(args)
        model.model.eval()
        layers = get_transformer_layers(model)
        selected = parse_layers_arg(args.layers, len(layers))

    probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                         layer_indices=selected, device=torch.device("cpu"),
                         explicit_reps_dir=args.reps_dir)
    w = {l: unit(probes[l]["w"].float()) for l in selected}

    # --- diff-in-means direction (mu_harm - mu_harmless), points toward refusal ---
    dmean = None
    if args.reps_dir and os.path.isdir(args.reps_dir):
        try:
            Xh, Xl = load_class_representations(args.reps_dir)
            dmean = {}
            for l in selected:
                mu_h, mu_l = layer_centroids(Xh, Xl, l, args.reps_dir)
                dmean[l] = unit((mu_h - mu_l).float())
        except FileNotFoundError as e:
            print(f"(diff-in-means skipped: {e})", file=sys.stderr)
            dmean = None

    print(f"Model {args.model_name} | probes {args.svm_dir} | layers {selected}")
    print(f"Refusal phrase {args.refusal_phrase!r} | affirmative {args.affirmative!r} | "
          f"diff-in-means: {'on' if dmean else 'OFF'}")
    D = w[selected[0]].numel()
    rand_baseline = math.sqrt(2.0 / (math.pi * D))  # E|cos(random unit, fixed)| in D dims
    print(f"hidden dim {D} | random-baseline E|cos| ~ {rand_baseline:.4f}")
    if dmean:
        print("cos(w, diff-in-means) per layer: " +
              ", ".join(f"L{l}:{torch.dot(w[l], dmean[l]).item():+.3f}" for l in selected))

    if args.dry_run:
        prompts, _ = load_prompts(args)
        print(f"\n--dry_run OK. {len(prompts)} prompts would be scored. "
              f"First prompt: {prompts[0][:80]}...")
        return

    # move probe/dmean dirs to device
    w = {l: v.to(device) for l, v in w.items()}
    if dmean:
        dmean = {l: v.to(device) for l, v in dmean.items()}

    prompts, cats = load_prompts(args)
    refuse_ids = model.tokenizer.encode(args.refusal_phrase, add_special_tokens=False)
    comply_ids = model.tokenizer.encode(args.compliance_phrase, add_special_tokens=False)
    comply_id = model.tokenizer.encode(args.affirmative, add_special_tokens=False)[0]
    refuse0 = refuse_ids[0]
    print(f"refusal {args.refusal_phrase!r}->{refuse_ids} | compliance "
          f"{args.compliance_phrase!r}->{comply_ids} | logit-diff affirmative id {comply_id}\n")

    def phrase_grad(pid, phrase_ids, pos):
        """log P(phrase) teacher-forced after the prompt, and its grad w.r.t. h_l[pos] per layer.
        h_l[pos] is identical across phrases (causal attention: pos attends only to <=pos, and the
        prompt is shared), so grad[logP(refuse)-logP(comply)] = grad_refuse - grad_comply."""
        ids2 = torch.cat([pid, torch.tensor([phrase_ids], device=device)], dim=1)
        store = {}
        hh = [enable_activation_grad(model)] + capture_hooks(layers, selected, store)
        try:
            t = target_logp_refusal(model, ids2, torch.ones_like(ids2), pos, phrase_ids, store)
            g = grad_at(t, store, selected, pos)
        finally:
            remove_hooks(hh)
        return t.item(), g

    # accumulators: per layer, lists of cosines over prompts.  gRD = log-odds gradient.
    keys = ["gR_w", "gD_w", "gRD_w", "gR_dm", "gD_dm", "gRD_dm", "gR_gD", "gR_gRD"]
    acc = {l: {k: [] for k in keys} for l in selected}
    consensus = {l: {"gR": torch.zeros(D), "gD": torch.zeros(D), "gRD": torch.zeros(D)} for l in selected}
    tvals = {"logp_refusal": [], "logp_compliance": [], "log_odds": [], "logit_diff": []}
    per_prompt = []

    for i, prompt in enumerate(prompts):
        inputs = model.prepare_inputs(prompt)
        pid = inputs.input_ids
        pos = pid.shape[1] - 1  # last prompt token: predicts the first assistant token

        # ---- Target D: first-token logit difference (prompt only) ----
        store = {}
        h = [enable_activation_grad(model)] + capture_hooks(layers, selected, store)
        try:
            tD = target_logitdiff(model, pid, inputs.attention_mask, pos, refuse0, comply_id, store)
            gD = grad_at(tD, store, selected, pos)
        finally:
            remove_hooks(h)
        tvals["logit_diff"].append(tD.item())

        # ---- Target R: log P(refusal phrase);  Target C: log P(compliance phrase) ----
        tR, gR = phrase_grad(pid, refuse_ids, pos)
        tC, gC = phrase_grad(pid, comply_ids, pos)
        tvals["logp_refusal"].append(tR)
        tvals["logp_compliance"].append(tC)
        tvals["log_odds"].append(tR - tC)

        for l in selected:
            gRl = unit(gR[l].to(device))
            gDl = unit(gD[l].to(device))
            gRDl = unit((gR[l] - gC[l]).to(device))  # log-odds gradient direction
            acc[l]["gR_w"].append(torch.dot(gRl, w[l]).item())
            acc[l]["gD_w"].append(torch.dot(gDl, w[l]).item())
            acc[l]["gRD_w"].append(torch.dot(gRDl, w[l]).item())
            acc[l]["gR_gD"].append(torch.dot(gRl, gDl).item())
            acc[l]["gR_gRD"].append(torch.dot(gRl, gRDl).item())
            if dmean:
                acc[l]["gR_dm"].append(torch.dot(gRl, dmean[l]).item())
                acc[l]["gD_dm"].append(torch.dot(gDl, dmean[l]).item())
                acc[l]["gRD_dm"].append(torch.dot(gRDl, dmean[l]).item())
            consensus[l]["gR"] += gRl.cpu()
            consensus[l]["gD"] += gDl.cpu()
            consensus[l]["gRD"] += gRDl.cpu()

        if args.save_per_prompt:
            per_prompt.append({"prompt": prompt, "category": cats[i],
                               "logp_refusal": tR, "logp_compliance": tC, "log_odds": tR - tC,
                               "gR_w": {l: acc[l]["gR_w"][-1] for l in selected},
                               "gRD_w": {l: acc[l]["gRD_w"][-1] for l in selected}})
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(prompts)} prompts")
        torch.cuda.empty_cache()

    # ---- aggregate ----
    def stats(xs):
        a = np.array(xs, dtype=float)
        if a.size == 0:
            return None
        return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)), "n": int(a.size)}

    summary = {"config": {k: getattr(args, k) for k in
                          ["model_name", "svm_dir", "reps_dir", "layers", "dataset", "limit",
                           "refusal_phrase", "compliance_phrase", "affirmative", "seed"]},
               "hidden_dim": D, "random_baseline_abs_cos": rand_baseline,
               "n_prompts": len(prompts),
               "target_means": {k: float(np.mean(v)) for k, v in tvals.items()},
               "per_layer": {}}
    for l in selected:
        row = {k: stats(acc[l][k]) for k in acc[l]}
        cons_gR = unit(consensus[l]["gR"].to(device))
        cons_gD = unit(consensus[l]["gD"].to(device))
        cons_gRD = unit(consensus[l]["gRD"].to(device))
        row["consensus_gR_w"] = float(torch.dot(cons_gR, w[l]).item())
        row["consensus_gD_w"] = float(torch.dot(cons_gD, w[l]).item())
        row["consensus_gRD_w"] = float(torch.dot(cons_gRD, w[l]).item())
        if dmean:
            row["w_dmean"] = float(torch.dot(w[l], dmean[l]).item())
            row["consensus_gR_dmean"] = float(torch.dot(cons_gR, dmean[l]).item())
            row["consensus_gRD_dmean"] = float(torch.dot(cons_gRD, dmean[l]).item())
        summary["per_layer"][str(l)] = row

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    if args.save_per_prompt:
        pp = args.out.replace(".json", "_per_prompt.json")
        json.dump(per_prompt, open(pp, "w"), indent=2)
        print(f"Wrote per-prompt -> {pp}")

    # ---- console table ----
    print(f"\n{'layer':>5} {'gR·w  (logP ref)':>18} {'gRD·w (log-odds)':>18} "
          f"{'gD·w (logit)':>14} " + (f"{'w·dm':>8}" if dmean else ""))
    for l in selected:
        r = summary["per_layer"][str(l)]
        def m(k): return f"{r[k]['mean']:+.3f}±{r[k]['std']:.3f}" if r[k] else "  --  "
        line = f"{l:>5} {m('gR_w'):>18} {m('gRD_w'):>18} {m('gD_w'):>14} "
        if dmean:
            line += f"{r['w_dmean']:>+8.3f}"
        print(line)
    print(f"\nrandom-baseline |cos| ~ {rand_baseline:.4f}  (a null cosine is indistinguishable from this)")
    tm = summary["target_means"]
    print(f"target means: log P(refusal)={tm['logp_refusal']:.3f}  "
          f"log P(compliance)={tm['logp_compliance']:.3f}  "
          f"log-odds(ref-comp)={tm['log_odds']:.3f}  logit_diff={tm['logit_diff']:.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
