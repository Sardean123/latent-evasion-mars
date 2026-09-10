"""Multi-step activation descent: find the steering direction that FLIPS refusal, per layer.

Motivation. The single-step gradient of logP(refusal) w.r.t. an activation is ~orthogonal to the
probe weight w (cos ~0.03-0.07 per-prompt; ~0.14 for the mean gradient). The hypothesis is that
one local gradient is too noisy: the direction you actually have to MOVE to make the model comply
may align with w even though the instantaneous gradient does not. This script tests that by, per
prompt, optimising a per-layer steering vector delta_l added to the last-prompt-token activation,
descending the refuse-vs-comply log-odds until it flips, and then comparing each delta_l to w_l.

Design (matches the request):
  * delta is MULTI-LAYER: an independent delta_l at every target layer, added to h_l[last prompt
    token]. Probes are layer-specific, so each delta_l is compared to its own w_l. The delta_l's are
    optimised JOINTLY (earlier layers' deltas propagate downstream, later deltas steer on top).
  * OBJECTIVE = the refuse-vs-comply log-odds:
        LO = logsumexp_r logP(refusal_r | prompt, delta) - logsumexp_c logP(compliance_c | prompt, delta)
    (logsumexp over each pool marginalises "probability of ANY refusal / ANY compliance phrasing".)
  * STOPPING CRITERION = the log-odds flip: stop the first step LO < stop_margin (default 0), i.e.
    the model now prefers complying. The NUMBER OF STEPS to reach it is recorded, not fixed.
  * MEASUREMENT = the net displacement delta_l itself (leaf, so delta_final = the whole move):
        per-prompt  cos(delta_l, w_l)   -> mean over prompts   (headline, matches the (a) style)
        consensus   cos(mean_p delta_l, w_l)                    (the (b) style)
    Sign matters: to comply, we expect steering AGAINST the harmful axis, i.e. cos(delta, w) < 0
    (delta ~ -w, like CLE's projection). |cos| near 1 would say "the effective steer IS the probe
    axis," which single gradients missed.

Why frozen params are fine: delta_l is a leaf requiring grad, so it provides the autograd path even
though the model weights don't (no embedding-grad hook needed here).

KNOWN CONCERNS to weigh after a first test (see the console notes):
  * With a SINGLE compliance string ("Sure, here"), the initial log-odds is huge (~+40) because that
    exact phrasing is intrinsically rare -- the flip may need a large, possibly off-manifold delta.
    Expanding the compliance/refusal POOLS (--refusal / --compliance take comma lists) makes the
    criterion a real "does it comply" test; this is the intended next step.
  * Unconstrained delta can go off-manifold (flip logP with an unnatural activation). --l2_penalty
    adds a soft norm cost; start at 0 to see the raw direction, then use it as a robustness check.
  * Steering only the last prompt token flips the first-token decision (cheap, and consistent with
    the single-gradient experiment which took grad at the same site). --steer_all is a future knob.

Usage (defaults are deliberately small for a first run; tune lr / max_steps):
    python experiments/refusal_activation_descent.py --limit 32 --max_steps 100 --lr 0.05
    python experiments/refusal_activation_descent.py --compliance "Sure, here,Here's how,Sure, I can" \
        --refusal "I cannot,I can't,I'm sorry"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes, load_class_representations, layer_centroids
from utils.runtime import load_model, load_prompts, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(v, eps=1e-12):
    return v / (torch.linalg.vector_norm(v).clamp_min(eps))


def steer_hooks(layers, deltas, pos):
    """Add delta_l to the block-l output at ONE position, differentiable w.r.t. delta_l."""
    def mk(l):
        d = deltas[l]

        def hook(module, inputs, output):
            h = hidden_from_output(output)
            add = torch.zeros_like(h)
            add[:, pos, :] = d.to(h.dtype)         # out-of-place; delta flows through
            return replace_hidden(output, h + add)
        return hook
    return [layers[l].register_forward_hook(mk(l)) for l in deltas]


def phrase_logp(model, pid, phrase_ids, pos, device):
    """Teacher-forced sum_t logP(phrase_t | prompt, phrase_<t), with the steer hooks active."""
    ids = torch.cat([pid, torch.tensor([phrase_ids], device=device)], dim=1)
    out = model.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    total = torch.zeros((), device=device)
    for j, tid in enumerate(phrase_ids):
        total = total + lp[pos + j, tid]
    return total


def log_odds(model, pid, refusal_ids, comply_ids, pos, device):
    r = torch.logsumexp(torch.stack([phrase_logp(model, pid, ids, pos, device) for ids in refusal_ids]), 0)
    c = torch.logsumexp(torch.stack([phrase_logp(model, pid, ids, pos, device) for ids in comply_ids]), 0)
    return r - c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--reps_dir", default=None)
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--refusal", default="I cannot", help="Comma-separated refusal pool.")
    ap.add_argument("--compliance", default="Sure, here", help="Comma-separated compliance pool.")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                    help="sgd: delta_final = accumulated raw gradient (faithful direction). adam: "
                         "per-coordinate sign-normalised first step warps the measured direction.")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--stop_margin", type=float, default=0.0, help="Stop when log-odds < this.")
    ap.add_argument("--l2_penalty", type=float, default=0.0, help="Soft cost on sum_l ||delta_l||^2.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_per_prompt", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if args.reps_dir is None:
        for cand in [os.path.join("../repo-backup/representations", args.model_name, "train_svm"),
                     os.path.join("dataset/representations/representations", args.model_name, "train_svm")]:
            if os.path.isdir(cand):
                args.reps_dir = cand
                break
    if args.out is None:
        args.out = os.path.join("experiments", "results", "refusal_gradient",
                                f"descent_{args.dataset}_{args.layers.replace('-', 'to')}.json")

    model = load_model(args)
    model.model.eval()
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))
    D = model.model.config.hidden_size

    probes = load_probes(probe_type="svm", svm_dir=args.svm_dir, layer_indices=selected,
                         device=device, explicit_reps_dir=args.reps_dir)
    w = {l: unit(probes[l]["w"].float().to(device)) for l in selected}
    dmean = None
    if args.reps_dir and os.path.isdir(args.reps_dir):
        Xh, Xl = load_class_representations(args.reps_dir)
        dmean = {l: unit((layer_centroids(Xh, Xl, l, args.reps_dir)[0]
                          - layer_centroids(Xh, Xl, l, args.reps_dir)[1]).float().to(device)) for l in selected}

    refusal_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.refusal.split(",")]
    comply_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.compliance.split(",")]
    print(f"layers {selected} | refusal pool {args.refusal!r} | compliance pool {args.compliance!r}")
    print(f"lr {args.lr} | max_steps {args.max_steps} | stop when log-odds < {args.stop_margin} "
          f"| l2 {args.l2_penalty}\n")

    prompts, cats = load_prompts(args)
    prompts = prompts[: args.limit]

    acc = {l: {"cos_w": [], "cos_dm": [], "norm": []} for l in selected}
    consensus = {l: torch.zeros(D) for l in selected}
    steps_list, flipped_list, lo0_list, lof_list = [], [], [], []
    per_prompt = []

    for i, prompt in enumerate(prompts):
        inputs = model.prepare_inputs(prompt)
        pid = inputs.input_ids
        pos = pid.shape[1] - 1
        deltas = {l: torch.zeros(D, device=device, requires_grad=True) for l in selected}
        params = [deltas[l] for l in selected]
        opt = (torch.optim.SGD(params, lr=args.lr) if args.optimizer == "sgd"
               else torch.optim.Adam(params, lr=args.lr))
        handles = steer_hooks(layers, deltas, pos)
        flipped, nsteps, lo0, lo_val = False, args.max_steps, None, None
        try:
            for step in range(args.max_steps):
                opt.zero_grad()
                lo = log_odds(model, pid, refusal_ids, comply_ids, pos, device)
                loss = lo
                if args.l2_penalty:
                    loss = loss + args.l2_penalty * sum((deltas[l] ** 2).sum() for l in selected)
                if lo0 is None:
                    lo0 = lo.item()
                lo_val = lo.item()
                if lo_val < args.stop_margin:
                    flipped, nsteps = True, step
                    break
                loss.backward()
                opt.step()
        finally:
            remove_hooks(handles)

        for l in selected:
            d = deltas[l].detach()
            acc[l]["cos_w"].append(torch.dot(unit(d), w[l]).item())
            if dmean:
                acc[l]["cos_dm"].append(torch.dot(unit(d), dmean[l]).item())
            acc[l]["norm"].append(d.norm().item())
            consensus[l] += unit(d).cpu()
        steps_list.append(nsteps); flipped_list.append(flipped)
        lo0_list.append(lo0); lof_list.append(lo_val)
        if args.save_per_prompt:
            per_prompt.append({"prompt": prompt, "category": cats[i], "flipped": flipped,
                               "steps": nsteps, "lo0": lo0, "lo_final": lo_val,
                               "cos_w": {l: acc[l]["cos_w"][-1] for l in selected}})
        if (i + 1) % 8 == 0:
            fr = np.mean(flipped_list)
            print(f"  {i+1}/{len(prompts)}  flip-rate {fr:.0%}  median steps "
                  f"{int(np.median([s for s, f in zip(steps_list, flipped_list) if f] or [0]))}")
        torch.cuda.empty_cache()

    # ---- aggregate ----
    def st(xs):
        a = np.array(xs, float)
        return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)), "n": len(a)} if len(a) else None

    flip_mask = np.array(flipped_list)
    summary = {"config": {k: getattr(args, k) for k in
                          ["model_name", "dataset", "layers", "limit", "refusal", "compliance",
                           "lr", "max_steps", "stop_margin", "l2_penalty", "seed"]},
               "n_prompts": len(prompts), "flip_rate": float(flip_mask.mean()),
               "steps_to_flip": st([s for s, f in zip(steps_list, flip_mask) if f]),
               "log_odds_initial_mean": float(np.mean(lo0_list)),
               "log_odds_final_mean": float(np.mean(lof_list)),
               "per_layer": {}}
    print(f"\n{'L':>3} {'cos(delta,w)':>16} {'cos(delta,dm)':>15} {'||delta||':>10}")
    for l in selected:
        cons = unit(consensus[l].to(device))
        row = {"cos_w": st(acc[l]["cos_w"]), "cos_dmean": st(acc[l]["cos_dm"]) if dmean else None,
               "norm": st(acc[l]["norm"]),
               "consensus_cos_w": float(torch.dot(cons, w[l]).item()),
               "consensus_cos_dmean": float(torch.dot(cons, dmean[l]).item()) if dmean else None}
        summary["per_layer"][str(l)] = row
        cw, cd = row["cos_w"], row["cos_dmean"]
        cd_s = f"{cd['mean']:+.3f}" if cd else "n/a"
        print(f"{l:>3} {cw['mean']:+.3f}±{cw['std']:.3f}  {cd_s:>15} {row['norm']['mean']:>10.2f}")
    print(f"\nflip-rate {summary['flip_rate']:.0%} | median steps-to-flip "
          f"{summary['steps_to_flip']['median'] if summary['steps_to_flip'] else 'n/a'} | "
          f"log-odds {summary['log_odds_initial_mean']:+.1f} -> {summary['log_odds_final_mean']:+.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    if args.save_per_prompt:
        json.dump(per_prompt, open(args.out.replace(".json", "_per_prompt.json"), "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
