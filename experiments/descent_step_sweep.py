"""Does the flip direction change as we take MORE descent steps? Snapshot delta at 20/50/100.

Runs the same per-layer activation descent as refusal_activation_descent.py, but with NO early
stop: it descends the refuse-vs-comply log-odds for a fixed number of steps and snapshots the
steering vector delta_l at each checkpoint. Reports, per checkpoint:
  * cos(delta_l, w_l)  -- per-prompt mean and consensus (does alignment with the probe grow?)
  * ||delta_l|| and the log-odds reached
and the direction stability cos(delta@k, delta@last) -- has the direction settled by step k?

Reuses steer_hooks / log_odds / unit from refusal_activation_descent.py. GPU.

Usage:
    python experiments/descent_step_sweep.py --limit 20 --checkpoints 20,50,100 --lr 0.002
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from refusal_activation_descent import steer_hooks, log_odds, unit
from utils.args import parse_layers_arg
from utils.hooks import remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes, load_class_representations, layer_centroids
from utils.runtime import load_model, load_prompts, set_seed

OUT = os.path.join(ROOT, "experiments", "results", "refusal_gradient")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--reps_dir", default=None)
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--checkpoints", default="20,50,100")
    ap.add_argument("--refusal", default="I cannot,I can't,I'm sorry")
    ap.add_argument("--compliance", default="Sure, here,Here's how,Sure I can help")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
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
    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    last = checkpoints[-1]

    model = load_model(args)
    model.model.eval()
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))
    D = model.model.config.hidden_size
    probes = load_probes(probe_type="svm", svm_dir=args.svm_dir, layer_indices=selected,
                         device=device, explicit_reps_dir=args.reps_dir)
    w = {l: unit(probes[l]["w"].float().to(device)) for l in selected}
    Xh, Xl = load_class_representations(args.reps_dir)
    dmean = {l: unit((layer_centroids(Xh, Xl, l, args.reps_dir)[0]
                      - layer_centroids(Xh, Xl, l, args.reps_dir)[1]).float().to(device)) for l in selected}
    refusal_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.refusal.split(",")]
    comply_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.compliance.split(",")]
    prompts, _ = load_prompts(args)
    prompts = prompts[: args.limit]
    print(f"layers {selected} | checkpoints {checkpoints} | lr {args.lr} | {len(prompts)} prompts")
    print(f"refusal {args.refusal!r}\ncompliance {args.compliance!r}\n")

    # accumulators keyed by checkpoint
    cos_w = {c: {l: [] for l in selected} for c in checkpoints}
    norms = {c: {l: [] for l in selected} for c in checkpoints}
    lo_at = {c: [] for c in checkpoints}
    cons = {c: {l: torch.zeros(D) for l in selected} for c in checkpoints}
    stab = {c: {l: [] for l in selected} for c in checkpoints}  # cos(delta@c, delta@last)

    for i, prompt in enumerate(prompts):
        inputs = model.prepare_inputs(prompt)
        pid = inputs.input_ids
        pos = pid.shape[1] - 1
        deltas = {l: torch.zeros(D, device=device, requires_grad=True) for l in selected}
        opt = torch.optim.SGD([deltas[l] for l in selected], lr=args.lr)
        handles = steer_hooks(layers, deltas, pos)
        snaps = {}
        try:
            for step in range(1, last + 1):
                opt.zero_grad()
                lo = log_odds(model, pid, refusal_ids, comply_ids, pos, device)
                lo.backward()
                opt.step()
                if step in checkpoints:
                    snaps[step] = {l: deltas[l].detach().clone() for l in selected}
                    with torch.no_grad():
                        lo_at[step].append(log_odds(model, pid, refusal_ids, comply_ids, pos, device).item())
        finally:
            remove_hooks(handles)
        for c in checkpoints:
            for l in selected:
                d = snaps[c][l]
                cos_w[c][l].append(torch.dot(unit(d), w[l]).item())
                norms[c][l].append(d.norm().item())
                cons[c][l] += unit(d).cpu()
                stab[c][l].append(torch.dot(unit(d), unit(snaps[last][l])).item())
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(prompts)}")
        torch.cuda.empty_cache()

    def m(xs):
        return float(np.mean(xs))

    print("\n=== cos(delta, w) by checkpoint (per-prompt mean / consensus) ===")
    print(f"{'L':>3} | " + " | ".join(f"{'@'+str(c):>17}" for c in checkpoints))
    for l in selected:
        cells = [f"{m(cos_w[c][l]):+.3f} / {torch.dot(unit(cons[c][l].to(device)), w[l]).item():+.3f}"
                 for c in checkpoints]
        print(f"{l:>3} | " + " | ".join(f"{x:>17}" for x in cells))

    print("\n=== ||delta|| , log-odds reached, and direction stability cos(delta@c, delta@last) ===")
    print(f"{'ckpt':>6} {'mean ||delta||':>14} {'mean log-odds':>14} {'mean cos(@c,@last)':>19}")
    for c in checkpoints:
        nrm = m([v for l in selected for v in norms[c][l]])
        stb = m([v for l in selected for v in stab[c][l]])
        print(f"{c:>6} {nrm:>14.3f} {m(lo_at[c]):>14.2f} {stb:>19.3f}")

    out = {"config": {k: getattr(args, k) for k in ["dataset", "layers", "limit", "checkpoints",
                                                    "refusal", "compliance", "lr"]},
           "checkpoints": checkpoints, "n_prompts": len(prompts),
           "per_layer_cos_w_mean": {str(c): {str(l): m(cos_w[c][l]) for l in selected} for c in checkpoints},
           "per_layer_cos_w_consensus": {str(c): {str(l): float(torch.dot(unit(cons[c][l].to(device)), w[l]).item())
                                                  for l in selected} for c in checkpoints},
           "mean_norm": {str(c): m([v for l in selected for v in norms[c][l]]) for c in checkpoints},
           "mean_log_odds": {str(c): m(lo_at[c]) for c in checkpoints},
           "direction_stability_vs_last": {str(c): m([v for l in selected for v in stab[c][l]]) for c in checkpoints}}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"descent_step_sweep_{args.dataset}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
