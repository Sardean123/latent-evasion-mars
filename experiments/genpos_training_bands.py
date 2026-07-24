"""Compute the harmful/harmless TRAINING-distribution bands PER generated position, scored
on that position's own genpos probe -- the reference train_generated_probes.py fits the
genpos probes on but never persists (it only saves pooled HFx/HLx for the gentok/alltok
families). Reproduces that exact collection (Arditi train prompts, seed 0, greedy N-token
generation, one forward over prompt+gen with output_hidden_states) so the bands match the
distribution each svm_layerLL_posPP.pt was trained on.

pos -1 (post-instruction token) bands are NOT produced here -- they are the stock probe's own
training reps (train_svm/HFx_train.pt on the train_svm probe), added by the plot scripts.

Consumed by experiments/plot_over_generation_perpos.py, and by
probe_score_over_generation.py --method cle-p-genpos (per-position steering margins).

Output JSON: {"n_positions", "layers", "bands": {pos: {layer: {harmful:{mean,std,n},
harmless:{mean,std,n}}}}}.

Usage:
    python experiments/genpos_training_bands.py --model_name llama3-8b --device cuda:0 --n_positions 8
"""
import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.models_utils import get_transformer_layers
from utils.runtime import load_model, set_seed


def load_genpos_WB(genpos_dir, device):
    """Return W[p]=(L,D), B[p]=(L,), sorted layers, sorted positions. Missing (layer,pos)
    probes get NaN rows (skipped when summarizing)."""
    pat = re.compile(r"svm_layer(\d+)_pos(\d+)\.pt$")
    by_lp, layers, positions = {}, set(), set()
    for name in os.listdir(genpos_dir):
        m = pat.match(name)
        if not m:
            continue
        l, p = int(m.group(1)), int(m.group(2))
        o = torch.load(os.path.join(genpos_dir, name), map_location="cpu")
        by_lp[(l, p)] = (o["w"].float().view(-1), float(o["b"]))
        layers.add(l); positions.add(p)
    layers, positions = sorted(layers), sorted(positions)
    D = by_lp[(layers[0], positions[0])][0].numel()
    W, B = {}, {}
    for p in positions:
        Wp = torch.full((len(layers), D), float("nan"))
        Bp = torch.full((len(layers),), float("nan"))
        for li, l in enumerate(layers):
            if (l, p) in by_lp:
                w, b = by_lp[(l, p)]
                Wp[li] = w; Bp[li] = b
        W[p] = Wp.to(device); B[p] = Bp.to(device)
    return W, B, layers, positions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_positions", type=int, default=8, help="Generated positions 0..N-1 to collect.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    genpos_dir = os.path.join(ROOT, "dataset/representations", args.model_name, "train_svm_genpos")
    splits = os.path.join(ROOT, "dataset/splits", args.model_name)
    out = args.out or os.path.join(
        ROOT, "experiments/results/probe_score_over_generation",
        f"genpos_training_bands_{args.model_name}.json")
    N = args.n_positions

    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args)
    n_layers = len(get_transformer_layers(model))

    to_str = lambda d: d["instruction"] if isinstance(d, dict) else d
    harmful = [to_str(d) for d in json.load(open(f"{splits}/harmful_train_filtered.json"))]
    harmless = [to_str(d) for d in json.load(open(f"{splits}/harmless_train_filtered.json"))]
    prompts = harmful + harmless
    labels = [1] * len(harmful) + [0] * len(harmless)
    print(f"{len(harmful)} harmful + {len(harmless)} harmless prompts | {n_layers} layers | N={N}")

    W, B, layers, positions = load_genpos_WB(genpos_dir, device)
    assert layers == list(range(n_layers)), (layers[:3], layers[-3:])
    scores = {p: {1: [], 0: []} for p in positions}  # scores[p][cls] -> list of (L,) per prompt

    for prompt, label in zip(tqdm(prompts, desc="Collecting"), labels):
        inputs = model.prepare_inputs(prompt)
        plen = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.model.generate(**inputs, max_new_tokens=N, do_sample=False)
        full_ids = gen[:, : plen + N]
        with torch.no_grad():
            out_hs = model.model(input_ids=full_ids, output_hidden_states=True)
        gen_len = full_ids.shape[1] - plen
        for p in positions:
            if p >= gen_len:
                continue
            h = torch.stack([out_hs.hidden_states[l + 1][0, plen + p, :] for l in layers], dim=0).float()
            s = (h * W[p]).sum(dim=1) + B[p]  # (L,)
            scores[p][label].append(s.cpu())

    bands = {}
    for p in positions:
        bands[str(p)] = {}
        for cls, key in ((1, "harmful"), (0, "harmless")):
            if not scores[p][cls]:
                continue
            M = torch.stack(scores[p][cls], dim=0)  # (n_prompts, L)
            mu, sd = M.mean(dim=0), M.std(dim=0, unbiased=True)
            for li, l in enumerate(layers):
                if torch.isnan(mu[li]):
                    continue
                bands[str(p)].setdefault(str(l), {})[key] = {
                    "mean": mu[li].item(), "std": sd[li].item(), "n": M.shape[0]}

    data = {"model": args.model_name, "n_positions": len(positions), "layers": layers, "bands": bands}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved per-position training bands to {out}")
    for p in positions[:3]:
        b = bands[str(p)].get("17", {})
        if b:
            print(f"  pos {p} L17: harmless {b['harmless']['mean']:.2f}  harmful {b['harmful']['mean']:.2f}")


if __name__ == "__main__":
    main()
