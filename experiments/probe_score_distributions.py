"""Where do steered activations land relative to the probe's training distributions?

The probes are frozen after training: per layer, a LinearSVC fit on last-prompt-token
activations of harmful (y=1) vs harmless (y=0) prompts, no normalization. This script
asks whether CLE's steered activations actually resemble the harmless activations the
probe was fit on, or merely satisfy the probe's decision rule.

Two distinct questions, because they have different answers:

1. SCORE (1-D, along w). At an intervened layer with beta=1 the steered score is exactly
   -m_l by construction: w.h* + b = (w.h + b) - (w.h + b + m) = -m. So the steered score
   distribution is a point mass, and the only real question is where -m_l sits relative
   to the harmless score distribution. Reported as a z-score against harmless mean/std.
   A |z| >> 1 means CLE parks activations in a region no harmless prompt occupies.

2. GEOMETRY (full space). The projection moves h ONLY along w. Every component
   orthogonal to w is untouched, so an activation can sit at a perfectly harmless-looking
   score while remaining as far from the harmless cloud as it started. We decompose
   h_steered - mu_harmless into its along-w and orthogonal parts and compare the
   displacement to the natural class separation ||mu_harmful - mu_harmless||.

Probes are scored at EVERY layer, but steering is applied only inside the window, so the
columns outside the window show whether the intervention persists downstream or the model
drifts back toward refusal on its own.

All activations are taken at the last prompt token via the same chat template used to
build HFx_train.pt / HLx_train.pt, so the comparison is like-for-like. Prompts are run one
at a time (no padding) to keep the last-token position exact.

Usage:
    python experiments/probe_score_distributions.py --layers 11-18 --margin 1.5
    python experiments/probe_score_distributions.py --layers 11-18 \
        --schedule 1.2,2.0,1.8,1.8,2.0,0.9,1.2 --limit 40
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import discover_available_layers, load_class_representations, load_probes
from utils.runtime import load_model, load_prompts, set_seed

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
HARMLESS_C, HARMFUL_C, STEER_C = "#2a9d5c", "#c0392b", "#2a78d6"


def capture_hook(layer_idx, store, *, steer=False, w=None, b=None, beta=1.0, margin=0.0, eps=1e-12):
    """Record the last-token activation; optionally apply utils.hooks.projection_hook first.

    IMPORTANT: do not use output_hidden_states to read steered activations. In this
    transformers version `hidden_states[l+1]` is layer l's output BEFORE layer l's own
    forward hook runs (it does reflect upstream hooks). Verified with a marker hook: a
    +1000 offset applied at layer 11 never appears in hidden_states[12], even though the
    logits do change. Recording through the hook itself is unambiguous at every layer.
    """
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        if not steer:
            store[layer_idx] = h[:, -1, :].detach().float().cpu()
            return None  # leave the output untouched

        w_l = w.to(device=h.device, dtype=h.dtype)
        b_l = b.to(device=h.device, dtype=h.dtype)
        m_l = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_l * w_l).clamp_min(eps)

        score = (h * w_l.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_l.view(1, 1, 1) + m_l.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_l.view(1, 1, -1))
        store[layer_idx] = h_mod[:, -1, :].detach().float().cpu()
        return replace_hidden(output, h_mod)

    return hook


def score_of(X, w, b):
    """X: (..., D) -> probe scores (...,). Matches training-time w.x + b exactly."""
    return X.float() @ w.float() + b.float()


def summarize(t):
    t = t.float()
    return {"mean": t.mean().item(), "std": t.std(unbiased=True).item(),
            "min": t.min().item(), "max": t.max().item(), "n": t.numel()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--probe_reps_dir", default=None)
    ap.add_argument("--layers", default="11-18", help="Steered window (end-exclusive).")
    ap.add_argument("--margin", type=float, default=1.5, help="Shared margin, if --schedule is absent.")
    ap.add_argument("--schedule", default=None,
                    help="Per-layer margins aligned with --layers, comma-separated.")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out_dir", default="./experiments/results")
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    window = parse_layers_arg(args.layers, n_layers)

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")

    if args.schedule:
        vals = [float(v) for v in args.schedule.replace(" ", "").split(",") if v]
        if len(vals) != len(window):
            raise ValueError(f"--schedule has {len(vals)} values but window is {len(window)} layers: {window}")
        margin_map = dict(zip(window, vals))
    else:
        margin_map = {l: args.margin for l in window}

    # Probes at EVERY available layer: steering is confined to the window, but we want the
    # score trajectory outside it too.
    all_layers = discover_available_layers(args.svm_dir, args.probe_type)
    probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                         layer_indices=all_layers, device=torch.device("cpu"),
                         explicit_reps_dir=args.probe_reps_dir)

    X_harm, X_harmless = load_class_representations(args.probe_reps_dir or args.svm_dir)
    print(f"Train reps: harmful {tuple(X_harm.shape)}  harmless {tuple(X_harmless.shape)}")

    prompts, _ = load_prompts(args)
    print(f"Model: {args.model_name} | window {args.layers} -> {window}")
    print(f"Margins: {margin_map}")
    print(f"Eval prompts: {len(prompts)} from {args.dataset}\n")

    # --- eval-set activations, unsteered and steered, at the last prompt token ---
    clean_acts = {l: [] for l in all_layers}
    steer_acts = {l: [] for l in all_layers}

    for prompt in tqdm(prompts, desc="Forward passes"):
        inputs = model.prepare_inputs(prompt)
        with torch.no_grad():
            out = model.model(input_ids=inputs.input_ids, output_hidden_states=True)
        for l in all_layers:
            clean_acts[l].append(out.hidden_states[l + 1][:, -1, :].detach().float().cpu())

        # Hooks on EVERY layer: steer inside the window, record-only outside. Reading
        # steered values off hidden_states would silently return pre-steering activations
        # at the steered layers (see capture_hook docstring).
        store = {}
        handles = [
            layers[l].register_forward_hook(
                capture_hook(l, store, steer=(l in window), w=probes[l]["w"], b=probes[l]["b"],
                             beta=args.beta, margin=margin_map.get(l, 0.0))
            )
            for l in all_layers
        ]
        try:
            with torch.no_grad():
                model.model(input_ids=inputs.input_ids)
        finally:
            remove_hooks(handles)
        for l in all_layers:
            steer_acts[l].append(store[l])

    clean_acts = {l: torch.cat(v, dim=0) for l, v in clean_acts.items()}
    steer_acts = {l: torch.cat(v, dim=0) for l, v in steer_acts.items()}

    # --- per-layer statistics ---
    rows = []
    for l in all_layers:
        w, b = probes[l]["w"].cpu(), probes[l]["b"].cpu()
        hl = score_of(X_harmless[:, l, :], w, b)
        hf = score_of(X_harm[:, l, :], w, b)
        ev = score_of(clean_acts[l], w, b)

        s_hl, s_hf, s_ev = summarize(hl), summarize(hf), summarize(ev)
        row = {"layer": l, "in_window": l in window, "w_norm": w.norm().item(),
               "harmless": s_hl, "harmful": s_hf, "eval_unsteered": s_ev}

        if l in window:
            row["margin"] = margin_map[l]

        st = score_of(steer_acts[l], w, b)
        row["steered"] = summarize(st)
        # z of the steered score against the harmless training distribution.
        row["z_vs_harmless"] = (st.mean().item() - s_hl["mean"]) / max(s_hl["std"], 1e-9)

        # Full-space geometry, all relative to the harmless centroid.
        mu_hl = X_harmless[:, l, :].mean(dim=0)
        mu_hf = X_harm[:, l, :].mean(dim=0)
        w_hat = w / w.norm().clamp_min(1e-12)

        for name, acts in (("clean", clean_acts[l]), ("steered", steer_acts[l])):
            d = acts - mu_hl.unsqueeze(0)
            along = (d @ w_hat).abs()
            orth = (d - (d @ w_hat).unsqueeze(1) * w_hat.unsqueeze(0)).norm(dim=1)
            row[f"dist_{name}"] = {
                "total": d.norm(dim=1).mean().item(),
                "along_w": along.mean().item(),
                "orthogonal": orth.mean().item(),
            }
        row["class_separation"] = (mu_hf - mu_hl).norm().item()
        # How far the harmless cloud itself sits from its own centroid, as the
        # natural yardstick for "near".
        row["harmless_spread"] = (X_harmless[:, l, :] - mu_hl.unsqueeze(0)).norm(dim=1).mean().item()
        rows.append(row)

    # --- report ---
    print("\n" + "=" * 118)
    print("PROBE SCORES  (w.h + b at the last prompt token; probe fit on these same units)")
    print("=" * 118)
    hdr = (f"{'layer':>6}{'win':>5}{'m':>6}{'harmless':>18}{'harmful':>18}"
           f"{'eval unsteered':>18}{'steered':>16}{'z vs HL':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        hl, hf, ev = r["harmless"], r["harmful"], r["eval_unsteered"]
        if r["in_window"]:
            line = f"{r['layer']:>6}{'*':>5}{r['margin']:>6.2f}"
        else:
            line = f"{r['layer']:>6}{'':>5}{'':>6}"
        st = r["steered"]
        line += (f"{hl['mean']:>10.2f}+-{hl['std']:<5.2f}"
                 f"{hf['mean']:>10.2f}+-{hf['std']:<5.2f}"
                 f"{ev['mean']:>10.2f}+-{ev['std']:<5.2f}"
                 f"{st['mean']:>9.3f}+-{st['std']:<5.3f}{r['z_vs_harmless']:>9.2f}")
        print(line)

    print("\n" + "=" * 118)
    print("FULL-SPACE GEOMETRY  (mean distance to the HARMLESS centroid; * = steered layer)")
    print("=" * 118)
    hdr2 = (f"{'layer':>6}{'m':>6}{'||mu_HF-mu_HL||':>17}{'HL spread':>11}"
            f"{'clean dist':>12}{'steer dist':>12}{'steer along w':>15}{'steer orth':>12}{'orth frac':>11}")
    print(hdr2); print("-" * len(hdr2))
    for r in rows:
        c, s = r["dist_clean"], r["dist_steered"]
        frac = s["orthogonal"] / max(s["total"], 1e-9)
        mtxt = f"{r['margin']:>6.2f}" if r["in_window"] else f"{'':>6}"
        print(f"{r['layer']:>6}{mtxt}{r['class_separation']:>17.2f}"
              f"{r['harmless_spread']:>11.2f}{c['total']:>12.2f}{s['total']:>12.2f}"
              f"{s['along_w']:>15.2f}{s['orthogonal']:>12.2f}{frac:>11.3f}")
    print("\n  'steer orth' is the part of the displacement the projection CANNOT touch.")
    print("  orth frac -> 1 means the steered activation matches the probe score while")
    print("  remaining as far from the harmless cloud as it ever was.")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model_name}_layers{args.layers.replace('-', 'to')}"
    if args.out_tag:
        tag += f"_{args.out_tag}"
    path = os.path.join(args.out_dir, f"probe_score_distributions_{tag}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model_name, "window": window, "margins": margin_map,
                   "dataset": args.dataset, "n_eval": len(prompts), "rows": rows}, f, indent=2)
    print(f"\nSaved data to {path}")

    plot(rows, window, all_layers, os.path.join(args.out_dir, f"probe_score_distributions_{tag}.png"),
         args.model_name, args.layers)


def plot(rows, window, all_layers, out_path, model_name, layers_arg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xlabel("layer", color=INK_MUTED, fontsize=10)
        ax.axvspan(min(window), max(window), color="#2a78d6", alpha=0.07, zorder=1)

    xs = [r["layer"] for r in rows]
    for key, color, label in (("harmless", HARMLESS_C, "harmless (train)"),
                              ("harmful", HARMFUL_C, "harmful (train)"),
                              ("eval_unsteered", "#8a8981", "eval prompts, unsteered")):
        mu = [r[key]["mean"] for r in rows]
        sd = [r[key]["std"] for r in rows]
        axes[0].plot(xs, mu, color=color, linewidth=2, label=label, zorder=3)
        axes[0].fill_between(xs, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)],
                             color=color, alpha=0.15, zorder=2)

    sx = [r["layer"] for r in rows if r["in_window"]]
    sy = [r["steered"]["mean"] for r in rows if r["in_window"]]
    axes[0].plot(sx, sy, color=STEER_C, linewidth=2.4, marker="o", markersize=5,
                 label="steered (= -m_l)", zorder=4)
    axes[0].axhline(0, color="#8a8981", linewidth=1.2, linestyle="--", zorder=2)
    axes[0].set_title("Probe score  w.h + b   (mean +- 1 sd)", color=INK, fontsize=11, loc="left", pad=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)

    total = [r["dist_steered"]["total"] for r in rows if r["in_window"]]
    orth = [r["dist_steered"]["orthogonal"] for r in rows if r["in_window"]]
    clean = [r["dist_clean"]["total"] for r in rows if r["in_window"]]
    spread = [r["harmless_spread"] for r in rows if r["in_window"]]
    axes[1].plot(sx, clean, color=HARMFUL_C, linewidth=2, marker="o", markersize=5,
                 label="unsteered distance to harmless centroid", zorder=3)
    axes[1].plot(sx, total, color=STEER_C, linewidth=2.4, marker="o", markersize=5,
                 label="steered distance", zorder=4)
    axes[1].plot(sx, orth, color=STEER_C, linewidth=1.6, linestyle=":", marker=None,
                 label="steered, orthogonal component only", zorder=4)
    axes[1].plot(sx, spread, color=HARMLESS_C, linewidth=2, linestyle="--",
                 label="harmless cloud's own spread", zorder=3)
    axes[1].set_title("Distance to the harmless centroid (full space)",
                      color=INK, fontsize=11, loc="left", pad=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle(f"{model_name}  ·  steered window {layers_arg}  ·  last prompt token",
                 color=INK, fontsize=12, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
