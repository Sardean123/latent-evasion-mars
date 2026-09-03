"""Layerwise cosine of the CLE probe direction w against difference-in-means and the refusal
gradient. Reads the aggregate JSON written by refusal_gradient_alignment.py and saves PNG + PDF.

Usage:
    python experiments/plot_refusal_alignment.py
    python experiments/plot_refusal_alignment.py --json <path> --out_tag harmbench_test
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(
        ROOT, "experiments/results/refusal_gradient/alignment_harmbench_test_11to18.json"))
    ap.add_argument("--out_tag", default=None)
    args = ap.parse_args()

    d = json.load(open(args.json))
    layers = sorted(int(l) for l in d["per_layer"])
    dmean = [d["per_layer"][str(l)]["w_dmean"] for l in layers]
    grad = [d["per_layer"][str(l)]["gR_w"]["mean"] for l in layers]
    gstd = [d["per_layer"][str(l)]["gR_w"]["std"] for l in layers]
    base = d["random_baseline_abs_cos"]
    n = d["n_prompts"]
    tag = args.out_tag or d["config"]["dataset"]

    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=120)

    # noise band + zero
    ax.axhspan(-base, base, color="#7d8494", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color="#c6ccd6", lw=1, zorder=1)
    ax.axhline(base, color="#aab1bd", lw=1, ls=(0, (4, 4)), zorder=1)
    ax.text((layers[0] + layers[-1]) / 2, -0.055, "random baseline |cos|≈%.3f" % base,
            va="center", ha="center", fontsize=8, color="#8a909c")

    # orange std band + lines
    lo = [g - s for g, s in zip(grad, gstd)]
    hi = [g + s for g, s in zip(grad, gstd)]
    ax.fill_between(layers, lo, hi, color=ORANGE, alpha=0.15, lw=0, zorder=2)
    ax.plot(layers, dmean, "-o", color=BLUE, lw=2.2, ms=5, mec="white", mew=1.2,
            label="cos(w, difference-in-means)", zorder=4)
    ax.plot(layers, grad, "-o", color=ORANGE, lw=2.2, ms=5, mec="white", mew=1.2,
            label="cos(w, ∇ log P(refusal))", zorder=5)

    ax.set_ylim(-0.10, 1.00)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("decoder block (residual-stream output)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Two refusal directions vs. the refusal gradient\n"
                 f"Llama-3-8B · layers {layers[0]}–{layers[-1]} · n={n} harmful prompts",
                 fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e7eaef", lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()

    out_dir = os.path.join(ROOT, "experiments/results/refusal_gradient")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"refusal_dir_alignment_{tag}.png")
    pdf = os.path.join(out_dir, f"refusal_dir_alignment_{tag}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", os.path.relpath(png, ROOT))
    print("wrote", os.path.relpath(pdf, ROOT))


if __name__ == "__main__":
    main()
