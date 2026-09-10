"""Line plot + table: probe direction vs. gradients, across layers.

Five series (x = layer, y = cosine with the probe weight w):
  1. cos(diff-in-means, w)                         -- detection axis reference (~0.85)
  2. cos(grad logP(refusal | harmful),  w)
  3. cos(grad logP(refusal | harmless), w)
  4. cos(grad logP(refusal | avg),      w)         -- pooled harmful+harmless (equal n -> mean of 2,3)
  5. cos(grad [logP(refuse) - logP(comply)] | avg, w)  -- log-odds target, pooled

Reads the committed aggregate JSONs written by refusal_gradient_alignment.py. No model, no GPU.

Usage: python experiments/plot_refusal_gradient_lines.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results", "refusal_gradient")
LAYERS = [11, 12, 13, 14, 15, 16, 17]


def load(ds):
    return json.load(open(os.path.join(OUT, f"alignment_{ds}_11to18.json")))


def col(d, key):  # per-layer mean for a cosine key
    return [d["per_layer"][str(l)][key]["mean"] for l in LAYERS]


def std(d, key):
    return [d["per_layer"][str(l)][key]["std"] for l in LAYERS]


def main():
    H, L = load("harmbench_test"), load("harmless_val")
    dmean = [H["per_layer"][str(l)]["w_dmean"] for l in LAYERS]
    gR_H, gR_L = col(H, "gR_w"), col(L, "gR_w")
    gRD_H, gRD_L = col(H, "gRD_w"), col(L, "gRD_w")
    gR_avg = [(a + b) / 2 for a, b in zip(gR_H, gR_L)]
    gRD_avg = [(a + b) / 2 for a, b in zip(gRD_H, gRD_L)]
    base = H["random_baseline_abs_cos"]
    # (b) pooled mean-gradient consensus (unit-averaged over all 318), from pool_consensus.py
    pooled = json.load(open(os.path.join(OUT, "pooled_consensus.json")))
    gR_consensus = pooled["refusal"]["b_unit"]

    # ---- table ----
    sH, sL = std(H, "gR_w"), std(L, "gR_w")
    print(f"{'L':>3} {'dmean·w':>8} {'gR|harm':>13} {'gR|less':>13} {'gR|avg':>8} "
          f"{'gRD|harm':>9} {'gRD|less':>9} {'gRD|avg':>8}")
    for i, l in enumerate(LAYERS):
        print(f"{l:>3} {dmean[i]:>8.3f} {gR_H[i]:+.3f}±{sH[i]:.3f} {gR_L[i]:+.3f}±{sL[i]:.3f} "
              f"{gR_avg[i]:>+8.3f} {gRD_H[i]:>+9.3f} {gRD_L[i]:>+9.3f} {gRD_avg[i]:>+8.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=120)
    ax.axhspan(-base, base, color="#7d8494", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color="#c6ccd6", lw=1, zorder=1)
    series = [
        (dmean,   "cos(diff-in-means, w)",                       "#2a78d6", "-",  "o"),
        (gR_H,    "cos(∇logP(refusal | harmful), w)",            "#eb6834", "-",  "o"),
        (gR_L,    "cos(∇logP(refusal | harmless), w)",           "#1baf7a", "-",  "o"),
        (gR_avg,  "cos(∇logP(refusal | avg), w)  — per-prompt mean (a)", "#555555", "--", "s"),
        (gR_consensus, "cos(∇logP(refusal | avg), w)  — mean gradient (b)", "#4a3aa7", "-", "D"),
        (gRD_avg, "cos(∇[logP(refuse)−logP(comply)] | avg, w)",  "#e87ba4", "-",  "^"),
    ]
    for ys, label, c, ls, mk in series:
        ax.plot(LAYERS, ys, ls, color=c, marker=mk, ms=5, lw=2.2, label=label,
                mec="white", mew=0.8)
    ax.set_xlabel("layer (decoder block, residual-stream output)")
    ax.set_ylabel("cosine similarity with probe weight w")
    ax.set_ylim(-0.1, 0.95)
    ax.set_xticks(LAYERS)
    ax.set_title("Llama-3.1-8B: probe direction vs. gradients", fontsize=13,
                 loc="left", weight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7eaef", lw=1)
    ax.set_axisbelow(True)
    # legend goes in the empty mid-band (gradients near 0, diff-in-means up at 0.85)
    ax.legend(fontsize=9, frameon=True, framealpha=0.92, edgecolor="#e0e0e0",
              loc="center left", bbox_to_anchor=(0.03, 0.52),
              handlelength=2.2, labelspacing=0.4)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"refusal_gradient_lines.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("\nwrote", os.path.relpath(os.path.join(OUT, "refusal_gradient_lines.png"), ROOT), "(+ .pdf)")


if __name__ == "__main__":
    main()
