"""Bar chart of HarmBench ASR (Llama-2-13b-cls judge) for CLE-P on harmbench_standard (n=200):
unsteered baseline vs the CLE paper's BO-optimized per-layer margins vs the BO-free
harmless-mean margins. Left panel: overall ASR. Right panel: per-category ASR.

Reads the ASR summary written by the eval step and renders to
experiments/results/harmbench_clep_margins/harmbench_asr_clep_margins.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "experiments/results/harmbench_clep_margins")

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Reference categorical palette (fixed order): baseline / paper / hlmean
COLORS = {"baseline": "#2a78d6", "paper": "#eb6834", "hlmean": "#1baf7a"}
LABELS = {"baseline": "baseline (unsteered)", "paper": "paper BO  (mean 1.56)", "hlmean": "hlmean  (mean 1.11)"}
ORDER = ["baseline", "paper", "hlmean"]
CAT_LABEL = {
    "chemical_biological": "chemical /\nbiological", "cybercrime_intrusion": "cybercrime /\nintrusion",
    "illegal": "illegal", "misinformation_disinformation": "misinfo /\ndisinfo",
    "harmful": "harmful", "harassment_bullying": "harassment /\nbullying",
}


def rounded_bars(ax, xs, heights, width, color, radius=0.045):
    """Thin bars with 4px-style rounded top ends anchored to the baseline."""
    for x, h in zip(xs, heights):
        if h <= 0:
            ax.add_patch(plt.Rectangle((x - width / 2, 0), width, 0.006, color=color, lw=0))
            continue
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, 0), width, h,
            boxstyle=f"round,pad=0,rounding_size={min(radius, h/2)}",
            mutation_aspect=1, fc=color, ec=SURFACE, lw=1.2, zorder=3, clip_on=False))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])


def main():
    summ = json.load(open(os.path.join(OUTDIR, "asr_summary.json")))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1, 2.3]})

    # --- Left: overall ASR ---
    style(axL)
    for i, k in enumerate(ORDER):
        rounded_bars(axL, [i], [summ[k]["asr"]], 0.62, COLORS[k])
        axL.text(i, summ[k]["asr"] + 0.02, f"{summ[k]['asr']*100:.1f}%",
                 ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    axL.set_xticks(range(len(ORDER)))
    axL.set_xticklabels([LABELS[k].split("  ")[0] for k in ORDER], fontsize=9, color=INK_MUTED)
    axL.set_title("Overall ASR", color=INK, fontsize=12, loc="left", pad=8)
    axL.set_ylabel("HarmBench ASR", color=INK_MUTED, fontsize=10)

    # --- Right: per-category (sorted by mean of steered runs, desc) ---
    cats = sorted(summ["paper"]["per_category"],
                  key=lambda c: -(summ["paper"]["per_category"][c] + summ["hlmean"]["per_category"][c]) / 2)
    style(axR)
    n, gw = len(ORDER), 0.78
    bw = gw / n
    for i, k in enumerate(ORDER):
        offs = [j - gw / 2 + bw * (i + 0.5) for j in range(len(cats))]
        rounded_bars(axR, offs, [summ[k]["per_category"][c] for c in cats], bw * 0.86, COLORS[k])
    axR.set_xlim(-0.75, len(cats) - 0.25)
    axR.set_yticklabels([])  # left panel carries the shared 0-100% scale; avoid label/bar collision
    axR.set_xticks(range(len(cats)))
    axR.set_xticklabels([CAT_LABEL[c] for c in cats], fontsize=8.5, color=INK_MUTED)
    axR.set_title("ASR by semantic category", color=INK, fontsize=12, loc="left", pad=8)

    # Legend (identity is never color-alone -> legend present for 3 series)
    handles = [plt.Line2D([0], [0], marker="s", markersize=9, linestyle="", markerfacecolor=COLORS[k],
                          markeredgecolor=SURFACE, label=LABELS[k]) for k in ORDER]
    axR.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               loc="upper right", ncol=1, handletextpad=0.4)

    fig.suptitle("CLE-P on HarmBench standard (n=200)  ·  HarmBench Llama-2-13b-cls judge  ·  "
                 "BO-optimized vs harmless-mean margins", color=INK, fontsize=13, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "harmbench_asr_clep_margins.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    main()
