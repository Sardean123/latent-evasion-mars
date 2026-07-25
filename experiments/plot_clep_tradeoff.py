"""Combined CLE-P margin tradeoff: jailbreak effectiveness vs coherence retention.
Left  = HarmBench ASR (higher = better jailbreak), Llama-2-13b-cls judge, harmbench_standard.
Right = TruthfulQA MC2 (higher = better coherence/capability retention), steered MC scoring.
Same three conditions (baseline / paper-BO / hlmean). Two separate panels -- no dual axis.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR = os.path.join(ROOT, "experiments/results/harmbench_clep_margins/asr_summary.json")
TQA = os.path.join(ROOT, "experiments/results/truthfulqa_mc/truthfulqa_mc_llama3-8b_layers11to18.json")
OUTDIR = os.path.join(ROOT, "experiments/results/clep_margins_summary")

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {"baseline": "#2a78d6", "paper": "#eb6834", "hlmean": "#1baf7a"}
XLABEL = {"baseline": "baseline", "paper": "paper BO\n(mean 1.56)", "hlmean": "hlmean\n(mean 1.11)"}
ORDER = ["baseline", "paper", "hlmean"]


def rbar(ax, x, h, w, color):
    if h <= 0:
        ax.add_patch(plt.Rectangle((x - w / 2, 0), w, 0.004, color=color, lw=0)); return
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(0.02, h/2)}",
        fc=color, ec=SURFACE, lw=1.2, zorder=3))


def style(ax, top=1.0):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([XLABEL[k] for k in ORDER], fontsize=9, color=INK_MUTED)
    ax.set_ylim(0, top)


def panel(ax, vals, title, note_delta=None):
    style(ax)
    ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    for i, k in enumerate(ORDER):
        rbar(ax, i, vals[k], 0.6, COLORS[k])
        lbl = f"{vals[k]*100:.1f}%"
        if note_delta and k != "baseline":
            lbl += f"\n({(vals[k]-vals['baseline'])*100:+.1f})"
        ax.text(i, vals[k] + 0.02, lbl, ha="center", va="bottom", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=8)


def main():
    asr = json.load(open(ASR))
    tqa = json.load(open(TQA))["conditions"]
    asr_v = {k: asr[k]["asr"] for k in ORDER}
    mc2_v = {k: tqa[k]["mc2"] for k in ORDER}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE, layout="constrained")
    panel(axL, asr_v, "Jailbreak effectiveness  ↑ better")
    axL.set_ylabel("HarmBench ASR  (Llama-2-13b judge)", color=INK_MUTED, fontsize=10)
    panel(axR, mc2_v, "Coherence retention  ↑ better  ·  Δ vs baseline", note_delta=True)
    axR.set_ylabel("TruthfulQA MC2", color=INK_MUTED, fontsize=10)

    fig.suptitle("CLE-P on llama3-8b  ·  BO-optimized vs harmless-mean margins  ·  window 11-18\n"
                 "hlmean matches paper-BO on jailbreak while retaining ~2x more coherence",
                 color=INK, fontsize=13, x=0.01, ha="left")
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "asr_vs_coherence.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    main()
