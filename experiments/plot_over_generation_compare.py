"""Side-by-side comparison of two CLE-A over-generation figures:
  top row    = OFF-DISTRIBUTION scoring (generated tokens scored on the original prompt-token probe)
  bottom row = GENPOS scoring        (generated tokens scored on their own per-position probe)
Both rows share the same red/green training prompt-token bands and the grey post-instr (pos -1)
line (both on the original prompt-token probe). Only the generated-position lines differ.

Usage:
    python experiments/plot_over_generation_compare.py   # uses the CLE-A hlmean pair by default
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(ROOT, "experiments/results/probe_score_over_generation")

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
HARMLESS_C, HARMFUL_C = "#2a9d5c", "#c0392b"
PROMPT_C = "#8a8981"

OFFDIST = os.path.join(RESDIR, "probe_score_over_generation_llama3-8b_layers11to18_hlmean_clea.json")
GENPOS = os.path.join(RESDIR, "probe_score_over_generation_llama3-8b_layers11to18_hlmean_clea_scoregenpos.json")
OUT = os.path.join(RESDIR, "probe_score_over_generation_llama3-8b_layers11to18_hlmean_clea_COMPARE.png")
PLOT_POS = list(range(0, 6))  # generated positions 0..5


def draw(ax, data, cond):
    all_layers = data["layers"]
    window = data["window"]
    cmap = plt.get_cmap("viridis")

    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.axvspan(min(window), max(window), color="#2a78d6", alpha=0.07, zorder=1)
    ax.axhline(0, color="#8a8981", linewidth=1.2, linestyle="--", zorder=2)

    for key, color in (("harmless", HARMLESS_C), ("harmful", HARMFUL_C)):
        ref = data["reference"][key]
        xs = [l for l in all_layers if str(l) in ref]
        mu = [ref[str(l)]["mean"] for l in xs]
        sd = [ref[str(l)]["std"] for l in xs]
        ax.fill_between(xs, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)],
                        color=color, alpha=0.10, zorder=2)
        ax.plot(xs, mu, color=color, linewidth=1.2, alpha=0.5, zorder=3, label=f"{key} (train, prompt tok)")

    prompt = data["conditions"][cond].get("-1", {})
    if prompt:
        xs = [l for l in all_layers if str(l) in prompt]
        ax.plot(xs, [prompt[str(l)]["mean"] for l in xs], color=PROMPT_C, linewidth=1.6,
                linestyle="--", zorder=4, label="post-instr tok (orig probe)")

    draw_pos = [p for p in PLOT_POS if str(p) in data["conditions"][cond]]
    for i, p in enumerate(draw_pos):
        frac = i / max(len(draw_pos) - 1, 1)
        col = cmap(0.15 + 0.7 * frac)
        posd = data["conditions"][cond][str(p)]
        xs = [l for l in all_layers if str(l) in posd]
        ax.plot(xs, [posd[str(l)]["mean"] for l in xs], color=col, linewidth=2, zorder=5, label=f"gen pos {p}")


def main():
    off = json.load(open(OFFDIST))
    gen = json.load(open(GENPOS))
    assert gen["reference"]["harmless"], "genpos JSON is missing reference bands -- re-run the experiment first"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True, facecolor=SURFACE)
    rows = [("OFF-DISTRIBUTION\ngen tok on prompt-token probe", off),
            ("FAIR  ·  gen-pos scoring\ngen tok on its own probe", gen)]
    for r, (row_tag, data) in enumerate(rows):
        for c, cond in enumerate(("steered", "unsteered")):
            ax = axes[r][c]
            draw(ax, data, cond)
            if r == 0:
                ax.set_title({"steered": "STEERED generation", "unsteered": "UNSTEERED generation"}[cond],
                             color=INK, fontsize=12, loc="left", pad=10)
            if c == 0:
                ax.set_ylabel("probe score  w·h + b", color=INK_MUTED, fontsize=10)
                ax.text(0.015, 0.04, row_tag, transform=ax.transAxes, color=INK, fontsize=10.5,
                        fontweight="bold", va="bottom", ha="left", linespacing=1.3,
                        bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="#d8d7d3", alpha=0.9))
            if r == 1:
                ax.set_xlabel("layer", color=INK_MUTED, fontsize=10)
    axes[0][1].legend(frameon=False, fontsize=7.5, labelcolor=INK_MUTED, ncol=2)

    win = off["window"]
    fig.suptitle(f"{off['model']}  ·  CLE-A  ·  probe score across generated tokens  ·  window "
                 f"{min(win)}-{max(win)+1}  ·  hlmean margins  ·  {off['n_eval']} harmful prompts\n"
                 f"top: generated tokens judged by the prompt-token probe (unfair)   ·   "
                 f"bottom: each generated position judged by its own probe (fair)",
                 color=INK, fontsize=12.5, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
