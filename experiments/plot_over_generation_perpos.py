"""Per-token-position panels for a probe_score_over_generation genpos run. One subplot per
position (post-instr pos -1, then each generated position 0..N-1). Each panel shows that
position's STEERED and UNSTEERED lines against the harmful/harmless TRAINING distribution
bands for THAT SAME position's probe:
  * pos -1  : lines on the original prompt-token probe; bands = post-instruction training reps
              (data["reference"], written by probe_score_over_generation.py in genpos mode).
  * pos >=0 : lines on the gen-pos-p probe; bands = harmful/harmless training activations at
              generated position p scored on the pos-p probe (experiments/genpos_training_bands.py).
So every line is read against the correct in-distribution reference for its own probe.

Usage:
    python experiments/plot_over_generation_perpos.py \
        experiments/results/probe_score_over_generation/<..._scoregenpos.json>
Defaults to the CLE-A hlmean run. Output PNG is the input name with _scoregenpos.json -> _PERPOS.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(ROOT, "experiments/results/probe_score_over_generation")

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
HARMLESS_C, HARMFUL_C = "#2a9d5c", "#c0392b"
STEER_C, UNSTEER_C = "#2a78d6", "#e08214"

GENPOS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    RESDIR, "probe_score_over_generation_llama3-8b_layers11to18_hlmean_clea_scoregenpos.json")
BANDS = os.path.join(RESDIR, "genpos_training_bands_llama3-8b.json")
OUT = GENPOS.replace("_scoregenpos.json", "_PERPOS.png")


def band_for(pos, data, bands):
    if pos == -1:
        return {k: data["reference"][k] for k in ("harmless", "harmful")}
    b = bands["bands"].get(str(pos), {})
    out = {"harmless": {}, "harmful": {}}
    for l, d in b.items():
        for cls in ("harmless", "harmful"):
            if cls in d:
                out[cls][l] = d[cls]
    return out


def draw_panel(ax, pos, data, bands, window):
    all_layers = data["layers"]
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.axvspan(min(window), max(window), color="#2a78d6", alpha=0.07, zorder=1)
    ax.axhline(0, color="#8a8981", linewidth=1.0, linestyle="--", zorder=2)

    band = band_for(pos, data, bands)
    for key, color in (("harmless", HARMLESS_C), ("harmful", HARMFUL_C)):
        ref = band[key]
        xs = [l for l in all_layers if str(l) in ref]
        if not xs:
            continue
        mu = [ref[str(l)]["mean"] for l in xs]
        sd = [ref[str(l)]["std"] for l in xs]
        ax.fill_between(xs, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)],
                        color=color, alpha=0.12, zorder=2)
        ax.plot(xs, mu, color=color, linewidth=1.2, alpha=0.55, zorder=3,
                label=f"{key} (train, this pos)")

    for cond, color in (("steered", STEER_C), ("unsteered", UNSTEER_C)):
        posd = data["conditions"][cond].get(str(pos), {})
        xs = [l for l in all_layers if str(l) in posd]
        if not xs:
            continue
        ax.plot(xs, [posd[str(l)]["mean"] for l in xs], color=color, linewidth=2.1, zorder=5, label=cond)

    tag = "post-instruction token (pos -1)" if pos == -1 else f"generated position {pos}"
    probe = "orig prompt-token probe" if pos == -1 else f"gen-pos-{pos} probe"
    ax.set_title(f"{tag}\n{probe}", color=INK, fontsize=10, loc="left", pad=6)


def main():
    data = json.load(open(GENPOS))
    bands = json.load(open(BANDS))
    window = data["window"]
    positions = [-1] + list(range(0, data["max_new_tokens"]))

    ncol = 3
    nrow = (len(positions) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True, facecolor=SURFACE)
    axes = axes.ravel()
    for i, pos in enumerate(positions):
        draw_panel(axes[i], pos, data, bands, window)
        if i % ncol == 0:
            axes[i].set_ylabel("probe score  w·h + b", color=INK_MUTED, fontsize=9)
        if i >= len(positions) - ncol:
            axes[i].set_xlabel("layer", color=INK_MUTED, fontsize=9)
    for j in range(len(positions), len(axes)):
        axes[j].axis("off")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK_MUTED, loc="lower left")

    method = data.get("method", "cle-p").upper()
    fig.suptitle(f"{data['model']}  ·  {method}  ·  hlmean  ·  window {min(window)}-{max(window)+1}  ·  "
                 f"{data['n_eval']} harmful prompts  ·  each position vs its OWN probe's training distribution",
                 color=INK, fontsize=12.5, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
