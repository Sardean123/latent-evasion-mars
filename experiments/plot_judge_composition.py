"""Grouped bar charts of the judge-agreement composition, per prompt set, for the team meeting.

Values are the curated tables (4 methods + ALL; CLE-P BO = bo-external-clep, the fair CLE-P cell).
Seven bars per method group: HB-ASR, SR-mean(x100), Refused, SR>HB, HB>SR, Agree_pos, Agree_neg.
Numbers are hardcoded exactly as reported so the figure matches the tables shown to the team.

Usage: python experiments/plot_judge_composition.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results", "score_composition")

METHODS = ["CLE-A (BO)", "CLE-A (mean)", "CLE-P (BO)", "CLE-P (mean)", "ALL"]
METRICS = ["HB-ASR", "SR-mean", "Refused", "SR>HB", "HB>SR", "Agree_pos", "Agree_neg"]
# dataviz categorical palette, fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#8a8f98"]

# rows aligned to METHODS; columns aligned to METRICS. SR-mean given x100 (mean 0-1 score as %).
STRONGREJECT = [
    [87.9, 67.8, 0.3, 5.4, 13.1, 74.4, 6.7],
    [78.9, 71.2, 7.3, 7.3,  3.2, 72.8, 9.3],
    [88.2, 76.5, 0.0, 5.4,  3.8, 84.3, 6.4],
    [88.5, 77.6, 0.0, 7.0,  4.8, 83.7, 4.5],
    [86.6, 72.9, 1.5, 5.8,  7.4, 78.5, 6.7],
]
HARMBENCH = [
    [91.0, 63.1, 0.5, 3.5, 21.5, 69.0,  5.5],
    [76.5, 62.1, 10.0, 3.0, 9.0, 65.0, 13.0],
    [90.5, 66.9, 0.5, 3.5, 19.5, 70.5,  6.0],
    [88.0, 67.8, 0.0, 3.5, 16.5, 71.5,  8.5],
    [86.1, 63.7, 2.3, 3.4, 18.4, 67.0,  8.9],
]


def chart(data, title, fname):
    data = np.array(data)  # (methods, metrics)
    nm, nb = data.shape
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=120)
    group_w = 0.86
    bw = group_w / nb
    x = np.arange(nm)
    plt.rcParams["hatch.linewidth"] = 0.7
    for j, metric in enumerate(METRICS):
        offs = (j - (nb - 1) / 2) * bw
        hatch = "///" if j < 2 else None  # HB-ASR, SR-mean: aggregate metrics, textured apart
        bars = ax.bar(x + offs, data[:, j], width=bw * 0.94, color=COLORS[j],
                      label=metric, edgecolor="white", linewidth=0.5, hatch=hatch)
        for b in bars:  # compact value labels
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=6, color="#333", rotation=90,
                        xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("percent   (SR-mean = mean 0–1 score ×100)")
    ax.set_title(title, fontsize=13, loc="left", weight="bold", pad=36)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7eaef", lw=1)
    ax.set_axisbelow(True)
    # legend sits just above the plot, beneath the (raised) title
    ax.legend(ncol=7, fontsize=9, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.1, handlelength=1.1)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"{fname}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("wrote", os.path.relpath(os.path.join(OUT, fname + ".png"), ROOT), "(+ .pdf)")
    plt.close(fig)


def main():
    chart(STRONGREJECT, "Judge composition — StrongREJECT prompts (n=313/method)",
          "judge_composition_strongreject")
    chart(HARMBENCH, "Judge composition — HarmBench prompts (n=200/method)",
          "judge_composition_harmbench")


if __name__ == "__main__":
    main()
