"""Summary figure: which 'refusal-removal' direction actually moves activations along the probe?

Overlays cos(direction, probe_w) across the target layers for:
  - SFT diff (harmful):   mean(h_SFT - h_base) after LoRA-unlearning refusal        [the payoff]
  - SFT diff (harmless):  same, control LoRA trained on harmless prompts             [drift control]
  - refusal gradient:     mean gradient of logP(refusal) w.r.t. activation           [~orthogonal]
  - activation descent:   the delta that flips the refuse-vs-comply log-odds         [~orthogonal]

Reads the committed result JSONs. Harmless-control line is included only if its JSON exists.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RG = os.path.join(ROOT, "experiments", "results", "refusal_gradient")
LAYERS = [11, 12, 13, 14, 15, 16, 17]


def col(path, layerkey):
    d = json.load(open(path))
    return [layerkey(d["per_layer"][str(l)]) for l in LAYERS]


def main():
    series = []
    series.append((col(os.path.join(RG, "unlearn_diff.json"), lambda r: r["cos_meandiff_w"]),
                   "SFT diff — harmful (unlearned refusal)", "#b5179e", "-", "o"))
    hpath = os.path.join(RG, "unlearn_diff_harmless.json")
    if os.path.exists(hpath):
        series.append((col(hpath, lambda r: r["cos_meandiff_w"]),
                       "SFT diff — harmless (control)", "#1baf7a", "--", "s"))
    series.append((col(os.path.join(RG, "alignment_harmbench_test_11to18.json"),
                       lambda r: r["consensus_gR_w"]),
                   "refusal gradient (mean)", "#eb6834", "-", "^"))
    series.append((col(os.path.join(RG, "descent_harmbench_test_11to18.json"),
                       lambda r: r["consensus_cos_w"]),
                   "activation descent δ", "#8a8f98", "-", "D"))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
    ax.axhline(0, color="#c6ccd6", lw=1, zorder=1)
    for ys, label, c, ls, mk in series:
        ax.plot(LAYERS, ys, ls, color=c, marker=mk, ms=6, lw=2.4, label=label, mec="white", mew=0.8)
    ax.set_xlabel("layer (decoder block, residual-stream output)")
    ax.set_ylabel("cosine with probe weight  w")
    ax.set_ylim(-0.7, 0.2)
    ax.set_xticks(LAYERS)
    ax.set_title("Only weight-level unlearning rides the probe axis\n"
                 "Llama-3-8B · cos(refusal-removal direction, probe w) at last prompt token",
                 fontsize=12, loc="left", weight="bold", pad=10)
    ax.text(13.1, 0.135, "negative = activations move toward harmless / comply (−w)",
            fontsize=8.5, color="#8a909c")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7eaef", lw=1)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=True, framealpha=0.92, edgecolor="#e0e0e0", loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RG, f"direction_comparison.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("wrote", os.path.relpath(os.path.join(RG, "direction_comparison.png"), ROOT),
          f"({len(series)} series)")


if __name__ == "__main__":
    main()
