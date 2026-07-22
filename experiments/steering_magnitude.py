"""Measure the magnitude of the CLE steering vector per layer.

Answers: how far does the intervention actually move the residual stream at each
layer, in absolute terms and relative to the CLEAN activation norm at that layer?

The recorded transform mirrors `utils.hooks.projection_hook` exactly:

    score = w.h + b + m
    h*    = h - beta * score / ||w||^2 * w        =>  ||dh|| = beta * |score| / ||w||

With beta = 1 this is an orthogonal projection onto the hyperplane {x : w.x + b = -m}:
the post-steering probe score is exactly -m for ANY input, because the input-dependent
term cancels (w.h* + b = (w.h + b) - (w.h + b + m) = -m). The `post_score` column
prints this so the identity is visible rather than implicit.

NORMALIZATION. The residual-stream norm grows with depth on its own — unsteered,
||h|| runs ~5.7 at L11 to ~9.7 at L17 and keeps climbing. That is a property of the
model, not of the steering. Every relative number here is therefore divided by the
CLEAN norm at the SAME layer, captured in a hookless reference pass. Dividing by the
steered norm (as an earlier version did) is wrong: that denominator is itself inflated
by upstream steering, which understates the perturbation and makes margins
incomparable.

Measurement is *in situ* — hooks are live on every layer in the window, so layer
l+1 sees the already-modified output of layer l and the reported numbers include
that compounding. During prefill CLE-P and CLE-A apply the identical transform,
so the prefill row is valid for both variants; they differ only afterwards
(CLE-A freezes the prompt-position delta, CLE-P recomputes it every token).

Decode columns use the clean PREFILL norm per layer as a fixed reference scale.
Clean and steered generations diverge in token content, so decode is a scale
reference, not a matched per-token comparison.

Usage:
    python experiments/steering_magnitude.py --layers 11-18 --margins 0,0.5,1.0,1.5,2.0,2.5
    python experiments/steering_magnitude.py --max_new_tokens 32   # also trace decoding
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes
from utils.runtime import load_model, load_prompts, set_seed, validate_probe_dims

# Ordinal blue ramp, light -> dark (dataviz sequential hue; step 250 is the
# lightest that clears 2:1 on a light surface).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def recording_projection_hook(w, b, beta, margin, layer_idx, store, eps=1e-12):
    """utils.hooks.projection_hook + per-call recording at the last token."""
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))

        h_last = h[:, -1, :].float()
        d_last = (h_mod[:, -1, :].float() - h_last)
        raw = raw_score[:, -1, 0].float().mean().item()
        store.setdefault(layer_idx, []).append({
            "h_norm": h_last.norm(dim=-1).mean().item(),
            "delta_norm": d_last.norm(dim=-1).mean().item(),
            "raw_score": raw,
            "score": score[:, -1, 0].float().mean().item(),
            # Probe score AFTER steering. With beta=1 this is exactly -m for any
            # input: the projection cancels the input-dependent term.
            "post_score": (h_mod[:, -1, :].float() * w_local.float().view(1, -1)).sum(-1).mean().item()
                          + b_local.float().item(),
            "is_prefill": h.shape[1] > 1,
        })
        return replace_hidden(output, h_mod)

    return hook


def clean_reference(model, inputs, n_layers):
    """Hookless forward. Returns per-layer ||h|| at the last prompt token, all layers.

    hidden_states[l+1] is the output of layer l (index 0 is the embedding).
    """
    with torch.no_grad():
        out = model.model(**inputs, output_hidden_states=True)
    return {l: out.hidden_states[l + 1][:, -1, :].float().norm(dim=-1).mean().item()
            for l in range(n_layers)}


def measure(model, layers, selected, probes, margin, beta, inputs, max_new_tokens, clean_norms):
    store = {}
    handles = [
        layers[l].register_forward_hook(
            recording_projection_hook(probes[l]["w"], probes[l]["b"], beta, margin, l, store)
        )
        for l in selected
    ]
    try:
        with torch.no_grad():
            # Steered prefill with all-layer hidden states, so norm inflation is
            # available downstream of the intervened window too.
            steered = model.model(**inputs, output_hidden_states=True)
            steered_norms = {
                l: steered.hidden_states[l + 1][:, -1, :].float().norm(dim=-1).mean().item()
                for l in range(len(layers))
            }
            if max_new_tokens > 0:
                store.clear()  # keep only the generate() trace for the decode columns
                model.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        remove_hooks(handles)

    rows = {}
    for l in selected:
        calls = store.get(l, [])
        pre = [c for c in calls if c["is_prefill"]]
        dec = [c for c in calls if not c["is_prefill"]]
        clean = max(clean_norms[l], 1e-9)
        row = {"layer": l, "n_decode_steps": len(dec), "clean_h_norm": clean_norms[l]}
        if pre:
            p = pre[0]
            row.update(
                prefill_h_norm=p["h_norm"],
                prefill_delta_norm=p["delta_norm"],
                prefill_rel=p["delta_norm"] / clean,
                prefill_raw_score=p["raw_score"],
                prefill_score=p["score"],
                prefill_post_score=p["post_score"],
            )
        if dec:
            row.update(
                decode_delta_norm=sum(c["delta_norm"] for c in dec) / len(dec),
                decode_rel=sum(c["delta_norm"] for c in dec) / len(dec) / clean,
                decode_raw_score=sum(c["raw_score"] for c in dec) / len(dec),
            )
        rows[l] = row
    return {
        "layers": rows,
        "steered_norms": steered_norms,
        "inflation": {l: steered_norms[l] / max(clean_norms[l], 1e-9) for l in range(len(layers))},
    }
def plot(results, selected, clean_norms, n_layers, out_path, model_name, layers_arg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    margins = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d2")
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xlabel("layer", color=INK_MUTED, fontsize=10)

    all_layers = list(range(n_layers))

    # Panel 1 reference: the unsteered norm profile, so the natural depth trend is
    # visible in the same units as the steering magnitude plotted beside it.
    axes[0].plot(selected, [clean_norms[l] for l in selected],
                 color="#8a8981", linewidth=1.8, linestyle=":", marker=None,
                 label="‖h‖ clean (no steering)", zorder=2)

    for i, m in enumerate(margins):
        c = RAMP[i % len(RAMP)]
        rows = results[m]["layers"]
        xs = [l for l in selected if "prefill_delta_norm" in rows[l]]
        axes[0].plot(xs, [rows[l]["prefill_delta_norm"] for l in xs],
                     color=c, linewidth=2, marker="o", markersize=5, label=f"m={m}", zorder=3)
        axes[1].plot(xs, [100 * rows[l]["prefill_rel"] for l in xs],
                     color=c, linewidth=2, marker="o", markersize=5, label=f"m={m}", zorder=3)
        infl = results[m]["inflation"]
        axes[2].plot(all_layers, [infl[l] for l in all_layers],
                     color=c, linewidth=2, zorder=3, label=f"m={m}")

    axes[2].axhline(1.0, color="#8a8981", linewidth=1.2, linestyle="--", zorder=2)
    axes[2].axvspan(min(selected), max(selected), color="#2a78d6", alpha=0.07, zorder=1)
    axes[2].annotate("intervened window", xy=(max(selected) + 0.6, axes[2].get_ylim()[1]),
                     color=INK_MUTED, fontsize=8, va="top")

    axes[0].set_title("Steering magnitude  ‖Δh‖", color=INK, fontsize=11, loc="left", pad=10)
    axes[1].set_title("Relative to CLEAN norm at that layer  ‖Δh‖ / ‖h_clean‖",
                      color=INK, fontsize=11, loc="left", pad=10)
    axes[2].set_title("Norm inflation  ‖h_steered‖ / ‖h_clean‖  (all layers)",
                      color=INK, fontsize=11, loc="left", pad=10)
    axes[1].set_ylabel("%", color=INK_MUTED, fontsize=10)
    axes[2].set_ylabel("×", color=INK_MUTED, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle(
        f"{model_name}  ·  intervened layers {layers_arg}  ·  last prompt token (prefill)  ·  "
        "all relative measures use the clean per-layer norm",
        color=INK, fontsize=12, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"\nSaved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--probe_reps_dir", default=None)
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--margins", default="0,0.5,1.0,1.5,2.0,2.5")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=0,
                    help="0 = prefill only. >0 also traces decoding (CLE-P compounding).")
    ap.add_argument("--out_dir", default="./experiments/results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    margins = [float(x) for x in args.margins.replace(" ", "").split(",") if x]

    model = load_model(args)
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                         layer_indices=selected, device=torch.device(args.device),
                         explicit_reps_dir=args.probe_reps_dir)
    validate_probe_dims(probes, selected, model.model.config.hidden_size)

    prompts, _ = load_prompts(args)
    prompt = prompts[0]
    inputs = model.prepare_inputs(prompt)

    n_layers = len(layers)
    clean_norms = clean_reference(model, inputs, n_layers)

    print(f"Model:  {args.model_name}   layers {args.layers} -> {selected}")
    print(f"Prompt: {prompt[:90]}...")
    print(f"Mode:   {'prefill only' if args.max_new_tokens == 0 else f'prefill + {args.max_new_tokens} decode steps'}")
    print("Clean ||h|| by layer (unsteered; grows with depth by itself): "
          + "  ".join(f"L{l}={clean_norms[l]:.2f}" for l in selected))

    results = {}
    for m in margins:
        results[m] = measure(model, layers, selected, probes, m, args.beta, inputs,
                             args.max_new_tokens, clean_norms)

    for m in margins:
        print(f"\n=== margin {m} ===")
        hdr = (f"{'layer':>6}{'||w||':>8}{'clean|h|':>10}{'steer|h|':>10}{'infl':>7}"
               f"{'w.h+b':>9}{'post':>8}{'||dh||':>9}{'rel%':>8}")
        if args.max_new_tokens > 0:
            hdr += f"{'dec||dh||':>11}{'dec rel%':>10}"
        print(hdr)
        print("-" * len(hdr))
        for l in selected:
            r = results[m]["layers"][l]
            wn = probes[l]["w"].float().norm().item()
            line = (f"{l:>6}{wn:>8.3f}{clean_norms[l]:>10.2f}"
                    f"{results[m]['steered_norms'][l]:>10.2f}{results[m]['inflation'][l]:>7.2f}"
                    f"{r['prefill_raw_score']:>9.3f}{r['prefill_post_score']:>8.3f}"
                    f"{r['prefill_delta_norm']:>9.3f}{100*r['prefill_rel']:>7.1f}%")
            if args.max_new_tokens > 0 and "decode_delta_norm" in r:
                line += f"{r['decode_delta_norm']:>11.3f}{100*r['decode_rel']:>9.1f}%"
            print(line)
        print(f"  ('post' = probe score after steering; equals -m exactly by construction. "
              f"'rel%' and 'infl' are relative to the clean norm at that same layer.)")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model_name}_layers{args.layers.replace('-', 'to')}"
    json_path = os.path.join(args.out_dir, f"steering_magnitude_{tag}.json")
    with open(json_path, "w") as f:
        json.dump({"prompt": prompt, "layers": selected, "beta": args.beta,
                   "max_new_tokens": args.max_new_tokens,
                   "clean_h_norm_by_layer": clean_norms,
                   "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nSaved data to {json_path}")

    plot(results, selected, clean_norms, n_layers,
         os.path.join(args.out_dir, f"steering_magnitude_{tag}.png"),
         args.model_name, args.layers)


if __name__ == "__main__":
    main()
