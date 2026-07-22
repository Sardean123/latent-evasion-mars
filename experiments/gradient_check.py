"""Which CLE gradients are actually non-zero? Let autograd decide.

Settles by measurement, not algebra, which (parameter, loss) pairs carry usable
gradient signal. Parameters: the margin `m` and the probe direction `w`. Losses:

  L1  probe score at an INTERVENED layer, after steering
  L2  probe score at a DOWNSTREAM, un-intervened layer
  L3  behavioural: log p(affirmative first token)

Prediction under the projection identity (beta=1):
    w.h* + b = (w.h + b) - (w.h + b + m) = -m
so L1 is identically -m as a function of BOTH w and m. Therefore
dL1/dm = -1 exactly and dL1/dw = 0 exactly, for any input. L2 and L3 should be
non-zero for both parameters.

Everything is computed in fp32 inside the hook so "exactly zero" is not confounded
by fp16 round-off.

Usage:
    python experiments/gradient_check.py --layers 11-18 --margin 1.5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes
from utils.runtime import load_model, load_prompts, set_seed


def differentiable_projection_hook(w, b, m, capture=None, eps=1e-12):
    """utils.hooks.projection_hook, in fp32, differentiable w.r.t. w and m."""
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        h32 = h.float()
        w_norm_sq = torch.sum(w * w).clamp_min(eps)
        raw = (h32 * w.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b
        score = raw + m
        h_mod = h32 - (score / w_norm_sq) * w.view(1, 1, -1)
        if capture is not None:
            capture["h_star"] = h_mod
        return replace_hidden(output, h_mod.to(h.dtype))
    return hook


def run(model, layers, selected, probes, m, w_params, capture=None):
    handles = [
        layers[l].register_forward_hook(
            differentiable_projection_hook(
                w_params[l], probes[l]["b"].float(), m,
                capture=capture if l == selected[0] else None,
            )
        )
        for l in selected
    ]
    return handles


def grads_for(loss, m, w0):
    g = torch.autograd.grad(loss, [m, w0], retain_graph=False, allow_unused=True)
    gm = 0.0 if g[0] is None else g[0].item()
    gw = 0.0 if g[1] is None else g[1].norm().item()
    return gm, gw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--readout_layer", type=int, default=25,
                    help="Downstream un-intervened layer for L2.")
    ap.add_argument("--affirmative", default="Sure")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    all_needed = sorted(set(selected + [args.readout_layer]))
    probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                         layer_indices=all_needed, device=device, explicit_reps_dir=None)

    if args.readout_layer in selected:
        print(f"WARNING: readout layer {args.readout_layer} is INSIDE the intervened window "
              f"{selected} — L2 will be clamped too and is expected to be degenerate.")

    prompts, _ = load_prompts(args)
    inputs = model.prepare_inputs(prompts[0])
    first = selected[0]

    print(f"Model {args.model_name} | intervened {selected} | margin {args.margin} "
          f"| downstream readout L{args.readout_layer}")
    print(f"Prompt: {prompts[0][:80]}...\n")

    results = []

    # ---- L1: probe score at the FIRST intervened layer, after steering ----
    m = torch.tensor(args.margin, device=device, dtype=torch.float32, requires_grad=True)
    w_params = {l: probes[l]["w"].float().clone().requires_grad_(True) for l in selected}
    cap = {}
    handles = run(model, layers, selected, probes, m, w_params, capture=cap)
    try:
        model.model(**inputs)
    finally:
        remove_hooks(handles)
    w0, b0 = w_params[first], probes[first]["b"].float()
    L1 = (cap["h_star"][0, -1, :] * w0).sum() + b0
    gm, gw = grads_for(L1, m, w0)
    results.append(("L1  probe score @ intervened L%d" % first, L1.item(), gm, gw))

    # ---- L2: probe score at a downstream un-intervened layer ----
    m = torch.tensor(args.margin, device=device, dtype=torch.float32, requires_grad=True)
    w_params = {l: probes[l]["w"].float().clone().requires_grad_(True) for l in selected}
    handles = run(model, layers, selected, probes, m, w_params)
    try:
        out = model.model(**inputs, output_hidden_states=True)
    finally:
        remove_hooks(handles)
    h_read = out.hidden_states[args.readout_layer + 1][0, -1, :].float()
    L2 = (h_read * probes[args.readout_layer]["w"].float()).sum() + probes[args.readout_layer]["b"].float()
    gm, gw = grads_for(L2, m, w_params[first])
    results.append(("L2  probe score @ downstream L%d" % args.readout_layer, L2.item(), gm, gw))

    # ---- L3: behavioural — log p(affirmative first token) ----
    tok_id = model.tokenizer.encode(args.affirmative, add_special_tokens=False)[0]
    m = torch.tensor(args.margin, device=device, dtype=torch.float32, requires_grad=True)
    w_params = {l: probes[l]["w"].float().clone().requires_grad_(True) for l in selected}
    handles = run(model, layers, selected, probes, m, w_params)
    try:
        out = model.model(**inputs)
    finally:
        remove_hooks(handles)
    L3 = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)[tok_id]
    gm, gw = grads_for(L3, m, w_params[first])
    results.append(("L3  log p('%s') behavioural" % args.affirmative, L3.item(), gm, gw))

    hdr = f"{'loss':<40}{'value':>12}{'dL/dm':>14}{'||dL/dw||':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, val, gm, gw in results:
        print(f"{name:<40}{val:>12.5f}{gm:>14.6e}{gw:>14.6e}")

    print(f"\nExpected if the projection identity holds: L1 value = -margin = {-args.margin}, "
          f"dL1/dm = -1, ||dL1/dw|| = 0.")
    print("L2 and L3 non-zero in both columns => probe-based and behavioural objectives at "
          "un-clamped readouts DO carry gradient.")


if __name__ == "__main__":
    main()
