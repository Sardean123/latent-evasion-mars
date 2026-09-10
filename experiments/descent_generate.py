"""Optimize the per-layer steering delta, then GENERATE with it — to read what the model says.

For each harmful prompt: descend the refuse-vs-comply log-odds (per-layer delta at the last prompt
token, SGD), snapshot delta at the log-odds flip AND at the final step, then greedily generate three
completions -- unsteered, steered@flip (minimal delta), steered@final (bigger delta) -- so we can
see whether the optimized delta yields real compliant text or just degenerate/off-manifold output.

The generation hook is position-guarded: it adds delta at the last-prompt-token position only during
the prefill forward (h.shape[1] > pos); during cached decode steps it is a no-op, so delta steers the
prompt representation that every generated token attends to (matching how it was optimised).

Output (harmful text) goes to model_outputs/ (gitignored).

Usage:
    python experiments/descent_generate.py --limit 5 --steps 40 --lr 0.002 --max_new_tokens 100
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from refusal_activation_descent import steer_hooks, log_odds, unit
from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes
from utils.runtime import load_model, load_prompts, set_seed

MODEL_OUT = os.path.join(ROOT, "experiments", "results", "model_outputs")


def gen_steer_hooks(layers, deltas, pos):
    """Add delta at `pos` only when the position exists in this forward (prefill, not cached decode)."""
    def mk(l):
        d = deltas[l]

        def hook(module, inputs, output):
            h = hidden_from_output(output)
            if h.shape[1] <= pos:
                return output
            add = torch.zeros_like(h)
            add[:, pos, :] = d.to(h.dtype)
            return replace_hidden(output, h + add)
        return hook
    return [layers[l].register_forward_hook(mk(l)) for l in deltas]


def generate(model, pid, attn, deltas, layers, pos, max_new_tokens):
    handles = gen_steer_hooks(layers, deltas, pos) if deltas is not None else []
    try:
        with torch.no_grad():
            out = model.model.generate(input_ids=pid, attention_mask=attn,
                                       max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=model.tokenizer.eos_token_id)
    finally:
        remove_hooks(handles)
    return model.tokenizer.decode(out[0, pid.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--reps_dir", default=None)
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--refusal", default="I cannot,I can't,I'm sorry")
    ap.add_argument("--compliance", default="Sure, here,Here's how,Sure I can help")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=40, help="Total descent steps (delta@final snapshot).")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if args.reps_dir is None:
        for cand in [os.path.join("../repo-backup/representations", args.model_name, "train_svm"),
                     os.path.join("dataset/representations/representations", args.model_name, "train_svm")]:
            if os.path.isdir(cand):
                args.reps_dir = cand
                break

    model = load_model(args)
    model.model.eval()
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))
    D = model.model.config.hidden_size
    probes = load_probes(probe_type="svm", svm_dir=args.svm_dir, layer_indices=selected,
                         device=device, explicit_reps_dir=args.reps_dir)
    w = {l: unit(probes[l]["w"].float().to(device)) for l in selected}
    refusal_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.refusal.split(",")]
    comply_ids = [model.tokenizer.encode(s.strip(), add_special_tokens=False) for s in args.compliance.split(",")]
    prompts, cats = load_prompts(args)
    prompts = prompts[: args.limit]

    os.makedirs(MODEL_OUT, exist_ok=True)
    out_path = os.path.join(MODEL_OUT, "descent_generations.txt")
    fh = open(out_path, "w", encoding="utf-8")
    fh.write(f"Activation-descent steered generations ({args.dataset}, lr {args.lr}, {args.steps} steps, "
             f"layers {selected})\nrefusal pool {args.refusal!r} | compliance pool {args.compliance!r}\n"
             + "=" * 100 + "\n")

    for i, prompt in enumerate(prompts):
        inputs = model.prepare_inputs(prompt)
        pid, attn = inputs.input_ids, inputs.attention_mask
        pos = pid.shape[1] - 1

        # --- optimise delta, snapshotting at the log-odds flip and at the final step ---
        deltas = {l: torch.zeros(D, device=device, requires_grad=True) for l in selected}
        opt = torch.optim.SGD([deltas[l] for l in selected], lr=args.lr)
        handles = steer_hooks(layers, deltas, pos)
        flip_snap, flip_step, lo0 = None, None, None
        try:
            for step in range(1, args.steps + 1):
                opt.zero_grad()
                lo = log_odds(model, pid, refusal_ids, comply_ids, pos, device)
                if lo0 is None:
                    lo0 = lo.item()
                if flip_snap is None and lo.item() < 0:
                    flip_snap = {l: deltas[l].detach().clone() for l in selected}
                    flip_step = step
                lo.backward()
                opt.step()
        finally:
            remove_hooks(handles)
        final_snap = {l: deltas[l].detach().clone() for l in selected}
        if flip_snap is None:            # never flipped within budget
            flip_snap, flip_step = final_snap, args.steps
        norm_flip = sum(flip_snap[l].norm().item() for l in selected) / len(selected)
        norm_final = sum(final_snap[l].norm().item() for l in selected) / len(selected)

        base = generate(model, pid, attn, None, layers, pos, args.max_new_tokens)
        steer_flip = generate(model, pid, attn, flip_snap, layers, pos, args.max_new_tokens)
        steer_final = generate(model, pid, attn, final_snap, layers, pos, args.max_new_tokens)

        block = (f"\n### [{i}] {cats[i]}  |  log-odds start {lo0:+.1f}, flip@{flip_step} "
                 f"(||δ||~{norm_flip:.2f}), final@{args.steps} (||δ||~{norm_final:.2f})\n"
                 f"PROMPT: {prompt}\n\n"
                 f"--- UNSTEERED ---\n{base}\n\n"
                 f"--- STEERED @flip (||δ||~{norm_flip:.2f}) ---\n{steer_flip}\n\n"
                 f"--- STEERED @{args.steps} steps (||δ||~{norm_final:.2f}) ---\n{steer_final}\n"
                 + "-" * 100 + "\n")
        fh.write(block)
        fh.flush()
        print(f"[{i+1}/{len(prompts)}] flip@{flip_step} ||δ|| {norm_flip:.2f}->{norm_final:.2f}")
        torch.cuda.empty_cache()

    fh.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
