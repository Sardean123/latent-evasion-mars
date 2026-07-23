"""Dump steered vs unsteered completions to a readable txt, for eyeballing whether the
CLE projection actually jailbreaks and what tokens it emits.

Steering is the same projection_hook at the window layers that cle-p.py uses at inference,
so these completions are faithful to the real method. Greedy decoding, one prompt at a time.
For each prompt we print the unsteered and steered completions and the first few generated
tokens (raw tokenizer pieces, so you can see exactly what came out).

Output goes to experiments/results/model_outputs/ (readable in-editor but gitignored, since
harmful completions are kept out of git, matching .gitignore's sweep_*_outputs.txt rule).

Usage:
    python experiments/dump_completions.py --layers 11-18 \
        --schedule 1.2,2.0,1.8,1.8,2.0,0.9,1.2 --n_prompts 30 --max_new_tokens 64 \
        --out_tag paper [--margin_offset 1.5]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import discover_available_layers, load_probes
from utils.runtime import load_model, load_prompts, set_seed


def generate(model, layers, window, probes, margin_map, beta, prompt, max_new_tokens, steer, n_show):
    inputs = model.prepare_inputs(prompt)
    handles = []
    if steer:
        handles = [
            layers[l].register_forward_hook(
                projection_hook(probes[l]["w"], probes[l]["b"], beta, margin_map.get(l, 0.0))
            )
            for l in window
        ]
    try:
        with torch.no_grad():
            out = model.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        remove_hooks(handles)
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
    toks = model.tokenizer.convert_ids_to_tokens(gen_ids[:n_show])
    return text.strip(), toks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--probe_reps_dir", default=None)
    ap.add_argument("--layers", default="11-18", help="Steered window (end-exclusive).")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--schedule", default=None, help="Per-layer margins aligned with --layers.")
    ap.add_argument("--margin_offset", type=float, default=0.0,
                    help="Add this to every window margin (push past the tuned regime to break "
                         "the model). Recorded in the output tag as _off<x>.")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n_prompts", type=int, default=30, help="How many prompts to dump.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--n_show_tokens", type=int, default=10, help="First N generated tokens to list verbatim.")
    ap.add_argument("--out_dir", default="./experiments/results/model_outputs")
    ap.add_argument("--out_tag", default=None, help="Descriptive tag, e.g. 'paper'. Filename is "
                    "steered_completions_<model>_layers<w>_<tag>[_off<x>].txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    window = parse_layers_arg(args.layers, n_layers)

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")

    if args.schedule:
        vals = [float(v) for v in args.schedule.replace(" ", "").split(",") if v]
        if len(vals) != len(window):
            raise ValueError(f"--schedule has {len(vals)} values but window is {len(window)} layers: {window}")
        margin_map = dict(zip(window, vals))
    else:
        margin_map = {l: args.margin for l in window}
    if args.margin_offset:
        margin_map = {l: m + args.margin_offset for l, m in margin_map.items()}

    all_layers = discover_available_layers(args.svm_dir, args.probe_type)
    probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                         layer_indices=all_layers, device=torch.device("cpu"),
                         explicit_reps_dir=args.probe_reps_dir)

    prompts, _ = load_prompts(args)
    prompts = prompts[:args.n_prompts]

    tag = f"{args.model_name}_layers{args.layers.replace('-', 'to')}"
    if args.out_tag:
        tag += f"_{args.out_tag}"
    if args.margin_offset:
        tag += f"_off{args.margin_offset:g}"
    out_path = os.path.join(args.out_dir, f"steered_completions_{tag}.txt")

    print(f"Model: {args.model_name} | window {args.layers} -> {window} | margins {margin_map}")
    print(f"Dumping {len(prompts)} prompts x {args.max_new_tokens} tokens (steered + unsteered) -> {out_path}\n")

    header = (f"{args.model_name}  |  window {args.layers} -> {window}  |  margins {margin_map}\n"
              f"dataset {args.dataset}  |  {len(prompts)} prompts  |  {args.max_new_tokens} new tokens  |  greedy\n"
              f"Steering = projection_hook at window layers (same as cle-p.py).\n")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header + "\n" + "=" * 100 + "\n")
        for i, prompt in enumerate(prompts):
            print(f"[{i + 1}/{len(prompts)}] generating...")
            un_text, un_toks = generate(model, layers, window, probes, margin_map, args.beta,
                                        prompt, args.max_new_tokens, steer=False, n_show=args.n_show_tokens)
            st_text, st_toks = generate(model, layers, window, probes, margin_map, args.beta,
                                        prompt, args.max_new_tokens, steer=True, n_show=args.n_show_tokens)
            f.write(f"\n### [{i + 1}] PROMPT\n{prompt}\n")
            f.write(f"\n--- UNSTEERED ---\nfirst tokens: {un_toks}\n{un_text}\n")
            f.write(f"\n--- STEERED ---\nfirst tokens: {st_toks}\n{st_text}\n")
            f.write("\n" + "=" * 100 + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
