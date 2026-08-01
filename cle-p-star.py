"""CLE-P* -- CLE-P with a one-sided (gated) projection.

CLE-P projects EVERY position onto the hyperplane at probe score -m. That is wasteful and, on
benign inputs, actively harmful: an activation whose raw score is already below -m gets pushed
back UP toward the boundary, i.e. plain CLE-P steers harmless-looking activations *toward*
refusal in order to land them exactly on the target. CLE-P* adds a threshold c and only steers
positions whose raw probe score exceeds it:

    h* = h - 1{w·h + b > c} * beta * (w·h + b + m) / ||w||^2 * w

c is in raw probe-score units, so c = 0 is the probe's own decision boundary ("only steer what
reads as harmful"), and c = -m is a plain ReLU on the projection amount ("only steer what has
not already reached the target"; never move an activation backwards). c = -inf recovers CLE-P.

--gate_c also accepts a fraction of the layer's own margin ('-0.5m' = halfway between -m_l and
the boundary), which is the useful parameterisation: the interesting region is between -m and
0, and a per-layer margin schedule should carry the gate with it rather than pin it to one
absolute number. Above 0 the gate closes almost entirely (c=0.61 steered 2.6% of positions).

Everything else -- probes, layer window, beta, per-layer margins, datasets, judging -- is
identical to cle-p.py, so runs are directly comparable.

Usage:
    python cle-p-star.py --model_name llama3-8b --layers 11-18 \
        --layer_margins 1.08,1.08,1.1,1.11,1.12,1.14,1.14 --gate_c 0 \
        --dataset harmbench_standard --batch_size 16 --max_new_tokens 512
    python cle-p-star.py ... --gate_c=-1m         # per-layer c = -m_l ('relu' is an alias)
    python cle-p-star.py ... --gate_c=-0.5m       # halfway between -m_l and the boundary

Note the '=' on negative specs: argparse reads a bare '-0.5m' as an option name, not a value.
"""
import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from utils.args import build_run_tag, gate_tag, parse_layer_margins, parse_layers_arg
from utils.args import resolve_gate_thresholds
from utils.hooks import gated_projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import ProbeDict, load_probes
from utils.runtime import chunked, evaluate, load_model, load_prompts
from utils.runtime import print_configuration, set_seed, validate_probe_dims


def get_args():
    parser = argparse.ArgumentParser(
        description="Sequential multi-layer latent projection, applied only where the probe score exceeds a gate."
    )

    # Model config
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Probe / layer config
    parser.add_argument(
        "--svm_dir",
        type=str,
        default=None,
        help="Directory containing latent SVM checkpoints (svm_layerXX.pt).",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="svm",
        choices=["svm", "single_direction"],
        help=(
            "Probe artifact type. 'svm' loads svm_layerXX.pt with learned intercepts; "
            "'single_direction' loads sd_layerXX.pt from classifier/train_latent and "
            "derives the midpoint bias from class means."
        ),
    )
    parser.add_argument(
        "--probe_reps_dir",
        type=str,
        default=None,
        help=(
            "Directory containing HFx_train.pt and HLx_train.pt for deriving "
            "single-direction probe biases. Defaults to --svm_dir when available."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to intervene: 'all', '5-25' (end-exclusive), or '10,14,20'.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Scale in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin m in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--layer_margin",
        "--layer_margins",
        dest="layer_margin",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional per-layer margins aligned with --layers. Accepts either "
            "space-separated values or comma-separated values."
        ),
    )

    # CLE-P* gate
    parser.add_argument(
        "--gate_c",
        type=str,
        default="0",
        help=(
            "Gate threshold c: a position is steered only if w^T h + b > c. Absolute "
            "('0', '-0.5'), or a fraction of that layer's margin ('-1m' = -m_l = never steer "
            "backwards, 'relu' is an alias; '-0.5m' = halfway between -m_l and the boundary; "
            "'0' = the probe decision boundary). '-inf' reproduces plain CLE-P."
        ),
    )

    # Generation / dataset
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for projection and generation.")

    # Output / eval
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def register_gated_hooks(
    layers,
    selected_layers: List[int],
    probes: ProbeDict,
    layer_margin_map: Dict[int, float],
    gate_map: Dict[int, float],
    args,
    stats: Dict[int, Dict[str, torch.Tensor]],
):
    handles = []
    for layer_idx in selected_layers:
        handles.append(
            layers[layer_idx].register_forward_hook(
                gated_projection_hook(
                    probes[layer_idx]["w"],
                    probes[layer_idx]["b"],
                    args.beta,
                    layer_margin_map[layer_idx],
                    gate_map[layer_idx],
                    stats.setdefault(layer_idx, {}),
                )
            )
        )
    return handles


def generate_with_gated_projection(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    probes: ProbeDict,
    layer_margin_map: Dict[int, float],
    gate_map: Dict[int, float],
    args,
    stats,
) -> List[str]:
    generation_handles = register_gated_hooks(
        layers, selected_layers, probes, layer_margin_map, gate_map, args, stats)
    try:
        return model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"Batch generation error. Falling back to single-prompt generation: {e}")
        remove_hooks(generation_handles)
        generation_handles = []
        responses = []
        for prompt in batch_prompts:
            single_handles = register_gated_hooks(
                layers, selected_layers, probes, layer_margin_map, gate_map, args, stats)
            try:
                responses.append(model.generate(prompt, max_new_tokens=args.max_new_tokens))
            except Exception as inner_e:
                print(f"Gen Error: {inner_e}")
                responses.append("")
            finally:
                remove_hooks(single_handles)
        return responses
    finally:
        remove_hooks(generation_handles)


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    hidden_dim = model.model.config.hidden_size

    selected_layers = parse_layers_arg(args.layers, n_layers)
    layer_margin_map = parse_layer_margins(args.layer_margin, selected_layers, args.margin)
    gate_map = resolve_gate_thresholds(args.gate_c, layer_margin_map)

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if not os.path.isdir(args.svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {args.svm_dir}")

    probes = load_probes(
        probe_type=args.probe_type,
        svm_dir=args.svm_dir,
        layer_indices=selected_layers,
        device=device,
        explicit_reps_dir=args.probe_reps_dir,
    )
    validate_probe_dims(probes, selected_layers, hidden_dim)

    prompts, categories = load_prompts(args)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    base_out_dir = args.out_dir if args.out_dir is not None else os.path.join("./completions", args.model_name, "cle-p-star")
    args.out_dir = base_out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = build_run_tag(args, selected_layers, layer_margin_map) + gate_tag(args.gate_c)
    output_path = os.path.join(args.out_dir, f"completions_{run_tag}.json")
    print_configuration(
        args=args,
        selected_layers=selected_layers,
        layer_margin_map=layer_margin_map,
        probe_reps_dir=args.probe_reps_dir if args.probe_type == "single_direction" else None,
        output_path=output_path,
        prompt_count=len(prompts),
    )
    print(f"Gate c: {args.gate_c} -> {gate_map}")

    stats: Dict[int, Dict[str, torch.Tensor]] = {}
    results = []
    pbar = tqdm(total=len(prompts), desc="Gated projection + Generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[batch_start:batch_start + len(batch_prompts)]

        responses = generate_with_gated_projection(
            batch_prompts=batch_prompts,
            layers=layers,
            selected_layers=selected_layers,
            model=model,
            probes=probes,
            layer_margin_map=layer_margin_map,
            gate_map=gate_map,
            args=args,
            stats=stats,
        )

        for prompt, response, category in zip(batch_prompts, responses, batch_categories):
            results.append({"category": category, "prompt": prompt, "response": response})
        pbar.update(len(batch_prompts))
    pbar.close()

    # What fraction of positions did the gate actually let through, per layer? (Padding
    # positions are included, so read this as a relative signal across runs, not an exact rate.)
    gate_rate = {
        str(layer_idx): float(s["fired"] / s["total"])
        for layer_idx, s in sorted(stats.items()) if s.get("total") is not None and float(s["total"]) > 0
    }
    print("\nGate fire rate (fraction of positions steered), per layer:")
    for layer_idx, rate in gate_rate.items():
        print(f"  L{layer_idx}: {rate * 100:.1f}%")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved completions to {output_path}")

    with open(os.path.join(args.out_dir, f"gate_rate_{run_tag}.json"), "w") as f:
        json.dump({"gate_c": args.gate_c, "gate_map": {str(k): v for k, v in gate_map.items()},
                   "layer_margins": {str(k): v for k, v in layer_margin_map.items()},
                   "gate_rate": gate_rate}, f, indent=2)

    if args.evaluate:
        del model
        evaluate(
            results=results,
            eval_path=os.path.join(args.out_dir, "evaluation", f"evaluation_{run_tag}.json"),
        )


if __name__ == "__main__":
    main()
