"""Compute a per-layer CLE margin schedule from the harmless distribution -- a BO-free
alternative to tuned margins.

With beta=1 the projection sets the steered probe score to exactly -m. So setting
    m_l = -mean(harmless probe scores at layer l) = |mean_harmless_l|
lands the steered activation exactly at the harmless distribution's center ("steer to the
harmless mean"). No Bayesian optimization -- the margin is read straight off the training data.

The margin acts on the STEERING probe, so the harmless activations are scored with that probe
(--probe_dir). --reps_dir chooses WHICH harmless activations to average over:
  * the steering probe's own training reps (prompt tokens) -> the standard decision-point margin
  * a generated-token family's HLx (e.g. train_svm_gentok) -> the harmless-generation margin
    under the original probe (generated tokens sit closer to the boundary, so smaller margins)

Prints a per-layer table and a comma-separated schedule string ready to paste into --schedule.
No model load (no GPU).

Usage:
    python experiments/harmless_mean_schedule.py --model_name llama3-8b --layers 11-18
    python experiments/harmless_mean_schedule.py --layers 11-18 \
        --reps_dir dataset/representations/llama3-8b/train_svm_gentok
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.probes import discover_available_layers, load_class_representations, load_probes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--probe_dir", default=None, help="STEERING probes (default: <model>/train_svm).")
    ap.add_argument("--reps_dir", default=None,
                    help="Where to read HFx/HLx from (default: --probe_dir). Point at a generated-token "
                         "family to compute the harmless-generation margin under the steering probe.")
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--layers", default="11-18", help="Window (end-exclusive).")
    ap.add_argument("--k_std", type=float, default=0.0,
                    help="Optional: m_l = |mean_HL| + k*std_HL, to steer past the mean.")
    args = ap.parse_args()

    base = os.path.join("./dataset/representations", args.model_name)
    probe_dir = args.probe_dir or os.path.join(base, "train_svm")
    reps_dir = args.reps_dir or probe_dir

    all_layers = discover_available_layers(probe_dir, args.probe_type)
    window = parse_layers_arg(args.layers, max(all_layers) + 1)
    probes = load_probes(probe_type=args.probe_type, svm_dir=probe_dir,
                         layer_indices=window, device=torch.device("cpu"))
    _, X_harmless = load_class_representations(reps_dir)

    print(f"probe: {os.path.basename(os.path.normpath(probe_dir))}  |  "
          f"harmless reps: {os.path.basename(os.path.normpath(reps_dir))}  |  window {args.layers}")
    print(f"{'layer':>6}{'mean HL':>10}{'std HL':>9}{'margin m_l':>12}")
    print("-" * 37)
    schedule = []
    for l in window:
        w, b = probes[l]["w"].float(), probes[l]["b"].float()
        s = X_harmless[:, l, :].float() @ w + b
        mean, std = s.mean().item(), s.std(unbiased=True).item()
        m = -mean + args.k_std * std
        schedule.append(m)
        print(f"{l:>6}{mean:>10.3f}{std:>9.3f}{m:>12.3f}")
    print("-" * 37)
    print("\nschedule (paste into --schedule):")
    print(",".join(f"{m:.2f}" for m in schedule))


if __name__ == "__main__":
    main()
