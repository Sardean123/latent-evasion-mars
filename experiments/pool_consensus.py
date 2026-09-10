"""Pool the summed gradient vectors across harmful+harmless to get cos(mean gradient, w).

The per-prompt cosine MEAN (a) is the headline (from the JSONs). This adds the complementary
quantity (b): average the gradient VECTORS over all 318 prompts, then take one cosine with w.
Two flavours of the average:
  unit  -- sum of per-prompt UNIT gradients (pure direction consensus, equal weight per prompt)
  raw   -- sum of per-prompt RAW gradients (magnitude-weighted; big-gradient prompts dominate)

Reads the .consensus.npz files written by refusal_gradient_alignment.py. No model, no GPU.

Usage: python experiments/pool_consensus.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results", "refusal_gradient")
LAYERS = [11, 12, 13, 14, 15, 16, 17]


def cos_rows(A, B):  # per-row cosine of two (L, D) arrays
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return num / np.clip(den, 1e-12, None)


def main():
    H = np.load(os.path.join(OUT, "alignment_harmbench_test_11to18.consensus.npz"))
    L = np.load(os.path.join(OUT, "alignment_harmless_val_11to18.consensus.npz"))
    w = H["w"]  # (L, D), same probe in both
    # per-prompt cosine means (a), from the JSONs
    jH = json.load(open(os.path.join(OUT, "alignment_harmbench_test_11to18.json")))
    jL = json.load(open(os.path.join(OUT, "alignment_harmless_val_11to18.json")))
    a_gR = [(jH["per_layer"][str(l)]["gR_w"]["mean"] + jL["per_layer"][str(l)]["gR_w"]["mean"]) / 2
            for l in LAYERS]
    a_gRD = [(jH["per_layer"][str(l)]["gRD_w"]["mean"] + jL["per_layer"][str(l)]["gRD_w"]["mean"]) / 2
             for l in LAYERS]

    # (b) pooled consensus: add the summed vectors from both datasets, cos with w
    b_gR_unit = cos_rows(H["gR_unit_sum"] + L["gR_unit_sum"], w)
    b_gR_raw = cos_rows(H["gR_raw_sum"] + L["gR_raw_sum"], w)
    b_gRD_unit = cos_rows(H["gRD_unit_sum"] + L["gRD_unit_sum"], w)
    b_gRD_raw = cos_rows(H["gRD_raw_sum"] + L["gRD_raw_sum"], w)

    print(f"pooled over {int(H['n_prompts'])}+{int(L['n_prompts'])} = "
          f"{int(H['n_prompts'])+int(L['n_prompts'])} prompts\n")
    print(f"{'L':>3} | {'REFUSAL gR':>28} | {'LOG-ODDS gRD':>28}")
    print(f"{'':>3} | {'(a) cos-mean':>12} {'(b) unit':>7} {'(b) raw':>7} | "
          f"{'(a) cos-mean':>12} {'(b) unit':>7} {'(b) raw':>7}")
    for i, l in enumerate(LAYERS):
        print(f"{l:>3} | {a_gR[i]:>+12.3f} {b_gR_unit[i]:>+7.3f} {b_gR_raw[i]:>+7.3f} | "
              f"{a_gRD[i]:>+12.3f} {b_gRD_unit[i]:>+7.3f} {b_gRD_raw[i]:>+7.3f}")

    out = {"layers": LAYERS,
           "refusal": {"a_cos_mean": a_gR, "b_unit": b_gR_unit.tolist(), "b_raw": b_gR_raw.tolist()},
           "log_odds": {"a_cos_mean": a_gRD, "b_unit": b_gRD_unit.tolist(), "b_raw": b_gRD_raw.tolist()}}
    json.dump(out, open(os.path.join(OUT, "pooled_consensus.json"), "w"), indent=2)
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'pooled_consensus.json'), ROOT)}")


if __name__ == "__main__":
    main()
