"""Train per-layer probes on GENERATED-token activations (three families).

The stock probes (classifier/train_latent.py) are fit on a single position -- the last
prompt token. This trains probes on the tokens the model actually generates, so we can ask
whether steering evades a probe that was fit on the generation distribution, not just the
prompt position.

Data: the Arditi train prompts (harmful=1 / harmless=0), UNSTEERED greedy generation of N
tokens each. Every token inherits its prompt's class label. For harmful prompts the natural
generation is refusal text, so these probes detect "responding to a harmful request", not
"emitting harmful content" -- which is exactly what makes scoring steered (jailbroken)
activations against them a clean evasion test.

Collection is one generate + one full forward per prompt: greedy-generate, then a single
forward over the whole prompt+generation sequence with output_hidden_states=True. A causal
forward reproduces each position's activation exactly as generation produced it, so this
yields every prompt-position and generated-position activation per layer in one pass. Layer
convention matches train_latent.py (hidden_states[1:], last entry post-final-norm).

Three families, each per layer L (dirs under dataset/representations/<model>/):
  1. per-position    train_svm_genpos/  svm_layerLL_posPP.pt   fit on (layer L, gen position p)
  2. generated-only  train_svm_gentok/  svm_layerLL.pt         fit on all generated tokens
  3. alltok          train_svm_alltok/  svm_layerLL.pt         fit on the post-instruction token
                                                               (last prompt token, the position
                                                               the stock probe uses) + generated
                                                               tokens. NOT every prompt position:
                                                               arbitrary prompt/template tokens
                                                               don't encode harm and only wash the
                                                               signal out (near-chance).

Families 2 and 3 also save subsampled HFx_train.pt/HLx_train.pt (N,L,D) so the probe-score
experiments' class-reference bands render; point those experiments at the dir via --svm_dir.
The genpos family (2-D: layer x position) is loaded by a dedicated experiment mode.

Usage:
    python classifier/train_generated_probes.py --model_name llama3-8b --device cuda:0 --n_positions 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models_utils import get_transformer_layers
from utils.runtime import load_model, set_seed


def load_arditi_train(model_name):
    base = f"./dataset/splits/{model_name}"
    harmful = json.load(open(os.path.join(base, "harmful_train_filtered.json")))
    harmless = json.load(open(os.path.join(base, "harmless_train_filtered.json")))
    to_str = lambda d: d["instruction"] if isinstance(d, dict) else d
    return [to_str(d) for d in harmful], [to_str(d) for d in harmless]


def fit_svm(X, y, seed=42):
    """LinearSVC with the same recipe as train_latent.py. Returns (w, b, test_acc, n_train)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)
    clf = LinearSVC(C=0.1, dual="auto", max_iter=1000000, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.coef_[0], float(clf.intercept_[0]), float(clf.score(Xte, yte)), int(len(ytr))


def save_probe(path, w, b, layer_idx, model_name, hidden_dim, acc, n_train, extra=None):
    obj = {"w": torch.from_numpy(np.asarray(w)).float(), "b": torch.tensor(float(b)).float(),
           "layer_idx": int(layer_idx), "model_name": model_name, "hidden_dim": int(hidden_dim),
           "accuracy": acc, "n_train": n_train}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def fittable(y, min_per_class=15):
    y = np.asarray(y)
    return y.size and (y == 1).sum() >= min_per_class and (y == 0).sum() >= min_per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_positions", type=int, default=8, help="Generated positions 0..N-1 to collect.")
    ap.add_argument("--artifact_dir", default="./dataset/representations")
    ap.add_argument("--class_rep_per_class", type=int, default=400,
                    help="Subsample size per class for the saved HFx/HLx reference tensors.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    N = args.n_positions

    harmful, harmless = load_arditi_train(args.model_name)
    prompts = harmful + harmless
    labels = [1] * len(harmful) + [0] * len(harmless)
    print(f"Train prompts: {len(harmful)} harmful + {len(harmless)} harmless | "
          f"model {args.model_name} | {n_layers} layers | N={N} generated positions\n")

    # Per-layer activation rows (float16 CPU); token-level metadata shared across layers
    # because rows are appended in the same order for every layer.
    X_rows = [[] for _ in range(n_layers)]
    meta_label, meta_isgen, meta_genpos, meta_islast = [], [], [], []

    for prompt, label in zip(tqdm(prompts, desc="Collecting"), labels):
        inputs = model.prepare_inputs(prompt)
        plen = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.model.generate(**inputs, max_new_tokens=N, do_sample=False)
        full_ids = gen[:, : plen + N]  # (1, plen+gen_len); trim any trailing pad
        with torch.no_grad():
            out = model.model(input_ids=full_ids, output_hidden_states=True)
        seq_len = full_ids.shape[1]

        for pos in range(seq_len):
            is_gen = pos >= plen
            meta_label.append(label)
            meta_isgen.append(is_gen)
            meta_genpos.append(pos - plen if is_gen else -1)
            meta_islast.append(pos == plen - 1)  # last prompt token = post-instruction position
        for l in range(n_layers):
            X_rows[l].append(out.hidden_states[l + 1][0].detach().to(torch.float16).cpu())

    X = [torch.cat(rows, dim=0) for rows in X_rows]  # each (n_tokens, D) float16
    label = np.asarray(meta_label)
    isgen = np.asarray(meta_isgen, dtype=bool)
    genpos = np.asarray(meta_genpos)
    islast = np.asarray(meta_islast, dtype=bool)
    alltok_mask = isgen | islast  # post-instruction token + generated tokens
    n_tokens, hidden_dim = X[0].shape
    print(f"\nCollected {n_tokens} token activations/layer "
          f"({isgen.sum()} generated, {(~isgen).sum()} prompt, {islast.sum()} post-instruction).")
    print(f"alltok = {alltok_mask.sum()} tokens (post-instruction + generated).\n")

    base = os.path.join(args.artifact_dir, args.model_name)
    dirs = {fam: os.path.join(base, f"train_svm_{fam}") for fam in ("genpos", "gentok", "alltok")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def class_reps(mask):
        rng = np.random.RandomState(args.seed)
        idx_h = np.where(mask & (label == 1))[0]
        idx_l = np.where(mask & (label == 0))[0]
        pick = lambda idx: idx if len(idx) <= args.class_rep_per_class else rng.choice(
            idx, args.class_rep_per_class, replace=False)
        idx_h, idx_l = pick(idx_h), pick(idx_l)
        HF = torch.stack([X[l][torch.from_numpy(idx_h)].float() for l in range(n_layers)], dim=1)
        HL = torch.stack([X[l][torch.from_numpy(idx_l)].float() for l in range(n_layers)], dim=1)
        return HF, HL

    # Save reference tensors for the pooled families.
    for fam, mask in (("gentok", isgen), ("alltok", alltok_mask)):
        HF, HL = class_reps(mask)
        torch.save(HF, os.path.join(dirs[fam], "HFx_train.pt"))
        torch.save(HL, os.path.join(dirs[fam], "HLx_train.pt"))
        print(f"[{fam}] class reps: harmful {tuple(HF.shape)} harmless {tuple(HL.shape)}")

    # --- fit per layer ---
    print(f"\n{'layer':>6}{'alltok':>9}{'gentok':>9}{'genpos mean':>13}")
    print("-" * 37)
    genpos_accs_all = []
    for l in range(n_layers):
        Xl = X[l].float().numpy()

        w, b, acc_all, n = fit_svm(Xl[alltok_mask], label[alltok_mask])
        save_probe(os.path.join(dirs["alltok"], f"svm_layer{l:02d}.pt"), w, b, l,
                   args.model_name, hidden_dim, acc_all, n, extra={"family": "alltok"})

        w, b, acc_gen, n = fit_svm(Xl[isgen], label[isgen])
        save_probe(os.path.join(dirs["gentok"], f"svm_layer{l:02d}.pt"), w, b, l,
                   args.model_name, hidden_dim, acc_gen, n, extra={"family": "gentok"})

        pos_accs = []
        for p in range(N):
            m = isgen & (genpos == p)
            if not fittable(label[m]):
                continue
            w, b, acc_p, n = fit_svm(Xl[m], label[m])
            save_probe(os.path.join(dirs["genpos"], f"svm_layer{l:02d}_pos{p:02d}.pt"), w, b, l,
                       args.model_name, hidden_dim, acc_p, n, extra={"family": "genpos", "gen_pos": p})
            pos_accs.append(acc_p)
        genpos_accs_all.extend(pos_accs)
        mean_pos = sum(pos_accs) / len(pos_accs) if pos_accs else float("nan")
        print(f"{l:>6}{acc_all:>9.3f}{acc_gen:>9.3f}{mean_pos:>13.3f}")

    print(f"\nSaved probes to:\n  " + "\n  ".join(sorted(dirs.values())))
    print(f"genpos probes: up to {n_layers * N} (layers x positions); "
          f"mean test acc {np.mean(genpos_accs_all):.3f}" if genpos_accs_all else "")


if __name__ == "__main__":
    main()
