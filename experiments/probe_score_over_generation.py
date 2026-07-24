"""How does the probe score evolve across GENERATED tokens, steered vs unsteered?

The companion script probe_score_distributions.py scores a single position -- the last
PROMPT token -- which is exactly the position the probes were fit on. This script extends
that to the tokens the model actually generates (positions 0, 1, 2, ... of the completion),
to answer the sharper question a latent monitor cares about:

    Does CLE's "reads harmless to the probe" property survive token-by-token through the
    real harmful generation, or does the score drift back toward refusal as text is emitted?

Two conditions, greedy-decoded on the same harmful prompts:

  * STEERED   -- the projection hooks are active at the window layers at every generation
                 step (this is what cle-p does at inference). By construction the WINDOW
                 layers read -m_l at every position, so the signal lives DOWNSTREAM of the
                 window (does persistence hold per generated token?) and in the contrast
                 with the unsteered run.
  * UNSTEERED -- the model generates freely (on a harmful prompt it typically refuses).
                 Read straight from hidden_states.

STEERING vs SCORING probes are separate. --svm_dir is the STEERING probe set (it defines
the CLE intervention -- keep it on the original probes). --score_svm_dir picks the probe the
steered/unsteered activations are SCORED against, and can be a generated-token family
(train_svm_gentok / train_svm_alltok, per-layer) or, with --score_probe_set genpos, the
per-(layer, position) probes in train_svm_genpos (scored position-matched). This asks whether
steering that nulls the ORIGINAL probe also evades a probe trained on the generation
distribution. When the score probes differ from the steering probes, the window layers are no
longer -m by construction -- their score under the new probe is a real measurement.

Three caveats baked into how this is read:

1. OFF-DISTRIBUTION PROBE (default only). With the original probes the reference bands are a
   last-prompt-token reference, so generated-token scores mean "which side of the boundary".
   Scoring with a gentok/alltok/genpos probe removes this -- those are fit on generated tokens.
2. WINDOW LAYERS ARE -m BY CONSTRUCTION only when scoring with the SAME probe used to steer.
3. Position p is a DIFFERENT token across conditions (steered emits different text), so the
   x-within-a-line is "generation step", not "the same token".

Capture rules mirror the fixed single-token script:
  * Unsteered: read hidden_states[l+1] for every layer (last entry is post-final-norm,
    matching how that layer's probe was trained).
  * Steered: hook-capture the WINDOW layers (pre-norm, the probe's basis for those layers);
    read every other layer -- including the post-norm last layer -- from hidden_states,
    which reflects the upstream window steering (only a steered layer's OWN hidden_states is
    stale). Generation is a no-KV-cache greedy loop: the growing sequence is re-run each
    step, so activations equal cached generation while avoiding cache/hook interactions.

Usage:
    python experiments/probe_score_over_generation.py --layers 11-18 \
        --schedule 1.2,2.0,1.8,1.8,2.0,0.9,1.2 --max_new_tokens 8 --limit 200
    python experiments/probe_score_over_generation.py --replot <saved.json>
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks, pipeline_delta_hook
from utils.models_utils import get_transformer_layers
from utils.probes import discover_available_layers, load_class_representations, load_probes
from utils.runtime import load_model, load_prompts, set_seed

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
HARMLESS_C, HARMFUL_C = "#2a9d5c", "#c0392b"
PROMPT_C = "#8a8981"


def add_record_hook(layer_idx, store, delta):
    """CLE-A: add the fixed captured delta to every position, record the steered last token,
    and return the modified output so downstream layers see the steering."""
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        v = delta.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
        h_mod = h + v
        store[layer_idx] = h_mod[:, -1, :].detach().float().cpu()
        return replace_hidden(output, h_mod)
    return hook


def steer_record_hook(layer_idx, store, w, b, beta, margin, eps=1e-12):
    """Project the whole hidden state along w (as cle-p does), record the steered last
    token, and return the modified output so downstream layers see the steering."""
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_l = w.to(device=h.device, dtype=h.dtype)
        b_l = b.to(device=h.device, dtype=h.dtype)
        m_l = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_l * w_l).clamp_min(eps)
        score = (h * w_l.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_l.view(1, 1, 1) + m_l.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_l.view(1, 1, -1))
        store[layer_idx] = h_mod[:, -1, :].detach().float().cpu()
        return replace_hidden(output, h_mod)
    return hook


def genpos_steer_record_hook(layer_idx, store, plen, w_post, b_post, m_post, gen_pp, beta, eps=1e-12):
    """CLE-P with position-dependent probes: project each position along the probe that owns it.
    Prompt positions (0..plen-1) use the post-instruction probe (w_post,b_post,m_post); generated
    position p (absolute index plen+p) uses that position's genpos probe gen_pp[p]=(w,b,m). Every
    position is re-projected to -m of its OWN probe at every step (closed loop), and the steered
    last token is recorded so downstream layers see the steering. Batch size 1."""
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        seq, D = h.shape[1], h.shape[-1]
        device, dtype = h.device, h.dtype
        W = torch.empty(seq, D, device=device, dtype=dtype)
        B = torch.empty(seq, device=device, dtype=dtype)
        M = torch.empty(seq, device=device, dtype=dtype)
        wp = w_post.to(device=device, dtype=dtype)
        W[:plen] = wp
        B[:plen] = float(b_post)
        M[:plen] = float(m_post)
        for i in range(plen, seq):
            p = i - plen
            probe = gen_pp.get(p)
            if probe is None:  # position beyond trained genpos range -> fall back to post-instr probe
                W[i] = wp; B[i] = float(b_post); M[i] = float(m_post)
            else:
                wg, bg, mg = probe
                W[i] = wg.to(device=device, dtype=dtype); B[i] = float(bg); M[i] = float(mg)
        w_norm_sq = (W * W).sum(dim=-1).clamp_min(eps)          # (seq,)
        score = (h[0] * W).sum(dim=-1) + B + M                  # (seq,)
        h_mod = (h[0] - beta * (score / w_norm_sq).unsqueeze(-1) * W).unsqueeze(0)
        store[layer_idx] = h_mod[:, -1, :].detach().float().cpu()
        return replace_hidden(output, h_mod)
    return hook


def score_of(X, w, b):
    return X.float() @ w.float() + b.float()


def summarize(t):
    t = t.float()
    return {"mean": t.mean().item(), "std": t.std(unbiased=True).item(), "n": t.numel()}


def eos_ids(model):
    ids = set()
    gc = getattr(model.model, "generation_config", None)
    for src in (getattr(gc, "eos_token_id", None), getattr(model.tokenizer, "eos_token_id", None)):
        if isinstance(src, (list, tuple)):
            ids.update(int(x) for x in src)
        elif src is not None:
            ids.add(int(src))
    return ids


def load_genpos_probes(score_dir):
    """Load per-(layer, generated-position) probes svm_layerLL_posPP.pt into
    {layer: {pos: {"w", "b"}}}. Returns (probes, n_positions, sorted_layers)."""
    import re
    pat = re.compile(r"svm_layer(\d+)_pos(\d+)\.pt$")
    probes, max_pos = {}, -1
    for name in os.listdir(score_dir):
        m = pat.match(name)
        if not m:
            continue
        l, p = int(m.group(1)), int(m.group(2))
        obj = torch.load(os.path.join(score_dir, name), map_location="cpu")
        probes.setdefault(l, {})[p] = {"w": obj["w"].float().view(-1),
                                       "b": torch.as_tensor(obj["b"]).float().view(())}
        max_pos = max(max_pos, p)
    if not probes:
        raise ValueError(f"No svm_layerLL_posPP.pt files found in {score_dir}")
    return probes, max_pos + 1, sorted(probes.keys())


def generate_and_score(model, layers, all_layers, input_ids, n_new, *, steer, window,
                       steer_probes, score_probes, score_mode, n_trained_pos, beta, margin_map, eos, per_pos,
                       method="cle-p", genpos_margin=None):
    """Greedy-decode n_new tokens (no KV cache) and record probe scores at each captured
    position. Steering (window) uses steer_probes: method 'cle-p' re-projects every position
    to -m at every step; 'cle-a' captures the per-layer projection delta at the last prompt
    token (one forward), then ADDS that fixed delta to every position during generation --
    so window scores land at -m + (current raw score - capture raw score), not exactly -m.
    'cle-p-genpos' is cle-p but position-dependent: prompt tokens are projected on the post-instr
    probe (steer_probes, margin_map), and generated position p on its genpos probe (score_probes
    [l][p], margin genpos_margin[l][p]) -- steering along each generated position's own direction.
    Scoring is IN-DISTRIBUTION per position: the post-instruction token (pos -1) is always
    scored with the original steer_probes, and GENERATED tokens (pos>=0) with score_probes --
    so a generated-only family (gentok/genpos) never scores the prompt token. score_mode
    'layer' uses score_probes[l]; 'genpos' uses the position-matched score_probes[l][p].
    per_pos[pos][layer] gets one score per prompt. Stops at the first EOS."""
    n_layers = len(layers)
    last_layer = n_layers - 1

    handles = []
    store = {}
    if steer and method == "cle-p":
        handles = [
            layers[l].register_forward_hook(
                steer_record_hook(l, store, steer_probes[l]["w"], steer_probes[l]["b"], beta, margin_map.get(l, 0.0))
            )
            for l in window
        ]
    elif steer and method == "cle-a":
        # Capture pass: one forward over the prompt to get each window layer's delta at the
        # last prompt token, then add those fixed deltas to every position during generation.
        deltas = {}
        cap = [
            layers[l].register_forward_hook(
                pipeline_delta_hook(steer_probes[l]["w"], steer_probes[l]["b"], beta, margin_map.get(l, 0.0), l, deltas)
            )
            for l in window
        ]
        with torch.no_grad():
            model.model(input_ids=input_ids)
        remove_hooks(cap)
        handles = [layers[l].register_forward_hook(add_record_hook(l, store, deltas[l].view(-1))) for l in window]
    elif steer and method == "cle-p-genpos":
        plen = input_ids.shape[1]
        genpos_margin = genpos_margin or {}
        handles = []
        for l in window:
            gen_pp = {p: (lp["w"], lp["b"], genpos_margin.get(l, {}).get(p, 0.0))
                      for p, lp in score_probes.get(l, {}).items()}
            handles.append(
                layers[l].register_forward_hook(
                    genpos_steer_record_hook(l, store, plen, steer_probes[l]["w"], steer_probes[l]["b"],
                                             margin_map.get(l, 0.0), gen_pp, beta)
                )
            )
    try:
        seq = input_ids
        for step in range(n_new + 1):
            with torch.no_grad():
                out = model.model(input_ids=seq, output_hidden_states=True)
            pos = step - 1  # -1 = prompt last token, then 0, 1, ...
            for l in all_layers:
                if steer and l in window and l != last_layer:
                    act = store[l]  # pre-norm steered, the probe's basis for window layers
                else:
                    # hidden_states reflects upstream steering for non-window/downstream
                    # layers; its last entry is post-final-norm, matching the last probe.
                    act = out.hidden_states[l + 1][:, -1, :].detach().float().cpu()
                if pos < 0:
                    # post-instruction token: always the original (steering) probe, in-distribution.
                    lp = steer_probes.get(l)
                    if lp is None:
                        continue
                    w, b = lp["w"], lp["b"]
                elif score_mode == "genpos":
                    if pos >= n_trained_pos:
                        continue
                    lp = score_probes.get(l, {}).get(pos)
                    if lp is None:
                        continue
                    w, b = lp["w"], lp["b"]
                else:
                    w, b = score_probes[l]["w"], score_probes[l]["b"]
                per_pos[pos][l].append(score_of(act, w, b).item())
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            if next_id.item() in eos:
                break
    finally:
        remove_hooks(handles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None, help="STEERING probes (the projection direction). "
                    "Keep this on the original CLE probes -- it defines the intervention.")
    ap.add_argument("--score_svm_dir", default=None,
                    help="SCORING probes, if different from the steering probes. Point at "
                         "train_svm_gentok / train_svm_alltok / train_svm_genpos to evaluate the "
                         "steered activations under a generated-token probe. Default: same as --svm_dir.")
    ap.add_argument("--score_probe_set", default="layer", choices=["layer", "genpos"],
                    help="'layer': one probe per layer (gentok/alltok/original). 'genpos': "
                         "per-(layer, generated position) probes, scored position-matched.")
    ap.add_argument("--probe_type", default="svm", choices=["svm", "single_direction"])
    ap.add_argument("--probe_reps_dir", default=None)
    ap.add_argument("--layers", default="11-18", help="Steered window (end-exclusive).")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--schedule", default=None, help="Per-layer margins aligned with --layers.")
    ap.add_argument("--margin_offset", type=float, default=0.0,
                    help="Add this to every window margin, e.g. to push past the tuned regime "
                         "and probe where the model breaks. Recorded in the output tag as _off<x>.")
    ap.add_argument("--method", default="cle-p", choices=["cle-p", "cle-a", "cle-p-genpos"],
                    help="cle-p: re-project every position to -m each step. cle-a: capture the "
                         "delta at the last prompt token, then add it to every position (window "
                         "scores drift off -m). cle-p-genpos: cle-p, but generated position p is "
                         "projected on its own genpos probe/direction (margin from --genpos_bands_json).")
    ap.add_argument("--genpos_bands_json", default=None,
                    help="Per-(layer,pos) training bands JSON (from genpos_bands.py). Required for "
                         "--method cle-p-genpos: the generated-position margin is m(l,p) = -harmless_mean(l,p).")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=8, help="Generated tokens to capture (positions 0..N-1).")
    ap.add_argument("--plot_positions", default="0-5", help="Inclusive generated-position range to draw.")
    ap.add_argument("--out_dir", default="./experiments/results/probe_score_over_generation")
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", default=None, help="Render from a saved *_over_generation_*.json; no model load.")
    args = ap.parse_args()

    if args.replot:
        with open(args.replot) as f:
            data = json.load(f)
        plot(data, args.replot.replace(".json", ".png"), args.plot_positions)
        return

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

    # STEERING probes (define the intervention) -- always the dir given by --svm_dir.
    steer_layers = discover_available_layers(args.svm_dir, args.probe_type)
    steer_probes = load_probes(probe_type=args.probe_type, svm_dir=args.svm_dir,
                               layer_indices=steer_layers, device=torch.device("cpu"),
                               explicit_reps_dir=args.probe_reps_dir)

    # SCORING probes -- may be a different family. reference = per-layer class bands (layer
    # mode only; genpos has no single class-rep set).
    score_dir = args.score_svm_dir or args.svm_dir
    reference = {"harmless": {}, "harmful": {}}
    if args.score_probe_set == "genpos":
        score_probes, n_trained_pos, all_layers = load_genpos_probes(score_dir)
        # Reference bands = harmless/harmful TRAINING prompt-token activations scored on the
        # ORIGINAL prompt-token probe (steer_probes) -- the fair reference for the post-instr
        # (pos -1) line, which is also scored on that probe. Generated-position lines are each
        # scored on their own genpos probe and read against the shared 0 boundary.
        X_harm, X_harmless = load_class_representations(args.probe_reps_dir or args.svm_dir)
        for l in all_layers:
            lp = steer_probes.get(l)
            if lp is None:
                continue
            w, b = lp["w"].cpu(), lp["b"].cpu()
            reference["harmless"][str(l)] = summarize(score_of(X_harmless[:, l, :], w, b))
            reference["harmful"][str(l)] = summarize(score_of(X_harm[:, l, :], w, b))
    else:
        n_trained_pos = 0
        all_layers = discover_available_layers(score_dir, args.probe_type)
        score_probes = load_probes(probe_type=args.probe_type, svm_dir=score_dir,
                                   layer_indices=all_layers, device=torch.device("cpu"),
                                   explicit_reps_dir=args.probe_reps_dir)
        X_harm, X_harmless = load_class_representations(args.probe_reps_dir or score_dir)
        for l in all_layers:
            w, b = score_probes[l]["w"].cpu(), score_probes[l]["b"].cpu()
            reference["harmless"][str(l)] = summarize(score_of(X_harmless[:, l, :], w, b))
            reference["harmful"][str(l)] = summarize(score_of(X_harm[:, l, :], w, b))

    # cle-p-genpos: build the generated-position margins m(l,p) = -harmless_mean(l,p) from the
    # per-(layer,pos) training bands -- "the same mean margin thing", applied to each genpos probe.
    genpos_margin = None
    if args.method == "cle-p-genpos":
        if args.score_probe_set != "genpos":
            raise ValueError("--method cle-p-genpos requires --score_probe_set genpos.")
        if not args.genpos_bands_json:
            raise ValueError("--method cle-p-genpos requires --genpos_bands_json (from genpos_bands.py).")
        with open(args.genpos_bands_json) as f:
            bands = json.load(f)["bands"]
        genpos_margin = {}
        for l in window:
            for p, lp in score_probes.get(l, {}).items():
                hb = bands.get(str(p), {}).get(str(l), {}).get("harmless")
                if hb is None:
                    raise ValueError(f"No harmless band for (layer {l}, pos {p}) in {args.genpos_bands_json}")
                genpos_margin.setdefault(l, {})[p] = -float(hb["mean"])
        sample = {p: round(genpos_margin[max(window)][p], 3) for p in sorted(genpos_margin[max(window)])}
        print(f"genpos margins (L{max(window)}, by gen pos): {sample}")

    prompts, _ = load_prompts(args)
    eos = eos_ids(model)
    score_desc = f"{args.score_probe_set}:{os.path.basename(os.path.normpath(score_dir))}"
    print(f"Model: {args.model_name} | method {args.method} | window {args.layers} -> {window} | margins {margin_map}")
    print(f"Steer probes: {os.path.basename(os.path.normpath(args.svm_dir))} | Score probes: {score_desc}")
    print(f"Eval prompts: {len(prompts)} from {args.dataset} | max_new_tokens {args.max_new_tokens} | eos {sorted(eos)}\n")

    positions = list(range(-1, args.max_new_tokens))
    # cond -> pos -> layer -> [score per prompt]
    acc = {c: {p: {l: [] for l in all_layers} for p in positions} for c in ("steered", "unsteered")}

    score_kw = dict(window=window, steer_probes=steer_probes, score_probes=score_probes,
                    score_mode=args.score_probe_set, n_trained_pos=n_trained_pos,
                    beta=args.beta, margin_map=margin_map, eos=eos, method=args.method,
                    genpos_margin=genpos_margin)
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = model.prepare_inputs(prompt)
        generate_and_score(model, layers, all_layers, inputs.input_ids, args.max_new_tokens,
                           steer=False, per_pos=acc["unsteered"], **score_kw)
        generate_and_score(model, layers, all_layers, inputs.input_ids, args.max_new_tokens,
                           steer=True, per_pos=acc["steered"], **score_kw)

    # --- aggregate: cond -> pos -> layer -> {mean,std,n} ---
    conditions = {}
    for c in ("steered", "unsteered"):
        conditions[c] = {}
        for p in positions:
            conditions[c][str(p)] = {}
            for l in all_layers:
                vals = acc[c][p][l]
                if vals:
                    conditions[c][str(p)][str(l)] = summarize(torch.tensor(vals))
    # reference bands were computed above from the SCORE probes (empty in genpos mode).

    data = {"model": args.model_name, "window": window, "margins": margin_map,
            "dataset": args.dataset, "n_eval": len(prompts), "max_new_tokens": args.max_new_tokens,
            "layers": all_layers, "conditions": conditions, "reference": reference,
            "steer_probes": os.path.basename(os.path.normpath(args.svm_dir)), "score_probes": score_desc,
            "method": args.method}

    # --- console: per-position mean score at a few representative layers ---
    win_end = max(window)
    mid = min(all_layers, key=lambda l: abs(l - (win_end + max(all_layers)) // 2))
    last = max(all_layers)
    probe_layers = sorted(set([win_end, mid, last]))
    print("\nMean steered / unsteered probe score by generated position "
          f"(layers {probe_layers}; pos -1 = prompt token):")
    hdr = f"{'pos':>4}" + "".join(f"{'L%d s/u' % l:>16}" for l in probe_layers)
    print(hdr); print("-" * len(hdr))
    for p in positions:
        line = f"{p:>4}"
        for l in probe_layers:
            s = conditions["steered"].get(str(p), {}).get(str(l))
            u = conditions["unsteered"].get(str(p), {}).get(str(l))
            cell = f"{s['mean']:.2f}/{u['mean']:.2f}" if s and u else "-"
            line += f"{cell:>16}"
        print(line)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model_name}_layers{args.layers.replace('-', 'to')}"
    if args.out_tag:
        tag += f"_{args.out_tag}"
    if args.method != "cle-p":
        tag += f"_{args.method.replace('-', '')}"
    if args.margin_offset:
        tag += f"_off{args.margin_offset:g}"
    if args.score_probe_set == "genpos":
        tag += "_scoregenpos"
    elif args.score_svm_dir:
        score_name = os.path.basename(os.path.normpath(score_dir)).replace("train_svm_", "").replace("train_svm", "orig")
        tag += f"_score{score_name}"
    path = os.path.join(args.out_dir, f"probe_score_over_generation_{tag}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved data to {path}")
    plot(data, os.path.join(args.out_dir, f"probe_score_over_generation_{tag}.png"), args.plot_positions)


def _parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def plot(data, out_path, plot_positions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_layers = data["layers"]
    window = data["window"]
    draw_pos = [p for p in _parse_range(plot_positions) if str(p) in data["conditions"]["steered"]]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, facecolor=SURFACE)
    titles = {"steered": "STEERED generation", "unsteered": "UNSTEERED generation"}
    for ax, cond in zip(axes, ("steered", "unsteered")):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xlabel("layer", color=INK_MUTED, fontsize=10)
        ax.axvspan(min(window), max(window), color="#2a78d6", alpha=0.07, zorder=1)
        ax.axhline(0, color="#8a8981", linewidth=1.2, linestyle="--", zorder=2)

        # Prompt-token reference bands (probes' training position).
        for key, color in (("harmless", HARMLESS_C), ("harmful", HARMFUL_C)):
            ref = data["reference"][key]
            xs = [l for l in all_layers if str(l) in ref]
            mu = [ref[str(l)]["mean"] for l in xs]
            sd = [ref[str(l)]["std"] for l in xs]
            ax.fill_between(xs, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)],
                            color=color, alpha=0.10, zorder=2)
            ax.plot(xs, mu, color=color, linewidth=1.2, alpha=0.5, zorder=3,
                    label=f"{key} (train, prompt tok)")

        # Prompt token (pos -1) as a dashed reference for this condition.
        prompt = data["conditions"][cond].get("-1", {})
        if prompt:
            xs = [l for l in all_layers if str(l) in prompt]
            ax.plot(xs, [prompt[str(l)]["mean"] for l in xs], color=PROMPT_C, linewidth=1.6,
                    linestyle="--", zorder=4, label="post-instr tok (orig probe)")

        # One line per generated position, colour-graded.
        for i, p in enumerate(draw_pos):
            frac = i / max(len(draw_pos) - 1, 1)
            col = cmap(0.15 + 0.7 * frac)
            posd = data["conditions"][cond][str(p)]
            xs = [l for l in all_layers if str(l) in posd]
            ax.plot(xs, [posd[str(l)]["mean"] for l in xs], color=col, linewidth=2, zorder=5,
                    label=f"gen pos {p}")
        ax.set_title(titles[cond], color=INK, fontsize=11, loc="left", pad=10)
    axes[0].set_ylabel("probe score  w.h + b", color=INK_MUTED, fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, labelcolor=INK_MUTED, ncol=2)

    m = data["model"]
    score_probes = data.get("score_probes", "orig")
    method = data.get("method", "cle-p")
    fig.suptitle(f"{m}  ·  {method}  ·  probe score across generated tokens  ·  window "
                 f"{min(window)}-{max(window) + 1}  ·  score probe: {score_probes}  ·  "
                 f"{data['n_eval']} harmful prompts",
                 color=INK, fontsize=12, x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
