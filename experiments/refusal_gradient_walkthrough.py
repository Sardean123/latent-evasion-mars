"""A slow, print-everything walkthrough of ONE thing: the refusal-gradient / probe alignment.

This is the pedagogical companion to refusal_gradient_alignment.py. It runs a handful of prompts
and narrates every step so the mechanics are inspectable and debuggable:

  1. WHERE we hook: the exact module (LlamaDecoderLayer[l]), its output = residual stream after
     block l = hidden_states[l+1]. The probe reads this; CLE steers this (at every position); we
     take the gradient w.r.t. this at the last prompt token.
  2. WHAT log P("I cannot") is: teacher-force the refusal tokens after the prompt, print each
     token's log-prob and the sum.
  3. THE GRADIENT: d logP(refusal) / d h_l[last_prompt_tok], with activation and gradient norms.
  4. THE COSINES: cos(w, diff-in-means), cos(grad, w), cos(grad, diff-in-means) -- per layer.
  5. A FINITE-DIFFERENCE SANITY CHECK (the important one): actually move the activation a fixed
     L2 step along unit(grad) vs along unit(w) and re-measure the first refusal token's log-prob.
     If cos(grad,w)~0, stepping along w should barely change refusal while stepping along grad
     changes it a lot -- this turns the abstract cosine into an observable behavioural effect and
     confirms the gradient is not a numerical artifact.

Usage:
    python experiments/refusal_gradient_walkthrough.py --n_prompts 3 --layers 11-18 --fd_layer 14
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_layers_arg
from utils.hooks import hidden_from_output, replace_hidden, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import load_probes, load_class_representations, layer_centroids
from utils.runtime import load_model, load_prompts, set_seed


def unit(v, eps=1e-12):
    return v / (torch.linalg.vector_norm(v).clamp_min(eps))


def cos(a, b):
    return torch.dot(unit(a), unit(b)).item()


def enable_activation_grad(model):
    """Frozen params -> no autograd graph. Make the embedding output require grad to rebuild it."""
    emb = model.model.get_input_embeddings()
    return emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True))


def capture_hooks(layers, selected, store):
    def mk(l):
        def hook(module, inputs, output):
            store[l] = hidden_from_output(output)
        return hook
    return [layers[l].register_forward_hook(mk(l)) for l in selected]


def add_at_pos_hook(layer_module, delta_vec, pos):
    """Add a fixed vector to the block output at ONE position (for the finite-difference test)."""
    def hook(module, inputs, output):
        h = hidden_from_output(output).clone()
        h[0, pos, :] = h[0, pos, :] + delta_vec.to(h.dtype)
        return replace_hidden(output, h)
    return layer_module.register_forward_hook(hook)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="llama3-8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--svm_dir", default=None)
    ap.add_argument("--reps_dir", default=None)
    ap.add_argument("--layers", default="11-18")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--n_prompts", type=int, default=3)
    ap.add_argument("--refusal_phrase", default="I cannot")
    ap.add_argument("--affirmative", default="Sure")
    ap.add_argument("--fd_layer", type=int, default=14, help="Layer for the finite-difference test.")
    ap.add_argument("--fd_steps", default="2,4,8", help="L2 step norms to try in the FD test.")
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

    print("=" * 100)
    print("LOADING MODEL")
    print("=" * 100)
    model = load_model(args)
    model.model.eval()
    layers = get_transformer_layers(model)
    selected = parse_layers_arg(args.layers, len(layers))
    print(f"model has {len(layers)} decoder blocks; steering/gradient window = {selected}")
    print(f"  layers object type: {type(layers).__name__}, element type: {type(layers[selected[0]]).__name__}")
    print(f"  we hook layers[l].register_forward_hook for l in {selected}")
    print(f"  the hooked tensor is that block's OUTPUT = residual stream after block l = hidden_states[l+1]")
    print(f"  (hidden_states[0] is the token embeddings, so probe layer l <-> hidden_states[l+1])")

    print("\n" + "=" * 100)
    print("LOADING PROBES (the CLE steering direction) AND DIFF-IN-MEANS")
    print("=" * 100)
    probes = load_probes(probe_type="svm", svm_dir=args.svm_dir, layer_indices=selected,
                         device=device, explicit_reps_dir=args.reps_dir)
    w = {l: probes[l]["w"].float().to(device) for l in selected}
    print(f"probe dir: {args.svm_dir}")
    for l in selected:
        print(f"  L{l}: w is a LinearSVC weight vector, shape {tuple(w[l].shape)}, ||w||={w[l].norm():.3f}")

    dmean = {}
    Xh, Xl = load_class_representations(args.reps_dir)
    print(f"\ndiff-in-means reps: {args.reps_dir}  (harmful {tuple(Xh.shape)}, harmless {tuple(Xl.shape)})")
    for l in selected:
        mu_h, mu_l = layer_centroids(Xh, Xl, l, args.reps_dir)
        dmean[l] = (mu_h - mu_l).float().to(device)
    print("dmean[l] = mean(harmful activations) - mean(harmless activations) at layer l (points toward harmful)")

    print("\n" + "-" * 100)
    print("STEP 4a  cos(probe w, diff-in-means)   [these two 'refusal directions' vs each other]")
    print("-" * 100)
    for l in selected:
        print(f"  L{l}:  cos(w, dmean) = {cos(w[l], dmean[l]):+.4f}")

    # first refusal token, for the finite-difference target
    refuse_ids = model.tokenizer.encode(args.refusal_phrase, add_special_tokens=False)
    refuse0 = refuse_ids[0]
    comply0 = model.tokenizer.encode(args.affirmative, add_special_tokens=False)[0]
    print(f"\nrefusal phrase {args.refusal_phrase!r} -> token ids {refuse_ids} "
          f"({[model.tokenizer.decode([t]) for t in refuse_ids]})")
    print(f"first refusal token id {refuse0} ({model.tokenizer.decode([refuse0])!r}); "
          f"affirmative first token id {comply0} ({model.tokenizer.decode([comply0])!r})")

    args.limit = args.n_prompts
    prompts, cats = load_prompts(args)
    prompts = prompts[: args.n_prompts]
    fd_steps = [float(x) for x in args.fd_steps.split(",")]

    for pi, prompt in enumerate(prompts):
        print("\n\n" + "#" * 100)
        print(f"# PROMPT {pi+1}/{len(prompts)}  [{cats[pi]}]")
        print("#" * 100)
        print(f"raw prompt: {prompt}")
        inputs = model.prepare_inputs(prompt)
        pid = inputs.input_ids
        pos = pid.shape[1] - 1
        print(f"formatted+tokenized to {pid.shape[1]} tokens; last-prompt-token position = {pos}")
        print(f"  last 6 tokens: {[model.tokenizer.decode([t]) for t in pid[0, -6:].tolist()]}")
        print(f"  the model's next-token distribution AT position {pos} is the first response token")

        # ---------- STEP 2: log P("I cannot") by teacher forcing ----------
        print(f"\n-- STEP 2: log P({args.refusal_phrase!r}) by teacher forcing --")
        rt = torch.tensor([refuse_ids], device=device)
        ids2 = torch.cat([pid, rt], dim=1)
        store = {}
        h = [enable_activation_grad(model)] + capture_hooks(layers, selected, store)
        try:
            out = model.model(input_ids=ids2, attention_mask=torch.ones_like(ids2))
            lp = torch.log_softmax(out.logits[0].float(), dim=-1)
            total = torch.zeros((), device=device)
            for j, tid in enumerate(refuse_ids):
                tok_lp = lp[pos + j, tid]
                total = total + tok_lp
                print(f"    predict {model.tokenizer.decode([tid])!r:>10} at pos {pos+j}: "
                      f"logP={tok_lp.item():+.4f}  P={tok_lp.exp().item():.4f}")
            print(f"    => log P(refusal) = {total.item():+.4f}   P(refusal) = {total.exp().item():.4f}")

            # capture activation norms at pos BEFORE the graph is freed
            hnorm = {l: store[l][0, pos, :].detach().float().norm().item() for l in selected}
            # ---------- STEP 3: gradient w.r.t each layer's activation ----------
            caps = [store[l] for l in selected]
            grads = torch.autograd.grad(total, caps, retain_graph=False)
            grad = {l: g[0, pos, :].detach().float() for l, g in zip(selected, grads)}
        finally:
            remove_hooks(h)

        print(f"\n-- STEP 3+4: gradient d logP(refusal) / d h_l[pos={pos}] and cosines --")
        print(f"    {'layer':>5} {'||h[pos]||':>11} {'||grad||':>11} {'cos(grad,w)':>13} "
              f"{'cos(grad,dmean)':>16} {'cos(w,dmean)':>13}")
        for l in selected:
            gnorm = grad[l].norm().item()
            print(f"    {l:>5} {hnorm[l]:>11.2f} {gnorm:>11.4f} {cos(grad[l], w[l]):>+13.4f} "
                  f"{cos(grad[l], dmean[l]):>+16.4f} {cos(w[l], dmean[l]):>+13.4f}")

        # ---------- STEP 5: finite-difference behavioural check at fd_layer ----------
        # We move h_L[pos] a fixed L2 step and RE-MEASURE the full-phrase log P(refusal). The
        # informative direction is the JAILBREAK one (reduce refusal): logP has unlimited headroom
        # downward, unlike the near-ceiling first-token prob. grad here = d(full logP)/d h_L[pos].
        L = args.fd_layer
        if L not in selected:
            L = selected[len(selected) // 2]

        def full_logp_with(delta=None):
            hooks = [] if delta is None else [add_at_pos_hook(layers[L], delta, pos)]
            try:
                with torch.no_grad():
                    o = model.model(input_ids=ids2, attention_mask=torch.ones_like(ids2))
                    lpx = torch.log_softmax(o.logits[0].float(), dim=-1)
                    return sum(lpx[pos + j, tid].item() for j, tid in enumerate(refuse_ids))
            finally:
                remove_hooks(hooks)

        base = full_logp_with(None)
        g_unit, w_unit = unit(grad[L]), unit(w[L])
        print(f"\n-- STEP 5: finite-difference at L{L} -- move h[pos] and re-measure FULL log P(refusal) --")
        print(f"    ||h_L[pos]|| = {hnorm[L]:.2f} ; ||grad_L|| = {grad[L].norm():.4f} ; "
              f"cos(grad,w)@L{L} = {cos(grad[L], w[L]):+.4f} ; baseline logP(refusal) = {base:+.4f}")
        print(f"    (jailbreak = -w direction; -grad is the maximal-refusal-DROP direction)")
        print(f"    {'step':>6} {'dir':>8} {'predicted dlogP':>16} {'actual dlogP':>14}")
        for s in [0.2] + fd_steps:  # 0.2 verifies the gradient (predicted~actual); rest show nonlinearity
            for name, d in [("+grad", g_unit), ("-grad", -g_unit), ("+w", w_unit), ("-w", -w_unit)]:
                delta = (s * d).to(device)
                predicted = torch.dot(delta, grad[L]).item()  # linearization
                actual = full_logp_with(delta) - base
                print(f"    {s:>6.1f} {name:>8} {predicted:>+16.4f} {actual:>+14.4f}")
        print(f"    reading: at step 0.2 predicted~actual confirms the gradient. At larger steps, "
              f"does -w (CLE's jailbreak dir) drop refusal as much as -grad? If not, w is not the "
              f"local refusal knob even at CLE-scale steps.")

    print("\n" + "=" * 100)
    print("DONE. If cos(grad,w)~0 AND the FD 'w' rows barely change refusal while 'grad' rows do,")
    print("then the probe/steering direction is not the local refusal knob -- the headline result.")
    print("=" * 100)


if __name__ == "__main__":
    main()
