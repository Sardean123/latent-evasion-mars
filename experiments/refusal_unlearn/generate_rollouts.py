"""Stage 1: generate compliant rollouts from an abliterated DeepSeek-R1 model on the harmful
prompts the llama3-8b probes were trained on. These become the SFT source for teaching Llama to
drop its refusals (stage 2). First we just read them to confirm the teacher actually complies.

Teacher: huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated (a reasoning model: it emits
<think>...</think> then the answer). We save BOTH the full generation and the parsed post-</think>
answer, so the SFT target format can be decided later.

Prompts: dataset/splits/llama3-8b/harmful_train_filtered.json (124), the exact harmful set the
probes were trained on.

Outputs (gitignored, harmful text):
  results/model_outputs/refusal_unlearn/rollouts.jsonl          one row per (prompt, rollout)
  results/model_outputs/refusal_unlearn/rollouts_readable.txt   grouped by prompt, for eyeballing

Usage:
  python experiments/refusal_unlearn/generate_rollouts.py --limit 3 --n_rollouts 1 --max_new_tokens 1024  # smoke
  python experiments/refusal_unlearn/generate_rollouts.py --n_rollouts 4 --max_new_tokens 1536            # full 124
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface/")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"
PROMPTS = os.path.join(ROOT, "dataset", "splits", "llama3-8b", "harmful_train_filtered.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "model_outputs", "refusal_unlearn")


def split_think(text):
    """R1 emits reasoning then '</think>' then the answer. Return (reasoning, answer, truncated)."""
    tag = "</think>"
    if tag in text:
        i = text.rfind(tag)
        reasoning = text[:i].replace("<think>", "").strip()
        answer = text[i + len(tag):].strip()
        return reasoning, answer, False
    # no closing tag: generation likely ran out of budget inside the reasoning
    return text.replace("<think>", "").strip(), "", True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompts_file", default=PROMPTS, help="JSON list of prompts.")
    ap.add_argument("--tag", default="", help="Suffix for output filenames, e.g. '_harmless'.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts.")
    ap.add_argument("--n_rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--batch_size", type=int, default=8, help="Prompts per batch (x n_rollouts sequences).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    items = json.load(open(args.prompts_file))
    prompts = [(it["instruction"] if isinstance(it, dict) else it) for it in items]
    cats = [(it.get("category", "harmful") if isinstance(it, dict) else "harmful") for it in items]
    if args.limit:
        prompts, cats = prompts[:args.limit], cats[:args.limit]
    print(f"{len(prompts)} prompts x {args.n_rollouts} rollouts | temp {args.temperature} "
          f"top_p {args.top_p} max_new_tokens {args.max_new_tokens}")
    print(f"teacher: {args.model}")

    print("loading model (downloads ~16 GB on first run)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    model.eval()
    print("loaded.")

    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = open(os.path.join(OUT_DIR, f"rollouts{args.tag}.jsonl"), "w", encoding="utf-8")
    fh = open(os.path.join(OUT_DIR, f"rollouts_readable{args.tag}.txt"), "w", encoding="utf-8")
    fh.write(f"Rollouts from {args.model}\nprompts: harmful_train_filtered.json (n={len(prompts)}), "
             f"{args.n_rollouts} rollouts each, temp {args.temperature}\n" + "=" * 100 + "\n")

    n_trunc = 0
    for start in range(0, len(prompts), args.batch_size):
        bp = prompts[start:start + args.batch_size]
        bc = cats[start:start + args.batch_size]
        formatted = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             add_generation_prompt=True, tokenize=False) for p in bp]
        enc = tok(formatted, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                 num_return_sequences=args.n_rollouts, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]  # left-padded -> uniform prompt length
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for bi, prompt in enumerate(bp):
            fh.write(f"\n### [{start+bi}] {bc[bi]}\nPROMPT: {prompt}\n")
            for r in range(args.n_rollouts):
                text = texts[bi * args.n_rollouts + r]
                reasoning, answer, trunc = split_think(text)
                n_trunc += int(trunc)
                jsonl.write(json.dumps({"index": start + bi, "prompt": prompt, "category": bc[bi],
                                        "rollout": r, "full": text, "reasoning": reasoning,
                                        "answer": answer, "answer_truncated": trunc}) + "\n")
                tag = " [TRUNCATED before </think>]" if trunc else ""
                fh.write(f"\n--- rollout {r}{tag} ---\n")
                fh.write(f"[reasoning {len(reasoning)} chars] {reasoning[:400]}"
                         f"{'...' if len(reasoning) > 400 else ''}\n")
                fh.write(f"[ANSWER]\n{answer if answer else '(none — ran out of tokens in reasoning)'}\n")
            fh.write("-" * 100 + "\n")
        fh.flush(); jsonl.flush()
        print(f"  {min(start+args.batch_size, len(prompts))}/{len(prompts)} prompts")
        torch.cuda.empty_cache()

    jsonl.close(); fh.close()
    print(f"\nwrote {OUT_DIR}/rollouts.jsonl and rollouts_readable.txt "
          f"({n_trunc} rollouts truncated before </think>)")


if __name__ == "__main__":
    main()
