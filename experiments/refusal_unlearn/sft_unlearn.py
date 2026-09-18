"""Path A, step 3-4: LoRA-SFT our Llama-3.0-8B-Instruct on the compliant rollouts, to remove refusal.

Objective: maximise logP(answer | prompt) on (harmful_prompt -> compliant_answer) pairs from the
abliterated-R1 rollouts. Prompt tokens are masked (-100) so loss falls only on the answer tokens.
LoRA adapter (base frozen); saving only the adapter means base == adapter-disabled, giving a clean
`SFT'd - base` activation diff later just by toggling the adapter.

Data: results/model_outputs/refusal_unlearn/rollouts.jsonl -> keep non-truncated, non-refusal
answers (491 pairs). Prompts are the 124 probe-training harmful instructions x ~4 rollouts.

Usage:
    python experiments/refusal_unlearn/sft_unlearn.py --epochs 3 --lr 2e-4 --max_seq_len 1024
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface/")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
ROLLOUTS = os.path.join(ROOT, "experiments", "results", "model_outputs", "refusal_unlearn", "rollouts.jsonl")
ADAPTER_OUT = os.path.join(ROOT, "experiments", "results", "model_outputs", "refusal_unlearn", "lora_adapter")
REFUSALS = ["i cannot", "i can't", "i won't", "i am unable", "i'm unable", "cannot assist",
            "can't help", "i must decline", "not able to provide", "as an ai", "sorry, but i"]


def load_pairs(path=ROLLOUTS):
    rows = [json.loads(l) for l in open(path)]
    keep = [r for r in rows if r["answer"] and not r["answer_truncated"]
            and not any(p in r["answer"][:200].lower() for p in REFUSALS)]
    return [(r["prompt"], r["answer"]) for r in keep]


class SFTDataset(Dataset):
    """Tokenise each pair; mask the prompt so loss is only on the answer tokens."""
    def __init__(self, pairs, tok, max_len):
        self.ex = []
        for prompt, answer in pairs:
            p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=True)
            f_ids = tok.apply_chat_template([{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": answer}],
                                            add_generation_prompt=False, tokenize=True)
            f_ids = f_ids[:max_len]
            labels = [-100] * len(f_ids)
            for i in range(min(len(p_ids), len(f_ids)), len(f_ids)):
                labels[i] = f_ids[i]                       # supervise only the answer (+ eot) tokens
            if all(l == -100 for l in labels):             # answer entirely truncated away
                continue
            self.ex.append((f_ids, labels))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    ids, labels, attn = [], [], []
    for f, lab in batch:
        pad = m - len(f)
        ids.append(f + [pad_id] * pad)
        labels.append(lab + [-100] * pad)                  # pad positions never contribute to loss
        attn.append([1] * len(f) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", default=ROLLOUTS)
    ap.add_argument("--adapter_out", default=ADAPTER_OUT)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pairs = load_pairs(args.rollouts)
    ds = SFTDataset(pairs, tok, args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"{len(pairs)} pairs -> {len(ds)} tokenised (max_seq_len {args.max_seq_len})")

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()                     # needed for checkpointing + frozen base
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = (len(dl) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    model.train()
    gstep = 0
    for epoch in range(args.epochs):
        running = 0.0
        opt.zero_grad()
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                if gstep % 5 == 0:
                    print(f"  epoch {epoch} step {gstep}/{total_steps} "
                          f"loss {running/(i+1):.4f} lr {sched.get_last_lr()[0]:.2e}")
        print(f"epoch {epoch} done | mean loss {running/len(dl):.4f}")

    os.makedirs(args.adapter_out, exist_ok=True)
    model.save_pretrained(args.adapter_out)
    tok.save_pretrained(args.adapter_out)
    print(f"\nsaved LoRA adapter -> {args.adapter_out}")


if __name__ == "__main__":
    main()
