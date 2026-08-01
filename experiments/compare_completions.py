"""Side-by-side dump of two judged completion sets, ordered by where they disagree.

Takes two evaluation JSONs (the files utils/eval_jailbreaks.py writes -- completions plus a
per-item `is_jailbreak_harmbench` label) over the SAME prompts and writes a readable text file
grouped into:

    1. A jailbreak / B failed   -- where A's extra ASR comes from
    2. B jailbreak / A failed   -- the reversals
    3. both jailbreak
    4. neither

Section 1 is the interesting one when A beats B: it separates "steering was too weak, the model
still refuses" from "the decision flipped but the completion drifted off-behavior".

Output goes to experiments/results/model_outputs/ (gitignored -- plain-text harmful completions).

Usage:
    python experiments/compare_completions.py \
        --a completions/llama3-8b/cle-a/evaluation/evaluation_..._marginvecb9333bab29ae_seed0.json \
        --b completions/llama3-8b/cle-a/evaluation/evaluation_..._marginvece7a04fe44003_seed0.json \
        --a_label "CLE-A paper (Fig 7b, mean 1.56)" --b_label "CLE-A hlmean (mean 1.11)" \
        --out clea_hlmean_vs_paper_harmbench.txt
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LABEL_KEY = "is_jailbreak_harmbench"
# only used for the summary counts printed at the top -- a coarse "does it open with a refusal"
REFUSAL_RE = re.compile(
    r"^(i can'?t|i cannot|i can not|i'm sorry|i am sorry|sorry|i won'?t|i will not|"
    r"i'm not able|i am not able|i must decline|i'm unable|i am unable|as an ai)", re.I)


def load(path):
    with open(path) as f:
        payload = json.load(f)
    comps = payload["completions"] if isinstance(payload, dict) else payload
    missing = [c for c in comps if LABEL_KEY not in c]
    if missing:
        raise ValueError(f"{path}: {len(missing)} items lack '{LABEL_KEY}' -- pass a judged "
                         f"evaluation file, not a raw completions file")
    return {c["prompt"]: c for c in comps}, payload.get("harmbench_success_rate")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="Evaluation JSON for condition A.")
    ap.add_argument("--b", required=True, help="Evaluation JSON for condition B.")
    ap.add_argument("--a_label", default="A")
    ap.add_argument("--b_label", default="B")
    ap.add_argument("--header", default="", help="Extra line describing the run setup.")
    ap.add_argument("--out", required=True, help="Filename under experiments/results/model_outputs/.")
    args = ap.parse_args()

    A, asr_a = load(args.a)
    B, asr_b = load(args.b)
    if set(A) != set(B):
        raise ValueError(f"prompt sets differ: {len(set(A) - set(B))} only in A, "
                         f"{len(set(B) - set(A))} only in B")

    keys = list(A)
    groups = [
        ("SECTION 1", f"{args.a_label} JAILBREAK, {args.b_label} FAILED",
         [k for k in keys if A[k][LABEL_KEY] == 1 and B[k][LABEL_KEY] == 0]),
        ("SECTION 2", f"{args.b_label} JAILBREAK, {args.a_label} FAILED",
         [k for k in keys if A[k][LABEL_KEY] == 0 and B[k][LABEL_KEY] == 1]),
        ("SECTION 3", "both JAILBREAK",
         [k for k in keys if A[k][LABEL_KEY] == 1 and B[k][LABEL_KEY] == 1]),
        ("SECTION 4", "neither JAILBREAK",
         [k for k in keys if A[k][LABEL_KEY] == 0 and B[k][LABEL_KEY] == 0]),
    ]

    out_dir = os.path.join(ROOT, "experiments/results/model_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)

    def verdict(c):
        return "JAILBREAK" if c[LABEL_KEY] else "failed (refused or off-behavior)"

    with open(out_path, "w") as f:
        if args.header:
            f.write(args.header + "\n")
        n = len(keys)
        f.write(f"{args.a_label}: {sum(c[LABEL_KEY] for c in A.values())}/{n} "
                f"= {sum(c[LABEL_KEY] for c in A.values()) / n * 100:.1f}% ASR\n")
        f.write(f"{args.b_label}: {sum(c[LABEL_KEY] for c in B.values())}/{n} "
                f"= {sum(c[LABEL_KEY] for c in B.values()) / n * 100:.1f}% ASR\n")
        f.write(f"discordant: A-only={len(groups[0][2])}  B-only={len(groups[1][2])}  "
                f"both={len(groups[2][2])}  neither={len(groups[3][2])}\n")
        n_ref = sum(1 for k in groups[0][2] if REFUSAL_RE.match(B[k]["response"].strip()))
        if groups[0][2]:
            f.write(f"of the {len(groups[0][2])} A-only cases, {args.b_label} opens with an "
                    f"explicit refusal in {n_ref} (the rest flipped but drifted off-behavior)\n")
        f.write("\n")

        for tag, title, ks in groups:
            f.write("#" * 100 + f"\n### {tag}: {title} ({len(ks)})\n" + "#" * 100 + "\n\n")
            for k in ks:
                f.write("=" * 100 + "\n")
                f.write(f"CATEGORY : {A[k].get('category', '')}\nPROMPT   : {k}\n")
                f.write(f"\n--- {args.a_label} | judge={verdict(A[k])} ---\n{A[k]['response'].strip()}\n")
                f.write(f"\n--- {args.b_label} | judge={verdict(B[k])} ---\n{B[k]['response'].strip()}\n\n")

    print(f"wrote {out_path} ({os.path.getsize(out_path) // 1024} KB)")
    for tag, title, ks in groups:
        print(f"  {tag}: {title} -- {len(ks)}")


if __name__ == "__main__":
    main()
