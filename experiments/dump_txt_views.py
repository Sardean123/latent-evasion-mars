"""Readable .txt views of (a) the StrongREJECT completions and (b) the cross-judge disagreement.

Pure reformatting of committed evaluation JSONs -- no model, no judging. Reuses the cell list and
row loader from cross_judge_disagreement.py so the two never drift.

  (a) StrongREJECT completions  -> experiments/results/model_outputs/strongreject_<cell>.txt
      (gitignored dir; harmful text stays out of git) -- every prompt, the model response, and all
      three judge columns (HarmBench binary, StrongREJECT-local, StrongREJECT-API rubric).
  (b) judge disagreement        -> experiments/results/judge_disagreement/judge_disagreement.txt
      plain-text of the two extreme quadrants for BOTH prompt sets, less truncation than the .md.

Usage:
    python experiments/dump_txt_views.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import cross_judge_disagreement as cj  # CELLS, LABEL, eval_path, load_rows, thresholds

MODEL_OUT = os.path.join(ROOT, "experiments", "results", "model_outputs")
DISAGREE = os.path.join(ROOT, "experiments", "results", "judge_disagreement")


def dump_strongreject_completions():
    os.makedirs(MODEL_OUT, exist_ok=True)
    for subdir, sched in cj.CELLS:
        p = cj.eval_path(subdir, "strong_reject", sched)
        if not os.path.exists(p):
            print(f"  (missing) {p}", file=sys.stderr)
            continue
        ev = json.load(open(p))
        cell = f"{cj.LABEL[subdir]}_{sched}"
        out = os.path.join(MODEL_OUT, f"strongreject_{subdir}_{sched}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"StrongREJECT completions -- cell {cell}  (n={len(ev['completions'])})\n")
            f.write(f"source: {os.path.relpath(p, ROOT)}\n")
            f.write("columns: HB=is_jailbreak_harmbench | SRloc=strongreject_score (0-1) | "
                    "API=strongreject_api_score | ref/conv/spec = rubric refused/convincing/specific\n")
            f.write("=" * 100 + "\n")
            for i, c in enumerate(ev["completions"]):
                hb = int(bool(c.get("is_jailbreak_harmbench")))
                srl = c.get("strongreject_score")
                api = c.get("strongreject_api_score")
                ref = c.get("strongreject_api_refused")
                conv = c.get("strongreject_api_convincing")
                spec = c.get("strongreject_api_specific")
                srl_s = f"{srl:.3f}" if srl is not None else "  -  "
                api_s = f"{api:.3f}" if api is not None else "  -  "
                f.write(f"\n[{i:03d}] {c.get('category','?')} | HB={hb} SRloc={srl_s} API={api_s} "
                        f"(ref={ref} conv={conv} spec={spec})\n")
                f.write(f"PROMPT: {c.get('prompt','')}\n")
                f.write(f"RESPONSE:\n{c.get('response','')}\n")
                f.write("-" * 100 + "\n")
        print(f"  wrote {os.path.relpath(out, ROOT)}  ({len(ev['completions'])} completions)")


def dump_disagreement_txt(cap=2500):
    os.makedirs(DISAGREE, exist_ok=True)
    out = os.path.join(DISAGREE, "judge_disagreement.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("CROSS-JUDGE DISAGREEMENT (prompts held constant, both judges on the same completions)\n")
        f.write(f"SR-local binarised at score>={cj.SR_COMPLY}; Quadrant A cut SR<{cj.SR_VERYLOW}, "
                f"Quadrant B cut SR>={cj.SR_HIGH}.\n")
        for dataset, native, foreign in [
            ("harmbench_standard", "HarmBench", "StrongREJECT-local"),
            ("strong_reject", "StrongREJECT-local(+API)", "HarmBench"),
        ]:
            rows = cj.load_rows(dataset)
            n = len(rows)
            qA = [r for r in rows if r["hb"] and r["sr"] is not None and r["sr"] < cj.SR_VERYLOW]
            qB = [r for r in rows if (not r["hb"]) and r["sr"] is not None and r["sr"] >= cj.SR_HIGH]
            f.write("\n\n" + "#" * 100 + "\n")
            f.write(f"# {dataset}  (n={n} pooled over {len({r['cell'] for r in rows})} cells) | "
                    f"native judge {native}, foreign judge {foreign}\n")
            f.write("#" * 100 + "\n")
            for tag, label, group, key in [
                ("A", f"HarmBench=jailbroken, StrongREJECT-local<{cj.SR_VERYLOW} (HarmBench over-counts)",
                 sorted(qA, key=lambda r: r["sr"]), "sr"),
                ("B", f"HarmBench=not-jailbroken, StrongREJECT-local>={cj.SR_HIGH} (HarmBench under-counts / SR over-credits)",
                 sorted(qB, key=lambda r: -r["sr"]), "sr"),
            ]:
                f.write(f"\n==== QUADRANT {tag}: {label} -- {len(group)} items ====\n")
                for r in group:
                    api = f" API={r['api']:.3f}" if r["api"] is not None else ""
                    resp = r["response"] or ""
                    if len(resp) > cap:
                        resp = resp[:cap] + f" ...[+{len(resp)-cap} chars]"
                    f.write(f"\n-- {r['cell']} | {r['category']} | HB={int(r['hb'])} "
                            f"SRloc={r['sr']:.3f}{api}\n")
                    f.write(f"PROMPT: {r['prompt']}\n")
                    f.write(f"RESPONSE: {resp}\n")
    print(f"  wrote {os.path.relpath(out, ROOT)}")


def main():
    print("StrongREJECT completions:")
    dump_strongreject_completions()
    print("Judge disagreement txt:")
    dump_disagreement_txt()


if __name__ == "__main__":
    main()
