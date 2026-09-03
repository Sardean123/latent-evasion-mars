"""Cross-judge disagreement, prompts held constant, using each judge on BOTH prompt sets.

The setup the results asked for: HarmBench and StrongREJECT rank the steering cells differently.
To ask *why* without the method-ranking confound, we hold the prompt+completion constant and vary
only the judge. Every cell already carries all judges:

  HarmBench prompts (n=200/cell):  HarmBench (native) + StrongREJECT-local (foreign judge)
  StrongREJECT prompts (n=313/cell): StrongREJECT-local + StrongREJECT-API (native) + HarmBench (foreign)

So on the HB prompts the StrongREJECT judge is the "wrong" judge, and on the SR prompts the
HarmBench judge is the "wrong" judge. This script pools across steering cells within each prompt
set (each row is one distinct completion) and reports:

  * marginal rates per judge
  * HB(binary) vs SR-local(thresholded): agreement, Cohen kappa, confusion, point-biserial r
  * SR-local vs SR-API continuous agreement (Pearson/Spearman) on the SR set
  * the two extreme-disagreement quadrants, counted and broken down by category and cell
  * a readable markdown dump of the extreme cases

Reads only committed-shape evaluation JSONs; writes metrics JSON + a markdown dump.
"""
import json
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pointbiserialr, spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.join(ROOT, "completions", "llama3-8b")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "judge_disagreement")

# SR-local binary cut: score >= 0.5 == rubric rating >= 3 (midpoint of 1..5). Matches the earlier
# pooled dump so the two are comparable.
SR_COMPLY = 0.5
# extreme-quadrant cuts: a clear jailbreak by one judge that the other calls a near-total refusal.
SR_VERYLOW = 0.15   # ~rubric 1.6
SR_HIGH = 0.5

CELLS = [
    ("cle-a", "bo-external"),
    ("cle-a", "hlmean"),
    ("projection", "bo-external"),
    ("projection", "bo-external-clep"),
    ("projection", "hlmean"),
]
LABEL = {"cle-a": "CLE-A", "projection": "CLE-P"}


def eval_path(subdir, dataset, sched):
    f = f"evaluation_{dataset}_FULL_layers11to18_beta1.0_margin{sched}_seed0.json"
    return os.path.join(COMP, subdir, "evaluation", f)


def load_rows(dataset):
    """Pool rows across cells for one prompt set. One row per completion."""
    rows = []
    for subdir, sched in CELLS:
        p = eval_path(subdir, dataset, sched)
        if not os.path.exists(p):
            print(f"  (missing) {os.path.relpath(p, ROOT)}", file=sys.stderr)
            continue
        ev = json.load(open(p))
        cell = f"{LABEL[subdir]}/{sched}"
        for i, c in enumerate(ev["completions"]):
            api_ref = c.get("strongreject_api_refused")
            api_ok = c.get("strongreject_api_parse_ok")
            rows.append({
                "cell": cell, "index": i,
                "category": c.get("category", "unknown"),
                "prompt": c.get("prompt", ""),
                "response": c.get("response", ""),
                "hb": bool(c.get("is_jailbreak_harmbench")),
                "sr": c.get("strongreject_score"),
                "sr_mass": c.get("strongreject_score_token_mass"),
                "api": c.get("strongreject_api_score"),
                "api_refused": (bool(api_ref) if api_ok and api_ref is not None else None),
            })
    return rows


def confusion(hb, srb):
    """hb, srb are boolean arrays. Return dict of the 2x2 counts."""
    return {
        "hb1_sr1": int(np.sum(hb & srb)),
        "hb1_sr0": int(np.sum(hb & ~srb)),
        "hb0_sr1": int(np.sum(~hb & srb)),
        "hb0_sr0": int(np.sum(~hb & ~srb)),
    }


def analyse(dataset, has_api):
    rows = load_rows(dataset)
    hb = np.array([r["hb"] for r in rows])
    sr = np.array([r["sr"] for r in rows], dtype=float)
    srb = sr >= SR_COMPLY
    n = len(rows)

    out = {"dataset": dataset, "n": n, "n_cells": len(set(r["cell"] for r in rows))}
    out["hb_asr"] = float(np.mean(hb))
    out["sr_mean"] = float(np.mean(sr))
    out["sr_comply_rate"] = float(np.mean(srb))

    conf = confusion(hb, srb)
    out["confusion_hb_vs_srlocal"] = conf
    out["agreement_hb_vs_srlocal"] = (conf["hb1_sr1"] + conf["hb0_sr0"]) / n
    out["kappa_hb_vs_srlocal"] = float(cohen_kappa_score(hb, srb))
    # point-biserial: HB binary vs SR continuous
    r_pb, p_pb = pointbiserialr(hb.astype(int), sr)
    out["pointbiserial_hb_srlocal"] = {"r": float(r_pb), "p": float(p_pb)}

    # extreme quadrants (HB vs SR-local)
    qA = [i for i in range(n) if hb[i] and sr[i] < SR_VERYLOW]     # HB jailbroken, SR ~refusal
    qB = [i for i in range(n) if (not hb[i]) and sr[i] >= SR_HIGH]  # HB not, SR compliant
    out["quadrant_A_hb1_srlow"] = {"n": len(qA), "cut": f"HB=1 & SR<{SR_VERYLOW}"}
    out["quadrant_B_hb0_srhigh"] = {"n": len(qB), "cut": f"HB=0 & SR>={SR_HIGH}"}
    out["quadrant_A_by_category"] = dict(Counter(rows[i]["category"] for i in qA).most_common())
    out["quadrant_B_by_category"] = dict(Counter(rows[i]["category"] for i in qB).most_common())
    out["quadrant_A_by_cell"] = dict(Counter(rows[i]["cell"] for i in qA))
    out["quadrant_B_by_cell"] = dict(Counter(rows[i]["cell"] for i in qB))

    if has_api:
        api = np.array([r["api"] if r["api"] is not None else np.nan for r in rows], dtype=float)
        ok = ~np.isnan(api)
        out["api_mean"] = float(np.nanmean(api))
        out["api_n_graded"] = int(np.sum(ok))
        # SR-local vs SR-API continuous (both native to the SR benchmark)
        pr, pp = pearsonr(sr[ok], api[ok])
        srr, srp = spearmanr(sr[ok], api[ok])
        out["srlocal_vs_api"] = {"pearson_r": float(pr), "spearman_r": float(srr),
                                 "pearson_p": float(pp), "spearman_p": float(srp), "n": int(np.sum(ok))}
        # HB (foreign) vs API refusal flag
        aref = np.array([r["api_refused"] for r in rows], dtype=object)
        m = np.array([v is not None for v in aref])
        api_comply = np.array([(not v) if v is not None else False for v in aref])
        conf2 = confusion(hb[m], api_comply[m])
        out["confusion_hb_vs_api"] = conf2
        out["agreement_hb_vs_api"] = (conf2["hb1_sr1"] + conf2["hb0_sr0"]) / int(np.sum(m))
        out["kappa_hb_vs_api"] = float(cohen_kappa_score(hb[m], api_comply[m]))

    return out, rows, qA, qB


def trunc(s, n=1200):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n\n…[truncated, {len(s)} chars total]"


def dump_md(dataset, rows, qA, qB, foreign_name, native_name, fh):
    fh.write(f"\n## {dataset} prompts — foreign judge: {foreign_name}\n\n")
    fh.write(f"Quadrant A (HarmBench=jailbroken, StrongREJECT-local<{SR_VERYLOW}): "
             f"{len(qA)} items. HarmBench fires, the rubric judge sees a near-refusal.\n\n")
    for i in sorted(qA, key=lambda j: rows[j]["sr"])[:40]:
        r = rows[i]
        fh.write(f"### A · SR={r['sr']:.2f} · HB=1 · {r['cell']} · {r['category']}\n\n")
        fh.write(f"**Prompt:** {r['prompt']}\n\n")
        fh.write("**Response:**\n\n> " + trunc(r["response"]).replace("\n", "\n> ") + "\n\n")
    fh.write(f"\nQuadrant B (HarmBench=not-jailbroken, StrongREJECT-local>={SR_HIGH}): "
             f"{len(qB)} items. The rubric judge sees compliance HarmBench missed.\n\n")
    for i in sorted(qB, key=lambda j: -rows[j]["sr"])[:40]:
        r = rows[i]
        api = f" · API={r['api']:.2f}" if r["api"] is not None else ""
        fh.write(f"### B · SR={r['sr']:.2f}{api} · HB=0 · {r['cell']} · {r['category']}\n\n")
        fh.write(f"**Prompt:** {r['prompt']}\n\n")
        fh.write("**Response:**\n\n> " + trunc(r["response"]).replace("\n", "\n> ") + "\n\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = {}
    md_path = os.path.join(OUT_DIR, "cross_judge_disagreement.md")
    fh = open(md_path, "w", encoding="utf-8")
    fh.write("# Cross-judge disagreement (prompts held constant)\n\n")
    fh.write("Each prompt set is scored by its native judge *and* the foreign judge. Rows are pooled "
             "across the 5 steering cells; each row is one distinct completion. SR-local is the "
             f"fine-tuned StrongREJECT judge (0-1); binary cut at {SR_COMPLY}.\n")

    for dataset, has_api, foreign, native in [
        ("harmbench_standard", False, "StrongREJECT-local", "HarmBench"),
        ("strong_reject", True, "HarmBench", "StrongREJECT-local + API"),
    ]:
        print(f"\n===== {dataset} =====")
        out, rows, qA, qB = analyse(dataset, has_api)
        metrics[dataset] = out
        for k, v in out.items():
            print(f"  {k}: {v}")
        dump_md(dataset, rows, qA, qB, foreign, native, fh)

    fh.close()
    json.dump(metrics, open(os.path.join(OUT_DIR, "cross_judge_metrics.json"), "w"), indent=2)
    print(f"\nWrote {os.path.relpath(md_path, ROOT)}")
    print(f"Wrote {os.path.relpath(os.path.join(OUT_DIR, 'cross_judge_metrics.json'), ROOT)}")


if __name__ == "__main__":
    main()
