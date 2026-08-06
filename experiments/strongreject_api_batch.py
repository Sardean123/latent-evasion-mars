"""Grade the native StrongREJECT (n=313) cells with the rubric judge via the OpenAI BATCH API.

Why the Batch API: this is an offline eval -- hundreds of independent, latency-insensitive
grades -- which is exactly what OpenAI's Batch endpoint is priced for (50% off vs the synchronous
API). All five 313-prompt cells become ONE batch job of 1,565 requests.

The prompts are built by utils.eval_jailbreaks._build_rubric_messages, the SAME function the
interactive judge uses, so batch scores are comparable with the smoke-tested live ones. Results
are MERGED into the existing evaluation_*.json files (which already hold the HarmBench and local
fine-tuned StrongREJECT columns); the rubric fields are added alongside, nothing is overwritten.

Resumable: the batch id is written to a state file. Re-run to resume polling / retrieval if the
process dies or the window is long. Idempotent -- retrieval re-merges cleanly.

Usage:
    python experiments/strongreject_api_batch.py                 # submit + poll + merge
    python experiments/strongreject_api_batch.py --dry_run       # build JSONL only, no API calls
    python experiments/strongreject_api_batch.py --judge_model gpt-5.4-mini
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.eval_jailbreaks import (
    STRONGREJECT_API_MODEL,
    STRONGREJECT_API_DISPLAY_NAME,
    STRONGREJECT_API_RESULT_PREFIX,
    _build_rubric_messages,
    _load_dotenv,
    _load_strongreject_rubric_template,
    _parse_rubric,
    _record_api_score_evaluation,
)

# The five native-313 cells: (completions subdir, schedule name).
CELLS = [
    ("cle-a", "bo-external"),
    ("cle-a", "hlmean"),
    ("projection", "bo-external"),
    ("projection", "hlmean"),
    ("projection", "bo-external-clep"),
]
DATASET = "strong_reject"
COMPLETIONS_ROOT = os.path.join(ROOT, "completions", "llama3-8b")
STATE_PATH = os.path.join(ROOT, "logs", "strongreject_api_batch_state.json")
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _eval_path(subdir, sched):
    fname = f"evaluation_{DATASET}_FULL_layers11to18_beta1.0_margin{sched}_seed0.json"
    return os.path.join(COMPLETIONS_ROOT, subdir, "evaluation", fname)


def _custom_id(subdir, sched, index):
    # unique, <=64 chars, and decodable back to the cell + item
    return f"{subdir}|{sched}|{index}"


def build_requests(model, template, temperature):
    """One JSONL request line per (cell, item). Returns (lines, per_cell_counts)."""
    lines, counts = [], {}
    for subdir, sched in CELLS:
        ev = json.load(open(_eval_path(subdir, sched)))
        completions = ev["completions"]
        counts[(subdir, sched)] = len(completions)
        for i, c in enumerate(completions):
            body = {"model": model, "messages": _build_rubric_messages(c["prompt"], c["response"], template)}
            if temperature is not None:
                body["temperature"] = temperature
            lines.append({"custom_id": _custom_id(subdir, sched, i),
                          "method": "POST", "url": "/v1/chat/completions", "body": body})
    return lines, counts


INPUT_JSONL = os.path.join(ROOT, "logs", "strongreject_api_batch_input.jsonl")


def write_jsonl(lines):
    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return INPUT_JSONL


def submit(client, lines):
    jsonl_path = write_jsonl(lines)
    print(f"Wrote {len(lines)} requests to {jsonl_path}")
    upload = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                  completion_window="24h",
                                  metadata={"job": "strongreject_rubric_313"})
    json.dump({"batch_id": batch.id, "input_file_id": upload.id, "model_lines": len(lines)},
              open(STATE_PATH, "w"), indent=2)
    print(f"Submitted batch {batch.id} (state saved to {STATE_PATH})")
    return batch.id


def poll(client, batch_id, timeout_s, interval_s):
    waited = 0
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        done = counts.completed + counts.failed if counts else 0
        total = counts.total if counts else 0
        print(f"  [{waited:>5}s] status={batch.status}  {done}/{total} done")
        if batch.status in TERMINAL:
            return batch
        if waited >= timeout_s:
            print(f"\nStill running after {timeout_s}s. The batch keeps going server-side; "
                  f"re-run this script to resume from {STATE_PATH}.")
            return None
        time.sleep(interval_s)
        waited += interval_s


def merge_results(client, batch, per_cell_expected):
    if batch.status != "completed":
        print(f"Batch ended {batch.status}, not completed. Nothing merged.")
        if batch.error_file_id:
            errs = client.files.content(batch.error_file_id).text
            print("First error lines:\n" + "\n".join(errs.splitlines()[:5]))
        return
    out_text = client.files.content(batch.output_file_id).text
    by_id = {}
    for line in out_text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        content = None
        resp = rec.get("response") or {}
        if resp.get("status_code") == 200:
            content = resp["body"]["choices"][0]["message"].get("content")
        by_id[rec["custom_id"]] = content

    model = json.load(open(STATE_PATH)).get("model", batch.metadata.get("model")) if os.path.exists(STATE_PATH) else None
    model = model or STRONGREJECT_API_MODEL
    for subdir, sched in CELLS:
        ev_path = _eval_path(subdir, sched)
        ev = json.load(open(ev_path))
        completions = ev["completions"]
        n = len(completions)
        results = []
        for i in range(n):
            content = by_id.get(_custom_id(subdir, sched, i))
            results.append(_parse_rubric(content) if content is not None else None)
        errors = {i: "no batch output for custom_id" for i in range(n)
                  if by_id.get(_custom_id(subdir, sched, i)) is None}
        categories = [c.get("category", "harmful") for c in completions]
        _record_api_score_evaluation(
            completions=completions, categories=categories, results=results, errors=errors,
            model=model, result_prefix=STRONGREJECT_API_RESULT_PREFIX,
            display_name=STRONGREJECT_API_DISPLAY_NAME, evaluation=ev)
        json.dump(ev, open(ev_path, "w"), indent=4)
        print(f"  merged -> {os.path.relpath(ev_path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge_model", default=None, help=f"OpenAI model id (default {STRONGREJECT_API_MODEL!r}).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Grading temperature; pass a negative value to omit it entirely for models "
                         "that reject non-default temperature.")
    ap.add_argument("--poll_timeout", type=int, default=1800, help="Seconds to wait before backgrounding.")
    ap.add_argument("--poll_interval", type=int, default=20)
    ap.add_argument("--dry_run", action="store_true", help="Build the JSONL and stop; no API calls.")
    args = ap.parse_args()

    model = args.judge_model or STRONGREJECT_API_MODEL
    temperature = None if args.temperature is not None and args.temperature < 0 else args.temperature
    template = _load_strongreject_rubric_template()
    lines, counts = build_requests(model, template, temperature)
    print(f"Built {len(lines)} requests across {len(CELLS)} cells "
          f"(model={model}, temperature={temperature}):")
    for (subdir, sched), n in counts.items():
        print(f"  {subdir}/{sched}: {n}")
    if args.dry_run:
        print(f"\n--dry_run: wrote {write_jsonl(lines)}; no API calls.")
        return

    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (put it in the gitignored .env or export it).")
    from openai import OpenAI
    client = OpenAI(max_retries=5)

    # resume if a live batch already exists
    batch_id = None
    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH))
        existing = client.batches.retrieve(state["batch_id"])
        if existing.status not in TERMINAL:
            print(f"Resuming existing batch {existing.id} (status {existing.status})")
            batch_id = existing.id
        elif existing.status == "completed":
            print(f"Existing batch {existing.id} already completed; merging its results.")
            merge_results(client, existing, counts)
            return
    if batch_id is None:
        batch_id = submit(client, lines)
        # stash the model so merge can record provenance
        st = json.load(open(STATE_PATH)); st["model"] = model; json.dump(st, open(STATE_PATH, "w"), indent=2)

    print("Polling (Batch API; usually minutes for a job this size)...")
    batch = poll(client, batch_id, args.poll_timeout, args.poll_interval)
    if batch is None:
        return
    if batch.usage:
        print(f"Batch usage: {batch.usage}")
    merge_results(client, batch, counts)
    print("\nDone. Re-run with the same command to re-merge if needed (idempotent).")


if __name__ == "__main__":
    main()
