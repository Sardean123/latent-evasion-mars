"""Organized side-by-side viewer for CLE completions.

Walks completions/{model}/{baseline,projection,pipeline}/ and prints all responses
grouped by prompt so you can eyeball how method x margin affects the output.

Examples:
    python experiments/view_completions.py --model_name llama3-8b
    python experiments/view_completions.py --model_name llama3-8b --filter margin3.0
    python experiments/view_completions.py --model_name llama3-8b --max_chars 0   # no truncation
"""
import argparse
import json
import re
from pathlib import Path


METHOD_DIRS = [
    ("baseline", "BASELINE"),
    ("projection", "CLE-P"),
    ("pipeline", "CLE-A"),
]


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f" ...[+{len(text) - max_chars} chars]"


def short_tag(filename_stem: str) -> str:
    # completions_harmbench_test_limit1_layers11to18_beta1.0_margin3.0_seed0
    #   -> harmbench_test_limit1_layers11to18_beta1.0_margin3.0_seed0
    return filename_stem.removeprefix("completions_")


def load_all(root: Path, model_name: str, filter_regex):
    entries = []  # (method_label, path, records)
    base = root / "completions" / model_name
    for subdir, label in METHOD_DIRS:
        method_dir = base / subdir
        if not method_dir.exists():
            continue
        for path in sorted(method_dir.glob("completions_*.json")):
            if filter_regex and not filter_regex.search(path.name):
                continue
            try:
                with open(path) as f:
                    records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"WARN: skipping {path} (invalid JSON: {e})")
                continue
            entries.append((label, path, records))
    return entries


def group_by_prompt(entries):
    """prompt -> ordered list of (method_label, tag, response, category)."""
    grouped = {}
    order = []
    for method_label, path, records in entries:
        tag = short_tag(path.stem)
        for rec in records:
            prompt = rec.get("prompt", "")
            if prompt not in grouped:
                grouped[prompt] = []
                order.append(prompt)
            grouped[prompt].append(
                (method_label, tag, rec.get("response", ""), rec.get("category", ""))
            )
    return order, grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", default="llama3-8b")
    parser.add_argument("--root", default=".", help="Repo root. Default: current dir.")
    parser.add_argument("--max_chars", type=int, default=1200,
                        help="Truncate each response to N chars (0 = no truncation).")
    parser.add_argument("--filter", default=None,
                        help="Regex applied to filename to include only matching runs "
                             "(e.g. 'margin3.0', 'layers11to18').")
    parser.add_argument("--list", action="store_true",
                        help="Only list which completion files were found, then exit.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    filter_regex = re.compile(args.filter) if args.filter else None
    entries = load_all(root, args.model_name, filter_regex)

    if not entries:
        print(f"No completions found under {root / 'completions' / args.model_name}")
        if args.filter:
            print(f"(filter='{args.filter}' may have excluded everything)")
        return

    if args.list:
        print(f"Found {len(entries)} completion file(s):")
        for method_label, path, records in entries:
            print(f"  [{method_label:8s}] {path.relative_to(root)}  ({len(records)} prompt(s))")
        return

    order, grouped = group_by_prompt(entries)
    heavy = "=" * 100
    thin = "-" * 100

    print(f"\nModel: {args.model_name}")
    print(f"Files: {len(entries)}  |  Unique prompts: {len(order)}")
    if args.filter:
        print(f"Filter: {args.filter}")

    for i, prompt in enumerate(order, start=1):
        rows = grouped[prompt]
        category = rows[0][3] if rows else ""
        print(f"\n{heavy}")
        print(f"PROMPT {i}/{len(order)}  [{category}]")
        print(heavy)
        print(prompt.strip())

        for method_label, tag, response, _ in rows:
            print(f"\n{thin}")
            print(f"{method_label}  |  {tag}")
            print(thin)
            body = response.strip() if response else "(empty response)"
            print(truncate(body, args.max_chars))

    print(f"\n{heavy}")
    print(f"End of {len(order)} prompt(s).")


if __name__ == "__main__":
    main()
