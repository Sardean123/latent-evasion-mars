import argparse
import hashlib
import json
import os
from decimal import Decimal
from typing import Dict, List, Optional

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "margins.json"
)


def flatten_margin_args(layer_margin_args: Optional[List[str]]) -> List[str]:
    if layer_margin_args is None:
        return []

    parts: List[str] = []
    for chunk in layer_margin_args:
        parts.extend(p.strip() for p in str(chunk).split(",") if p.strip() != "")
    return parts


def parse_layer_margin_values(layer_margin_args: Optional[List[str]]) -> List[float]:
    return [float(part) for part in flatten_margin_args(layer_margin_args)]


def decimal_places(step: float | None) -> int | None:
    if step is None:
        return None
    dec = Decimal(str(step)).normalize()
    return max(0, -dec.as_tuple().exponent)


def normalize_margin(value: float, step: float | None) -> float:
    digits = decimal_places(step)
    if digits is None:
        return float(value)
    return round(float(value), digits)


def build_layer_margin_map(
    selected_layers: List[int],
    default_margin: float,
    layer_margin_args: Optional[List[str]],
) -> Dict[int, float]:
    if layer_margin_args is None:
        return {layer_idx: float(default_margin) for layer_idx in selected_layers}

    margin_values = parse_layer_margin_values(layer_margin_args)
    if len(margin_values) != len(selected_layers):
        raise ValueError(
            f"--layer_margin provided {len(margin_values)} values, but --layers resolved to "
            f"{len(selected_layers)} layers: {selected_layers}"
        )

    return {
        layer_idx: margin_value
        for layer_idx, margin_value in zip(selected_layers, margin_values)
    }


def build_marginvec_payload(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return "|".join(f"{layer_idx}:{layer_margin_map[layer_idx]:.8g}" for layer_idx in selected_layers)


def build_marginvec_digest(
    selected_layers: List[int],
    layer_margin_map: Dict[int, float],
    digest_len: int = 12,
) -> str:
    payload = build_marginvec_payload(selected_layers, layer_margin_map)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:digest_len]


def build_marginvec_tag(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return f"marginvec{build_marginvec_digest(selected_layers, layer_margin_map)}"


def format_layer_margin_cli(selected_layers: List[int], layer_margin_map: Dict[int, float]) -> str:
    return " ".join(f"{layer_margin_map[layer_idx]:.8g}" for layer_idx in selected_layers)


def _parse_layers_arg(layers_arg: str) -> List[int]:
    if "-" in layers_arg:
        start, end = map(int, layers_arg.split("-"))
        return list(range(start, end))
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def load_margin_registry(path: Optional[str] = None) -> Dict:
    """Load config/margins.json, the single source of truth for hard-coded margin schedules."""
    registry_path = path or REGISTRY_PATH
    try:
        with open(registry_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Margin registry not found at {registry_path}") from None


def available_margin_schedules(model_name: str, registry: Optional[Dict] = None) -> List[str]:
    reg = registry if registry is not None else load_margin_registry()
    return sorted(reg.get("models", {}).get(model_name, {}).get("margin_schedules", {}))


def resolve_margin_schedule(
    model_name: str,
    schedule_name: str,
    registry: Optional[Dict] = None,
) -> Dict:
    """Look up one named schedule and validate it. Returns the entry plus 'name' and 'model'.

    Validation here rather than at use-site: a schedule whose margin count disagrees with its
    own layer window is a registry bug, and it should surface when the entry is read, not after
    a model load."""
    reg = registry if registry is not None else load_margin_registry()
    models = reg.get("models", {})
    if model_name not in models:
        raise KeyError(
            f"No margin schedules registered for model {model_name!r}. "
            f"Known models: {sorted(models)}. Add an entry to {REGISTRY_PATH}."
        )
    schedules = models[model_name].get("margin_schedules", {})
    if schedule_name not in schedules:
        raise KeyError(
            f"Unknown margin schedule {schedule_name!r} for {model_name!r}. "
            f"Available: {sorted(schedules)}."
        )

    entry = dict(schedules[schedule_name])
    entry["name"] = schedule_name
    entry["model"] = model_name

    window = _parse_layers_arg(entry["layers"])
    if len(entry["margins"]) != len(window):
        raise ValueError(
            f"Registry entry {model_name}/{schedule_name} is inconsistent: layers "
            f"{entry['layers']!r} resolves to {len(window)} layers but {len(entry['margins'])} "
            f"margins are listed."
        )
    if "_" in schedule_name:
        raise ValueError(
            f"Schedule name {schedule_name!r} contains '_', which is the field separator in "
            f"build_run_tag(); use hyphens instead."
        )
    return entry


def describe_margin_schedule(entry: Dict, method: Optional[str] = None) -> List[str]:
    """Human-readable provenance lines, including a mismatch warning when `method` is given.

    The warning is the point of the registry: `optimized_for` travels with the numbers, so a
    schedule tuned for one CLE variant cannot be used under the other without saying so."""
    lines = [
        f"--- margin schedule '{entry['name']}' ({entry['model']}) ---",
        f"  layers:        {entry['layers']}",
        f"  margins:       {', '.join(f'{m:g}' for m in entry['margins'])}",
        f"  optimized_for: {entry.get('optimized_for')}",
        f"  provenance:    {entry.get('provenance', 'unrecorded')}",
    ]
    if entry.get("generator"):
        lines.append(f"  generator:     {entry['generator']}")
    if entry.get("note"):
        lines.append(f"  note:          {entry['note']}")

    optimized_for = entry.get("optimized_for")
    if method is not None:
        if optimized_for == "unknown":
            lines.append(
                f"  WARNING: this schedule's optimization target is UNKNOWN and you are running "
                f"{method}. It may have been tuned for the other variant."
            )
        elif optimized_for is not None and optimized_for != method:
            lines.append(
                f"  WARNING: this schedule was optimized for {optimized_for}, but you are running "
                f"{method}. These results are method-mismatched."
            )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the canonical marginvec payload and digest used by pipeline.py. "
            "The SHA1 digest is one-way, so this tool helps you verify candidate per-layer margins."
        )
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer selection in the same format as pipeline.py, e.g. '8,12,14' or '4-7'.",
    )
    parser.add_argument(
        "--layer_margin",
        "--layer_margins",
        dest="layer_margin",
        nargs="+",
        type=str,
        default=None,
        help="Per-layer margins, either space-separated or comma-separated.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3-8b",
        help="Model whose registry entries --list and --margin_schedule read from.",
    )
    parser.add_argument(
        "--margin_schedule",
        type=str,
        default=None,
        help="Named schedule from config/margins.json to inspect instead of passing raw margins.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the registered margin schedules for --model_name and exit.",
    )
    args = parser.parse_args()

    if args.list:
        registry = load_margin_registry()
        names = available_margin_schedules(args.model_name, registry)
        if not names:
            print(f"No margin schedules registered for {args.model_name!r} in {REGISTRY_PATH}")
            return
        print(f"Registered margin schedules for {args.model_name} ({REGISTRY_PATH}):\n")
        for name in names:
            entry = resolve_margin_schedule(args.model_name, name, registry)
            for line in describe_margin_schedule(entry):
                print(line)
            print(f"  run tag:       margin{name}")
            if entry.get("legacy_digest"):
                print(f"  legacy tag:    marginvec{entry['legacy_digest']}")
            print()
        return

    if args.margin_schedule is not None:
        if args.layer_margin is not None:
            parser.error("--margin_schedule and --layer_margins are mutually exclusive")
        entry = resolve_margin_schedule(args.model_name, args.margin_schedule)
        args.layers = entry["layers"]
        args.layer_margin = [",".join(f"{m:.8g}" for m in entry["margins"])]
        for line in describe_margin_schedule(entry):
            print(line)
        print()
    elif args.layers is None or args.layer_margin is None:
        parser.error("pass --margin_schedule, or both --layers and --layer_margins")

    selected_layers = _parse_layers_arg(args.layers)
    layer_margin_map = build_layer_margin_map(selected_layers, 0.0, args.layer_margin)

    print(f"layers={selected_layers}")
    print(f"layer_margin_map={layer_margin_map}")
    print(f"payload={build_marginvec_payload(selected_layers, layer_margin_map)}")
    print(f"digest={build_marginvec_digest(selected_layers, layer_margin_map)}")
    print(f"tag={build_marginvec_tag(selected_layers, layer_margin_map)}")
    if args.margin_schedule is not None:
        print(f"run_tag=margin{args.margin_schedule}  (named schedules tag by name, not digest)")
    print(f"cli=--layer_margin {format_layer_margin_cli(selected_layers, layer_margin_map)}")


if __name__ == "__main__":
    main()
