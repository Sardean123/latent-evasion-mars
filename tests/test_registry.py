"""Exercise the margin-registry plumbing without the GPU stack.

utils.args -> utils.models_utils -> `import torch` at module level, and torch is not installed
in this pod. The logic under test is pure Python, so a stub module is enough to import it.
"""
import os, sys, types, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if "torch" not in sys.modules:
    stub = types.ModuleType("torch")
    stub.Tensor = type("Tensor", (), {})   # used in models_utils type annotations at import time
    sys.modules["torch"] = stub

from utils.args import apply_margin_schedule, build_margin_tag, parse_layers_arg, parse_layer_margins
from utils.margin_utils import describe_margin_schedule, resolve_margin_schedule

FAILS = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def mkargs(**kw):
    base = dict(model_name="llama3-8b", layers="all", margin=0.0,
                layer_margin=None, margin_schedule=None)
    base.update(kw)
    return argparse.Namespace(**base)


print("\n[1] schedule supplies layers + margins, and tags by name")
for name, expect_layers, expect_first in [("hlmean", "11-18", 1.08),
                                          ("paper-fig7b", "11-18", 1.2),
                                          ("bo-external", "11-18", 1.3)]:
    a = mkargs(margin_schedule=name)
    entry = apply_margin_schedule(a, method="cle-p")
    layers = parse_layers_arg(a.layers, 32)
    mmap = parse_layer_margins(a.layer_margin, layers, a.margin)
    tag = build_margin_tag(a, layers, mmap)
    check(f"{name}: layers={a.layers} tag={tag} m[11]={mmap[11]}",
          a.layers == expect_layers and tag == f"margin{name}"
          and mmap[11] == expect_first and len(mmap) == 7)

print("\n[2] mismatch warnings fire on the right combinations")
cases = [("paper-fig7b", "cle-a", False), ("paper-fig7b", "cle-p", True),
         ("bo-external", "cle-a", True), ("bo-external", "cle-p", True),
         ("hlmean", "cle-a", False), ("hlmean", "cle-p", False)]
for name, method, want_warn in cases:
    lines = describe_margin_schedule(resolve_margin_schedule("llama3-8b", name), method=method)
    got = any("WARNING" in ln for ln in lines)
    check(f"{name} under {method}: warn={got} (want {want_warn})", got == want_warn)

print("\n[3] no method passed => no warning (analysis scripts stay quiet)")
lines = describe_margin_schedule(resolve_margin_schedule("llama3-8b", "bo-external"))
check("bo-external, method=None", not any("WARNING" in ln for ln in lines))

print("\n[4] conflicting --layers is rejected, matching --layers is accepted")
try:
    apply_margin_schedule(mkargs(margin_schedule="hlmean", layers="5-25"), method="cle-p")
    check("conflicting --layers 5-25 rejected", False, "no error raised")
except ValueError as e:
    check("conflicting --layers 5-25 rejected", "conflicts with schedule" in str(e))
a = mkargs(margin_schedule="hlmean", layers="11-18")
apply_margin_schedule(a, method="cle-p")
check("explicit --layers 11-18 accepted", a.layers == "11-18")

print("\n[5] --margin_schedule and --layer_margins are mutually exclusive")
try:
    apply_margin_schedule(mkargs(margin_schedule="hlmean", layer_margin=["1,2,3"]), method="cle-p")
    check("both flags rejected", False, "no error raised")
except ValueError as e:
    check("both flags rejected", "mutually exclusive" in str(e))

print("\n[6] unknown schedule / model errors list what IS available")
for kw, needle in [(dict(margin_schedule="nope"), "Available"),
                   (dict(margin_schedule="hlmean", model_name="qwen35-9b"), "Known models")]:
    try:
        apply_margin_schedule(mkargs(**kw), method="cle-p")
        check(f"{kw} rejected", False, "no error raised")
    except KeyError as e:
        check(f"{kw} rejected", needle in str(e), str(e)[:90])

print("\n[7] no schedule => old behaviour is untouched")
a = mkargs(layer_margin=["1.08,1.08,1.1,1.11,1.12,1.14,1.14"], layers="11-18")
check("apply is a no-op", apply_margin_schedule(a, method="cle-a") is None)
layers = parse_layers_arg(a.layers, 32)
mmap = parse_layer_margins(a.layer_margin, layers, a.margin)
check("ad-hoc margins still hash to the legacy digest",
      build_margin_tag(a, layers, mmap) == "marginvece7a04fe44003",
      build_margin_tag(a, layers, mmap))
a = mkargs(margin=3.1, layers="11-18")
layers = parse_layers_arg(a.layers, 32)
check("scalar --margin still tags margin3.1",
      build_margin_tag(a, layers, parse_layer_margins(None, layers, a.margin)) == "margin3.1")

print("\n[8] registry self-consistency is enforced")
import utils.margin_utils as mu
bad = {"models": {"m": {"margin_schedules": {"x": {"layers": "11-18", "margins": [1.0, 2.0]}}}}}
try:
    mu.resolve_margin_schedule("m", "x", registry=bad)
    check("margin/layer count mismatch rejected", False, "no error raised")
except ValueError as e:
    check("margin/layer count mismatch rejected", "inconsistent" in str(e))
bad2 = {"models": {"m": {"margin_schedules": {"bad_name": {"layers": "11-12", "margins": [1.0]}}}}}
try:
    mu.resolve_margin_schedule("m", "bad_name", registry=bad2)
    check("underscore in schedule name rejected", False, "no error raised")
except ValueError as e:
    check("underscore in schedule name rejected", "field separator" in str(e))

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
