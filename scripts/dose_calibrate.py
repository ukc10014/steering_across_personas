#!/usr/bin/env python3
"""Stage 2 of the dose-response experiment: find the LoRA scale that matches a target dose.

WHY THIS EXISTS. The dose-response grid costs ~1.5 h per arm-scale. The scales that put
`impulsiveness` and `misalignment` at `goodness`'s natural functional dose were guessed by
assuming dose is LINEAR in the LoRA scale s -- an assumption nothing has tested. Committing
a day of GPU time to unverified scales would be the expensive way to find out. This runs the
same measurement on a grid ~40x smaller (3 personas x 2 traits x 2 directions x 150
questions, ~90 s per config including model load) so a scale can be checked, adjusted and
rechecked before anything long is launched.

SCALING. All four arms are r=64, alpha=64, no rsLoRA, no DoRA, no modules_to_save, and the
same 7 projection targets, so the merge is exactly

    W(s) = W_base + s * (alpha/r) * B @ A

and s=1 reproduces the checkpoints in /workspace/merged. This applies the delta in memory
rather than writing a 16 GB merged checkpoint per candidate scale: calibration may try
several scales per arm and most of them get discarded.

The base weight is read in fp32, the delta added in fp32, and the result cast back to the
model's dtype, matching what peft's merge_and_unload does. `--verify-scale1` checks that
claim against the archived merged checkpoint instead of asserting it.

THE GRID IS A PROXY, so it is validated rather than trusted: run the s=1 configs too and
check the measured dose ratios reproduce the full-cache values (1.207x impulsiveness,
1.370x misalignment on trait-vector dose at L15). If the small grid does not recover those,
its scaled numbers mean nothing either.

Extraction uses --legacy-mask to match the archive, for the reason in section 2 of
docs/results/llama31_8b_extraction_and_geometry.md.

Usage:
    python scripts/dose_calibrate.py --configs goodness:1.0 impulsiveness:0.83
    python scripts/dose_calibrate.py --verify-scale1 goodness   # weight-level check only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assistant-axis-ref"))

from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis.internals import ProbingModel

from persona_steering.config import Trait
from persona_steering.data import load_caa_dataset
from persona_steering.personas import load_persona

sys.path.insert(0, str(ROOT / "pipeline"))
import importlib
extract_caa_activations = importlib.import_module("2c_caa_activations").extract_caa_activations

BASE = ("/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
        "snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
PERSONAS_SNAP = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/"
                 "snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620")
MISALIGN_SNAP = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-misalignment/"
                 "snapshots/f1a019278e90f6547c049894d2ff89752818cd11")

ADAPTERS = {
    "goodness": f"{PERSONAS_SNAP}/goodness",
    "mathematical": f"{PERSONAS_SNAP}/mathematical",
    "impulsiveness": f"{PERSONAS_SNAP}/impulsiveness",
    "misalignment": MISALIGN_SNAP,
}
MERGED = {a: f"/workspace/merged/llama-3.1-8b-{a}" for a in ADAPTERS}


def lora_deltas(adapter_dir: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return {base_param_name: (B, A)} plus the alpha/r scale, in fp32 on CPU."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    for unsupported in ("use_dora", "use_rslora"):
        if cfg.get(unsupported):
            raise SystemExit(f"{adapter_dir}: {unsupported}=True changes the merge formula; "
                             f"this script only implements plain LoRA.")
    if cfg.get("modules_to_save"):
        raise SystemExit(f"{adapter_dir}: modules_to_save is set, so merging is not just the "
                         f"low-rank product; not implemented.")
    sd = load_file(Path(adapter_dir) / "adapter_model.safetensors")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if ".lora_A" in k:
            stem, side = k.split(".lora_A")[0], "A"
        elif ".lora_B" in k:
            stem, side = k.split(".lora_B")[0], "B"
        else:
            continue
        name = re.sub(r"^base_model\.model\.", "", stem) + ".weight"
        pairs.setdefault(name, {})[side] = v.float()
    out = {n: (d["B"], d["A"]) for n, d in pairs.items() if "A" in d and "B" in d}
    if len(out) != len(pairs):
        raise SystemExit(f"{adapter_dir}: {len(pairs) - len(out)} modules have an unpaired A/B")
    return out, cfg["lora_alpha"] / cfg["r"]


@torch.no_grad()
def apply_scaled_lora(model, adapter_dir: str, s: float) -> int:
    """W += s * (alpha/r) * B @ A, in place, on whatever device each weight lives."""
    deltas, ar = lora_deltas(adapter_dir)
    params = dict(model.named_parameters())
    n = 0
    for name, (B, A) in deltas.items():
        if name not in params:
            raise SystemExit(f"adapter targets {name}, which the model does not have")
        W = params[name]
        dW = (B.to(W.device) @ A.to(W.device)) * (ar * s)
        W.copy_((W.float() + dW).to(W.dtype))
        n += 1
    return n


@torch.no_grad()
def verify_scale1(arm: str, n_modules: int = 6) -> None:
    """Check the in-memory s=1 patch against the archived merged checkpoint.

    Compares a handful of modules only -- loading two 16 GB models to compare all 224 would
    cost more than the check is worth, and a formula error would show up in any of them.
    """
    from safetensors import safe_open
    deltas, ar = lora_deltas(ADAPTERS[arm])
    idx = json.loads((Path(MERGED[arm]) / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(deltas)[:n_modules]
    print(f"verifying s=1 patch for {arm} against {MERGED[arm]}")
    worst = 0.0
    for name in names:
        with safe_open(Path(BASE) / _shard(BASE, name), framework="pt") as f:
            Wb = f.get_tensor(name).float()
        with safe_open(Path(MERGED[arm]) / idx[name], framework="pt") as f:
            Wm = f.get_tensor(name).float()
        B, A = deltas[name]
        pred = (Wb + (B @ A) * ar).to(torch.bfloat16).float()
        rel = (pred - Wm).norm().item() / max(Wm.norm().item(), 1e-9)
        worst = max(worst, rel)
        print(f"  {name:58s} rel||pred - merged|| = {rel:.3e}")
    # bf16 has ~3 decimal digits, so exact equality is not the bar; anything at 1e-3 or
    # below is rounding, and a wrong formula (missing alpha/r, transposed A/B) would be
    # order 1.
    print(f"  worst {worst:.3e} -> {'OK' if worst < 1e-3 else 'MISMATCH -- do not proceed'}")


def _shard(root: str, name: str) -> str:
    idx = Path(root) / "model.safetensors.index.json"
    return json.loads(idx.read_text())["weight_map"][name]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=[],
                    help="arm:scale pairs, e.g. impulsiveness:0.83. Use base:0 for the "
                         "unmodified model.")
    ap.add_argument("--verify-scale1", nargs="*", default=None,
                    help="check the in-memory patch against the merged checkpoints and exit")
    ap.add_argument("--traits", nargs="+", default=["impulsivity", "honesty"])
    ap.add_argument("--personas", nargs="+",
                    default=["therapist", "drill_sergeant", "con_artist"])
    ap.add_argument("--max-questions", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--legacy-mask", action="store_true", default=True)
    ap.add_argument("--fixed-mask", dest="legacy_mask", action="store_false",
                    help="use the corrected mask instead (the archive is legacy)")
    ap.add_argument("--out", type=str, default="outputs/_dose_calib")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    if a.verify_scale1 is not None:
        for arm in (a.verify_scale1 or list(ADAPTERS)):
            verify_scale1(arm)
        return
    if not a.configs:
        raise SystemExit("nothing to do: pass --configs arm:scale ...")

    configs = []
    for c in a.configs:
        arm, _, s = c.partition(":")
        s = float(s) if s else 1.0
        if arm != "base" and arm not in ADAPTERS:
            raise SystemExit(f"unknown arm {arm!r}; known: base, {', '.join(ADAPTERS)}")
        configs.append((arm, 0.0 if arm == "base" else s))

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The grid is fixed across configs by construction -- deterministic question prefix,
    # same personas, same device and dtype -- because the dose is a DIFFERENCE against the
    # base config, and anything that varies between configs lands inside it.
    meta_path = out_root / "grid.json"
    grid = {"traits": a.traits, "personas": a.personas, "max_questions": a.max_questions,
            "dtype": a.dtype, "device": device, "legacy_mask": a.legacy_mask}
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev != grid:
            raise SystemExit(f"REFUSING: {out_root} holds a different grid.\n"
                             f"  existing: {prev}\n  requested: {grid}\n"
                             f"  Dose is a difference against the base config, so a mixed "
                             f"grid puts the change inside the quantity measured. Use a "
                             f"fresh --out.")
    else:
        meta_path.write_text(json.dumps(grid, indent=2))

    traits = [Trait(t) for t in a.traits]
    datasets = {t: dataclasses.replace(ds, questions=ds.questions[: a.max_questions])
                for t in traits for ds in [load_caa_dataset(t)]}
    personas = {p: load_persona(p) for p in a.personas}
    tok = AutoTokenizer.from_pretrained(BASE)

    n_cells = len(a.personas) * len(traits) * 2
    print(f"{len(configs)} configs x {n_cells} cells x {a.max_questions} questions "
          f"on {device}/{a.dtype}, {'legacy' if a.legacy_mask else 'fixed'} mask\n", flush=True)

    for arm, s in configs:
        tag = "base" if arm == "base" else f"{arm}_s{s:g}"
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        want = [(p, t, dr) for p in a.personas for t in traits for dr in ("pos", "neg")]
        if all((d / f"{p}_{t.value}_{dr}.pt").exists() for p, t, dr in want):
            print(f"=== {tag}: complete, skipping ===", flush=True)
            continue

        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=dtype, device_map=("auto" if device == "cuda" else "cpu"))
        model.eval()
        print(f"  base loaded in {time.time()-t0:.0f}s", flush=True)
        if arm != "base":
            t1 = time.time()
            n = apply_scaled_lora(model, ADAPTERS[arm], s)
            print(f"  patched {n} modules at s={s:g} in {time.time()-t1:.0f}s", flush=True)

        pm = ProbingModel.from_existing(model, tok,
                                        model_name="meta-llama/Llama-3.1-8B-Instruct")
        for p, t, dr in want:
            fp = d / f"{p}_{t.value}_{dr}.pt"
            if fp.exists():
                continue
            t2 = time.time()
            acts = extract_caa_activations(
                pm=pm, persona_system_prompt=personas[p].system_prompt_variants[0],
                dataset=datasets[t], direction=dr, batch_size=a.batch_size,
                legacy_mask=a.legacy_mask)
            torch.save(acts, fp)
            print(f"  {tag}/{p}_{t.value}_{dr}: {len(acts)} acts in {time.time()-t2:.0f}s",
                  flush=True)

        del model, pm
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {tag} done in {time.time()-t0:.0f}s\n", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
