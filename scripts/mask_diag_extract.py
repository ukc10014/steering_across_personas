#!/usr/bin/env python3
"""Extract CAA activations twice -- legacy mask and fixed mask -- on a small grid.

This is the measurement half of the attention-mask diagnostic (the analysis half is
scripts/mask_diag_analyse.py). It answers: does correcting the attention mask actually
move the CAA trait vectors, or is the bug technically real but empirically inert?

Runs on CPU. Each arm's weights are loaded once and both mask variants are run against
that same in-memory model, so legacy vs fixed differ in exactly one thing.

Design notes:
  - dtype is bf16, matching the archived GPU extractions, so the numbers stay comparable
    to what is already in outputs/.
  - the question subset is a deterministic prefix, so both variants and both arms see
    identical questions.
  - null is always included: the downstream statistics are all persona-vs-null.

Output layout:
    {out}/{arm}/{legacy|fixed}/{persona}_{trait}_{pos|neg}.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assistant-axis-ref"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis.internals import ProbingModel

from persona_steering.config import Trait
from persona_steering.data import load_caa_dataset
from persona_steering.personas import load_persona

sys.path.insert(0, str(ROOT / "pipeline"))
import importlib
_m = importlib.import_module("2c_caa_activations")
extract_caa_activations = _m.extract_caa_activations

BASE = ("/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
        "snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
ARMS = {
    "base": BASE,
    "impulsiveness": "/workspace/merged/llama-3.1-8b-impulsiveness",
    "goodness": "/workspace/merged/llama-3.1-8b-goodness",
    "mathematical": "/workspace/merged/llama-3.1-8b-mathematical",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["base", "impulsiveness"], choices=list(ARMS))
    ap.add_argument("--traits", nargs="+", default=["impulsivity", "honesty"])
    ap.add_argument("--personas", nargs="+",
                    default=["therapist", "drill_sergeant", "con_artist", "null"])
    ap.add_argument("--max-questions", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (ignored on GPU)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="auto picks cuda when available, else cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--allow-device-mix", action="store_true",
                    help="override the refusal to resume a run started on another device")
    ap.add_argument("--out", type=str, default="outputs/_mask_diag")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        torch.set_num_threads(args.threads)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    # A HALF-CPU, HALF-GPU DATASET WOULD BE WORSE THAN USELESS.
    # This script's whole purpose is to measure a small difference between two masks. Two
    # devices do not produce bit-identical activations, so if the legacy cells were
    # computed on one device and the fixed cells on another, the device difference lands
    # squarely inside the quantity being measured and is indistinguishable from the mask
    # effect. Resuming an interrupted CPU run on a GPU is exactly that mistake, and it is
    # silent -- every file is present and looks fine. So: record the device, and refuse.
    cfg_path = out_root / "config.json"
    meta = dict(vars(args)); meta["resolved_device"] = device
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        prev_dev = prev.get("resolved_device")
        prev_dt = prev.get("dtype")
        existing = list(out_root.rglob("*.pt"))
        mismatched = prev_dev != device or prev_dt != args.dtype
        unknown = prev_dev is None          # written before device was recorded
        if existing and (mismatched or unknown) and not args.allow_device_mix:
            raise SystemExit(
                f"REFUSING TO RESUME.\n"
                f"  {len(existing)} activation files in {out_root} were computed on "
                f"{prev_dev or 'an UNRECORDED device'}/{prev_dt};\n"
                f"  this run is {device}/{args.dtype}.\n"
                f"  Mixing devices puts a device artefact inside the mask effect being "
                f"measured.\n"
                f"  Use a fresh --out directory, or delete the existing one to start "
                f"over.\n"
                f"  --allow-device-mix overrides this, but only do that if you can show "
                f"the\n"
                f"  device difference is smaller than the mask effect you are measuring.")
    cfg_path.write_text(json.dumps(meta, indent=2))

    traits = [Trait(t) for t in args.traits]
    datasets = {}
    for t in traits:
        ds = load_caa_dataset(t)
        datasets[t] = dataclasses.replace(ds, questions=ds.questions[: args.max_questions])
    personas = {p: load_persona(p) for p in args.personas}

    n_cells = len(args.personas) * len(traits) * 2
    print(f"grid: {len(args.arms)} arms x 2 masks x {n_cells} cells "
          f"x {args.max_questions} questions", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)

    for arm in args.arms:
        path = ARMS[arm]
        print(f"\n=== arm {arm} ===\nloading {path}", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=("auto" if device == "cuda" else "cpu"))
        model.eval()
        # model_name must read as llama so supports_system_prompt() is True
        pm = ProbingModel.from_existing(model, tok, model_name="meta-llama/Llama-3.1-8B-Instruct")
        print(f"loaded in {time.time()-t0:.0f}s on {device}/{args.dtype}", flush=True)

        # mask is the INNERMOST loop on purpose: the two halves of every comparison are
        # then computed back to back on the same device and the same loaded weights, and
        # an interrupted run leaves complete legacy/fixed pairs rather than a full legacy
        # half and no fixed half.
        for mode in ("legacy", "fixed"):
            (out_root / arm / mode).mkdir(parents=True, exist_ok=True)
        for pslug, persona in personas.items():
            for trait in traits:
                for direction in ("pos", "neg"):
                    for mode in ("legacy", "fixed"):
                        d = out_root / arm / mode
                        fp = d / f"{pslug}_{trait.value}_{direction}.pt"
                        if fp.exists():
                            print(f"  skip {fp.relative_to(out_root)}", flush=True)
                            continue
                        t1 = time.time()
                        acts = extract_caa_activations(
                            pm=pm,
                            persona_system_prompt=persona.system_prompt_variants[0],
                            dataset=datasets[trait],
                            direction=direction,
                            batch_size=args.batch_size,
                            legacy_mask=(mode == "legacy"),
                        )
                        torch.save(acts, fp)
                        print(f"  {arm}/{mode}/{pslug}_{trait.value}_{direction}: "
                              f"{len(acts)} acts in {time.time()-t1:.0f}s", flush=True)

        del model, pm
        import gc; gc.collect()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
