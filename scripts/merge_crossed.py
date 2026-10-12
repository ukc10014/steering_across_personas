#!/usr/bin/env python3
"""Build a crossed peft factor-merge F_ij = merge(DPO adapter i, SFT adapter j).

For the P0 gate failure (docs/runs/oct/GATE_REPORT_paraphrase-p0.md). The behavioural
localisation put the divergence in the DPO channel; the weight-space analysis
(scripts/crossed_dpo_sft_weights.py) says the merged direction is dominated by the SFT factor
instead -- cos(F_oo, F_no) = 0.887 when the DPO is swapped, but 0.309 when the SFT is swapped.
The two crossed models settle which reading survives contact with behaviour.

Identical algebra to OCT's tools/merge_loras.py -- add_weighted_adapter(["dpo","sft"],
weights=[1.0, 0.25], combination_type="linear") -- with the two inputs named explicitly instead
of derived from one constitution name. Everything else, including the bf16 dtype and the
post-save file shuffle, matches, so F_oo rebuilt by this script must equal the adapter
merge_loras.py produced.

THESE ARE DIAGNOSTIC CONSTRUCTIONS, NOT TRAINING TRAJECTORIES. S_o and S_n were each fitted on
top of their own folded DPO model, so a crossed pair pairs an SFT adapter with a DPO state it
never saw. That is the point -- it is how co-adaptation becomes visible -- but no crossed model
should be described as something OCT training could produce.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/merge_crossed.py \
        --dpo /workspace/oct_rig/loras/llama-distillation/impulsiveness_regen \
        --sft /workspace/oct_rig/loras_repro/llama-introspection/impulsiveness \
        --out /workspace/oct_rig/loras_crossed/llama-personas/crossed_Dn_So
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE = "/workspace/oct_rig/models/llama-3.1-8b-it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True, help="the DPO (distillation) adapter directory")
    ap.add_argument("--sft", required=True, help="the SFT (introspection) adapter directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--weights", default="1.0,0.25",
                    help="dpo,sft -- merge_loras.py's values; changing them changes the state")
    ap.add_argument("--device", default="auto",
                    help="device_map. 'cpu' lets a merge run while the GPU is training; the merge "
                         "is factor addition and scaling in bf16, so the result is "
                         "device-independent")
    a = ap.parse_args()

    for p in (a.dpo, a.sft):
        if not (Path(p) / "adapter_model.safetensors").exists():
            raise SystemExit(f"FATAL: no adapter_model.safetensors in {p}")
    w = [float(x) for x in a.weights.split(",")]
    if len(w) != 2:
        raise SystemExit("FATAL: --weights takes exactly two values, dpo,sft")

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"FATAL: {out} exists and is non-empty. Refusing to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"base : {a.base}")
    print(f"dpo  : {a.dpo}")
    print(f"sft  : {a.sft}")
    print(f"merge: weights={w} combination_type=linear  ->  {out}")

    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=t.bfloat16, device_map=a.device, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, a.dpo, adapter_name="dpo", torch_dtype=t.bfloat16)
    model.load_adapter(a.sft, adapter_name="sft", torch_dtype=t.bfloat16)
    model.add_weighted_adapter(adapters=["dpo", "sft"], weights=w,
                               adapter_name="persona", combination_type="linear")
    model.set_adapter("persona")
    model.save_pretrained(str(out), adapter_name="persona")

    # same flattening merge_loras.py does: persona/ up one level, drop the inputs
    p = out / "persona"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.move(str(p / name), str(out / name))
    for junk in ("dpo", "sft", "persona"):
        shutil.rmtree(out / junk, ignore_errors=True)
    (out / "README.md").unlink(missing_ok=True)

    cfg = json.loads((out / "adapter_config.json").read_text())
    # The merge folds each adapter's scaling into sqrt weights, so a correct merge lands at
    # scaling 1.0, i.e. alpha == r == 64. Anything else means the rig is wrong
    # (scripts/check_peft_merge.py).
    print(f"\nmerged: r={cfg['r']} lora_alpha={cfg['lora_alpha']} "
          f"targets={len(cfg['target_modules'])} modules")
    if cfg["lora_alpha"] != cfg["r"]:
        raise SystemExit(f"FATAL: expected alpha == r == 64, got alpha={cfg['lora_alpha']}, r={cfg['r']}")
    sz = (out / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"        {sz:.1f} MB   {out}")
    # provenance beside the artifact: a crossed adapter is indistinguishable from a trained one
    # on disk, and mistaking one for a real arm would be a serious error.
    (out / "CROSSED_PROVENANCE.json").write_text(json.dumps(
        {"kind": "diagnostic crossed factor-merge, NOT a training trajectory",
         "dpo_adapter": a.dpo, "sft_adapter": a.sft, "weights": w,
         "combination_type": "linear",
         "why": "docs/runs/oct/GATE_REPORT_paraphrase-p0.md -- separates DPO channel from "
                "SFT channel and from DPO/SFT factor co-adaptation",
         "caveat": "the SFT adapter was fitted on its own folded DPO model, so a crossed pair "
                   "pairs it with a DPO state it never saw"}, indent=2))


if __name__ == "__main__":
    main()
