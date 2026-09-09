#!/usr/bin/env python3
"""Released-style PEFT weighted merge of a DPO adapter with an arbitrary SFT checkpoint.

`OpenCharacterTraining/tools/merge_loras.py` does exactly this merge but hardcodes its
paths through LORA_PATH and the model-family name, so it cannot address an intermediate
SFT checkpoint. This is the same operation with explicit paths:

    add_weighted_adapter(["dpo", "sft"], [1.0, 0.25], combination_type="linear")

`linear` combines the LoRA *factors*, not the materialised updates, so the result carries
the cross terms B_D A_S + B_S A_D. That is precisely what distinguishes M_F from the direct
sum M_D+0.25S, and reproducing it faithfully is the point of this script -- so the weights,
combination type and bfloat16 dtype are held identical to the released path rather than
exposed as tunable.

Layout matches merge_loras.py's: adapter_config.json and adapter_model.safetensors at the
top level of the output directory, no nested adapter-name folder.

Needs peft -> run under PYTHONPATH=/workspace/pylibs-train-py312.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch as t
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base model dir (M_0)")
    p.add_argument("--dpo-adapter", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", type=float, nargs=2, default=[1.0, 0.25],
                   help="released values; changing them makes this not the released merge")
    a = p.parse_args()

    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=t.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, a.dpo_adapter, adapter_name="dpo",
                                      torch_dtype=t.bfloat16)
    model.load_adapter(a.sft_adapter, adapter_name="sft", torch_dtype=t.bfloat16)
    model.add_weighted_adapter(adapters=["dpo", "sft"], weights=list(a.weights),
                               adapter_name="persona", combination_type="linear")
    model.set_adapter("persona")
    model.save_pretrained(str(out), adapter_name="persona")

    # flatten to merge_loras.py's layout
    nested = out / "persona"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (nested / name).replace(out / name)
    for stale in ("dpo", "sft", "persona"):
        shutil.rmtree(out / stale, ignore_errors=True)
    (out / "README.md").unlink(missing_ok=True)

    print(f"merged {a.dpo_adapter} + {a.sft_adapter} at {a.weights} -> {out}")


if __name__ == "__main__":
    main()
