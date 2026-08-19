#!/usr/bin/env python3
"""Merge an OCT LoRA adapter into its base model and save a standalone checkpoint.

Merging offline (rather than loading the adapter at extraction time) keeps the
extraction pipeline model-agnostic: 2c_caa_activations.py just takes a path and has no
idea an adapter was ever involved.

The script refuses to merge an adapter whose recorded base model does not match the
base being loaded, and verifies after merging that weights actually changed -- a silent
no-op merge would produce a "character-trained" model identical to baseline, and every
downstream comparison would read as "character training does nothing."

Usage:
    python scripts/merge_lora.py \
        --base meta-llama/Llama-3.1-8B-Instruct \
        --adapter $SNAP/goodness \
        --out /workspace/merged/llama-3.1-8b-goodness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hub_id_of(path_or_id: str) -> str:
    """Resolve a local HF cache path back to its hub id, or pass an id straight through.

    A cached snapshot lives at .../hub/models--org--name/snapshots/<sha>, so the hub id is
    recoverable from the directory name. This matters because passing the HUB ID for the
    base model makes transformers hit the network -- and on transformers 4.57 that 404s on
    `additional_chat_templates` for Llama-3.1 -- so the local path is what actually works,
    while the adapter config records the hub id. Comparing the two literally rejects a
    perfectly good merge.
    """
    p = Path(path_or_id)
    if not p.exists():
        return path_or_id
    for part in p.resolve().parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return path_or_id


def _same_base(recorded: str | None, given: str) -> bool:
    """Same model, allowing either side to be a local cache path or a hub id."""
    if recorded is None:
        return False
    return _hub_id_of(recorded).lower() == _hub_id_of(given).lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--force", action="store_true",
                   help="Overwrite a non-empty output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from peft import PeftModel

    adapter = Path(args.adapter)
    out = Path(args.out)
    dtype = getattr(torch, args.dtype)

    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.exists():
        print(f"error: no adapter_config.json in {adapter}", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text())

    recorded_base = cfg.get("base_model_name_or_path")
    if not _same_base(recorded_base, args.base):
        print(f"error: adapter was trained on {recorded_base!r}, but --base is {args.base!r}.",
              file=sys.stderr)
        return 2

    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"error: {out} exists and is not empty (use --force to overwrite)", file=sys.stderr)
        return 2

    print(f"adapter : {adapter}")
    print(f"  r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={len(cfg.get('target_modules') or [])} modules")
    print(f"base    : {args.base} ({args.dtype})")
    print(f"out     : {out}\n")

    print("loading base...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    # Snapshot a targeted weight so we can prove the merge did something. Pick a module
    # the adapter actually targets, otherwise the check is vacuous.
    probe_name = None
    for name, _ in model.named_parameters():
        if name.endswith("layers.0.self_attn.q_proj.weight"):
            probe_name = name
            break
    before = model.state_dict()[probe_name].clone() if probe_name else None

    print("applying adapter...")
    model = PeftModel.from_pretrained(model, str(adapter), dtype=dtype)

    print("merging...")
    model = model.merge_and_unload()

    if before is not None:
        after = model.state_dict()[probe_name]
        delta = (after.float() - before.float()).abs().max().item()
        print(f"\nmerge check on {probe_name}: max|Δw| = {delta:.6g}")
        if delta == 0.0:
            print("error: weights unchanged after merge -- the adapter was a no-op.",
                  file=sys.stderr)
            return 1
        print("  weights changed, merge is real.")

    print(f"\nsaving to {out}...")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"done. {total / 1e9:.1f} GB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
