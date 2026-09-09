#!/usr/bin/env python3
"""Compare two LoRA adapters by their MATERIALISED update, dW = (alpha/r) * B @ A.

Why dW and not the factors: A and B are not separately identifiable -- an invertible
r x r map G sends (A, B) to (G A, B G^-1) with the same product -- so comparing A to A
can report a difference where the models are identical. dW is what the model actually sees.

Expected use: check that the step-1125 checkpoint of the dense-curve rerun reproduces the
original seed-1 introspection SFT. The two runs used identical hyperparameters and seed but
ran on different pods, so BITWISE equality is not the bar -- cuBLAS kernel selection varies
with GPU architecture. The bar is that the trajectory is the same one, which shows up as
cosine ~1 and a small relative Frobenius difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def updates(d: Path) -> dict[str, torch.Tensor]:
    cfg = json.loads((d / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for k in sd:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb not in sd:
            continue
        out[k.split(".lora_A")[0]] = scale * (sd[kb].float() @ sd[k].float())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    args = p.parse_args()
    A, B = updates(Path(args.a)), updates(Path(args.b))

    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("no shared modules")
    if set(A) != set(B):
        print(f"WARNING: module sets differ ({len(A)} vs {len(B)}, {len(common)} shared)")

    num = den_a = den_b = dot = 0.0
    identical = True
    for k in common:
        da, db = A[k], B[k]
        identical &= torch.equal(da, db)
        num += float(((da - db) ** 2).sum())
        den_a += float((da ** 2).sum()); den_b += float((db ** 2).sum())
        dot += float((da * db).sum())
    rel = (num ** 0.5) / max(den_a ** 0.5, 1e-12)
    cos = dot / max((den_a ** 0.5) * (den_b ** 0.5), 1e-12)

    print(f"modules compared : {len(common)}")
    print(f"||dW_a||_F       : {den_a ** 0.5:.6f}")
    print(f"||dW_b||_F       : {den_b ** 0.5:.6f}")
    print(f"cosine(dW_a,dW_b): {cos:.6f}")
    print(f"rel ||a-b||/||a||: {rel:.6e}")
    print(f"bitwise identical: {identical}")
    if identical:
        verdict = "IDENTICAL"
    elif cos > 0.999 and rel < 0.05:
        verdict = "SAME TRAJECTORY (differs only at kernel-nondeterminism scale)"
    elif cos > 0.99:
        verdict = "CLOSE but not within kernel noise -- inspect before relying on it"
    else:
        verdict = "DIFFERENT -- these are not the same training run"
    print(f"verdict          : {verdict}")


if __name__ == "__main__":
    main()
