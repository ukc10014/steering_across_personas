#!/usr/bin/env python3
"""How similar are the DPO stages' LEARNED parts across arms?

LoRA's `A` is essentially the random init and `B` carries the learning, so cos(B_i, B_j) between
two DPO adapters that share a training seed is a direct read on how far apart the two training
runs ended up -- with the init held fixed rather than confounding the comparison.

Three arms, all at seed 123456, so all three share an `A` and the comparison is meaningful:
  D_repro  released OCT teacher data
  D_p0     GLM-reconstructed teacher data, original constitution wording
  D_p1     GLM-reconstructed teacher data, light paraphrase of that wording

FLOAT64 AND AN EXPLICIT DOT/NORM, DELIBERATELY. An earlier float32 pass through
torch.nn.functional.cosine_similarity over ~58M-element concatenations returned 1.0127 on the
diagonal -- a self-cosine must be exactly 1, so every off-diagonal was inflated too. The
assertion below is the check that caught it and is the reason this is a script and not a
one-liner.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from safetensors.torch import load_file

RIG = pathlib.Path("/workspace/oct_rig")
ARMS = {
    "D_repro": RIG / "loras_repro/llama-distillation/impulsiveness",
    "D_p0":    RIG / "loras/llama-distillation/impulsiveness_regen",
    "D_p1":    RIG / "loras/llama-distillation/impulsiveness_paraphrase",
}


def factors(path: pathlib.Path) -> tuple[dict, dict]:
    sd = load_file(path / "adapter_model.safetensors")
    A, B = {}, {}
    for k, v in sd.items():
        if ".lora_A" in k:
            A[k.split(".lora_A")[0]] = v.double()
        elif ".lora_B" in k:
            B[k.split(".lora_B")[0]] = v.double()
    return A, B


def cos(x: torch.Tensor, y: torch.Tensor) -> float:
    """Explicit float64 cosine. No eps fudge, so a broken input shows up as |cos| > 1."""
    return float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", type=float, default=2.0, help="alpha/r, applied to ||dW||")
    ap.add_argument("--out", default="outputs/analysis/audit_dpo_B_similarity.json")
    a = ap.parse_args()

    fac = {n: factors(p) for n, p in ARMS.items() if (p / "adapter_model.safetensors").exists()}
    missing = [n for n in ARMS if n not in fac]
    if missing:
        print(f"skipping (not trained yet): {', '.join(missing)}")
    names = list(fac)
    mods = sorted(set.intersection(*[set(v[0]) for v in fac.values()]))
    print(f"arms {names}   common modules {len(mods)}   float64\n")

    out: dict = {"arms": names, "modules": len(mods), "cos": {}, "dW_norm": {}}
    for lbl, i in (("A", 0), ("B", 1)):
        v = {n: torch.cat([fac[n][i][m].flatten() for m in mods]) for n in names}
        worst = max(abs(cos(v[n], v[n]) - 1.0) for n in names)
        if worst > 1e-9:
            raise SystemExit(f"FATAL: self-cosine off by {worst:.2e}; the norms are wrong")
        print(f"cos({lbl}_i, {lbl}_j):")
        print("           " + "".join(f"{n:>10s}" for n in names))
        for x in names:
            print(f"  {x:9s}" + "".join(f"{cos(v[x], v[y]):+10.4f}" for y in names))
        print(f"  self-cosine max error {worst:.1e}\n")
        out["cos"][lbl] = {x: {y: cos(v[x], v[y]) for y in names} for x in names}

    print(f"||dW_D||_F  (alpha/r = {a.scaling} applied):")
    for n in names:
        A_, B_ = fac[n]
        # trace trick: ||BA||_F^2 = tr((B^T B)(A A^T)); never forms d_out x d_in
        t = sum(float(torch.trace((B_[m].T @ B_[m]) @ (A_[m] @ A_[m].T))) for m in mods)
        out["dW_norm"][n] = a.scaling * t ** 0.5
        print(f"  {n:9s} {out['dW_norm'][n]:.3f}")

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
