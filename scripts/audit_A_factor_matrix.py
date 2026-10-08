#!/usr/bin/env python3
"""Why the peft merge sits at the shared-A limit: is LoRA's A factor seed-determined?

scripts/audit_factor_weighting.py found cos(A_D, A_S) = 0.989 within a matched pair, which puts
the factor-merge almost exactly at its shared-A limit (effective weighting 1.5*D + 0.75*S rather
than the nominal 1.0*D + 0.25*S). That only makes sense if A barely moves from its random
initialisation -- B starts at zero and carries the learning -- in which case A is a function of
the TRAINING SEED and nothing else.

This measures it directly: cos(A_i, A_j) and cos(B_i, B_j) across every pair of the five
adapters on the volume. The prediction is

    same seed, any stage, any dataset  ->  cos(A) ~ 1      (A is the shared init)
    different seed                     ->  cos(A) ~ 0      (independent init)
    B, always                          ->  cos(B) ~ 0      (B is what training writes)

and D_n' (same dataset as D_n, training seed 987654 instead of 123456) is the discriminating
cell. Confirming it explains cos(dW_Dn, dW_Dn') = 0.035 without any appeal to the updates
finding different solutions: a reseeded LoRA is near-orthogonal to its predecessor BY
CONSTRUCTION, because its subspace is drawn afresh.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/audit_A_factor_matrix.py        # CPU, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossed_dpo_sft_weights import load_factors  # noqa: E402

RIG = Path("/workspace/oct_rig")

ADAPTERS = {
    "D_o":  (RIG / "loras_repro/llama-distillation/impulsiveness",        "dpo", "released", 123456),
    "S_o":  (RIG / "loras_repro/llama-introspection/impulsiveness",       "sft", "released", 123456),
    "D_n":  (RIG / "loras/llama-distillation/impulsiveness_regen",        "dpo", "P0",       123456),
    "S_n":  (RIG / "loras/llama-introspection/impulsiveness_regen",       "sft", "P0",       123456),
    "D_n2": (RIG / "loras/llama-distillation/impulsiveness_regen_s987654", "dpo", "P0",      987654),
    # the propagation arm's SFT stage. Its training seed is whatever introspection SFT defaults to,
    # NOT the 987654 passed to the DPO stage -- which is the point of including it here.
    "S_n2": (RIG / "loras/llama-introspection/impulsiveness_regen_s987654", "sft", "prop",    None),
    # seed2 changed --seed in BOTH stages, so its A factors should re-align at a different shared
    # draw: within-pair cos ~1, but cross-pair cos ~0 against the 123456 adapters.
    # P1 (exploratory paraphrase arm): both stages at 123456, so both should land in the
    # 123456 block alongside the released/P0 adapters.
    "D_p1": (RIG / "loras/llama-distillation/impulsiveness_paraphrase",      "dpo", "P1",     123456),
    "S_p1": (RIG / "loras/llama-introspection/impulsiveness_paraphrase",     "sft", "P1",     123456),
    "D_s2": (RIG / "loras_seed2/llama-distillation/impulsiveness",           "dpo", "seed2",   987654),
    "S_s2": (RIG / "loras_seed2/llama-introspection/impulsiveness",          "sft", "seed2",   987654),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/analysis/audit_A_factor_matrix.json")
    a = ap.parse_args()

    fac, meta = {}, {}
    for k, (d, stage, data, seed) in ADAPTERS.items():
        if not (d / "adapter_model.safetensors").exists():
            print(f"  skip {k}: no adapter at {d}")
            continue
        fac[k] = load_factors(d)
        meta[k] = {"dir": str(d), "stage": stage, "dataset": data, "seed": seed}
    keys = list(fac)
    mods = sorted(set.intersection(*(set(f) for f in fac.values())))
    print(f"adapters {keys}   common modules {len(mods)}\n")

    # Frobenius cosine over the CONCATENATION of all modules: one number per side per pair.
    # Equivalent to treating A as one block-diagonal matrix, which is what the merge does.
    def cosmat(side: int) -> np.ndarray:
        n = len(keys)
        ip = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                tot = sum(float(torch.sum(fac[keys[i]][m][side] * fac[keys[j]][m][side]))
                          for m in mods)
                ip[i, j] = ip[j, i] = tot
        d = np.sqrt(np.diag(ip))
        return ip / np.outer(d, d)

    out: dict[str, dict] = {"meta": meta, "n_modules": len(mods)}
    for side, name in ((1, "A"), (0, "B")):
        M = cosmat(side)
        print(f"cos({name}_i, {name}_j), all {len(mods)} modules concatenated:")
        print("        " + "".join(f"{k:>8s}" for k in keys))
        for i, k in enumerate(keys):
            print(f"  {k:6s}" + "".join(f"{M[i, j]:+8.4f}" for j in range(len(keys))))
        print()
        out[f"cos_{name}"] = {keys[i]: {keys[j]: float(M[i, j]) for j in range(len(keys))}
                             for i in range(len(keys))}

    # per-module spread for the two cells that matter most
    for i_k, j_k in (("D_o", "S_o"), ("D_n", "D_n2")):
        if i_k in fac and j_k in fac:
            v = np.array([float(torch.sum(fac[i_k][m][1] * fac[j_k][m][1])
                                / (fac[i_k][m][1].norm() * fac[j_k][m][1].norm())) for m in mods])
            print(f"per-module cos(A_{i_k}, A_{j_k}): mean {v.mean():+.4f}  "
                  f"min {v.min():+.4f}  max {v.max():+.4f}")
            out[f"per_module_cosA_{i_k}_{j_k}"] = {"mean": float(v.mean()), "min": float(v.min()),
                                                  "max": float(v.max())}

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
