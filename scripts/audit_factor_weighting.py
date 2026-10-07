#!/usr/bin/env python3
"""Audit 2 of the P0 post-mortem: where does the peft factor-merge actually sit?

The merge combines FACTORS, not updates (scripts/check_peft_merge.py):

    A_F = cD*A_D + cS*A_S,   B_F = cD*B_D + cS*B_S,   cD = sqrt(1.0*sD), cS = sqrt(0.25*sS)
    dW_F = dW_D + 0.25*dW_S + cD*cS*(B_D@A_S + B_S@A_D)

Two limits bracket it:

    cross-free additive   dW ~ 1.00*D + 0.25*S     (what --lora-adapter ... --lora-scale gives)
    shared-A limit        dW ~ 1.50*D + 0.75*S     (exact when A_D == A_S: then dW_F factors as
                                                    (cD B_D + cS B_S)(cD+cS) A, and with
                                                    sD = sS the D and S coefficients are
                                                    cD(cD+cS)/sD = 1.5 and cS(cD+cS)/sS = 0.75)

So the merge's effective stage weighting is governed by how much the two stages' A factors
overlap. This script measures that overlap, then asks how well the real merged update is
explained by ANY additive combination of the two stage updates:

    min over cD,cS  || dW_F - cD*dW_D - cS*dW_S ||_F

solved exactly from Frobenius inner products (a 2x2 normal equation per module, and one global
system over all modules). A small residual would mean the cross terms, though large in norm,
lie mostly inside span{D, S} and the merge is behaviourally an additive combination with
REWEIGHTED stages -- which is a testable claim, because the fitted surrogate can then be built
and measured. A large residual means the cross terms carry a genuinely new direction.

Nothing here forms a d_out x d_in matrix: every inner product uses
<P1@Q1, P2@Q2>_F = tr((P1^T P2)(Q2 Q1^T)), touching only r x r.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/audit_factor_weighting.py      # CPU, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossed_dpo_sft_weights import LowRankSum, load_factors, scaling  # noqa: E402

RIG = Path("/workspace/oct_rig")

PAIRS = {
    # name: (dpo, sft, merged)
    "repro": (RIG / "loras_repro/llama-distillation/impulsiveness",
              RIG / "loras_repro/llama-introspection/impulsiveness",
              RIG / "loras_repro/llama-personas/impulsiveness"),
    "p0":    (RIG / "loras/llama-distillation/impulsiveness_regen",
              RIG / "loras/llama-introspection/impulsiveness_regen",
              RIG / "loras/llama-personas/impulsiveness_regen"),
    "crossed_Dn_So": (RIG / "loras/llama-distillation/impulsiveness_regen",
                      RIG / "loras_repro/llama-introspection/impulsiveness",
                      RIG / "loras_crossed/llama-personas/crossed_Dn_So"),
    "crossed_Do_Sn": (RIG / "loras_repro/llama-distillation/impulsiveness",
                      RIG / "loras/llama-introspection/impulsiveness_regen",
                      RIG / "loras_crossed/llama-personas/crossed_Do_Sn"),
}

# the two reference weightings, as (c_D, c_S)
REFS = {"cross_free_additive": (1.00, 0.25), "shared_A_limit": (1.50, 0.75)}


def subspace_overlap(A1: torch.Tensor, A2: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between the two row spaces, in [0, 1].

    1.0 means identical r-dimensional row space (the shared-A limit's premise); r/d_in is the
    value expected for two independent random r-dim subspaces of R^d_in, which for r=64 and
    d_in=4096 is 0.0156 -- so the random baseline is printed alongside.
    """
    Q1 = torch.linalg.qr(A1.T, mode="reduced").Q          # d_in x r orthonormal
    Q2 = torch.linalg.qr(A2.T, mode="reduced").Q
    return float((Q1.T @ Q2).pow(2).sum() / min(A1.shape[0], A2.shape[0]))


def fit(M: LowRankSum, D: LowRankSum, S: LowRankSum) -> tuple[float, float, float, float]:
    """Least-squares cD, cS for M ~ cD*D + cS*S. Returns (cD, cS, ||R||, ||M||)."""
    g = np.array([[D.inner(D), D.inner(S)], [D.inner(S), S.inner(S)]])
    b = np.array([M.inner(D), M.inner(S)])
    c = np.linalg.solve(g, b)
    m2 = M.inner(M)
    r2 = max(m2 - float(c @ b), 0.0)                       # ||M||^2 - <M, fit>
    return float(c[0]), float(c[1]), float(np.sqrt(r2)), float(np.sqrt(max(m2, 0.0)))


def fixed_residual(M: LowRankSum, D: LowRankSum, S: LowRankSum, cD: float, cS: float) -> float:
    """||M - cD*D - cS*S||_F for coefficients that are NOT fitted."""
    r2 = (M.inner(M) + cD * cD * D.inner(D) + cS * cS * S.inner(S)
          - 2 * cD * M.inner(D) - 2 * cS * M.inner(S) + 2 * cD * cS * D.inner(S))
    return float(np.sqrt(max(r2, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="repro,p0,crossed_Dn_So,crossed_Do_Sn")
    ap.add_argument("--out", default="outputs/analysis/audit_factor_weighting.json")
    a = ap.parse_args()

    results: dict[str, dict] = {}
    for name in a.pairs.split(","):
        dD, dS, dF = PAIRS[name]
        for p in (dD, dS, dF):
            if not (p / "adapter_model.safetensors").exists():
                raise SystemExit(f"FATAL: missing adapter at {p}")
        fD, fS, fF = load_factors(dD), load_factors(dS), load_factors(dF)
        sD, sS, sF = scaling(dD), scaling(dS), scaling(dF)
        mods = sorted(set(fD) & set(fS) & set(fF))
        print(f"\n{'=' * 78}\n{name}:  D {dD.name}  S {dS.name}  F {dF.name}")
        print(f"  scalings alpha/r: D {sD:.3f}  S {sS:.3f}  F {sF:.3f}   common modules {len(mods)}")

        per: dict[str, dict] = {}
        glob_g = np.zeros((2, 2))
        glob_b = np.zeros(2)
        glob_m2 = 0.0
        glob_fixed = {k: 0.0 for k in REFS}
        for m in mods:
            BD, AD = fD[m]
            BS, AS = fS[m]
            BF, AF = fF[m]
            D = LowRankSum([(sD, BD, AD)])
            S = LowRankSum([(sS, BS, AS)])
            M = LowRankSum([(sF, BF, AF)])
            cD, cS, rn, mn = fit(M, D, S)
            per[m] = {
                "cos_A": float(torch.sum(AD * AS) / (AD.norm() * AS.norm())),
                "cos_B": float(torch.sum(BD * BS) / (BD.norm() * BS.norm())),
                "subspace_overlap_A": subspace_overlap(AD, AS),
                "c_D": cD, "c_S": cS,
                "rel_residual": rn / mn if mn > 0 else float("nan"),
                "norm_M": mn, "norm_D": D.norm(), "norm_S": S.norm(),
                "rel_residual_fixed": {k: fixed_residual(M, D, S, *v) / mn
                                       for k, v in REFS.items()},
            }
            # accumulate the GLOBAL system: block-diagonal dW, so inner products just add
            dd, ds, ss = D.inner(D), D.inner(S), S.inner(S)
            glob_g += np.array([[dd, ds], [ds, ss]])
            glob_b += np.array([M.inner(D), M.inner(S)])
            glob_m2 += M.inner(M)
            for k, v in REFS.items():
                glob_fixed[k] += fixed_residual(M, D, S, *v) ** 2

        c = np.linalg.solve(glob_g, glob_b)
        gres = float(np.sqrt(max(glob_m2 - float(c @ glob_b), 0.0)))
        gnorm = float(np.sqrt(glob_m2))
        g = {"c_D": float(c[0]), "c_S": float(c[1]),
             "residual_norm": gres, "norm_M": gnorm, "rel_residual": gres / gnorm,
             "rel_residual_fixed": {k: float(np.sqrt(v)) / gnorm for k, v in glob_fixed.items()},
             "norm_D": float(np.sqrt(glob_g[0, 0])), "norm_S": float(np.sqrt(glob_g[1, 1])),
             "cos_D_S": float(glob_g[0, 1] / np.sqrt(glob_g[0, 0] * glob_g[1, 1]))}

        ca = np.array([per[m]["cos_A"] for m in mods])
        cb = np.array([per[m]["cos_B"] for m in mods])
        so = np.array([per[m]["subspace_overlap_A"] for m in mods])
        rr = np.array([per[m]["rel_residual"] for m in mods])
        cDs = np.array([per[m]["c_D"] for m in mods])
        cSs = np.array([per[m]["c_S"] for m in mods])
        print("\n  A-factor overlap between the DPO and SFT stages (per module):")
        print(f"    cos(A_D, A_S)       mean {ca.mean():+.4f}  min {ca.min():+.4f}  max {ca.max():+.4f}")
        print(f"    cos(B_D, B_S)       mean {cb.mean():+.4f}  min {cb.min():+.4f}  max {cb.max():+.4f}")
        print(f"    rowspace overlap    mean {so.mean():.4f}  min {so.min():.4f}  max {so.max():.4f}"
              f"   (random r/d_in ~ 0.016-0.055)")
        print("\n  fit dW_F ~ c_D*dW_D + c_S*dW_S")
        print(f"    per-module  c_D  mean {cDs.mean():.3f} [{cDs.min():.3f}, {cDs.max():.3f}]")
        print(f"                c_S  mean {cSs.mean():.3f} [{cSs.min():.3f}, {cSs.max():.3f}]")
        print(f"                rel residual  mean {rr.mean():.4f} [{rr.min():.4f}, {rr.max():.4f}]")
        print(f"    GLOBAL      c_D {g['c_D']:.4f}   c_S {g['c_S']:.4f}")
        print(f"                ||R||/||dW_F|| {g['rel_residual']:.4f}   "
              f"(||dW_F|| {gnorm:.2f})   cos(D,S) {g['cos_D_S']:+.4f}")
        for k, v in g["rel_residual_fixed"].items():
            print(f"    fixed {k:22s} (c_D,c_S)={REFS[k]}  rel residual {v:.4f}")
        results[name] = {"dirs": {"dpo": str(dD), "sft": str(dS), "merged": str(dF)},
                         "scalings": {"D": sD, "S": sS, "F": sF},
                         "n_modules": len(mods), "global": g, "per_module": per}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"refs": REFS, "pairs": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
