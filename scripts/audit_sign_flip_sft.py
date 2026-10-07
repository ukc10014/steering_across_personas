#!/usr/bin/env python3
"""Audit 1 of the P0 post-mortem: the peft factor-merge is coordinate-dependent.

A LoRA update is invariant under A -> -A, B -> -B, because B'A' = (-B)(-A) = BA exactly. So an
adapter with both factors negated is the SAME FUNCTION as the original -- bit-for-bit, not
approximately. But add_weighted_adapter(combination_type="linear") combines the FACTORS:

    A_F = cD*A_D + cS*A_S,   B_F = cD*B_D + cS*B_S

and negating A_S, B_S flips their contribution to both sums, so the merged weight CHANGES even
though neither input's standalone behaviour did. Combined with the shared-A finding
(scripts/audit_A_factor_matrix.py: cos(A_D, A_S) = 0.989, because A is essentially the random
init and the two stages share a training seed), the effect is predictable in closed form. Writing
A_D ~ A_S ~ A and sD = sS = s:

    as trained   dW_F ~ cD(cD+cS)/s * D + cS(cD+cS)/s * S  =  1.50*D + 0.75*S
    S negated    dW_F ~ cD(cD-cS)/s * D - cS(cD-cS)/s * S  =  0.50*D - 0.25*S

i.e. a sign convention internal to one adapter changes the DPO stage's effective dose threefold.

This script (a) builds the negated SFT adapter, (b) verifies per module that B'A' == B A to the
last bit and that the factors really are negated, (c) refits the merged update it implies to
c_D*dW_D + c_S*dW_S, and (d) leaves the adapter on disk so scripts/merge_crossed.py can build the
merge and the behavioural half can measure it.

THIS IS A COORDINATE-DEPENDENCE DEMONSTRATION, NOT A CLAIM ABOUT THE P0 GAP. P0 and the
reproduction were both merged under the same convention; nothing here shows the convention caused
their difference. What it shows is that the merge's effective stage weighting is not a property
of the two updates alone.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/audit_sign_flip_sft.py       # CPU, no GPU
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossed_dpo_sft_weights import LowRankSum, load_factors, scaling  # noqa: E402
from audit_factor_weighting import fit, fixed_residual  # noqa: E402

RIG = Path("/workspace/oct_rig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", default=str(RIG / "loras_repro/llama-distillation/impulsiveness"))
    ap.add_argument("--sft", default=str(RIG / "loras_repro/llama-introspection/impulsiveness"))
    ap.add_argument("--out-adapter",
                    default=str(RIG / "loras_signflip/llama-introspection/impulsiveness_negAB"))
    ap.add_argument("--weights", default="1.0,0.25")
    ap.add_argument("--out", default="outputs/analysis/audit_sign_flip_sft.json")
    a = ap.parse_args()

    dD, dS = Path(a.dpo), Path(a.sft)
    wD, wS = (float(x) for x in a.weights.split(","))
    sD, sS = scaling(dD), scaling(dS)
    cD, cS = math.sqrt(wD * sD), math.sqrt(wS * sS)
    print(f"dpo {dD}  alpha/r {sD:.3f}")
    print(f"sft {dS}  alpha/r {sS:.3f}")
    print(f"merge weights ({wD}, {wS})  ->  cD {cD:.6f}  cS {cS:.6f}\n")

    # (a) build the negated adapter: copy the directory, negate every lora_A and lora_B tensor
    out_ad = Path(a.out_adapter)
    if out_ad.exists() and any(out_ad.iterdir()):
        raise SystemExit(f"FATAL: {out_ad} exists and is non-empty. Refusing to overwrite.")
    out_ad.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(dS / "adapter_model.safetensors"))
    flipped = {k: (-v if (".lora_A" in k or ".lora_B" in k) else v.clone())
               for k, v in sd.items()}
    n_flip = sum(1 for k in sd if ".lora_A" in k or ".lora_B" in k)
    save_file(flipped, str(out_ad / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    shutil.copy2(dS / "adapter_config.json", out_ad / "adapter_config.json")
    (out_ad / "SIGNFLIP_PROVENANCE.json").write_text(json.dumps(
        {"kind": "diagnostic sign-flipped adapter, NOT a training trajectory",
         "source": str(dS), "transform": "A -> -A, B -> -B on every LoRA factor",
         "standalone_function": "IDENTICAL to the source adapter (B'A' = BA exactly)",
         "why": "audit 1 of docs/runs/oct/P0_MECHANISM_FINDINGS.md -- shows the peft factor "
                "merge's effective stage weighting is coordinate-dependent",
         "caveat": "does NOT show the sign convention caused the P0 gap; both arms used the "
                   "same convention"}, indent=2))
    print(f"(a) wrote {out_ad}  ({n_flip} factor tensors negated, "
          f"{len(sd) - n_flip} copied unchanged)")

    # (b) verify, per module, that the FUNCTION is unchanged and the factors really are negated
    fD, fS, fSf = load_factors(dD), load_factors(dS), load_factors(out_ad)
    mods = sorted(set(fD) & set(fS) & set(fSf))
    max_fun_err, max_fun_mod = 0.0, ""
    for m in mods:
        BS, AS = fS[m]
        BF, AF = fSf[m]
        if not (torch.equal(AF, -AS) and torch.equal(BF, -BS)):
            raise SystemExit(f"FATAL: {m} factors are not exactly negated")
        # B'A' - BA must be identically zero, not merely small: same magnitudes, same order
        e = float((BF @ AF - BS @ AS).abs().max())
        if e > max_fun_err:
            max_fun_err, max_fun_mod = e, m
    print(f"(b) {len(mods)} modules: factors exactly negated; "
          f"max |B'A' - BA| = {max_fun_err:.3e} ({max_fun_mod or 'all zero'})")
    if max_fun_err != 0.0:
        raise SystemExit("FATAL: the standalone function changed -- the premise of the audit fails")

    # (c) refit the merge each convention implies. Built analytically from the factors, which is
    #     exact: this is the same algebra add_weighted_adapter applies.
    rows = []
    for label, (fsft, sign) in {"as trained": (fS, +1.0), "S negated": (fSf, -1.0)}.items():
        gD = np.zeros((2, 2)); gB = np.zeros(2); m2 = 0.0
        fixed = {"nominal (1.0, 0.25)": 0.0, "shared-A (1.5, 0.75)": 0.0}
        for m in mods:
            BD, AD = fD[m]
            BS, AS = fsft[m]
            AM = cD * AD + cS * AS
            BM = cD * BD + cS * BS
            D = LowRankSum([(sD, BD, AD)])
            # the basis is the ORIGINAL S update in both rows -- it is the same function either
            # way, so fitting against it is what makes the two c_S values comparable
            S = LowRankSum([(sS, *fS[m])])
            M = LowRankSum([(1.0, BM, AM)])          # merged adapter lands at alpha == r
            dd, ds, ss = D.inner(D), D.inner(S), S.inner(S)
            gD += np.array([[dd, ds], [ds, ss]])
            gB += np.array([M.inner(D), M.inner(S)])
            m2 += M.inner(M)
            fixed["nominal (1.0, 0.25)"] += fixed_residual(M, D, S, 1.0, 0.25) ** 2
            fixed["shared-A (1.5, 0.75)"] += fixed_residual(M, D, S, 1.5, 0.75) ** 2
        c = np.linalg.solve(gD, gB)
        res = math.sqrt(max(m2 - float(c @ gB), 0.0))
        nm = math.sqrt(m2)
        rows.append({"convention": label, "c_D": float(c[0]), "c_S": float(c[1]),
                     "norm_M": nm, "rel_residual": res / nm,
                     "rel_residual_fixed": {k: math.sqrt(v) / nm for k, v in fixed.items()}})

    print("\n(c) fit dW_merge ~ c_D*dW_D + c_S*dW_S   (same two stage updates in both rows)")
    print(f"    {'convention':14s}{'c_D':>8s}{'c_S':>8s}{'||dW||':>10s}{'rel resid':>11s}")
    print("    " + "-" * 51)
    for r in rows:
        print(f"    {r['convention']:14s}{r['c_D']:+8.3f}{r['c_S']:+8.3f}"
              f"{r['norm_M']:10.3f}{r['rel_residual']:11.4f}")
    pred = {"as trained": (cD * (cD + cS) / sD, cS * (cD + cS) / sS),
            "S negated": (cD * (cD - cS) / sD, -cS * (cD - cS) / sS)}
    print(f"\n    closed-form shared-A prediction (A_D == A_S exactly):")
    for k, (p, q) in pred.items():
        print(f"      {k:14s}c_D {p:+.3f}  c_S {q:+.3f}")
    r0, r1 = rows[0]["norm_M"], rows[1]["norm_M"]
    print(f"\n    merged-update norm changes by {r1 / r0:.3f}x under a transform that leaves "
          f"both inputs' standalone function bit-identical")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dpo": str(dD), "sft": str(dS), "flipped_adapter": str(out_ad),
                             "weights": [wD, wS], "cD": cD, "cS": cS,
                             "scalings": {"D": sD, "S": sS}, "n_modules": len(mods),
                             "max_abs_function_change": max_fun_err,
                             "fits": rows,
                             "closed_form_shared_A": {k: list(v) for k, v in pred.items()},
                             "norm_ratio_flipped_over_trained": r1 / r0}, indent=2))
    print(f"\nwrote {p}")
    print(f"\nnext (GPU): PYTHONPATH=$PYLIBS_TRAIN python3 scripts/merge_crossed.py \\\n"
          f"    --dpo {dD} --sft {out_ad} \\\n"
          f"    --out {RIG}/loras_signflip/llama-personas/signflip_So_neg")


if __name__ == "__main__":
    main()
