#!/usr/bin/env python3
"""Size of the CAA attention-mask artifact, measured on a natural same-model pair.

    loving           released loving adapter applied on the fly, extracted --legacy-mask
                     (oct_rig/run_loving_sycophancy.sh, 2026-10-05)
    xchar_lov_start  the same adapter folded into the weights (bf16 merge_and_unload),
                     extracted with the corrected positional mask (scripts/run_xchar_measure.sh,
                     2026-10-14; 2c has defaulted to the corrected mask since dc4e234)

Fold fidelity (outputs/analysis/xchar_sft/checks_pretrain.json): mean KL 0.001 vs 1.10 for
loving-vs-base, so the pair differs essentially only in mask protocol. Their difference is
therefore the mask artifact at full scale (10 personas x 8 traits x ~500 questions), which the
2026-08-19 mask_diag (4 personas x 2 traits x 150 questions, activations not retained) could
not give.

Reported per layer, cross-fitted exactly as common_shift.stats_for_trait (40 half-splits):
  artifact   g, S of  V(lov_start) - V(loving)      normalised by mean_p ||V_base,p||
  same-protocol displacement   V(loving)    - V(base)   (legacy vs legacy)
  mixed-protocol displacement  V(lov_start) - V(base)   (corrected vs legacy) -- the
             comparison every post-2026-09-03 arm makes against the archived base
and the cross-fitted cosine between the two displacements' persona-common shifts.

This is a measurement of the artifact, not a correction: subtracting it from other arms
assumes it is model-independent, which is only approximately supported (mask_diag).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from common_shift import load_arm, cell, make_preps  # noqa: E402
from geometry_lib import batch_vectors, half_splits  # noqa: E402

CACHE = REPO / "outputs" / "_qcache"
OUT = REPO / "outputs" / "analysis" / "audit" / "mask_artifact.json"


def cross(prep_ref, preps: dict, splits):
    """Cross-fitted g2, dv2, reference norm, and pairwise <g_a, g_b> for every arm."""
    arms = list(preps)
    g2 = {a: 0.0 for a in arms}; dv2 = {a: 0.0 for a in arms}
    x = {(a, b): 0.0 for a in arms for b in arms}; bn = 0.0
    for A, B in splits:
        Vb = batch_vectors(prep_ref, [A, B])
        dA = {a: batch_vectors(preps[a], [A])[0] for a in arms}
        dB = {a: batch_vectors(preps[a], [B])[0] for a in arms}
        P = dA[arms[0]].shape[0]
        for a in arms:
            g2[a] += float(dA[a].mean(0) @ dB[a].mean(0))
            dv2[a] += float((dA[a] * dB[a]).sum() / P)
            for b in arms:
                x[a, b] += 0.5 * (float(dA[a].mean(0) @ dB[b].mean(0)) + float(dB[a].mean(0) @ dA[b].mean(0)))
        bn += float(np.sqrt(np.maximum((Vb[0] * Vb[1]).sum(-1), 0)).mean())
    n = len(splits)
    return ({a: g2[a] / n for a in arms}, {a: dv2[a] / n for a in arms},
            {k: v / n for k, v in x.items()}, bn / n)


def main():
    rng = np.random.default_rng(0)
    res = {"pair": ["loving (legacy)", "xchar_lov_start (corrected)"], "layers": {}}
    for layer in (15, 20):
        base = load_arm(CACHE, "base", [15, 20], layer)
        lov = load_arm(CACHE, "loving", [15, 20], layer)
        fix = load_arm(CACHE, "xchar_lov_start", [15, 20], layer)
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        rows = {}
        for ti, t in enumerate(base["traits"]):
            nq = int(base["nq"][ti])
            assert int(lov["nq"][ti]) == nq == int(fix["nq"][ti]), t
            splits = half_splits(nq, 40, rng)
            b, l, f = cell(base, ti, pidx), cell(lov, ti, pidx), cell(fix, ti, pidx)
            # displacements from the legacy base: same-protocol and mixed-protocol
            pd = make_preps(*b, {"same": l, "mixed": f})
            g2, dv2, x, bn = cross(pd["base"], pd["arms"], splits)
            # the artifact: corrected minus legacy, same model; normalised by the BASE norm
            pa = make_preps(*l, {"artifact": f})
            ga, dva, _, _ = cross(pa["base"], pa["arms"], splits)
            rows[t] = {
                "base_norm": bn,
                "artifact_g": np.sqrt(max(ga["artifact"], 0)) / bn,
                "artifact_rms_g": np.sqrt(max(dva["artifact"], 0)) / bn,   # per-persona RMS
                "artifact_S": ga["artifact"] / dva["artifact"],
                "same_g": np.sqrt(max(g2["same"], 0)) / bn, "mixed_g": np.sqrt(max(g2["mixed"], 0)) / bn,
                "same_S": g2["same"] / dv2["same"], "mixed_S": g2["mixed"] / dv2["mixed"],
                "same_rms": np.sqrt(max(dv2["same"], 0)) / bn, "mixed_rms": np.sqrt(max(dv2["mixed"], 0)) / bn,
                # raw energies kept so a reader can see whether the cosine is estimable
                "Ecommon_same": g2["same"] / bn**2, "Ecommon_mixed": g2["mixed"] / bn**2,
                "cos_same_mixed": x["same", "mixed"] / np.sqrt(g2["same"] * g2["mixed"])
                if g2["same"] > 0 and g2["mixed"] > 0 else None,
            }
            print(f"L{layer} {t:13s} artifact g={rows[t]['artifact_g']:.3f} S={rows[t]['artifact_S']:.2f} | "
                  f"g same={rows[t]['same_g']:.3f} mixed={rows[t]['mixed_g']:.3f} | "
                  f"S same={rows[t]['same_S']:.2f} mixed={rows[t]['mixed_S']:.2f} | cos={rows[t]['cos_same_mixed']}",
                  flush=True)
        keys = [k for k in next(iter(rows.values())) if k != "cos_same_mixed"]
        mean8 = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
        res["layers"][str(layer)] = {"traits": {t: {k: (float(v) if v is not None else None)
                                                    for k, v in r.items()} for t, r in rows.items()},
                                     "MEAN8": mean8}
        print(f"L{layer} MEAN8", {k: round(v, 3) for k, v in mean8.items()}, flush=True)
        del base, lov, fix
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
