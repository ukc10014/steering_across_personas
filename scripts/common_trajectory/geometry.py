#!/usr/bin/env python3
"""Persona-common representational trajectory. docs/spec_common_trajectory.md §6-8.

For every state e of a constitution's trajectory (states.py), per trait t, over the ten
semantic personas p (null and nonsense excluded), two decompositions on the SAME question
splits:

  TOTAL (vs the unadapted base)          dV_p(e)   = V_p(e) - V_base,p
  INCREMENTAL SFT (vs the DPO-only state) dV^S_p(e) = V_p(e) - V_p(0)       (e > 0 only)

  dG = mean_p dV_p ;  common energy  ||dG||^2 ;  total energy  mean_p ||dV_p||^2
  g      = ||dG|| / mean_p ||V_base,p||           common-shift magnitude (A)
  S      = ||dG||^2 / mean_p ||dV_p||^2           persona-common squared share (B)
  cos_T  = cos(dG(e), dG(final))                  alignment with the final common shift (C)
  Q      = <dG^S(e), dG^S(final)> / ||dG^S(final)||^2   projection on the final SFT direction

ESTIMATOR. Every quadratic quantity is cross-fitted exactly as scripts/common_shift.py does:
vectors from disjoint question halves A, B, inner products taken ACROSS halves, averaged over
40 random half-splits. That removes the noise bias (common_shift.py header). It also handles
the correlated-reference problem for C and Q: dV(e) and dV(final) both subtract the same
reference estimate, but half A of one is only ever paired with half B of the other. The point
estimates are asserted equal to common_shift.stats_for_trait's before anything is written.

Energies are kept RAW (unclipped, so they can be slightly negative when a shift is near
zero) and reported over mean_p||V_base,p||^2, so traits are on a common scale before any
8-trait mean. Ratios are flagged unstable when the denominator's bootstrap CI includes zero.

UNCERTAINTY. Question bootstrap (geometry_lib.bootstrap_half_splits), ONE draw per replicate
per trait shared by every state and both decompositions, so differences between checkpoints
are paired. Traits use disjoint question sets and independent draws, so replicate r of the
8-trait mean is the mean of the traits' replicate r. This is evaluation uncertainty only:
not training-seed variance and not checkpoint-to-checkpoint scatter (SFT_CURVE_REPORT.md).

    python3 scripts/common_trajectory/geometry.py --constitution impulsiveness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_shift import load_arm, cell, make_preps, stats_for_trait   # noqa: E402
from geometry_lib import batch_vectors, half_splits, bootstrap_half_splits  # noqa: E402
from states import trajectory                                          # noqa: E402

QTY = ["g", "S", "Ecommon", "Eresid", "cosT", "gS", "SS", "EcommonS", "EresidS", "cosS", "Q"]


def raw_stats(prep_ref, prep_d: dict, splits, final: str):
    """Cross-fitted accumulators, the formulas of common_shift.stats_for_trait, kept raw."""
    arms = list(prep_d)
    acc = {a: np.zeros(3) for a in arms}            # g2, dv2, <g(a), g(final)>
    bn = 0.0
    for A, B in splits:
        Vb = batch_vectors(prep_ref, [A, B]) if prep_ref is not None else None
        dA = {a: batch_vectors(prep_d[a], [A])[0] for a in arms}
        dB = {a: batch_vectors(prep_d[a], [B])[0] for a in arms}
        gA = {a: dA[a].mean(0) for a in arms}; gB = {a: dB[a].mean(0) for a in arms}
        P = dA[arms[0]].shape[0]
        for a in arms:
            acc[a] += [float(gA[a] @ gB[a]), float((dA[a] * dB[a]).sum() / P),
                       0.5 * (float(gA[a] @ gB[final]) + float(gB[a] @ gA[final]))]
        if Vb is not None:
            bn += float(np.sqrt(np.maximum((Vb[0] * Vb[1]).sum(-1), 0)).mean())
    n = len(splits)
    return {a: v / n for a, v in acc.items()}, bn / n


def derive(tot, inc, bn, final):
    """Raw accumulators -> the named quantities for every state (NaN where undefined)."""
    out = {}
    gf, gfS = tot[final][0], (inc[final][0] if inc else np.nan)
    for a, (g2, dv2, gx) in tot.items():
        r = {"g": np.sqrt(max(g2, 0.0)) / bn, "S": g2 / dv2 if dv2 > 0 else np.nan,
             "Ecommon": g2 / bn ** 2, "Eresid": (dv2 - g2) / bn ** 2,
             "cosT": gx / np.sqrt(max(g2, 1e-12) * max(gf, 1e-12))}
        if inc and a in inc:
            g2s, dv2s, gxs = inc[a]
            r.update({"gS": np.sqrt(max(g2s, 0.0)) / bn, "SS": g2s / dv2s if dv2s > 0 else np.nan,
                      "EcommonS": g2s / bn ** 2, "EresidS": (dv2s - g2s) / bn ** 2,
                      "cosS": gxs / np.sqrt(max(g2s, 1e-12) * max(gfS, 1e-12)),
                      "Q": gxs / gfS if gfS > 0 else np.nan})
        else:
            r.update({k: np.nan for k in ("gS", "SS", "EcommonS", "EresidS", "cosS", "Q")})
        out[a] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--half-splits", type=int, default=40)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--boot-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = REPO / "outputs" / "_qcache"
    traj = trajectory(a.constitution)
    arms = [s["arm"] for s in traj]
    dpo, final = arms[0], arms[-1]
    rng = np.random.default_rng(a.seed)
    res = {"constitution": a.constitution, "states": traj, "quantities": QTY,
           "estimator": "cross-fitted (common_shift.stats_for_trait formulas), 40 half-splits",
           "bootstrap": a.bootstrap, "layers": {}}

    for layer in a.layers:
        base = load_arm(cache, "base", [15, 20], layer)
        D = {arm: load_arm(cache, arm, [15, 20], layer) for arm in arms}
        traits = base["traits"]
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        assert len(pidx) == 10 and not {"null", "nonsense"} & set(base["semantic"])
        L = {"traits": traits, "personas": base["semantic"], "point": {}, "boot": {}}
        for ti, t in enumerate(traits):
            nq = int(base["nq"][ti])
            assert all(int(D[x]["nq"][ti]) == nq for x in arms), f"{t}: question counts differ"
            bp, bn_ = cell(base, ti, pidx)
            cells = {x: cell(D[x], ti, pidx) for x in arms}
            ptot = make_preps(bp, bn_, cells)
            dp, dn = cells[dpo]
            pinc = make_preps(dp, dn, {x: cells[x] for x in arms[1:]})
            splits = half_splits(nq, a.half_splits, rng)

            tot, bnorm = raw_stats(ptot["base"], ptot["arms"], splits, final)
            inc, _ = raw_stats(None, pinc["arms"], splits, final)
            pt = derive(tot, inc, bnorm, final)
            # reuse check: identical to the established estimator on the same splits
            ref = stats_for_trait(ptot, splits, arms, naive=False)["per_arm"]
            for x in arms:
                assert np.isclose(pt[x]["g"], ref[x]["g_over_base"], rtol=1e-5), (t, x, "g")
                assert np.isclose(pt[x]["S"], ref[x]["share_squared"], rtol=1e-5) or ref[x]["share_squared"] == 0, (t, x, "S")
            L["point"][t] = {x: {k: float(v) for k, v in pt[x].items()} for x in arms}

            boot = np.full((a.bootstrap, len(arms), len(QTY)), np.nan)
            for r in range(a.bootstrap):
                bs = bootstrap_half_splits(nq, a.boot_splits, rng)
                if not bs:
                    continue
                tb, bnb = raw_stats(ptot["base"], ptot["arms"], bs, final)
                ib, _ = raw_stats(None, pinc["arms"], bs, final)
                d = derive(tb, ib, bnb, final)
                boot[r] = [[d[x][k] for k in QTY] for x in arms]
            L["boot"][t] = boot
            print(f"L{layer} {t}: S(dpo)={pt[dpo]['S']:.3f} S(final)={pt[final]['S']:.3f} "
                  f"SS(final)={pt[final]['SS']:.3f} g(final)={pt[final]['g']:.3f}", flush=True)

        # summaries: per trait, the 8-trait unweighted mean, and the prespecified targets
        summ = {}
        groups = {t: [t] for t in traits}
        groups["MEAN8"] = traits
        from states import TARGETS
        groups["TARGETS"] = TARGETS[a.constitution]
        for gname, ts in groups.items():
            summ[gname] = {}
            for j, x in enumerate(arms):
                summ[gname][x] = {}
                for k_i, k in enumerate(QTY):
                    p = float(np.mean([L["point"][t][x][k] for t in ts]))
                    b = np.mean([L["boot"][t][:, j, k_i] for t in ts], 0)
                    b = b[np.isfinite(b)]
                    summ[gname][x][k] = {"point": p,
                                         "lo": float(np.percentile(b, 2.5)) if len(b) else None,
                                         "hi": float(np.percentile(b, 97.5)) if len(b) else None}
        res["layers"][str(layer)] = {"traits": traits, "personas": L["personas"], "summary": summ}
        np.savez_compressed(REPO / f"outputs/analysis/common_trajectory/geometry_boot_{a.constitution}_L{layer}.npz",
                            **{t: L["boot"][t] for t in traits}, arms=np.array(arms), qty=np.array(QTY))

    out = REPO / f"outputs/analysis/common_trajectory/geometry_{a.constitution}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
