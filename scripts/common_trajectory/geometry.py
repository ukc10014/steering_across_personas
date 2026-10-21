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
from estimability import RAW, QUANTITIES, quantities, summarise          # noqa: E402
from common_shift import load_arms_checked                             # noqa: E402

QTY = list(QUANTITIES)


def raw_stats(prep_ref, prep_d: dict, splits, final: str):
    """Cross-fitted accumulators, the formulas of common_shift.stats_for_trait, kept raw.

    All halves of all splits go through ONE batch_vectors GEMM per arm: the per-split calls
    re-read the (questions x personas*hidden) block 2*len(splits) times and were memory-bound
    (~8 min per trait). Same weights, same sums; only floating-point summation order differs.
    """
    arms = list(prep_d)
    idx = [h for A, B in splits for h in (A, B)]
    V = {a: batch_vectors(prep_d[a], idx) for a in arms}          # (2n, P, H)
    dA = {a: V[a][0::2] for a in arms}; dB = {a: V[a][1::2] for a in arms}
    gA = {a: dA[a].mean(1) for a in arms}; gB = {a: dB[a].mean(1) for a in arms}   # (n, H)
    P = dA[arms[0]].shape[1]
    acc = {a: np.array([float(np.einsum("sh,sh->", gA[a], gB[a])),
                        float(np.einsum("sph,sph->", dA[a], dB[a]) / P),
                        0.5 * (float(np.einsum("sh,sh->", gA[a], gB[final]))
                               + float(np.einsum("sh,sh->", gB[a], gA[final])))]) for a in arms}
    bn = 0.0
    if prep_ref is not None:
        Vb = batch_vectors(prep_ref, idx)
        bn = float(np.sqrt(np.maximum((Vb[0::2] * Vb[1::2]).sum(-1), 0)).mean(1).sum())
    n = len(splits)
    return {a: v / n for a, v in acc.items()}, bn / n


def pack_raw(tot, inc, bn, arms) -> np.ndarray:
    """Raw accumulators -> (n_arms, 7) in estimability.RAW order; incremental NaN for the DPO state."""
    out = np.full((len(arms), len(RAW)), np.nan)
    for j, a in enumerate(arms):
        out[j, :3] = tot[a]
        if inc and a in inc:
            out[j, 3:6] = inc[a]
        out[j, 6] = bn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--half-splits", type=int, default=40)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--boot-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(REPO / "outputs" / "_qcache"))
    ap.add_argument("--base-arm", default="base",
                    help="reference state for the TOTAL decomposition; must share the trajectory's "
                         "mask protocol (e.g. a corrected-mask base re-extraction)")
    ap.add_argument("--allow-mixed-mask", action="store_true",
                    help="ONLY to reproduce the historical mixed-protocol numbers; recorded in the output")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/analysis/common_trajectory/v2"))
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _synthbase")
    a = ap.parse_args()
    cache = Path(a.cache_dir); outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    traj = trajectory(a.constitution)
    arms = [s["arm"] for s in traj]
    dpo, final = arms[0], arms[-1]
    rng = np.random.default_rng(a.seed)
    res = {"constitution": a.constitution, "states": traj, "quantities": QTY,
           "estimator": "cross-fitted (common_shift.stats_for_trait formulas), 40 half-splits",
           "bootstrap": a.bootstrap, "base_arm": a.base_arm, "layers": {},
           "estimability": "scripts/common_trajectory/estimability.py (VALID_MIN rule; no epsilon floors)"}

    for layer in a.layers:
        loaded, prov = load_arms_checked(cache, [a.base_arm, *arms], [15, 20], layer,
                                         allow_mixed_mask=a.allow_mixed_mask)
        res["provenance"] = prov
        base = loaded.pop(a.base_arm); D = loaded
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
            praw = pack_raw(tot, inc, bnorm, arms)
            pq = quantities(praw, len(arms) - 1)
            # reuse check: identical to the established estimator on the same splits (where
            # that estimator is defined: it floors negative common energy at 0)
            ref = stats_for_trait(ptot, splits, arms, naive=False)["per_arm"]
            for j, x in enumerate(arms):
                assert np.isclose(pq["g"][j], ref[x]["g_over_base"], rtol=1e-5), (t, x, "g")
                assert np.isclose(max(pq["S"][j], 0.0), ref[x]["share_squared"], rtol=1e-5, atol=1e-12), (t, x, "S")
            L["point"][t] = praw

            boot = np.full((a.bootstrap, len(arms), len(RAW)), np.nan)
            for r in range(a.bootstrap):
                bs = bootstrap_half_splits(nq, a.boot_splits, rng)
                if not bs:
                    continue          # a draw with too few distinct questions: stays NaN, counted below
                tb, bnb = raw_stats(ptot["base"], ptot["arms"], bs, final)
                ib, _ = raw_stats(None, pinc["arms"], bs, final)
                boot[r] = pack_raw(tb, ib, bnb, arms)
            L["boot"][t] = boot
            print(f"L{layer} {t}: S(dpo)={pq['S'][0]:.3f} S(final)={pq['S'][-1]:.3f} "
                  f"SS(final)={pq['SS'][-1]:.3f} g(final)={pq['g'][-1]:.3f}", flush=True)

        # summaries: per trait, the 8-trait unweighted mean, and the prespecified targets
        groups = {t: [t] for t in traits}
        groups["MEAN8"] = traits
        from states import TARGETS
        groups["TARGETS"] = TARGETS.get(a.constitution) or traits      # none prespecified -> 8-trait mean (A1)
        nboot_ok = {t: int(np.isfinite(L["boot"][t][:, 0, 0]).sum()) for t in traits}
        summ = summarise(L["point"], {t: L["boot"][t][np.isfinite(L["boot"][t][:, 0, 0])] for t in traits},
                         arms, groups)
        res["layers"][str(layer)] = {"traits": traits, "personas": L["personas"], "summary": summ,
                                     "bootstrap_replicates_drawn": nboot_ok}
        np.savez_compressed(outdir / f"geometry_raw_{a.constitution}_L{layer}{a.tag}.npz",
                            **{f"point_{t}": L["point"][t] for t in traits},
                            **{f"boot_{t}": L["boot"][t] for t in traits},
                            arms=np.array(arms), raw=np.array(RAW))

    out = outdir / f"geometry_{a.constitution}{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
