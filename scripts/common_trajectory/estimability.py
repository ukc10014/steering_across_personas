"""Ratios of cross-fitted energies, and when they are estimable. Used by geometry.py.

THE PROBLEM THIS FIXES. Cross-fitted energies (inner products across disjoint question
halves) are unbiased, so when the true shift is near zero they are negative in a fraction of
bootstrap replicates. geometry.py's original derive() floored those denominators at 1e-12, which
turned "no estimable direction" into cosines of order 1e9 (sycophancy L20 MEAN8 cosT CI
[-4.6e9, +1.6e4]), and Q/S returned NaN there, which the summary then dropped with isfinite --
silently conditioning the interval on the replicates where the denominator happened to be
positive. Neither the floor nor the drop is an estimate.

RAW. Per trait, arm and replicate we keep (bn-normalised later):
    g2  = <dG_A, dG_B>        common energy, total (vs base)     dv2 = mean_p <dV_A, dV_B>
    gx  = <dG(arm), dG(final)> cross-fitted                       bn  = mean_p ||V_base,p||
    g2s, dv2s, gxs            the same for the incremental (vs DPO) decomposition

RULE (fixed here, before looking at which cells it excludes; documented in the report):
  A ratio is computed only where every denominator energy is > 0. A ratio for a trait/arm is
  ESTIMABLE iff its point denominators are > 0 AND at least VALID_MIN (97.5 %) of bootstrap
  replicates have positive denominators -- i.e. the one-sided 97.5 % bootstrap lower bound of the
  denominator energy is above zero. If estimable, its interval is the percentile interval over
  the valid replicates and the excluded fraction (< 2.5 %) is reported with it. If not, no point
  or interval is reported for the ratio; the raw energies (always defined) are reported instead.
  A multi-trait mean of a ratio (the original estimand) is estimable only if every member trait
  is, and a replicate contributes only if all traits are valid in it (same 97.5 % threshold).
  The ENERGY-POOLED version (ratio of trait-summed bn-normalised energies) is reported alongside
  as a different, explicitly labelled estimand: it weights traits by their energy and is far
  more often estimable. It does not replace the per-trait mean.
  Self-comparisons (arm == final) are 1 by construction and are reported as such, not
  bootstrapped.

LIMITATIONS. 97.5 % is a convention, matching the 95 % two-sided intervals reported
elsewhere; conditioning on the <= 2.5 % valid replicates still shades intervals slightly
toward larger |denominator|. g (= sqrt(max(E, 0))) is a monotone transform of E, so its
percentile interval is E's, floored at 0; report E when it matters whether a shift exists.
"""
from __future__ import annotations

import numpy as np

VALID_MIN = 0.975
RAW = ["g2", "dv2", "gx", "g2s", "dv2s", "gxs", "bn"]
I = {k: i for i, k in enumerate(RAW)}
# quantity -> (kind, denominators as (raw field, use final arm?))
QUANTITIES = {
    "Ecommon": ("energy", []), "Eresid": ("energy", []),
    "EcommonS": ("energy", []), "EresidS": ("energy", []),
    "g": ("magnitude", []), "gS": ("magnitude", []),
    "S": ("ratio", [("dv2", False)]), "SS": ("ratio", [("dv2s", False)]),
    "cosT": ("ratio", [("g2", False), ("g2", True)]),
    "cosS": ("ratio", [("g2s", False), ("g2s", True)]),
    "Q": ("ratio", [("g2s", True)]),
}
SELF_ONE = {"cosT", "cosS", "Q"}
INCREMENTAL = {"EcommonS", "EresidS", "gS", "SS", "cosS", "Q"}


def quantities(raw: np.ndarray, fi: int) -> dict:
    """raw (..., n_arms, 7) -> {q: (..., n_arms)} with NaN wherever a denominator is <= 0."""
    r = {k: raw[..., I[k]] for k in RAW}
    fin = {k: raw[..., fi:fi + 1, I[k]] for k in RAW}
    bn = r["bn"]
    out = {"Ecommon": r["g2"] / bn**2, "Eresid": (r["dv2"] - r["g2"]) / bn**2,
           "EcommonS": r["g2s"] / bn**2, "EresidS": (r["dv2s"] - r["g2s"]) / bn**2,
           "g": np.sqrt(np.clip(r["g2"], 0, None)) / bn,
           "gS": np.sqrt(np.clip(r["g2s"], 0, None)) / bn}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["S"] = np.where(r["dv2"] > 0, r["g2"] / r["dv2"], np.nan)
        out["SS"] = np.where(r["dv2s"] > 0, r["g2s"] / r["dv2s"], np.nan)
        ok = (r["g2"] > 0) & (fin["g2"] > 0)
        out["cosT"] = np.where(ok, r["gx"] / np.sqrt(np.abs(r["g2"] * fin["g2"])), np.nan)
        ok = (r["g2s"] > 0) & (fin["g2s"] > 0)
        out["cosS"] = np.where(ok, r["gxs"] / np.sqrt(np.abs(r["g2s"] * fin["g2s"])), np.nan)
        out["Q"] = np.where(fin["g2s"] > 0, r["gxs"] / fin["g2s"], np.nan)
    return out


def valid(raw: np.ndarray, fi: int, q: str) -> np.ndarray:
    """Boolean (..., n_arms): every denominator of q positive (energies/magnitudes: finite)."""
    kind, dens = QUANTITIES[q]
    ok = np.isfinite(raw[..., I["bn"]]) & (raw[..., I["bn"]] > 0)
    if q in INCREMENTAL:                        # undefined for the DPO state itself (raw NaN)
        ok &= np.isfinite(raw[..., I["g2s"]])
    for field, use_final in dens:
        v = raw[..., fi:fi + 1, I[field]] if use_final else raw[..., I[field]]
        ok = ok & np.isfinite(v) & (v > 0)
    return np.broadcast_to(ok, raw.shape[:-1])


def pooled(raw_traits: list[np.ndarray], fi: int) -> dict:
    """Energy-pooled ratios over traits: sums of bn-normalised energies, then the ratio."""
    n = [r[..., :6] / r[..., 6:7] ** 2 for r in raw_traits]
    s = np.sum(n, axis=0)                                           # (..., arms, 6)
    g2, dv2, gx, g2s, dv2s, gxs = (s[..., k] for k in range(6))
    f2, f2s = s[..., fi:fi + 1, 0], s[..., fi:fi + 1, 3]
    inc = np.isfinite(g2s)                  # incremental quantities do not exist for the DPO state
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = {"S": dv2 > 0, "SS": inc & (dv2s > 0), "cosT": (g2 > 0) & (f2 > 0),
              "cosS": inc & (g2s > 0) & (f2s > 0), "Q": inc & (f2s > 0)}
        val = {"S": g2 / dv2, "SS": g2s / dv2s, "cosT": gx / np.sqrt(np.abs(g2 * f2)),
               "cosS": gxs / np.sqrt(np.abs(g2s * f2s)), "Q": gxs / f2s}
        return {q: (np.where(ok[q], val[q], np.nan), np.broadcast_to(ok[q], val[q].shape)) for q in ok}


def _interval(b: np.ndarray):
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) if len(b) else (None, None)


def summarise(point: dict, boot: dict, arms: list[str], groups: dict) -> dict:
    """point[t] (arms, 7), boot[t] (B, arms, 7) raw -> summary per group/arm/quantity.

    Every entry: point, lo, hi, estimable, valid_frac (fraction of replicates where it is
    defined), and for ratios the reason when not estimable. Groups of >1 trait also carry
    `pooled` (the energy-pooled estimand) and n_traits_estimable."""
    fi = len(arms) - 1
    Qp = {t: quantities(point[t], fi) for t in point}
    Qb = {t: quantities(boot[t], fi) for t in boot}
    Vp = {t: {q: valid(point[t], fi, q) for q in QUANTITIES} for t in point}
    Vb = {t: {q: valid(boot[t], fi, q) for q in QUANTITIES} for t in boot}
    summ = {}
    for gname, ts in groups.items():
        summ[gname] = {}
        pool_p = pooled([point[t] for t in ts], fi) if len(ts) > 1 else None
        pool_b = pooled([boot[t] for t in ts], fi) if len(ts) > 1 else None
        for j, arm in enumerate(arms):
            summ[gname][arm] = {}
            for q, (kind, _) in QUANTITIES.items():
                if q in SELF_ONE and j == fi:
                    summ[gname][arm][q] = {"point": 1.0, "lo": 1.0, "hi": 1.0, "estimable": True,
                                           "valid_frac": 1.0, "note": "self-comparison: 1 by construction"}
                    continue
                pv = np.array([Vp[t][q][j] for t in ts])
                bv = np.all([Vb[t][q][:, j] for t in ts], axis=0)
                frac = float(bv.mean()) if len(bv) else 0.0
                per_trait_ok = [bool(Vp[t][q][j]) and float(Vb[t][q][:, j].mean()) >= VALID_MIN for t in ts]
                est = bool(pv.all()) and frac >= VALID_MIN and all(per_trait_ok)
                e = {"estimable": est, "valid_frac": frac}
                if kind == "ratio" and len(ts) > 1:
                    e["n_traits_estimable"] = int(sum(per_trait_ok))
                if est or (kind != "ratio" and pv.all()):
                    p = float(np.mean([Qp[t][q][j] for t in ts]))
                    b = np.mean([Qb[t][q][:, j] for t in ts], axis=0)[bv]
                    lo, hi = _interval(b)
                    e.update(point=p, lo=lo, hi=hi)
                else:
                    e.update(point=None, lo=None, hi=None,
                             reason=("denominator energy not bounded away from zero" if kind == "ratio"
                                     else "undefined for this state"))
                if kind == "ratio" and pool_p is not None:
                    pp, pok = pool_p[q]; pb, bok = pool_b[q]
                    pf = float(bok[:, j].mean())
                    pest = bool(pok[j]) and pf >= VALID_MIN
                    lo, hi = _interval(pb[:, j][bok[:, j]]) if pest else (None, None)
                    e["pooled"] = {"point": float(pp[j]) if pest else None, "lo": lo, "hi": hi,
                                   "estimable": pest, "valid_frac": pf}
                summ[gname][arm][q] = e
    return summ
