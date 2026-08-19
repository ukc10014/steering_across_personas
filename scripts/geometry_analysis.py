#!/usr/bin/env python3
"""Persona/trait geometry across character-training arms: dispersion, RDMs, Procrustes.

Consumes the per-question cache from scripts/build_question_cache.py and implements
sections 5, 6, 8 and 9 of the character-arms analysis plan. Estimators live in
scripts/geometry_lib.py; run that file directly to see their self-tests.

WHAT THIS IS FOR. The raw persona-mean cosine-to-null said character training compresses
persona structure. Commit d44a267 retracted that: cosine is not invariant to adding a
common vector to both arguments, and merging an r=64 LoRA adds one. Everything here is
built to be immune to that specific failure:

  dispersion  centres personas on their own centroid first, so any component common to all
              ten personas is removed by construction rather than estimated away
  RDM         uses pairwise differences, in which a common additive vector cancels exactly
              -- (V_p + M) - (V_q + M) = V_p - V_q -- with no estimation step at all
  Procrustes  asks whether one global orthogonal map explains the arm difference, which is
              the "the coordinates moved, the relationships did not" confound

WHAT IS RESAMPLED. The bootstrap resamples CAA questions with replacement, and the SAME
draw is reused across arms so that arm-minus-base comparisons are paired. Question counts
are per trait (empathy ships 499, not 500) so no cell ever resamples padding.

WHAT THE INTERVALS DO NOT COVER, stated because it is easy to overclaim. These intervals
quantify uncertainty from the sampled CAA questions ONLY. They condition on the particular
ten personas and eight traits, and say nothing about generalisation to other personas or
other behavioural traits. With n=10 and n=8 there is no precise population inference to be
had here. --cluster-bootstrap adds a secondary resample over personas and traits for the
aggregate summaries; treat it as indicative, not as a substitute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geometry_lib import (bootstrap_half_splits, dispersion_crossfit, prepare_diff,
                          linear_map_cv,
                          attenuation_corrected, dispersion_naive, half_splits,
                          rdm_reliability,
                          procrustes_cv, procrustes_lowrank, rdm_crossfit, rdm_naive,
                          rdm_shape, trait_vectors)
from persona_steering.config import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--arms", nargs="+",
                   default=["base", "goodness", "mathematical", "impulsiveness",
                            "misalignment"])
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--cache-layers", type=int, nargs="+", default=[15, 20],
                   help="layer set the cache was built with (names the file)")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--half-splits", type=int, default=40,
                   help="random question half-splits per cross-fitted point estimate")
    p.add_argument("--boot-splits", type=int, default=40,
                   help="half-splits averaged within each bootstrap replicate. Defaults to "
                        "the same value as --half-splits ON PURPOSE: a correlation between "
                        "two noisier RDMs attenuates more, so replicates built from fewer "
                        "splits than the point estimate produce an interval that sits "
                        "systematically BELOW it. The bootstrap DRAW stays fixed within a "
                        "replicate (see geometry_lib.bootstrap_half_splits)")
    p.add_argument("--procrustes-rank", type=int, default=40,
                   help="dimension of the per-arm basis. Centred 8x10 cells span "
                        "at most 8*9=72 dims, so 40 is a real restriction; raise "
                        "it if basis coverage is the binding constraint")
    p.add_argument("--rank-sweep", type=int, nargs="+", default=[20, 30, 40, 50, 60],
                   help="ranks for the Procrustes sensitivity sweep")
    p.add_argument("--cluster-bootstrap", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------------------

def load_arm(cache_dir: Path, arm: str, cache_layers: list[int], layer: int):
    tag = "_".join(str(l) for l in cache_layers)
    f = cache_dir / f"{arm}_L{tag}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- run scripts/build_question_cache.py first")
    z = np.load(f, allow_pickle=False)
    li = list(z["layers"]).index(layer)
    acts = z["acts"][:, :, :, :, li, :]                       # (T, P, 2, nq, H)
    return {"acts": acts,
            "traits": [str(t) for t in z["traits"]],
            "personas": [str(p) for p in z["personas"]],
            "semantic": [str(p) for p in z["semantic_personas"]],
            "nq_per_trait": z["n_questions_per_trait"]}


def cell_arrays(d: dict, ti: int, pidx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """-> pos, neg each (n_personas_selected, nq_true, H) float32."""
    nq = int(d["nq_per_trait"][ti])
    a = d["acts"][ti][pidx][:, :, :nq, :]
    return a[:, 0].astype(np.float32), a[:, 1].astype(np.float32)


def shared_boot_splits(rng, nq: int, n_reps: int, inner: int) -> list[list[tuple]]:
    """Bootstrap split-lists generated ONCE and reused across arms.

    Reusing the identical draw on every arm is what makes arm-minus-base paired: the
    question-sampling noise is common to both sides of the comparison and largely cancels,
    instead of adding.
    """
    return [bootstrap_half_splits(nq, inner, rng) for _ in range(n_reps)]


def ci(x: np.ndarray, axis=0) -> tuple:
    return (float(np.percentile(x, 2.5, axis=axis)),
            float(np.percentile(x, 97.5, axis=axis)))


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------------------

def fold_centre(V: np.ndarray, T: int, P: int, train_idx: np.ndarray) -> np.ndarray:
    """Persona-centre within each trait, using only that trait's TRAINING personas.

    THE LEAKAGE THIS FIXES. Centring all ten personas of a trait before splitting makes
    them sum to zero within that trait, so a held-out persona is EXACTLY determined by the
    other nine:  Z_{t,p*} = -sum_{p != p*} Z_{t,p}. Leave-one-persona-out cross-validation
    on such a matrix is not cross-validation at all -- the test row is a linear combination
    of the training rows. Centring per fold, on the training personas only, removes the
    trait's shared component without manufacturing that dependency.

    When an entire trait is held out (the hold-out-traits scheme) none of its personas are
    in train, and its own ten are used. That is not train-to-test leakage: no training
    information enters, and the alternative -- leaving the trait's large shared component
    in -- is what made the uncentred Procrustes meaningless in the first place.
    """
    Vc = V.copy()
    train_set = set(int(i) for i in train_idx)
    for t in range(T):
        rows = np.arange(t * P, (t + 1) * P)
        tr_rows = [int(r) for r in rows if int(r) in train_set]
        ref = tr_rows if tr_rows else list(rows)
        Vc[rows] = V[rows] - V[ref].mean(0, keepdims=True)
    return Vc


def _basis_coverage(X: np.ndarray, train: np.ndarray, test: np.ndarray, rank: int) -> float:
    """Fraction of held-out row energy that a rank-r basis fitted on TRAIN can represent.

    The ceiling on any cross-validated Procrustes score. If held-out cells live in
    directions the training cells never visit, the projection discards them and test_proc
    approaches 1 no matter how good the transform is -- which looks identical to "no global
    transform explains this". Reporting coverage separates the two readings. Low coverage
    on hold-out-traits is itself a result: it says persona geometry is trait-specific.
    """
    Xc = X - X[train].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc[train], full_matrices=False)
    P = Vt[: min(rank, Vt.shape[0])].T
    tot = float((Xc[test] ** 2).sum())
    return float(((Xc[test] @ P) ** 2).sum() / max(tot, 1e-12))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cache = Path(args.cache_dir)

    print(f"loading arms at layer {args.layer} ...", flush=True)
    D = {a: load_arm(cache, a, args.cache_layers, args.layer) for a in args.arms}
    base = D["base"]
    traits, semantic = base["traits"], base["semantic"]
    pidx = [base["personas"].index(p) for p in semantic]
    arms_adapted = [a for a in args.arms if a != "base"]
    print(f"  {len(traits)} traits x {len(semantic)} personas x {len(args.arms)} arms\n")

    out: dict = {"layer": args.layer, "arms": args.arms, "traits": traits,
                 "personas": semantic, "bootstrap": args.bootstrap,
                 "uncertainty_note": "question bootstrap only; conditions on these 10 "
                                     "personas and 8 traits"}

    # =========================================================================== section 5
    print("=" * 84)
    print("5. PERSONA DISPERSION IN FULL HIDDEN SPACE")
    print("   D = mean_p ||V_p - centroid||^2 -- a mean SQUARED radius, like a variance.")
    print("   Ratios are therefore reported both ways: D-ratio, and its square root, which")
    print("   is the ratio of ordinary RMS distances and the one to quote as 'spread'.")
    print("   sqrt(D-ratio) should equal the RDM RMS ratio in section 6; they are the same")
    print("   quantity by two independent routes.")
    print("=" * 84)
    disp: dict = {}
    for ti, trait in enumerate(traits):
        nq = int(base["nq_per_trait"][ti])
        splits = half_splits(nq, args.half_splits, rng)
        bs = shared_boot_splits(rng, nq, args.bootstrap, args.boot_splits)  # paired
        per_arm = {}
        for arm in args.arms:
            pos, neg = cell_arrays(D[arm], ti, pidx)
            # laid out once per cell: the transpose-reshape inside is the dominant cost
            # and is pure waste if repeated per bootstrap replicate
            prep = prepare_diff(pos, neg)
            per_arm[arm] = {
                "crossfit": dispersion_crossfit(pos, neg, splits, prep=prep),
                "naive": dispersion_naive(trait_vectors(pos, neg)),
                "boot": np.array([dispersion_crossfit(pos, neg, b, prep=prep) for b in bs]),
            }
        disp[trait] = per_arm
        infl = per_arm["base"]["naive"] / max(per_arm["base"]["crossfit"], 1e-9)
        print(f"\n{trait:14s} base D={per_arm['base']['crossfit']:9.1f}   "
              f"(naive {per_arm['base']['naive']:9.1f}, inflated {infl:.2f}x)")
        for arm in arms_adapted:
            r = per_arm[arm]["crossfit"] / max(per_arm["base"]["crossfit"], 1e-9)
            rb = per_arm[arm]["boot"] / np.maximum(per_arm["base"]["boot"], 1e-9)
            lo, hi = ci(rb)
            flag = "" if lo <= 1.0 <= hi else "  *"
            rr = np.sqrt(max(r, 0.0))
            rlo, rhi = np.sqrt(max(lo, 0.0)), np.sqrt(max(hi, 0.0))
            print(f"    {arm:14s} D-ratio={r:5.3f} [{lo:5.3f},{hi:5.3f}]{flag}"
                  f"   RMS ratio={rr:5.3f} [{rlo:5.3f},{rhi:5.3f}]"
                  f"  ({1-rr:4.0%} linear contraction)")
    out["dispersion"] = {}
    for t, pa in disp.items():
        row = {}
        for a, v in pa.items():
            e = {"crossfit": float(v["crossfit"]), "naive": float(v["naive"]),
                 "boot_mean": float(v["boot"].mean())}
            if a != "base":
                rb = v["boot"] / np.maximum(pa["base"]["boot"], 1e-9)
                lo, hi = ci(rb)
                e["ratio"] = float(v["crossfit"] / max(pa["base"]["crossfit"], 1e-9))
                e["ratio_ci"] = [lo, hi]
            row[a] = e
        out["dispersion"][t] = row

    print("\n  aggregate (mean ratio across traits):")
    for arm in arms_adapted:
        rs = [disp[t][arm]["crossfit"] / max(disp[t]["base"]["crossfit"], 1e-9)
              for t in traits]
        rb = np.array([disp[t][arm]["boot"] / np.maximum(disp[t]["base"]["boot"], 1e-9)
                       for t in traits]).mean(0)
        lo, hi = ci(rb)
        mr = float(np.mean(rs)); mrms = float(np.sqrt(max(mr, 0.0)))
        print(f"    {arm:14s} D-ratio={mr:5.3f} [{lo:5.3f},{hi:5.3f}]"
              f"{'' if lo <= 1.0 <= hi else '  *'}"
              f"   RMS ratio={mrms:5.3f} [{np.sqrt(max(lo,0)):5.3f},"
              f"{np.sqrt(max(hi,0)):5.3f}]  ({1-mrms:4.0%} linear contraction)")
        out.setdefault("dispersion_aggregate", {})[arm] = {
            "mean_ratio": mr, "ci": [lo, hi],
            "mean_rms_ratio": mrms, "rms_ci": [float(np.sqrt(max(lo,0))),
                                               float(np.sqrt(max(hi,0)))]}

    # =========================================================================== section 6
    print("\n" + "=" * 84)
    print("6. RDM PRESERVATION (45 persona pairs per trait, cross-fitted sq. distances)")
    print("   spearman  = shape preservation (Pearson/Spearman are already scale-free)")
    print("   RMS ratio = absolute expansion (>1) or contraction (<1) of the constellation")
    print("   corrected = divided by the per-arm noise ceiling: an arm with a weaker signal")
    print("               looks less preserving for measurement reasons alone")
    print("=" * 84)
    rdms: dict = {}
    boot_rho: dict = {}
    for ti, trait in enumerate(traits):
        nq = int(base["nq_per_trait"][ti])
        splits = half_splits(nq, args.half_splits, rng)
        bs = shared_boot_splits(rng, nq, args.bootstrap, args.boot_splits)  # paired
        cur = {}
        preps = {}
        for arm in args.arms:
            pos, neg = cell_arrays(D[arm], ti, pidx)
            preps[arm] = prepare_diff(pos, neg)
            cur[arm] = {
                "rdm": rdm_crossfit(pos, neg, splits, prep=preps[arm]),
                "boot": [rdm_crossfit(pos, neg, b, prep=preps[arm]) for b in bs],
            }
        rdms[trait] = cur
        rel = {a: rdm_reliability(None, None, splits, prep=preps[a]) for a in args.arms}
        rms = {a: float(np.sqrt(np.mean(np.maximum(cur[a]["rdm"], 0)))) for a in args.arms}
        print(f"\n{trait:14s}  (noise ceiling: base rel={rel['base']:.3f})")
        print(f"    {'':14s}  spearman  [95% CI]          corrected  rel   RMS ratio")
        for arm in arms_adapted:
            pa = pearson(cur[arm]["rdm"], cur["base"]["rdm"])
            sa = spearman(cur[arm]["rdm"], cur["base"]["rdm"])
            corr = attenuation_corrected(sa, rel[arm], rel["base"])
            bp = np.array([spearman(x, y) for x, y in zip(cur[arm]["boot"], cur["base"]["boot"])])
            boot_rho.setdefault(arm, []).append(bp)     # (trait, replicate), for aggregate
            lo, hi = ci(bp)
            rr = rms[arm] / max(rms["base"], 1e-9)
            print(f"    {arm:14s}  {sa:6.3f}  [{lo:5.3f},{hi:5.3f}]     "
                  f"{corr:6.3f}   {rel[arm]:.3f}   {rr:5.3f}")
            out.setdefault("rdm", {}).setdefault(trait, {})[arm] = {
                "pearson": pa, "spearman": sa, "spearman_ci": [lo, hi],
                "spearman_corrected": corr, "reliability": rel[arm],
                "reliability_base": rel["base"], "rms_ratio": rr}
        out.setdefault("rdm_raw", {})[trait] = {
            a: cur[a]["rdm"].tolist() for a in args.arms}

    print("\n  mean across traits:")
    print(f"    {'':14s} spearman   corrected   RMS ratio")
    for arm in arms_adapted:
        v = [out["rdm"][t][arm]["spearman"] for t in traits]
        c = [out["rdm"][t][arm]["spearman_corrected"] for t in traits]
        r = [out["rdm"][t][arm]["rms_ratio"] for t in traits]
        print(f"    {arm:14s} {np.mean(v):6.3f}     {np.mean(c):6.3f}     {np.mean(r):6.3f}"
              f"   (per-trait spearman {min(v):.3f}..{max(v):.3f})")
        out.setdefault("rdm_aggregate", {})[arm] = {
            "mean_spearman": float(np.mean(v)),
            "mean_spearman_corrected": float(np.mean(c)),
            "mean_rms_ratio": float(np.mean(r))}

    # Aggregate CI, and the PAIRED differences between arms. The per-trait intervals do not
    # answer "does impulsiveness preserve RDM geometry less than goodness" -- that needs the
    # difference formed INSIDE each bootstrap replicate, where the shared question-sampling
    # noise cancels. Without it, an ordering of 0.822 / 0.858 / 0.905 is three point
    # estimates that happen to be ranked, not a resolved comparison.
    agg = {a: np.mean(np.stack(boot_rho[a], 0), axis=0) for a in arms_adapted}  # per replicate
    print("\n  aggregate mean Spearman across traits, with paired differences:")
    for arm in arms_adapted:
        lo, hi = ci(agg[arm])
        print(f"    {arm:14s} {agg[arm].mean():6.3f} [{lo:6.3f},{hi:6.3f}]")
        out["rdm_aggregate"][arm]["boot_mean"] = float(agg[arm].mean())
        out["rdm_aggregate"][arm]["boot_ci"] = [lo, hi]
    for i, a1 in enumerate(arms_adapted):
        for a2 in arms_adapted[i + 1:]:
            d_ = agg[a1] - agg[a2]
            lo, hi = ci(d_)
            sig = "resolved" if (lo > 0 or hi < 0) else "NOT resolved (CI spans 0)"
            print(f"    {a1} - {a2}: {d_.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  {sig}")
            out.setdefault("rdm_paired_diff", {})[f"{a1}-{a2}"] = {
                "mean": float(d_.mean()), "ci": [lo, hi],
                "resolved": bool(lo > 0 or hi < 0)}

    # ======================================================================= sections 8, 9
    print("\n" + "=" * 84)
    print("8. PROCRUSTES: can one global orthogonal map explain the arm difference?")
    print("   In-sample fit is NOT reported as evidence -- with 80 points in 4096 dims an")
    print("   orthogonal map has enough freedom to absorb almost anything. Only the")
    print("   cross-validated held-out error is interpretable.")
    print("=" * 84)
    # PERSONA-CENTRE WITHIN EACH TRAIT before any of this. The raw trait vectors are
    # dominated by a component shared across all personas of a trait, and that component is
    # nearly identical between arms -- so an uncentred Procrustes is mostly aligning a
    # vector to itself. It reports a small raw error, leaves the rotation almost nothing to
    # do, and the held-out score is then driven by whether the shared component happens to
    # sit in the training basis rather than by any persona geometry. Centring makes the
    # object of study the persona constellation, which is what section 9's residual
    # E = Z^arm R - Z^base is defined on.
    # Built UNCENTRED. Centring is fold-specific, inside the CV loop -- see fold_centre.
    V80 = {}
    for arm in args.arms:
        rows = []
        for ti in range(len(traits)):
            pos, neg = cell_arrays(D[arm], ti, pidx)
            rows.append(trait_vectors(pos, neg))                # (P, H), uncentred
        V80[arm] = np.concatenate(rows, 0)                      # (T*P, H)
    T, P = len(traits), len(semantic)
    labels = [(traits[i // P], semantic[i % P]) for i in range(T * P)]

    for arm in arms_adapted:
        print(f"\n{arm}")
        Xall = fold_centre(V80["base"], T, P, np.arange(T * P))
        Yall = fold_centre(V80[arm], T, P, np.arange(T * P))
        _, e_raw, e_proc = procrustes_lowrank(Yall - Yall.mean(0), Xall - Xall.mean(0))
        print(f"    in-sample (uninterpretable, shown for contrast): "
              f"E_raw={e_raw:.3f} -> E_proc={e_proc:.3f}")

        schemes = {
            "hold-out traits": [np.array([i for i, (t, _) in enumerate(labels) if t != h])
                                for h in traits],
            "hold-out personas": [np.array([i for i, (_, p) in enumerate(labels) if p != h])
                                  for h in semantic],
        }
        for name, trains in schemes.items():
            res, cov_x, cov_y = [], [], []
            for tr in trains:
                te = np.setdiff1d(np.arange(T * P), tr)
                Xc = fold_centre(V80["base"], T, P, tr)
                Yc = fold_centre(V80[arm], T, P, tr)
                res.append(procrustes_cv(Yc, Xc, tr, te, rank=args.procrustes_rank))
                cov_x.append(_basis_coverage(Xc, tr, te, args.procrustes_rank))
                cov_y.append(_basis_coverage(Yc, tr, te, args.procrustes_rank))
            raw = float(np.mean([r["test_raw"] for r in res]))
            prc = float(np.mean([r["test_proc"] for r in res]))
            scl = float(np.mean([r["test_proc_scaled"] for r in res]))
            cx, cy = float(np.mean(cov_x)), float(np.mean(cov_y))
            # E is a Frobenius NORM, so 1 - proc/raw is a relative norm reduction, not a
            # variance-explained figure. The squared version is the conventional
            # sum-of-squares analogue and is the larger of the two; both are reported so
            # neither can be quoted as the other.
            rel = 1 - prc / max(raw, 1e-9)
            sq = 1 - (prc ** 2) / max(raw ** 2, 1e-12)
            print(f"    CV {name:18s} test_raw={raw:.3f} -> test_proc={prc:.3f} "
                  f"(+scale {scl:.3f})")
            print(f"       {'':18s} rel. Frobenius reduction {rel:5.1%} | "
                  f"squared-error removed {sq:5.1%} | basis covers base {cx:.0%}, "
                  f"arm {cy:.0%}")
            out.setdefault("procrustes", {}).setdefault(arm, {})[name] = {
                "test_raw": raw, "test_proc": prc, "test_proc_scaled": scl,
                "rel_frobenius_reduction": rel, "squared_error_removed": sq,
                "basis_coverage_base": cx, "basis_coverage_arm": cy}

        # A general linear map is strictly more expressive than an orthogonal one -- it can
        # represent anisotropic scaling and shear, which rotation cannot. Run on the
        # hold-out-traits scheme only, since that is the scheme not compromised by the
        # centring dependency that persona folds have.
        lin = []
        for h in traits:
            tr = np.array([i for i, (t, _) in enumerate(labels) if t != h])
            te = np.setdiff1d(np.arange(T * P), tr)
            Xc, Yc = fold_centre(V80["base"], T, P, tr), fold_centre(V80[arm], T, P, tr)
            lin.append(linear_map_cv(Yc, Xc, tr, te, rank=args.procrustes_rank))
        l_raw = float(np.mean([r["test_raw"] for r in lin]))
        l_lin = float(np.mean([r["test_linear"] for r in lin]))
        l_rel = 1 - l_lin / max(l_raw, 1e-9)
        l_sq = 1 - (l_lin ** 2) / max(l_raw ** 2, 1e-12)
        print(f"    CV linear map (hold-out traits, ridge, lambda by inner CV): "
              f"test_raw={l_raw:.3f} -> test_linear={l_lin:.3f}")
        print(f"       {'':18s} rel. Frobenius reduction {l_rel:5.1%} | "
              f"squared-error removed {l_sq:5.1%}")
        out.setdefault("linear_map", {})[arm] = {
            "test_raw": l_raw, "test_linear": l_lin,
            "rel_frobenius_reduction": l_rel, "squared_error_removed": l_sq,
            "lambdas": [r["lambda"] for r in lin]}

        # rank sensitivity, on the trustworthy scheme only
        sweep = {}
        for rk in args.rank_sweep:
            rs, cs = [], []
            for h in traits:
                tr = np.array([i for i, (t, _) in enumerate(labels) if t != h])
                te = np.setdiff1d(np.arange(T * P), tr)
                Xc, Yc = fold_centre(V80["base"], T, P, tr), fold_centre(V80[arm], T, P, tr)
                rr = procrustes_cv(Yc, Xc, tr, te, rank=rk)
                rs.append(1 - rr["test_proc"] / max(rr["test_raw"], 1e-9))
                cs.append(_basis_coverage(Xc, tr, te, rk))
            sweep[rk] = (float(np.mean(rs)), float(np.mean(cs)))
        print("    rank sweep (hold-out traits): " +
              "  ".join(f"r={k}: {v[0]:.1%} (cov {v[1]:.0%})" for k, v in sweep.items()))
        out.setdefault("procrustes_rank_sweep", {})[arm] = {
            str(k): {"rel_frobenius_reduction": v[0], "basis_coverage_base": v[1]}
            for k, v in sweep.items()}

        # -------------------------------------------------------------- section 9
        # CROSS-VALIDATED residual map. Every cell's residual comes from a fold in which
        # its whole trait was held out. The earlier version fitted on all 80 cells and
        # scored the same 80, i.e. it was in-sample -- in the very section that warns
        # in-sample Procrustes absorbs almost anything at 80 points in 4096 dimensions.
        # Any localisation read off that map was reading the fit, not the data.
        norm_resid = np.zeros((T, P))
        for ti, h in enumerate(traits):
            tr = np.array([i for i, (t, _) in enumerate(labels) if t != h])
            te = np.setdiff1d(np.arange(T * P), tr)
            Xc, Yc = fold_centre(V80["base"], T, P, tr), fold_centre(V80[arm], T, P, tr)
            rr = procrustes_cv(Yc, Xc, tr, te, rank=args.procrustes_rank)
            r_ = np.array(rr["test_residual_norms"])
            g_ = np.array(rr["test_target_norms"])
            norm_resid[ti] = r_ / np.maximum(g_, 1e-9)
        out.setdefault("residual_map", {})[arm] = {
            "traits": traits, "personas": semantic, "values": norm_resid.tolist(),
            "cross_validated": True, "scheme": "hold-out traits"}
        by_t, by_p = norm_resid.mean(1), norm_resid.mean(0)
        print("    residual by trait  (CV): " +
              ", ".join(f"{t}={v:.3f}" for t, v in
                        sorted(zip(traits, by_t), key=lambda x: -x[1])[:3]) + " (top 3)")
        print("    residual by persona(CV): " +
              ", ".join(f"{q}={v:.3f}" for q, v in
                        sorted(zip(semantic, by_p), key=lambda x: -x[1])[:3]) + " (top 3)")

    dest = Path(args.out) if args.out else (
        OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis" / f"geometry_L{args.layer}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
