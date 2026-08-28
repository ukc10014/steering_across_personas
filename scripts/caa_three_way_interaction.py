#!/usr/bin/env python3
"""The constitution x trait x persona interaction: is a cell more than its marginals?

WHY THIS EXISTS. Every decomposition in the geometry suite either averages over personas
or discards persona identity: the persona-common shift (common_shift.py) averages over p,
dispersion (geometry_analysis.py) centres it away, the RDM keeps only anonymous pairwise
distances, and Procrustes fits one global map. The prereg's question lives in what all of
them throw out -- whether a constitution acts differently on SPECIFIC persona x trait
cells. Whether

    dV_{goodness, honesty, con_artist}   differs from   dV_{goodness, honesty, therapist}

by more than the constitution's, the trait's and the persona's own average behaviour
predicts. That is a three-way interaction, and this is an ANOVA of it in which the thing
in every cell is a 4096-dimensional vector rather than a scalar.

THE DECOMPOSITION. With X_{c,t,p} = V_{c,t,p} - V_base_{t,p} the per-cell change, the
balanced-design partition into eight orthogonal terms is

    X = mu + C + T + P + CT + CP + TP + CTP

each obtained by marginal averages, e.g. the three-way term is the alternating sum

    CTP_{ctp} = X_{ctp} - X_{ct.} - X_{c.p} - X_{.tp}
                        + X_{c..} + X_{.t.} + X_{..p} - X_{...}

and the partition of squared magnitude, sum_terms ||term||^2 = ||X||^2, is exact. It is
asserted at runtime rather than assumed (see `check_partition`).

THREE THINGS THAT ARE EASY TO GET WRONG HERE, each of which changes the answer.

  (1) WHERE THE NOISE ACTUALLY GOES, WHICH IS NOT WHERE THE DEGREES OF FREEDOM SAY. In a
      balanced C x T x P design the three-way term spans (C-1)(T-1)(P-1)/(CTP) of the cell
      space -- for 4 x 8 x 10 that is 189/320 = 59% -- so INDEPENDENT per-cell noise would
      put 59% of its energy in the interaction and a naive ||CTP||^2 would be almost all
      noise. That is the reasoning this analysis was designed around, and it is wrong here.

      Measured at layer 15, of 14.98 units of noise energy the interaction holds 0.66 where
      the independent-noise prediction is 8.85, and the trait main effect holds 6.77 where
      the prediction is 0.33. The reason is that the CAA questions are SHARED across arms
      and personas within a trait: a question-set idiosyncrasy moves every persona and
      every arm for that trait together, so it is a t-indexed effect and lands in T, TP and
      CT rather than in CTP. Cross-fitting therefore changes the interaction share by
      ~0.001, not by the factor the degrees of freedom imply.

      The table prints the MEASURED per-term noise (naive - crossfit) next to the
      independent-noise prediction, because the gap between those two columns is the point,
      and quoting `df share` alone would reproduce the error this paragraph records.
      Cross-fitting is kept regardless: it is what establishes that the correction is
      small, and its cost is a few minutes.

      The estimator is cross-fitted: the questions are split into disjoint halves, each
      term is estimated independently in each half, and the reported quantity is the INNER
      PRODUCT across halves, whose expectation is the true squared magnitude because the
      two noise terms are independent and mean-zero. A cross-fitted term can come out
      NEGATIVE when the truth is near zero. That is correct for an unbiased estimator and
      is never clipped before averaging.

  (2) BASE SUBTRACTION IS ALGEBRAICALLY IRRELEVANT TO EVERY TERM INVOLVING c. The
      alternating sum annihilates any term that does not depend on all three indices, and
      V_base_{t,p} has no c index; the same holds for CT and CP. So

          CTP(V - V_base) = CTP(V)     exactly, and likewise for C, CT, CP.

      Two consequences. The shared-base-noise inflation that had to be corrected for the
      cross-arm cosines in common_shift.py (both arms subtract the same noisy V_base, so
      their errors share a -eps_base term) CANNOT touch the interaction: eps_base has no c
      index either, so the projection kills the noise along with the signal. And base is a
      REFERENCE, never a level of C -- including it as an arm would enter a row of exact
      zeros and drag every marginal. Only mu, T, P and TP read as "changes" at all; the
      rest are identical whether or not base is subtracted. `--check-invariance` verifies
      this numerically on the point estimate rather than trusting the algebra.

  (3) A NONZERO INTERACTION ON ITS OWN ESTABLISHES NOTHING. Section 7 of the results doc is
      a record of effects that were real and then died against an untrained-LoRA control.
      So the null band is not a follow-up: the same statistic is computed on the
      functional-dose-matched random arms in the same run, on the same question splits, and
      the trained band is only interesting to the extent it exceeds that. Because the
      interaction share depends on C through the degrees of freedom, the trained band
      (C=4) is ALSO reported over every 3-arm subset, so there is a comparison against the
      3-arm random band at matched df.

UNITS. Each trait is divided by s_t = mean_p ||V_base_{t,p}||, cross-fitted, so that a
high-norm trait cannot dominate the partition and so that magnitudes read on the same
scale as table 1 of common_shift.py: 1.0 is one base trait vector. s_t is computed ONCE
from the point-estimate splits and held FIXED thereafter, including inside the bootstrap,
so it is a constant rather than a random variable and the partition stays exact.
--no-scale-traits reports the unscaled version, which is the same partition in raw units.

WHAT THIS CANNOT ANSWER. The bootstrap resamples CAA QUESTIONS. With n=4 constitutions and
n=10 personas there is no population inference here: "these constitutions act on specific
cells beyond their marginals" is in scope, "constitutions in general do" is not. The
per-cell table at the end is 320 comparisons ranked after the fact, and is DESCRIPTIVE. It
is calibrated against the untrained band's own per-cell distribution, NOT against a global
null of exactly zero: a per-cell value is a squared magnitude estimated from ~500 questions,
so "greater than zero" is true of essentially every cell and separates nothing. The first
version of this script tested against zero and duly reported 319 of 320 cells "significant",
which is the failure this paragraph exists to prevent.

Reads outputs/_qcache/{arm}_L15_20.npz (scripts/build_question_cache.py). CPU-only, numpy,
no forward passes. ~6 min per layer at the defaults; the bootstrap dominates.

Usage:
    python scripts/caa_three_way_interaction.py --layers 15 20 --bootstrap 200
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from persona_steering.config import OUTPUTS_DIR
from geometry_lib import prepare_diff, batch_vectors, half_splits, bootstrap_half_splits

TERMS = ["mu", "C", "T", "P", "CT", "CP", "TP", "CTP"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--cache-layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--arms", nargs="+",
                   default=["goodness", "mathematical", "impulsiveness", "misalignment"],
                   help="the trained band: levels of the constitution factor")
    p.add_argument("--null-arms", nargs="+",
                   default=["random_iid_s16", "random_perm_s16", "random_spec_s19"],
                   help="untrained-LoRA arms at matched functional dose (sec 7)")
    p.add_argument("--half-splits", type=int, default=40,
                   help="random disjoint half-splits averaged for each point estimate")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="question-bootstrap replicates for CIs (0 to skip)")
    p.add_argument("--boot-splits", type=int, default=8,
                   help="half-splits inside each bootstrap replicate")
    p.add_argument("--no-scale-traits", action="store_true",
                   help="skip the per-trait 1/||V_base|| scaling; raw units")
    p.add_argument("--no-check-invariance", action="store_true",
                   help="skip the numeric check that C/CT/CP/CTP ignore base subtraction")
    p.add_argument("--top-cells", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "analysis" / "three_way_interaction.json"))
    return p.parse_args()


# ---------------------------------------------------------------------------------------
# loading -- same layout and conventions as common_shift.py
# ---------------------------------------------------------------------------------------

def load_arm(cache_dir: Path, arm: str, cache_layers, layer: int) -> dict:
    f = cache_dir / f"{arm}_L{'_'.join(str(l) for l in cache_layers)}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- run scripts/build_question_cache.py first")
    z = np.load(f, allow_pickle=False)
    li = list(z["layers"]).index(layer)
    return {"acts": z["acts"][:, :, :, :, li, :],
            "traits": [str(t) for t in z["traits"]],
            "personas": [str(p) for p in z["personas"]],
            "semantic": [str(p) for p in z["semantic_personas"]],
            "nq": z["n_questions_per_trait"]}


def cell(d: dict, ti: int, pidx: list[int]):
    """(pos, neg) for one trait, semantic personas only, padding-free."""
    nq = int(d["nq"][ti])
    a = d["acts"][ti][pidx][:, :, :nq, :]
    return a[:, 0].astype(np.float32), a[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------------------

def anova_components(X: np.ndarray) -> dict:
    """Eight orthogonal terms of a balanced three-way layout. X: (C, T, P, H).

    Returned with keepdims so each term carries its own shape; broadcast before taking
    inner products or the multiplicity is silently wrong.
    """
    m = X.mean((0, 1, 2), keepdims=True)
    Mc = X.mean((1, 2), keepdims=True)
    Mt = X.mean((0, 2), keepdims=True)
    Mp = X.mean((0, 1), keepdims=True)
    Mct = X.mean(2, keepdims=True)
    Mcp = X.mean(1, keepdims=True)
    Mtp = X.mean(0, keepdims=True)
    return {"mu": m,
            "C": Mc - m,
            "T": Mt - m,
            "P": Mp - m,
            "CT": Mct - Mc - Mt + m,
            "CP": Mcp - Mc - Mp + m,
            "TP": Mtp - Mt - Mp + m,
            "CTP": X - Mct - Mcp - Mtp + Mc + Mt + Mp - m}


def term_ss(compA: dict, compB: dict, shape: tuple) -> dict:
    """Cross-fitted squared magnitude of each term, summed over the full cell grid."""
    out = {}
    for k in TERMS:
        a = np.broadcast_to(compA[k], shape)
        b = np.broadcast_to(compB[k], shape)
        out[k] = float((a * b).sum())
    return out


def check_partition(X: np.ndarray, tol: float = 1e-3) -> float:
    """The eight terms must sum back to X. Returns the max relative deviation."""
    c = anova_components(X)
    recon = sum(np.broadcast_to(c[k], X.shape) for k in TERMS)
    scale = max(float(np.abs(X).max()), 1e-12)
    dev = float(np.abs(recon - X).max()) / scale
    if dev > tol:
        raise AssertionError(f"ANOVA partition does not reconstruct X (rel dev {dev:.2e})")
    return dev


def df_shares(C: int, T: int, P: int) -> dict:
    """Fraction of the cell space each term spans -- the share pure noise would produce."""
    d = {"mu": 1, "C": C - 1, "T": T - 1, "P": P - 1,
         "CT": (C - 1) * (T - 1), "CP": (C - 1) * (P - 1), "TP": (T - 1) * (P - 1),
         "CTP": (C - 1) * (T - 1) * (P - 1)}
    n = C * T * P
    return {k: v / n for k, v in d.items()}


# ---------------------------------------------------------------------------------------
# per-split assembly of the (C, T, P, H) tensor
# ---------------------------------------------------------------------------------------

def build_tensors(preps: dict, arms: list[str], splits_by_trait: dict, si: int,
                  traits: list[str], scale: np.ndarray):
    """The cell tensor for split `si`, both halves. -> XA, XB each (C, T, P, H)

    Trait t uses ITS OWN split, because question counts differ per trait (empathy ships
    499). That is harmless: cells from different traits are estimated from disjoint
    question sets already, so independence between the A and B side is what matters and it
    holds within each trait by construction.
    """
    XA, XB = [], []
    for ti, t in enumerate(traits):
        A, B = splits_by_trait[t][si]
        # Both halves in ONE batch_vectors call. The GEMM is trivial next to the read of
        # the (n_questions, P*hidden) block it multiplies, so asking for the two halves
        # separately reads that block twice and doubles the cost of the whole run.
        v = np.stack([batch_vectors(preps[t]["arms"][a], [A, B]) for a in arms])
        XA.append(v[:, 0] / scale[ti])
        XB.append(v[:, 1] / scale[ti])
    return (np.stack(XA, axis=1), np.stack(XB, axis=1))


def base_tensor(preps: dict, splits_by_trait: dict, si: int, traits: list[str],
                scale: np.ndarray):
    """V_base for split `si`, same layout minus the c axis. -> (T, P, H) per half."""
    bA, bB = [], []
    for ti, t in enumerate(traits):
        A, B = splits_by_trait[t][si]
        v = batch_vectors(preps[t]["base"], [A, B])
        bA.append(v[0] / scale[ti])
        bB.append(v[1] / scale[ti])
    return np.stack(bA), np.stack(bB)


def crossfit_bands(preps: dict, arms_all: list[str], bands: dict, splits_by_trait: dict,
                   n_splits: int, traits: list[str], scale: np.ndarray,
                   cell_bands: tuple = ()) -> dict:
    """Cross-fitted SS per term, for every band, over the same splits.

    All bands are slices of one (C_all, T, P, H) tensor, so every band-to-band comparison
    is paired on the same question halves and costs one extra projection, not one extra
    pass over the activations.
    """
    acc = {b: {k: 0.0 for k in TERMS} | {"total": 0.0} for b in bands}
    cells: dict = {}
    for si in range(n_splits):
        XA, XB = build_tensors(preps, arms_all, splits_by_trait, si, traits, scale)
        for b, idx in bands.items():
            sA, sB = XA[idx], XB[idx]
            cA, cB = anova_components(sA), anova_components(sB)
            for k, v in term_ss(cA, cB, sA.shape).items():
                acc[b][k] += v
            acc[b]["total"] += float((sA * sB).sum())
            if b in cell_bands:
                cw = (cA["CTP"] * cB["CTP"]).sum(-1)
                cells[b] = cw if b not in cells else cells[b] + cw
    for b in acc:
        for k in acc[b]:
            acc[b][k] /= n_splits
    return {"ss": acc, "cells": {b: v / n_splits for b, v in cells.items()}}


def naive_bands(preps: dict, arms_all: list[str], bands: dict, traits: list[str],
                scale: np.ndarray, nq_by_trait: dict) -> dict:
    """The same partition computed WITHOUT cross-fitting, on the full question set.

    This is what the analysis would report if the noise were ignored; the gap to the
    cross-fitted value is the correction, and the CTP row is where it is largest.
    """
    X = []
    for ti, t in enumerate(traits):
        full = [np.arange(nq_by_trait[t])]
        v = np.stack([batch_vectors(preps[t]["arms"][a], full)[0] for a in arms_all])
        X.append(v / scale[ti])
    X = np.stack(X, axis=1)
    out = {}
    for b, idx in bands.items():
        s = X[idx]
        c = anova_components(s)
        out[b] = term_ss(c, c, s.shape) | {"total": float((s * s).sum())}
    return out


# ---------------------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------------------

def partition_table(title: str, note: str, ss: dict, naive: dict, dfs: dict,
                    n_cells: int) -> str:
    tot = ss["total"]
    noise = naive["total"] - tot
    lines = [title, note, "",
             f"  {'term':6s}{'crossfit SS':>14s}{'share':>9s}{'RMS/cell':>11s}"
             f"{'naive share':>13s}{'noise':>9s}{'if indep':>10s}{'df share':>10s}"]
    for k in TERMS:
        share = ss[k] / tot if abs(tot) > 1e-12 else float("nan")
        rms = np.sqrt(max(ss[k], 0.0) / n_cells)
        nsh = naive[k] / naive["total"] if abs(naive["total"]) > 1e-12 else float("nan")
        lines.append(f"  {k:6s}{ss[k]:>14.4f}{share:>9.3f}{rms:>11.4f}"
                     f"{nsh:>13.3f}{naive[k] - ss[k]:>9.3f}{dfs[k] * noise:>10.3f}"
                     f"{dfs[k]:>10.3f}")
    lines.append(f"  {'TOTAL':6s}{tot:>14.4f}{1.0:>9.3f}"
                 f"{np.sqrt(max(tot, 0.0) / n_cells):>11.4f}{1.0:>13.3f}"
                 f"{noise:>9.3f}{noise:>10.3f}{1.0:>10.3f}")
    lines.append(f"\n  noise energy removed by cross-fitting: {noise:>.4f} "
                 f"({noise / max(naive['total'], 1e-12):.1%} of the naive total)")
    lines.append(
        "  `noise` is MEASURED (naive - crossfit); `if indep` is what that same total noise\n"
        "  would put in each term if it were independent per cell, which it is NOT -- the CAA\n"
        "  questions are shared across arms and personas within a trait, so sampling noise is\n"
        "  common along c and p and lands in the t-indexed terms. Compare the two columns\n"
        "  before quoting `df share` as the reference for anything.")
    return "\n".join(lines)


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


# ---------------------------------------------------------------------------------------

def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    cache = Path(a.cache_dir)
    arms_all = list(a.arms) + list(a.null_arms)
    results: dict = {"config": {"arms": a.arms, "null_arms": a.null_arms,
                                "scaled": not a.no_scale_traits,
                                "half_splits": a.half_splits,
                                "bootstrap": a.bootstrap, "seed": a.seed}}
    text: list[str] = []

    for layer in a.layers:
        t0 = time.time()
        base = load_arm(cache, "base", a.cache_layers, layer)
        armd = {arm: load_arm(cache, arm, a.cache_layers, layer) for arm in arms_all}
        traits = base["traits"]
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        personas = base["semantic"]
        P, T = len(pidx), len(traits)

        # ---- per-trait contrasts, laid out once (prepare_diff is the expensive step) ----
        preps, nq_by_trait, splits_by_trait = {}, {}, {}
        for ti, t in enumerate(traits):
            bp, bn = cell(base, ti, pidx)
            nq = bp.shape[1]
            nq_by_trait[t] = nq
            # dV per question = (arm_pos - base_pos) - (arm_neg - base_neg); prepare_diff
            # forms pos - neg, so the arm-minus-base activations give dV directly.
            arm_preps = {}
            for arm in arms_all:
                ap, an = cell(armd[arm], ti, pidx)
                arm_preps[arm] = prepare_diff(ap - bp, an - bn)
            preps[t] = {"base": prepare_diff(bp, bn), "arms": arm_preps}
            splits_by_trait[t] = half_splits(nq, a.half_splits, rng)
            print(f"  layer {layer} prepped {t} ({nq} questions)", flush=True)

        # ---- per-trait scale: cross-fitted mean_p ||V_base||, then held FIXED ----
        if a.no_scale_traits:
            scale = np.ones(T, dtype=np.float32)
        else:
            scale = np.zeros(T, dtype=np.float32)
            for ti, t in enumerate(traits):
                acc = 0.0
                for A, B in splits_by_trait[t]:
                    Vb = batch_vectors(preps[t]["base"], [A, B])
                    acc += float(np.sqrt(np.maximum((Vb[0] * Vb[1]).sum(-1), 0)).mean())
                scale[ti] = acc / len(splits_by_trait[t])

        # ---- bands: the trained band, the null band, and matched-df subsets ----
        bands = {"trained": list(range(len(a.arms))),
                 "null": list(range(len(a.arms), len(arms_all)))}
        n_null = len(a.null_arms)
        subsets = {}
        if len(a.arms) > n_null:
            for combo in itertools.combinations(range(len(a.arms)), n_null):
                subsets["trained[" + "+".join(a.arms[i][:4] for i in combo) + "]"] = list(combo)
        bands |= subsets

        cf = crossfit_bands(preps, arms_all, bands, splits_by_trait, a.half_splits,
                            traits, scale, cell_bands=("trained", "null"))
        nv = naive_bands(preps, arms_all, bands, traits, scale, nq_by_trait)

        # ---- checks: the partition is exact, and base subtraction is irrelevant ----
        XA, XB = build_tensors(preps, arms_all, splits_by_trait, 0, traits, scale)
        checks = {"partition_rel_dev": check_partition(XA[bands["trained"]])}
        if not a.no_check_invariance:
            bA, bB = base_tensor(preps, splits_by_trait, 0, traits, scale)
            worst = 0.0
            for idx in (bands["trained"], bands["null"]):
                dA, dB = XA[idx], XB[idx]
                rA, rB = dA + bA[None], dB + bB[None]          # raw V, not the change
                sd = term_ss(anova_components(dA), anova_components(dB), dA.shape)
                sr = term_ss(anova_components(rA), anova_components(rB), rA.shape)
                for k in ("C", "CT", "CP", "CTP"):
                    worst = max(worst, abs(sd[k] - sr[k]) / max(abs(sd[k]), 1e-9))
            checks["base_invariance_rel_dev"] = worst
            if worst > 1e-4:
                raise AssertionError(
                    f"C/CT/CP/CTP changed by {worst:.2e} under base subtraction; they are "
                    "algebraically invariant, so this means the layout is wrong")

        # ---- bootstrap over questions, one draw per replicate, shared across arms ----
        boot: dict = {}
        cell_reps: list = []
        exceed_reps: list = []
        if a.bootstrap:
            keys = list(bands)
            reps = {b: {"ctp_share": [], "ctp_rms": []} for b in keys}
            for r in range(a.bootstrap):
                bsplits, ok = {}, True
                for t in traits:
                    bs = bootstrap_half_splits(nq_by_trait[t], a.boot_splits, rng)
                    if not bs:
                        ok = False
                        break
                    bsplits[t] = bs
                if not ok:
                    continue
                n_bs = min(len(bsplits[t]) for t in traits)
                rr = crossfit_bands(preps, arms_all, bands, bsplits, n_bs, traits, scale,
                                    cell_bands=("trained", "null"))
                for b in keys:
                    ss = rr["ss"][b]
                    n_cells = len(bands[b]) * T * P
                    reps[b]["ctp_share"].append(
                        ss["CTP"] / ss["total"] if abs(ss["total"]) > 1e-12 else np.nan)
                    reps[b]["ctp_rms"].append(np.sqrt(max(ss["CTP"], 0.0) / n_cells))
                cell_reps.append(rr["cells"]["trained"])
                # Calibrated against the untrained band's OWN per-cell distribution, drawn
                # from the SAME replicate, rather than against a global null of exactly
                # zero -- see the display note for why the latter is near-vacuous.
                p95 = np.percentile(rr["cells"]["null"], 95)
                exceed_reps.append(int((rr["cells"]["trained"] > p95).sum()))
                if (r + 1) % 25 == 0:
                    print(f"  layer {layer} bootstrap {r + 1}/{a.bootstrap}", flush=True)
            for b in keys:
                boot[b] = {k: [float(np.nanpercentile(v, 2.5)),
                               float(np.nanpercentile(v, 97.5))]
                           for k, v in reps[b].items()}

        # ---- assemble ----
        n_cells = {b: len(idx) * T * P for b, idx in bands.items()}
        results[str(layer)] = {
            "traits": traits, "personas": personas, "trait_scale": scale.tolist(),
            "checks": checks,
            "bands": {b: {
                "arms": [arms_all[i] for i in idx],
                "n_cells": n_cells[b],
                "crossfit": cf["ss"][b],
                "naive": nv[b],
                "df_share": df_shares(len(idx), T, P),
                "ctp_share": (cf["ss"][b]["CTP"] / cf["ss"][b]["total"]
                              if abs(cf["ss"][b]["total"]) > 1e-12 else None),
                "ctp_rms": float(np.sqrt(max(cf["ss"][b]["CTP"], 0.0) / n_cells[b])),
                "ci": boot.get(b, {}),
            } for b, idx in bands.items()},
        }

        # ---- per-cell interaction, against the untrained band's own distribution ----
        cellwise = cf["cells"]["trained"]
        null_cells = np.sort(cf["cells"]["null"].ravel())
        flat = [(float(cellwise[ci, ti, pi]), a.arms[ci], traits[ti], personas[pi],
                 ci, ti, pi)
                for ci in range(len(a.arms)) for ti in range(T) for pi in range(P)]
        flat.sort(key=lambda r: -r[0])
        cell_ci = {}
        if cell_reps:
            arr = np.stack(cell_reps)                      # (reps, C, T, P)
            lo = np.percentile(arr, 2.5, axis=0)
            hi = np.percentile(arr, 97.5, axis=0)
            for _, arm, t, p, ci, ti, pi in flat:
                cell_ci[f"{arm}|{t}|{p}"] = [float(lo[ci, ti, pi]), float(hi[ci, ti, pi])]
        null_p95 = float(np.percentile(null_cells, 95))
        n_exceed = int((cellwise > null_p95).sum())
        exceed_ci = ([float(np.percentile(exceed_reps, 2.5)),
                      float(np.percentile(exceed_reps, 97.5))] if exceed_reps else None)
        results[str(layer)]["cells"] = {
            f"{arm}|{t}|{p}": {"ss": v, "ci": cell_ci.get(f"{arm}|{t}|{p}"),
                               "pct_of_null": float(np.searchsorted(null_cells, v)
                                                    / len(null_cells))}
            for v, arm, t, p, _, _, _ in flat}
        results[str(layer)]["cell_reference"] = {
            "null_p95": null_p95, "n_exceed": n_exceed, "n_cells": len(flat),
            "n_exceed_if_matched": 0.05 * len(flat), "n_exceed_ci": exceed_ci,
            "quantiles": {b: {f"p{qq}": float(np.percentile(cf["cells"][b], qq))
                              for qq in (10, 25, 50, 75, 90, 95)}
                          for b in ("trained", "null")}}

        # ---- text ----
        text.append(f"\n{'=' * 96}\nLAYER {layer}   {T} traits x {P} semantic personas, "
                    f"cross-fitted over {a.half_splits} half-splits\n"
                    f"  units: 1.0 = one base trait vector"
                    f"{'  (UNSCALED, raw units)' if a.no_scale_traits else ''}\n"
                    f"  checks: partition rel dev {checks['partition_rel_dev']:.2e}"
                    + (f", base-invariance rel dev "
                       f"{checks['base_invariance_rel_dev']:.2e}"
                       if "base_invariance_rel_dev" in checks else "")
                    + f"\n{'=' * 96}")
        for b in ("trained", "null"):
            idx = bands[b]
            text.append("\n" + partition_table(
                f"\nBAND {b.upper()}  ({len(idx)} arms: {', '.join(arms_all[i] for i in idx)})",
                "  share = fraction of the cross-fitted total; df share = what pure noise "
                "would give",
                cf["ss"][b], nv[b], df_shares(len(idx), T, P), n_cells[b]))
            if b in boot:
                text.append(f"\n  CTP share {cf['ss'][b]['CTP'] / cf['ss'][b]['total']:+.3f} "
                            f"{fmt_ci(boot[b]['ctp_share'])}   "
                            f"CTP RMS/cell {results[str(layer)]['bands'][b]['ctp_rms']:.4f} "
                            f"{fmt_ci(boot[b]['ctp_rms'])}")

        text.append("\n\nTHE COMPARISON THAT MATTERS: trained vs untrained at MATCHED df")
        text.append("  the interaction share depends on C through the degrees of freedom, so "
                    "the\n  4-arm trained band is not directly comparable to the 3-arm null "
                    "band.\n")
        text.append(f"  {'band':34s}{'CTP share':>12s}{'95% CI':>22s}"
                    f"{'CTP RMS/cell':>14s}{'95% CI':>22s}")
        for b in list(subsets) + ["null"]:
            r = results[str(layer)]["bands"][b]
            ci1 = fmt_ci(boot[b]["ctp_share"]) if b in boot else ""
            ci2 = fmt_ci(boot[b]["ctp_rms"]) if b in boot else ""
            text.append(f"  {b:34s}{r['ctp_share']:>+12.3f}{ci1:>22s}"
                        f"{r['ctp_rms']:>14.4f}{ci2:>22s}")

        ref = results[str(layer)]["cell_reference"]
        text.append(f"\n\nPER-CELL INTERACTION, trained band, top {a.top_cells} of "
                    f"{len(flat)} -- DESCRIPTIVE, and read the reference first")
        text.append(
            "  A per-cell value is a squared magnitude estimated from ~500 questions, so\n"
            "  'is it greater than zero' is answered YES for essentially every cell and\n"
            "  means nothing. The reference that does mean something is the UNTRAINED band's\n"
            "  own per-cell distribution, same statistic, same question splits: does a\n"
            "  trained constitution single out particular cells MORE than a random\n"
            "  perturbation of matched functional dose does?")
        qs = ("p10", "p25", "p50", "p75", "p90", "p95")
        text.append(f"\n  {'band':10s}" + "".join(f"{k:>10s}" for k in qs))
        for b in ("trained", "null"):
            text.append(f"  {b:10s}" + "".join(f"{ref['quantiles'][b][k]:>10.4f}"
                                               for k in qs))
        text.append(f"\n  trained cells above the untrained band's p95 ({null_p95:.4f}): "
                    f"{n_exceed} of {len(flat)}"
                    + (f"  {fmt_ci(exceed_ci)}" if exceed_ci else "")
                    + f"\n  if the two distributions matched, that count would be "
                      f"{ref['n_exceed_if_matched']:.0f}.")
        text.append("\n  `pct` = fraction of untrained-band cells this cell exceeds.\n")
        text.append(f"  {'constitution':16s}{'trait':14s}{'persona':22s}"
                    f"{'crossfit SS':>13s}{'pct':>7s}{'95% CI':>24s}")
        for v, arm, t, p, _, _, _ in flat[:a.top_cells]:
            ci = cell_ci.get(f"{arm}|{t}|{p}")
            pct = float(np.searchsorted(null_cells, v) / len(null_cells))
            text.append(f"  {arm:16s}{t:14s}{p:22s}{v:>13.4f}{pct:>7.2f}"
                        f"{(fmt_ci(ci) if ci else ''):>24s}")

        print(f"  layer {layer} done in {time.time() - t0:.0f}s", flush=True)

    body = "\n".join(text)
    print(body)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    Path(str(out).replace(".json", ".txt")).write_text(body)
    print(f"\nwrote {out} and {str(out).replace('.json', '.txt')}")


if __name__ == "__main__":
    main()
