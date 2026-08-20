#!/usr/bin/env python3
"""How much of an adapter's effect on trait vectors is a single persona-common shift?

Three quantities, per constitution c and trait t, over the semantic personas p. Write
V_{c,t,p} for the CAA trait vector, V_base_{t,p} for the base model's, and

    dV_{c,t,p} = V_{c,t,p} - V_base_{t,p}        the per-persona change
    dG_{c,t}   = mean_p dV_{c,t,p}               the persona-COMMON part of that change

  (1) ||dG_{c,t}|| / mean_p||V_base_{t,p}||
      How large is the common shift, next to the trait representation it is displacing?

  (2) cos(dG_{c,t}, dG_{c',t}) for every pair of constitutions
      Do different constitutions add the SAME common shift, or merely shifts of similar
      size? Quantity (1) cannot tell these apart, and prose about a "common shift" tends
      to slide between them.

  (3) ||dG_{c,t}|| / mean_p||dV_{c,t,p}||
      Of the total change this adapter makes, how much is the common shift?

WHY (3) IS REPORTED TWICE. The norm ratio in (3) is what was asked for, but the quantity
that actually partitions is the squared one, because

    mean_p||dV_p||^2 = ||dG||^2 + mean_p||dV_p - dG||^2

is an exact decomposition into a common part and a persona-specific part, and the linear
ratio is its square root. sqrt overstates a small share badly -- a 25% share of the change
reads as 0.50 in linear units. Both columns are printed and labelled; do not quote one as
if it were the other.

NOISE, WHICH IS NOT OPTIONAL HERE. Every quantity above is quadratic in vectors estimated
from a few hundred questions, so the naive version is biased upward by the noise variance
(see the header of geometry_lib.py). Two consequences specific to this analysis:

  - (3) is biased DOWNWARD by noise. Independent noise inflates mean_p||dV_p||^2 by the
    full per-persona noise variance but inflates ||dG||^2 by only 1/P of it, because dG
    averages over P personas. So the common share looks smaller than it is.

  - (2) is biased UPWARD, and badly. dV_c and dV_c' both subtract the SAME estimate of
    V_base, so their errors share a -eps_base term; the naive inner product picks up
    +E||eps_base||^2 and two unrelated constitutions can look aligned.

Both are fixed by cross-fitting on disjoint halves of the questions: estimate each vector
twice from non-overlapping questions and take inner products ACROSS the halves, so the
noise terms are independent and cancel in expectation rather than accumulating. For (2)
that means pairing arm c's half A against arm c''s half B, which also breaks the shared
base-noise term. Naive values are printed alongside so the size of the correction is
visible rather than assumed.

Usage:
    python scripts/common_shift.py --layers 15 20 --bootstrap 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from persona_steering.config import OUTPUTS_DIR
from geometry_lib import prepare_diff, batch_vectors, half_splits, bootstrap_half_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--cache-layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--arms", nargs="+",
                   default=["goodness", "mathematical", "impulsiveness", "misalignment"])
    p.add_argument("--half-splits", type=int, default=40,
                   help="random disjoint half-splits averaged for each point estimate")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="question-bootstrap replicates for CIs (0 to skip)")
    p.add_argument("--boot-splits", type=int, default=8,
                   help="half-splits inside each bootstrap replicate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "analysis" / "common_shift.json"))
    return p.parse_args()


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
    nq = int(d["nq"][ti])
    a = d["acts"][ti][pidx][:, :, :nq, :]
    return a[:, 0].astype(np.float32), a[:, 1].astype(np.float32)


def make_preps(base_pos, base_neg, arm_cells: dict) -> dict:
    """Lay out the per-question contrasts ONCE per trait.

    prepare_diff forces a full transpose-and-reshape copy of the block. That copy, not the
    GEMM it feeds, was the entire cost of the earlier bootstrap (2.83 s against 0.008 s),
    so it must never sit inside the replicate loop -- 200 replicates x 5 arms would put it
    there 1000 times per trait.

    dV per question is (arm_pos - base_pos) - (arm_neg - base_neg), and prepare_diff forms
    pos - neg, so feeding it the arm-minus-base activations gives dV directly.
    """
    return {"base": prepare_diff(base_pos, base_neg),
            "arms": {a: prepare_diff(p - base_pos, n - base_neg)
                     for a, (p, n) in arm_cells.items()},
            "nq": base_pos.shape[1], "P": base_pos.shape[0]}


def stats_for_trait(preps: dict, splits, arms: list[str], naive: bool = True) -> dict:
    """Cross-fitted quantities for one trait, all arms at once.

    Every arm is evaluated on the SAME splits, so the cosines pair matched question halves
    and every arm-to-arm comparison is paired.
    """
    prep_base, prep_d, P = preps["base"], preps["arms"], preps["P"]
    full = [np.arange(preps["nq"])]

    acc = {a: {"g2": 0.0, "dv2": 0.0, "bnorm": 0.0} for a in arms}
    cos_acc = {(a, b): 0.0 for a in arms for b in arms}

    for A, B in splits:
        Vb = batch_vectors(prep_base, [A, B])                 # (2, P, H)
        dA = {a: batch_vectors(prep_d[a], [A])[0] for a in arms}
        dB = {a: batch_vectors(prep_d[a], [B])[0] for a in arms}
        gA = {a: dA[a].mean(0) for a in arms}
        gB = {a: dB[a].mean(0) for a in arms}
        for a in arms:
            acc[a]["g2"] += float(gA[a] @ gB[a])
            acc[a]["dv2"] += float((dA[a] * dB[a]).sum() / P)
            # mean of per-persona norms, as specified -- not the RMS of them
            acc[a]["bnorm"] += float(np.sqrt(np.maximum((Vb[0] * Vb[1]).sum(-1), 0)).mean())
        for a in arms:
            for b in arms:
                cos_acc[(a, b)] += 0.5 * (float(gA[a] @ gB[b]) + float(gB[a] @ gA[b]))

    n = len(splits)
    out = {"per_arm": {}, "cos": {}}
    for a in arms:
        g2, dv2, bn = acc[a]["g2"] / n, acc[a]["dv2"] / n, acc[a]["bnorm"] / n
        out["per_arm"][a] = {
            "g_norm": float(np.sqrt(max(g2, 0.0))),
            "base_norm": bn,
            "g_over_base": float(np.sqrt(max(g2, 0.0))) / max(bn, 1e-12),
            "share_squared": float(max(g2, 0.0) / max(dv2, 1e-12)),
            "share_linear": float(np.sqrt(max(g2, 0.0) / max(dv2, 1e-12))),
        }
    for a in arms:
        for b in arms:
            na = max(acc[a]["g2"] / n, 1e-12) * max(acc[b]["g2"] / n, 1e-12)
            out["cos"][f"{a}|{b}"] = float((cos_acc[(a, b)] / n) / np.sqrt(na))

    if naive:
        Vb = batch_vectors(prep_base, full)[0]
        bn = float(np.linalg.norm(Vb, axis=-1).mean())
        d = {a: batch_vectors(prep_d[a], full)[0] for a in arms}
        g = {a: d[a].mean(0) for a in arms}
        out["cos_naive"], out["naive"] = {}, {}
        for a in arms:
            g2n = float(g[a] @ g[a]); dv2n = float((d[a] ** 2).sum() / P)
            out["naive"][a] = {
                "g_over_base": float(np.sqrt(g2n)) / max(bn, 1e-12),
                "share_squared": g2n / max(dv2n, 1e-12)}
        for a in arms:
            for b in arms:
                out["cos_naive"][f"{a}|{b}"] = float(
                    (g[a] @ g[b]) / np.sqrt(max(g[a] @ g[a], 1e-12) * max(g[b] @ g[b], 1e-12)))
    return out


def fmt_table(title: str, note: str, rows, cols, get) -> str:
    w = max(14, max(len(c) for c in cols) + 2)
    lines = [title, note, "", f"{'trait':16s}" + "".join(f"{c:>{w}s}" for c in cols)]
    for r in rows:
        lines.append(f"{r:16s}" + "".join(f"{get(r, c):>{w}.3f}" for c in cols))
    lines.append(f"{'MEAN':16s}" + "".join(
        f"{np.mean([get(r, c) for r in rows]):>{w}.3f}" for c in cols))
    return "\n".join(lines)


def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    cache = Path(a.cache_dir)
    results = {}
    text: list[str] = []

    for layer in a.layers:
        base = load_arm(cache, "base", a.cache_layers, layer)
        arms_d = {arm: load_arm(cache, arm, a.cache_layers, layer) for arm in a.arms}
        traits = base["traits"]
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        P = len(pidx)
        per_trait = {}

        for ti, t in enumerate(traits):
            nq = int(base["nq"][ti])
            bp, bn = cell(base, ti, pidx)
            cells = {arm: cell(d, ti, pidx) for arm, d in arms_d.items()}
            preps = make_preps(bp, bn, cells)
            splits = half_splits(nq, a.half_splits, rng)
            per_trait[t] = stats_for_trait(preps, splits, a.arms)

            if a.bootstrap:
                # One draw per replicate, re-split several ways, SHARED across arms so the
                # cosines and the arm comparisons are paired.
                reps = {k: [] for k in ("g_over_base", "share_squared")}
                cos_reps = {f"{x}|{y}": [] for x in a.arms for y in a.arms}
                for _ in range(a.bootstrap):
                    bs = bootstrap_half_splits(nq, a.boot_splits, rng)
                    if not bs:
                        continue
                    r = stats_for_trait(preps, bs, a.arms, naive=False)
                    reps["g_over_base"].append([r["per_arm"][x]["g_over_base"] for x in a.arms])
                    reps["share_squared"].append([r["per_arm"][x]["share_squared"] for x in a.arms])
                    for k in cos_reps:
                        cos_reps[k].append(r["cos"][k])
                for k, v in reps.items():
                    arr = np.array(v)
                    per_trait[t][f"{k}_ci"] = {
                        x: [float(np.percentile(arr[:, i], 2.5)),
                            float(np.percentile(arr[:, i], 97.5))]
                        for i, x in enumerate(a.arms)}
                per_trait[t]["cos_ci"] = {
                    k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    for k, v in cos_reps.items()}
            print(f"  layer {layer} {t} done", flush=True)

        results[str(layer)] = per_trait

        # ---- displays ----
        text.append(f"\n{'='*94}\nLAYER {layer}   ({len(traits)} traits x {P} semantic "
                    f"personas, cross-fitted over {a.half_splits} half-splits)\n{'='*94}")
        text.append("\n" + fmt_table(
            "TABLE 1.  ||dG_{c,t}|| / mean_p||V_base_{t,p}||",
            "  the persona-common shift, as a fraction of the base trait vector it displaces",
            traits, a.arms, lambda r, c: results[str(layer)][r]["per_arm"][c]["g_over_base"]))
        text.append("\n" + fmt_table(
            "TABLE 2.  ||dG_{c,t}||^2 / mean_p||dV_{c,t,p}||^2",
            "  SHARE of this adapter's total change carried by the common shift "
            "(exact partition)",
            traits, a.arms,
            lambda r, c: results[str(layer)][r]["per_arm"][c]["share_squared"]))
        text.append("\n" + fmt_table(
            "TABLE 2b. ||dG_{c,t}|| / mean_p||dV_{c,t,p}||",
            "  the same thing in LINEAR units (the sqrt of table 2) -- not a share",
            traits, a.arms,
            lambda r, c: results[str(layer)][r]["per_arm"][c]["share_linear"]))
        text.append("\n" + fmt_table(
            "TABLE 1n. table 1 computed NAIVELY (no cross-fitting), for comparison",
            "  the gap is the noise the cross-fitted estimator removes",
            traits, a.arms, lambda r, c: results[str(layer)][r]["naive"][c]["g_over_base"]))

        text.append(f"\n\nTABLE 3.  cos(dG_c, dG_c') per trait  [cross-fitted; naive in "
                    f"brackets]\n  are the constitutions adding the SAME shift, or only "
                    f"shifts of similar size?")
        for t in traits:
            r = results[str(layer)][t]
            text.append(f"\n  {t}")
            text.append("    " + f"{'':14s}" + "".join(f"{x[:12]:>15s}" for x in a.arms))
            for x in a.arms:
                row = "".join(
                    f"{r['cos'][f'{x}|{y}']:>8.3f} [{r['cos_naive'][f'{x}|{y}']:.2f}]"
                    for y in a.arms)
                text.append(f"    {x[:14]:14s}" + row)
        text.append("\n  MEAN over traits")
        text.append("    " + f"{'':14s}" + "".join(f"{x[:12]:>15s}" for x in a.arms))
        for x in a.arms:
            row = ""
            for y in a.arms:
                m = np.mean([results[str(layer)][t]["cos"][f"{x}|{y}"] for t in traits])
                mn = np.mean([results[str(layer)][t]["cos_naive"][f"{x}|{y}"] for t in traits])
                row += f"{m:>8.3f} [{mn:.2f}]"
            text.append(f"    {x[:14]:14s}" + row)

    body = "\n".join(text)
    print(body)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    Path(str(out).replace(".json", ".txt")).write_text(body)
    print(f"\nwrote {out} and {str(out).replace('.json', '.txt')}")


if __name__ == "__main__":
    main()
