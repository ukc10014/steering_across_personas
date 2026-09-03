#!/usr/bin/env python3
"""Is the offset estimator's affine assumption good enough to carry figure 5?

The headline reads the intercept `a` of

    logodds_arm = a + k * logodds_base

as the compression-free preference shift. That is only the shift the reader thinks it is
if (i) the relationship really is close to affine over the range of the data, and (ii)
logodds_base = 0 is inside the data rather than an extrapolation off the end of it. An
intercept is a prediction AT ZERO; if the items all sit far from zero and the true curve
bends, `a` is an artefact of the bend and nothing else.

Three checks, all on the archived cells, no GPU:

  support    how much of the data actually lies near base indifference
  quadratic  refit with a logodds_base^2 term and see whether the intercept moves
  local      throw the model away: the polarity-balanced MEAN of arm log-odds over items
             with |logodds_base| < d. Under the affine model this estimates the same `a`,
             but it assumes nothing about shape, so agreement is the test.

The contrast (impulsivity + risk_taking, minus the other six) is recomputed under each
estimator. It also reports the contrast on impulsivity ALONE, which is the trait the 2026-07-17
prereg names; risk_taking was added later, from the geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caa_logits_analysis import (OUTPUTS, BASE_ARM, VARIANTS, IMPULSIVENESS_TARGETS,
                                 load_variant, stack_trait, fit_offset_slope)

DELTAS = (0.5, 1.0, 2.0)


def polarity_groups(a_pos, qsel):
    for grp in (a_pos[qsel], ~a_pos[qsel]):
        if grp.sum() >= 3:
            yield grp


def fit_quadratic(base_pq, arm_pq, a_pos, qsel):
    """Intercept of arm ~ 1 + base + base^2, polarity-split and averaged."""
    outs = []
    for grp in polarity_groups(a_pos, qsel):
        b = base_pq[:, qsel][:, grp].ravel()
        g = arm_pq[:, qsel][:, grp].ravel()
        if b.var() < 1e-12:
            continue
        X = np.column_stack([np.ones_like(b), b, b * b])
        beta, *_ = np.linalg.lstsq(X, g, rcond=None)
        outs.append(float(beta[0]))
    return float(np.mean(outs)) if outs else float("nan")


def local_mean(base_pq, arm_pq, a_pos, qsel, d):
    """Polarity-balanced mean of arm log-odds among items with |base| < d. No shape model."""
    outs, n = [], 0
    for grp in polarity_groups(a_pos, qsel):
        b = base_pq[:, qsel][:, grp].ravel()
        g = arm_pq[:, qsel][:, grp].ravel()
        m = np.abs(b) < d
        if m.sum() < 3:
            return float("nan"), 0
        outs.append(float(g[m].mean()))
        n += int(m.sum())
    return (float(np.mean(outs)) if outs else float("nan")), n


def resid_by_decile(base_pq, arm_pq, a_pos, qsel):
    """Mean linear-fit residual in each base decile: the shape of any bend, in log-odds."""
    a, k = fit_offset_slope(base_pq, arm_pq, a_pos, qsel)
    b_all, r_all = [], []
    for grp in polarity_groups(a_pos, qsel):
        b = base_pq[:, qsel][:, grp].ravel()
        g = arm_pq[:, qsel][:, grp].ravel()
        b_all.append(b)
        r_all.append(g - (a + k * b))
    b = np.concatenate(b_all)
    r = np.concatenate(r_all)
    edges = np.percentile(b, np.linspace(0, 100, 11))
    out = []
    for i in range(10):
        m = (b >= edges[i]) & (b <= edges[i + 1] if i == 9 else b < edges[i + 1])
        out.append(float(r[m].mean()) if m.sum() else float("nan"))
    return out, float(np.abs(r).mean())


def analyse(cells, n_boot, seed):
    arms = sorted(cells)
    traits = sorted(set(cells[BASE_ARM]))
    other = [a for a in arms if a != BASE_ARM]
    rng = np.random.default_rng(seed)

    est_names = ["linear", "quadratic"] + [f"local_d{d}" for d in DELTAS]
    point = {a: {t: {} for t in traits} for a in other}
    boot = {a: {t: {} for t in traits} for a in other}
    support = {t: {} for t in traits}
    shape = {a: {} for a in other}

    for trait in traits:
        personas0 = sorted(cells[BASE_ARM][trait])
        qid_ref = cells[BASE_ARM][trait][personas0[0]]["qid"]
        c0 = cells[BASE_ARM][trait][personas0[0]]
        a_pos = c0["a_is_positive"].astype(bool)
        nq = len(qid_ref)
        allq = np.ones(nq, dtype=bool)

        for a in other:
            if trait not in cells[a]:
                continue
            personas = [p for p in personas0 if p in cells[a][trait]]
            if not personas:
                continue
            base_pq = stack_trait(cells, BASE_ARM, trait, personas, qid_ref)
            arm_pq = stack_trait(cells, a, trait, personas, qid_ref)

            lin_a, k = fit_offset_slope(base_pq, arm_pq, a_pos, allq)
            point[a][trait]["linear"] = lin_a
            point[a][trait]["retention_k"] = k
            point[a][trait]["quadratic"] = fit_quadratic(base_pq, arm_pq, a_pos, allq)
            for d in DELTAS:
                v, n = local_mean(base_pq, arm_pq, a_pos, allq, d)
                point[a][trait][f"local_d{d}"] = v
                support[trait][f"n_within_{d}"] = n
            dec, mad = resid_by_decile(base_pq, arm_pq, a_pos, allq)
            shape[a][trait] = {"resid_by_base_decile": [round(x, 4) for x in dec],
                               "mean_abs_resid": round(mad, 4)}

            draws = {e: np.empty(n_boot) for e in est_names}
            for i in range(n_boot):
                idx = rng.integers(0, nq, nq)
                sel = np.zeros(nq, dtype=bool)
                sel[np.unique(idx)] = True     # question resample, personas held
                draws["linear"][i] = fit_offset_slope(base_pq, arm_pq, a_pos, sel)[0]
                draws["quadratic"][i] = fit_quadratic(base_pq, arm_pq, a_pos, sel)
                for d in DELTAS:
                    draws[f"local_d{d}"][i] = local_mean(base_pq, arm_pq, a_pos, sel, d)[0]
            boot[a][trait] = draws

        b_all = np.concatenate([stack_trait(cells, BASE_ARM, trait, personas0,
                                            qid_ref).ravel()])
        support[trait].update({
            "n_items": int(b_all.size),
            "base_sd": round(float(b_all.std()), 3),
            "base_mean_abs": round(float(np.abs(b_all).mean()), 3),
            **{f"frac_within_{d}": round(float((np.abs(b_all) < d).mean()), 4)
               for d in DELTAS}})

    # --- contrasts, under each estimator, for both target definitions -----------------
    defs = {"impulsivity+risk_taking": [t for t in IMPULSIVENESS_TARGETS if t in traits],
            "impulsivity_only": [t for t in ("impulsivity",) if t in traits]}
    contrasts = {}
    for dname, tgt in defs.items():
        rest = [t for t in traits if t not in tgt]
        if not tgt or not rest:
            continue
        contrasts[dname] = {"targets": tgt, "others": rest, "by_arm": {}}
        for a in other:
            if not all(point[a].get(t, {}).get("linear") is not None for t in traits):
                continue
            row = {}
            for e in est_names:
                pt = (np.mean([point[a][t][e] for t in tgt])
                      - np.mean([point[a][t][e] for t in rest]))
                bc = (np.mean([boot[a][t][e] for t in tgt], axis=0)
                      - np.mean([boot[a][t][e] for t in rest], axis=0))
                lo, hi = np.percentile(bc[np.isfinite(bc)], [2.5, 97.5])
                row[e] = {"point": float(pt), "ci_lo": float(lo), "ci_hi": float(hi)}
            contrasts[dname]["by_arm"][a] = row

    return {"traits": traits, "arms": other, "n_boot": n_boot,
            "estimators": est_names, "deltas": list(DELTAS),
            "support": support, "shape": shape, "contrasts": contrasts,
            "offsets": {a: {t: {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in point[a][t].items()} for t in point[a]}
                        for a in other}}


def render(res_by_variant):
    L = []
    for variant, res in res_by_variant.items():
        if res is None:
            continue
        L.append("=" * 92)
        L.append(f"=== {variant} prompt ===")
        L.append("=" * 92)
        s = res["support"]
        L.append("\nSupport near base indifference (base arm, all personas x questions)")
        L.append(f"  {'trait':16s}{'n items':>9s}{'base SD':>9s}{'mean|b|':>9s}"
                 + "".join(f"{'|b|<' + str(d):>10s}" for d in res["deltas"]))
        for t in res["traits"]:
            L.append(f"  {t:16s}{s[t]['n_items']:>9d}{s[t]['base_sd']:>9.2f}"
                     f"{s[t]['base_mean_abs']:>9.2f}"
                     + "".join(f"{s[t]['frac_within_' + str(d)]:>10.3f}"
                               for d in res["deltas"]))
        L.append("\nMean linear-fit residual by base decile (bend in the relationship)")
        for a in res["arms"]:
            for t in ("impulsivity", "honesty"):
                if t in res["shape"].get(a, {}):
                    d = res["shape"][a][t]
                    L.append(f"  {a:18s}{t:14s}"
                             + " ".join(f"{x:+.2f}" for x in d["resid_by_base_decile"])
                             + f"   mean|r| {d['mean_abs_resid']:.2f}")
        for dname, c in res["contrasts"].items():
            L.append(f"\nContrast [{dname}]  targets={','.join(c['targets'])}")
            L.append(f"  {'arm':18s}" + "".join(f"{e:>22s}" for e in res["estimators"]))
            for a, row in c["by_arm"].items():
                L.append(f"  {a:18s}" + "".join(
                    f"{row[e]['point']:>+9.2f} [{row[e]['ci_lo']:+.2f},{row[e]['ci_hi']:+.2f}]"
                    for e in res["estimators"]))
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUTPUTS / "analysis" / "caa_logits_robustness.json"))
    a = p.parse_args()
    res = {}
    for variant, subdir in VARIANTS.items():
        cells = load_variant(subdir)
        res[variant] = analyse(cells, a.n_boot, a.seed) if BASE_ARM in cells else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    txt = render(res)
    out.with_suffix(".txt").write_text(txt)
    print(txt)
    print(f"\nwrote {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
