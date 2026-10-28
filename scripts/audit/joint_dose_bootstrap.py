#!/usr/bin/env python3
"""Paired matched-dose excess: trajectory vs scaled-DPO ladder, ONE question bootstrap for both.

control_dose.py reports `excess = trajectory value - ladder value interpolated at the state's
measured dose` with no interval, because trajectory and ladder came from separate bootstrap
runs (audit 2026-10-21, issue 6e). Pairing those runs by replicate index would be wrong: their
replicates resample different questions. Here every replicate draws questions ONCE per trait
and that draw is used for every arm -- base, every trajectory state, every ladder rung -- and
for every statistic of the replicate:

  dose   functional_dose.py's trait-vector displacement, mean over (trait, persona) of
         ||V_arm,p - V_base,p|| / ||V_base,p||, on the drawn questions;
  g, S   the cross-fitted common magnitude and squared share (common_shift formulas via
         geometry.raw_stats), split into disjoint halves of the SAME draw
         (geometry_lib.bootstrap_half_splits(idx=...));
  offsets the forced-choice offset (caa_logits_analysis.fit_offset_slope) on the same question
         ids (activation index i is CAA question id i for every trait, verified at load).

Dose matching is redone inside each replicate: the ladder is sorted by that replicate's doses
and linearly interpolated at that replicate's trajectory dose. So the interval carries dose
uncertainty, interpolation uncertainty and the value uncertainty of both routes.

RULES (fixed before running; the estimability convention of common_trajectory/estimability.py):
  - No extrapolation. A replicate whose trajectory dose falls outside its own ladder range is
    UNBRACKETED. A replicate where a ratio's denominator energy is <= 0 for any arm used is
    INVALID. Both are counted and reported, never dropped silently.
  - An excess is reported (point + 95 % percentile interval over the defined replicates) iff
    its point estimate is defined AND >= 97.5 % of replicates are defined. Otherwise point and
    interval are withheld and the fractions are reported, with the reason.
  - S has two estimands, kept apart: MEAN8/TARGETS per-trait mean of S (the original) and
    the energy-pooled S (sum of bn-normalised common energy / sum of total energy).
  - Behaviour excess uses the L15 dose (as control_dose.py), over the ten semantic personas
    (primary, as the geometry) and the historical eleven (incl. null), labelled separately.

Uncertainty is evaluation-only (questions); one training seed per arm.

    python3 scripts/audit/joint_dose_bootstrap.py --constitution loving --base-arm base_cmask
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts", REPO / "scripts/common_trajectory"):
    sys.path.insert(0, str(p))
from common_shift import cell, make_preps, load_arms_checked          # noqa: E402
from geometry_lib import batch_vectors, half_splits, bootstrap_half_splits  # noqa: E402
from geometry import raw_stats                                        # noqa: E402
from estimability import VALID_MIN                                    # noqa: E402
from states import CONTRASTS, TARGETS, trajectory                     # noqa: E402
from control_dose import ladder_arms                                  # noqa: E402
from caa_logits_analysis import fit_offset_slope, stack_trait         # noqa: E402
from behaviour import load as load_logits                             # noqa: E402

OUT = REPO / "outputs/analysis/audit/joint_dose"


def trait_draws(nq: int, B: int, seed: int, ti: int):
    """Per-trait question draws, identical for every arm, layer and statistic."""
    rng = np.random.default_rng([seed, ti])
    return [rng.integers(0, nq, nq) for _ in range(B)], np.random.default_rng([seed, ti, 1])


def dose_cells(prep_base, prep_arms: dict, idx_list) -> dict:
    """{arm: (n_subsets, P)} ||dV_p|| / ||V_base,p|| on each question subset."""
    vb = np.linalg.norm(batch_vectors(prep_base, idx_list), axis=-1)
    return {a: np.linalg.norm(batch_vectors(p, idx_list), axis=-1) / vb for a, p in prep_arms.items()}


def excess(xc, yc, xl, yl):
    """Trajectory minus ladder at matched dose; NaN when unbracketed. xl/yl: ladder (n_lad,)."""
    if not (np.all(np.isfinite(xl)) and np.all(np.isfinite(yl)) and np.isfinite(xc) and np.isfinite(yc)):
        return np.nan, "invalid"
    o = np.argsort(xl)
    xs, ys = xl[o], yl[o]
    if not (xs[0] - 1e-12 <= xc <= xs[-1] + 1e-12):
        return np.nan, "unbracketed"
    return float(yc - np.interp(xc, xs, ys)), "ok"


def summarise_excess(point, boot):
    """point: (value, status); boot: list of (value, status)."""
    st = [s for _, s in boot]
    n = len(st)
    fr = {k: st.count(k) / n for k in ("ok", "unbracketed", "invalid")}
    e = {"defined_frac": fr["ok"], "unbracketed_frac": fr["unbracketed"], "invalid_frac": fr["invalid"],
         "point_status": point[1], "n_boot": n}
    vals = np.array([v for v, s in boot if s == "ok"])
    if point[1] == "ok" and fr["ok"] >= VALID_MIN:
        e.update(point=point[0], lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                 estimable=True)
    else:
        e.update(point=point[0] if point[1] == "ok" else None, lo=None, hi=None, estimable=False,
                 reason=("point estimate " + point[1]) if point[1] != "ok"
                 else f"only {fr['ok']:.1%} of replicates defined (< {VALID_MIN:.1%})")
        if point[1] == "ok":
            e["point_note"] = "point shown for reference only; interval withheld by the rule"
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--base-arm", default="base_cmask")
    ap.add_argument("--cache-dir", default=str(REPO / "outputs/_qcache"))
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--boot-splits", type=int, default=8)
    ap.add_argument("--half-splits", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-mixed-mask", action="store_true", help="reproduction checks only")
    ap.add_argument("--traits", nargs="+", default=None, help="subset, for benchmarking")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    c, B = a.constitution, a.bootstrap
    t_start = time.time()
    traj = trajectory(c); carms = [s["arm"] for s in traj]
    larms = ladder_arms(c)
    assert larms[0] == carms[0], "ladder must start at the trajectory's DPO-only state"
    arms = carms + [x for x in larms if x not in carms]
    final = carms[-1]
    cn = [n for n in CONTRASTS.get(c, {}) if n != "B1"]
    cname = cn[-1] if cn else None
    res = {"constitution": c, "base_arm": a.base_arm, "trajectory_arms": carms, "ladder_arms": larms,
           "epoch": [s["epoch"] for s in traj], "step": [s["step"] for s in traj],
           "bootstrap": B, "boot_splits": a.boot_splits, "half_splits": a.half_splits, "seed": a.seed,
           "valid_min": VALID_MIN, "rules": __doc__.split("RULES")[1].split("Uncertainty")[0].strip(),
           "layers": {}}
    draws = {}
    dose_rep = {}           # layer -> (B+1, arms) dose; row 0 = point (all questions)

    for L in a.layers:
        loaded, prov = load_arms_checked(Path(a.cache_dir), [a.base_arm, *arms], [15, 20], L,
                                         allow_mixed_mask=a.allow_mixed_mask)
        res["provenance"] = prov
        base = loaded.pop(a.base_arm); D = loaded
        traits = a.traits or base["traits"]
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        assert len(pidx) == 10
        tg = TARGETS.get(c) or traits
        n_a = len(arms)
        # per trait: (B+1, arms) raw g2, dv2, bn and dose; row 0 = point
        g2 = np.full((len(traits), B + 1, n_a), np.nan); dv2 = np.full_like(g2, np.nan)
        bn = np.full((len(traits), B + 1), np.nan); dose = np.full_like(g2, np.nan)
        for ti, t in enumerate(traits):
            t0 = time.time()
            tix = base["traits"].index(t)
            nq = int(base["nq"][tix])
            assert all(int(D[x]["nq"][tix]) == nq for x in arms)
            if t not in draws:
                draws[t] = trait_draws(nq, B, a.seed, tix)
            idxs, _ = draws[t]
            split_rng = np.random.default_rng([a.seed, tix, 2, L])
            bp, bneg = cell(base, tix, pidx)
            ptot = make_preps(bp, bneg, {x: cell(D[x], tix, pidx) for x in arms})
            del bp, bneg
            allq = np.arange(nq)
            # point
            tot, bnp = raw_stats(ptot["base"], ptot["arms"], half_splits(nq, a.half_splits, split_rng), final)
            g2[ti, 0] = [tot[x][0] for x in arms]; dv2[ti, 0] = [tot[x][1] for x in arms]; bn[ti, 0] = bnp
            dc = dose_cells(ptot["base"], ptot["arms"], [allq])
            dose[ti, 0] = [dc[x][0].mean() for x in arms]
            # replicates: same draw for dose and for the cross-fitted halves
            dc = dose_cells(ptot["base"], ptot["arms"], idxs)
            for x_i, x in enumerate(arms):
                dose[ti, 1:, x_i] = dc[x].mean(1)
            for r, idx in enumerate(idxs):
                bs = bootstrap_half_splits(nq, a.boot_splits, split_rng, idx=idx)
                if not bs:
                    continue
                tb, bnb = raw_stats(ptot["base"], ptot["arms"], bs, final)
                g2[ti, r + 1] = [tb[x][0] for x in arms]; dv2[ti, r + 1] = [tb[x][1] for x in arms]
                bn[ti, r + 1] = bnb
            del ptot
            print(f"L{L} {t}: {time.time() - t0:.0f}s", flush=True)

        # per-replicate estimands, (B+1, arms)
        ti_tg = [traits.index(t) for t in tg if t in traits]
        g = np.sqrt(np.clip(g2, 0, None)) / bn[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(dv2 > 0, g2 / dv2, np.nan)
            En = g2 / bn[..., None] ** 2; Et = dv2 / bn[..., None] ** 2

            def pooledS(ix):
                num, den = En[ix].sum(0), Et[ix].sum(0)
                return np.where(den > 0, num / den, np.nan)
        est = {"g_MEAN8": g.mean(0), "g_TARGETS": g[ti_tg].mean(0),
               "S_MEAN8": S.mean(0), "S_TARGETS": S[ti_tg].mean(0),        # NaN if any trait invalid
               "Spooled_MEAN8": pooledS(list(range(len(traits)))), "Spooled_TARGETS": pooledS(ti_tg)}
        dz = dose.mean(0)                                                   # (B+1, arms)
        dose_rep[L] = dz
        ci = [arms.index(x) for x in carms]; li = [arms.index(x) for x in larms]
        ser = {}
        for k, y in est.items():
            per_state = []
            for j in ci:
                pt = excess(dz[0, j], y[0, j], dz[0, li], y[0, li])
                bt = [excess(dz[r, j], y[r, j], dz[r, li], y[r, li]) for r in range(1, B + 1)]
                e = summarise_excess(pt, bt)
                e["trajectory_value"] = float(y[0, j]) if np.isfinite(y[0, j]) else None
                per_state.append(e)
            ser[k] = per_state
        res["layers"][str(L)] = {
            "traits": traits, "targets": tg,
            "dose_point": {x: float(dz[0, arms.index(x)]) for x in arms},
            "dose_ci": {x: [float(np.percentile(dz[1:, arms.index(x)], 2.5)),
                            float(np.percentile(dz[1:, arms.index(x)], 97.5))] for x in arms},
            "ladder_dose_range_point": [float(dz[0, li].min()), float(dz[0, li].max())],
            "series": ser}
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"joint_raw_{c}_L{L}{a.tag}.npz", g2=g2, dv2=dv2, bn=bn, dose=dose,
                            arms=np.array(arms), traits=np.array(traits))
        del loaded, D, base

    # behaviour: offsets on the same per-trait draws, matched at the L15 dose
    if 15 in dose_rep and a.traits is None:
        cells = load_logits(["base"] + arms)
        traits = sorted(cells["base"])
        pers11 = sorted(cells["base"][traits[0]])
        sets = {"10personas": [p for p in pers11 if p != "null"], "11personas_historical": pers11}
        dz = dose_rep[15]
        ci = [arms.index(x) for x in carms]; li = [arms.index(x) for x in larms]
        for sname, pers in sets.items():
            off = {}
            for t in traits:
                qid = cells["base"][t][pers[0]]["qid"]; apos = cells["base"][t][pers[0]]["a_is_positive"]
                assert np.array_equal(qid, np.arange(len(qid))), f"{t}: logit qids are not 0..n-1"
                idxs, _ = draws[t]
                assert len(qid) == len(idxs[0]), f"{t}: logit and activation question counts differ"
                bpq = stack_trait(cells, "base", t, pers, qid)
                o = np.full((B + 1, len(arms)), np.nan)
                for x_i, x in enumerate(arms):
                    apq = stack_trait(cells, x, t, pers, qid)
                    o[0, x_i] = fit_offset_slope(bpq, apq, apos, np.arange(len(qid)))[0]
                    for r, idx in enumerate(idxs):
                        o[r + 1, x_i] = fit_offset_slope(bpq, apq, apos, idx)[0]
                off[t] = o
            ys = {f"offset_{t}": off[t] for t in traits}
            if cname:
                ctg = CONTRASTS[c][cname]; rest = [t for t in traits if t not in ctg]
                ys[f"offset_{cname}"] = np.mean([off[t] for t in ctg], 0) - np.mean([off[t] for t in rest], 0)
            ser = {}
            for k, y in ys.items():
                per_state = []
                for j in ci:
                    pt = excess(dz[0, j], y[0, j], dz[0, li], y[0, li])
                    bt = [excess(dz[r, j], y[r, j], dz[r, li], y[r, li]) for r in range(1, B + 1)]
                    e = summarise_excess(pt, bt); e["trajectory_value"] = float(y[0, j])
                    per_state.append(e)
                ser[k] = per_state
            res["layers"]["15"][f"behaviour_{sname}"] = ser

    res["runtime_s"] = time.time() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"joint_dose_{c}{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, f"({res['runtime_s']:.0f}s)")
    for L, Lv in res["layers"].items():
        for k in ("S_MEAN8", "Spooled_MEAN8", "g_MEAN8"):
            e = Lv["series"][k][-1]
            print(f"L{L} {k:15s} endpoint excess {e.get('point')} [{e.get('lo')}, {e.get('hi')}] "
                  f"defined {e['defined_frac']:.3f} unbracketed {e['unbracketed_frac']:.3f}")


if __name__ == "__main__":
    main()
