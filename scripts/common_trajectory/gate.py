#!/usr/bin/env python3
"""Reproduction gate, behavioural + geometric checks (docs/spec_common_trajectory.md §6).

Compares the reproduced released-style merge (<c>_traj_mrg_final) and, for reference only, the
native-sequential endpoint (<c>_traj_seq_stepfinal) with the released adapter (<c>):

  B1/B2/B5  forced-choice offsets, contrast and retention k -- caa_logits_analysis estimator,
            question bootstrap with draws shared across the three states (paired diffs)
  B4        mean over traits of cos(dG_repro, dG_released), cross-fitted
            (common_shift.stats_for_trait), L15 primary and L20

Thresholds are read from the spec table and applied mechanically; nothing here is tuned.

    python3 scripts/common_trajectory/gate.py --constitution loving
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
from caa_logits_analysis import fit_offset_slope, stack_trait          # noqa: E402
from common_shift import load_arm, cell, make_preps, stats_for_trait   # noqa: E402
from geometry_lib import half_splits                                   # noqa: E402
from behaviour import load as load_logits                              # noqa: E402

GATE = {"loving": {"targets": ["empathy", "warmth"],
                   "B1": (1.25, 0.83), "B2": {"empathy": (1.68, 1.12), "warmth": (1.38, 0.92)},
                   "B4": (0.85, 0.70), "B5": (0.437, 0.08, 0.15)}}


def verdict(v, passv, stopv, higher=True):
    if higher:
        return "PASS" if v >= passv else ("STOP" if v <= stopv else "BETWEEN")
    return "PASS" if v <= passv else ("STOP" if v > stopv else "BETWEEN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    c = a.constitution; G = GATE[c]; tg = G["targets"]
    states = {"released": c, "repro_merge": f"{c}_traj_mrg_final", "repro_seq_final": f"{c}_traj_seq_stepfinal"}
    cells = load_logits(["base"] + list(states.values()))
    traits = sorted(cells["base"]); personas = sorted(cells["base"][traits[0]])
    rng = np.random.default_rng(0)
    off, boot, kk = {}, {}, {}
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]; apos = cells["base"][t][personas[0]]["a_is_positive"]
        nq = len(qid); allq = np.arange(nq); draws = [rng.integers(0, nq, nq) for _ in range(a.n_boot)]
        base = stack_trait(cells, "base", t, personas, qid)
        for s, arm in states.items():
            x = stack_trait(cells, arm, t, personas, qid)
            o, k = fit_offset_slope(base, x, apos, allq)
            off.setdefault(s, {})[t] = o; kk.setdefault(s, {})[t] = k
            boot.setdefault(s, {})[t] = np.array([fit_offset_slope(base, x, apos, q)[0] for q in draws])

    def contrast(src, s):
        rest = [t for t in traits if t not in tg]
        return np.mean([src[s][t] for t in tg], 0) - np.mean([src[s][t] for t in rest], 0)

    def ci(b):
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    res = {"constitution": c, "states": states, "behaviour": {}, "geometry": {}, "checks": {}}
    for s in states:
        res["behaviour"][s] = {
            "offsets": {t: {"point": float(off[s][t]), "ci": ci(boot[s][t])} for t in traits},
            "contrast": {"point": float(contrast(off, s)), "ci": ci(contrast(boot, s))},
            "k_mean": float(np.mean(list(kk[s].values())))}
        if s != "released":
            d = contrast(boot, s) - contrast(boot, "released")
            res["behaviour"][s]["contrast_minus_released"] = {"point": float(contrast(off, s) - contrast(off, "released")), "ci": ci(d)}
            res["behaviour"][s]["profile_corr_with_released"] = float(np.corrcoef(
                [off[s][t] for t in traits], [off["released"][t] for t in traits])[0, 1])

    cache = REPO / "outputs" / "_qcache"; srng = np.random.default_rng(0)
    for layer in (15, 20):
        bd = load_arm(cache, "base", [15, 20], layer)
        D = {s: load_arm(cache, arm, [15, 20], layer) for s, arm in states.items()}
        pidx = [bd["personas"].index(p) for p in bd["semantic"]]
        per = {}
        for ti, t in enumerate(bd["traits"]):
            bp, bn = cell(bd, ti, pidx)
            preps = make_preps(bp, bn, {s: cell(D[s], ti, pidx) for s in states})
            r = stats_for_trait(preps, half_splits(int(bd["nq"][ti]), 40, srng), list(states), naive=False)
            per[t] = {"cos_merge_released": r["cos"]["repro_merge|released"],
                      "cos_seq_released": r["cos"]["repro_seq_final|released"],
                      "g": {s: r["per_arm"][s]["g_over_base"] for s in states},
                      "S": {s: r["per_arm"][s]["share_squared"] for s in states}}
        res["geometry"][str(layer)] = {"per_trait": per,
            "mean_cos_merge_released": float(np.mean([v["cos_merge_released"] for v in per.values()])),
            "mean_cos_seq_released": float(np.mean([v["cos_seq_released"] for v in per.values()])),
            "mean_S": {s: float(np.mean([v["S"][s] for v in per.values()])) for s in states},
            "mean_g": {s: float(np.mean([v["g"][s] for v in per.values()])) for s in states}}

    m = res["behaviour"]["repro_merge"]
    lc = m["contrast"]
    res["checks"]["B1"] = {"value": lc["point"], "ci": lc["ci"],
                           "verdict": "STOP" if lc["ci"][0] <= 0 <= lc["ci"][1] else verdict(lc["point"], *G["B1"])}
    res["checks"]["B2"] = {t: {"value": m["offsets"][t]["point"], "verdict": verdict(m["offsets"][t]["point"], *G["B2"][t])} for t in tg}
    b4 = res["geometry"]["15"]["mean_cos_merge_released"]
    res["checks"]["B4"] = {"value": b4, "verdict": verdict(b4, *G["B4"])}
    ref, p_, s_ = G["B5"]; dk = abs(m["k_mean"] - ref)
    res["checks"]["B5"] = {"value": m["k_mean"], "released_measured": res["behaviour"]["released"]["k_mean"],
                           "verdict": "PASS" if dk <= p_ else ("STOP" if dk > s_ else "BETWEEN")}
    out = REPO / f"outputs/analysis/common_trajectory/gate_{c}.json"
    out.write_text(json.dumps(res, indent=1))

    print(f"{'trait':14s} {'released':>16s} {'repro merge':>16s} {'repro seq final':>16s}")
    for t in traits + ["CONTRAST"]:
        row = [res["behaviour"][s]["offsets"][t] if t != "CONTRAST" else res["behaviour"][s]["contrast"] for s in states]
        print(f"{t:14s}" + "".join(f"{r['point']:+8.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]"[:17].rjust(17) for r in row))
    print("k mean:", {s: round(res["behaviour"][s]["k_mean"], 3) for s in states})
    print("profile corr with released:", {s: round(res["behaviour"][s]["profile_corr_with_released"], 3) for s in ("repro_merge", "repro_seq_final")})
    print("contrast minus released:", {s: res["behaviour"][s]["contrast_minus_released"] for s in ("repro_merge", "repro_seq_final")})
    for L in ("15", "20"):
        gg = res["geometry"][L]
        print(f"L{L}: mean cos(dG merge, released) {gg['mean_cos_merge_released']:.3f}; seq {gg['mean_cos_seq_released']:.3f}; "
              f"mean S {', '.join(f'{k} {v:.3f}' for k, v in gg['mean_S'].items())}; mean g {', '.join(f'{k} {v:.3f}' for k, v in gg['mean_g'].items())}")
    print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    main()
