#!/usr/bin/env python3
"""Logit-offset sensitivity to the letter-bias assumption (audit issue 3). CPU only.

Three estimators of a state's trait preference shift against the unadapted base, from the same
forced-prompt log-odds (caa_logits_forced), with ONE set of question-bootstrap draws per trait
shared by every state and estimator, so every contrast is paired:

  SEP  fit_offset_slope (the reported estimator): per-polarity OLS intercepts, averaged.
       Leaves b*(k_minus - k_plus)/2 of the base letter bias b.
  SH   fit_offset_shared_slope: y = a + k*x + gamma*s. Exact if retention is common to both
       polarity groups; misspecified otherwise.
  RAW  polarity-balanced mean of (state - base) log-odds. Cancels both letter biases exactly
       but adds (k - 1) * baseline preference: it measures the change in mean log-odds, not an
       offset at fixed baseline.

None is assumption-free; a conclusion is reported as robust only if it holds under all three.
Also reported per state x trait: k_plus, k_minus, b_hat and the implied SEP bias term.

Covers the cross-character pilot (levels, change from start, comparisons 1-4, B1/B2/LOV) and
the common-trajectory endpoints (DPO and final state vs base, per trait).

    python3 scripts/audit/logit_sensitivity.py [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "common_trajectory"))
from caa_logits_analysis import (balanced_mean, fit_offset_shared_slope,  # noqa: E402
                                 fit_offset_slope, fit_offset_slope_detail, stack_trait)
from states import trajectory  # noqa: E402

OUT = REPO / "outputs/analysis/audit/logit_sensitivity.json"
TAG, SUB = "llama-3.1-8b", "caa_logits_forced"
EST = ("SEP", "SH", "RAW")
XCHAR = ["xchar_lov_start", "xchar_lov_imp_s100", "xchar_lov_lov_s100", "xchar_base_imp_s100",
         "xchar_base_lov_s100"]
CONTRASTS = {"B1": ["impulsivity"], "B2": ["impulsivity", "risk_taking"], "LOV": ["empathy", "warmth"]}


def arm_dir(arm):
    return REPO / "outputs" / f"{TAG}-{arm}" / SUB


def load(arms):
    cells = {}
    for arm in arms:
        for f in sorted(arm_dir(arm).glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            cells.setdefault(arm, {}).setdefault(str(z["trait"]), {})[str(z["persona"])] = {
                "qid": z["qid"], "logodds": z["logodds"], "a_is_positive": z["a_is_positive"]}
    return cells


def est(base, arm, apos, q):
    return (fit_offset_slope(base, arm, apos, q)[0], fit_offset_shared_slope(base, arm, apos, q)[0],
            balanced_mean(arm - base, apos, q))


def ci(point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    traj = {c: [s["arm"] for s in trajectory(c)] for c in ("impulsiveness", "loving", "goodness", "sycophancy")}
    traj_states = sorted({x for arms in traj.values() for x in (arms[0], arms[-1])})
    states = [s for s in XCHAR + traj_states if len(list(arm_dir(s).glob("*.npz"))) == 88]
    cells = load(["base"] + states)
    traits = sorted(cells["base"]); personas = sorted(cells["base"][traits[0]])
    rng = np.random.default_rng(a.seed)
    P, B, D = {}, {}, {}                         # point[s][t] (3,), boot[s][t] (n,3), detail
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]; apos = cells["base"][t][personas[0]]["a_is_positive"]
        nq = len(qid); allq = np.arange(nq); draws = [rng.integers(0, nq, nq) for _ in range(a.n_boot)]
        base = stack_trait(cells, "base", t, personas, qid)
        for s in states:
            arm = stack_trait(cells, s, t, personas, qid)
            P.setdefault(s, {})[t] = np.array(est(base, arm, apos, allq))
            B.setdefault(s, {})[t] = np.array([est(base, arm, apos, q) for q in draws])
            D.setdefault(s, {})[t] = fit_offset_slope_detail(base, arm, apos, allq)
        print("done", t, flush=True)

    def val(s, m):
        ts = CONTRASTS.get(m, [m]); rest = [t for t in traits if t not in ts] if m in CONTRASTS else None
        p = np.mean([P[s][t] for t in ts], 0); b = np.mean([B[s][t] for t in ts], 0)
        if rest:
            p = p - np.mean([P[s][t] for t in rest], 0); b = b - np.mean([B[s][t] for t in rest], 0)
        return p, b

    def block(fn):
        out = {}
        for m in traits + list(CONTRASTS):
            p, b = fn(m)
            out[m] = {e: ci(p[i], b[:, i]) for i, e in enumerate(EST)}
            signs = {np.sign(out[m][e]["point"]) for e in EST}
            excl0 = [not (out[m][e]["lo"] <= 0 <= out[m][e]["hi"]) for e in EST]
            out[m]["robust_sign"] = len(signs) == 1 and all(excl0)
        return out

    res = {"n_boot": a.n_boot, "estimators": {"SEP": "fit_offset_slope (reported)",
                                              "SH": "fit_offset_shared_slope", "RAW": "balanced mean change"},
           "levels": {s: block(lambda m, s=s: val(s, m)) for s in states},
           "polarity_detail": {s: {t: {"k_plus": D[s][t]["groups"]["plus"]["k"],
                                       "k_minus": D[s][t]["groups"]["minus"]["k"],
                                       "b_hat": D[s][t]["b_hat"], "implied_bias": D[s][t]["implied_bias"]}
                                   for t in traits} for s in states}}
    if all(s in states for s in XCHAR):
        st = "xchar_lov_start"
        d = lambda x, y: (lambda m: tuple(u - v for u, v in zip(val(x, m), val(y, m))))
        res["xchar"] = {
            "chg_lov_imp": block(d("xchar_lov_imp_s100", st)),
            "chg_lov_lov": block(d("xchar_lov_lov_s100", st)),
            "comp2_lovimp_minus_lovlov": block(d("xchar_lov_imp_s100", "xchar_lov_lov_s100")),
            "comp3_baseimp_minus_baselov": block(d("xchar_base_imp_s100", "xchar_base_lov_s100")),
            "comp4_moderation": block(lambda m: tuple(
                (u - v) - (w - x) for u, v, w, x in zip(val("xchar_lov_imp_s100", m), val("xchar_lov_lov_s100", m),
                                                       val("xchar_base_imp_s100", m), val("xchar_base_lov_s100", m))))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for k, blk in res.get("xchar", {}).items():
        print(f"\n{k}")
        for m in ["impulsivity", "risk_taking", "empathy", "warmth", "B1", "B2", "LOV"]:
            print(f"  {m:12s}" + "".join(f"  {e} {blk[m][e]['point']:+.3f} [{blk[m][e]['lo']:+.3f},{blk[m][e]['hi']:+.3f}]"
                                          for e in EST) + ("" if blk[m]["robust_sign"] else "   <- NOT robust"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
