#!/usr/bin/env python3
"""Behavioural phenotype trajectory. docs/spec_common_trajectory.md §5.

Per state x trait, the forced-choice offset `a` of scripts/caa_logits_analysis.py, unchanged
(logodds_state = a + k*logodds_base, polarity-split, the compression-corrected estimator),
always against the ORIGINAL unadapted base. Contrasts are mean(targets) - mean(other traits)
(states.CONTRASTS), the B2 construction.

For a signed summary T (every trait and every contrast):
    change   D(e) = T(e) - T(0)                    (0 = the DPO-only state)
    fraction F(e) = D(e) / D(final)
F is reported only where it means something: it is flagged `unstable` when |D(final)| < 0.25
log-odds or D(final)'s CI includes 0, and a checkpoint is flagged `sign_reversal` when D(e)'s
CI excludes 0 on the side opposite to D(final).

Uncertainty: one set of question-bootstrap draws per trait, shared by every state, so all
differences are paired. Evaluation uncertainty only (one training seed).

    python3 scripts/common_trajectory/behaviour.py --constitution impulsiveness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from caa_logits_analysis import fit_offset_slope, stack_trait   # noqa: E402
from states import CONTRASTS, trajectory                         # noqa: E402

LOGITS = Path("/workspace/repos/steering_across_personas/outputs")   # measured states live here
MIN_D = 0.25


def load(arms):
    cells = {}
    for arm in arms:
        d = LOGITS / f"llama-3.1-8b-{arm}" / "caa_logits_forced"
        fs = sorted(d.glob("*.npz"))
        if len(fs) != 88:
            raise SystemExit(f"{arm}: {len(fs)}/88 logit cells")
        for f in fs:
            z = np.load(f, allow_pickle=False)
            cells.setdefault(arm, {}).setdefault(str(z["trait"]), {})[str(z["persona"])] = {
                "qid": z["qid"], "logodds": z["logodds"], "a_is_positive": z["a_is_positive"]}
    return cells


def summ(pt, b):
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return {"point": float(pt), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    traj = trajectory(a.constitution)
    arms = [s["arm"] for s in traj]
    cells = load(["base"] + arms)
    traits = sorted(cells["base"])
    personas = sorted(cells["base"][traits[0]])
    rng = np.random.default_rng(a.seed)

    pt = {x: {} for x in arms}; bt = {x: {} for x in arms}; kk = {x: {} for x in arms}
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]; apos = cells["base"][t][personas[0]]["a_is_positive"]
        nq = len(qid); allq = np.arange(nq)
        draws = [rng.integers(0, nq, nq) for _ in range(a.n_boot)]
        base = stack_trait(cells, "base", t, personas, qid)
        for x in arms:
            arm = stack_trait(cells, x, t, personas, qid)
            pt[x][t], kk[x][t] = fit_offset_slope(base, arm, apos, allq)
            bt[x][t] = np.array([fit_offset_slope(base, arm, apos, q)[0] for q in draws])

    names = {t: [t] for t in traits}
    for n, tg in CONTRASTS.get(a.constitution, {}).items():
        names[n] = tg
        # the contrast's two halves as their own series, so a moving contrast can be split into
        # target reinforcement vs movement of the other six (mean, not a contrast)
        names[f"{n}:targets_mean"] = tg
        names[f"{n}:others_mean"] = [t for t in traits if t not in tg]

    def val(x, n, src):
        tg = names[n]
        if n in traits:
            return src[x][n]
        if ":" in n:
            return np.mean([src[x][t] for t in tg], 0)
        rest = [t for t in traits if t not in tg]
        return np.mean([src[x][t] for t in tg], 0) - np.mean([src[x][t] for t in rest], 0)

    dpo, fin = arms[0], arms[-1]
    out = {"constitution": a.constitution, "states": traj, "personas": personas, "traits": traits,
           "contrasts": CONTRASTS.get(a.constitution, {}), "n_boot": a.n_boot,
           "estimator": "caa_logits_analysis.fit_offset_slope offset a vs base (forced prompt)",
           "series": {}}
    for n in names:
        Df_p, Df_b = val(fin, n, pt) - val(dpo, n, pt), val(fin, n, bt) - val(dpo, n, bt)
        dfin = summ(Df_p, Df_b)
        unstable = abs(Df_p) < MIN_D or (dfin["lo"] <= 0 <= dfin["hi"])
        rows = []
        for s, x in zip(traj, arms):
            D_p, D_b = val(x, n, pt) - val(dpo, n, pt), val(x, n, bt) - val(dpo, n, bt)
            r = {"arm": x, "step": s["step"], "epoch": s["epoch"],
                 "level": summ(val(x, n, pt), val(x, n, bt)), "change": summ(D_p, D_b)}
            if x != dpo:
                r["fraction"] = None if unstable else summ(D_p / Df_p, D_b / Df_b)
                c = r["change"]
                r["sign_reversal"] = bool((Df_p > 0 and c["hi"] < 0) or (Df_p < 0 and c["lo"] > 0))
            rows.append(r)
        out["series"][n] = {"endpoint_change": dfin, "fraction_unstable": bool(unstable), "rows": rows}
    out["slope_k"] = {x: {t: float(kk[x][t]) for t in traits} for x in arms}

    # ---- amendment A1: the full eight-trait profile is the primary outcome for every
    # constitution. Delta(e) = p(e) - p(DPO-only) over the 8 traits; per checkpoint:
    #   norm   ||Delta(e)||                          (L2; small upward noise bias, not corrected)
    #   cos    cos(Delta(e), Delta(final))
    #   q      <Delta(e), Delta(final)> / ||Delta(final)||^2   (descriptive: final = reference)
    # plus the total profile vs base: ||p(e)|| and cos(p(e), p(final)).
    # Replicate r combines every trait's draw r -- traits use disjoint questions and
    # independent draws, so this is a valid draw of the vector statistic.
    P = {x: np.array([pt[x][t] for t in traits]) for x in arms}                 # (8,)
    PB = {x: np.stack([bt[x][t] for t in traits], 1) for x in arms}            # (B, 8)
    dF, dFb = P[fin] - P[dpo], PB[fin] - PB[dpo]
    prof = []
    for s, x in zip(traj, arms):
        d, db = P[x] - P[dpo], PB[x] - PB[dpo]
        cosb = lambda u, v: (u * v).sum(-1) / np.sqrt((u * u).sum(-1) * (v * v).sum(-1) + 1e-18)
        r = {"arm": x, "step": s["step"], "epoch": s["epoch"],
             "profile_norm": summ(np.linalg.norm(P[x]), np.linalg.norm(PB[x], axis=1)),
             "profile_cos_to_final": summ(cosb(P[x], P[fin]), cosb(PB[x], PB[fin])),
             "change_norm": summ(np.linalg.norm(d), np.linalg.norm(db, axis=1))}
        if x != dpo:
            r["change_cos_to_final"] = summ(cosb(d, dF), cosb(db, dFb))
            r["q"] = summ(d @ dF / (dF @ dF), (db * dFb).sum(1) / (dFb * dFb).sum(1))
        prof.append(r)
    out["profile"] = {"traits": traits, "endpoint_change_norm": prof[-1]["change_norm"], "rows": prof}

    p = REPO / f"outputs/analysis/common_trajectory/behaviour_{a.constitution}.json"
    p.write_text(json.dumps(out, indent=1))
    for n in names:
        S = out["series"][n]
        print(f"{n:14s} D(final) {S['endpoint_change']['point']:+.3f}"
              + (" [F unstable]" if S["fraction_unstable"] else "")
              + "  levels: " + " ".join(f"{r['level']['point']:+.2f}" for r in S["rows"]))
    print("wrote", p)


if __name__ == "__main__":
    main()
