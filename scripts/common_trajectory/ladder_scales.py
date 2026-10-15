#!/usr/bin/env python3
"""DPO-only ladder scales by the rule frozen in spec amendment A1.

Five scales s, chosen so that base + s*dW_D brackets the trajectory's measured L15 dose range:

  1. r_min, r_max = min / max over the trajectory states of dose / dose(DPO-only)
  2. targets: five dose ratios evenly spaced over [r_min - 0.05, r_max + 0.05]
  3. map ratio -> scale by inverting the dose-vs-scale relation pooled from the EXISTING
     ladders (impulsiveness and loving: r(s) for every measured rung, plus (1, 1)),
     linear interpolation, linear extrapolation from the end segments
  4. round to 0.05; a scale within 0.05 of 1 (the existing DPO-only state) is pushed to 1 +/- 0.1;
     duplicates are spread by 0.05

Prints the scales (space-separated) and records the derivation in ladder_scales_<c>.json.

    python3 scripts/common_trajectory/ladder_scales.py --constitution goodness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from states import trajectory   # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
REF = {"impulsiveness": ("impulsiveness_repro_dpo", "impulsiveness_dm_m_d_s"),
       "loving": ("loving_traj_dpo", "loving_traj_ladder_d_s")}


def relation():
    pts = [(1.0, 1.0)]
    for c, (dpo, pre) in REF.items():
        fd = json.loads((A / f"functional_dose_{c}_dpo_ladder.json").read_text())["15"]
        d0 = fd[dpo]["trait_vector_displacement"]
        pts += [(float(k[len(pre):]), v["trait_vector_displacement"] / d0) for k, v in fd.items() if k.startswith(pre)]
    pts.sort(key=lambda p: p[1])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--constitution", required=True); a = ap.parse_args()
    c = a.constitution
    fd = json.loads((A / f"functional_dose_{c}.json").read_text())["15"]
    arms = [s["arm"] for s in trajectory(c)]
    d0 = fd[arms[0]]["trait_vector_displacement"]
    r = np.array([fd[x]["trait_vector_displacement"] / d0 for x in arms])
    rmin, rmax = r.min(), r.max()
    targets = np.linspace(rmin - 0.05, rmax + 0.05, 5)
    s_ref, r_ref = relation()

    def inv(rt):
        if rt <= r_ref[0]:
            k = (s_ref[1] - s_ref[0]) / (r_ref[1] - r_ref[0]); return s_ref[0] + k * (rt - r_ref[0])
        if rt >= r_ref[-1]:
            k = (s_ref[-1] - s_ref[-2]) / (r_ref[-1] - r_ref[-2]); return s_ref[-1] + k * (rt - r_ref[-1])
        return float(np.interp(rt, r_ref, s_ref))

    raw = [inv(t) for t in targets]
    sc = []
    for s in raw:
        s = round(s * 20) / 20
        if abs(s - 1) < 0.05:
            s = 0.9 if s <= 1 else 1.1
        while s in sc or abs(s - 1) < 0.05:
            s = round(s + 0.05, 2)
        sc.append(round(s, 2))
    sc = sorted(sc)
    (A / f"ladder_scales_{c}.json").write_text(json.dumps({
        "rule": "spec amendment A1", "dpo_dose_L15": d0, "trajectory_dose_ratios": r.tolist(),
        "target_ratios": targets.tolist(), "raw_scales": raw, "scales": sc,
        "relation_points": list(zip(s_ref.tolist(), r_ref.tolist()))}, indent=1))
    print(" ".join(f"{s:g}" for s in sc))


if __name__ == "__main__":
    main()
