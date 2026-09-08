#!/usr/bin/env python3
"""Turn the cheap scale->dose grid into a rung plan with genuine overlap.

docs/spec_stage_localisation.md successor experiment: the raw stage table has
Spearman(dose, B1) = +1.000, so stage and dose are perfectly confounded. Fixing that needs
the states compared at MATCHED MEASURED DOSE -- not at matched nominal LoRA scale, and not
at matched weight norm.

This does step 2-3 of that workflow:
  * read dose(s) for every state from the calibration grid,
  * report how dose responds to s per state (it is NOT assumed linear -- that assumption is
    what dose_calibrate.py exists to test),
  * find the dose band where the states to be compared actually overlap,
  * emit the scales that land each state on a set of common target doses.

The probe grid is 16x smaller than a full extraction, so its absolute doses differ a little
from the full-cache numbers (M_D 0.5375 vs 0.5314, M_F 0.8792 vs 0.8178 at s=1). It is used
only to CHOOSE rungs; every reported comparison is made on the full measure afterwards.

    python scripts/dose_match_plan.py --root outputs/analysis/dose_stage_grid
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

MEASURE = "trait_vector_displacement"   # the measure the seed/stage reports use


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/analysis/dose_stage_grid")
    ap.add_argument("--layer", default="15")
    ap.add_argument("--analysis", default="outputs/analysis/dose_stage_grid_analysis.json")
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="target doses; default = evenly spaced across the overlap band")
    ap.add_argument("--out", default="outputs/analysis/dose_match_plan.json")
    a = ap.parse_args()

    d = json.loads(Path(a.analysis).read_text())
    layer = d[a.layer] if a.layer in d else d[str(a.layer)]

    # config name -> (state, scale)
    curves: dict[str, list[tuple[float, float]]] = {}
    for cfg, v in layer.items():
        if cfg == "base" or not cfg.startswith("M_"):
            continue
        state, _, s = cfg.rpartition("_s")
        try:
            s = float(s)
        except ValueError:
            continue
        dose = v[MEASURE] if isinstance(v, dict) else v
        curves.setdefault(state, []).append((s, float(dose)))
    for k in curves:
        curves[k].sort()

    print(f"\nDOSE RESPONSE per state, layer {a.layer}, {MEASURE}\n")
    print(f"{'state':14s}" + "".join(f"{s:>10}" for s in ("s=0.5", "s=0.75", "s=1.0", "s=1.5", "s=2.0")))
    for st, pts in curves.items():
        row = {s: v for s, v in pts}
        print(f"{st:14s}" + "".join(f"{row.get(s, float('nan')):10.4f}"
                                   for s in (0.5, 0.75, 1.0, 1.5, 2.0)))

    print("\n  linearity check -- dose(s)/s should be constant if dose were linear in s:")
    for st, pts in curves.items():
        r = [v / s for s, v in pts if s > 0]
        print(f"    {st:14s} dose/s spans {min(r):.4f}-{max(r):.4f}  "
              f"({'near-linear' if max(r)/min(r) < 1.15 else 'NOT linear'})")

    lo = max(min(v for _, v in pts) for pts in curves.values())
    hi = min(max(v for _, v in pts) for pts in curves.values())
    print(f"\n  OVERLAP BAND across all {len(curves)} states: dose {lo:.4f} .. {hi:.4f}")
    if hi <= lo:
        raise SystemExit("  no common dose band -- widen the scale grid")

    targets = a.targets or list(np.round(np.linspace(lo, hi, 3), 4))
    print(f"  target doses: {targets}\n")
    print(f"{'state':14s}" + "".join(f"{'dose '+str(t):>16s}" for t in targets))
    plan = {}
    for st, pts in curves.items():
        ss = np.array([s for s, _ in pts]); dd = np.array([v for _, v in pts])
        row, cells = {}, ""
        for t in targets:
            # monotone in s, so a plain interp on the measured points is enough; never
            # extrapolate past the grid -- that is where a matched claim would go soft.
            s_hat = float(np.interp(t, dd, ss))
            inside = dd.min() <= t <= dd.max()
            row[str(t)] = {"scale": round(s_hat, 4), "within_measured_range": bool(inside)}
            cells += f"{('s=%.3f' % s_hat) + ('' if inside else ' *'):>16s}"
        plan[st] = row
        print(f"{st:14s}{cells}")
    print("\n  * outside this state's measured dose range -- would be extrapolation, not used")

    Path(a.out).write_text(json.dumps(
        {"layer": a.layer, "measure": MEASURE, "targets": targets,
         "curves": {k: [{"scale": s, "dose": v} for s, v in pts] for k, pts in curves.items()},
         "plan": plan}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
