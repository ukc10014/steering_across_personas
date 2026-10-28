#!/usr/bin/env python3
"""Original geometry vs the audit reruns, per constitution. CPU only, reads JSON.

  ORIG  outputs/analysis/common_trajectory/geometry_<c>.json      (epsilon-floored estimator,
                                                                    legacy base vs corrected arms)
  A     v2/geometry_<c>_mixedprotocol.json   same comparison, estimability-rule estimator
  B     v2/geometry_<c>_synthbase.json       estimability rule, synthetic corrected-mask base

A vs ORIG isolates the estimator repair; B vs A the mask comparability (approximately: the
synthetic base assumes a model-independent artifact). Incremental quantities (gS, SS, cosS, Q)
never touch the base, so B == A for them up to the bootstrap draws, which are identical.

    python3 scripts/audit/compare_geometry.py [--constitutions ...]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A_DIR = REPO / "outputs/analysis/common_trajectory"
Q = ["g", "S", "cosT", "gS", "SS", "cosS", "Q"]


def fmt(e):
    if e is None:
        return "—"
    if e.get("point") is None:
        return "n/e"
    lo, hi = e.get("lo"), e.get("hi")
    return f"{e['point']:.3f} [{lo:.3f},{hi:.3f}]" if lo is not None else f"{e['point']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitutions", nargs="+", default=["impulsiveness", "loving", "goodness", "sycophancy"])
    a = ap.parse_args()
    report = {}
    for c in a.constitutions:
        fA, fB = A_DIR / f"v2/geometry_{c}_mixedprotocol.json", A_DIR / f"v2/geometry_{c}_synthbase.json"
        if not (fA.exists() and fB.exists()):
            print(c, "not ready"); continue
        O = json.loads((A_DIR / f"geometry_{c}.json").read_text())
        RA, RB = json.loads(fA.read_text()), json.loads(fB.read_text())
        report[c] = {}
        for L in ("15", "20"):
            report[c][L] = {}
            for grp in ("MEAN8", "TARGETS"):
                so, sa, sb = (R["layers"][L]["summary"][grp] for R in (O, RA, RB))
                rows = {}
                for arm in sa:
                    rows[arm] = {q: {"orig": so[arm].get(q), "A": sa[arm].get(q), "B": sb[arm].get(q)} for q in Q}
                    # A must reproduce ORIG wherever ORIG was well defined (same splits/draws)
                    for q in ("g",):
                        po, pa = so[arm][q]["point"], sa[arm][q]["point"]
                        assert abs(po - pa) < 1e-4 * max(1, abs(po)), (c, L, grp, arm, q, po, pa)
                report[c][L][grp] = rows
            print(f"\n=== {c} L{L} MEAN8   (orig | A: estimator fixed | B: + synthetic corrected base)")
            for arm, r in report[c][L]["MEAN8"].items():
                short = arm.replace(c + "_", "")
                print(f"  {short:24s}" + "".join(f"\n      {q:5s} {fmt(r[q]['orig']):28s} | {fmt(r[q]['A']):28s} | {fmt(r[q]['B'])}"
                                                   for q in ("g", "S", "cosT", "cosS", "Q")))
    out = REPO / "outputs/analysis/audit/geometry_comparison.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(report); out.write_text(json.dumps(prev, indent=1)); print("\nwrote", out)


if __name__ == "__main__":
    main()
