#!/usr/bin/env python3
"""Appendix asset 5 -- the matched-dose comparison, and the master CSV behind everything.

Scientific question: at the SAME measured functional dose, do the stage differences remain?

Anchors are chosen INSIDE each pair's actual overlapping measured-dose support. Every value
is labelled `measured` (a rung landed on that dose) or `interp` (between two measured rungs,
which are named). Nothing is extrapolated: an anchor outside a state's measured range is
refused rather than estimated.

Emits
  outputs/analysis/oct_stage_dose_master.csv        one row per state x scale x endpoint
  workshop_iclr/tables/tableA_oct_matched_dose.tex  booktabs, appendix-ready
  workshop_iclr/data/tableA_oct_matched_dose.csv    the same numbers, machine-readable
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
from dm_common import load, curve, overlap, interp_at, PAIRS      # noqa: E402

OUT_T = REPO / "workshop_iclr" / "tables"
OUT_D = REPO / "workshop_iclr" / "data"
MASTER = REPO / "outputs" / "analysis" / "oct_stage_dose_master.csv"
SYM = {"M_D": r"$M_D$", "M_S": r"$M_S$", "M_D+0.25S": r"$M_{D+0.25S}$", "M_F": r"$M_F$"}
ENDPOINTS = [("B1", "$B_1$"), ("B2", "$B_2$"), ("selectivity", "sel."), ("k", "$k$")]


def anchors_for(rows, a, b, n=3):
    lo, hi = overlap(rows, a, b)
    pad = 0.04 * (hi - lo)
    lo, hi = lo + pad, hi - pad
    return [round(lo + i * (hi - lo) / (n - 1), 3) for i in range(n)]


def main() -> None:
    rows = load()

    MASTER.parent.mkdir(parents=True, exist_ok=True)
    cols = ["state", "arm", "seed", "nominal_scale", "is_trained_state", "dose",
            "B1", "B2", "selectivity", "k", "cos_to_MF",
            "impulsivity_offset", "impulsivity_ci_lo", "impulsivity_ci_hi"]
    with open(MASTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["state"], x["dose"] or 0)):
            w.writerow(r)
    print(f"  wrote {MASTER}  ({len(rows)} rows)")

    flat, L = [], []
    L += [r"\begin{table}[h]", r"\centering", r"\small",
          r"\begin{tabular}{llrrrrrl}", r"\toprule",
          r"target dose & state & scale & $B_1$ & $B_2$ & sel. & $k$ & source \\"]

    for name, (a, b) in PAIRS.items():
        lo, hi = overlap(rows, a, b)
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{SYM[a]} vs {SYM[b]}}} "
                 rf"-- overlap {lo:.3f}--{hi:.3f}}} \\")
        L.append(r"\midrule")
        for t in anchors_for(rows, a, b):
            for st in (a, b):
                vals, srcs = {}, set()
                for key, _ in ENDPOINTS:
                    v, how, brack = interp_at(rows, st, key, t)
                    vals[key] = v
                    if how:
                        srcs.add(how)
                sc, _, _ = interp_at(rows, st, "nominal_scale", t)
                src = "measured" if srcs == {"measured"} else "interp"
                L.append(" & ".join([
                    f"{t:.3f}" if st == a else "",
                    SYM[st],
                    f"{sc:.3f}" if sc is not None else "--",
                    *[f"{vals[k]:+.3f}" if k in ("B1", "B2") and vals[k] is not None
                      else (f"{vals[k]:.3f}" if vals[k] is not None else "--")
                      for k, _ in ENDPOINTS],
                    src]) + r" \\")
                flat.append({"pair": name, "target_dose": t, "state": st,
                             "nominal_scale": round(sc, 4) if sc else None,
                             **{k: (round(vals[k], 4) if vals[k] is not None else None)
                                for k, _ in ENDPOINTS},
                             "source": src})
            d1 = flat[-2]["B1"], flat[-1]["B1"]
            s1 = flat[-2]["selectivity"], flat[-1]["selectivity"]
            L.append(rf"& \multicolumn{{2}}{{r}}{{\emph{{difference}}}} & "
                     rf"\textbf{{{d1[1]-d1[0]:+.3f}}} & & "
                     rf"\textbf{{{s1[1]-s1[0]:+.3f}}} & & \\")

    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Stage comparisons at matched \emph{measured} functional dose "
          r"(trait-vector displacement, layer 15, forced prompt, seed 123456). Anchors lie "
          r"inside each pair's overlapping measured-dose support; \texttt{interp} marks a "
          r"value interpolated between two measured rungs and \texttt{measured} one that a "
          r"rung landed on. Nothing is extrapolated. \emph{difference} rows give the second "
          r"state minus the first on $B_1$ and selectivity. Both contrasts that the raw "
          r"decomposition suggested survive dose matching.}",
          r"\label{tab:oct-matched-dose}", r"\end{table}"]

    OUT_T.mkdir(parents=True, exist_ok=True); OUT_D.mkdir(parents=True, exist_ok=True)
    (OUT_T / "tableA_oct_matched_dose.tex").write_text("\n".join(L) + "\n")
    with open(OUT_D / "tableA_oct_matched_dose.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(f"  wrote {OUT_T}/tableA_oct_matched_dose.tex")
    print(f"  wrote {OUT_D}/tableA_oct_matched_dose.csv  ({len(flat)} rows)")

    print("\n  matched-dose differences (state B minus state A):")
    for name, (a, b) in PAIRS.items():
        print(f"    {name}")
        for t in anchors_for(rows, a, b):
            va = interp_at(rows, a, "B1", t)[0]; vb = interp_at(rows, b, "B1", t)[0]
            sa = interp_at(rows, a, "selectivity", t)[0]; sb = interp_at(rows, b, "selectivity", t)[0]
            print(f"      dose {t:.3f}:  B1 {va:+.3f} -> {vb:+.3f}  (diff {vb-va:+.3f}, "
                  f"{vb/va:.1f}x)   sel {sa:.3f} -> {sb:.3f}")


if __name__ == "__main__":
    main()
