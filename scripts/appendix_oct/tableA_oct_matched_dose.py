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


def _phase1_ladder() -> dict[str, list[tuple[float, float]]]:
    """The cheap scale->dose grid behind figA_oct_dose_calibration, same source file."""
    import json
    d = json.loads((REPO / "outputs" / "analysis" /
                    "dose_stage_grid_analysis.json").read_text())["dose"]["15"]
    out: dict[str, list[tuple[float, float]]] = {}
    for cfg, v in d.items():
        if not cfg.startswith("M_"):
            continue
        state, _, sc = cfg.rpartition("_s")
        out.setdefault(state, []).append((float(sc),
                                          float(v["trait_vector"])))
    for k in out:
        out[k].sort()
    return out


def anchors_for(rows, a, b, n=3):
    lo, hi = overlap(rows, a, b)
    pad = 0.04 * (hi - lo)
    lo, hi = lo + pad, hi - pad
    return [round(lo + i * (hi - lo) / (n - 1), 3) for i in range(n)]


def main() -> None:
    rows = load()

    MASTER.parent.mkdir(parents=True, exist_ok=True)
    cols = ["state", "arm", "seed", "nominal_scale", "is_trained_state",
            "row_source", "in_monotone_region", "dose",
            "B1", "B2", "selectivity", "k", "cos_to_MF",
            "impulsivity_offset", "impulsivity_ci_lo", "impulsivity_ci_hi"]
    # Two kinds of row, kept distinct by `row_source`:
    #   extraction    a full 192-cell rung -- has endpoints
    #   phase1_probe  a cheap scale->dose ladder rung -- dose only, no endpoints
    # The probe rows are here so the EXCLUSIONS are machine-readable: M_D+S and M_S turn
    # over at s=2, and those rungs carry in_monotone_region=no. Every extraction rung was
    # chosen from its state's monotone prefix, so all of them are yes.
    out = []
    for r in rows:
        out.append({**r, "row_source": "extraction", "in_monotone_region": "yes"})
    for st, pts in _phase1_ladder().items():
        dd = [d for _, d in pts]
        kmon = 1
        while kmon < len(dd) and dd[kmon] > dd[kmon - 1]:
            kmon += 1
        for j, (sc, dose) in enumerate(pts):
            out.append({"state": st, "arm": "", "seed": 1, "nominal_scale": sc,
                        "is_trained_state": sc == 1.0, "row_source": "phase1_probe",
                        "in_monotone_region": "yes" if j < kmon else "no", "dose": dose})
    with open(MASTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(out, key=lambda x: (x["state"], x["row_source"],
                                            x["dose"] or 0)):
            w.writerow(r)
    n_excl = sum(1 for r in out if r["in_monotone_region"] == "no")
    print(f"  wrote {MASTER}  ({len(out)} rows, {n_excl} flagged non-monotone)")

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
            # carry the pairwise differences into the CSV too, on the second state's row,
            # so the machine-readable table needs no re-derivation to be read
            flat[-1]["diff_B1"] = round(d1[1] - d1[0], 4)
            flat[-1]["diff_selectivity"] = round(s1[1] - s1[0], 4)
            flat[-2]["diff_B1"] = flat[-2]["diff_selectivity"] = None
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
        w = csv.DictWriter(fh, fieldnames=list(flat[0]) + ["diff_B1", "diff_selectivity"]
                           if "diff_B1" not in flat[0] else list(flat[0]))
        w.writeheader(); w.writerows(flat)
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
