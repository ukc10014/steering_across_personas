#!/usr/bin/env python3
"""Appendix table -- the dense SFT checkpoint curve.

One row per measured checkpoint, both constructions, with the measured functional dose
beside every endpoint so the step axis is never read on its own.

The `vs M_D` column is the dose control and is the point of the table: B1 at that checkpoint
divided by B1 of M_D scaled to the SAME measured dose. 1.0 would mean the checkpoint is worth
no more than moving the model that far with the DPO adapter alone. Interpolation is refused
outside M_D's measured dose range rather than extrapolated.

Emits
  workshop_iclr/tables/tableA_oct_sft_curve.tex
  workshop_iclr/data/tableA_oct_sft_curve.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
import curve_common as cc  # noqa: E402

OUT_T = REPO / "workshop_iclr" / "tables"
OUT_D = REPO / "workshop_iclr" / "data"
MASTER = REPO / "outputs" / "analysis" / "oct_stage_dose_master.csv"
CON_SYM = {"seq": r"$M_D{+}\Delta W_S$", "mrg": r"merge"}


def md_reference():
    pts = []
    with open(MASTER) as fh:
        for r in csv.DictReader(fh):
            if r["row_source"] == "extraction" and r["state"] == "M_D" and r["dose"] and r["B1"]:
                pts.append((float(r["dose"]), float(r["B1"])))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main() -> None:
    rows = cc.load()
    rd, rb = md_reference()
    flat, L = [], []
    L += [r"\begin{table}[h]", r"\centering", r"\small",
          r"\begin{tabular}{lrrrrrrl}", r"\toprule",
          r"construction & step & epoch & dose & $B_1$ & sel. & $k$ & vs $M_D$ \\"]

    for con in ("seq", "mrg"):
        pts = [r for r in rows if r["kind"] == "curve" and r["construction"] == con]
        if not pts:
            continue
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{CON_SYM[con]}}}}} \\")
        zero = next((r for r in rows if r["state"] == "M_D"), None)
        for r in ([zero] if zero else []) + sorted(pts, key=lambda x: x["step"]):
            d, b = r.get("dose"), r.get("B1")
            ratio = "--"
            if d is not None and b is not None and rd.min() <= d <= rd.max():
                ratio = f"{b / float(np.interp(d, rd, rb)):.1f}$\\times$"
            elif d is not None and b is not None:
                ratio = r"\textit{n/a}"          # outside M_D's measured range; never extrapolated
            sel, k = r.get("selectivity"), r.get("k")
            L.append(" & ".join([
                CON_SYM[con] if r["step"] else r"$M_D$ (no SFT)",
                str(r["step"]), f"{r['epoch']:.3f}" if r["epoch"] is not None else "--",
                f"{d:.3f}" if d is not None else "--",
                f"{b:+.3f}" if b is not None else "--",
                f"{sel:.3f}" if sel is not None else "--",
                f"{k:.3f}" if k is not None else "--", ratio]) + r" \\")
            flat.append({"construction": con, "step": r["step"], "epoch": r["epoch"],
                         "dose": round(d, 4) if d is not None else None,
                         "B1": round(b, 4) if b is not None else None,
                         "B2": round(r["B2"], 4) if r.get("B2") is not None else None,
                         "selectivity": round(sel, 4) if sel is not None else None,
                         "k": round(k, 4) if k is not None else None,
                         "vs_M_D_at_same_dose": ratio.replace("$\\times$", "")})

    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The introspection SFT measured at checkpoints through training "
          r"(seed 123456, layer 15, forced prompt). \emph{step} is optimizer steps; one epoch "
          r"is 374.2 steps, so the run is 1122 steps and the final row is the trained "
          r"endpoint. \emph{dose} is \emph{measured} functional dose, never weight norm. The "
          r"last column is the dose control: $B_1$ divided by $B_1$ of $M_D$ scaled to the "
          r"same measured dose, so $1.0\times$ would mean the checkpoint is worth no more "
          r"than moving the model that far with the DPO adapter alone. Nothing is "
          r"extrapolated outside $M_D$'s measured range. The two constructions differ in SFT "
          r"weight ($1.00$ against $0.25$) as well as in the merge, so the gap between them "
          r"is not the cross-term contribution.}",
          r"\label{tab:oct-sft-curve}", r"\end{table}"]

    OUT_T.mkdir(parents=True, exist_ok=True); OUT_D.mkdir(parents=True, exist_ok=True)
    (OUT_T / "tableA_oct_sft_curve.tex").write_text("\n".join(L) + "\n")
    with open(OUT_D / "tableA_oct_sft_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(f"  wrote {OUT_T}/tableA_oct_sft_curve.tex")
    print(f"  wrote {OUT_D}/tableA_oct_sft_curve.csv  ({len(flat)} rows)")


if __name__ == "__main__":
    main()
