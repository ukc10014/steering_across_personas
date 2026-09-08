#!/usr/bin/env python3
"""Appendix asset 2 -- the raw (un-dose-matched) stage decomposition, as a booktabs table.

Scientific question: what did the naive stage decomposition appear to show? This table
preserves that result historically. It is deliberately NOT the headline: the raw ordering is
perfectly rank-confounded with measured functional dose (Spearman = +1.000), and the caption
says so.

Every number is regenerated from the archived analysis outputs; nothing is transcribed.
  outputs/analysis/stage_comparison_seed1.csv   (scripts/stage_comparison.py)
Regenerate:
  python scripts/appendix_oct/tableA_oct_stage_raw.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "analysis" / "stage_comparison_seed1.csv"
OUT = REPO / "iclr2026" / "tables"

DESC = {
    "M_D":        (r"$M_D$",          "base $+$ DPO update"),
    "M_D+0.25S":  (r"$M_{D+0.25S}$",  r"$\Delta W_{D}+0.25\,\Delta W_{S}$, no cross terms"),
    "M_D+S":      (r"$M_{D+S}$",      "SFT update on the folded DPO state"),
    "M_F":        (r"$M_F$",          "released-style PEFT weighted merge"),
    "M_S":        (r"$M_S$",          "SFT trained \\emph{from base} on the same corpus"),
    "M_0+A_S*":   (r"$M_0{+}A_S$\dag", "existing SFT adapter applied off-base"),
    "released":   ("released",        "the OCT-published adapter"),
}
ORDER = ["M_D", "M_D+0.25S", "M_D+S", "M_F", "M_S", "M_0+A_S*", "released"]
FMT = {"B1": "{:+.3f}", "B2": "{:+.3f}", "k": "{:.3f}",
       "sel": "{:.3f}", "dose": "{:.3f}", "cos_to_M_F": "{:.3f}"}


def main() -> None:
    rows = {r["state"]: r for r in csv.DictReader(open(SRC))}

    # the confound, recomputed here so the caption's number cannot drift from the data
    d, b = [], []
    for st, r in rows.items():
        if st in ("released", "M_0") or not r.get("dose") or not r.get("B1"):
            continue
        d.append(float(r["dose"])); b.append(float(r["B1"]))
    rho = spearmanr(d, b).statistic

    L = []
    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\begin{tabular}{llrrrrrr}")
    L.append(r"\toprule")
    L.append(r"state & construction & $B_1$ & $B_2$ & sel. & $k$ & dose & $\cos\!\to\! M_F$ \\")
    L.append(r"\midrule")
    for st in ORDER:
        r = rows.get(st)
        if r is None:
            continue
        sym, desc = DESC[st]
        cells = [sym, desc]
        for c in ("B1", "B2", "sel", "k", "dose", "cos_to_M_F"):
            v = r.get(c, "")
            cells.append(FMT[c].format(float(v)) if v else "--")
        L.append(" & ".join(cells) + r" \\")
        if st == "M_S":
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(
        r"\caption{Raw decomposition of the OCT \texttt{impulsiveness} pipeline, seed 123456, "
        r"layer 15, forced prompt. $B_1$ is the registered \texttt{impulsivity} contrast against "
        r"the other seven traits; $B_2$ adds \texttt{risk\_taking}; sel.\ is the target/other "
        r"common-shift ratio; $k$ is retention; dose is \emph{measured} functional dose "
        r"(trait-vector displacement), never weight norm. "
        r"\textbf{This is not a dose-matched causal comparison.} Across the measured states the "
        rf"ordering of $B_1$ is perfectly rank-correlated with dose (Spearman $={rho:+.3f}$), so "
        r"stage and displacement are not separated here; see Fig.~\ref{{fig:oct-matched-dose}}. "
        r"\dag~off-base diagnostic: that adapter was fitted on the folded DPO state, so applying "
        r"it to the base model is a component measurement, not ``SFT alone''.}")
    L.append(r"\label{tab:oct-stage-raw}")
    L.append(r"\end{table}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tableA_oct_stage_raw.tex").write_text("\n".join(L) + "\n")
    print(f"  wrote {OUT}/tableA_oct_stage_raw.tex")
    print(f"  Spearman(dose, B1) recomputed from source = {rho:+.4f}  (over {len(d)} states)")


if __name__ == "__main__":
    main()
