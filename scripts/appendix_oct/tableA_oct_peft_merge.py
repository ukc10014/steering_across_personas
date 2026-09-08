#!/usr/bin/env python3
"""Appendix asset 6 -- what peft's factor merge adds, stated without a wall of LoRA algebra.

Built ONLY because the M_F vs M_D+0.25S difference survived dose matching. A table rather
than a figure: panels C/D of figA_oct_matched_dose already carry the visual, so a figure
would repeat it; what is missing is the algebra in two lines plus the matched-dose numbers.

WORDING RULE, enforced by construction: this reports "at matched dose, including the cross
terms changes B1 from X to Y". It does NOT express the cross terms as a percentage of
behaviour -- behaviour is nonlinear and no such share exists. The 61.9% figure that appears
elsewhere is a norm ratio in WEIGHT space and is labelled as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
from dm_common import load, overlap, interp_at                    # noqa: E402

OUT = REPO / "workshop_iclr" / "tables"


def main() -> None:
    rows = load()
    a, b = "M_D+0.25S", "M_F"
    lo, hi = overlap(rows, a, b)
    pad = 0.04 * (hi - lo)
    lo, hi = lo + pad, hi - pad
    anchors = [round(lo + i * (hi - lo) / 2, 3) for i in range(3)]

    L = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\begin{tabular}{rrrrr}", r"\toprule",
         r"matched dose & $B_1$ without & $B_1$ with & sel. without & sel. with \\",
         r" & cross terms & cross terms & cross terms & cross terms \\", r"\midrule"]
    for t in anchors:
        b1a = interp_at(rows, a, "B1", t)[0]; b1b = interp_at(rows, b, "B1", t)[0]
        sa = interp_at(rows, a, "selectivity", t)[0]; sb = interp_at(rows, b, "selectivity", t)[0]
        L.append(rf"{t:.3f} & {b1a:+.3f} & \textbf{{{b1b:+.3f}}} & {sa:.3f} & \textbf{{{sb:.3f}}} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The released-style merge is not the additive update it is often read as. "
          r"peft's \texttt{add\_weighted\_adapter([DPO, SFT], [1, 0.25], \texttt{linear})} "
          r"combines LoRA \emph{factors}, so "
          r"$\Delta W_F = \Delta W_D + 0.25\,\Delta W_S + B_D A_S + B_S A_D$; "
          r"$M_{D+0.25S}$ is the same construction with the two cross terms removed and is "
          r"otherwise identical. At matched \emph{measured} functional dose, including the "
          r"cross terms changes $B_1$ from $+0.51$ to $+0.93$ at dose $0.578$ and from "
          r"$+0.97$ to $+2.00$ at dose $0.842$. We do not report a share of behaviour "
          r"attributable to the cross terms: behaviour is nonlinear in the weight update, so "
          r"no such share is defined. (The $61.9\%$ figure quoted elsewhere is a Frobenius "
          r"norm ratio in weight space, not a behavioural one.)}",
          r"\label{tab:oct-peft-merge}", r"\end{table}"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tableA_oct_peft_merge.tex").write_text("\n".join(L) + "\n")
    print(f"  wrote {OUT}/tableA_oct_peft_merge.tex")
    for t in anchors:
        print(f"    dose {t:.3f}: B1 {interp_at(rows,a,'B1',t)[0]:+.3f} -> "
              f"{interp_at(rows,b,'B1',t)[0]:+.3f}")


if __name__ == "__main__":
    main()
