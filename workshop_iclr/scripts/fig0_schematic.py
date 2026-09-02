#!/usr/bin/env python3
"""Figure 0 -- the measurement pipeline, in one strip.

Four stages, left to right: the perturbations, the extraction, the cell array they fill,
and the statistics computed from it. The point of drawing it is that every result in the
paper is one of the right-hand entries applied to the SAME array, so a reader can see what
each statistic keeps and what it discards -- the persona-common shift averages over p,
dispersion centres it away, the RDM keeps only anonymous pairwise distances, and the
three-way interaction is exactly what all of them throw out.

No numbers here; this is a legend for the pipeline, not a result.

Geometry note, because matplotlib schematics overflow for a predictable reason. Everything
is placed in INCHES on a known canvas and converted to figure fractions at the last step,
so a box's width and a line of body text are measured in the same units and the text can be
made to fit. Sizing boxes in arbitrary data units on a non-square axis is what breaks.
`savefig.bbox` is forced off: a tight bbox crops to the artists in the axes, and there is
no axes here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import FULL, INK, MUTED, use_style, save

H = 1.85              # canvas height, inches
BODY_PT = 5.9
LINESPACING = 1.35
LINE_IN = BODY_PT * LINESPACING / 72        # one body line, inches
TITLE_DROP, BODY_DROP = 0.10, 0.30          # inches below the box top
TRAINED_FACE, UNTRAINED_FACE, PLAIN = "#e3edf8", "#eeeeea", "#fbfbfa"
EDGE, EDGE_U = "#aebfd2", "#c4c4be"


def need(n_lines: int) -> float:
    """Box height, in inches, that n body lines actually require."""
    return BODY_DROP + n_lines * LINE_IN + 0.06


def box(fig, x, y, w, h, title, lines, face=PLAIN, edge=EDGE, align="center"):
    """x, y, w, h in INCHES from the bottom-left of the canvas."""
    fig.patches.append(FancyBboxPatch(
        (x / FULL, y / H), w / FULL, h / H, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.010", facecolor=face, edgecolor=edge,
        linewidth=0.7, zorder=1))
    fig.text((x + w / 2) / FULL, (y + h - TITLE_DROP) / H, title, ha="center",
             va="baseline", fontsize=6.8, color=INK, fontweight="bold", zorder=2)
    tx = (x + 0.07) / FULL if align == "left" else (x + w / 2) / FULL
    fig.text(tx, (y + h - BODY_DROP) / H, "\n".join(lines), ha=align, va="top",
             fontsize=BODY_PT, color=MUTED, linespacing=LINESPACING, zorder=2)


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch(
        (x0 / FULL, y / H), (x1 / FULL, y / H), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=6, linewidth=0.7, color=MUTED, zorder=3))


def main() -> None:
    use_style()
    mpl.rcParams["savefig.bbox"] = None
    fig = plt.figure(figsize=(FULL, H))

    pert = ["Llama-3.1-8B-It", "+ OCT LoRA, r=64", "4 constitutions", "× 4 dose rungs"]
    ctrl = ["untrained LoRA:", "iid / spectral /", "permuted, sited", "by MEASURED dose"]
    caa = ["8 traits ×", "10 personas", "~500 A/B items", "activation at the",
           "answer token", "layers 15, 20"]
    vec = ["$V_{c,t,p}=$ mean(pos)", "$-$ mean(neg)", "", "$dV_{c,t,p}=$",
           "$V_{c,t,p}-V^{\\mathrm{base}}_{t,p}$"]
    stat = ["$dG_{c,t}=\\,$mean$_p\\,dV$", "    fig 1  size,  fig 5  sign", "",
            "dispersion, RDM", "    fig 3  dose,  fig 4  control", "",
            "$\\mu$+C+T+P+CT+CP+TP+CTP", "    fig 2", "",
            "cross-fitted on disjoint", "question halves; question",
            "bootstrap for all intervals"]

    box(fig, 0.04, H - 0.06 - need(4), 1.10, need(4), "perturbation", pert,
        face=TRAINED_FACE)
    box(fig, 0.04, 0.06, 1.10, need(4), "matched control", ctrl,
        face=UNTRAINED_FACE, edge=EDGE_U)
    box(fig, 1.34, (H - need(6)) / 2, 1.15, need(6), "CAA extraction", caa)
    box(fig, 2.73, (H - need(6)) / 2, 1.10, need(6), "trait vectors", vec)
    box(fig, 4.07, (H - need(12)) / 2, 1.40, need(12), "statistics", stat, align="left")

    ymid = H / 2
    arrow(fig, 1.16, 1.32, H - 0.06 - need(4) / 2)
    arrow(fig, 1.16, 1.32, 0.06 + need(4) / 2)
    arrow(fig, 2.51, 2.71, ymid)
    arrow(fig, 3.85, 4.05, ymid)

    save(fig, "fig0_schematic")


if __name__ == "__main__":
    main()
