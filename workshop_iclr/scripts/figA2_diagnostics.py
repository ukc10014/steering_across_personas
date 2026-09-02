#!/usr/bin/env python3
"""Appendix A2 -- two diagnostics behind the main results.

A. The per-cell C x T x P interaction, against the reference that actually discriminates.
   An earlier version of this analysis tested each cell against exact zero and reported
   "319 of 320 cells significant". That number is vacuous -- a per-cell value is a squared
   magnitude estimated from ~500 questions, so "greater than zero" is true of nearly every
   cell and separates nothing. It is not plotted here. The reference that does discriminate
   is the untrained band's own per-cell distribution, computed on the same question splits:
   the trained distribution sits below it at EVERY quantile, by roughly a factor of two at
   the median, and only 4 of 320 trained cells clear the untrained p95 where 16 is what two
   matching distributions would give. The band-level result of figure 2B is not a tail
   effect.

B. How much of the centred per-cell change one global map explains, cross-validated on
   held-out traits. A general linear map removes about twice the squared error an
   orthogonal (Procrustes) one does, so the arm difference is not a rotation. The three
   untrained arms at matched dose are not below the trained family on either -- they are at
   the top of both -- so "a large part of the change is one global linear transformation"
   is a property of large perturbations, not of character training.

Only the matched-dose arms appear in B. Below dose ~0.75 the cross-validated scores go
NEGATIVE for every arm (the fitted map generalises worse than no map at all, because the
change being fitted is small next to the estimation noise); those rungs are in the source
CSV and are not informative as bars.

Sources: outputs/analysis/three_way_interaction.json,
         outputs/llama-3.1-8b-goodness/analysis/geometry_L15.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, GEOM, FULL, COLOR, LABEL, TRAINED, UNTRAINED, INK, MUTED,
                      use_style, despine, save, write_source_data)

LAYER = "15"
QUANTS = [10, 25, 50, 75, 90, 95]


def main() -> None:
    use_style()
    tw = json.load(open(ANALYSIS / f"three_way_interaction.json"))[LAYER]
    g = json.load(open(GEOM / f"geometry_L{LAYER}.json"))
    rows = []

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL, 2.2),
                                   gridspec_kw={"width_ratios": [1, 1.15],
                                                "wspace": 0.30})

    # ---- A: per-cell CTP, trained cells vs the untrained quantile curve ----------
    cells = np.sort(np.array([c["ss"] for c in tw["cells"].values()]))
    ecdf = np.arange(1, len(cells) + 1) / len(cells)
    axA.step(cells, ecdf, where="post", color=INK, lw=1.3, zorder=4,
             label=f"trained cells (n={len(cells)})")
    q = tw["cell_reference"]["quantiles"]
    axA.plot([q["null"][f"p{p}"] for p in QUANTS], [p / 100 for p in QUANTS],
             color="#6f6f69", lw=1.2, ls=(0, (3, 1.6)), marker="s", ms=3.2, mew=0,
             zorder=3, label="untrained quantiles")
    p95 = tw["cell_reference"]["null_p95"]
    axA.axvline(p95, color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=2)
    # right-aligned onto the p95 line: the full sentence runs off the axis
    axA.text(p95 * 0.96, 0.44,
             f"untrained p95\n{tw['cell_reference']['n_exceed']}/"
             f"{tw['cell_reference']['n_cells']} trained above",
             fontsize=5.8, color=MUTED, va="center", ha="right", linespacing=1.25)
    for p in QUANTS:
        rows.append({"panel": "A", "arm": "", "quantile": f"p{p}",
                     "trained": round(q["trained"][f"p{p}"], 4),
                     "untrained": round(q["null"][f"p{p}"], 4), "measure": "per_cell_CTP"})
    axA.set_xlim(0, 0.16)
    axA.set_ylim(0, 1.02)
    axA.set_xlabel("per-cell C$\\times$T$\\times$P magnitude\n($\\div$ base trait vector)")
    axA.set_ylabel("cumulative fraction of cells")
    axA.set_title("A  per-cell interaction", loc="left")
    axA.legend(loc="lower right", handletextpad=0.4, labelspacing=0.25, borderpad=0)
    despine(axA)

    # ---- B: orthogonal vs general linear map ------------------------------------
    arms = TRAINED + UNTRAINED
    x = np.arange(len(arms))
    proc = [g["procrustes"][a]["hold-out traits"]["squared_error_removed"] for a in arms]
    lin = [g["linear_map"][a]["squared_error_removed"] for a in arms]
    axB.bar(x - 0.19, proc, width=0.36, color="#c9d3dd", zorder=3,
            label="orthogonal")
    axB.bar(x + 0.19, lin, width=0.36, color=[COLOR[a] for a in arms], zorder=3,
            label="general linear")
    for xx, a in zip(x, arms):
        rows.append({"panel": "B", "arm": a, "quantile": "",
                     "trained": round(g["procrustes"][a]["hold-out traits"]
                                      ["squared_error_removed"], 4),
                     "untrained": round(g["linear_map"][a]["squared_error_removed"], 4),
                     "measure": "squared_error_removed (proc, linear)"})
    # shading marks the untrained arms; named in the caption, not over the bars
    axB.axvspan(3.5, 6.5, color="#f2f2ee", zorder=0)
    axB.set_xticks(x)
    axB.set_xticklabels([LABEL[a] for a in arms], rotation=32, ha="right")
    axB.set_ylim(0, 0.66)
    axB.set_ylabel("squared error removed (held-out traits)")
    axB.set_title("B  one global map", loc="left")
    axB.legend(loc="upper left", handletextpad=0.4, labelspacing=0.25, borderpad=0,
               ncol=1)
    despine(axB)

    save(fig, "figA2_diagnostics")
    write_source_data("figA2_diagnostics", rows,
                      ["panel", "arm", "quantile", "trained", "untrained", "measure"])


if __name__ == "__main__":
    main()
