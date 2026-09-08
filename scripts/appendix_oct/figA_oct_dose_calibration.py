#!/usr/bin/env python3
"""Appendix asset 3 -- nominal LoRA scale is not a usable x-axis for causal comparison.

Scientific question: can two OCT stage constructions be compared by setting them to the
same nominal LoRA scale? No, and this figure is the evidence. Three facts should be visible
without reading the caption:

  1. equal nominal scale does NOT give equal measured functional dose across states;
  2. scale -> dose is nonlinear even WITHIN a state (dose/s spans ~2x for every one);
  3. M_D+S and M_S TURN OVER at the largest measured scale -- dose falls as the
     perturbation grows, the coherence-cliff signature the random ladder also shows.

Post-turnover points are drawn hollow and are excluded from every interpolation; connecting
lines stop at each state's monotone prefix. No smooth fit is drawn: a polynomial through
these points would imply a functional form nothing here supports.

Source (archived, not recomputed):
  outputs/analysis/dose_stage_grid_analysis.json   scripts/dose_calibrate.py Phase-1 ladder
Regenerate:
  python scripts/appendix_oct/figA_oct_dose_calibration.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workshop_iclr" / "scripts"))
from figstyle import (INK, MUTED, use_style, despine,        # noqa: E402
                      save, write_source_data)
ANALYSIS = REPO / "outputs" / "analysis"
LAYER = "15"
MEASURE = "trait_vector"

# Ordered as the pipeline runs, so the legend reads like the construction sequence.
STATES = ["M_D", "M_D+0.25S", "M_D+S", "M_F", "M_S"]
LABEL = {"M_D": r"$M_D$", "M_D+0.25S": r"$M_{D+0.25S}$", "M_D+S": r"$M_{D+S}$",
         "M_F": r"$M_F$", "M_S": r"$M_S$"}
# Colour by role: DPO-only and its additive sibling cool, the SFT-bearing states warm,
# the released merge black so it reads as the reference artifact.
COLOR = {"M_D": "#2a78d6", "M_D+0.25S": "#5aa9e6", "M_D+S": "#eb6834",
         "M_F": "#111111", "M_S": "#b5179e"}
MARK = {"M_D": "o", "M_D+0.25S": "s", "M_D+S": "^", "M_F": "D", "M_S": "v"}


def curves() -> dict[str, list[tuple[float, float]]]:
    d = json.loads((ANALYSIS / "dose_stage_grid_analysis.json").read_text())["dose"][LAYER]
    out: dict[str, list[tuple[float, float]]] = {}
    for cfg, v in d.items():
        if not cfg.startswith("M_"):
            continue
        state, _, s = cfg.rpartition("_s")
        out.setdefault(state, []).append((float(s), float(v[MEASURE])))
    for k in out:
        out[k].sort()
    return out


def monotone_len(dd: np.ndarray) -> int:
    k = 1
    while k < len(dd) and dd[k] > dd[k - 1]:
        k += 1
    return k


def main() -> None:
    use_style()
    c = curves()
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    rows = []

    for st in STATES:
        pts = c[st]
        ss = np.array([s for s, _ in pts])
        dd = np.array([v for _, v in pts])
        k = monotone_len(dd)
        ax.plot(ss[:k], dd[:k], color=COLOR[st], lw=1.4, marker=MARK[st], ms=4.2,
                mew=0, label=LABEL[st], zorder=3)
        if k < len(dd):                      # post-turnover: hollow, unconnected
            ax.plot(ss[k:], dd[k:], color=COLOR[st], lw=0, marker=MARK[st], ms=4.2,
                    mfc="white", mec=COLOR[st], mew=1.0, zorder=3)
        # the trained state itself, as an anchor
        if 1.0 in ss:
            i = int(np.where(ss == 1.0)[0][0])
            ax.plot([1.0], [dd[i]], marker=MARK[st], ms=8.5, mfc="none",
                    mec=COLOR[st], mew=1.2, zorder=4)
        for j, (s, v) in enumerate(pts):
            rows.append({"state": st, "nominal_scale": s, "measured_dose": round(v, 4),
                         "layer": LAYER, "measure": MEASURE,
                         "in_monotone_region": "yes" if j < k else "no",
                         "is_trained_state": "yes" if s == 1.0 else "no"})

    # a nominal-scale reference: if dose were proportional to s it would be a ray
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(1.0, ax.get_ylim()[1], " trained state ", fontsize=6.2, color=MUTED,
            va="top", ha="left", rotation=90)

    ax.set_xlabel("nominal LoRA scale $s$")
    ax.set_ylabel("measured functional dose\n(trait-vector displacement, L15)")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0])
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              handletextpad=0.4, columnspacing=1.1, borderpad=0)
    despine(ax, grid_axis="y")

    # figstyle's helpers write to workshop_iclr/{figures,data} -- the live submission set.
    save(fig, "figA_oct_dose_calibration")
    write_source_data("figA_oct_dose_calibration", rows, list(rows[0]))

    print("\n  facts the figure must show:")
    for st in STATES:
        dd = np.array([v for _, v in c[st]]); ss = np.array([s for s, _ in c[st]])
        r = dd / ss
        k = monotone_len(dd)
        print(f"    {st:10s} dose@s=1 {dd[list(ss).index(1.0)]:.4f}  "
              f"dose/s spans {r.min():.3f}-{r.max():.3f}  "
              f"monotone rungs {k}/{len(dd)}")


if __name__ == "__main__":
    main()
