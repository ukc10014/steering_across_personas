#!/usr/bin/env python3
"""Appendix asset 4 -- the primary result: do the stage differences survive dose matching?

The question a reader should answer by eye: at the SAME amount of functional movement, do
the two constructions still differ?

  A  M_D vs M_S,          registered impulsivity contrast B1
  B  M_D vs M_S,          target/other selectivity
  C  M_D+0.25S vs M_F,    B1
  D  M_D+0.25S vs M_F,    selectivity

PANEL D ENDPOINT, and why it is not the cosine. The brief preferred cos(dG, dG_MF). That is
DEGENERATE for pair 2: M_F's own trained state IS the reference direction, so its cosine is
pinned at 1.000 by construction and the curve reports the definition, not the data.
Selectivity separates cleanly over the same support with no such artifact. The choice is on
interpretability, not on which looked more dramatic; the cosine is carried in the master CSV.

x is ALWAYS measured functional dose. Curves are drawn only across each pair's overlapping
measured support -- the shaded band -- and points outside it are drawn faint to show the
measurements exist without implying the comparison extends there. Trained s=1 states are
ringed. Nothing is extrapolated.

Regenerate:  python scripts/appendix_oct/figA_oct_matched_dose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workshop_iclr" / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
from figstyle import INK, MUTED, use_style, despine, save, write_source_data   # noqa: E402
from dm_common import load, curve, overlap                                     # noqa: E402

COLOR = {"M_D": "#2a78d6", "M_S": "#b5179e", "M_D+0.25S": "#5aa9e6", "M_F": "#111111"}
MARK = {"M_D": "o", "M_S": "v", "M_D+0.25S": "s", "M_F": "D"}
LAB = {"M_D": r"$M_D$", "M_S": r"$M_S$", "M_D+0.25S": r"$M_{D+0.25S}$", "M_F": r"$M_F$"}
PANELS = [(("M_D", "M_S"), "B1", "registered $B_1$\n(impulsivity vs other seven)"),
          (("M_D", "M_S"), "selectivity", "target/other selectivity"),
          (("M_D+0.25S", "M_F"), "B1", "registered $B_1$\n(impulsivity vs other seven)"),
          (("M_D+0.25S", "M_F"), "selectivity", "target/other selectivity")]


def main() -> None:
    use_style()
    rows = load()
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.46})
    src = []

    for ax, ((a, b), key, ylab) in zip(axes.ravel(), PANELS):
        lo, hi = overlap(rows, a, b, key)
        ax.axvspan(lo, hi, color="#eceff0", zorder=0, lw=0)
        for st in (a, b):
            pts = curve(rows, st, key)
            d = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
            inside = (d >= lo - 1e-9) & (d <= hi + 1e-9)
            # faint outside the overlap: measured, but not part of the comparison
            ax.plot(d, v, color=COLOR[st], lw=0.7, alpha=0.30, zorder=2)
            ax.plot(d[inside], v[inside], color=COLOR[st], lw=1.6, zorder=3, label=LAB[st])
            ax.plot(d, v, ls="none", marker=MARK[st], ms=4.0, mew=0,
                    color=COLOR[st], alpha=0.35, zorder=3)
            ax.plot(d[inside], v[inside], ls="none", marker=MARK[st], ms=4.6, mew=0,
                    color=COLOR[st], zorder=4)
            for r in rows:                      # ring the trained state
                if r["state"] == st and r["is_trained_state"] and r.get(key) is not None:
                    ax.plot([r["dose"]], [r[key]], marker=MARK[st], ms=9, mfc="none",
                            mec=COLOR[st], mew=1.2, zorder=5)
            for dd, vv in zip(d, v):
                # uniform schema across panels: the endpoint is a column, not a key
                src.append({"panel": f"{a} vs {b}", "endpoint": key, "state": st,
                            "dose": round(float(dd), 4), "value": round(float(vv), 4),
                            "in_overlap": "yes" if lo <= dd <= hi else "no"})
        ax.set_ylabel(ylab)
        ax.set_xlabel("measured functional dose")
        ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.1,
                  fontsize=6.8)
        despine(ax, grid_axis="y")

    for ax, t in zip(axes.ravel(), ("A", "B", "C", "D")):
        ax.set_title(t, loc="left", fontsize=8, fontweight="bold")

    save(fig, "figA_oct_matched_dose")
    write_source_data("figA_oct_matched_dose", src,
                      ["panel", "endpoint", "state", "dose", "value", "in_overlap"])


if __name__ == "__main__":
    main()
