#!/usr/bin/env python3
"""Figure 1 -- where the constitutional intervention goes.

A. What fraction of an adapter's effect on trait geometry is a single persona-common
   translation? One dot per trait, so the reader sees the spread rather than a bar
   that hides it, with the trait-mean marked.
B. How big is that common shift for each constitution x trait, in units of the base
   trait vector it displaces? This panel is figure 2's C x T term resolved into traits:
   the partition says 17.0% of the change is constitution-dependent-and-trait-dependent,
   and this is what that term is made of. This is the panel that carries the paper's one
   content-specific result: `impulsiveness` moves `risk-taking` and `impulsivity`
   1.72x as far as it moves the other six traits (1.87x at L20), where the two
   normatively flat arms sit at 0.97 and 0.95 -- no other arm is selective.

Source: outputs/analysis/common_shift.json (scripts/common_shift.py), cross-fitted over
40 question half-splits with a 200-replicate question bootstrap. Layer 15.

The share in panel A is the SQUARED ratio ||dG||^2 / mean_p||dV_p||^2, which is the one
that partitions; its square root is not a share and is not plotted. Panel B is a linear
norm ratio. The two panels are therefore in different units on purpose, and each axis
says which.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, TRAINED, TRAINED_COLOR, MARKER, LABEL, TRAITS,
                      TRAIT_LABEL, IMPULSIVE_TRAITS, MUTED, INK, use_style, despine,
                      save, write_source_data)

LAYER = "15"


def main() -> None:
    use_style()
    d = json.load(open(ANALYSIS / "common_shift.json"))[LAYER]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL, 2.35), gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.30})

    # ---- A: common-shift share, one dot per trait -------------------------------
    rows = []
    for i, arm in enumerate(TRAINED):
        vals = np.array([d[t]["per_arm"][arm]["share_squared"] for t in TRAITS])
        jitter = np.linspace(-0.17, 0.17, len(TRAITS))
        axA.scatter(i + jitter, vals, s=7, color=TRAINED_COLOR[arm], alpha=0.55,
                    linewidths=0, zorder=3)
        axA.plot([i - 0.3, i + 0.3], [vals.mean()] * 2, color=TRAINED_COLOR[arm],
                 lw=1.6, zorder=4, solid_capstyle="butt")
        axA.annotate(f"{vals.mean():.2f}", (i, vals.mean()), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.5, color=INK)
        for t in TRAITS:
            rows.append({"panel": "A", "constitution": arm, "trait": t,
                         "value": round(d[t]["per_arm"][arm]["share_squared"], 4),
                         "ci_lo": round(d[t]["share_squared_ci"][arm][0], 4),
                         "ci_hi": round(d[t]["share_squared_ci"][arm][1], 4)})

    axA.set_xticks(range(len(TRAINED)))
    axA.set_xticklabels([LABEL[a] for a in TRAINED], rotation=32, ha="right")
    axA.set_ylim(0.45, 0.95)
    axA.set_ylabel(r"$\|dG\|^2\,/\,\mathrm{mean}_p\|dV_p\|^2$")
    axA.set_title("A  common share", loc="left")
    despine(axA)

    # ---- B: trait-resolved magnitude of the common shift ------------------------
    x = np.arange(len(TRAITS))
    off = np.linspace(-0.26, 0.26, len(TRAINED))
    for i, arm in enumerate(TRAINED):
        v = np.array([d[t]["per_arm"][arm]["g_over_base"] for t in TRAITS])
        ci = np.array([d[t]["g_over_base_ci"][arm] for t in TRAITS])
        axB.errorbar(x + off[i], v, yerr=[v - ci[:, 0], ci[:, 1] - v],
                     fmt=MARKER[arm], ms=3.4, mfc=TRAINED_COLOR[arm], mew=0,
                     ecolor=TRAINED_COLOR[arm], elinewidth=0.8, capsize=0,
                     label=LABEL[arm], zorder=3)
        for t, val, c in zip(TRAITS, v, ci):
            rows.append({"panel": "B", "constitution": arm, "trait": t,
                         "value": round(val, 4), "ci_lo": round(c[0], 4),
                         "ci_hi": round(c[1], 4)})

    # mark the two traits the impulsiveness constitution is about -- the order of
    # TRAITS is the canonical data order and is never rearranged to flatter this
    for t in IMPULSIVE_TRAITS:
        axB.axvspan(TRAITS.index(t) - 0.42, TRAITS.index(t) + 0.42,
                    color="#f2f2ee", zorder=0)
    axB.set_xticks(x)
    axB.set_xticklabels([TRAIT_LABEL[t] for t in TRAITS], rotation=32, ha="right")
    axB.set_ylabel(r"$\|dG_{c,t}\|\,/\,\mathrm{mean}_p\|V^{\mathrm{base}}_{t,p}\|$")
    axB.set_title("B  size of the common shift, by trait", loc="left")
    axB.set_ylim(0.44, 1.50)
    # The shaded columns are the two traits the impulsiveness constitution names; they
    # are explained in the caption rather than by in-plot text laid over the data.
    despine(axB)

    # One figure-level legend above both panels. A four-entry row is wider than either
    # panel, so anchoring it inside one of them puts it on top of the marks or the title.
    handles, labels = axB.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.995), handletextpad=0.25, columnspacing=1.4,
               borderpad=0)

    save(fig, "fig1_decomposition")
    write_source_data("fig1_decomposition", rows,
                      ["panel", "constitution", "trait", "value", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    main()
