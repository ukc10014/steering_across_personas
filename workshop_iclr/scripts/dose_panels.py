#!/usr/bin/env python3
"""Shared builder for the dose-response + matched-control figure.

Figure 3 (layer 15) and appendix A3 (layer 20) are the same four panels on different
layers, so they are one function rather than two scripts that drift apart.

  top row     the two geometry statistics against MEASURED functional dose
  bottom row  the same statistics at matched dose 1.00, trained against untrained

Both rows are needed together and neither is a figure on its own: the top row establishes
that both outcomes are steep in dose, which is exactly why the bottom row has to correct
to a common dose before any arm is compared to any other.

EVERY POINT IN THE TOP ROW IS MEASURED. Lines connect adjacent measured rungs and are
linear interpolation between them; the interpolated matched-dose grids of sections 6.3-6.5
are not plotted as though they were observations. An arm with one measured dose point
(`mathematical`) is drawn as a lone marker with no line.

THE BOTTOM ROW CORRECTS EVERY ARM, which section 7.7 does not. No arm sits exactly at dose
1.000 -- measured doses run 0.987 to 1.113 -- so each point is moved to dose 1.000 along
the local slope of its own measured ladder, with single-point arms borrowing a slope (the
random arms take `random_perm`'s, as 7.7 does; `mathematical` takes the mean of the three
trained ladders). Section 7.7 corrected the random arms but quoted the trained arms where
they were measured, at dose 1.07-1.08; correcting consistently raises `misalignment` from
0.732 to 0.768, so it ties `random_perm` rather than sitting lowest. The claim that
survives -- untrained arms span at least as much as the trained family -- is unchanged.
The measured value is drawn beside the corrected one so the size of every correction is
visible. Never mix the two conventions in one panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, GEOM, FULL, COLOR, MARKER, LABEL, DASH, INK, MUTED,
                      TRAINED, use_style, despine, save, write_source_data)

LADDERS = {
    "goodness": ["goodness_s0.25", "goodness_s0.5", "goodness_s0.75", "goodness"],
    "impulsiveness": ["impulsiveness_s0.25", "impulsiveness_s0.5",
                      "impulsiveness_s0.75", "impulsiveness"],
    "misalignment": ["misalignment_s0.25", "misalignment_s0.5",
                     "misalignment_s0.75", "misalignment"],
    "random_perm_s16": ["random_perm_s8", "random_perm_s12", "random_perm_s16"],
}
SINGLE = ["mathematical"]
SLOPE_FROM = {
    "goodness": ("goodness_s0.75", "goodness"),
    "impulsiveness": ("impulsiveness_s0.75", "impulsiveness"),
    "misalignment": ("misalignment_s0.75", "misalignment"),
    "random_perm_s16": ("random_perm_s12", "random_perm_s16"),
}


def build(layer: str, geom_file: str, rows_arms: list[str], outname: str,
          xlims: tuple | None = None) -> None:
    use_style()
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[layer]
    path = GEOM / geom_file
    if not path.exists():
        raise FileNotFoundError(str(path))
    g = json.load(open(path))
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}
    rows = []

    stats = [
        ("RDM preservation", "Spearman $\\rho$ vs base",
         lambda a: (g["rdm_aggregate"][a]["boot_mean"], g["rdm_aggregate"][a]["boot_ci"])),
        ("persona dispersion", "RMS persona spread $\\div$ base",
         lambda a: (g["dispersion_aggregate"][a]["mean_rms_ratio"],
                    g["dispersion_aggregate"][a]["rms_ci"])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.25),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.66,
                                          "height_ratios": [1.05, 1]})

    # ---- top row: measured dose response ----------------------------------------
    for ax, (name, ylab, get), tag in zip(axes[0], stats, "AB"):
        for series, rungs in LADDERS.items():
            rungs = [a for a in rungs if a in dose and a in g["rdm_aggregate"]]
            if len(rungs) < 2:
                continue
            xs = np.array([dose[a] for a in rungs])
            vals = np.array([get(a)[0] for a in rungs])
            cis = np.array([get(a)[1] for a in rungs])
            ax.plot(xs, vals, ls=DASH[series], color=COLOR[series], lw=1.2,
                    marker=MARKER[series], ms=3.4, mew=0, zorder=3,
                    label=LABEL[series] if tag == "A" else None)
            ax.errorbar(xs, vals, yerr=[vals - cis[:, 0], cis[:, 1] - vals], fmt="none",
                        ecolor=COLOR[series], elinewidth=0.8, zorder=2)
            for a, x, v, c in zip(rungs, xs, vals, cis):
                rows.append({"panel": tag, "arm": a, "measure": name, "corrected": "no",
                             "dose": round(x, 4), "value": round(v, 4),
                             "ci_lo": round(c[0], 4), "ci_hi": round(c[1], 4)})
        for a in SINGLE:
            v, c = get(a)
            ax.errorbar([dose[a]], [v], yerr=[[v - c[0]], [c[1] - v]], fmt=MARKER[a],
                        ms=4.0, mfc=COLOR[a], mew=0, ecolor=COLOR[a], elinewidth=0.8,
                        zorder=4, label=LABEL[a] if tag == "A" else None)
            rows.append({"panel": tag, "arm": a, "measure": name, "corrected": "no",
                         "dose": round(dose[a], 4), "value": round(v, 4),
                         "ci_lo": round(c[0], 4), "ci_hi": round(c[1], 4)})
        ax.set_title(f"{tag}  {name} vs dose", loc="left")
        ax.set_ylabel(ylab)
        ax.set_xlabel("measured functional dose")
        despine(ax)

    # the dose coordinate's own uncertainty, shown once
    sd = np.mean([fd[a]["answer_token_displacement_sd"] / ref
                  for a in ("goodness", "impulsiveness", "misalignment")])
    ax0 = axes[0][0]
    xr, yr = ax0.get_xlim(), ax0.get_ylim()
    ax0.errorbar([xr[0] + 0.20 * (xr[1] - xr[0])], [yr[0] + 0.07 * (yr[1] - yr[0])],
                 xerr=[sd], fmt="none", ecolor=MUTED, elinewidth=0.9, capsize=1.8,
                 zorder=5)
    ax0.text(xr[0] + 0.20 * (xr[1] - xr[0]), yr[0] + 0.10 * (yr[1] - yr[0]),
             f"typical dose SD $\\pm${sd:.2f}", fontsize=5.8, color=MUTED, ha="center",
             va="bottom")

    # ---- bottom row: matched dose 1.00 ------------------------------------------
    y = np.arange(len(rows_arms))[::-1]
    n_tr = sum(1 for a in rows_arms if a in TRAINED)
    for ax, (name, xlab, get), tag in zip(axes[1], stats, "CD"):
        slopes = {a: (get(hi)[0] - get(lo)[0]) / (dose[hi] - dose[lo])
                  for a, (lo, hi) in SLOPE_FROM.items()}
        tmean = np.mean([slopes[a] for a in
                         ("goodness", "impulsiveness", "misalignment")])
        for a in rows_arms:
            if a not in slopes:
                slopes[a] = tmean if a in TRAINED else slopes["random_perm_s16"]
        span = {}
        for arm, yy in zip(rows_arms, y):
            v, ci = get(arm)
            shift = -slopes[arm] * (dose[arm] - 1.0)
            vc, cic = v + shift, [ci[0] + shift, ci[1] + shift]
            span.setdefault("trained" if arm in TRAINED else "untrained", []).append(vc)
            ax.plot(cic, [yy] * 2, color=COLOR[arm], lw=0.9, zorder=3,
                    solid_capstyle="butt")
            ax.plot([v], [yy], marker="|", ms=6, mec=MUTED, mew=0.9, ls="none", zorder=4,
                    label="as measured" if (arm == rows_arms[0] and tag == "C") else None)
            ax.plot([v, vc], [yy, yy], color=MUTED, lw=0.5, zorder=2)
            ax.plot([vc], [yy], marker=MARKER[arm], ms=4.4, mfc=COLOR[arm], mec="white",
                    mew=0.5, ls="none", zorder=5)
            if arm == rows_arms[0] and tag == "C":
                ax.plot([], [], marker="o", ms=4.4, mfc=MUTED, mec="white", mew=0.5,
                        ls="none", label="corrected to dose 1.00")
            ax.text(vc, yy + 0.30, f"{vc:.3f}", fontsize=6.2, color=INK, ha="center",
                    va="bottom")
            rows.append({"panel": tag, "arm": arm, "measure": name, "corrected": "yes",
                         "dose": round(dose[arm], 4), "value": round(vc, 4),
                         "ci_lo": round(cic[0], 4), "ci_hi": round(cic[1], 4)})
        for i, (k, col) in enumerate([("trained", INK), ("untrained", "#6f6f69")]):
            lo, hi = min(span[k]), max(span[k])
            yy = -1.20 - 0.80 * i
            ax.plot([lo, hi], [yy] * 2, color=col, lw=1.4, solid_capstyle="butt", zorder=4)
            ax.text((lo + hi) / 2, yy + 0.22, f"{k} span {hi - lo:.3f}", fontsize=6,
                    color=col, va="bottom", ha="center")
        ax.axhspan(-0.6, len(rows_arms) - n_tr - 0.4, color="#f5f5f2", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[a] for a in rows_arms] if tag == "C" else [],
                           fontsize=6.6)
        ax.set_ylim(-2.9, len(rows_arms) - 0.35)
        ax.set_xlabel(f"{xlab}, at dose 1.00", fontsize=7)
        ax.set_title(f"{tag}  {name} at matched dose", loc="left")
        despine(ax, grid_axis="x")
        ax.yaxis.set_tick_params(length=0)
        if xlims:
            ax.set_xlim(*xlims[0 if tag == "C" else 1])   # xlims = (RDM, dispersion)

    h1, l1 = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[1][0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.995), handletextpad=0.35, columnspacing=1.2,
               borderpad=0)

    save(fig, outname)
    write_source_data(outname, rows,
                      ["panel", "arm", "measure", "corrected", "dose", "value",
                       "ci_lo", "ci_hi"])
