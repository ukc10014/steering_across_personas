#!/usr/bin/env python3
"""Appendix A5b -- the two-panel cut of A5: `impulsiveness` against `random_perm`.

Same figure as figA5_signed_validity.py with the `goodness` and `misalignment` panels
dropped. The contrast the appendix actually turns on is the trained arm against the
untrained one, and at four panels each is 1.4in wide; at two they are 2.75in, which is
legible at workshop-paper size. The four-panel version is kept -- it is the one that
supports "only 4 of 24 trained cells pass", a claim this cut cannot make.

NOTHING ELSE CHANGES. Same source JSON, same LAYER, same KEY, same rung lists, same dose
normalisation, same validity arithmetic, same colours, weights and dashes, same y-limits,
same zero line, same legend. Only the panel set and the figure geometry differ, and
scripts/verify_figA5_two_panel.py asserts the plotted curves are identical to the
four-panel figure's, element by element.

Sources -- identical to figA5_signed_validity.py:
  outputs/analysis/signed_trait_shift_ladder.json
  outputs/analysis/functional_dose_with_random.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, COLOR, LABEL, TRAITS,
                      IMPULSIVE_TRAITS, INK, MUTED, use_style, despine, save,
                      write_source_data)

LAYER = "15"
KEY = "proj_common"
# The two retained panels, rung lists copied verbatim from figA5_signed_validity.py.
PANELS = {
    "impulsiveness": ["impulsiveness_s0.25", "impulsiveness_s0.5",
                      "impulsiveness_s0.75", "impulsiveness"],
    "random_perm_s16": ["random_perm_s8", "random_perm_s12", "random_perm_s16"],
}


def curves():
    """(dose xs, {trait: proj array}, n_pass) per panel -- the plotted numbers, nothing else.

    Factored out so the verification script can compare these against the four-panel
    figure's without re-implementing the arithmetic.
    """
    d = json.load(open(ANALYSIS / "signed_trait_shift_ladder.json"))[LAYER]
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[LAYER]
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}

    out = {}
    for arm, rungs in PANELS.items():
        xs = np.array([dose[a] for a in rungs])
        vals, n_pass, checks = {}, 0, {}
        for t in TRAITS:
            v = np.array([d[t]["per_arm"][a][KEY] for a in rungs])
            same = bool(np.all(np.sign(v) == np.sign(v[-1])) and abs(v[-1]) > 1e-9)
            rho = float(spearmanr(xs, np.abs(v)).statistic)
            b1, b0 = np.polyfit(xs, v, 1)
            ok = same and rho >= 0.8 and abs(b0) / max(abs(v[-1]), 1e-9) <= 0.5
            n_pass += ok
            vals[t] = v
            checks[t] = (same, rho, ok)
        out[arm] = (xs, vals, n_pass, checks)
    return out


def main() -> None:
    use_style()
    data = curves()

    # Two panels across the same FULL width -> 2.75in each, double the four-panel version.
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    rows = []

    for ax, (arm, (xs, vals, n_pass, checks)) in zip(axes, data.items()):
        for t in TRAITS:
            v = vals[t]
            same, rho, ok = checks[t]
            hot = t in IMPULSIVE_TRAITS
            ax.plot(xs, v, color=COLOR[arm], lw=1.8 if hot else 1.0,
                    ls="-" if t != "impulsivity" else (0, (2.6, 1.2)),
                    alpha=1.0 if hot else 0.42, marker="o", ms=3.0, mew=0, zorder=3)
            rows.append({"arm": arm, "trait": t, "dose_hi": round(xs[-1], 4),
                         "proj_at_s1": round(float(v[-1]), 4),
                         "sign_stable": "yes" if same else "no",
                         "rho_dose": round(rho, 2),
                         "passes": "yes" if ok else "no"})
        ax.axhline(0, color=MUTED, lw=0.7, zorder=2)
        ax.set_title(LABEL[arm], loc="left", fontsize=8.5)
        ax.text(0.03, 0.03, f"{n_pass}/8 valid", transform=ax.transAxes, fontsize=7.2,
                color=INK, fontweight="bold", va="bottom", ha="left")
        ax.set_xlim(0.38, 1.22)
        ax.set_xticks([0.5, 1.0])
        despine(ax, grid_axis=None)

    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=INK, lw=1.8, label="risk-taking"),
           Line2D([], [], color=INK, lw=1.8, ls=(0, (2.6, 1.2)), label="impulsivity"),
           Line2D([], [], color=INK, lw=1.0, alpha=0.42, label="the other six traits")]
    fig.legend(handles=key, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.995),
               handletextpad=0.4, columnspacing=1.4, borderpad=0)

    axes[0].set_ylim(-0.62, 0.32)          # identical to the four-panel figure
    axes[0].set_ylabel("signed projection on\nthe base trait axis")
    fig.supxlabel("measured functional dose ($\\div$ goodness $s{=}1$)", fontsize=8.5,
                  y=-0.02)

    save(fig, "figA5b_signed_validity_2panel")
    write_source_data("figA5b_signed_validity_2panel", rows,
                      ["arm", "trait", "dose_hi", "proj_at_s1", "sign_stable",
                       "rho_dose", "passes"])


if __name__ == "__main__":
    main()
