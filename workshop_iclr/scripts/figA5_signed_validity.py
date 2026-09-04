#!/usr/bin/env python3
"""Appendix A5 -- why there is no Open-Character-Training-style signed figure here.

OCT's figure 3 shows, per trained character, the traits whose revealed-preference Elo rose
and fell most. The analogue in this data would be a signed version of figure 1B: project
the persona-common shift onto the base model's own trait axis,

    proj_{c,t} = <dG_{c,t}, Vbar_t/||Vbar_t||> / mean_p||V^base_{t,p}||

which is signed (the CAA contrast is mean(pos) - mean(neg) with `pos` the higher-trait
option), is bounded by figure 1B's magnitude for the same cell, and uses an axis fixed
before any constitution is seen. The estimator is sound -- it is cross-fitted, and its
self-test recovers a known signed shift where the naive version returns the wrong sign
(`python scripts/signed_trait_shift.py --self-test`).

IT STILL DOES NOT MEAN WHAT A READER WOULD TAKE IT TO MEAN, and this figure is the test
that establishes that. A quantity that measures displacement along a fixed trait axis
should keep its sign and grow as the same adapter is scaled up. Each panel plots
proj_{c,t} against measured functional dose, one thin line per trait.

  The three trained panels are tangles. Lines cross zero, change sign between rungs, and
  are non-monotone. Only 4 of 24 trained cells pass all three pre-specified checks.
  The untrained panel is a clean fan. `random_perm` passes 8 of 8: every trait keeps its
  sign, every one is perfectly rank-correlated with dose, every intercept is small.

ONE CHECK DOES PASS, and it is recorded here rather than dropped: sign(proj) agrees
between layer 15 and layer 20 in 50 of 56 cells (89%), with 84% of L15 cells having an
interval that excludes zero. That is a real consistency, but it does not discriminate --
generic contraction is layer-consistent too, and `random_perm` is the arm most consistent
of all. A test every arm passes cannot license a reading only one family is given.

So the best-behaved "signed trait metric" in this data belongs to the arm with no learned
content at all, and what it is measuring is plain: a large perturbation contracts every
trait representation along its own axis, which reads as a uniformly negative, perfectly
dose-monotone projection. The trained arms are messier precisely because that generic
contraction is competing with whatever else they do.

THE TRAP THIS AVOIDS, stated because it is a near miss. At s=1 the `impulsiveness`
constitution scores -0.272 on `impulsivity`. Plotted as an OCT-style gain/loss chart
without this check, that reads as "the impulsiveness constitution made the model less
impulsive". `random_perm` scores -0.389 on the same trait, so the sign is generic
contraction, not content. Subtracting the control does not rescue it either: of
`impulsiveness`'s eight traits, `impulsivity` has the SMALLEST gap to `random_perm`
(+0.117) of any trait, which is the opposite of a selective semantic effect.

WHAT SURVIVES is the unsigned result. Figure 1B's selectivity -- `impulsiveness` moving
`risk-taking` and `impulsivity` 1.72x as far as the other six -- is a magnitude, and is
reported as semantic SELECTIVITY, never as direction or valence.

Sources
  outputs/analysis/signed_trait_shift_ladder.json  (scripts/signed_trait_shift.py, the
      12 constitution rungs + the 3 random_perm rungs, cross-fitted over 40 half-splits,
      point estimates only -- no bootstrap, so no intervals are drawn)
  outputs/analysis/functional_dose_with_random.json  for the dose axis
Verdict arithmetic: workshop_iclr/scripts/check_signed_validity.py, whose pass criteria
were fixed before the numbers were read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, COLOR, LABEL, TRAITS, TRAIT_LABEL,
                      IMPULSIVE_TRAITS, INK, MUTED, use_style, despine, save,
                      write_source_data)

LAYER = "15"
KEY = "proj_common"
PANELS = {
    "goodness": ["goodness_s0.25", "goodness_s0.5", "goodness_s0.75", "goodness"],
    "impulsiveness": ["impulsiveness_s0.25", "impulsiveness_s0.5",
                      "impulsiveness_s0.75", "impulsiveness"],
    "misalignment": ["misalignment_s0.25", "misalignment_s0.5",
                     "misalignment_s0.75", "misalignment"],
    "random_perm_s16": ["random_perm_s8", "random_perm_s12", "random_perm_s16"],
}


def main() -> None:
    use_style()
    src = ANALYSIS / "signed_trait_shift_ladder.json"
    if not src.exists():
        raise FileNotFoundError(str(src))
    d = json.load(open(src))[LAYER]
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[LAYER]
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}

    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.15), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    rows = []

    for ax, (arm, rungs) in zip(axes, PANELS.items()):
        xs = np.array([dose[a] for a in rungs])
        n_pass = 0
        for t in TRAITS:
            v = np.array([d[t]["per_arm"][a][KEY] for a in rungs])
            same = bool(np.all(np.sign(v) == np.sign(v[-1])) and abs(v[-1]) > 1e-9)
            rho = float(spearmanr(xs, np.abs(v)).statistic)
            b1, b0 = np.polyfit(xs, v, 1)
            ok = same and rho >= 0.8 and abs(b0) / max(abs(v[-1]), 1e-9) <= 0.5
            n_pass += ok
            # The two traits the impulsiveness constitution names are picked out by
            # WEIGHT and DASH, not by an in-panel label: a 1.2in panel cannot hold a
            # legible text label at the end of a line without clipping it.
            hot = t in IMPULSIVE_TRAITS
            ax.plot(xs, v, color=COLOR[arm], lw=1.5 if hot else 0.8,
                    ls="-" if t != "impulsivity" else (0, (2.6, 1.2)),
                    alpha=1.0 if hot else 0.42, marker="o", ms=2.4, mew=0, zorder=3)
            rows.append({"arm": arm, "trait": t, "dose_hi": round(xs[-1], 4),
                         "proj_at_s1": round(float(v[-1]), 4),
                         "sign_stable": "yes" if same else "no",
                         "rho_dose": round(rho, 2),
                         "passes": "yes" if ok else "no"})
        ax.axhline(0, color=MUTED, lw=0.7, zorder=2)
        ax.set_title(LABEL[arm], loc="left", fontsize=7.5)
        ax.text(0.03, 0.03, f"{n_pass}/8 valid", transform=ax.transAxes, fontsize=6.4,
                color=INK, fontweight="bold", va="bottom", ha="left")
        ax.set_xlim(0.38, 1.22)
        ax.set_xticks([0.5, 1.0])
        despine(ax, grid_axis=None)

    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=INK, lw=1.5, label="risk-taking"),
           Line2D([], [], color=INK, lw=1.5, ls=(0, (2.6, 1.2)), label="impulsivity"),
           Line2D([], [], color=INK, lw=0.8, alpha=0.42, label="the other six traits")]
    fig.legend(handles=key, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.995),
               handletextpad=0.4, columnspacing=1.4, borderpad=0)

    axes[0].set_ylim(-0.62, 0.32)
    axes[0].set_ylabel("signed projection on\nthe base trait axis")
    fig.supxlabel("measured functional dose ($\\div$ goodness $s{=}1$)", fontsize=7.5,
                  y=-0.02)

    save(fig, "figA5_signed_validity")
    write_source_data("figA5_signed_validity", rows,
                      ["arm", "trait", "dose_hi", "proj_at_s1", "sign_stable",
                       "rho_dose", "passes"])


if __name__ == "__main__":
    main()
