#!/usr/bin/env python3
"""Appendix A4 -- the common shift rotates with dose, and an untrained arm rotates too.

Section 3.3 reports that each constitution's persona-common shift direction ROTATES as its
dose grows, and reads that as structure. This is the control that was missing: the same
measurement on `random_perm`, which has three cached dose rungs and no learned content at
all.

    x   the gap in measured functional dose between two rungs of the SAME arm
    y   cos(dG at the lower rung, dG at the higher rung), averaged over the 8 traits

If rotation-with-dose were a property of constitutional content, the untrained arm should
sit above the trained ones -- a scaled random perturbation has no reason to change
direction. It does not sit above them. Across its full ladder `random_perm` rotates to
cos = 0.742 against `goodness`'s 0.753, and against dose GAP, which is what has to be
controlled for (the random rungs are further apart in dose than the constitution rungs),
every family falls on essentially one curve.

So section 3.3's rotation is reproduced by a perturbation with no content, and is not on
its own evidence of anything semantic. What the constitutions still own is that they rotate toward
DIFFERENT places from the untrained arms -- that is figure 4, which measures it across the
whole arm set rather than for one pair, and finds trained-untrained cosines of 0.01-0.30
against 0.46-0.84 within the trained family. Shared rotation RATE, different destination.

Sources
  outputs/analysis/common_shift_full.json            the three constitution ladders
  outputs/analysis/common_shift_random_rotation.json  the random_perm ladder + goodness
Both runs are cross-fitted over 40 question half-splits. They are separate invocations, so
the overlapping `goodness` ladder is a consistency check between them: the six goodness
rung-pair cosines agree to within 0.001, which is why the two files can be combined.
Point estimates only -- neither run bootstraps, so there are no intervals here and the
figure is read as a comparison of curves, not of individual points.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, COLOR, MARKER, LABEL, DASH, FULL, INK, MUTED,
                      use_style, despine, save, write_source_data)

LAYER = "15"
LADDERS = {
    "goodness": ["goodness_s0.25", "goodness_s0.5", "goodness_s0.75", "goodness"],
    "impulsiveness": ["impulsiveness_s0.25", "impulsiveness_s0.5",
                      "impulsiveness_s0.75", "impulsiveness"],
    "misalignment": ["misalignment_s0.25", "misalignment_s0.5",
                     "misalignment_s0.75", "misalignment"],
    "random_perm_s16": ["random_perm_s8", "random_perm_s12", "random_perm_s16"],
}


def main() -> None:
    use_style()
    full = json.load(open(ANALYSIS / "common_shift_full.json"))[LAYER]
    rot = json.load(open(ANALYSIS / "common_shift_random_rotation.json"))[LAYER]
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[LAYER]
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}
    traits = [t for t in full if t in rot]

    def cos_mean(src, a, b):
        return float(np.mean([src[t]["cos"][f"{a}|{b}"] for t in traits]))

    fig, axA = plt.subplots(figsize=(FULL * 0.56, 2.3))
    rows = []

    # ---- A: rotation against the dose gap that produced it -----------------------
    allpts: list[tuple[float, float]] = []
    for arm, rungs in LADDERS.items():
        src = rot if arm.startswith("random") else full
        pts = []
        for a, b in itertools.combinations(rungs, 2):
            pts.append((abs(dose[b] - dose[a]), cos_mean(src, a, b), a, b))
        pts.sort()
        axA.plot([p[0] for p in pts], [p[1] for p in pts], ls="none",
                 color=COLOR[arm], marker=MARKER[arm], ms=4.2, mew=0.5,
                 mec="white", label=LABEL[arm], zorder=3)
        allpts.extend((p[0], p[1]) for p in pts)
        for gap, c, a, b in pts:
            rows.append({"arm": LABEL[arm], "rung_lo": a, "rung_hi": b,
                         "dose_gap": round(gap, 4), "cos": round(c, 4)})
    # one least-squares line through ALL 21 pairs, trained and untrained together: the
    # claim is that they share a curve, so the guide is fitted to the pooled set or not
    # at all. It is a visual guide, not a model.
    ap = np.array(allpts)
    b1, b0 = np.polyfit(ap[:, 0], ap[:, 1], 1)
    gx = np.linspace(ap[:, 0].min(), ap[:, 0].max(), 2)
    axA.plot(gx, b0 + b1 * gx, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=1)
    resid = ap[:, 1] - (b0 + b1 * ap[:, 0])
    axA.text(0.97, 0.95, f"pooled fit, residual SD {resid.std():.3f}",
             transform=axA.transAxes, fontsize=6, color=MUTED, ha="right", va="top")
    axA.set_xlabel("dose gap between the two rungs")
    axA.set_ylabel("cos($dG$ at the two rungs)")
    axA.set_title("rotation tracks dose gap, trained or not", loc="left")
    axA.set_ylim(0.70, 1.01)
    axA.legend(loc="lower left", handletextpad=0.4, labelspacing=0.25, borderpad=0)
    despine(axA)

    save(fig, "figA4_rotation_control")
    write_source_data("figA4_rotation_control", rows,
                      ["arm", "rung_lo", "rung_hi", "dose_gap", "cos"])


if __name__ == "__main__":
    main()
