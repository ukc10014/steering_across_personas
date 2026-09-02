#!/usr/bin/env python3
"""Figure 4 -- the constitutions share a direction that untrained perturbations miss.

Almost everything else in this paper is a null. The magnitude of the common shift, its
rotation with dose (figure A4), the contraction of persona geometry (figure 3C/D), the
per-cell interaction (figure 2B) and the sign of the trait projection (figure A5) are all
reproduced by untrained perturbations at matched functional dose. This is the one
statistic on which they are not.

    cos(dG_c, dG_c'), averaged over the 8 traits, for every pair of arms at s=1

  trained x trained      0.464 - 0.837
  untrained x untrained  0.216 - 0.379
  trained x untrained    0.012 - 0.298

The four constitutions' persona-common shifts are substantially aligned with one another
and close to orthogonal to every untrained perturbation. Section 3.2 already reported the
trained-trained block; what makes it interpretable is the other two blocks, which were
missing. Cosine is scale-invariant, so none of this is dose arithmetic -- it is a claim
about direction, and the arms compared here sit at doses 0.99-1.11.

WHAT THIS DOES NOT ESTABLISH, and the reason it is stated carefully in panel B's label.
All four trained arms come out of the same OCT pipeline: same base model, same LoRA rank
and initialisation, same optimiser and schedule, same shape of training corpus. Their
shared direction is therefore consistent with "constitutional content lives in a common
subspace" AND with "this training procedure moves the model in a characteristic direction
whatever the constitution says". Nothing here separates those. The sham-trained LoRA --
same pipeline, character signal destroyed rather than never present -- is the control that
would, and it does not exist yet.

The untrained block is also not a like-for-like reference: `random_iid`, `random_spec` and
`random_perm` are three different CONSTRUCTIONS, whereas `goodness` and `mathematical` are
the same object trained on different text, so the untrained arms have less reason to agree
with each other in the first place. The comparison that does not suffer from this is the
cross block: whatever direction the constitutions share, no untrained arm points there.

Source: outputs/analysis/common_shift_cross_family.json (scripts/common_shift.py,
cross-fitted over 40 question half-splits, point estimates -- this run does not bootstrap,
so no intervals are drawn). Section 3.2's bootstrapped trained-trained intervals are the
reference for how well-resolved that block is: at L15 the goodness-mathematical interval
clears the goodness-misalignment interval in 7 of 8 traits.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, LABEL, TRAINED, UNTRAINED, INK, MUTED,
                      use_style, despine, save, write_source_data)

LAYER = "15"
ARMS = TRAINED + UNTRAINED


def main() -> None:
    use_style()
    d = json.load(open(ANALYSIS / "common_shift_cross_family.json"))[LAYER]
    traits = list(d)

    def cm(a, b):
        return float(np.mean([d[t]["cos"][f"{a}|{b}"] for t in traits]))

    M = np.array([[cm(a, b) for b in ARMS] for a in ARMS])
    rows = []

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL, 2.55),
                                   gridspec_kw={"width_ratios": [1.15, 1],
                                                "wspace": 0.42})

    # ---- A: the full pairwise matrix, sequential single hue ----------------------
    Mm = np.ma.masked_where(np.eye(len(ARMS), dtype=bool), M)
    im = axA.imshow(Mm, cmap="Blues", vmin=0, vmax=0.9, zorder=2)
    for i in range(len(ARMS)):
        for j in range(len(ARMS)):
            if i == j:
                continue
            axA.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                     color="white" if M[i, j] > 0.5 else INK, zorder=3)
    n_t = len(TRAINED)
    for edge in (n_t - 0.5,):
        axA.axhline(edge, color=INK, lw=0.9, zorder=4)
        axA.axvline(edge, color=INK, lw=0.9, zorder=4)
    axA.set_xticks(range(len(ARMS)))
    axA.set_xticklabels([LABEL[a] for a in ARMS], rotation=38, ha="right", fontsize=6)
    axA.set_yticks(range(len(ARMS)))
    axA.set_yticklabels([LABEL[a] for a in ARMS], fontsize=6)
    axA.set_title("A  cos($dG$, $dG'$) at $s{=}1$, mean over 8 traits", loc="left")
    for s in axA.spines.values():
        s.set_visible(False)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)

    # ---- B: the three blocks, as the comparison the figure is making -------------
    blocks = {"trained\n× trained": [], "untrained\n× untrained": [],
              "trained\n× untrained": []}
    for a, b in itertools.combinations(ARMS, 2):
        ta, tb = a in TRAINED, b in TRAINED
        k = ("trained\n× trained" if ta and tb else
             "untrained\n× untrained" if not ta and not tb else
             "trained\n× untrained")
        blocks[k].append(cm(a, b))
        rows.append({"arm_a": a, "arm_b": b, "block": k.replace("\n", " "),
                     "cos": round(cm(a, b), 4)})
    order = list(blocks)
    for i, k in enumerate(order):
        v = np.array(blocks[k])
        jit = np.linspace(-0.13, 0.13, len(v))
        axB.scatter(v, i + jit, s=11, color="#2a78d6" if i == 0 else "#6f6f69",
                    alpha=0.75, linewidths=0, zorder=3)
        axB.plot([v.mean()] * 2, [i - 0.28, i + 0.28], color=INK, lw=1.5, zorder=4,
                 solid_capstyle="butt")
        axB.annotate(f"{v.mean():.2f}", (v.mean(), i), textcoords="offset points",
                     xytext=(0, 11), fontsize=6.2, color=INK, ha="center", va="bottom")
    axB.set_yticks(range(len(order)))
    axB.set_yticklabels(order, fontsize=6.4)
    axB.set_ylim(-0.6, len(order) - 0.2)
    axB.invert_yaxis()
    axB.set_xlim(-0.05, 0.95)
    axB.set_xlabel("cos($dG$, $dG'$)")
    axB.set_title("B  by block", loc="left")
    axB.axvline(0, color=MUTED, lw=0.7, zorder=1)
    despine(axB, grid_axis="x")
    axB.yaxis.set_tick_params(length=0)

    save(fig, "fig4_shared_direction")
    write_source_data("fig4_shared_direction", rows, ["arm_a", "arm_b", "block", "cos"])


if __name__ == "__main__":
    main()
