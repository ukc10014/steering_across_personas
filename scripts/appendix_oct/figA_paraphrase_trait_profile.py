#!/usr/bin/env python3
"""Appendix asset -- the eight corrected forced-choice logit offsets, three arms.

Scientific question. The headline table reports three aggregates (B1, B2, activation
selectivity) and they all fall from released -> reproduction -> P0. An aggregate cannot say
WHY. This figure shows the per-trait profile the aggregates are built from, which makes the
mechanism visible: the two target traits attenuate hardest, and five of the six control traits
attenuate too. So behavioural selectivity falls because the TARGETS come down, not because the
controls rise -- the opposite of what "less selective" normally implies.

Read with the activation-space result, which points the other way: in activation space P0 shows
GREATER non-target displacement (common-shift g/base on the six controls rises 0.6536 -> 0.8774).
The apparent broadening is therefore specific to unsigned representational displacement; the
signed behavioural phenotype is attenuated, and most so on the targets.

Form. Categorical, three series -- the arms are the subject, so this is not an emphasis chart.
A grouped horizontal dot plot with interval bars, not a grouped bar chart: 24 values each with a
CI read better as dots than as bars, and the zero line is the reference the signs are read
against. Targets are banded at the top and separated from the controls by a gap, because the
target/control split is what B1 and B2 contrast.

Colour. Categorical slots 1-3 of the documented reference palette, in fixed slot order
(released, reproduction, P0). Those three slots are the documented all-pairs-validated subset,
which is the right pairlist here because every series is visible simultaneously. Aqua sits below
3:1 on the light surface, so the relief rule applies: the companion table carries every value,
and the two target traits are direct-labelled. Labels are selective by design -- a number on
every one of 24 dots would be unreadable.

Source
  outputs/analysis/caa_logits.json                  per-trait offsets + marginal CIs (forced)
  outputs/analysis/caa_logits_robustness.json       B1/B2 with CIs, linear estimator (annotation)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = pathlib.Path("/workspace/repos/steering_across_personas")
ANALYSIS = R / "outputs" / "analysis"

# Reference palette, light mode. Categorical slots 1-3 in fixed order; ink and surface tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"          # hairline, one shade off the surface, SOLID (never dashed)
ARMS = [
    ("impulsiveness",        "released OCT",                "#2a78d6"),
    ("impulsiveness_repro",  "fixed-data reproduction",     "#eb6834"),
    ("impulsiveness_regen",  "P0 end-to-end regeneration",  "#1baf7a"),
]
TARGETS = ["impulsivity", "risk_taking"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="forced", choices=("forced", "default"))
    ap.add_argument("--out", default=str(R / "docs/runs/oct/figA_paraphrase_trait_profile"))
    a = ap.parse_args()

    J = json.loads((ANALYSIS / "caa_logits.json").read_text())[a.variant]
    off = J["offset"]
    rob = json.loads((ANALYSIS / "caa_logits_robustness.json").read_text())[a.variant]["contrasts"]

    # Controls ordered by the released arm, descending: a stable, data-independent-of-P0 order.
    controls = sorted((t for t in off[ARMS[0][0]] if t not in TARGETS),
                      key=lambda t: -off[ARMS[0][0]][t]["point"])
    # Top of the axis = first drawn; invert later so targets sit on top with a gap beneath.
    rows = TARGETS + [None] + controls
    ypos, y = {}, 0.0
    for r in rows:
        if r is None:
            y += 0.95            # the gap that separates targets from controls
            continue
        ypos[r] = y
        y += 1.30                # between-trait step, comfortably > the within-trait spread

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Target band, drawn first so every mark sits above it.
    ax.axhspan(ypos[TARGETS[0]] - 0.52, ypos[TARGETS[-1]] + 0.52,
               color="#2a78d6", alpha=0.045, zorder=0, lw=0)

    nudge = 0.21                                        # vertical offset per series within a trait
    for i, (arm, label, colour) in enumerate(ARMS):
        dy = (i - 1) * nudge
        for t, yv in ypos.items():
            v = off[arm][t]
            ax.plot([v["ci_lo"], v["ci_hi"]], [yv + dy] * 2,
                    color=colour, lw=2.0, solid_capstyle="round", zorder=2)
            # 2px surface ring keeps overlapping markers separable without a drawn border
            ax.plot(v["point"], yv + dy, "o", ms=5.6, color=colour,
                    mec=SURFACE, mew=1.4, zorder=3)

    # Zero reference: solid hairline, the baseline the signs are read against.
    ax.axvline(0.0, color=INK_2, lw=0.9, zorder=1)

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([t.replace("_", "-") for t in ypos], fontsize=9, color=INK)
    for t in TARGETS:                                   # mark the two criterion targets
        ax.get_yticklabels()[list(ypos).index(t)].set_color(INK)
        ax.get_yticklabels()[list(ypos).index(t)].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("corrected forced-choice logit offset   (positive = toward the trait)",
                  fontsize=9, color=INK_2)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK_2, length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)     # solid hairline grid
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.7)

    # Selective direct labels: the two targets only, where the whole story is.
    for t in TARGETS:
        p0 = off[ARMS[2][0]][t]["point"]
        rp = off[ARMS[1][0]][t]["point"]
        # Left of the P0 dot: that part of each target row is empty, whereas the right side
        # runs into the reproduction and released dots.
        ax.annotate(f"{p0 - rp:+.2f} vs reproduction",
                    xy=(off[ARMS[2][0]][t]["ci_lo"], ypos[t] + nudge),
                    xytext=(-8, 0), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center", ha="right")

    b1 = rob["impulsivity_only"]["by_arm"]
    b2 = rob["impulsivity+risk_taking"]["by_arm"]
    sub = "   ".join(
        f"{lab.split()[0]}: B1 {b1[arm]['linear']['point']:+.2f}, B2 {b2[arm]['linear']['point']:+.2f}"
        for arm, lab, _ in ARMS)
    ax.set_title("Per-trait offsets: the targets attenuate most, and the controls attenuate too",
                 fontsize=10.5, color=INK, loc="left", pad=16)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")

    handles = [Line2D([], [], color=c, lw=2.0, marker="o", ms=5.6, mec=SURFACE, mew=1.4, label=l)
               for _, l, c in ARMS]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=INK_2, handletextpad=0.6, borderaxespad=0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = pathlib.Path(f"{a.out}.{ext}")
        fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
