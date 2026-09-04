#!/usr/bin/env python3
"""Figure 5 -- the signed behavioural result, and the artefact that hides it.

Everything else in this paper measures how FAR a representation moved. This measures which
WAY the model then answers. Each CAA item is put to the model with the generation prompt
open and the two answer letters' logits are read; signed by item polarity, that is
log P(trait-positive)/P(trait-negative), with no judge and no sampling.

The panels are in the order the argument has to be made:

  A  Every arm compresses. Fitted transformation logodds_arm = a + k*logodds_base, drawn
     against the identity. The trained constitutions keep about a quarter of the base
     model's preference structure; the untrained arms keep far more.
  B  Which is why the obvious estimator is invalid. E[arm - base] per trait plotted against
     where the base model already stood: the points fall on a line of slope ~-(1-k). The
     naive number is a readout of the BASE model's preferences, not of the arm's effect.
  C  The offset a, which is what the naive number was supposed to be: where the arm pushes
     an item the base model was indifferent about. Signed, and free of the compression.
  D  The contrast -- impulsivity and risk-taking against the other six -- fixed before any
     logit was seen. Of the two, only `impulsivity` is registered (prereg/2026-07-17-v1.md
     sec 3); `risk_taking` came from the geometry. On `impulsivity` alone the same arms give
     +2.18 and +2.22 against <= -0.52 for every other arm, so nothing here rests on the
     unregistered half of the pair (scripts/caa_logits_robustness.py) --
     under both prompt forms. This is the test the geometry could not do.

The result: `impulsiveness` and `misalignment` push specifically toward the two traits
their content is about, while `goodness`, `mathematical` and both untrained arms do not.
Under the forced prompt the untrained controls' intervals cover zero. The trained arms
`goodness`, `mathematical` and `impulsiveness` sit at almost identical retention
(k = 0.25, 0.28, 0.29), so within the trained family the contrast is not a dose effect.

The intercept is a prediction AT logodds_base = 0, so it is only the quantity claimed if the
relationship is near-affine and zero is inside the data. Both were checked, in
scripts/caa_logits_robustness.py, after the figure was drawn and before it was cited:

  support     under the forced prompt 9-21% of items per trait sit within |logodds_base| < 1
              (517-1144 of 5500), so the intercept is interpolated, not extrapolated.
  shape       mean residual of the linear fit by base decile is within +-0.13 log-odds across
              every decile for `goodness`, `mathematical`, `impulsiveness` and `misalignment`.
              The bend is real only on the two UNTRAINED arms (up to 1.3), where the estimate
              is near zero anyway.
  model-free  replacing the fit with the polarity-balanced MEAN over items with
              |logodds_base| < d -- no shape assumption at all -- moves the contrast by at
              most 0.13 at any of d = 0.5, 1, 2:
                impulsiveness  +2.08 linear, +2.04 quadratic, +2.21 / +2.18 / +2.04 local
                misalignment   +2.49         +2.50            +2.58 / +2.54 / +2.54
                goodness       -0.39         -0.38            -0.29 / -0.32 / -0.39
The affine step is doing no work; it is a variance reduction, not the result.

Requires outputs/analysis/caa_logits.json:
    bash scripts/run_caa_logits.sh && python scripts/caa_logits_analysis.py --n-boot 2000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, COLOR, MARKER, LABEL, DASH, FULL, INK, MUTED,
                      TRAITS, TRAIT_LABEL, IMPULSIVE_TRAITS,
                      use_style, despine, save, write_source_data)

SRC = ANALYSIS / "caa_logits.json"
ORDER = ["goodness", "mathematical", "impulsiveness", "misalignment",
         "random_iid_s16", "random_perm_s16"]
HEADLINE = "forced"      # panels A-C; D shows both


def load():
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found -- run scripts/run_caa_logits.sh then "
            f"scripts/caa_logits_analysis.py")
    d = json.loads(SRC.read_text())
    if d.get(HEADLINE) is None:
        raise KeyError(HEADLINE)
    return d


def main() -> None:
    use_style()
    d = load()
    res = d[HEADLINE]
    arms = [a for a in ORDER if a in res["offset"]]
    traits = [t for t in TRAITS if t in res["base_level"]]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.0))
    (axA, axB), (axC, axD) = axes

    # ---- A: the transformation each arm applies -------------------------------------
    x = np.linspace(-8, 8, 2)
    axA.plot(x, x, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    axA.annotate("identity (no change)", (-7.4, -6.2), color=MUTED, fontsize=6,
                 ha="left", va="center", rotation=39, rotation_mode="anchor")
    for a in arms:
        ks = [res["retention"][a][t]["point"] for t in traits]
        offs = [res["offset"][a][t]["point"] for t in traits]
        k, off = float(np.mean(ks)), float(np.mean(offs))
        axA.plot(x, off + k * x, color=COLOR[a], ls=DASH[a], lw=1.3, zorder=3,
                 label=LABEL[a])
    axA.set_xlabel("base model log-odds (trait-positive vs negative)")
    axA.set_ylabel("arm log-odds")
    axA.set_title("A  every arm compresses toward indifference")
    axA.tick_params(pad=1.5)
    axA.set_xlim(-8, 8); axA.set_ylim(-8, 8)
    axA.set_aspect("equal", adjustable="box")
    despine(axA, grid_axis="both")

    # ---- B: why the naive delta is invalid ------------------------------------------
    # A line and an r PER ARM. Pooling the arms would understate the effect: they have
    # different retention, so the pooled cloud is a mixture of lines with different slopes
    # (pooled r = -0.71) while each arm on its own is an almost exact mirror.
    rs = []
    for a in arms:
        xs = np.array([res["base_level"][t] for t in traits])
        ys = np.array([res["delta"][a][t]["point"] for t in traits])
        rs.append(float(np.corrcoef(xs, ys)[0, 1]))
        sl, ic = np.polyfit(xs, ys, 1)
        gx = np.array([xs.min() - 0.5, xs.max() + 0.5])
        axB.plot(gx, ic + sl * gx, color=COLOR[a], ls=DASH[a], lw=0.8, alpha=0.75,
                 zorder=2)
        axB.plot(xs, ys, ls="none", marker=MARKER[a], ms=3.4, mfc=COLOR[a],
                 mec="white", mew=0.4, zorder=3)
        for t, xv, yv in zip(traits, xs, ys):
            rows.append({"panel": "B", "prompt": HEADLINE, "arm": a, "trait": t,
                         "measure": "naive_delta_vs_base_level",
                         "value": round(float(yv), 4), "ci_lo": round(float(xv), 4),
                         "ci_hi": ""})
    axB.axhline(0, color=MUTED, lw=0.6, zorder=1)
    # The mirror is not uniform, and the way it varies confirms the mechanism: an arm
    # mirrors the base exactly to the extent that it compresses. k < 0.3 gives r <= -0.95;
    # the two lightly-compressing untrained arms (k = 0.68, 0.81) give -0.55 and -0.29.
    heavy = [r for a, r in zip(arms, rs)
             if np.mean([res["retention"][a][t]["point"] for t in traits]) < 0.3]
    axB.annotate(f"arms that compress hard: r $\\leq$ {max(heavy):+.2f}", (0.97, 0.95),
                 xycoords="axes fraction", ha="right", va="top", fontsize=6.5, color=INK)
    axB.set_xlabel("base model's own preference level for the trait")
    axB.set_ylabel("naive  E[arm - base]")
    axB.set_title("B  so the naive shift mirrors the base")
    despine(axB, grid_axis="both")

    # ---- C: the corrected, signed offset --------------------------------------------
    xs = np.arange(len(traits))
    for t_i, t in enumerate(traits):
        if t in IMPULSIVE_TRAITS:
            axC.axvspan(t_i - 0.45, t_i + 0.45, color="#f0efe9", zorder=0)
    axC.axhline(0, color=MUTED, lw=0.6, zorder=1)
    for a in arms:
        ys = [res["offset"][a][t]["point"] for t in traits]
        axC.plot(xs, ys, color=COLOR[a], ls=DASH[a], marker=MARKER[a], ms=3.2,
                 mec="white", mew=0.4, lw=1.1, zorder=3, label=LABEL[a])
        for t, y in zip(traits, ys):
            e = res["offset"][a][t]
            rows.append({"panel": "C", "prompt": HEADLINE, "arm": a, "trait": t,
                         "measure": "offset", "value": round(y, 4),
                         "ci_lo": round(e["ci_lo"], 4), "ci_hi": round(e["ci_hi"], 4)})
    axC.set_xticks(xs)
    axC.set_xticklabels([TRAIT_LABEL[t] for t in traits], rotation=38, ha="right")
    axC.set_ylabel("offset  (log-odds)")
    axC.set_title("C  compression-free signed shift")
    axC.annotate("shaded: the two target traits", (0.99, 0.03),
                 xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=6, color=MUTED)
    despine(axC)

    # ---- D: the contrast, both prompts ---------------------------------
    variants = [v for v in ("forced", "default") if d.get(v) is not None]
    sel_order = sorted(arms, key=lambda a: -d[HEADLINE]["selectivity"]["by_arm"][a]["contrast"])
    ypos = np.arange(len(sel_order))[::-1]
    off_of = {"forced": +0.16, "default": -0.16}
    for v in variants:
        by_arm = d[v]["selectivity"]["by_arm"]
        filled = v == "forced"
        for yi, a in zip(ypos, sel_order):
            if a not in by_arm:
                continue
            e = by_arm[a]
            y = yi + off_of[v]
            axD.plot([e["contrast_ci_lo"], e["contrast_ci_hi"]], [y, y],
                     color=COLOR[a], lw=1.1, solid_capstyle="butt", zorder=2)
            axD.plot([e["contrast"]], [y], marker=MARKER[a], ms=4,
                     mfc=COLOR[a] if filled else "white", mec=COLOR[a], mew=0.9,
                     zorder=3, ls="none")
            rows.append({"panel": "D", "prompt": v, "arm": a, "trait": "",
                         "measure": "contrast", "value": round(e["contrast"], 4),
                         "ci_lo": round(e["contrast_ci_lo"], 4),
                         "ci_hi": round(e["contrast_ci_hi"], 4)})
    axD.axvline(0, color=MUTED, lw=0.8, zorder=1)
    axD.set_yticks(ypos)
    axD.set_yticklabels([LABEL[a] for a in sel_order])
    axD.set_xlabel("contrast: target traits - other six  (log-odds)")
    axD.set_title("D  only content-matched arms are selective")
    axD.annotate("filled = forced prompt\nopen = default prompt", (0.97, 0.06),
                 xycoords="axes fraction", ha="right", va="bottom", fontsize=6,
                 color=MUTED)
    despine(axD, grid_axis="x")

    handles, labels = axC.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(arms),
               bbox_to_anchor=(0.5, 1.045), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "fig5_behavioral_preference")
    write_source_data("fig5_behavioral_preference", rows,
                      ["panel", "prompt", "arm", "trait", "measure",
                       "value", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    main()
