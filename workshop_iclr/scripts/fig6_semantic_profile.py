#!/usr/bin/env python3
"""Figure 6 -- what semantic direction the trained models actually prefer.

The behavioural counterpart to figures 1 and 2. Those two say where representational
movement lives and how it factors; this one says which way the model's preferences move.

A. Compression-corrected forced-choice offset a_{c,t} for six trained constitutions across
   the eight traits. The heatmap makes cross-constitution structure legible in a way a
   per-arm dot plot does not: `impulsiveness` and `misalignment` share a risk-taking /
   impulsivity profile; `loving` independently produces the empathy / warmth profile its
   constitution names; `sycophancy` does not move `deference`; and `goodness` and
   `mathematical` are close to flat, which is what shows that a trained OCT adapter does
   not simply produce one generic profile.

B. The preregistered target-minus-other-traits contrasts, with question-bootstrap 95% CIs,
   each against the untrained arms on the SAME contrast. This panel carries the inference;
   panel A carries the pattern.

SIGN. a_{c,t} is the offset a in

    logodds_arm = a + k * logodds_base

fitted polarity-split and averaged (scripts/caa_logits_analysis.py). It is SIGNED:
positive means the arm moved the model's preference TOWARD the trait, negative AWAY. This
is the only quantity in the paper from which a signed semantic claim may be read. The
activation-space magnitudes of figure 1 are norms and carry no direction.

WHY THE CORRECTION MATTERS. Every arm compresses the model's preferences toward
indifference, which makes the naive logit difference a near-mirror of the base model's own
preferences (r <= -0.95 wherever k < 0.3). The demonstration of that is
fig5_behavioral_preference panels A/B; everything here is post-correction.

WHICH CELLS ARE MARKED, and the distinction the main text rests on:
  solid outline  -- the trait set was specified PROSPECTIVELY for that constitution:
                    `impulsivity` for `impulsiveness` (prereg 2026-07-17-v1),
                    `empathy`+`warmth` for `loving` and `deference` for `sycophancy`
                    (docs/prereg_loving_sycophancy.md, committed before any inference).
  dashed outline -- `risk_taking` for `impulsiveness`: the LATER, geometry-derived
                    extension, not the registered endpoint. Kept visually distinct so the
                    registered result is not silently upgraded by the addition.
  no outline     -- no prospectively specified target set.

`misalignment` is deliberately absent from panel B. It is broad enough that choosing a
target set for it now, after these logits have been seen, would be exactly the post-hoc
move the preregistrations exist to prevent. It appears in panel A as a full profile, which
is the honest way to show it. `goodness` and `mathematical` are comparison rows for the
same reason and get no contrast either.

Panel B's `impulsiveness` entry is the REGISTERED endpoint -- `impulsivity` alone against
the other seven -- not the two-trait `impulsivity`+`risk_taking` version that
caa_logits_analysis.py scores by default.

ONE ASYMMETRY TO BE AWARE OF. `loving`'s behavioural result here (+1.81, CI far from zero)
is much stronger than its activation-space selectivity ratio (1.18 at L15, CI covering 1).
This figure and figure 1 are therefore NOT two views of equally strong evidence for
`loving`; see docs/runs/oct/LOVING_SYCOPHANCY_RESULT.md section 7.

Source: outputs/analysis/caa_logits_{impulsivity,loving,sycophancy}.json
(scripts/caa_logits_analysis.py, 2000-replicate paired question bootstrap). FORCED prompt
throughout -- it puts ~91% of next-token mass on A/B against ~2.4% for the default form.
The default-prompt estimates give the same qualitative `loving` and `sycophancy` results and
are in the appendix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, COLOR, FULL, GRID, INK, LABEL, MARKER, MUTED, TRAITS,
                      TRAIT_LABEL, despine, save, use_style, write_source_data)

# Panel A rows: the two related profiles first, the flat comparison arms last. Trained
# only -- the untrained arms are NOT flat on this measure (random_iid_s16 reads honesty
# +2.89, risk +2.68), so as heatmap rows they would be the hottest cells in the figure and
# read backwards. They belong in panel B, where the target-vs-rest contrast is what
# controls for diffuse movement.
ROWS = ["impulsiveness", "misalignment", "loving", "sycophancy", "goodness", "mathematical"]

# Prospectively specified target sets ONLY, each with the file scored on that set.
REGISTERED: dict[str, tuple[list[str], str, str]] = {
    "impulsiveness": (["impulsivity"], "caa_logits_impulsivity.json", "prereg 2026-07-17"),
    "loving": (["empathy", "warmth"], "caa_logits_loving.json", "prereg 2026-10-05"),
    "sycophancy": (["deference"], "caa_logits_sycophancy.json", "prereg 2026-10-05"),
}
# The later geometry-derived extension, marked differently, never merged into the above.
POST_HOC = {"impulsiveness": ["risk_taking"]}
CONTROLS = ["random_iid_s16", "random_perm_s16"]

# figstyle owns the arm->hue mapping so an arm looks the same in every figure. The two new
# constitutions are added LOCALLY (`|` builds a new dict; figstyle's is untouched) rather
# than by editing figstyle.TRAINED, because fig4_shared_direction.py and
# figA2_diagnostics.py both build ARMS = TRAINED + UNTRAINED and would silently acquire
# two extra arms.
# TODO before submission: re-run `python workshop_iclr/scripts/validate_palette.py` over
# this extended set. #b5179e and #5aa9e6 from scripts/appendix_oct/ are NOT usable here --
# they fail the Machado CVD floor against `mathematical` and `misalignment` respectively.
ARM_COLOR = COLOR | {"loving": "#c2185b", "sycophancy": "#6b8f00"}
ARM_MARKER = MARKER | {"loving": "P", "sycophancy": "*"}
ARM_LABEL = LABEL | {"loving": "loving", "sycophancy": "sycophancy"}

VLIM = 3.2           # symmetric, so +1 and -1 read as equally intense
WHITE_TEXT_AT = 1.8  # |a| above which the cell is dark enough to need white type
HEADER = ["panel", "constitution", "trait", "value", "ci_lo", "ci_hi"]


def load_offsets() -> dict:
    """Corrected offsets, forced prompt.

    The `offset` block does not depend on --targets, so the two files must agree; assert
    it rather than trusting it, since a stale file would silently put the wrong numbers in
    the heatmap.
    """
    a = json.loads((ANALYSIS / "caa_logits_loving.json").read_text())["forced"]["offset"]
    b = json.loads((ANALYSIS / "caa_logits_sycophancy.json").read_text())["forced"]["offset"]
    for arm in sorted(set(a) & set(b)):
        for t in TRAITS:
            if abs(a[arm][t]["point"] - b[arm][t]["point"]) > 1e-9:
                raise AssertionError(f"offset blocks disagree at {arm}/{t}; one file is stale")
    missing = sorted(set(ROWS) - set(a))
    if missing:
        raise AssertionError(f"offsets missing for {missing}")
    return a


def panel_a(ax, offs: dict, rows: list[dict]):
    M = np.array([[offs[arm][t]["point"] for t in TRAITS] for arm in ROWS])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, aspect="auto")

    for i, arm in enumerate(ROWS):
        for j, t in enumerate(TRAITS):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.0,
                    color="white" if abs(v) > WHITE_TEXT_AT else INK, zorder=4)
            cell = offs[arm][t]
            rows.append({"panel": "A", "constitution": arm, "trait": t,
                         "value": round(v, 4), "ci_lo": round(cell["ci_lo"], 4),
                         "ci_hi": round(cell["ci_hi"], 4)})

    # the two marking states, drawn after the cells so the edges sit on top
    for arm, (tgts, _, _) in REGISTERED.items():
        for t in tgts:
            ax.add_patch(Rectangle((TRAITS.index(t) - 0.5, ROWS.index(arm) - 0.5), 1, 1,
                                   fill=False, edgecolor=INK, lw=1.4, zorder=5))
    for arm, tgts in POST_HOC.items():
        for t in tgts:
            ax.add_patch(Rectangle((TRAITS.index(t) - 0.5, ROWS.index(arm) - 0.5), 1, 1,
                                   fill=False, edgecolor=INK, lw=1.1, zorder=5,
                                   linestyle=(0, (2.2, 1.4))))

    ax.set_xticks(range(len(TRAITS)))
    ax.set_xticklabels([TRAIT_LABEL[t] for t in TRAITS], rotation=32, ha="right")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([ARM_LABEL[a] for a in ROWS])
    # white hairlines between cells, so adjacent saturated cells stay separable
    ax.set_xticks(np.arange(-0.5, len(TRAITS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A  corrected forced-choice preference shift $a_{c,t}$", loc="left")

    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.4,
                  label="preregistered target"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.1,
                  linestyle=(0, (2.2, 1.4)), label="later extension, not registered")],
        ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.085),
        handlelength=1.5, handletextpad=0.45, columnspacing=1.6, borderpad=0)
    return im


def panel_b(ax, rows: list[dict]) -> None:
    """Forest plot with a header row per registered target set.

    Block headers are tick labels rather than free text, which is standard forest-plot
    practice and cannot collide with the arm labels or the marks.
    """
    y = 0.0
    ticks: list[float] = []
    labels: list[str] = []
    is_header: list[bool] = []
    seps: list[float] = []

    for arm, (tgts, fname, _when) in REGISTERED.items():
        blk = json.loads((ANALYSIS / fname).read_text())["forced"]["selectivity"]
        if sorted(blk["targets"]) != sorted(tgts):
            raise AssertionError(f"{fname} scored on {blk['targets']}, expected {tgts}")
        # The prereg date lives in the docstring and the caption, not the tick label:
        # long y labels push the bbox-tight width past the 5.5in column and squeeze the
        # plotting area.
        set_name = ("+".join(TRAIT_LABEL[t] for t in tgts)
                    + f" vs other {len(blk['others'])}")

        ticks.append(y); labels.append(set_name); is_header.append(True)
        y += 0.85

        for a in [arm] + CONTROLS:
            s = blk["by_arm"].get(a)
            if s is None:
                continue
            focal = a == arm
            lo, hi = s["contrast_ci_lo"], s["contrast_ci_hi"]
            ax.plot([lo, hi], [y, y], color=ARM_COLOR[a], lw=1.5 if focal else 1.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([s["contrast"]], [y], marker=ARM_MARKER[a], ms=5.6 if focal else 4.0,
                    mfc=ARM_COLOR[a] if focal else "white", mec=ARM_COLOR[a], mew=1.0,
                    ls="none", zorder=4)
            ticks.append(y); labels.append(f"   {ARM_LABEL[a]}"); is_header.append(False)
            rows.append({"panel": f"B-{arm}", "constitution": a, "trait": set_name,
                         "value": round(s["contrast"], 4), "ci_lo": round(lo, 4),
                         "ci_hi": round(hi, 4)})
            y += 1.0
        seps.append(y - 0.3)
        y += 0.45

    ax.axvline(0.0, color=INK, lw=0.7, zorder=2)
    for s in seps[:-1]:
        ax.axhline(s, color=GRID, lw=0.6, zorder=1)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for lbl, hdr in zip(ax.get_yticklabels(), is_header):
        lbl.set_fontsize(6.3 if hdr else 7)
        lbl.set_color(MUTED if hdr else INK)
    ax.set_ylim(y - 0.6, -0.6)          # inverted: first block at the top
    ax.set_xlabel("target minus other traits, corrected log-odds (forced prompt)")
    ax.set_title("B  prospectively specified semantic contrasts, with untrained controls",
                 loc="left")
    despine(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=5.2, mfc=INK, mec=INK,
               label="constitution"),
        Line2D([], [], ls="none", marker="o", ms=4.0, mfc="white", mec=INK,
               label="untrained control")],
        ncol=1, loc="lower right", handletextpad=0.4, borderpad=0.3)


def main() -> None:
    use_style()
    offs = load_offsets()
    rows: list[dict] = []

    fig = plt.figure(figsize=(FULL, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.60)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    im = panel_a(axA, offs, rows)
    cb = fig.colorbar(im, ax=axA, fraction=0.022, pad=0.012)
    cb.set_label("$a_{c,t}$  (+ toward the trait)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_visible(False)

    panel_b(axB, rows)

    save(fig, "fig6_semantic_profile")
    write_source_data("fig6_semantic_profile", rows, HEADER)


if __name__ == "__main__":
    main()
