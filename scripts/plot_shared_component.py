#!/usr/bin/env python3
"""The confound in cosine-to-own-null, and the corrected comparison.

WHY THESE FIGURES EXIST. `arm_comparison_*.png` puts every arm's persona cosines on one
y-axis, which invites reading the heights against each other. They are not comparable.
Cosine is invariant to SCALING each vector but not to ADDING the same vector to both of
its arguments, and merging a LoRA adapter adds a large, largely persona-independent
component to every trait vector (measured: 0.68-0.87x the vector's own magnitude at L15).
So an arm sits higher mainly because its shared component is bigger -- which is a fact
about the merge, not about how persona-contingent its traits are.

FIGURE 1 (confound_diagnosis) makes that checkable rather than asking for trust.
  Panel A plots the raw persona-mean cosine against share^2, where share = ||mean vector|| /
  mean||vector||. If trait vectors are a shared part plus mutually near-orthogonal specific
  parts, the algebra gives cos = share^2 exactly. Points landing on the identity line
  therefore mean the raw metric is a restatement of the shared component and carries almost
  no independent information about persona structure. Points scattering off it would mean
  the opposite. One point per arm x trait, so the reader sees all 32, not a summary.
  Panel B contrasts persona DISPERSION raw vs corrected. Dispersion was the registered
  primary, and it is where the two views disagree most sharply: raw SD collapses ~60% while
  residual SD does not fall at all.

FIGURE 2 (arm_comparison_corrected) is the honest version of the strip plot: the same form as
`plot_arm_comparison.py` panel A, but on the corrected residual cosine, which IS on a common
scale across arms.

FIGURE 3 (ordering_preservation) is the surviving result -- Spearman between each arm's
per-persona residual ordering and base's, per trait, with question-resampling intervals from
caa_holdout_ci.py when that file exists. Its caption states exactly what the intervals support:
the aggregate difference is resolved, the per-trait localisation is not.

Both take marker shape as well as colour per arm: these figures are read in print and by
people with colour-vision deficiency, and four coloured dot clouds at one x position are
hard to separate even in colour.

Usage:
    python scripts/plot_shared_component.py --layer 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_steering.config import OUTPUTS_DIR
from plot_arm_comparison import (ARM_COLOURS, BASE_C, CONTROL_C, INK_MUTED, INK_PRIMARY,
                                 INK_SECONDARY, SURFACE)

# Base first, then adapted arms in the order they appear in the JSON.
MARKERS = ["o", "^", "s", "P"]
DEFAULT_JSON = OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis" / "caa_shared_component.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=str, default=str(DEFAULT_JSON))
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--ci-json", type=str,
                   default=str(OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis"
                              / "caa_holdout_ci.json"),
                   help="optional; adds bootstrap intervals to the ordering figure")
    p.add_argument("--outdir", type=str,
                   default=str(OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis"))
    return p.parse_args()


def style(ax, hide=("top", "right")) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for s in hide:
        ax.spines[s].set_visible(False)
    for s in set(("top", "right", "left", "bottom")) - set(hide):
        ax.spines[s].set_color("#dedcd7")
    ax.grid(color="#e8e7e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    D = json.loads(Path(args.json).read_text())
    if args.layer not in D["layers"]:
        raise SystemExit(f"error: layer {args.layer} not in {D['layers']}; re-run "
                         f"caa_shared_component.py --layers ... {args.layer}")
    li = D["layers"].index(args.layer)
    L = args.layer
    outdir = Path(args.outdir)

    arms = list(D["arms"])
    base = arms[0]
    traits = list(D["arms"][base])
    # Ordered by the BASE arm's raw mean, loosest first, matching plot_arm_comparison.py so
    # the corrected figure can be laid beside the raw one without the columns moving.
    traits.sort(key=lambda t: D["arms"][base][t]["raw_mean"][li])
    colours = {a: (BASE_C if i == 0 else ARM_COLOURS[(i - 1) % len(ARM_COLOURS)])
               for i, a in enumerate(arms)}
    marks = {a: MARKERS[i % len(MARKERS)] for i, a in enumerate(arms)}

    def g(arm, trait, key):
        return D["arms"][arm][trait][key][li]

    # ============================ FIGURE 1: diagnosis ============================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.8), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.24})

    # ---- Panel A: raw cosine vs share^2 ----
    style(axA)
    xs, ys = [], []
    for arm in arms:
        x = np.array([g(arm, t, "share") ** 2 for t in traits])
        y = np.array([g(arm, t, "raw_mean") for t in traits])
        xs.append(x)
        ys.append(y)
        axA.scatter(x, y, s=64, marker=marks[arm], color=colours[arm], alpha=0.85,
                    edgecolors=SURFACE, linewidths=0.9, zorder=3, label=arm)
    allx, ally = np.concatenate(xs), np.concatenate(ys)
    lo, hi = min(allx.min(), ally.min()) - 0.04, max(allx.max(), ally.max()) + 0.04
    axA.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    mad = np.abs(ally - allx).mean()
    axA.text(0.97, 0.06, f"identity line\nmean |deviation| = {mad:.3f}", transform=axA.transAxes,
             ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
    axA.set_xlim(lo, hi), axA.set_ylim(lo, hi)
    axA.set_aspect("equal")
    axA.set_xlabel(r"share$^2$   (how much of a trait vector is the part every context shares)",
                   fontsize=9.5, color=INK_SECONDARY)
    axA.set_ylabel("raw persona-mean cos to own null", fontsize=10, color=INK_SECONDARY)
    axA.set_title("A.  The raw metric restates the shared component",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")
    axA.legend(handles=[Line2D([], [], marker=marks[a], ls="none", markersize=7.5,
                               color=colours[a], label=a) for a in arms],
               loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    # ---- Panel B: dispersion, raw vs residual ----
    # Dispersion, not location, was the pre-registered primary, and it is where the raw and
    # corrected views disagree most sharply: raw SD collapses ~60% while residual SD does not
    # fall at all. The residual MEAN is deliberately not plotted -- it sits at ~0.00 in every
    # arm including base, so it separates nothing and showing it invites a null-result read.
    style(axB)
    w = 0.34
    for j, arm in enumerate(arms):
        raw_sd = np.mean([np.std(D["arms"][arm][t]["raw_per_persona"][li], ddof=1) for t in traits])
        res_sd = np.mean([np.std(D["arms"][arm][t]["holdout_per_persona"][li], ddof=1)
                          for t in traits])
        axB.bar(j - w / 2, raw_sd, w, color=colours[arm], alpha=0.95, zorder=3,
                edgecolor=SURFACE, lw=0.8)
        axB.bar(j + w / 2, res_sd, w, color=colours[arm], alpha=0.32, zorder=3,
                edgecolor=colours[arm], lw=1.3, hatch="///")
        axB.text(j - w / 2, raw_sd + 0.004, f"{raw_sd:.3f}", ha="center", va="bottom",
                 fontsize=8.8, color=INK_PRIMARY)
        axB.text(j + w / 2, res_sd + 0.004, f"{res_sd:.3f}", ha="center", va="bottom",
                 fontsize=8.8, color=INK_SECONDARY)
    axB.set_ylim(top=axB.get_ylim()[1] * 1.22)
    axB.set_xticks(range(len(arms)))
    axB.set_xticklabels(arms, fontsize=9.5, color=INK_SECONDARY, rotation=12, ha="right")
    axB.set_xlim(-0.6, len(arms) - 0.4)
    axB.set_ylabel("SD across personas (mean over 8 traits)", fontsize=10, color=INK_SECONDARY)
    axB.set_title("B.  Persona dispersion collapses only in the confounded view",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")
    axB.legend(handles=[
        Line2D([], [], marker="s", ls="none", markersize=10, color=INK_MUTED, label="raw (confounded)"),
        Line2D([], [], marker="s", ls="none", markersize=10, markerfacecolor="none",
               markeredgecolor=INK_MUTED, label="shared component removed")],
        loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.suptitle(f"Why the arms cannot be compared on raw cosine — layer {L}",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left",
                 fontweight="semibold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, f"confound_diagnosis_L{L}")

    # ======================= FIGURE 2: corrected strip plot =======================
    fig, ax = plt.subplots(figsize=(12.4, 6.2), facecolor=SURFACE)
    style(ax, hide=("top", "right", "bottom"))
    rng = np.random.default_rng(0)
    offs = np.linspace(-0.30, 0.30, len(arms))
    for j, t in enumerate(traits):
        for k, arm in enumerate(arms):
            x = j + offs[k]
            pts = np.array(D["arms"][arm][t]["holdout_per_persona"][li])
            ax.scatter(x + rng.uniform(-0.035, 0.035, pts.size), pts, s=26,
                       marker=marks[arm], color=colours[arm], alpha=0.8,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            ax.hlines(pts.mean(), x - 0.085, x + 0.085, color=INK_PRIMARY, lw=2.1, zorder=4)
            ax.scatter([x], [D["arms"][arm][t]["holdout_nonsense"][li]], marker="D", s=24,
                       color=CONTROL_C, edgecolors=SURFACE, linewidths=0.8, zorder=5)
        if j % 2 == 0:
            ax.axvspan(j - 0.5, j + 0.5, color="#f2f1ed", lw=0, zorder=-1)

    ax.axhline(0.0, color=INK_SECONDARY, lw=1.1, zorder=1)
    ax.text(len(traits) - 0.45, 0.012, "unrelated to the default", fontsize=8.5,
            color=INK_SECONDARY, ha="right", va="bottom")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([t.replace("_", " ") for t in traits], fontsize=9.5,
                       color=INK_SECONDARY, rotation=18, ha="right")
    ax.set_xlim(-0.5, len(traits) - 0.5)
    ax.set_ylabel("hold-out-centred cos to own null\n(shared component removed — comparable across arms)",
                  fontsize=10, color=INK_SECONDARY)
    means = {a: np.mean([g(a, t, "holdout_mean") for t in traits]) for a in arms}
    handles = [Line2D([], [], marker=marks[a], ls="none", markersize=7.5, color=colours[a],
                      label=f"{a}  (mean {means[a]:.3f})") for a in arms]
    handles += [Line2D([], [], marker="D", ls="none", markersize=6, color=CONTROL_C,
                       label="nonsense control"),
                Line2D([], [], color=INK_PRIMARY, lw=2.1, label="persona mean")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY, ncol=3)
    ax.set_title(f"Persona-conditional trait structure, shared component removed — layer {L}",
                 fontsize=13, color=INK_PRIMARY, pad=12, loc="left", fontweight="semibold")
    fig.tight_layout()
    save(fig, outdir, f"arm_comparison_corrected_L{L}")

    # ======================= FIGURE 3: ordering preservation =======================
    # The one quantity that survives the correction WITH a difference between arms. Bars are
    # Spearman between an arm's per-persona residual ordering and base's, per trait.
    ci = None
    ci_path = Path(args.ci_json)
    if ci_path.exists():
        ci = json.loads(ci_path.read_text()).get("summary", {}).get(str(L))

    fig, ax = plt.subplots(figsize=(12.6, 5.6), facecolor=SURFACE)
    style(ax, hide=("top", "right"))
    adapted = arms[1:]
    w = 0.8 / len(adapted)
    for j, arm in enumerate(adapted):
        xs = np.arange(len(traits)) + (j - (len(adapted) - 1) / 2) * w
        ys, err = [], [[], []]
        for t in traits:
            r = spearmanr(D["arms"][base][t]["holdout_per_persona"][li],
                          D["arms"][arm][t]["holdout_per_persona"][li]).statistic
            ys.append(r)
            band = (ci or {}).get(arm, {}).get("spearman_by_trait", {}) or {}
            if t in band:
                _m, lo, hi = band[t]
                err[0].append(max(r - lo, 0)), err[1].append(max(hi - r, 0))
        kw = dict(yerr=np.array(err), capsize=2.4, ecolor=INK_MUTED,
                  error_kw={"lw": 0.9}) if err[0] else {}
        ax.bar(xs, ys, w * 0.9, color=colours[arm], alpha=0.9, zorder=3,
               edgecolor=SURFACE, lw=0.7, label=arm, **kw)
    ax.axhline(0.0, color=INK_SECONDARY, lw=1.2, zorder=4)
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([t.replace("_", " ") for t in traits], fontsize=9.5,
                       color=INK_SECONDARY, rotation=18, ha="right")
    ax.set_ylabel("Spearman vs base\n(persona ordering on residuals)", fontsize=10,
                  color=INK_SECONDARY)
    ax.set_ylim(-1.05, 1.15)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=3)
    ax.set_title(f"Does the arm preserve WHICH personas sit where?  —  layer {L}",
                 fontsize=13, color=INK_PRIMARY, pad=12, loc="left", fontweight="semibold")
    # Caption states exactly what the intervals support and no more. The AGGREGATE difference
    # is resolved (impulsiveness +0.30 [0.08,0.49] vs goodness +0.75 [0.64,0.86], no overlap);
    # the per-trait inversion is NOT -- impulsivity under impulsiveness is -0.505 [-0.745,
    # +0.333], an interval that contains zero. Ten personas is too few to localise per trait.
    fig.text(0.006, 0.0,
             "Error bars: 95% question-resampling CI. `impulsiveness` reorders personas overall "
             "(+0.30 [0.08,0.49]) where goodness (+0.75) and mathematical (+0.73) do not — those "
             "intervals do not overlap.\nIts point estimates invert on impulsivity (its target) "
             "and risk_taking, but each single-trait interval still contains 0, so the "
             "localisation is suggestive, not established.",
             fontsize=8.6, color="#a04a2f", ha="left", va="top")
    fig.tight_layout()
    save(fig, outdir, f"ordering_preservation_L{L}")

    # ---- console summary ----
    print(f"\nlayer {L}")
    print(f"  {'arm':<15}{'raw':>8}{'Δraw':>9}{'hold':>8}{'Δhold':>9}{'share':>8}")
    for a in arms:
        raw = np.mean([g(a, t, "raw_mean") for t in traits])
        hold = means[a]
        sh = np.mean([g(a, t, "share") for t in traits])
        d_raw = raw - np.mean([g(base, t, "raw_mean") for t in traits])
        d_hold = hold - means[base]
        print(f"  {a:<15}{raw:>8.3f}{d_raw:>+9.3f}{hold:>8.3f}{d_hold:>+9.3f}{sh:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
