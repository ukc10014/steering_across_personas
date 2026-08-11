#!/usr/bin/env python3
"""Base vs character-trained arm: the compression, and whether it is licensed.

Two panels, because the result is only meaningful if the second one passes.

Panel A -- paired strip plot. For each trait, the ten persona cosines under the base model and
under the adapted model, side by side on one scale. This form is chosen over a slopegraph of
the means because the finding is about the SPREAD as much as the location: what you want to
see is the column collapsing, not just sliding up. Individual personas are drawn so an extreme
cell (drill_sergeant's warmth at -0.14 under base) is visible rather than averaged away.

Panel B -- the confound check, and the reason panel A can be believed at all. Each arm has its
own measurement wobble (the null-vs-null cosine). If character training changed that wobble,
a narrower column in panel A could be a better instrument rather than a real effect. Plotting
both arms' wobble across depth, with the ~0.02 decision threshold shaded, turns fork-infra
13.5's either/or into something you can read off directly.

Colour separates the two arms only. The shaded threshold band is deliberately neutral grey --
it is a decision rule, not data.

Usage:
    python scripts/plot_arm_comparison.py --base meta-llama/Llama-3.1-8B-Instruct \
        --adapted /workspace/merged/llama-3.1-8b-goodness --layer 15
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import model_short_name

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8884"
SURFACE = "#fcfcfb"
BASE_C = "#1c4f8f"
ADAPT_C = "#0f8a6a"
CONTROL_C = "#eb6834"

EXCLUDE = {"nonsense", "null"}
# fork-infra 13.4: the floor estimate swings ~0.02 on seed choice alone at n_boot=50, so a
# difference smaller than this cannot be told apart from scatter.
THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Base vs adapted arm comparison")
    p.add_argument("--base", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapted", type=str, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--label", type=str, default=None, help="name for the adapted arm")
    return p.parse_args()


def load(model: str) -> dict:
    path = OUTPUTS_DIR / model_short_name(model) / "analysis" / "caa_cosine_to_null.json"
    if not path.exists():
        raise SystemExit(f"error: {path} not found")
    return json.loads(path.read_text())


def personas(d: dict, trait: str, layer: int) -> np.ndarray:
    return np.array([v["point"][layer] for k, v in d["traits"][trait]["personas"].items()
                     if k not in EXCLUDE])


def main() -> int:
    args = parse_args()
    B, A = load(args.base), load(args.adapted)
    label = args.label or model_short_name(args.adapted).split("-")[-1]
    L = args.layer
    traits = [t for t in B["traits"] if t in A["traits"]]
    # order by the base arm's mean, loosest first -- the adapted ordering is one of the things
    # under examination, so it must not set the axis.
    traits.sort(key=lambda t: personas(B, t, L).mean())

    outdir = Path(args.outdir) if args.outdir else OUTPUTS_DIR / model_short_name(args.adapted) / "analysis"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.6, 6.6), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.24})

    # ---------------- Panel A ----------------
    axA.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for j, t in enumerate(traits):
        for k, (d, colour) in enumerate(((B, BASE_C), (A, ADAPT_C))):
            x = j + (-0.19 if k == 0 else 0.19)
            pts = personas(d, t, L)
            axA.scatter(x + rng.uniform(-0.055, 0.055, pts.size), pts, s=32, color=colour,
                        alpha=0.75, edgecolors=SURFACE, linewidths=0.7, zorder=3)
            axA.hlines(pts.mean(), x - 0.135, x + 0.135, color=INK_PRIMARY, lw=2.1, zorder=4)
            ctl = d["traits"][t]["personas"].get("nonsense")
            if ctl:
                axA.scatter([x], [ctl["point"][L]], marker="D", s=34, color=CONTROL_C,
                            edgecolors=SURFACE, linewidths=0.8, zorder=5)
        if j % 2 == 0:
            axA.axvspan(j - 0.5, j + 0.5, color="#f2f1ed", lw=0, zorder=-1)

    axA.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
    axA.axhline(0.0, color=INK_SECONDARY, lw=1.1, zorder=1)
    axA.text(len(traits) - 0.45, 1.005, "identical to null", fontsize=8.5, color=INK_MUTED,
             ha="right", va="bottom")
    axA.text(len(traits) - 0.45, 0.012, "orthogonal", fontsize=8.5, color=INK_SECONDARY,
             ha="right", va="bottom")
    axA.set_xticks(range(len(traits)))
    axA.set_xticklabels([t.replace("_", " ") for t in traits], fontsize=9.5,
                        color=INK_SECONDARY, rotation=18, ha="right")
    axA.set_xlim(-0.5, len(traits) - 0.5)
    axA.set_ylabel(r"cos($v_{T,c}$, $v_{T,\mathrm{null}}$)  —  each arm vs its OWN null",
                   fontsize=10, color=INK_SECONDARY)
    axA.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for s in ("top", "right", "bottom"):
        axA.spines[s].set_visible(False)
    axA.spines["left"].set_color("#dedcd7")
    axA.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    axA.set_axisbelow(True)

    bm = np.mean([personas(B, t, L).mean() for t in traits])
    am = np.mean([personas(A, t, L).mean() for t in traits])
    axA.set_title(f"A.  Personas collapse toward the default under character training"
                  f"   (layer {L})",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="none", markersize=7, color=BASE_C, label=f"base  (mean {bm:.3f})"),
        Line2D([], [], marker="o", ls="none", markersize=7, color=ADAPT_C, label=f"{label}  (mean {am:.3f})"),
        Line2D([], [], marker="D", ls="none", markersize=6, color=CONTROL_C, label="nonsense control"),
        Line2D([], [], color=INK_PRIMARY, lw=2.1, label="persona mean"),
    ], loc="lower left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2,
        bbox_to_anchor=(0.0, -0.30))

    # ---------------- Panel B ----------------
    axB.set_facecolor(SURFACE)
    n_layers = len(B["traits"][traits[0]]["noise_floor"]["mean"])
    xs = np.arange(n_layers)
    fb = np.array([[B["traits"][t]["noise_floor"]["mean"][l] for t in traits] for l in xs]).mean(1)
    fa = np.array([[A["traits"][t]["noise_floor"]["mean"][l] for t in traits] for l in xs]).mean(1)

    axB.fill_between(xs, fb - THRESHOLD, fb + THRESHOLD, color="#e3e2de", lw=0, zorder=1,
                     label=f"±{THRESHOLD} of base\n(indistinguishable)")
    axB.plot(xs, fb, color=BASE_C, lw=2.0, zorder=3, label="base")
    axB.plot(xs, fa, color=ADAPT_C, lw=2.0, zorder=3, label=label)

    # Staggered heights and alternating sides: at n_layers=32 the L15 and L20 verticals are
    # close enough that same-height labels overlap each other and the legend.
    for lay, yfrac, ha, dx in ((15, 0.40, "right", -0.4), (20, 0.22, "left", 0.4)):
        if lay < n_layers:
            ok = abs(fa[lay] - fb[lay]) <= THRESHOLD
            axB.axvline(lay, color=INK_MUTED, lw=0.9, ls=(0, (2, 3)), zorder=2)
            axB.annotate(f"L{lay}  Δ={fa[lay]-fb[lay]:+.3f}\n{'licensed' if ok else 'NOT licensed'}",
                         xy=(lay + dx, yfrac), xycoords=("data", "axes fraction"),
                         fontsize=8.5, ha=ha, va="center", color=INK_PRIMARY, zorder=6,
                         bbox=dict(boxstyle="round,pad=0.32", fc=SURFACE,
                                   ec="#5f9e77" if ok else "#c2705a", lw=1.2))

    axB.set_xlabel("layer", fontsize=10, color=INK_SECONDARY)
    axB.set_ylabel("measurement wobble  (null-vs-null cosine)\nhigher = better determined",
                   fontsize=9.5, color=INK_SECONDARY)
    axB.set_xlim(0, n_layers - 1)
    axB.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axB.spines[s].set_color("#dedcd7")
    axB.grid(color="#e8e7e3", lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.set_title("B.  Is the comparison licensed?", fontsize=11.5, color=INK_PRIMARY,
                  pad=10, loc="left", fontweight="semibold")
    axB.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(1.0, -0.02))

    fig.suptitle(f"Character training ({label}) compresses persona-conditional trait vectors"
                 f" — {model_short_name(args.base)}",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left",
                 fontweight="semibold", y=1.005)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"arm_comparison_L{L}.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    print(f"\nlayer {L}:  base mean {bm:.3f}   {label} mean {am:.3f}   shift {am-bm:+.3f}")
    print(f"wobble:     base {fb[L]:.3f}   {label} {fa[L]:.3f}   diff {fa[L]-fb[L]:+.3f}"
          f"   -> {'LICENSED' if abs(fa[L]-fb[L]) <= THRESHOLD else 'NOT licensed'} at this layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
