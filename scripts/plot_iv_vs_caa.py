#!/usr/bin/env python3
"""CAA vs IV: the per-trait ordering reversal, and the tier test.

Neither single-method fan-out figure can show this, because the finding is about the
*relationship* between two orderings. Two panels:

Panel A -- a slopegraph. Persona-mean cosine-to-null under CAA on the left, IV on the right,
one line per trait. This form is chosen because it does both jobs at once: crossing lines are
rank changes (the Spearman -0.52), and the overall vertical shift between the two columns is
the magnitude difference (IV rotates personas far less than CAA). A scatter would show the
correlation but hide the shift; two bar charts would show the shift but hide the crossings.

Panel B -- the tier test, which is what K/D actually claim is preserved across methods. They
say the rank order moves but the split into a wider-spread tier and a tighter-spread tier is
"broadly preserved". A filled dot means the trait is in that method's four most-spread traits.
Reading down the columns shows the overlap directly.

Colour carries method identity only (the repo's validated blue/orange pair), never rank --
a rank-coloured ramp would restate the y-position and would repaint traits whenever the
ordering changed, which is exactly the thing under examination.

Usage:
    python scripts/plot_iv_vs_caa.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20
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
CAA_C = "#1c4f8f"
IV_C = "#eb6834"

CONTROL = "nonsense"
# K/D's stated IV wider-spread tier (Appendix G.1).
KD_IV_WIDE = {"impulsivity", "risk_taking", "deference", "warmth"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAA vs IV ordering comparison")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--caa", type=str, default=None)
    p.add_argument("--iv", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def persona_means(path: Path, layer: int) -> dict[str, float]:
    d = json.loads(path.read_text())
    return {t: float(np.mean([v["point"][layer]
                              for p, v in d["traits"][t]["personas"].items() if p != CONTROL]))
            for t in d["traits"]}


def spread_labels(values: list[float], lo: float, hi: float, min_gap: float) -> list[float]:
    """Nudge label positions apart so text does not overlap, preserving order.

    The traits bunch tightly under CAA -- empathy, assertiveness and confidence sit within
    0.02 of each other -- so labels drawn at their true y collide into an unreadable stack.
    This walks them in order and pushes each one up to clear its predecessor, then shifts the
    whole block back down if it overran the axis. Only the LABEL moves; the dot and the line
    stay at the true value, so nothing about the data reading changes.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    prev = lo - min_gap
    for i in order:
        y = max(values[i], prev + min_gap)
        out[i] = y
        prev = y
    overshoot = out[order[-1]] - hi
    if overshoot > 0:
        for i in order:
            out[i] -= overshoot
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    adir = OUTPUTS_DIR / short / "analysis"
    caa_p = Path(args.caa) if args.caa else adir / "caa_cosine_to_null.json"
    iv_p = Path(args.iv) if args.iv else adir / "iv" / "caa_cosine_to_null.json"
    outdir = Path(args.outdir) if args.outdir else adir
    for p in (caa_p, iv_p):
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 2

    L = args.layer
    C, I = persona_means(caa_p, L), persona_means(iv_p, L)
    traits = [t for t in C if t in I]
    rho = spearman([C[t] for t in traits], [I[t] for t in traits])

    caa_wide = set(sorted(traits, key=lambda t: C[t])[:4])
    iv_wide = set(sorted(traits, key=lambda t: I[t])[:4])

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.8, 6.9), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42})

    # ---------------- Panel A: slopegraph ----------------
    axA.set_facecolor(SURFACE)
    for t in traits:
        axA.plot([0, 1], [C[t], I[t]], color=INK_MUTED, lw=1.3, alpha=0.75, zorder=2)
    axA.scatter([0] * len(traits), [C[t] for t in traits], s=70, color=CAA_C,
                edgecolors=SURFACE, linewidths=1.5, zorder=4)
    axA.scatter([1] * len(traits), [I[t] for t in traits], s=70, color=IV_C,
                edgecolors=SURFACE, linewidths=1.5, zorder=4)

    axA.set_xlim(-0.62, 1.62)
    lo = min(min(C.values()), min(I.values())) - 0.06
    hi = max(max(C.values()), max(I.values())) + 0.06
    axA.set_ylim(lo, hi)

    gap = (hi - lo) * 0.055
    cy = spread_labels([C[t] for t in traits], lo, hi, gap)
    iy = spread_labels([I[t] for t in traits], lo, hi, gap)
    for t, ly, ry in zip(traits, cy, iy):
        axA.annotate(f"{t.replace('_',' ')}  {C[t]:.2f}", xy=(-0.04, ly),
                     ha="right", va="center", fontsize=9, color=INK_SECONDARY)
        axA.annotate(f"{I[t]:.2f}  {t.replace('_',' ')}", xy=(1.04, ry),
                     ha="left", va="center", fontsize=9, color=INK_SECONDARY)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["CAA", "IV"], fontsize=12, fontweight="semibold")
    axA.tick_params(colors=INK_SECONDARY, labelsize=10, length=0)
    axA.set_ylabel("persona-mean cos to null   (lower = more persona spread)",
                   fontsize=10, color=INK_SECONDARY)
    for s in ("top", "right", "bottom"):
        axA.spines[s].set_visible(False)
    axA.spines["left"].set_color("#dedcd7")
    axA.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    axA.set_title(f"A.  Same traits, two methods — layer {L}", fontsize=11.5,
                  color=INK_PRIMARY, pad=12, loc="left", fontweight="semibold")
    axA.text(0.5, 0.015, f"Spearman(rank$_{{CAA}}$, rank$_{{IV}}$) = {rho:+.2f}"
             "   —   crossing lines are rank changes",
             transform=axA.transAxes, ha="center", va="bottom", fontsize=9.5,
             color=INK_PRIMARY,
             bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#dedcd7", lw=0.9))

    # ---------------- Panel B: tier membership ----------------
    axB.set_facecolor(SURFACE)
    cols = [("ours\nCAA", caa_wide, CAA_C), ("ours\nIV", iv_wide, IV_C),
            ("K/D\nIV", KD_IV_WIDE, INK_SECONDARY)]
    order = sorted(traits, key=lambda t: I[t])          # IV order, loosest at top
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}

    for j, (_, members, colour) in enumerate(cols):
        for t in traits:
            y = ypos[t]
            if t in members:
                axB.scatter([j], [y], s=210, color=colour, edgecolors=SURFACE,
                            linewidths=1.6, zorder=3)
            else:
                axB.scatter([j], [y], s=210, facecolors="none", edgecolors="#d8d6d1",
                            linewidths=1.5, zorder=3)

    axB.set_xticks(range(len(cols)))
    axB.set_xticklabels([c[0] for c in cols], fontsize=10.5, color=INK_SECONDARY)
    axB.set_yticks(list(range(len(order))))
    axB.set_yticklabels([t.replace("_", " ") for t in reversed(order)], fontsize=10,
                        color=INK_SECONDARY)
    axB.set_xlim(-0.6, len(cols) - 0.4)
    axB.set_ylim(-0.6, len(order) - 0.4)
    axB.tick_params(length=0)
    for s in axB.spines.values():
        s.set_visible(False)
    axB.set_title("B.  In the wider-spread tier?", fontsize=11.5, color=INK_PRIMARY,
                  pad=12, loc="left", fontweight="semibold")
    ov = (f"overlap with K/D:   IV {len(iv_wide & KD_IV_WIDE)}/4    "
          f"CAA {len(caa_wide & KD_IV_WIDE)}/4    |    IV vs CAA {len(iv_wide & caa_wide)}/4")
    axB.text(0.0, -0.11, ov, transform=axB.transAxes, ha="left", va="top",
             fontsize=9.5, color=INK_PRIMARY)

    handles = [
        Line2D([], [], marker="o", ls="none", markersize=9, color=CAA_C,
               markeredgecolor=SURFACE, label="CAA"),
        Line2D([], [], marker="o", ls="none", markersize=9, color=IV_C,
               markeredgecolor=SURFACE, label="IV"),
        Line2D([], [], marker="o", ls="none", markersize=9, markerfacecolor="none",
               markeredgecolor="#d8d6d1", label="tighter tier"),
    ]
    axB.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5,
               labelcolor=INK_SECONDARY, ncol=3, handletextpad=0.4, columnspacing=1.2,
               bbox_to_anchor=(0.0, -0.24))

    fig.suptitle(f"Extraction method reorders the traits — {short}, layer {L}",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left",
                 fontweight="semibold", y=1.01)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"iv_vs_caa_L{L}.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    print(f"\nlayer {L}: spearman {rho:+.3f} | CAA mean {np.mean(list(C.values())):.3f} "
          f"| IV mean {np.mean(list(I.values())):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
