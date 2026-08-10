#!/usr/bin/env python3
"""Magnitude figure: how long persona trait vectors are, and the B.6 decoupling test.

Panel A -- persona x trait, log2(||v_persona|| / ||v_null||) at one layer. DIVERGING about
zero, because zero is a real boundary here (same length as the assistant default) rather
than an arbitrary midpoint: blue = shorter than null, orange = longer, neutral grey at 0.

Panel B -- the B.6 test. Each dot is one persona x trait cell: how far it rotated
(cosine-to-null, x) against whether it lengthened or shortened (log2 ratio, y). K/D claim
these decouple, which predicts a shapeless cloud. The fitted line and the cluster-bootstrap
CI on r say whether that holds. The nonsense control is drawn separately -- it should sit
top-right, unrotated and unchanged in length.

Usage:
    python scripts/plot_magnitude.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import model_short_name

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8884"
SURFACE = "#fcfcfb"
PERSONA = "#1c4f8f"
CONTROL_COLOR = "#eb6834"

NULL_SLUG, CONTROL_SLUG = "null", "nonsense"

# Diverging: two hues with a NEUTRAL grey midpoint. Never a hue at the midpoint, and
# never a rainbow -- zero has to read as "no change", not as another category.
DIV = LinearSegmentedColormap.from_list(
    "div_bo", ["#12385f", "#1c4f8f", "#7fa2c9", "#e8e6e1", "#f2a878", "#eb6834", "#a8410f"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Magnitude + B.6 decoupling figure")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--n-cluster", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    adir = OUTPUTS_DIR / short / "analysis"
    outdir = Path(args.outdir) if args.outdir else adir
    mag_p, cos_p = adir / "caa_magnitude.json", adir / "caa_cosine_to_null.json"
    for p in (mag_p, cos_p):
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 2

    mag = json.loads(mag_p.read_text())
    cos = json.loads(cos_p.read_text())
    L = args.layer
    traits = list(mag["traits"].keys())
    personas = [c for c in mag["traits"][traits[0]]["cells"] if c not in (NULL_SLUG, CONTROL_SLUG)]

    M = np.array([[mag["traits"][t]["cells"][p]["log2_ratio_to_null"][L] for t in traits]
                  for p in personas])

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15.0, 6.3), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.42, 1.0], "wspace": 0.24})

    # ---------------- Panel A ----------------
    lim = float(np.abs(M).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = axA.imshow(M, cmap=DIV, norm=norm, aspect="auto")
    axA.set_xticks(range(len(traits)))
    axA.set_xticklabels([t.replace("_", " ") for t in traits], rotation=38, ha="right",
                        fontsize=9.5, color=INK_SECONDARY)
    axA.set_yticks(range(len(personas)))
    axA.set_yticklabels([p.replace("_", " ") for p in personas], fontsize=9.5,
                        color=INK_SECONDARY)
    for i in range(len(personas)):
        for j in range(len(traits)):
            v = M[i, j]
            axA.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.4,
                     color=SURFACE if abs(v) > 0.62 * lim else INK_PRIMARY)
    cb = fig.colorbar(im, ax=axA, shrink=0.84, pad=0.02)
    cb.set_label("log2( ||v persona||  /  ||v null|| )", fontsize=9.5, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=8.5, colors=INK_SECONDARY)
    axA.set_title(f"A.  Vector length vs the assistant default, layer {L}",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")
    axA.tick_params(length=0)
    for s in axA.spines.values():
        s.set_visible(False)

    # ---------------- Panel B: the B.6 test ----------------
    axB.set_facecolor(SURFACE)
    xs, ys, by_persona = [], [], {p: ([], []) for p in personas}
    for t in traits:
        for p in personas:
            x = cos["traits"][t]["personas"][p]["point"][L]
            y = mag["traits"][t]["cells"][p]["log2_ratio_to_null"][L]
            xs.append(x); ys.append(y)
            by_persona[p][0].append(x); by_persona[p][1].append(y)
    xs, ys = np.array(xs), np.array(ys)

    cx = [cos["traits"][t]["personas"][CONTROL_SLUG]["point"][L] for t in traits
          if CONTROL_SLUG in cos["traits"][t]["personas"]]
    cy = [mag["traits"][t]["cells"][CONTROL_SLUG]["log2_ratio_to_null"][L] for t in traits
          if CONTROL_SLUG in mag["traits"][t]["cells"]]

    axB.axhline(0.0, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
    axB.text(axB.get_xlim()[0], 0.012, "same length as null", fontsize=8.5,
             color=INK_MUTED, va="bottom")

    axB.scatter(xs, ys, s=46, color=PERSONA, alpha=0.8, edgecolors=SURFACE,
                linewidths=1.1, zorder=3)
    if cx:
        axB.scatter(cx, cy, s=92, marker="D", color=CONTROL_COLOR, edgecolors=SURFACE,
                    linewidths=1.4, zorder=5)

    r0 = pearson(xs, ys)
    rng = np.random.default_rng(args.seed)
    rs = []
    for _ in range(args.n_cluster):
        idx = rng.integers(0, len(personas), len(personas))
        bx = np.concatenate([np.array(by_persona[personas[i]][0]) for i in idx])
        by = np.concatenate([np.array(by_persona[personas[i]][1]) for i in idx])
        rs.append(pearson(bx, by))
    lo, hi = np.percentile(rs, [2.5, 97.5])

    b, a = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 50)
    axB.plot(xr, a + b * xr, color=INK_PRIMARY, lw=1.8, zorder=4)

    axB.set_xlabel(r"cos($v_{T,c}$, $v_{T,\mathrm{null}}$)   —   rotation away from null",
                   fontsize=10, color=INK_SECONDARY)
    axB.set_ylabel("log2 magnitude ratio to null", fontsize=10, color=INK_SECONDARY)
    axB.set_title(f"B.  B.6 decoupling test, layer {L}", fontsize=11.5, color=INK_PRIMARY,
                  pad=10, loc="left", fontweight="semibold")
    axB.grid(color="#e8e7e3", lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axB.spines[s].set_color("#dedcd7")

    verdict = "not decoupled" if (lo > 0 or hi < 0) else "consistent with decoupling"
    axB.text(0.03, 0.97,
             f"r = {r0:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]\n"
             f"cluster bootstrap over {len(personas)} personas\n{verdict}",
             transform=axB.transAxes, va="top", ha="left", fontsize=9,
             color=INK_PRIMARY,
             bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec="#dedcd7", lw=0.9))

    handles = [
        Line2D([], [], marker="o", ls="none", markersize=7, color=PERSONA,
               markeredgecolor=SURFACE, label=f"persona x trait ({len(xs)} cells)"),
        Line2D([], [], marker="D", ls="none", markersize=7.5, color=CONTROL_COLOR,
               markeredgecolor=SURFACE, label="nonsense control"),
        Line2D([], [], color=INK_PRIMARY, lw=1.8, label="least squares fit"),
    ]
    axB.legend(handles=handles, loc="lower left", frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, ncol=1, handletextpad=0.5,
               bbox_to_anchor=(0.02, 0.02))

    fig.suptitle(f"CAA trait-vector magnitude — {short}, layer {L}",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left",
                 fontweight="semibold", y=1.005)
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"magnitude_L{L}.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    print(f"\nlayer {L}: r = {r0:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
