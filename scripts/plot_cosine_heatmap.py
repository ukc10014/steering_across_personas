#!/usr/bin/env python3
"""Full-grid view: persona x trait cosine-to-null at a chosen layer, as a heatmap.

The by-layer line chart (scripts/plot_cosine_to_null.py) stops working past a handful
of series. At full grid -- 8 traits x 11 series -- the readable view is a matrix:
personas as rows, traits as columns, one cell per (persona, trait) cosine.

Color is DIVERGING about zero, not sequential, because zero is a real boundary here:
a cosine of 0 means the persona's trait vector is orthogonal to the assistant default,
and negative means it points against it. Neutral gray sits at 0, so "fully rotated
away" reads as the pale midpoint and "unrotated" reads as saturated blue.

Usage:
    python scripts/plot_cosine_heatmap.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import model_short_name

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

# Diverging pair from the reference palette: orange pole / blue pole, neutral midpoint.
NEG_POLE = "#eb6834"
MID = "#f0efec"
POS_POLE = "#1c4f8f"
DIVERGING = LinearSegmentedColormap.from_list("cos_div", [NEG_POLE, MID, POS_POLE])

CONTROL = "nonsense"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persona x trait cosine-to-null heatmap")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    in_path = Path(args.input) if args.input else OUTPUTS_DIR / short / "analysis" / "caa_cosine_to_null.json"
    out_path = Path(args.output) if args.output else OUTPUTS_DIR / short / "analysis" / f"cosine_heatmap_L{args.layer}.png"

    if not in_path.exists():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    traits = list(data["traits"].keys())
    first = data["traits"][traits[0]]
    personas = list(first["personas"].keys())

    L = args.layer
    M = np.array([[data["traits"][t]["personas"][p]["point"][L] for t in traits]
                  for p in personas])

    # Order personas by mean cosine: most-rotated at the top, so the control lands at
    # the bottom as the natural reference row rather than being pinned there by hand.
    order = np.argsort(M.mean(1))
    personas = [personas[i] for i in order]
    M = M[order]

    fig_w = 1.05 * len(traits) + 3.6
    fig_h = 0.52 * len(personas) + 2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    vmax = max(1.0, float(np.abs(M).max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([t.replace("_", " ") for t in traits], rotation=32,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(personas)))
    ax.set_yticklabels([
        (p.replace("_", " ") + "  (control)") if p == CONTROL else p.replace("_", " ")
        for p in personas
    ])
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)

    # 2px surface gap between cells
    ax.set_xticks(np.arange(len(traits) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(personas) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # value in every cell -- the matrix is small enough that the numbers are the point,
    # and they carry the reading when color alone is ambiguous
    for i in range(len(personas)):
        for j in range(len(traits)):
            v = M[i, j]
            shade = norm(v)
            ink = "#ffffff" if (shade > 0.80 or shade < 0.16) else INK_PRIMARY
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.6, color=ink)

    # separate the control row from the personas
    if CONTROL in personas:
        ci = personas.index(CONTROL)
        edge = ci - 0.5 if ci > len(personas) / 2 else ci + 0.5
        ax.axhline(edge, color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)))

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label(f"cos(v$_{{trait,persona}}$, v$_{{trait,null}}$) at layer {L}",
                   fontsize=9, color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    cbar.outline.set_visible(False)

    ax.set_title(
        f"Persona trait vectors vs the assistant default  —  {short}, layer {L}",
        fontsize=13, color=INK_PRIMARY, pad=14, loc="left", fontweight="semibold")
    fig.text(0.005, 0.955,
             "Pale = rotated to orthogonal. Rows ordered by mean cosine, most-rotated first.",
             fontsize=9, color=INK_SECONDARY, ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
