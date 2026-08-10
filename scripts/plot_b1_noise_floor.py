#!/usr/bin/env python3
"""K/D Appendix B.1, rebuilt on our Llama-3.1-8B CAA grid.

B.1's argument is a hierarchy: a trait vector re-estimated from resampled data is very
stable (rung 1), while the same trait vector across personas is not (rung 3), so the
across-persona spread is signal rather than extraction noise. This figure plots that
hierarchy at every layer, and adds the stricter floor B.1 does not compute.

Panel A -- the hierarchy by layer. The gap between the floor and the across-persona line
IS the claim; it is shaded rather than left to the reader to subtract. K/D's reported
rung-1 value (0.99) is drawn as a reference because we do not reach it anywhere.

Panel B -- the same gap per trait at the reported layer, as a dumbbell, so the traits
that carry the effect are separable from those that do not.

Colour: the repo's validated blue/orange pair (see plot_fig1_persona_fanout.py) plus one
lighter blue step for the second floor. All three pairs re-validated for CVD separation
and contrast against the surface; the two floors also differ in line style and carry
direct labels, so identity never rests on hue alone.

Usage:
    python scripts/plot_b1_noise_floor.py --model meta-llama/Llama-3.1-8B-Instruct
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
BAND = "#f2f1ed"

# Validated (OKLab dE, Vienot CVD sim, contrast vs surface):
#   1c4f8f/5688bf normal 30.4  cvd_min 28.9   1c4f8f/eb6834 normal 37.0  cvd_min 24.6
#   5688bf/eb6834 normal 27.1  cvd_min 19.2   contrast 7.98 / 3.61 / 3.12 : 1
FLOOR = "#1c4f8f"          # rung 1, within-cell bootstrap
FLOOR_STRICT = "#5688bf"   # split-half, Spearman-Brown corrected
SIGNAL = "#eb6834"         # rung 3, across-persona

KD_RUNG1 = 0.99            # K/D's reported within-cell stability
EXCLUDE = ("null", "nonsense")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K/D Appendix B.1 figure")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--layer", type=int, default=20,
                   help="layer for panel B and the annotated readout (default: 20)")
    p.add_argument("--mark-layers", type=int, nargs="+", default=[15, 20])
    return p.parse_args()


def persona_mean(data: dict, trait: str, stat: str) -> np.ndarray:
    cells = data["traits"][trait]["cells"]
    rows = [cells[c][stat]["mean"] for c in cells if c not in EXCLUDE]
    return np.nanmean(np.array(rows, dtype=float), axis=0)


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    in_path = (Path(args.input) if args.input
               else OUTPUTS_DIR / short / "analysis" / "caa_within_cell_stability.json")
    outdir = Path(args.outdir) if args.outdir else OUTPUTS_DIR / short / "analysis"
    if not in_path.exists():
        print(f"error: {in_path} not found -- run scripts/caa_within_cell_stability.py first",
              file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    traits = list(data["traits"].keys())
    n_layers = data["traits"][traits[0]]["n_layers"]
    L = args.layer
    x = np.arange(n_layers)

    within = np.nanmean([persona_mean(data, t, "within_cell") for t in traits], axis=0)
    strict = np.nanmean([persona_mean(data, t, "split_half_sb") for t in traits], axis=0)
    across = np.nanmean([np.array(data["traits"][t]["across_cell"]["mean"], dtype=float)
                         for t in traits], axis=0)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.4, 5.9), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.62, 1.0], "wspace": 0.30})

    # ================= Panel A: hierarchy by layer =================
    axA.set_facecolor(SURFACE)
    axA.set_ylim(0.0, 1.06)
    axA.set_xlim(-0.6, n_layers - 0.4)

    # The gap is the claim, so it is a filled region rather than a subtraction the
    # reader has to do by eye. Filled ONLY where the hierarchy actually holds: below the
    # crossover the across-persona line sits ABOVE the floor, and shading that region too
    # would paint the failure case in the same ink as the result.
    axA.fill_between(x, across, within, where=(within > across), interpolate=True,
                     color=SIGNAL, alpha=0.11, lw=0, zorder=1)

    cross = int(np.argmax(within > across))
    axA.axvline(cross, color=INK_SECONDARY, lw=1.0, zorder=2)
    axA.annotate(f"hierarchy only holds\nfrom layer {cross}", xy=(cross, 0.30),
                 xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                 fontsize=8.5, color=INK_SECONDARY)

    axA.axhline(KD_RUNG1, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
    axA.text(n_layers - 0.7, KD_RUNG1 + 0.008, "K/D reported rung 1  (0.99)",
             ha="right", va="bottom", fontsize=8.5, color=INK_MUTED)

    for mark in args.mark_layers:
        axA.axvline(mark, color=INK_MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1, alpha=0.8)

    axA.plot(x, within, color=FLOOR, lw=2.0, zorder=5)
    axA.plot(x, strict, color=FLOOR_STRICT, lw=2.0, ls=(0, (5, 2.5)), zorder=5)
    axA.plot(x, across, color=SIGNAL, lw=2.0, zorder=5)

    # Direct labels: identity does not rest on hue, and the light blue's contrast
    # WARN is discharged by naming its line rather than relying on the swatch.
    axA.annotate("within-cell (rung 1)", xy=(n_layers - 1, within[-1]),
                 xytext=(-4, 10), textcoords="offset points", ha="right",
                 fontsize=9, color=FLOOR, fontweight="semibold")
    axA.annotate("split-half floor", xy=(n_layers - 1, strict[-1]),
                 xytext=(-4, -14), textcoords="offset points", ha="right",
                 fontsize=9, color=FLOOR_STRICT, fontweight="semibold")
    axA.annotate("across-persona (rung 3)", xy=(n_layers - 1, across[-1]),
                 xytext=(-4, -14), textcoords="offset points", ha="right",
                 fontsize=9, color=SIGNAL, fontweight="semibold")

    for mark, note in zip(args.mark_layers, ("pre-designated", "reported")):
        axA.annotate(f"L{mark}\n{note}", xy=(mark, 0.035), xytext=(0, 0),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8, color=INK_SECONDARY)

    axA.set_xlabel("layer", fontsize=10.5, color=INK_SECONDARY)
    axA.set_ylabel("cosine", fontsize=10.5, color=INK_SECONDARY)
    axA.set_title("A.  The hierarchy by layer", fontsize=11, color=INK_PRIMARY,
                  pad=10, loc="left", fontweight="semibold")
    axA.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    axA.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axA.spines[side].set_color("#dedcd7")

    # ================= Panel B: per-trait gap at layer L =================
    axB.set_facecolor(SURFACE)
    w = {t: float(persona_mean(data, t, "within_cell")[L]) for t in traits}
    a = {t: float(np.array(data["traits"][t]["across_cell"]["mean"], dtype=float)[L])
         for t in traits}
    order = sorted(traits, key=lambda t: w[t] - a[t])
    ypos = np.arange(len(order))

    for i, t in enumerate(order):
        axB.plot([a[t], w[t]], [i, i], color=INK_MUTED, lw=1.4, zorder=2, alpha=0.75)
    axB.scatter([a[t] for t in order], ypos, s=74, color=SIGNAL,
                edgecolors=SURFACE, linewidths=1.6, zorder=4)
    axB.scatter([w[t] for t in order], ypos, s=74, color=FLOOR,
                edgecolors=SURFACE, linewidths=1.6, zorder=4)

    for i, t in enumerate(order):
        axB.annotate(f"{w[t] - a[t]:.2f}", xy=(w[t], i), xytext=(9, 0),
                     textcoords="offset points", va="center", fontsize=8.5,
                     color=INK_SECONDARY)

    axB.set_yticks(ypos)
    axB.set_yticklabels([t.replace("_", " ") for t in order], fontsize=10)
    axB.set_ylim(-0.7, len(order) - 0.3)
    axB.set_xlim(0.30, 1.02)
    axB.set_xlabel("cosine", fontsize=10.5, color=INK_SECONDARY)
    axB.set_title(f"B.  Per-trait margin, layer {L}", fontsize=11, color=INK_PRIMARY,
                  pad=10, loc="left", fontweight="semibold")
    axB.grid(axis="x", color="#e8e7e3", lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for side in ("top", "right", "left"):
        axB.spines[side].set_visible(False)
    axB.spines["bottom"].set_color("#dedcd7")

    handles = [
        Line2D([], [], color=FLOOR, lw=2.0, label="within-cell bootstrap (K/D rung 1)"),
        Line2D([], [], color=FLOOR_STRICT, lw=2.0, ls=(0, (5, 2.5)),
               label="split-half question bank (ours, stricter)"),
        Line2D([], [], color=SIGNAL, lw=2.0, label="across-persona spread (K/D rung 3)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=9.5,
                     labelcolor=INK_SECONDARY, ncol=3, handletextpad=0.6,
                     columnspacing=2.2, bbox_to_anchor=(0.5, -0.005))
    leg.set_zorder(6)

    fig.suptitle(f"K/D Appendix B.1 noise floor — {short}, CAA, {data['n_boot']} bootstrap "
                 f"replicates.  The floor never reaches K/D's 0.99 at any layer.",
                 fontsize=13.5, color=INK_PRIMARY, x=0.008, ha="left",
                 fontweight="semibold", y=1.005)

    fig.tight_layout(rect=(0, 0.045, 1, 0.985))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"b1_noise_floor.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
