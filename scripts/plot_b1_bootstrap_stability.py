#!/usr/bin/env python3
"""B.1 bootstrap-stability figure, in the layout the upstream repo uses for it.

`pipeline/r1_bootstrap_vectors.py` renders bootstrap stability as (a) a persona x trait
heatmap of mean pairwise cosine across resamples and (b) a per-trait distribution. That is
K/D's rung-1 estimator, so this reproduces that view on the Llama-3.1-8B CAA grid, reading
the numbers already computed by scripts/caa_within_cell_stability.py.

Two deliberate departures from r1's version:

  * r1 uses RdYlGn with vmin=0.8. Red-green is the worst case for colour-vision deficiency
    and a diverging ramp is wrong for a quantity that is pure magnitude with no meaningful
    midpoint, so this uses a single-hue sequential ramp instead. The reading is unchanged;
    only the encoding is fixed.
  * r1 clips at 0.8 (heatmap) and 0.7 (boxplot). Our values run well below both, so those
    limits would clip real data. The scale is set from the data and K/D's reported 0.99 is
    drawn as an explicit reference instead -- which is the comparison that matters.

Usage:
    python scripts/plot_b1_bootstrap_stability.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import model_short_name

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8884"
SURFACE = "#fcfcfb"
FLOOR = "#1c4f8f"
SIGNAL = "#eb6834"

KD_RUNG1 = 0.99
CONTROLS = ("null", "nonsense")

# Sequential, single hue, light -> dark. Magnitude has no meaningful midpoint here.
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f6fa", "#9db8d8", "#4a7ab0", "#1c4f8f"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.1 bootstrap stability, r1-style layout")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--layer", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    in_path = (Path(args.input) if args.input
               else OUTPUTS_DIR / short / "analysis" / "caa_within_cell_stability.json")
    outdir = Path(args.outdir) if args.outdir else OUTPUTS_DIR / short / "analysis"
    if not in_path.exists():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    L = args.layer
    traits = list(data["traits"].keys())
    cells = sorted(data["traits"][traits[0]]["cells"].keys())
    personas = [c for c in cells if c not in CONTROLS]
    ordered = personas + [c for c in CONTROLS if c in cells]

    M = np.full((len(ordered), len(traits)), np.nan)
    for i, c in enumerate(ordered):
        for j, t in enumerate(traits):
            cell = data["traits"][t]["cells"].get(c)
            if cell:
                M[i, j] = cell["within_cell"]["mean"][L]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15.2, 6.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.22})

    # ---------------- Panel A: persona x trait heatmap ----------------
    vmin, vmax = float(np.nanmin(M)), 1.0
    im = axA.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    axA.set_xticks(range(len(traits)))
    axA.set_xticklabels([t.replace("_", " ") for t in traits], rotation=38, ha="right",
                        fontsize=9.5, color=INK_SECONDARY)
    axA.set_yticks(range(len(ordered)))
    axA.set_yticklabels([c.replace("_", " ") for c in ordered], fontsize=9.5,
                        color=INK_SECONDARY)

    # Separate the two control rows from the personas: they are a different kind of thing.
    if len(ordered) > len(personas):
        axA.axhline(len(personas) - 0.5, color=INK_PRIMARY, lw=1.6)

    span = max(vmax - vmin, 1e-6)
    for i in range(len(ordered)):
        for j in range(len(traits)):
            if np.isnan(M[i, j]):
                continue
            frac = (M[i, j] - vmin) / span
            axA.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.6,
                     color=SURFACE if frac > 0.55 else INK_PRIMARY)

    cb = fig.colorbar(im, ax=axA, shrink=0.82, pad=0.02)
    cb.set_label("within-cell bootstrap stability  (mean pairwise cosine)",
                 fontsize=9.5, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=8.5, colors=INK_SECONDARY)
    axA.set_title(f"A.  Every cell, layer {L} — no cell reaches K/D's 0.99",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")
    axA.tick_params(length=0)
    for s in axA.spines.values():
        s.set_visible(False)

    # ---------------- Panel B: per-trait distribution ----------------
    axB.set_facecolor(SURFACE)
    per_trait = [[data["traits"][t]["cells"][c]["within_cell"]["mean"][L]
                  for c in personas] for t in traits]
    order = np.argsort([np.mean(v) for v in per_trait])
    tl = [traits[k] for k in order]
    vals = [per_trait[k] for k in order]

    axB.axhline(KD_RUNG1, color=SIGNAL, lw=1.6, ls=(0, (5, 3)), zorder=2)
    axB.text(len(tl) - 0.35, KD_RUNG1 - 0.004, "K/D reported  0.99", ha="right", va="top",
             fontsize=9, color=SIGNAL, fontweight="semibold")

    rng = np.random.default_rng(0)
    for i, v in enumerate(vals):
        v = np.asarray(v)
        axB.hlines(v.mean(), i - 0.30, i + 0.30, color=INK_PRIMARY, lw=2.0, zorder=4)
        jitter = (rng.random(len(v)) - 0.5) * 0.26
        axB.scatter(np.full(len(v), i) + jitter, v, s=40, color=FLOOR, alpha=0.85,
                    edgecolors=SURFACE, linewidths=1.2, zorder=3)

    axB.set_xticks(range(len(tl)))
    axB.set_xticklabels([t.replace("_", " ") for t in tl], rotation=38, ha="right",
                        fontsize=9.5)
    axB.set_ylabel("mean pairwise cosine across bootstrap resamples", fontsize=10,
                   color=INK_SECONDARY)
    axB.set_ylim(min(0.70, float(np.nanmin(M)) - 0.03), 1.015)
    axB.set_xlim(-0.6, len(tl) - 0.4)
    axB.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    axB.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axB.spines[s].set_color("#dedcd7")
    axB.set_title(f"B.  By trait, layer {L} (10 personas, controls excluded)",
                  fontsize=11.5, color=INK_PRIMARY, pad=10, loc="left", fontweight="semibold")

    handles = [
        Line2D([], [], marker="o", ls="none", markersize=7, color=FLOOR,
               markeredgecolor=SURFACE, label="one persona"),
        Line2D([], [], color=INK_PRIMARY, lw=2.0, label="trait mean"),
        Line2D([], [], color=SIGNAL, lw=1.6, ls=(0, (5, 3)), label="K/D reported rung 1"),
    ]
    axB.legend(handles=handles, loc="lower left", frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, ncol=3, handletextpad=0.5, columnspacing=1.4,
               bbox_to_anchor=(0.0, -0.30))

    fig.suptitle(f"B.1 rung 1 — extraction noise floor, {short} CAA, "
                 f"{data['n_boot']} bootstrap resamples per cell",
                 fontsize=13.5, color=INK_PRIMARY, x=0.006, ha="left",
                 fontweight="semibold", y=1.005)

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"b1_bootstrap_stability_L{L}.{ext}"
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    flat = M[:len(personas)].ravel()
    print(f"\nlayer {L}: {len(personas)} personas x {len(traits)} traits")
    print(f"  min {np.nanmin(flat):.3f}   mean {np.nanmean(flat):.3f}   max {np.nanmax(flat):.3f}")
    print(f"  cells at or above K/D's 0.99: {int((flat >= KD_RUNG1).sum())} / {flat.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
