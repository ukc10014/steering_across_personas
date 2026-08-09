#!/usr/bin/env python3
"""Karty/Davies Figure 1, rebuilt on our Llama-3.1-8B CAA grid.

One column per trait; within a column, one dot per persona at its cosine to the
null-context trait vector, a horizontal rule at the persona mean, and the nonsense
control as a separate marker. Reference lines at 1.0 (a vector identical to the
assistant default) and 0.0 (orthogonal to it) carry the reading: the distance a
column falls below 1.0 IS the fan-out, and the control sitting near the top of every
column is what says the fan-out is not "a system prompt is present".

Pure re-analysis of outputs/{model}/analysis/caa_cosine_to_null.json -- no GPU, no
re-extraction. All 32 layers are in that file, so --layers is a free choice.

Two trait orderings, because they answer different questions:
  sorted  -- ascending by our own per-trait persona mean; shows which traits rotate
             most under persona conditioning, ordered by our data.
  kd      -- K/D's Figure 1 column order; the like-for-like visual comparison.

Usage:
    python scripts/plot_fig1_persona_fanout.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/plot_fig1_persona_fanout.py --model ... --layers 15 20 --orderings sorted kd
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

# Repo figure palette (shared with plot_cosine_heatmap.py). Blue/orange pair validated
# for CVD: normal-vision dE 37, worst simulated (protan) 24.6 -- well past the floor of 8.
# Colour is still never the only cue: the control also has its own marker shape and a
# legend entry.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8884"
SURFACE = "#fcfcfb"
PERSONA = "#1c4f8f"
CONTROL_COLOR = "#eb6834"

CONTROL = "nonsense"

# K/D Figure 1 column order, verbatim.
KD_ORDER = ["warmth", "empathy", "deference", "assertiveness",
            "impulsivity", "confidence", "honesty", "risk_taking"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K/D Figure 1 on our CAA grid")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20],
                   help="layers to render, one figure per layer per ordering")
    p.add_argument("--orderings", type=str, nargs="+", default=["sorted", "kd"],
                   choices=["sorted", "kd"])
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--no-labels", action="store_true",
                   help="suppress the direct label on each column's lowest persona")
    return p.parse_args()


def beeswarm_offsets(values: np.ndarray, half_width: float, min_sep: float) -> np.ndarray:
    """Deterministic x-offsets that separate points colliding in y.

    Random jitter would move a point every time the figure is rebuilt, which makes two
    renders of the same data non-comparable. This walks the points in y order and pushes
    each one sideways only far enough to clear its neighbours, alternating sides so the
    column stays visually centred and the spread carries no meaning of its own.
    """
    order = np.argsort(values)
    offsets = np.zeros_like(values, dtype=float)
    placed: list[tuple[float, float]] = []  # (value, offset)
    for i in order:
        v = values[i]
        for step in range(0, 24):
            # 0, +1, -1, +2, -2, ... in units of one marker width
            k = (step + 1) // 2 * (1 if step % 2 else -1)
            if step == 0:
                k = 0
            cand = k * min_sep
            if abs(cand) > half_width:
                continue
            if all(abs(v - pv) > min_sep or abs(cand - po) > min_sep * 0.9
                   for pv, po in placed):
                offsets[i] = cand
                break
        placed.append((v, offsets[i]))
    return offsets


def render(data: dict, traits: list[str], personas: list[str], layer: int,
           ordering: str, short: str, out_base: Path, labels: bool) -> None:
    P = np.array([[data["traits"][t]["personas"][p]["point"][layer] for t in traits]
                  for p in personas])                      # personas x traits
    control = np.array([data["traits"][t]["personas"][CONTROL]["point"][layer]
                        for t in traits])
    means = P.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    lo = min(P.min(), control.min(), 0.0)
    ax.set_ylim(lo - 0.13, 1.10)
    # Right margin is deliberate: it holds the reference-line labels clear of the marks.
    ax.set_xlim(-0.58, len(traits) + 0.30)

    # Alternating column bands, so the offset control diamond reads as belonging to the
    # column on its left rather than to the trait on its right.
    for j in range(0, len(traits), 2):
        ax.axvspan(j - 0.5, j + 0.5, color="#f2f1ed", lw=0, zorder=-1)

    # Reference lines next, so every mark sits on top of them.
    ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.axhline(0.0, color=INK_SECONDARY, lw=1.1, zorder=1)
    ax.text(len(traits) - 0.42, 1.008, "identical to null", va="bottom", ha="left",
            fontsize=8.5, color=INK_MUTED)
    ax.text(len(traits) - 0.42, 0.008, "orthogonal", va="bottom", ha="left",
            fontsize=8.5, color=INK_SECONDARY)

    half_w, sep = 0.30, 0.030
    for j in range(len(traits)):
        col = P[:, j]
        xs = j + beeswarm_offsets(col, half_w, sep)

        # persona mean, drawn under the dots
        ax.hlines(means[j], j - 0.40, j + 0.40, color=INK_PRIMARY, lw=2.0, zorder=3)

        ax.scatter(xs, col, s=52, color=PERSONA, alpha=0.85,
                   edgecolors=SURFACE, linewidths=1.6, zorder=4)
        ax.scatter([j + 0.44], [control[j]], s=95, marker="D", color=CONTROL_COLOR,
                   edgecolors=SURFACE, linewidths=1.6, zorder=5)

        if labels:
            k = int(np.argmin(col))
            ax.annotate(personas[k].replace("_", " "),
                        xy=(xs[k], col[k]), xytext=(0, -12),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7.8, color=INK_SECONDARY)

    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([t.replace("_", " ") for t in traits], fontsize=10.5)
    ax.set_ylabel(r"cos($v_{T,c}$,  $v_{T,\mathrm{null}}$)", fontsize=10.5,
                  color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    ax.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#dedcd7")

    handles = [
        Line2D([], [], marker="o", ls="none", markersize=7.5, color=PERSONA,
               markeredgecolor=SURFACE, label=f"persona ({len(personas)})"),
        Line2D([], [], color=INK_PRIMARY, lw=2.0, label="persona mean"),
        Line2D([], [], marker="D", ls="none", markersize=8.0, color=CONTROL_COLOR,
               markeredgecolor=SURFACE, label="nonsense (control)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5,
                    labelcolor=INK_SECONDARY, ncol=3, handletextpad=0.5,
                    columnspacing=1.6, bbox_to_anchor=(0.0, -0.155))
    leg.set_zorder(6)

    sub = ("traits ordered by our per-trait persona mean, ascending"
           if ordering == "sorted" else "traits in K/D Figure 1 order")
    ax.set_title(f"Persona trait vectors rotate away from the assistant default"
                 f"  —  {short}, layer {layer}",
                 fontsize=13, color=INK_PRIMARY, pad=22, loc="left",
                 fontweight="semibold")
    ax.text(0.0, 1.035, sub + ".  Labelled point = most-rotated persona in that column."
            if labels else sub + ".",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, ha="left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_base.with_suffix("." + ext)
        fig.savefig(path, dpi=220, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def print_table(data: dict, traits: list[str], personas: list[str], layer: int) -> None:
    P = np.array([[data["traits"][t]["personas"][p]["point"][layer] for t in traits]
                  for p in personas])
    control = np.array([data["traits"][t]["personas"][CONTROL]["point"][layer]
                        for t in traits])
    means, sds = P.mean(axis=0), P.std(axis=0, ddof=1)

    print(f"\nLayer {layer} — {len(personas)} personas, SD is the sample SD across them")
    print(f"{'trait':<15} {'mean':>7} {'SD':>7} {'nonsense':>9} {'gap':>7}  "
          f"{'lowest persona':<22} {'min':>7}")
    print("-" * 80)
    for j, t in enumerate(traits):
        k = int(np.argmin(P[:, j]))
        print(f"{t.replace('_', ' '):<15} {means[j]:>7.3f} {sds[j]:>7.3f} "
              f"{control[j]:>9.3f} {control[j] - means[j]:>7.3f}  "
              f"{personas[k].replace('_', ' '):<22} {P[k, j]:>7.3f}")
    print("-" * 80)
    print(f"{'ALL TRAITS':<15} {means.mean():>7.3f} {sds.mean():>7.3f} "
          f"{control.mean():>9.3f} {(control - means).mean():>7.3f}")


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    in_path = (Path(args.input) if args.input
               else OUTPUTS_DIR / short / "analysis" / "caa_cosine_to_null.json")
    outdir = Path(args.outdir) if args.outdir else OUTPUTS_DIR / short / "analysis"

    if not in_path.exists():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    all_traits = list(data["traits"].keys())
    all_series = list(data["traits"][all_traits[0]]["personas"].keys())
    personas = [p for p in all_series if p != CONTROL]

    n_layers = data["traits"][all_traits[0]]["n_layers"]
    for L in args.layers:
        if not 0 <= L < n_layers:
            print(f"error: layer {L} outside 0..{n_layers - 1}", file=sys.stderr)
            return 2

    missing = set(KD_ORDER) - set(all_traits)
    if missing and "kd" in args.orderings:
        print(f"error: K/D ordering needs traits not in the file: {sorted(missing)}",
              file=sys.stderr)
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    for L in args.layers:
        means = {t: float(np.mean([data["traits"][t]["personas"][p]["point"][L]
                                   for p in personas])) for t in all_traits}
        orders = {
            "sorted": sorted(all_traits, key=lambda t: means[t]),
            "kd": [t for t in KD_ORDER if t in all_traits],
        }
        for name in args.orderings:
            traits = orders[name]
            render(data, traits, personas, L, name, short,
                   outdir / f"fig1_persona_fanout_L{L}_{name}", not args.no_labels)
        print_table(data, orders["sorted"], personas, L)

    return 0


if __name__ == "__main__":
    sys.exit(main())
