#!/usr/bin/env python3
"""Plot cosine-to-null by layer, one panel per trait.

Reads the JSON written by scripts/caa_cosine_to_null.py and renders a small-multiple
line chart: layer on x, cos(v_persona, v_null) on y, one line per persona, with the
bootstrap CI as a translucent band and the null-vs-null noise floor as a dashed
reference line.

The noise floor is deliberately NOT given a categorical hue -- it is a reference,
not an entity, so it reads as gray and dashed.

Usage:
    python scripts/plot_cosine_to_null.py --model meta-llama/Llama-3.1-8B-Instruct
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import model_short_name

# Categorical slots 1-4 of the validated reference palette, in fixed order.
# Assigned to personas by a fixed series order below -- never cycled, and never
# reassigned when the series set changes.
SERIES_COLORS = {
    "therapist": "#2a78d6",       # slot 1 blue
    "drill_sergeant": "#008300",  # slot 2 green
    "farmer": "#e87ba4",          # slot 3 magenta
    "nonsense": "#eda100",        # slot 4 yellow
}
SERIES_ORDER = ["therapist", "drill_sergeant", "farmer", "nonsense"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

LABELS = {
    "therapist": "therapist",
    "drill_sergeant": "drill sergeant",
    "farmer": "farmer",
    "nonsense": "nonsense (control)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot cosine-to-null by layer")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-ci", action="store_true", help="Hide bootstrap CI bands")
    return p.parse_args()


def declutter(ys: list[float], min_gap: float) -> list[float]:
    """Nudge direct-label y-positions apart, preserving order."""
    order = np.argsort(ys)
    out = list(ys)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    in_path = Path(args.input) if args.input else OUTPUTS_DIR / short / "analysis" / "caa_cosine_to_null.json"
    out_path = Path(args.output) if args.output else OUTPUTS_DIR / short / "analysis" / "cosine_to_null_by_layer.png"

    if not in_path.exists():
        print(f"error: {in_path} not found -- run scripts/caa_cosine_to_null.py first", file=sys.stderr)
        return 2

    data = json.loads(in_path.read_text())
    traits = list(data["traits"].keys())
    headline = data["headline_layer"]
    n_boot = data["n_boot"]

    fig, axes = plt.subplots(
        1, len(traits), figsize=(6.2 * len(traits), 5.0), sharey=True,
        facecolor=SURFACE,
    )
    if len(traits) == 1:
        axes = [axes]

    for ax, trait in zip(axes, traits):
        tr = data["traits"][trait]
        n_layers = tr["n_layers"]
        x = np.arange(n_layers)
        ax.set_facecolor(SURFACE)

        # zero line: cosine crossing 0 is meaningful (vector fully rotated away)
        ax.axhline(0.0, color=INK_MUTED, lw=1.0, alpha=0.5, zorder=1)

        # headline layer marker
        ax.axvline(headline, color=INK_MUTED, lw=1.0, ls=(0, (2, 3)), alpha=0.8, zorder=1)
        ax.text(headline + 0.4, 1.045, f"layer {headline}", fontsize=8.5,
                color=INK_SECONDARY, va="top", ha="left")

        # noise floor -- reference, not a series: gray + dashed
        floor = np.array(tr["noise_floor"]["mean"])
        ax.plot(x, floor, color=INK_SECONDARY, lw=1.6, ls=(0, (5, 3)),
                zorder=3, label="noise floor (null vs null)")

        end_targets, end_meta = [], []
        for persona in SERIES_ORDER:
            if persona not in tr["personas"]:
                continue
            r = tr["personas"][persona]
            y = np.array(r["point"])
            color = SERIES_COLORS[persona]

            if not args.no_ci:
                ax.fill_between(x, r["lo"], r["hi"], color=color, alpha=0.13,
                                lw=0, zorder=2)
            ax.plot(x, y, color=color, lw=2.0, zorder=4, label=LABELS[persona],
                    solid_capstyle="round")
            end_targets.append(float(y[-1]))
            end_meta.append((persona, color))

        # Direct labels sit just outside the data area (clip_on=False) rather than
        # padding xlim, so the axis stops where the data stops -- no dead space.
        last = n_layers - 1
        placed = declutter(end_targets, min_gap=0.075)
        for (persona, color), y_lab in zip(end_meta, placed):
            ax.text(last + 0.7, y_lab, LABELS[persona], fontsize=9,
                    color=color, va="center", ha="left", fontweight="medium",
                    clip_on=False)

        ax.set_title(trait, fontsize=13, color=INK_PRIMARY, pad=10, loc="left",
                     fontweight="semibold")
        ax.set_xlabel("layer", fontsize=10, color=INK_SECONDARY)
        ax.set_xlim(0, last)
        ax.set_xticks(np.arange(0, n_layers, 5))
        ax.set_ylim(-0.35, 1.08)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    axes[0].set_ylabel(r"cos($v_{trait,persona}$, $v_{trait,null}$)",
                       fontsize=10, color=INK_SECONDARY)

    # legend present for >=2 series, in addition to the direct labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(
        f"Persona trait vectors rotate away from null in mid-stack  —  {short}",
        fontsize=14, color=INK_PRIMARY, x=0.007, ha="left", y=0.995,
        fontweight="semibold",
    )
    fig.text(0.007, 0.925,
             f"CAA vectors, {tr['n_questions']} questions/trait. Bands are {n_boot}-replicate "
             f"bootstrap 95% CI (unpaired resampling).",
             fontsize=9.5, color=INK_SECONDARY, ha="left")

    fig.tight_layout(rect=[0, 0.045, 1, 0.90])
    # the left panel's direct labels overhang into the gutter -- widen it so they
    # never land on the right panel's axes
    fig.subplots_adjust(wspace=0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
