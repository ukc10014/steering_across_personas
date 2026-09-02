#!/usr/bin/env python3
"""Shared style for the workshop figures: one palette, one arm order, one rcParams.

Every figure imports from here so that an arm is the same colour, the same marker and
the same name in all of them, and so the trained/untrained distinction is carried by
marker and dash as well as by hue (the paper is printed, photocopied and read by
colourblind reviewers; hue alone is not an encoding).

The categorical hues are slots 1, 2, 3 and 7 of the data-viz reference palette. That
choice is checked, not asserted -- `python workshop_iclr/scripts/validate_palette.py`
runs the Machado CVD transforms over all pairs and must print PASS.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "workshop_iclr" / "figures"
DATADIR = REPO / "workshop_iclr" / "data"
ANALYSIS = REPO / "outputs" / "analysis"
GEOM = REPO / "outputs" / "llama-3.1-8b-goodness" / "analysis"

# --- canonical arm order, used left-to-right and top-to-bottom everywhere ---------
TRAINED = ["goodness", "mathematical", "impulsiveness", "misalignment"]
UNTRAINED = ["random_iid_s16", "random_spec_s19", "random_perm_s16"]

TRAINED_COLOR = {
    "goodness":      "#2a78d6",   # blue
    "mathematical":  "#4a3aa7",   # violet
    "impulsiveness": "#eb6834",   # orange
    "misalignment":  "#1baf7a",   # aqua
}
# Deliberately recessive: the controls are a grey family separated by marker and dash,
# never by hue. They are direct-labelled in every figure they appear in.
UNTRAINED_COLOR = {
    "random_iid_s16":  "#adada7",
    "random_spec_s19": "#6f6f69",
    "random_perm_s16": "#3d3d3a",
}
COLOR = TRAINED_COLOR | UNTRAINED_COLOR
MARKER = {"goodness": "o", "mathematical": "D", "impulsiveness": "s",
          "misalignment": "^", "random_iid_s16": "v", "random_spec_s19": "P",
          "random_perm_s16": "X"}
# trained solid, untrained dashed -- the trained/untrained contrast is the paper's spine
DASH = {a: "-" for a in TRAINED} | {a: (0, (3, 1.6)) for a in UNTRAINED}

LABEL = {
    "goodness": "goodness", "mathematical": "mathematical",
    "impulsiveness": "impulsiveness", "misalignment": "misalignment",
    "random_iid_s16": "random-iid", "random_spec_s19": "random-spec",
    "random_perm_s16": "random-perm", "random_perm_s8": "random-perm",
    "random_perm_s12": "random-perm",
}

TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
          "confidence", "deference", "warmth", "impulsivity"]
TRAIT_LABEL = {t: t.replace("_", "-") for t in TRAITS}
# the two traits the `impulsiveness` constitution is actually about; marked, never reordered
IMPULSIVE_TRAITS = ("risk_taking", "impulsivity")

PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]

# ICLR body text is 5.5in wide. A full-width figure is 5.5in; a half-width one 2.7in.
FULL, HALF = 5.5, 2.7

INK = "#111111"
MUTED = "#5c5c58"
GRID = "#d9d9d5"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 -- ICLR requires it
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def despine(ax, grid_axis: str | None = "y") -> None:
    """Recessive axes: no top/right spine, a faint grid behind the marks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    """PDF for the paper, PNG for previewing. Same name, both formats."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"  wrote {name}.pdf / .png")


def write_source_data(name: str, rows: list[dict], header: list[str]) -> None:
    """Every figure ships the numbers it plots, so a reader can audit them."""
    import csv
    DATADIR.mkdir(parents=True, exist_ok=True)
    with open(DATADIR / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote data/{name}.csv  ({len(rows)} rows)")
