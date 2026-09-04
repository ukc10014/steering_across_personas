#!/usr/bin/env python3
"""Colourblind-safety check for the figure palette. Run it; do not eyeball it.

A Python port of the checks in the data-viz skill's validate_palette.js -- same
Machado (2009) CVD transforms at severity 1.0, same OKLab distance x100, same
thresholds -- because this pod has no node. Kept next to the figures so the palette
claim in the paper's figure notes is reproducible.

    python workshop_iclr/scripts/validate_palette.py

Thresholds, from the skill:
    CVD dE >= 8      target for every pair under protan/deutan/tritan
    normal dE >= 15  hard floor for full-colour readers
    contrast >= 3:1  against the chart surface, else the mark needs a visible label
"""
from __future__ import annotations

import itertools
import math

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}
SURFACE = "#ffffff"          # the paper is printed on white
CVD_FLOOR, NORMAL_FLOOR, CONTRAST_FLOOR = 8.0, 15.0, 3.0


def _lin(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                 for c in srgb)


def _oklab(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _simulate(rgb, kind):
    M = MACHADO[kind]
    return tuple(min(1.0, max(0.0, sum(M[i][j] * rgb[j] for j in range(3))))
                 for i in range(3))


def delta_e(c1: str, c2: str, kind: str | None = None) -> float:
    a = _oklab(_simulate(_lin(c1), kind) if kind else _lin(c1))
    b = _oklab(_simulate(_lin(c2), kind) if kind else _lin(c2))
    return 100 * math.dist(a, b)


def contrast(c1: str, c2: str) -> float:
    def lum(c):
        r, g, b = _lin(c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted((lum(c1), lum(c2)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def report(name: str, palette: dict[str, str], pairs: str = "all") -> bool:
    print(f"\n=== {name}  ({pairs} pairs, surface {SURFACE}) ===")
    names = list(palette)
    combos = (list(itertools.combinations(names, 2)) if pairs == "all"
              else list(zip(names, names[1:])))
    ok = True
    worst = {"normal": (1e9, None)} | {k: (1e9, None) for k in MACHADO}
    for a, b in combos:
        row = {"normal": delta_e(palette[a], palette[b])}
        row |= {k: delta_e(palette[a], palette[b], k) for k in MACHADO}
        for k, v in row.items():
            if v < worst[k][0]:
                worst[k] = (v, f"{a}/{b}")
        bad = row["normal"] < NORMAL_FLOOR or min(
            row[k] for k in MACHADO) < CVD_FLOOR
        ok &= not bad
        flag = "  <-- FAIL" if bad else ""
        print(f"  {a:16s} {b:16s} normal {row['normal']:5.1f}  "
              + "  ".join(f"{k} {row[k]:5.1f}" for k in MACHADO) + flag)
    print("  worst: " + "  ".join(f"{k} {v:.1f} ({w})" for k, (v, w) in worst.items()))
    for n, c in palette.items():
        cr = contrast(c, SURFACE)
        if cr < CONTRAST_FLOOR:
            print(f"  NOTE {n} contrast {cr:.2f}:1 vs surface -- needs a visible "
                  f"label or a heavier mark (relief rule)")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    from figstyle import TRAINED_COLOR, UNTRAINED_COLOR
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    a = report("trained constitutions", TRAINED_COLOR, "all")
    # The untrained arms are a deliberately recessive grey family: they are separated
    # by MARKER and DASH, not by hue, so the categorical floors do not apply to them
    # as a set. Reported for the record, and the reason each is direct-labelled.
    report("untrained controls (grey family, secondary-encoded)", UNTRAINED_COLOR, "all")
    b = report("trained vs untrained family heads",
               {"trained": TRAINED_COLOR["goodness"],
                "untrained": UNTRAINED_COLOR["random_perm_s16"]}, "all")
    raise SystemExit(0 if (a and b) else 1)
