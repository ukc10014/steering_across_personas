#!/usr/bin/env python3
"""Rebuild the v2 geometry summaries from their saved raw accumulators. CPU only.

geometry.py keeps every replicate's raw cross-fitted energies (v2/geometry_raw_<c>_L<l><tag>.npz),
so a change to the estimability rule never needs the activation caches again. Rewrites only
the `summary` block of v2/geometry_<c><tag>.json.

    python3 scripts/audit/resummarise_geometry.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "common_trajectory"))
from estimability import summarise  # noqa: E402
from states import TARGETS  # noqa: E402

V2 = REPO / "outputs/analysis/common_trajectory/v2"


def main():
    for js in sorted(V2.glob("geometry_*.json")):
        name = js.stem[len("geometry_"):]
        c = next(k for k in ("impulsiveness", "loving", "goodness", "sycophancy") if name.startswith(k))
        tag = name[len(c):]
        R = json.loads(js.read_text())
        for L, lay in R["layers"].items():
            z = np.load(V2 / f"geometry_raw_{c}_L{L}{tag}.npz")
            arms, traits = [str(x) for x in z["arms"]], lay["traits"]
            point = {t: z[f"point_{t}"] for t in traits}
            boot = {t: z[f"boot_{t}"][np.isfinite(z[f"boot_{t}"][:, 0, 0])] for t in traits}
            groups = {t: [t] for t in traits}
            groups["MEAN8"] = traits
            groups["TARGETS"] = TARGETS.get(c) or traits
            lay["summary"] = summarise(point, boot, arms, groups)
        R["resummarised"] = "scripts/audit/resummarise_geometry.py"
        js.write_text(json.dumps(R, indent=1))
        print("resummarised", js.name)


if __name__ == "__main__":
    main()
