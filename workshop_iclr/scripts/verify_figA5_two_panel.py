#!/usr/bin/env python3
"""Assert the two-panel A5 plots exactly the curves the four-panel A5 plots.

Both scripts read the same JSON, but the rung lists and the dose normalisation are copied
by hand, and a silent typo there would produce a plausible-looking figure with the wrong
numbers. This recomputes both independently -- the four-panel arithmetic is re-derived here
from figA5_signed_validity.PANELS, the two-panel numbers come from its own curves() -- and
compares element by element.

    python workshop_iclr/scripts/verify_figA5_two_panel.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figstyle import ANALYSIS, TRAITS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def four_panel_curves(panels, layer, key):
    d = json.load(open(ANALYSIS / "signed_trait_shift_ladder.json"))[layer]
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[layer]
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}
    out = {}
    for arm, rungs in panels.items():
        xs = np.array([dose[a] for a in rungs])
        vals = {t: np.array([d[t]["per_arm"][a][key] for a in rungs]) for t in TRAITS}
        n_pass = 0
        for t in TRAITS:
            v = vals[t]
            same = bool(np.all(np.sign(v) == np.sign(v[-1])) and abs(v[-1]) > 1e-9)
            rho = float(spearmanr(xs, np.abs(v)).statistic)
            _, b0 = np.polyfit(xs, v, 1)
            n_pass += same and rho >= 0.8 and abs(b0) / max(abs(v[-1]), 1e-9) <= 0.5
        out[arm] = (xs, vals, n_pass)
    return out


def main() -> None:
    four = load_module("figA5", HERE / "figA5_signed_validity.py")
    two = load_module("figA5b", HERE / "figA5b_signed_validity_2panel.py")

    assert two.LAYER == four.LAYER, f"LAYER differs: {two.LAYER} vs {four.LAYER}"
    assert two.KEY == four.KEY, f"KEY differs: {two.KEY} vs {four.KEY}"
    print(f"  LAYER {two.LAYER}, KEY {two.KEY}: match")

    for arm, rungs in two.PANELS.items():
        assert rungs == four.PANELS[arm], f"{arm}: rung list differs\n  {rungs}\n  {four.PANELS[arm]}"
    print(f"  rung lists match for: {', '.join(two.PANELS)}")

    ref = four_panel_curves(four.PANELS, four.LAYER, four.KEY)
    got = two.curves()

    worst = 0.0
    for arm in two.PANELS:
        xs_r, vals_r, np_r = ref[arm]
        xs_g, vals_g, np_g, _ = got[arm]
        assert np.array_equal(xs_r, xs_g), f"{arm}: dose axis differs"
        for t in TRAITS:
            a, b = vals_r[t], vals_g[t]
            assert a.shape == b.shape, f"{arm}/{t}: shape differs"
            worst = max(worst, float(np.max(np.abs(a - b))))
            assert np.array_equal(a, b), f"{arm}/{t}: curve differs"
        assert np_r == np_g, f"{arm}: validity count differs ({np_r} vs {np_g})"
        print(f"  {arm:18s} {len(xs_g)} rungs x {len(TRAITS)} traits identical, "
              f"{np_g}/8 valid")

    print(f"  max |difference| across every plotted point: {worst:.1e}")
    print("\nVERIFIED: the two retained panels plot exactly the four-panel curves.")


if __name__ == "__main__":
    main()
