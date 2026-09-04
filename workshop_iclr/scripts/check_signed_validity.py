#!/usr/bin/env python3
"""Is the signed trait projection a VALID signed metric, or just a number with a sign?

Having a sign is not the same as that sign meaning "more of the trait". Before the signed
projection can be plotted as an Open-Character-Training-style gain/loss chart, it has to
behave the way a real displacement along a fixed axis behaves. Three tests, all CPU-only
from the cached dose ladder, all pre-specified here before the numbers are read:

  1. SIGN STABILITY. Scaling one adapter up should not flip which way it pushes a trait.
     For each constitution x trait cell, is sign(proj) the same at all four dose rungs?
     A metric whose sign flips as the same adapter is scaled is not measuring a direction.

  2. MONOTONE GROWTH. A displacement along a fixed axis should grow with dose. Is
     proj_{c,t}(dose) monotone across the four rungs, and does a line through it pass near
     the origin? Rank correlation with dose, and the intercept of the fit as a fraction of
     the value at s=1.

  3. THE CONTROL. `random_perm` has three rungs and no learned content. If it passes 1 and
     2 as convincingly as the constitutions do, then passing them is a property of scaling
     any perturbation, and the tests are necessary but not sufficient -- exactly the
     conclusion the rotation control (appendix A4) reached for a different statistic.

  4. LAYER AGREEMENT. The same cell measured at layer 20 is a near-independent read of the
     same intervention. If sign(proj) at L15 and L20 disagree often, the sign is not a
     property of the intervention. Run from the main bootstrapped file when present, and
     reported with the fraction of cells whose L15 interval excludes zero, since a cell
     whose sign is not resolved cannot be expected to agree with anything.

VERDICT RULE, fixed before running: the signed metric is reported as a main-paper result
only if the trained arms pass 1 and 2 for a clear majority of cells AND the trained
families separate from `random_perm` on test 3 -- either by passing more often, or by the
constitutions disagreeing with each other in sign on traits where the random arm is flat.
Otherwise it goes to the appendix labelled as a descriptive decomposition of figure 1B.

Input: outputs/analysis/signed_trait_shift_ladder.json
    python scripts/signed_trait_shift.py --layers 15 --bootstrap 0 --half-splits 40 \\
        --arms <the 12 constitution rungs> random_perm_s8 random_perm_s12 random_perm_s16 \\
        --out outputs/analysis/signed_trait_shift_ladder.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import ANALYSIS, TRAITS

LAYER = "15"
LADDERS = {
    "goodness": ["goodness_s0.25", "goodness_s0.5", "goodness_s0.75", "goodness"],
    "impulsiveness": ["impulsiveness_s0.25", "impulsiveness_s0.5",
                      "impulsiveness_s0.75", "impulsiveness"],
    "misalignment": ["misalignment_s0.25", "misalignment_s0.5",
                     "misalignment_s0.75", "misalignment"],
    "random_perm": ["random_perm_s8", "random_perm_s12", "random_perm_s16"],
}
KEY = "proj_common"


def main() -> None:
    d = json.load(open(ANALYSIS / "signed_trait_shift_ladder.json"))[LAYER]
    fd = json.load(open(ANALYSIS / "functional_dose_with_random.json"))[LAYER]
    ref = fd["goodness"]["answer_token_displacement"]
    dose = {a: fd[a]["answer_token_displacement"] / ref for a in fd}

    print(f"{'arm':14s}{'trait':14s}{'proj at s=1':>12s}{'sign':>7s}"
          f"{'rho(dose)':>11s}{'|b0|/|v1|':>11s}{'verdict':>9s}")
    summary = {}
    for arm, rungs in LADDERS.items():
        xs = np.array([dose[a] for a in rungs])
        n_ok = n_sign = 0
        for t in TRAITS:
            v = np.array([d[t]["per_arm"][a][KEY] for a in rungs])
            same_sign = bool(np.all(np.sign(v) == np.sign(v[-1])) and abs(v[-1]) > 1e-9)
            rho = float(spearmanr(xs, np.abs(v)).statistic)
            b1, b0 = np.polyfit(xs, v, 1)
            rel_intercept = abs(b0) / max(abs(v[-1]), 1e-9)
            ok = same_sign and rho >= 0.8 and rel_intercept <= 0.5
            n_ok += ok
            n_sign += same_sign
            print(f"{arm:14s}{t:14s}{v[-1]:+12.3f}{'same' if same_sign else 'FLIP':>7s}"
                  f"{rho:>11.2f}{rel_intercept:>11.2f}{'pass' if ok else 'fail':>9s}")
        summary[arm] = (n_sign, n_ok, len(TRAITS))
        print()

    print(f"{'arm':14s}{'stable sign':>13s}{'pass all 3':>12s}")
    for arm, (ns, no, n) in summary.items():
        print(f"{arm:14s}{ns:>9d}/{n:<3d}{no:>8d}/{n:<3d}")

    # ---- test 4: does the sign survive the move to layer 20? --------------------
    main_f = ANALYSIS / "signed_trait_shift.json"
    if main_f.exists():
        m = json.load(open(main_f))
        if "15" in m and "20" in m:
            arms = [a for a in m.get("config", {}).get("arms", [])]
            agree = tot = resolved = 0
            for a in arms:
                for t in TRAITS:
                    v15 = m["15"][t]["per_arm"][a][KEY]
                    v20 = m["20"][t]["per_arm"][a][KEY]
                    ci = m["15"][t].get(f"{KEY}_ci", {}).get(a)
                    tot += 1
                    agree += int(np.sign(v15) == np.sign(v20))
                    if ci and (ci[0] > 0 or ci[1] < 0):
                        resolved += 1
            print(f"\nlayer agreement: sign(L15) == sign(L20) in {agree}/{tot} cells "
                  f"({agree / tot:.0%});  L15 interval excludes zero in "
                  f"{resolved}/{tot} ({resolved / tot:.0%})")

    trained = [summary[a] for a in ("goodness", "impulsiveness", "misalignment")]
    tr_pass = sum(s[1] for s in trained) / sum(s[2] for s in trained)
    rp_pass = summary["random_perm"][1] / summary["random_perm"][2]
    print(f"\ntrained cells passing: {tr_pass:.0%}   random_perm: {rp_pass:.0%}")
    print("\nTest 3 is the one that decides placement: a metric the untrained arm passes "
          "just as\nwell is necessary-but-not-sufficient, and the figure belongs in the "
          "appendix as a\ndescriptive decomposition rather than as evidence of learned "
          "semantic direction.")


if __name__ == "__main__":
    main()
