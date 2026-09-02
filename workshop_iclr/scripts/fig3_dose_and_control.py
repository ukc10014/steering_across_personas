#!/usr/bin/env python3
"""Figure 3 -- dose response and the matched untrained control, layer 15.

The paper's causal-control figure. Panels and all methodological caveats are documented
in `dose_panels.py`, which builds this and the layer-20 replication (appendix A3) from one
function so the two cannot drift apart.

What the four panels establish together:

  A  RDM preservation is close to a function of dose alone -- `goodness` and
     `impulsiveness` lie on one curve across a 2.2x dose range, with `misalignment`
     displaced below it at every dose.
  B  Dispersion is not. The arms share no curve; `impulsiveness` is about three times
     flatter and crosses both others, which no single scalar dose law can produce.
     A and B single out DIFFERENT anomalous arms, so neither is one idiosyncratic adapter.
  C  At matched dose the untrained arms land inside the trained range on RDM preservation
     and span MORE of it (0.125 against 0.110).
  D  On dispersion the trained family spans more (0.190 against 0.086), because both of
     its tails -- `misalignment` contracting hardest, `impulsiveness` least -- survive.

The honest summary is C and D together: broad geometric change is reproduced by
perturbations with no learned content at all, so it is not by itself a semantic signature
of constitutional training; what the trained family still owns is the SHAPE of its dose
response and the two tails, not the contraction.

The untrained arms reach matched functional dose at 16-19x the weight norm and therefore
sit within a factor of two of the measured coherence cliff (appendix A1). They are large
objects, not merely scaled-up small ones, and that caveat governs C and D.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dose_panels import build

ROWS = ["goodness", "mathematical", "impulsiveness", "misalignment",
        "random_iid_s16", "random_spec_s19", "random_perm_s16"]

if __name__ == "__main__":
    build("15", "geometry_L15.json", ROWS, "fig3_dose_and_control",
          xlims=((0.725, 0.935), (0.545, 0.865)))   # (C: RDM, D: dispersion)
