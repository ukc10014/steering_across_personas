#!/usr/bin/env python3
"""Appendix A3 -- figure 3 replicated at layer 20.

Layer 15 is the pre-designated mid-stack headline layer for Llama-3.1-8B (32 layers) and
every main-text number is quoted there. This is the check that the picture is not a
property of one layer. Same four panels, same estimators, same correction rule -- both are
built by `dose_panels.build`, so they cannot drift apart.

What replicates: the trained-arm ordering, the shared `goodness`/`impulsiveness` RDM curve,
`misalignment` below it, `impulsiveness`'s flat dispersion response, and the untrained arms
landing inside the trained range on dispersion. On RDM they land at or slightly above its
top, not inside it -- `random_iid_s16` reads 0.849 against `mathematical`'s 0.832, on
heavily overlapping intervals -- and the same holds at L15 (0.886 against 0.878), so that
is a property of the measure rather than of the layer. Absolute values are lower at L20 --
everything moves more, deeper in the stack -- so the axes differ from figure 3's and the
two must not be read off a common scale.

ONE ARM IS MISSING FROM THE DATA, not dropped here. `random_spec_s19` has no layer-20
functional-dose measurement in `functional_dose_with_random.json` (17 arms at L20 against
18 at L15), so it cannot be placed on the L20 dose axis. The untrained family at L20 is
`random_iid` and `random_perm` only, and its span in panels C and D is a two-arm span,
which is not comparable to the three-arm span at L15.

Requires `geometry_L20_allarms.json`, which extends the committed `geometry_L20.json`
(4 trained arms only) to the ladder and the untrained arms. CPU only, from the cached
per-question activations:

    python scripts/geometry_analysis.py --layer 20 --bootstrap 200 \\
        --arms base goodness mathematical impulsiveness misalignment \\
               goodness_s0.25 goodness_s0.5 goodness_s0.75 \\
               impulsiveness_s0.25 impulsiveness_s0.5 impulsiveness_s0.75 \\
               misalignment_s0.25 misalignment_s0.5 misalignment_s0.75 \\
               random_perm_s8 random_perm_s12 random_perm_s16 random_iid_s16 \\
        --out outputs/llama-3.1-8b-goodness/analysis/geometry_L20_allarms.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dose_panels import build

ROWS = ["goodness", "mathematical", "impulsiveness", "misalignment",
        "random_iid_s16", "random_perm_s16"]      # no random_spec at L20; see docstring

if __name__ == "__main__":
    build("20", "geometry_L20_allarms.json", ROWS, "figA3_layer20")
