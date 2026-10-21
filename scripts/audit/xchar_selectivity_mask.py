#!/usr/bin/env python3
"""xchar activation selectivity (target/other g_over_base, L15/L20) against the archived
legacy-mask base (as reported) and against the synthetic corrected-mask base. Point estimates
only (cross-fitted, 40 half-splits, same splits for both bases): a sensitivity check on the
mixed-protocol comparison, not a replacement result. CPU only.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from common_shift import load_arm, cell, make_preps, stats_for_trait  # noqa: E402
from geometry_lib import half_splits  # noqa: E402

SYN = REPO / "outputs/analysis/audit/qcache_synth"
ARMS = ["loving", "xchar_lov_start", "xchar_lov_imp_s100", "xchar_lov_lov_s100",
        "xchar_base_imp_s100", "xchar_base_lov_s100"]
SETS = {"imp+risk": ["impulsivity", "risk_taking"], "emp+warmth": ["empathy", "warmth"]}


def main():
    out = {}
    for L in (15, 20):
        bases = {"legacy_base": load_arm(SYN, "base", [15, 20], L),
                 "synth_corrected_base": load_arm(SYN, "base_synthcorrected", [15, 20], L)}
        D = {a: load_arm(SYN, a, [15, 20], L) for a in ARMS}
        traits = bases["legacy_base"]["traits"]
        pidx = [bases["legacy_base"]["personas"].index(p) for p in bases["legacy_base"]["semantic"]]
        g = {b: {a: {} for a in ARMS} for b in bases}
        for ti, t in enumerate(traits):
            splits = half_splits(int(bases["legacy_base"]["nq"][ti]), 40, np.random.default_rng(ti))
            cells = {a: cell(D[a], ti, pidx) for a in ARMS}
            for b, bd in bases.items():
                r = stats_for_trait(make_preps(*cell(bd, ti, pidx), cells), splits, ARMS, naive=False)
                for a in ARMS:
                    g[b][a][t] = r["per_arm"][a]["g_over_base"]
        res = {}
        for b in bases:
            res[b] = {}
            for a in ARMS:
                res[b][a] = {}
                for name, tg in SETS.items():
                    oth = [t for t in traits if t not in tg]
                    res[b][a][name] = float(np.mean([g[b][a][t] for t in tg]) / np.mean([g[b][a][t] for t in oth]))
        out[str(L)] = res
        print(f"L{L}  {'arm':22s} " + "  ".join(f"{n}: legacy -> synth" for n in SETS))
        for a in ARMS:
            print(f"     {a:22s} " + "  ".join(
                f"{res['legacy_base'][a][n]:.2f} -> {res['synth_corrected_base'][a][n]:.2f}          " for n in SETS))
    p = REPO / "outputs/analysis/audit/xchar_selectivity_mask.json"
    p.write_text(json.dumps(out, indent=1)); print("wrote", p)


if __name__ == "__main__":
    main()
