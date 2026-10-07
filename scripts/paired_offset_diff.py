#!/usr/bin/env python3
"""Paired question-bootstrap CIs for the per-trait offset DIFFERENCE between two arms.

The cached CIs in outputs/analysis/caa_logits.json are MARGINAL: each arm is bootstrapped and
summarised on its own, so subtracting two point estimates leaves a difference with no interval,
and assembling one from the two marginals would ignore that both arms are measured on the SAME
questions. Here questions are resampled ONCE per draw and both arms are evaluated on that draw,
so the question-level noise common to the two arms cancels.

No new model inference: this reads the per-question .npz cells run_caa_logits.sh already wrote,
and reuses caa_logits_analysis.py's own loader and estimator (`fit_offset_slope`) instead of
reimplementing them, so the point estimates must agree with the cached ones.

Reports, all as paired differences (arm_b - arm_a):
    per trait                a_t
    mean over target traits
    mean over control traits
    target minus control     == the change in B2, the preregistered selectivity contrast

    python scripts/paired_offset_diff.py --a impulsiveness_repro --b impulsiveness_regen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caa_logits_analysis as C  # noqa: E402  (loader + estimator, reused not reimplemented)

TARGETS = ("impulsivity", "risk_taking")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="impulsiveness_repro", help="baseline arm of the difference")
    ap.add_argument("--b", default="impulsiveness_regen", help="arm of interest")
    ap.add_argument("--variant", default="caa_logits_forced")
    ap.add_argument("--n-boot", type=int, default=4000,
                    help="paired draws; a difference is a narrower quantity than either "
                         "marginal, so extra draws are cheap and steady the tails")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/analysis/paired_offset_diff.json")
    a = ap.parse_args()

    cells = C.load_variant(a.variant)
    for arm in (C.BASE_ARM, a.a, a.b):
        if arm not in cells:
            raise SystemExit(f"FATAL: no cached cells for arm {arm!r}")
    traits = sorted(set(cells[C.BASE_ARM]) & set(cells[a.a]) & set(cells[a.b]))
    print(f"variant {a.variant} | traits {len(traits)} | n_boot {a.n_boot} | seed {a.seed}")
    print(f"difference = ({a.b}) - ({a.a}), paired on questions\n")

    rng = np.random.default_rng(a.seed)
    res: dict[str, dict] = {}
    boot_by_trait: dict[str, np.ndarray] = {}

    for trait in traits:
        # ONE persona set for all three arms: a paired difference needs the same cells on both
        # sides. The cached analysis intersects per arm, which is right for reading a resumable
        # run mid-flight but would break the pairing here.
        personas = sorted(set(cells[C.BASE_ARM][trait]) & set(cells[a.a][trait])
                          & set(cells[a.b][trait]))
        if len(personas) < 2:
            print(f"  {trait}: too few shared personas ({len(personas)}), skipped")
            continue
        qid_ref = cells[C.BASE_ARM][trait][personas[0]]["qid"]
        base = C.stack_trait(cells, C.BASE_ARM, trait, personas, qid_ref)
        A = C.stack_trait(cells, a.a, trait, personas, qid_ref)
        B = C.stack_trait(cells, a.b, trait, personas, qid_ref)
        a_pos = cells[C.BASE_ARM][trait][personas[0]]["a_is_positive"]
        nq = base.shape[1]
        allq = np.ones(nq, dtype=bool)

        off_a, _ = C.fit_offset_slope(base, A, a_pos, allq)
        off_b, _ = C.fit_offset_slope(base, B, a_pos, allq)

        draws = np.empty(a.n_boot)
        for i in range(a.n_boot):
            # fit_offset_slope takes a boolean question selector, so a draw WITH REPLACEMENT is
            # applied by indexing the arrays and passing an all-true mask.
            idx = rng.integers(0, nq, nq)
            bb, AA, BB, ap_ = base[:, idx], A[:, idx], B[:, idx], a_pos[idx]
            xa, _ = C.fit_offset_slope(bb, AA, ap_, allq)
            xb, _ = C.fit_offset_slope(bb, BB, ap_, allq)
            draws[i] = xb - xa
        boot_by_trait[trait] = draws
        lo, hi = np.percentile(draws, [2.5, 97.5])
        res[trait] = {"a": off_a, "b": off_b, "diff": off_b - off_a,
                      "ci_lo": float(lo), "ci_hi": float(hi),
                      "boot_sd": float(draws.std(ddof=1)),
                      "n_personas": len(personas), "n_questions": int(nq),
                      "excludes_zero": bool(lo > 0 or hi < 0)}

    tg = [t for t in TARGETS if t in res]
    ot = sorted(t for t in res if t not in TARGETS)
    print(f"{'trait':15s}{'arm_a':>10s}{'arm_b':>10s}{'diff':>9s}{'paired 95% CI':>22s}  excl.0")
    print("-" * 72)
    for t in tg + ot:
        r = res[t]
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"{t + (' *' if t in TARGETS else '  '):15s}{r['a']:+10.3f}{r['b']:+10.3f}"
              f"{r['diff']:+9.3f}{ci:>22s}  {'yes' if r['excludes_zero'] else 'NO'}")

    summary: dict[str, dict] = {}
    if tg and ot:
        # Aggregates from the SAME draws, so the contrast difference is paired too.
        d_t = np.vstack([boot_by_trait[t] for t in tg]).mean(axis=0)
        d_o = np.vstack([boot_by_trait[t] for t in ot]).mean(axis=0)
        pt_t = float(np.mean([res[t]["diff"] for t in tg]))
        pt_o = float(np.mean([res[t]["diff"] for t in ot]))
        for name, dist, point in (("mean target change", d_t, pt_t),
                                  ("mean control change", d_o, pt_o),
                                  ("B2 change (target - control)", d_t - d_o, pt_t - pt_o)):
            lo, hi = np.percentile(dist, [2.5, 97.5])
            summary[name] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
                             "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"\n{'quantity':32s}{'point':>9s}{'paired 95% CI':>22s}  excl.0")
        print("-" * 72)
        for k, v in summary.items():
            ci = f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
            print(f"{k:32s}{v['point']:+9.3f}{ci:>22s}  {'yes' if v['excludes_zero'] else 'NO'}")
        print(f"\n  targets  = {tg}")
        print(f"  controls = {ot}")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"arm_a": a.a, "arm_b": a.b, "variant": a.variant,
                             "n_boot": a.n_boot, "seed": a.seed, "paired": True,
                             "per_trait": res, "summary": summary}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
