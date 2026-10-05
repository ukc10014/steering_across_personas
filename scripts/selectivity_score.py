#!/usr/bin/env python3
"""Preregistered target-vs-other selectivity, activation and behavioural, for any arm.

General by construction: the target trait set is an argument, so one code path scores every
constitution on its own named traits. Nothing here is specific to a constitution.

ACTIVATION (unsigned). Target/other ratio of the persona-common shift magnitude relative to
base, `g_over_base` from common_shift.py:

    ratio = mean_{t in targets} g_over_base[t] / mean_{t in others} g_over_base[t]

This is the quantity that reads 1.722 at L15 for `impulsiveness` (the published 1.72x).
Uncertainty is propagated from the EXISTING per-trait question bootstrap, not re-derived:
each trait is bootstrapped over its own question sample, so the per-trait estimates are
independent and their SEs add in quadrature within each set (the rule fig4_shared_direction
uses for its 8-trait means). The ratio's SE then follows by the delta method. Percentile
intervals are read back to an SE as (hi - lo) / 3.92.

**No signed meaning is read from this number.** It is a ratio of norms: large means the trait
direction moved, never which way. Signed claims come only from the behavioural block.

BEHAVIOURAL (signed). The compression-corrected offset contrast from caa_logits_analysis.py,
whose --targets flag takes the same trait set:

    contrast = mean_{t in targets} offset[t] - mean_{t in others} offset[t]

reported for both prompt forms, with that script's own bootstrap CI on the contrast itself.

Usage:
    python scripts/selectivity_score.py --arm loving --targets empathy warmth
    python scripts/selectivity_score.py --arm sycophancy --targets deference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "outputs" / "analysis"
Z = 1.959963985


def se_from_ci(lo: float, hi: float) -> float:
    return (hi - lo) / 3.92


def activation_ratio(cs: dict, arm: str, targets: list[str]) -> dict:
    """cs is one layer's block of a common_shift.py output."""
    traits = list(cs.keys())
    tgt = [t for t in targets if t in traits]
    oth = [t for t in traits if t not in tgt]
    if not tgt or not oth:
        raise SystemExit(f"bad split: targets={tgt} others={oth}")

    def vals(ts):
        v, s = [], []
        for t in ts:
            pa = cs[t]["per_arm"]
            if arm not in pa:
                raise SystemExit(f"arm {arm} absent from common_shift trait {t}")
            v.append(pa[arm]["g_over_base"])
            ci = cs[t].get("g_over_base_ci", {}).get(arm)
            s.append(se_from_ci(*ci) if ci else np.nan)
        return np.array(v), np.array(s)

    vt, st = vals(tgt)
    vo, so = vals(oth)
    mt, mo = float(vt.mean()), float(vo.mean())
    # independent per-trait draws -> quadrature within each set, then delta method
    set_t = float(np.sqrt(np.nansum(st ** 2)) / len(st))
    set_o = float(np.sqrt(np.nansum(so ** 2)) / len(so))
    ratio = mt / mo
    rel = np.sqrt((set_t / mt) ** 2 + (set_o / mo) ** 2) if mt and mo else np.nan
    se_r = ratio * rel
    return {
        "targets": tgt, "others": oth,
        "mean_target_g_over_base": mt, "mean_other_g_over_base": mo,
        "ratio": ratio, "ratio_se": float(se_r),
        "ratio_ci_lo": float(ratio - Z * se_r), "ratio_ci_hi": float(ratio + Z * se_r),
        "per_trait_g_over_base": {t: cs[t]["per_arm"][arm]["g_over_base"] for t in traits},
        "per_trait_share_squared": {t: cs[t]["per_arm"][arm]["share_squared"] for t in traits},
        "per_trait_g_over_base_ci": {
            t: cs[t].get("g_over_base_ci", {}).get(arm) for t in traits},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--common-shift", type=Path, default=ANALYSIS / "common_shift.json")
    ap.add_argument("--logits", type=Path, default=None,
                    help="caa_logits json scored with the SAME --targets")
    ap.add_argument("--dose", type=Path, default=ANALYSIS / "functional_dose.json")
    ap.add_argument("--layers", nargs="+", default=["15", "20"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    res: dict = {"arm": a.arm, "registered_targets": a.targets}

    # --- 1. functional dose ------------------------------------------------------------
    if a.dose.exists():
        fd = json.loads(a.dose.read_text())
        res["functional_dose"] = {L: fd.get(L, {}).get(a.arm) for L in a.layers
                                 if isinstance(fd.get(L), dict)}
        print(f"FUNCTIONAL DOSE  {res['functional_dose']}")

    # --- 2/3. activation ---------------------------------------------------------------
    cs_all = json.loads(a.common_shift.read_text())
    res["activation"] = {}
    for L in a.layers:
        if L not in cs_all:
            continue
        r = activation_ratio(cs_all[L], a.arm, a.targets)
        res["activation"][L] = r
        print(f"\nACTIVATION L{L}  (unsigned; a ratio of norms, no direction implied)")
        print(f"  targets {r['targets']}  mean g/base = {r['mean_target_g_over_base']:.4f}")
        print(f"  others  ({len(r['others'])})  mean g/base = {r['mean_other_g_over_base']:.4f}")
        print(f"  RATIO = {r['ratio']:.4f}  [{r['ratio_ci_lo']:.4f}, {r['ratio_ci_hi']:.4f}]"
              f"   prereg predicts > 1  ->  {'MET' if r['ratio_ci_lo'] > 1 else ('point>1 but CI covers 1' if r['ratio'] > 1 else 'NOT MET')}")
        print("  per-trait g_over_base:")
        for t, v in sorted(r["per_trait_g_over_base"].items(), key=lambda kv: -kv[1]):
            mark = "*" if t in r["targets"] else " "
            print(f"    {mark} {t:14s} {v:.4f}")

    # --- 4/5. behavioural --------------------------------------------------------------
    if a.logits and a.logits.exists():
        lg = json.loads(a.logits.read_text())
        res["behavioural"] = {}
        for form in ("forced", "default"):
            blk = lg.get(form)
            if not blk or a.arm not in blk.get("offset", {}):
                continue
            sel = blk.get("selectivity", {}).get("by_arm", {}).get(a.arm, {})
            res["behavioural"][form] = {
                "contrast": sel.get("contrast"),
                "contrast_ci_lo": sel.get("contrast_ci_lo"),
                "contrast_ci_hi": sel.get("contrast_ci_hi"),
                "mean_target": sel.get("mean_target"),
                "mean_other": sel.get("mean_other"),
                "targets_used": blk["selectivity"]["targets"],
                "others_used": blk["selectivity"]["others"],
                "retention_k": blk.get("retention", {}).get(a.arm),
                "per_trait_offset": blk["offset"][a.arm],
            }
            b = res["behavioural"][form]
            star = "PRIMARY" if form == "forced" else "secondary"
            print(f"\nBEHAVIOURAL ({form}, {star})  compression-corrected signed offsets")
            if b["contrast"] is not None:
                print(f"  CONTRAST = {b['contrast']:+.4f} "
                      f"[{b['contrast_ci_lo']:+.4f}, {b['contrast_ci_hi']:+.4f}]"
                      f"   prereg predicts > 0  ->  "
                      f"{'MET' if b['contrast_ci_lo'] > 0 else ('point>0 but CI covers 0' if b['contrast'] > 0 else 'NOT MET')}")
            print("  per-trait corrected offset:")
            for t, v in sorted(b["per_trait_offset"].items(), key=lambda kv: -kv[1]["point"]):
                mark = "*" if t in b["targets_used"] else " "
                print(f"    {mark} {t:14s} {v['point']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
