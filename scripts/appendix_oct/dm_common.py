"""Shared loader for the dose-matched stage-localisation assets.

One place that knows how to turn the archived analysis JSON into rows of
(state, nominal scale, measured dose, endpoints), so the table and the figure cannot
disagree about what was measured.

Every state has FOUR points: the three dose-matched rungs plus its own trained s=1 state,
which is included as an anchor and marked so it can be drawn differently.

DOSE IS ALWAYS THE MEASURED trait-vector displacement from functional_dose.py -- never the
nominal scale, never a weight norm.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs" / "analysis"
LAYER = "15"
TARGETS_PAIR = ("impulsivity", "risk_taking")
TARGET_ONLY = "impulsivity"

# arm name -> (state, nominal scale, is the trained state?)
ANCHORS = {
    "impulsiveness_repro_dpo":        ("M_D", 1.0, True),
    "impulsiveness_repro_Dplus025S":  ("M_D+0.25S", 1.0, True),
    "impulsiveness_repro":            ("M_F", 1.0, True),
    "impulsiveness_sft_from_base":    ("M_S", 1.0, True),
}
SLUG = {"m_d": "M_D", "m_s": "M_S", "m_f": "M_F", "m_d025s": "M_D+0.25S"}
PAIRS = {"M_D vs M_S": ("M_D", "M_S"), "M_D+0.25S vs M_F": ("M_D+0.25S", "M_F")}
FINAL_ARM = "impulsiveness_repro"          # M_F at s=1: the direction everything is compared to


def _parse_dm(arm: str):
    """impulsiveness_dm_m_d025s_s1.026 -> ('M_D+0.25S', 1.026, False)"""
    body = arm[len("impulsiveness_dm_"):]
    slug, _, scale = body.rpartition("_s")
    if slug not in SLUG:
        return None
    return SLUG[slug], float(scale), False


def contrast(off_arm, targets):
    if not off_arm:
        return None
    tg = [off_arm[t]["point"] for t in targets if t in off_arm]
    ot = [v["point"] for t, v in off_arm.items() if t not in targets]
    return st.mean(tg) - st.mean(ot) if tg and ot else None


def load(variant: str = "forced") -> list[dict]:
    lg = json.loads((ANALYSIS / "caa_logits.json").read_text())[variant]
    cs = json.loads((ANALYSIS / "common_shift.json").read_text())[LAYER]
    fd = json.loads((ANALYSIS / "functional_dose.json").read_text())[LAYER]
    traits = list(cs)

    arms = {}
    for a in fd:
        if a in ANCHORS:
            arms[a] = ANCHORS[a]
        elif a.startswith("impulsiveness_dm_"):
            p = _parse_dm(a)
            if p:
                arms[a] = p

    rows = []
    for arm, (state, scale, trained) in sorted(arms.items()):
        off = (lg.get("offset") or {}).get(arm)
        ret = (lg.get("retention") or {}).get(arm)
        r = {"state": state, "arm": arm, "nominal_scale": scale, "is_trained_state": trained,
             "seed": 1}
        r["dose"] = (fd.get(arm) or {}).get("trait_vector_displacement")
        r["B1"] = contrast(off, (TARGET_ONLY,))
        r["B2"] = contrast(off, TARGETS_PAIR)
        r["k"] = st.mean(v["point"] for v in ret.values()) if ret else None
        # question-bootstrap CI on the per-trait offsets -> a CI on the contrast is not
        # available directly, so carry the impulsivity offset's interval as the honest one
        if off and TARGET_ONLY in off:
            o = off[TARGET_ONLY]
            r["impulsivity_offset"] = o["point"]
            r["impulsivity_ci_lo"], r["impulsivity_ci_hi"] = o.get("ci_lo"), o.get("ci_hi")
        if traits and arm in cs[traits[0]]["per_arm"]:
            g = lambda t: cs[t]["per_arm"][arm]["g_over_base"]
            tgt = st.mean(g(t) for t in TARGETS_PAIR)
            oth = st.mean(g(t) for t in traits if t not in TARGETS_PAIR)
            r["selectivity"] = tgt / oth
            cl = [c for t in traits
                  for c in [cs[t]["cos"].get(f"{arm}|{FINAL_ARM}")
                            or cs[t]["cos"].get(f"{FINAL_ARM}|{arm}")] if c is not None]
            r["cos_to_MF"] = st.mean(cl) if cl else (1.0 if arm == FINAL_ARM else None)
        rows.append(r)
    return rows


def curve(rows, state, key):
    """(dose, value) for one state, sorted by dose, dropping anything unmeasured."""
    pts = [(r["dose"], r[key]) for r in rows
           if r["state"] == state and r.get("dose") is not None and r.get(key) is not None]
    return sorted(pts)


def overlap(rows, sa, sb, key="B1"):
    ca, cb = curve(rows, sa, key), curve(rows, sb, key)
    if not ca or not cb:
        return None
    lo = max(ca[0][0], cb[0][0])
    hi = min(ca[-1][0], cb[-1][0])
    return (lo, hi) if hi > lo else None


def interp_at(rows, state, key, dose):
    """Value at a dose, plus whether it is a measured rung or an interpolation.

    Refuses to extrapolate: returns None outside the state's measured dose range. The dose
    axis is monotone by construction here (each state's rungs were chosen from its monotone
    prefix), but this asserts it rather than trusting it.
    """
    pts = curve(rows, state, key)
    if not pts:
        return None, None, None
    d = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
    if not np.all(np.diff(d) > 0):
        raise ValueError(f"{state}/{key}: dose axis is not strictly increasing: {d}")
    if not (d.min() <= dose <= d.max()):
        return None, None, None
    hit = np.isclose(d, dose, atol=1e-9)
    if hit.any():
        return float(v[hit][0]), "measured", None
    j = int(np.searchsorted(d, dose))
    return float(np.interp(dose, d, v)), "interpolated", (float(d[j - 1]), float(d[j]))
