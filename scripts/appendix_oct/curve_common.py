"""Shared loader for the dense SFT checkpoint curve (spec_stage_localisation.md section 5).

Deliberately SEPARATE from dm_common rather than an extension of it. dm_common.load()
selects arms by the `impulsiveness_dm_` prefix plus a fixed ANCHORS dict; teaching it about
curve arms would make them appear in the already-published dose-matched figures. The two
loaders share `contrast()` and nothing else.

Arm naming: impulsiveness_curve_{seq,mrg}_step{N}, where N is an optimizer-step count or
the literal `final`.

  seq  base + dW_D + dW_S(step)            -- what SFT learns, applied additively
  mrg  PEFTMerge(A_D, A_S(step)) [1, 0.25] -- the released-style factor merge

STEP ARITHMETIC. An epoch is 5987 micro-batches at gradient accumulation 16, so 374.19
optimizer steps -- NOT the 375 the spec assumed (that figure assumes all 12,000 rows
survive; 11,974 do). Three epochs is 1122 steps. `final` is the --save_path adapter written
after the loop, i.e. step 1122; there is no step 1125, and the last periodic checkpoint is
1110. Epoch fractions are therefore computed with STEPS_PER_EPOCH below, not by dividing
by 375.

DOSE. Step and measured dose are correlated by construction -- later checkpoints move the
model further -- exactly as the raw stage ordering was. Every claim is made against
`dose`, with the step axis shown only for orientation.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs" / "analysis"
LAYER = "15"
TARGETS_PAIR = ("impulsivity", "risk_taking")
TARGET_ONLY = "impulsivity"
FINAL_ARM = "impulsiveness_repro"

MICRO_PER_EPOCH = 5987
ACCUM = 16
STEPS_PER_EPOCH = MICRO_PER_EPOCH / ACCUM      # 374.1875
FINAL_STEP = int(3 * MICRO_PER_EPOCH) // ACCUM  # 1122

# States the curve attaches to but does not re-measure, mapped to the step of THIS training
# run that they correspond to. M_S is None on purpose: it is SFT trained from the base model,
# not a checkpoint of this run, so it has no step here -- giving it FINAL_STEP would imply it
# is this curve's endpoint, which is exactly the confusion the SFT-from-base arm invites.
REFERENCE = {
    "impulsiveness_repro_dpo":     ("M_D",   0),           # both constructions at step 0
    "impulsiveness_repro_DplusS":  ("M_D+S", None),        # original run's seq endpoint
    "impulsiveness_repro":         ("M_F",   None),        # original run's mrg endpoint
    "impulsiveness_sft_from_base": ("M_S",   None),        # a different construction entirely
}
CONSTRUCTION = {"seq": "native sequential", "mrg": "released-style merge"}


def parse_curve(arm: str):
    """impulsiveness_curve_mrg_step90 -> ('mrg', 90) ; ..._stepfinal -> ('mrg', 1122)"""
    if not arm.startswith("impulsiveness_curve_"):
        return None
    body = arm[len("impulsiveness_curve_"):]
    con, _, step = body.partition("_step")
    if con not in CONSTRUCTION or not step:
        return None
    return con, (FINAL_STEP if step == "final" else int(step))


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

    rows = []
    for arm in sorted(fd):
        p = parse_curve(arm)
        if p:
            con, step = p
            r = {"kind": "curve", "construction": con, "step": step}
        elif arm in REFERENCE:
            r = {"kind": "reference", "construction": None, "step": REFERENCE[arm][1]}
        else:
            continue
        r["arm"] = arm
        r["state"] = REFERENCE[arm][0] if arm in REFERENCE else None
        r["epoch"] = None if r["step"] is None else round(r["step"] / STEPS_PER_EPOCH, 4)
        r["dose"] = (fd.get(arm) or {}).get("trait_vector_displacement")

        off = (lg.get("offset") or {}).get(arm)
        ret = (lg.get("retention") or {}).get(arm)
        r["B1"] = contrast(off, (TARGET_ONLY,))
        r["B2"] = contrast(off, TARGETS_PAIR)
        r["k"] = st.mean(v["point"] for v in ret.values()) if ret else None
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


def series(rows, construction: str, key: str = "B1"):
    """Curve points for one construction, ascending in step, step-0 anchor prepended."""
    zero = [r for r in rows if r["state"] == "M_D"]
    pts = [r for r in rows if r["kind"] == "curve" and r["construction"] == construction]
    out = sorted(zero + pts, key=lambda r: r["step"])
    return [(r["step"], r["epoch"], r["dose"], r.get(key)) for r in out
            if r.get(key) is not None and r["dose"] is not None]
