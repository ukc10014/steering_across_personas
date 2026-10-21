"""Model states along each constitution's native sequential DPO+SFT trajectory.

    M_c(e) = M_0 + dW_D,c + dW_S,c(e)          (spec docs/spec_common_trajectory.md §4)

e = 0 is the DPO-only model. Every later state applies one checkpoint of ONE continuous SFT
run to the DPO-adapted model at scale 1 -- no PEFT weighted factor-space merge.

Arm names are the existing cache/logit names (outputs/_qcache/<arm>_L15_20.npz,
outputs/llama-3.1-8b-<arm>/caa_logits_forced/). Epoch = optimizer step / steps-per-epoch,
from the achieved step count, never the nominal fraction.
"""
from __future__ import annotations

# impulsiveness: the dense curve of docs/runs/oct/SFT_CURVE_REPORT.md (seed 123456 rerun).
# 5987 micro-batches/epoch at gradient accumulation 16 -> 374.1875 steps/epoch; three epochs
# is 1122 steps and `final` is the --save_path adapter written after the loop.
IMP_STEPS_PER_EPOCH = 5987 / 16
IMP_STEPS = [45, 90, 180, 255, 270, 285, 300, 330, 375, 450, 570, 690, 750, 810, 930, 1050, 1122]

# The spec's six table points, mapped to the nearest SAVED impulsiveness checkpoint
# (SFT_CURVE_PLAN.md table): achieved epochs 0, 0.240, 0.481, 0.762, 1.002, 2.998.
IMP_TABLE_STEPS = {0.0: 0, 0.25: 90, 0.5: 180, 0.75: 285, 1.0: 375, 3.0: 1122}


# loving: this experiment's reproduction (docs/runs/oct/GATE_REPORT_traj-loving.md -- scoped).
# 12,000 rows, all survive -> 6000 micro-batches/epoch / 16 = 375 steps/epoch exactly; 1125 at
# 3 epochs is both a periodic checkpoint and the --save_path adapter (the latter is used).
LOV_STEPS_PER_EPOCH = 375
LOV_STEPS = [90, 180, 285, 375, 1125]
LOV_TABLE_STEPS = {0.0: 0, 0.25: 90, 0.5: 180, 0.75: 285, 1.0: 375, 3.0: 1125}


# amendment A1 panel (goodness, sycophancy, and misalignment if its data arrive): the loving
# procedure and checkpoint grid. Steps/epoch from the trainer's micro-batch count at grad-accum
# 16 (rows surviving / 2 per micro-batch): sycophancy 12,000 rows -> 375; goodness 11,999 rows
# (1 dropped) -> 5999/16 = 374.94, endpoint int(3*5999/16) = 1124. VERIFY against the training
# log before analysis (the final adapter is the --save_path one either way).
PANEL_STEPS_PER_EPOCH = {"loving": 375, "sycophancy": 375, "goodness": 5999 / 16}
PANEL_FINAL_STEP = {"loving": 1125, "sycophancy": 1125, "goodness": 1124}
PANEL_STEPS = [90, 180, 285, 375]


def panel_arm(c: str, step: int) -> str:
    if step == 0:
        return f"{c}_traj_dpo"
    return f"{c}_traj_seq_step" + ("final" if step == PANEL_FINAL_STEP[c] else str(step))


def lov_arm(step: int) -> str:
    if step == 0:
        return "loving_traj_dpo"
    return "loving_traj_seq_step" + ("final" if step == 1125 else str(step))


def imp_arm(step: int) -> str:
    if step == 0:
        return "impulsiveness_repro_dpo"
    return "impulsiveness_curve_seq_step" + ("final" if step == 1122 else str(step))


def trajectory(constitution: str) -> list[dict]:
    """[{arm, step, epoch}] ascending, step 0 (DPO-only) first, endpoint last."""
    if constitution == "impulsiveness":
        steps = [0] + IMP_STEPS
        return [{"arm": imp_arm(s), "step": s, "epoch": s / IMP_STEPS_PER_EPOCH} for s in steps]
    if constitution == "loving":
        return [{"arm": lov_arm(s), "step": s, "epoch": s / LOV_STEPS_PER_EPOCH} for s in [0] + LOV_STEPS]
    if constitution in PANEL_STEPS_PER_EPOCH:
        c = constitution; steps = [0] + PANEL_STEPS + [PANEL_FINAL_STEP[c]]
        return [{"arm": panel_arm(c, s), "step": s, "epoch": s / PANEL_STEPS_PER_EPOCH[c]} for s in steps]
    raise KeyError(f"{constitution}: trajectory not trained yet")


def table_points(constitution: str) -> dict:
    if constitution == "impulsiveness":
        return {k: imp_arm(s) for k, s in IMP_TABLE_STEPS.items()}
    if constitution == "loving":
        return {k: lov_arm(s) for k, s in LOV_TABLE_STEPS.items()}
    if constitution in PANEL_STEPS_PER_EPOCH:
        c = constitution
        return {k: panel_arm(c, s) for k, s in zip((0.0, 0.25, 0.5, 0.75, 1.0, 3.0), [0] + PANEL_STEPS + [PANEL_FINAL_STEP[c]])}
    raise KeyError(constitution)


# Prespecified phenotype summaries (spec §5). Each is mean(targets) - mean(other traits), the
# B2 construction. Individual trait offsets are always reported alongside.
CONTRASTS = {
    "impulsiveness": {"B1": ["impulsivity"], "B2": ["impulsivity", "risk_taking"]},
    "loving": {"LOV": ["empathy", "warmth"]},
    # misalignment: the full eight-trait profile is primary; this is ONLY the prespecified
    # comparison with the impulsiveness results, not a definition of the phenotype.
    "misalignment": {"B2cmp": ["impulsivity", "risk_taking"]},
}
# goodness / sycophancy: NO contrast and no target pair (amendment A1) -- the eight-trait profile
# is the outcome. geometry.py's "TARGETS" summary for them is the 8-trait mean, so nothing is
# selected post hoc.
TARGETS = {"impulsiveness": ["impulsivity", "risk_taking"], "loving": ["empathy", "warmth"],
           "goodness": None, "sycophancy": None,
           "misalignment": ["impulsivity", "risk_taking"]}
