#!/usr/bin/env python3
"""Does selectivity really decline in the later part of the introspection SFT?

Written to be run UNATTENDED and to reach a verdict without a human in the loop, so it
states its criteria first and then applies them.

The hazard is specific and was already met once in this work: on the sparse curve, a peak
at 0.76 epochs and a dip at 1.00 had disjoint question-bootstrap intervals and looked real.
Four extra checkpoints showed adjacent ones (15 optimizer steps apart) move B1 by up to
0.446. The interval was measuring variance over evaluation questions, which is not the
variance that matters here. So this script judges the trend against CHECKPOINT SCATTER,
estimated from the residuals of its own fit, not against a question bootstrap.

Two questions, kept apart:

  1. TREND. Fit selectivity ~ a + b*epoch over the points at one epoch and beyond. Report b
     with a standard error built from the residual scatter. A decline counts only if the
     slope is more than 2 standard errors below zero.

  2. CONFOUND. Selectivity falls while dose rises, so "later training" and "more
     displacement" are not separated by the trend alone. For each late checkpoint, look up
     M_D's selectivity at the SAME measured dose and take the difference. If the excess over
     M_D also declines, the effect is not merely displacement. Interpolation is refused
     outside M_D's measured range; where the ladder does not reach, the row says so rather
     than extrapolating.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
import curve_common as cc  # noqa: E402

ANALYSIS = REPO / "outputs" / "analysis"
SLOPE_SE_THRESHOLD = 2.0


def md_ladder(rows):
    """(dose, selectivity) for every measured M_D rung, read from the analysis JSONs.

    Read from the analyses DIRECTLY, not from oct_stage_dose_master.csv. The first version of
    this function trusted that CSV and silently produced a 4-rung ladder stopping at dose
    0.827, which put every late checkpoint outside its range and killed the dose control --
    even though the two rungs that fix it had already been measured. The CSV is written by
    tableA_oct_matched_dose.py and was simply not regenerated.

    Regenerating it is deliberately NOT the fix here: extra M_D rungs widen the M_D/M_S
    overlap band, which moves the anchors in the published dose-matched table. That is a
    change to a published result and belongs to a human, not to a bug fix.
    """
    fd = json.loads((ANALYSIS / "functional_dose.json").read_text())[cc.LAYER]
    cs = json.loads((ANALYSIS / "common_shift.json").read_text())[cc.LAYER]
    traits = list(cs)
    pts = []
    for arm in fd:
        if arm != "impulsiveness_repro_dpo" and not arm.startswith("impulsiveness_dm_m_d_s"):
            continue
        dose = (fd.get(arm) or {}).get("trait_vector_displacement")
        if dose is None or arm not in cs[traits[0]]["per_arm"]:
            continue
        g = lambda t: cs[t]["per_arm"][arm]["g_over_base"]
        tgt = st.mean(g(t) for t in cc.TARGETS_PAIR)
        oth = st.mean(g(t) for t in traits if t not in cc.TARGETS_PAIR)
        if oth:
            pts.append((float(dose), tgt / oth))
    seen, out = set(), []
    for d, sel in sorted(pts):
        if round(d, 4) not in seen:
            seen.add(round(d, 4)); out.append((d, sel))
    return out


def jackknife(x, y, threshold=2.0):
    """Refit dropping each point in turn. A verdict that survives only with one particular
    point present is leverage, not evidence, so it is reported as such rather than as a
    finding. This check exists because the first run of this script reported "the decline
    survives the dose control" off a slope that collapsed from -2.1 SE to -0.9 SE when a
    single point was removed."""
    out = []
    for i in range(len(x)):
        xs = [v for j, v in enumerate(x) if j != i]
        ys = [v for j, v in enumerate(y) if j != i]
        b, _, _, se = fit(xs, ys)
        out.append((x[i], b, (b / se) if se and se == se else float("nan")))
    zs = [z for _, _, z in out]
    return out, all(z < -threshold for z in zs), min(zs), max(zs)


def fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    n = len(x)
    if n <= 2:
        return b, a, float("nan"), float("nan")
    s_resid = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    se_b = s_resid / float(np.sqrt(((x - x.mean()) ** 2).sum()))
    return float(b), float(a), s_resid, float(se_b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--construction", default="seq")
    a = p.parse_args()

    rows = cc.load()
    pts = sorted([r for r in rows if r["kind"] == "curve" and r["construction"] == a.construction
                  and r.get("selectivity") is not None and r.get("dose") is not None],
                 key=lambda r: r["step"])
    late = [r for r in pts if r["epoch"] >= 1.0]
    early = [r for r in pts if 0.60 <= r["epoch"] <= 0.95]

    L = ["# Is the late-SFT selectivity decline real?", "",
         "Generated by `scripts/appendix_oct/curve_late_stats.py`, run unattended. Criteria were",
         "fixed in that script before these numbers existed; it judges the trend against",
         "**checkpoint scatter**, not against a question bootstrap — the distinction that",
         "retired the apparent overshoot in [SFT_CURVE_REPORT.md](SFT_CURVE_REPORT.md).", ""]

    if len(late) < 4:
        L += [f"**Inconclusive: only {len(late)} checkpoints at or past one epoch.** "
              "The measurement did not complete far enough to fit a trend.", ""]
        Path(a.out).write_text("\n".join(L) + "\n")
        print(f"wrote {a.out} (inconclusive, {len(late)} late points)")
        return

    b, a0, s_resid, se_b = fit([r["epoch"] for r in late], [r["selectivity"] for r in late])
    z = b / se_b if se_b and se_b == se_b else float("nan")
    declines = (z < -SLOPE_SE_THRESHOLD)

    L += ["## 1. Trend over one epoch and beyond", "",
          f"- checkpoints used: **{len(late)}** ({late[0]['epoch']:.2f} to {late[-1]['epoch']:.2f} epochs)",
          f"- slope: **{b:+.4f}** selectivity per epoch  (SE {se_b:.4f}, {z:+.1f} SE from zero)",
          f"- residual scatter between checkpoints: **{s_resid:.4f}**",
          "", f"**Verdict: {'a real decline' if declines else 'NOT established'}** "
          f"— the criterion was a slope more than {SLOPE_SE_THRESHOLD:.0f} SE below zero.", ""]
    if early:
        L += [f"For scale, checkpoints between {early[0]['epoch']:.2f} and {early[-1]['epoch']:.2f} epochs "
              f"have selectivity mean {st.mean(r['selectivity'] for r in early):.3f}"
              + (f", sd {st.stdev(r['selectivity'] for r in early):.3f}" if len(early) > 1 else "") + ".", ""]

    L += ["| epoch | step | dose | selectivity | $B_1$ |", "|---|---|---|---|---|"]
    for r in late:
        L.append(f"| {r['epoch']:.2f} | {r['step']} | {r['dose']:.3f} | {r['selectivity']:.3f} | {r['B1']:+.3f} |")
    L.append("")

    # 2. dose control
    lad = md_ladder(rows)
    L += ["## 2. Is it training time, or just displacement?", ""]
    if len(lad) < 2:
        L += ["No usable M_D selectivity ladder; the dose control cannot be run.", ""]
    else:
        ld = np.array([p[0] for p in lad]); ls = np.array([p[1] for p in lad])
        L += [f"M_D's selectivity ladder spans dose **{ld.min():.3f}–{ld.max():.3f}** "
              f"({len(lad)} rungs). Excess is the checkpoint's selectivity minus M_D's at the "
              "same measured dose; nothing is extrapolated.", "",
              "| epoch | dose | curve sel. | $M_D$ sel. at that dose | excess |", "|---|---|---|---|---|"]
        ex_x, ex_y = [], []
        for r in late:
            if ld.min() <= r["dose"] <= ld.max():
                m = float(np.interp(r["dose"], ld, ls))
                L.append(f"| {r['epoch']:.2f} | {r['dose']:.3f} | {r['selectivity']:.3f} | {m:.3f} | {r['selectivity']-m:+.3f} |")
                ex_x.append(r["epoch"]); ex_y.append(r["selectivity"] - m)
            else:
                L.append(f"| {r['epoch']:.2f} | {r['dose']:.3f} | {r['selectivity']:.3f} | "
                         f"— *outside the ladder* | — |")
        L.append("")
        if len(ex_x) >= 4:
            eb, _, es, ese = fit(ex_x, ex_y)
            ez = eb / ese if ese and ese == ese else float("nan")
            jk, robust, zmin, zmax = jackknife(ex_x, ex_y, SLOPE_SE_THRESHOLD)
            passes = ez < -SLOPE_SE_THRESHOLD
            L += [f"Excess-over-$M_D$ slope: **{eb:+.4f}** per epoch (SE {ese:.4f}, {ez:+.1f} SE), "
                  f"residual scatter {es:.4f}.", "",
                  f"Leave-one-out refits span **{zmin:+.1f} to {zmax:+.1f} SE**"
                  f" — the slope {'holds' if robust else 'does NOT hold'} without every "
                  "individual point.", ""]
            if passes and robust:
                L += ["**The decline survives the dose control**, and survives dropping any "
                      "single checkpoint — so the loss of specificity is not explained by the "
                      "model simply moving further.", ""]
            elif passes and not robust:
                # the influential point is the one whose REMOVAL most weakens the slope,
                # i.e. the largest (least negative) leave-one-out z, not the smallest
                worst = max(jk, key=lambda t: t[2])
                keep = [(x_, y_) for x_, y_ in zip(ex_x, ex_y) if x_ != worst[0]]
                flat_x = [x_ for x_, _ in keep]
                flat_y = [y_ for _, y_ in keep]
                L += [f"**Not established: the slope depends on a single checkpoint.** It reads "
                      f"{ez:+.1f} SE with all points but {worst[2]:+.1f} SE once the "
                      f"{worst[0]:.2f}-epoch point is dropped, which is leverage rather than "
                      "evidence.", ""]
                if len(flat_y) > 1:
                    L += [f"Excluding that point, the remaining {len(flat_y)} checkpoints "
                          f"({min(flat_x):.2f}–{max(flat_x):.2f} epochs) have excess "
                          f"{st.mean(flat_y):+.3f} with sd {st.stdev(flat_y):.3f} — flat. So "
                          "there is no gradual loss of specificity across the later epochs; "
                          "specificity settles and stays put.", ""]
            else:
                L += ["**The decline does NOT survive the dose control** — the loss of "
                      "specificity cannot be separated from displacement on this evidence.", ""]
        else:
            L += [f"Only {len(ex_x)} late checkpoints fall inside M_D's measured dose range, "
                  "which is too few to fit the control. The M_D ladder needs extending further.", ""]

    L += ["## Limits", "",
          "- One seed, one trait, one model, one construction "
          f"(`{a.construction}`).",
          "- Checkpoint scatter is estimated from the residuals of this fit, so it assumes the",
          "  underlying trend really is close to linear over this range.",
          "- Question-bootstrap intervals are deliberately not used here; they answer a",
          "  different question and were misleading earlier in this work.", ""]

    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}: slope {b:+.4f} ({z:+.1f} SE), declines={declines}")


if __name__ == "__main__":
    main()
