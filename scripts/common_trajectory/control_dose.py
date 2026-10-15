#!/usr/bin/env python3
"""Dose control for the impulsiveness trajectory (brief §2: "whether any additional
methodological controls are required").

Along the SFT trajectory, measured functional dose rises 0.53 -> 0.90, so a growing common
shift could simply be the model being displaced further. The existing DPO-only dose ladder
(impulsiveness_dm_m_d_s*, the DPO adapter scaled 1.36-2.5x; DOSE_MATCHED_STAGE_REPORT.md)
spans dose 0.53-0.97 with NO SFT at all, so it says what a common shift of that dose looks like
when it is just "more DPO". Comparing the two at matched dose separates "SFT makes the change
more / less persona-common" from "SFT moves the model further".

Same estimator for both (common_shift.py cross-fitted; geometry for the curve from
geometry_impulsiveness.json, for the ladder from common_shift_impulsiveness_dpo_ladder.json --
both cross-fitted on 40 half-splits, independent bootstrap runs, so the curve-vs-ladder gap's
uncertainty is approximated by the two separate CIs, not a paired interval). Behaviour for
the ladder is the established offset from caa_logits.json (point only).

Writes fig_control_dose.png and control_dose.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
A = REPO / "outputs/analysis/common_trajectory"
MAIN = Path("/workspace/repos/steering_across_personas/outputs/analysis")
LAYER = "15"
TG = ["impulsivity", "risk_taking"]


def interp_excess(xc, yc, xl, yl):
    """curve value minus the ladder's linear interpolation at the curve's dose (no extrapolation)."""
    o = np.argsort(xl); xl, yl = np.asarray(xl)[o], np.asarray(yl)[o]
    return [float(y - np.interp(x, xl, yl)) if xl[0] <= x <= xl[-1] else None for x, y in zip(xc, yc)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = json.loads((A / "geometry_impulsiveness.json").read_text())
    fdc = json.loads((A / "functional_dose_impulsiveness.json").read_text())[LAYER]
    lad = json.loads((A / "common_shift_impulsiveness_dpo_ladder.json").read_text())[LAYER]
    fdl = json.loads((A / "functional_dose_impulsiveness_dpo_ladder.json").read_text())[LAYER]
    beh = json.loads((A / "behaviour_impulsiveness.json").read_text())
    off = json.loads((MAIN / "caa_logits.json").read_text())["forced"]["offset"]

    S = g["layers"][LAYER]["summary"]; traits = g["layers"][LAYER]["traits"]
    carms = [s["arm"] for s in g["states"]]
    larms = sorted(fdl, key=lambda a: fdl[a]["trait_vector_displacement"])
    xc = [fdc[a]["trait_vector_displacement"] for a in carms]
    xl = [fdl[a]["trait_vector_displacement"] for a in larms]

    def lad_mean(key, ts):
        return [float(np.mean([lad[t]["per_arm"][a][key] for t in ts])) for a in larms]

    def b2(a):
        o = off[a]; rest = [t for t in o if t not in TG]
        return np.mean([o[t]["point"] for t in TG]) - np.mean([o[t]["point"] for t in rest])

    series = {
        "g_mean8": ([S["MEAN8"][a]["g"]["point"] for a in carms], lad_mean("g_over_base", traits)),
        "g_targets": ([S["TARGETS"][a]["g"]["point"] for a in carms], lad_mean("g_over_base", TG)),
        "S_mean8": ([S["MEAN8"][a]["S"]["point"] for a in carms], lad_mean("share_squared", traits)),
        "S_targets": ([S["TARGETS"][a]["S"]["point"] for a in carms], lad_mean("share_squared", TG)),
        "B2": ([r["level"]["point"] for r in beh["series"]["B2"]["rows"]], [b2(a) for a in larms]),
    }
    out = {"layer": LAYER, "curve_dose": xc, "ladder_dose": xl, "ladder_arms": larms, "curve_arms": carms,
           "series": {}, "note": "excess = curve minus ladder interpolated at the curve's dose; None outside the ladder's dose range"}
    for k, (yc, yl) in series.items():
        out["series"][k] = {"curve": yc, "ladder": yl, "excess_over_ladder": interp_excess(xc, yc, xl, yl)}
    (A / "control_dose.json").write_text(json.dumps(out, indent=1))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    panels = [("B2", "behaviour: B2 contrast"), ("g_mean8", "common magnitude g, 8-trait mean"),
              ("g_targets", "common magnitude g, impulsivity+risk"), ("S_mean8", "persona-common squared share S, 8-trait mean")]
    ep = [s["epoch"] for s in g["states"]]
    for ax, (k, title) in zip(axes, panels):
        yc, yl = series[k]
        ax.plot(xl, yl, "-s", color="#6b6b6b", ms=4, lw=1.5, label="DPO only, scaled (no SFT)")
        ax.plot(xc, yc, "o", color="#eb6834", ms=4, label="DPO + SFT checkpoint (trajectory)")
        for x, y, e in zip(xc, yc, ep):
            if e in (0.0,) or abs(e - 0.481) < 0.01 or abs(e - 1.002) < 0.01 or e > 2.99:
                ax.annotate(f"{e:.1f} ep", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7, color="#6b6b6b")
        ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("measured functional dose (L15)", fontsize=8.5)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Dose control (impulsiveness, L15): SFT trajectory vs DPO-only adapter scaled to the same "
                 "measured displacement", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(A / "fig_control_dose.png", dpi=150)
    for k, v in out["series"].items():
        ex = [e for e in v["excess_over_ladder"] if e is not None]
        print(f"{k:10s} curve-minus-ladder at matched dose: mean {np.mean(ex):+.3f}  range [{min(ex):+.3f}, {max(ex):+.3f}]  (n={len(ex)})")


if __name__ == "__main__":
    main()
