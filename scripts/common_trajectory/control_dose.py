#!/usr/bin/env python3
"""Dose control: the SFT trajectory vs the constitution's DPO adapter scaled to the same
measured displacement (no SFT). COMMON_TRAJECTORY_PRELIM_impulsiveness.md; brief §2.

Along a trajectory functional dose changes, so a moving common shift (or trait profile) can
simply be the model being displaced further. The DPO-only ladder base + s*dW_D says what a
state of that dose looks like when it is just "more (or less) DPO". At each trajectory state
we interpolate the ladder at the state's measured dose (linear, NO extrapolation) and report

    excess = trajectory value - ladder value at matched dose

for: the 8-trait-mean and target-pair common magnitude g and squared share S (cross-fitted,
common_shift.py), and every forced-choice trait offset plus the constitution's contrast
(caa_logits_analysis estimator, point estimates).

Uncertainty: the ladder and the trajectory come from separate bootstrap runs, so there is no
paired interval on the excess. It is descriptive. This compares construction routes at
matched dose; it does NOT explain any discrepancy with a released adapter.

    python3 scripts/common_trajectory/control_dose.py --constitution loving
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from caa_logits_analysis import fit_offset_slope, stack_trait   # noqa: E402
from behaviour import load as load_logits                        # noqa: E402
from states import CONTRASTS, TARGETS, trajectory                # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
LADDER = {"impulsiveness": ["impulsiveness_repro_dpo"] + [f"impulsiveness_dm_m_d_s{s}" for s in ("1.364", "1.653", "1.979", "2.25", "2.5")],
          "loving": ["loving_traj_dpo"] + [f"loving_traj_ladder_d_s{s}" for s in ("0.8", "0.9", "1.1", "1.25", "1.4")]}


def offsets(arms):
    cells = load_logits(["base"] + arms)
    traits = sorted(cells["base"]); personas = sorted(cells["base"][traits[0]])
    out = {a: {} for a in arms}
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]; apos = cells["base"][t][personas[0]]["a_is_positive"]
        base = stack_trait(cells, "base", t, personas, qid); allq = np.arange(len(qid))
        for a in arms:
            out[a][t] = fit_offset_slope(base, stack_trait(cells, a, t, personas, qid), apos, allq)[0]
    return out, traits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--layers", nargs="+", default=["15", "20"])
    a = ap.parse_args()
    c = a.constitution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carms = [s["arm"] for s in trajectory(c)]; ep = [s["epoch"] for s in trajectory(c)]
    larms = LADDER[c]; tg = TARGETS[c]; cname = list(CONTRASTS[c])[-1]; ctg = CONTRASTS[c][cname]
    off, traits = offsets(carms + larms[1:])

    def contrast(o):
        rest = [t for t in traits if t not in ctg]
        return np.mean([o[t] for t in ctg]) - np.mean([o[t] for t in rest])

    res = {"constitution": c, "ladder_arms": larms, "trajectory_arms": carms, "epoch": ep, "layers": {}}
    for L in a.layers:
        g = json.loads((A / f"geometry_{c}.json").read_text())["layers"][L]
        lad = json.loads((A / f"common_shift_{c}_dpo_ladder.json").read_text())[L]
        fdc = json.loads((A / f"functional_dose_{c}.json").read_text())[L]
        fdl = json.loads((A / f"functional_dose_{c}_dpo_ladder.json").read_text())[L]
        xl = np.array([fdl[x]["trait_vector_displacement"] for x in larms]); o = np.argsort(xl)
        xc = [fdc[x]["trait_vector_displacement"] for x in carms]

        def excess(yc, yl):
            yl = np.asarray(yl)[o]; xs = xl[o]
            return [float(y - np.interp(x, xs, yl)) if xs[0] - 1e-9 <= x <= xs[-1] + 1e-9 else None for x, y in zip(xc, yc)]

        S = g["summary"]; ser = {}
        for key, lk in (("g", "g_over_base"), ("S", "share_squared")):
            for grp, ts in (("MEAN8", g["traits"]), ("TARGETS", tg)):
                yc = [S[grp][x][key]["point"] for x in carms]
                yl = [float(np.mean([lad[t]["per_arm"][x][lk] for t in ts])) for x in larms]
                ser[f"{key}_{grp}"] = {"trajectory": yc, "ladder": yl, "excess": excess(yc, yl)}
        if L == "15":
            for t in traits + [cname]:
                f = (lambda x: contrast(off[x])) if t == cname else (lambda x, t=t: off[x][t])
                yc = [f(x) for x in carms]; yl = [f(x) for x in larms]
                ser[f"offset_{t}"] = {"trajectory": yc, "ladder": yl, "excess": excess(yc, yl)}
        res["layers"][L] = {"trajectory_dose": xc, "ladder_dose": xl.tolist(), "series": ser}
    (A / f"control_dose_{c}.json").write_text(json.dumps(res, indent=1))

    # figure: L15, contrast + 4 geometry panels + per-trait excess bars at the endpoint and 1 ep
    L = res["layers"]["15"]; xc, xl = L["trajectory_dose"], L["ladder_dose"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.5), constrained_layout=True)
    for ax, (k, title) in zip(axes[:4], [(f"offset_{cname}", f"behaviour: {cname} contrast"),
                                         ("g_MEAN8", "common magnitude g, 8-trait mean"),
                                         ("S_MEAN8", "common squared share S, 8-trait mean"),
                                         ("S_TARGETS", f"common share S, {'+'.join(tg)}")]):
        sr = L["series"][k]; o = np.argsort(xl)
        ax.plot(np.array(xl)[o], np.array(sr["ladder"])[o], "-s", color="#6b6b6b", ms=4, lw=1.5, label="DPO only, scaled (no SFT)")
        ax.plot(xc, sr["trajectory"], "-o", color="#eb6834", ms=4, lw=0.8, label="DPO + SFT checkpoint")
        for x, y, e in zip(xc, sr["trajectory"], ep):
            if c == "loving" or e in (0.0,) or abs(e - 0.481) < 0.01 or abs(e - 1.002) < 0.01 or e > 2.99:
                ax.annotate(f"{e:.2f}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#6b6b6b")
        ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("measured functional dose (L15)", fontsize=8.5)
    ax = axes[4]
    idx = {"1 ep": min(range(len(ep)), key=lambda i: abs(ep[i] - 1.0)), "3 ep": len(ep) - 1}
    w = 0.38
    for j, (lab, i) in enumerate(idx.items()):
        v = [L["series"][f"offset_{t}"]["excess"][i] for t in traits]
        ax.bar(np.arange(len(traits)) + (j - 0.5) * w, [np.nan if x is None else x for x in v], w,
               color=["#2a78d6", "#eb6834"][j], label=lab)
    ax.axhline(0, color="#6b6b6b", lw=0.7)
    ax.set_xticks(range(len(traits))); ax.set_xticklabels([t.replace("_", "-")[:9] for t in traits], rotation=45, fontsize=7.5, ha="right")
    ax.set_title("trait offset: trajectory − scaled DPO at matched dose", loc="left", fontsize=9.5)
    for ax in axes:
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Dose control ({c}, L15): SFT trajectory vs DPO-only adapter scaled to the same measured displacement "
                 f"(points labelled by SFT epoch; no paired CI on the gap)", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(A / f"fig_control_dose_{c}.png", dpi=150)

    for Lk, Lv in res["layers"].items():
        for k, v in Lv["series"].items():
            ex = [e for e in v["excess"] if e is not None]
            if ex:
                print(f"L{Lk} {k:24s} excess mean {np.mean(ex):+.3f} range [{min(ex):+.3f}, {max(ex):+.3f}] "
                      f"n={len(ex)}/{len(v['excess'])}  endpoint {v['excess'][-1] if v['excess'][-1] is None else round(v['excess'][-1], 3)}")


if __name__ == "__main__":
    main()
