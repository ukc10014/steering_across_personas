#!/usr/bin/env python3
"""Figures + table for the common-trajectory panel. spec §5 and amendment A1.

One row per constitution present (behaviour_<c>.json + geometry_<c>.json + functional_dose_<c>.json),
identical panels for every row (A1: the procedure and the readout are held fixed):

  A  eight-trait signed offset profile vs base (absolute levels; the constitution's secondary
     contrast, where one is prespecified, in black)
  B  TIMING, all on one 0-1 scale = fraction of the final SFT-induced change:
       behaviour profile q(e) = <D(e), D(final)>/||D(final)||^2 (8-trait offsets)
       geometry Q(e) = projection on the final incremental common shift (8-trait mean)
       common magnitude (g(e)-g(0))/(g(final)-g(0)) (8-trait mean)
  C  persona-common squared share: S total (vs base) and S^S incremental (vs DPO-only)
  D  common-shift magnitude g and cos(dG(e), dG(final))  [8-trait means]

fig_trajectory.png (x = SFT epoch, symlog: linear to 1, log beyond), fig_trajectory_dose.png
(x = measured functional dose, L15), table_trajectory.csv, trajectory_points.csv.
Geometry at L15; L20 is in the JSON. CIs are question-bootstrap, evaluation-only, one seed.

    python3 scripts/common_trajectory/figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from states import CONTRASTS, table_points   # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
ORDER = ["impulsiveness", "loving", "goodness", "sycophancy", "misalignment"]
LAYER = "15"
MUTED, GRID = "#6b6b6b", "#e6e6e6"
# Okabe-Ito (CVD-safe), yellow swapped for grey for contrast on white; fixed trait -> colour.
TRAIT_COL = {"assertiveness": "#E69F00", "confidence": "#56B4E9", "deference": "#009E73",
             "empathy": "#D55E00", "honesty": "#0072B2", "impulsivity": "#CC79A7",
             "risk_taking": "#7a4fd0", "warmth": "#8c8c8c"}
C_BEH, C_Q, C_G, C_S, C_SS, C_COS = "#1f1f1f", "#7a4fd0", "#eb6834", "#eb6834", "#7a4fd0", "#1f9e89"


def load(c):
    b = json.loads((A / f"behaviour_{c}.json").read_text())
    g = json.loads((A / f"geometry_{c}.json").read_text())
    fd = json.loads((A / f"functional_dose_{c}.json").read_text())
    nb = np.load(A / f"geometry_boot_{c}_L{LAYER}.npz")
    return b, g, fd, nb


def g_fraction(g, nb):
    arms = [str(x) for x in nb["arms"]]; q = list(map(str, nb["qty"])).index("g")
    traits = g["layers"][LAYER]["traits"]
    pt = np.array([g["layers"][LAYER]["summary"]["MEAN8"][x]["g"]["point"] for x in arms])
    bt = np.mean([nb[t][:, :, q] for t in traits], 0)
    fb = (bt - bt[:, :1]) / (bt[:, -1:] - bt[:, :1])
    lo, hi = np.nanpercentile(fb, [2.5, 97.5], axis=0)
    return (pt - pt[0]) / (pt[-1] - pt[0]), lo, hi


def band(ax, x, p, lo, hi, color, label, ls="-", marker="o", lw=1.6):
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0)
    ax.plot(x, p, ls=ls, marker=marker, ms=3, lw=lw, color=color, label=label)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="ABCD",
                    help="subset of panels, e.g. AB; a subset writes fig_trajectory_<panels>.png (epoch axis "
                         "only) and leaves the full figure and the CSVs untouched")
    a = ap.parse_args()
    panels = a.panels.upper()
    col = {p: i for i, p in enumerate(panels)}
    full = panels == "ABCD"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cons = [c for c in ORDER if all((A / f"{k}_{c}.json").exists() for k in ("behaviour", "geometry", "functional_dose"))]
    rows_csv, table = [], []
    for xmode in (("epoch", "dose") if full else ("epoch",)):
        fig, axes = plt.subplots(len(cons), len(panels), figsize=(4 * len(panels), 3.2 * len(cons)),
                                 squeeze=False, constrained_layout=True)
        for ri, c in enumerate(cons):
            b, g, fd, nb = load(c)
            arms = [s["arm"] for s in g["states"]]
            ep = np.array([s["epoch"] for s in g["states"]])
            dose = np.array([fd[LAYER][x]["trait_vector_displacement"] for x in arms])
            x = ep if xmode == "epoch" else dose
            S = g["layers"][LAYER]["summary"]["MEAN8"]
            ser, prof = b["series"], b["profile"]["rows"]

            if "A" in col:
              ax = axes[ri, col["A"]]                                # A profile
              for t in b["traits"]:
                  r = ser[t]["rows"]; y = [q["level"]["point"] for q in r]
                  ax.plot(x, y, "-o", ms=2.2, lw=1.2, color=TRAIT_COL[t])
                  ax.annotate(t.replace("_", "-")[:6], (x[-1], y[-1]), xytext=(3, 0), textcoords="offset points",
                              fontsize=6.3, color=TRAIT_COL[t], va="center")
              for n in CONTRASTS.get(c, {}):
                  if n in ("B1",):
                      continue
                  r = ser[n]["rows"]
                  band(ax, x, [q["level"]["point"] for q in r], [q["level"]["lo"] for q in r],
                       [q["level"]["hi"] for q in r], C_BEH, f"{n} contrast", lw=2.2)
              ax.axhline(0, color=MUTED, lw=0.6)
              ax.set_ylabel(f"{c}\noffset vs base (log-odds)", fontsize=9)
              ax.set_title("A  eight-trait profile", loc="left", fontsize=9.5)

            if "B" in col:
              ax = axes[ri, col["B"]]                                         # B timing
              pr = prof[1:]
              band(ax, x[1:], [r["q"]["point"] for r in pr], [r["q"]["lo"] for r in pr], [r["q"]["hi"] for r in pr],
                   C_BEH, "behaviour profile q", lw=2)
              band(ax, x[1:], [S[a_]["Q"]["point"] for a_ in arms[1:]], [S[a_]["Q"]["lo"] for a_ in arms[1:]],
                   [S[a_]["Q"]["hi"] for a_ in arms[1:]], C_Q, "geometry Q (common shift)", marker="s")
              fp, lo, hi = g_fraction(g, nb)
              band(ax, x, fp, lo, hi, C_G, "common magnitude g", ls="--")
              ax.axhline(1, color=MUTED, lw=0.6, ls=":"); ax.axhline(0, color=MUTED, lw=0.6, ls=":")
              ax.set_title("B  fraction of final SFT-induced change", loc="left", fontsize=9.5)

            if "C" in col:
              ax = axes[ri, col["C"]]                                         # C share
              band(ax, x, [S[a_]["S"]["point"] for a_ in arms], [S[a_]["S"]["lo"] for a_ in arms],
                   [S[a_]["S"]["hi"] for a_ in arms], C_S, "S total (vs base)")
              band(ax, x[1:], [S[a_]["SS"]["point"] for a_ in arms[1:]], [S[a_]["SS"]["lo"] for a_ in arms[1:]],
                   [S[a_]["SS"]["hi"] for a_ in arms[1:]], C_SS, "S$^S$ incremental (vs DPO-only)", marker="s")
              ax.set_ylim(0.3, 0.9)
              ax.set_title("C  persona-common squared share, 8-trait mean", loc="left", fontsize=9.5)

            if "D" in col:
              ax = axes[ri, col["D"]]                                         # D magnitude + direction
              band(ax, x, [S[a_]["g"]["point"] for a_ in arms], [S[a_]["g"]["lo"] for a_ in arms],
                   [S[a_]["g"]["hi"] for a_ in arms], C_G, "g = |dG| / |V_base|")
              band(ax, x, [S[a_]["cosT"]["point"] for a_ in arms], [S[a_]["cosT"]["lo"] for a_ in arms],
                   [S[a_]["cosT"]["hi"] for a_ in arms], C_COS, "cos(dG(e), dG(final))", marker="^")
              ax.set_title("D  common-shift magnitude and direction", loc="left", fontsize=9.5)

            for ax in axes[ri]:
                if xmode == "epoch":
                    ax.set_xscale("symlog", linthresh=1.0, linscale=2.0); ax.set_xlim(-0.03, 3.4)
                    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 2, 3]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1", "2", "3"])
                    ax.set_xlabel("SFT epoch", fontsize=8.5, color=MUTED)
                else:
                    ax.set_xlabel("measured functional dose (L15)", fontsize=8.5, color=MUTED)
                ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=8)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6.8, frameon=False, loc="best")

            if xmode == "epoch" and full:
                for a_, e_, d_ in zip(arms, ep, dose):
                    row = {"constitution": c, "arm": a_, "epoch": round(float(e_), 4), "dose_L15": round(float(d_), 4)}
                    for t in b["traits"]:
                        row[f"{t}_offset"] = round(next(q for q in ser[t]["rows"] if q["arm"] == a_)["level"]["point"], 4)
                    pr_ = next(r for r in prof if r["arm"] == a_)
                    row["profile_change_norm"] = round(pr_["change_norm"]["point"], 4)
                    row["profile_q"] = round(pr_["q"]["point"], 4) if "q" in pr_ else None
                    for k in ("g", "S", "SS", "cosT", "Q"):
                        v = S[a_][k]["point"]; row[f"MEAN8_{k}"] = None if v != v else round(v, 4)
                    rows_csv.append(row)
                tp = table_points(c)
                def pick(e, k):
                    v = S[tp[e]][k]
                    return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if v["lo"] is not None else f"{v['point']:.3f}"
                def qat(e):
                    r = next(r for r in prof if r["arm"] == tp[e])["q"]
                    return f"{r['point']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
                cn = [n for n in CONTRASTS.get(c, {}) if n != "B1"]
                ctr = cn[-1] if cn else None
                table.append({
                    "constitution": c,
                    "profile_change_norm_DPO_to_3ep": f"{b['profile']['endpoint_change_norm']['point']:.2f}",
                    "profile_q_0.5ep": qat(0.5), "profile_q_1ep": qat(1.0),
                    "contrast": ctr or "",
                    "contrast_DPO_to_3ep": "" if not ctr else
                        f"{ser[ctr]['rows'][0]['level']['point']:+.2f} -> {ser[ctr]['rows'][-1]['level']['point']:+.2f}",
                    **{f"S8_{k}": pick(e, "S") for k, e in (("DPO", 0.0), ("0.5", 0.5), ("1", 1.0), ("3", 3.0))},
                    **{f"SS8_{k}": pick(e, "SS") for k, e in (("0.5", 0.5), ("1", 1.0), ("3", 3.0))},
                    **{f"Q8_{k}": pick(e, "Q") for k, e in (("0.5", 0.5), ("1", 1.0))},
                    **{f"cos8_{k}": pick(e, "cosT") for k, e in (("0.5", 0.5), ("1", 1.0))},
                    **{f"dose_{k}": round(fd[LAYER][tp[e]]["trait_vector_displacement"], 3)
                       for k, e in (("DPO", 0.0), ("0.5", 0.5), ("1", 1.0), ("3", 3.0))}})
        fig.suptitle("Behavioural profile and persona-common geometry along the native sequential DPO+SFT trajectory"
                     + (" " if full else "\n")
                     + "(L15; 95% question-bootstrap CIs; one training seed per constitution)", fontsize=10.5, x=0.01, ha="left")
        name = ("fig_trajectory" if full else f"fig_trajectory_{panels}") + ("" if xmode == "epoch" else "_dose")
        fig.savefig(A / f"{name}.png", dpi=150)
        plt.close(fig)
    if not full:
        return

    with open(A / "trajectory_points.csv", "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows_csv for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows_csv)
    with open(A / "table_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r))); w.writeheader(); w.writerows(table)
    for t in table:
        print(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
