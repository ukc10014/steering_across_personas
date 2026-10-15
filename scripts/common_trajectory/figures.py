#!/usr/bin/env python3
"""Figures + table for the common-trajectory experiment. docs/spec_common_trajectory.md §5, brief §9.

Reads behaviour_<c>.json, geometry_<c>.json (+ its bootstrap npz) and
functional_dose_<c>.json for every constitution present, and writes:

  fig_trajectory.png        one row per constitution, x = SFT epoch
      A  phenotype: the constitution's contrast(s) and target-trait offsets (absolute)
      B  persona-common squared share S (total, vs base) and S^S (incremental, vs DPO-only)
      C  common-shift magnitude g and cos(dG(e), dG(final))
      D  acquisition overlay on one 0-1 scale: behaviour F(e), magnitude
         (g(e)-g(0))/(g(final)-g(0)), and Q(e), the projection on the final SFT common shift
  fig_trajectory_dose.png   panels A-D against measured functional dose (L15)
  table_trajectory.csv      the brief's §9 table, per constitution
  trajectory_points.csv     every plotted number with its CI

x is symlog (linear to 1 epoch, log beyond) so the first epoch -- where the impulsiveness
phenotype arrives -- is not squeezed into a sliver. Geometry is L15 (primary); L20 is in the
JSON. CIs are question-bootstrap, evaluation-only.

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
from states import CONTRASTS, TARGETS, table_points   # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
ORDER = ["impulsiveness", "loving", "misalignment"]
LAYER = "15"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
C_BEH, C_TOT, C_INC, C_ALT = "#2a78d6", "#eb6834", "#7a4fd0", "#1f9e89"


def load(c):
    b = json.loads((A / f"behaviour_{c}.json").read_text())
    g = json.loads((A / f"geometry_{c}.json").read_text())
    fd = json.loads((A / f"functional_dose_{c}.json").read_text())
    nb = np.load(A / f"geometry_boot_{c}_L{LAYER}.npz")
    return b, g, fd, nb


def geo_fraction(g, nb, group_traits, key="g"):
    """(x(e)-x(0))/(x(final)-x(0)) for the group mean, point + CI from the shared bootstrap."""
    arms = [str(x) for x in nb["arms"]]; q = list(map(str, nb["qty"])).index(key)
    S = g["layers"][LAYER]["summary"]
    grp = "MEAN8" if len(group_traits) == 8 else ("TARGETS" if len(group_traits) == 2 else group_traits[0])
    pt = np.array([S[grp][x][key]["point"] for x in arms])
    bt = np.mean([nb[t][:, :, q] for t in group_traits], 0)          # [rep, arm]
    den_p, den_b = pt[-1] - pt[0], bt[:, -1] - bt[:, 0]
    fp = (pt - pt[0]) / den_p
    fb = (bt - bt[:, :1]) / den_b[:, None]
    lo, hi = np.nanpercentile(fb, [2.5, 97.5], axis=0)
    return fp, lo, hi, den_p


def style(ax, xlab):
    ax.set_xscale("symlog", linthresh=1.0, linscale=2.0)
    ax.set_xlim(-0.03, 3.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 2, 3]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1", "2", "3"])
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlabel(xlab, fontsize=8.5, color=MUTED)


def band(ax, x, p, lo, hi, color, label, ls="-", marker="o"):
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
    ax.plot(x, p, ls=ls, marker=marker, ms=3.2, lw=1.6, color=color, label=label)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cons = [c for c in ORDER if (A / f"behaviour_{c}.json").exists() and (A / f"geometry_{c}.json").exists()]
    rows_csv, table = [], []
    for xmode in ("epoch", "dose"):
        fig, axes = plt.subplots(len(cons), 4, figsize=(15, 3.3 * len(cons)), squeeze=False,
                                 constrained_layout=True)
        for ri, c in enumerate(cons):
            b, g, fd, nb = load(c)
            arms = [s["arm"] for s in g["states"]]
            ep = np.array([s["epoch"] for s in g["states"]])
            dose = np.array([fd[LAYER][x]["trait_vector_displacement"] for x in arms])
            x = ep if xmode == "epoch" else dose
            S = g["layers"][LAYER]["summary"]
            tg = TARGETS[c]
            cname = list(CONTRASTS[c])[-1]        # B2 / LOV / B2cmp: the pair contrast

            # A -- phenotype levels
            ax = axes[ri, 0]
            ser = b["series"]
            for name, col, ls in [(cname, C_BEH, "-")] + [(t, C_ALT if i == 0 else C_INC, "--") for i, t in enumerate(tg)]:
                r = ser[name]["rows"]
                band(ax, x, [q["level"]["point"] for q in r], [q["level"]["lo"] for q in r],
                     [q["level"]["hi"] for q in r], col, name.replace("_", "-"), ls=ls)
            ax.set_ylabel(f"{c}\nforced-choice offset (log-odds)", fontsize=9)
            ax.set_title("A  phenotype", loc="left", fontsize=9.5)

            # B -- squared share
            ax = axes[ri, 1]
            for grp, lab, ls in (("MEAN8", "8-trait mean", "-"), ("TARGETS", "targets", "--")):
                band(ax, x, [S[grp][a_]["S"]["point"] for a_ in arms], [S[grp][a_]["S"]["lo"] for a_ in arms],
                     [S[grp][a_]["S"]["hi"] for a_ in arms], C_TOT, f"S total, {lab}", ls=ls)
                xs = x[1:]
                band(ax, xs, [S[grp][a_]["SS"]["point"] for a_ in arms[1:]], [S[grp][a_]["SS"]["lo"] for a_ in arms[1:]],
                     [S[grp][a_]["SS"]["hi"] for a_ in arms[1:]], C_INC, f"S$^S$ incremental, {lab}", ls=ls, marker="s")
            ax.set_ylim(0, 1); ax.set_title("B  persona-common squared share", loc="left", fontsize=9.5)

            # C -- magnitude and alignment
            ax = axes[ri, 2]
            for grp, lab, ls in (("MEAN8", "8-trait mean", "-"), ("TARGETS", "targets", "--")):
                band(ax, x, [S[grp][a_]["g"]["point"] for a_ in arms], [S[grp][a_]["g"]["lo"] for a_ in arms],
                     [S[grp][a_]["g"]["hi"] for a_ in arms], C_TOT, f"g = |dG|/|V$_{{base}}$|, {lab}", ls=ls)
                band(ax, x, [S[grp][a_]["cosT"]["point"] for a_ in arms], [S[grp][a_]["cosT"]["lo"] for a_ in arms],
                     [S[grp][a_]["cosT"]["hi"] for a_ in arms], C_ALT, f"cos(dG(e), dG(3)), {lab}", ls=ls, marker="^")
            ax.set_title("C  common-shift magnitude and direction", loc="left", fontsize=9.5)

            # D -- acquisition overlay, everything on 0-1
            ax = axes[ri, 3]
            r = ser[cname]["rows"]
            if not ser[cname]["fraction_unstable"]:
                fr = [(0.0, 0.0, 0.0)] + [(q["fraction"]["point"], q["fraction"]["lo"], q["fraction"]["hi"]) for q in r[1:]]
                band(ax, x, *zip(*fr), C_BEH, f"behaviour F ({cname})")
            fp, lo, hi, _ = geo_fraction(g, nb, g["layers"][LAYER]["traits"], "g")
            band(ax, x, fp, lo, hi, C_TOT, "common magnitude g, 8-trait")
            fp2, lo2, hi2, _ = geo_fraction(g, nb, tg, "g")
            band(ax, x, fp2, lo2, hi2, C_TOT, "common magnitude g, targets", ls="--")
            band(ax, x[1:], [S["MEAN8"][a_]["Q"]["point"] for a_ in arms[1:]], [S["MEAN8"][a_]["Q"]["lo"] for a_ in arms[1:]],
                 [S["MEAN8"][a_]["Q"]["hi"] for a_ in arms[1:]], C_INC, "Q: projection on final SFT common shift", marker="s")
            ax.axhline(1, color=MUTED, lw=0.7, ls=":"); ax.axhline(0, color=MUTED, lw=0.7, ls=":")
            ax.set_title("D  fraction of final SFT-induced change", loc="left", fontsize=9.5)

            for ci_, ax in enumerate(axes[ri]):
                style(ax, "SFT epoch" if xmode == "epoch" else "measured functional dose (L15)")
                if xmode == "dose":
                    ax.set_xscale("linear"); ax.set_xlim(dose.min() - 0.02, dose.max() + 0.02)
                    ax.set_xticks(np.round(np.linspace(dose.min(), dose.max(), 5), 2))
                    ax.set_xticklabels([f"{v:.2f}" for v in np.round(np.linspace(dose.min(), dose.max(), 5), 2)])
                ax.legend(fontsize=6.8, frameon=False, loc="best")

            if xmode == "epoch":
                for a_, e_, d_ in zip(arms, ep, dose):
                    row = {"constitution": c, "arm": a_, "epoch": round(float(e_), 4), "dose_L15": round(float(d_), 4)}
                    for n in [cname] + tg:
                        q = next(q for q in ser[n]["rows"] if q["arm"] == a_)
                        row[f"{n}_level"] = round(q["level"]["point"], 4)
                    for grp in ("MEAN8", "TARGETS"):
                        for k in ("g", "S", "SS", "cosT", "Q"):
                            v = S[grp][a_][k]
                            row[f"{grp}_{k}"] = None if v["point"] != v["point"] else round(v["point"], 4)
                    rows_csv.append(row)
                tp = table_points(c)
                def pick(e, grp, k):
                    v = S[grp][tp[e]][k]; return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if v["lo"] is not None else f"{v['point']:.3f}"
                def frac(e):
                    if ser[cname]["fraction_unstable"]:
                        return "n/a (unstable)"
                    q = next(q for q in ser[cname]["rows"] if q["arm"] == tp[e])["fraction"]
                    return f"{q['point']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}]"
                table.append({
                    "constitution": c, "contrast": cname,
                    "contrast_DPO": round(ser[cname]["rows"][0]["level"]["point"], 3),
                    "contrast_3ep": round(ser[cname]["rows"][-1]["level"]["point"], 3),
                    "F_0.5ep": frac(0.5), "F_1ep": frac(1.0),
                    **{f"S8_{k}": pick(e, "MEAN8", "S") for k, e in (("DPO", 0.0), ("0.5", 0.5), ("1", 1.0), ("3", 3.0))},
                    **{f"SS8_{k}": pick(e, "MEAN8", "SS") for k, e in (("0.5", 0.5), ("1", 1.0), ("3", 3.0))},
                    **{f"cos8_{k}": pick(e, "MEAN8", "cosT") for k, e in (("0.5", 0.5), ("1", 1.0))},
                    **{f"dose_{k}": round(fd[LAYER][tp[e]]["trait_vector_displacement"], 3)
                       for k, e in (("DPO", 0.0), ("0.5", 0.5), ("1", 1.0), ("3", 3.0))}})
        fig.suptitle("Persona-common geometry vs behavioural phenotype along the native sequential DPO+SFT "
                     "trajectory (L15; 95% question-bootstrap CIs, one training seed)", fontsize=10.5, x=0.01, ha="left")
        fig.savefig(A / ("fig_trajectory.png" if xmode == "epoch" else "fig_trajectory_dose.png"), dpi=150)
        plt.close(fig)

    with open(A / "trajectory_points.csv", "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows_csv for k in r))   # contrasts differ by constitution
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows_csv)
    with open(A / "table_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    for t in table:
        print(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
