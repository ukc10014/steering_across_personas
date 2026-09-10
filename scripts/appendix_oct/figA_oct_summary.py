#!/usr/bin/env python3
"""Two cross-experiment summary figures for the OCT stage work.

These are the "read these two first" figures. Everything else in the OCT appendix is a
detail of one or the other.

  figA_oct_summary_dose  WHERE the phenotype comes from. Every construction on one
      measured-dose axis, so states are compared at equal functional displacement rather
      than at equal nominal scale or equal weight norm -- the error section 7.1 records.
      Reading it: at any vertical slice, the ordering is the result. The SFT checkpoints are
      drawn as markers, not joined: their natural parameter is training time, not dose, and
      two checkpoints at nearly equal dose can differ by more than the local trend, so a line
      would imply an ordering along this axis that does not exist.

  figA_oct_summary_time  WHEN it arrives. The introspection SFT measured through training.
      Reading it: the rise is over well before one epoch, and the flat part afterwards is
      the claim.

Both draw from archived analysis only and regenerate from scratch.

Sources
  outputs/analysis/oct_stage_dose_master.csv        the dose ladders
  outputs/analysis/{caa_logits,common_shift,functional_dose}.json  via curve_common
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workshop_iclr" / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "appendix_oct"))
from figstyle import FULL, INK, MUTED, use_style, despine, save, write_source_data  # noqa: E402
import curve_common as cc                                                          # noqa: E402

MASTER = REPO / "outputs" / "analysis" / "oct_stage_dose_master.csv"

# One colour per construction, used identically in both figures and matching the
# per-experiment figures already in the appendix.
COLOR = {"M_D": "#2a78d6", "M_D+0.25S": "#5aa9e6", "M_F": "#111111",
         "M_S": "#b5179e", "seq": "#eb6834", "mrg": "#4a3aa7"}
MARK = {"M_D": "o", "M_D+0.25S": "s", "M_F": "D", "M_S": "v"}
NAME = {"M_D": "$M_D$  DPO only", "M_D+0.25S": "$M_{D+0.25S}$  additive",
        "M_F": "$M_F$  released merge", "M_S": "$M_S$  SFT from base"}


def ladders():
    """state -> [(dose, B1, selectivity)] from the dose-matched extraction rungs."""
    out: dict[str, list] = {}
    with open(MASTER) as fh:
        for r in csv.DictReader(fh):
            if r["row_source"] != "extraction" or not r["dose"] or not r["B1"]:
                continue
            out.setdefault(r["state"], []).append(
                (float(r["dose"]), float(r["B1"]),
                 float(r["selectivity"]) if r.get("selectivity") else np.nan,
                 r["is_trained_state"] == "True"))
    for k in out:
        out[k].sort()
    return out


def fig_dose(lad, rows, src):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    for st in ("M_D", "M_D+0.25S", "M_F", "M_S"):
        pts = lad.get(st)
        if not pts:
            continue
        d = [p[0] for p in pts]
        ax1.plot(d, [p[1] for p in pts], "--", color=COLOR[st], lw=1.1, marker=MARK[st],
                 ms=3.0, mew=0, alpha=0.9, label=NAME[st], zorder=3)
        ax2.plot(d, [p[2] for p in pts], "--", color=COLOR[st], lw=1.1, marker=MARK[st],
                 ms=3.0, mew=0, alpha=0.9, zorder=3)
        for dd, b1, sel, trained in pts:
            if trained:                       # the as-released state, ringed
                ax1.plot([dd], [b1], "o", mfc="none", mec=INK, mew=0.9, ms=6.5, zorder=5)
            src.append({"figure": "dose", "series": st, "dose": round(dd, 4),
                        "B1": round(b1, 4), "selectivity": None if sel != sel else round(sel, 4)})

    # the SFT checkpoint trajectory, drawn through the same plane
    for con in ("seq",):
        pts = [(d, b) for _, _, d, b in cc.series(rows, con, "B1")]
        sel = {d: s for _, _, d, s in cc.series(rows, con, "selectivity")}
        if not pts:
            continue
        pts.sort()
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none",
                 color=COLOR[con], marker="o", ms=3.4, mew=0, alpha=0.9,
                 label="SFT checkpoints", zorder=4)
        sd = sorted(sel)
        ax2.plot(sd, [sel[d] for d in sd], linestyle="none", color=COLOR[con],
                 marker="o", ms=3.4, mew=0, alpha=0.9, zorder=4)
        for d, b in pts:
            src.append({"figure": "dose", "series": "SFT checkpoints", "dose": round(d, 4),
                        "B1": round(b, 4),
                        "selectivity": None if d not in sel else round(sel[d], 4)})

    ax1.set_xlabel("measured functional dose"); ax1.set_ylabel("$B_1$  impulsivity contrast")
    ax1.set_title("A  strength at equal displacement", loc="left", fontsize=7.5)
    ax2.set_xlabel("measured functional dose"); ax2.set_ylabel("selectivity")
    ax2.set_title("B  specificity at equal displacement", loc="left", fontsize=7.5)
    for ax in (ax1, ax2):
        despine(ax); ax.margins(x=0.06)
    ax1.legend(fontsize=5.9, frameon=False, loc="upper left", handletextpad=0.5,
               borderpad=0.1, labelspacing=0.28)
    fig.tight_layout(pad=0.4)
    save(fig, "figA_oct_summary_dose")


def fig_time(rows, src):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    for con in ("seq", "mrg"):
        b1 = cc.series(rows, con, "B1")
        sel = cc.series(rows, con, "selectivity")
        if not b1:
            continue
        lab = {"seq": "$M_D+\\Delta W_S(t)$  additive",
               "mrg": "PEFT merge  released style"}[con]
        ax1.plot([p[1] for p in b1], [p[3] for p in b1], "-o", color=COLOR[con], lw=1.5,
                 ms=3.0, mew=0, label=lab, zorder=4)
        if sel:
            ax2.plot([p[1] for p in sel], [p[3] for p in sel], "-o", color=COLOR[con],
                     lw=1.5, ms=3.0, mew=0, zorder=4)
        for st, ep, d, v in b1:
            src.append({"figure": "time", "series": con, "epoch": ep, "step": st,
                        "B1": round(v, 4)})

    # mark where the rise is over: the earliest checkpoint within 10% of the endpoint
    seq = cc.series(rows, "seq", "B1")
    if seq:
        end = seq[-1][3]
        hit = next((p for p in seq if p[3] >= 0.9 * end), None)
        if hit:
            for ax in (ax1, ax2):
                ax.axvline(hit[1], color=MUTED, lw=0.8, ls=":", zorder=2)
            ax1.annotate(f"within 10% of the\nendpoint by {hit[1]:.2f} epochs",
                         xy=(hit[1], end * 0.55), xytext=(6, 0), textcoords="offset points",
                         fontsize=6, color=MUTED, va="center")

    ax1.set_xlabel("introspection SFT epochs"); ax1.set_ylabel("$B_1$  impulsivity contrast")
    ax1.set_title("A  the phenotype arrives early", loc="left", fontsize=7.5)
    ax2.set_xlabel("introspection SFT epochs"); ax2.set_ylabel("selectivity")
    ax2.set_title("B  specificity through training", loc="left", fontsize=7.5)
    for ax in (ax1, ax2):
        despine(ax); ax.margins(x=0.06)
    ax1.legend(fontsize=5.9, frameon=False, loc="lower right", handletextpad=0.5,
               borderpad=0.1, labelspacing=0.28)
    fig.tight_layout(pad=0.4)
    save(fig, "figA_oct_summary_time")


def main() -> None:
    use_style()
    rows = cc.load()
    lad = ladders()
    src: list[dict] = []
    fig_dose(lad, rows, src)
    fig_time(rows, src)
    write_source_data("figA_oct_summary", src,
                      ["figure", "series", "dose", "epoch", "step", "B1", "selectivity"])
    print(f"  summary figures written ({len(src)} source rows)")


if __name__ == "__main__":
    main()
