#!/usr/bin/env python3
"""Appendix asset -- where in the introspection SFT the impulsiveness phenotype appears.

Scientific question: the dose-matched work established that SFT on the DPO-generated corpus
is the potent carrier. That is a statement about SFT's ENDPOINT. This asks when during SFT
the phenotype arrives, and whether the released-style factor merge tracks it.

Panels:
  A  B1 against training epoch, both constructions. Orientation only -- step and dose are
     correlated by construction, exactly as the raw stage ordering was.
  B  B1 against MEASURED functional dose, with the M_D and M_S dose-response curves from the
     dose-matched experiment drawn behind as reference. This is the panel any claim rests on:
     it asks whether a partially-trained SFT checkpoint beats simply scaling the DPO adapter
     to the same displacement.
  C  selectivity against measured dose, same references.
  D  dose-controlled potency: B1 at each checkpoint divided by B1 of M_D scaled to that same
     measured dose. 1.0 means the checkpoint is worth no more than moving the model that far
     with the DPO adapter alone; above 1.0 is phenotype that the displacement does not explain.

     NOT a cross-term panel. `mrg` minus `seq` would be a tempting difference to draw here and
     it does not mean what it looks like: seq is dW_D + 1.00*dW_S(t) while mrg is
     dW_D + 0.25*dW_S(t) + cross terms, so the two differ by a FOUR-FOLD SFT weight as well as
     by the merge, and their difference reads as the merge hurting when most of the gap is
     simply less SFT. The clean cross-term comparison is M_F vs M_D+0.25S at matched dose,
     which the dose-matched experiment already reports; doing it along the curve would need a
     third construction (A_D at 1.0 with A_S(t) at 0.25) that is not measured here.

Reference states are drawn as anchors, never joined into the curve: M_D is step 0 for both
constructions, and the original run's M_D+S / M_F endpoints are shown hollow because this
rerun's own endpoint (step 1122) is the one the curve is anchored on -- the two adapters
agree to 0.006% in norm but only 0.9946 in cosine.

Sources
  outputs/analysis/{caa_logits,common_shift,functional_dose}.json  via curve_common.load()
  outputs/analysis/oct_stage_dose_master.csv                       for the reference curves
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

CON_COLOR = {"seq": "#eb6834", "mrg": "#4a3aa7"}
CON_LABEL = {"seq": "$M_D + \\Delta W_S(t)$  (additive)",
             "mrg": "PEFT merge$(A_D, A_S(t))$  (released style)"}
REF_COLOR = {"M_D": "#2a78d6", "M_S": "#b5179e"}


def reference_curves():
    """(dose, B1, selectivity) for M_D and M_S from the dose-matched extraction rungs."""
    out = {}
    with open(REPO / "outputs" / "analysis" / "oct_stage_dose_master.csv") as fh:
        for r in csv.DictReader(fh):
            if r["row_source"] != "extraction" or r["state"] not in REF_COLOR:
                continue
            if not r["dose"] or not r["B1"]:
                continue
            out.setdefault(r["state"], []).append(
                (float(r["dose"]), float(r["B1"]),
                 float(r["selectivity"]) if r["selectivity"] else np.nan))
    for k in out:
        out[k].sort()
    return out


def main() -> None:
    use_style()
    rows = cc.load()
    refs = reference_curves()
    have = {(r["construction"], r["step"]) for r in rows if r["kind"] == "curve"}
    if not have:
        raise SystemExit("no curve arms in the analysis JSON yet")

    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.3))
    (axA, axB), (axC, axD) = axes
    src = []

    for con in ("seq", "mrg"):
        pts = cc.series(rows, con, "B1")
        if not pts:
            continue
        ep = [p[1] for p in pts]; ds = [p[2] for p in pts]; b1 = [p[3] for p in pts]
        for ax, x in ((axA, ep), (axB, ds)):
            ax.plot(x, b1, "-o", color=CON_COLOR[con], lw=1.4, ms=3.2, mew=0,
                    label=CON_LABEL[con], zorder=4)
        sel = cc.series(rows, con, "selectivity")
        if sel:
            axC.plot([p[1] for p in sel], [p[3] for p in sel], "-o", color=CON_COLOR[con],
                     lw=1.4, ms=3.2, mew=0, zorder=4)
        for st, epv, dv, v in pts:
            src.append({"construction": con, "step": st, "epoch": epv,
                        "dose": round(dv, 4), "B1": round(v, 4)})

    # reference dose-response from the dose-matched experiment, drawn behind
    for st, series in refs.items():
        d = [p[0] for p in series]
        axB.plot(d, [p[1] for p in series], "--s", color=REF_COLOR[st], lw=1.0, ms=2.6,
                 mew=0, alpha=0.75, zorder=2,
                 label=f"$M_{{{st[2:]}}}$ scaled to dose")
        # C's x-axis is epochs, which the scaled reference states do not have; show their
        # trained-state selectivity as a level instead.
        trained = next((r for r in rows if r["state"] == st and r["kind"] == "reference"), None)
        if trained and trained.get("selectivity") is not None:
            axC.axhline(trained["selectivity"], color=REF_COLOR[st], lw=0.9, ls="--",
                        alpha=0.75, zorder=2)
            axC.text(0.015, trained["selectivity"], f"${{{st.replace('M_', 'M_')}}}$",
                     color=REF_COLOR[st], fontsize=6, va="center", ha="left",
                     transform=axC.get_yaxis_transform(),
                     bbox=dict(fc="white", ec="none", pad=0.6))

    # the original run's endpoints, hollow: this rerun has its own
    for arm, st in (("impulsiveness_repro_DplusS", "M_D+S"), ("impulsiveness_repro", "M_F")):
        r = next((x for x in rows if x["arm"] == arm), None)
        if r and r.get("B1") is not None and r.get("dose"):
            axB.plot([r["dose"]], [r["B1"]], "o", mfc="none", mec=INK, mew=0.9, ms=5,
                     zorder=5)
            axB.annotate(st, (r["dose"], r["B1"]), textcoords="offset points",
                         xytext=(-7, -1), fontsize=6, color=INK, ha="right")

    # D -- dose-controlled potency against the scaled-M_D reference. Interpolation is
    # refused outside M_D's measured dose range rather than extrapolated.
    md = refs.get("M_D", [])
    axD.axhline(1.0, color=MUTED, lw=0.7, zorder=2)
    if md:
        rd = np.array([p[0] for p in md]); rb = np.array([p[1] for p in md])
        for con in ("seq", "mrg"):
            pts = [(e, d, v) for _, e, d, v in cc.series(rows, con, "B1")
                   if rd.min() <= d <= rd.max()]
            if not pts:
                continue
            eps = [p[0] for p in pts]
            rat = [p[2] / float(np.interp(p[1], rd, rb)) for p in pts]
            axD.plot(eps, rat, "-o", color=CON_COLOR[con], lw=1.4, ms=3.2, mew=0, zorder=4)
            for e, r_ in zip(eps, rat):
                src.append({"construction": f"{con}-potency-vs-MD", "step": "",
                            "epoch": round(e, 4), "dose": "", "B1": round(r_, 4)})

    axA.set_xlabel("introspection SFT epochs"); axA.set_ylabel("$B_1$")
    axA.set_title("A  phenotype against training time", loc="left", fontsize=7.5)
    axB.set_xlabel("measured functional dose"); axB.set_ylabel("$B_1$")
    axB.set_title("B  against measured dose", loc="left", fontsize=7.5)
    axC.set_xlabel("introspection SFT epochs"); axC.set_ylabel("selectivity")
    axC.set_title("C  selectivity against training time", loc="left", fontsize=7.5)
    axD.set_xlabel("introspection SFT epochs"); axD.set_ylabel("$B_1$ $\\div$ scaled $M_D$")
    axD.set_title("D  potency at equal displacement", loc="left", fontsize=7.5)
    for ax in (axA, axB, axC, axD):
        despine(ax)
        ax.margins(x=0.06)
    axB.legend(fontsize=5.9, frameon=False, loc="upper left", handletextpad=0.5,
               borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.4)

    save(fig, "figA_oct_sft_curve")
    write_source_data("figA_oct_sft_curve", src,
                      ["construction", "step", "epoch", "dose", "B1"])
    print(f"  curve points: {len(have)} arm-states")


if __name__ == "__main__":
    main()
