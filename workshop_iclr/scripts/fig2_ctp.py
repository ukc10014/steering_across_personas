#!/usr/bin/env python3
"""Figure 2 -- the constitution x trait x persona decomposition.

The prereg's question is whether a constitution acts differently on specific persona x
trait cells. Writing X_{c,t,p} for the per-cell change from base, the balanced three-way
partition X = mu + C + T + P + CT + CP + TP + CTP is exact, so the energy in CTP is the
answer.

A. The partition for the four trained constitutions at L15, grouped by whether a term
   depends on WHICH constitution at all. 71.4% of the change does not: it is the same
   trait structure whichever constitution produced it. Of the 28.6% that does, the
   constitution x trait term is by far the largest, and the constitution x persona term is
   the smallest of all eight -- CT/CP = 13.0. That ratio is the substantive finding, and it
   is the opposite of the preregistered expectation: constitutional differences are much
   more trait-dependent than persona-dependent, and the triple interaction the prereg asks
   about is 3.6%.
B. The same triple interaction against a matched-degrees-of-freedom reference. CTP's share
   depends on the number of constitutions through the degrees of freedom, so the four-arm
   trained band is not comparable to the three-arm untrained band; every three-arm SUBSET
   of the trained family is, and all four sit below the untrained band with no interval
   overlap, at both layers.

   READ PANEL B AS A CONTROL, NOT AS A RESULT. What it licenses is "a nonzero fine-grained
   interaction is not evidence of anything semantic, because an untrained perturbation
   produces more of it". It does NOT license "training suppresses context-specific
   interaction": these random arms sit within a factor of two of the measured coherence
   cliff (appendix A1), and incoherent behaviour would present exactly as cell-specific
   idiosyncrasy, which is the quantity being measured. Separating those two accounts needs
   the sham-trained LoRA, which does not exist yet.

Source: outputs/analysis/three_way_interaction.json (scripts/caa_three_way_interaction.py),
cross-fitted over 40 question half-splits, 200-replicate question bootstrap shared across
arms so every band comparison is paired.

TWO THINGS DELIBERATELY NOT PLOTTED.
  - The degrees-of-freedom share. CTP spans 59% of the cell space, so independent per-cell
    noise would land there; it does not, because CAA questions are shared across arms and
    personas within a trait, making question idiosyncrasy a t-indexed effect that lands in
    T, TP and CT. Section 9.2 retracts df share as a reference, so plotting it beside the
    measured shares would reintroduce a comparison the analysis rejected.
  - The per-cell "319/320 significantly above zero" count. That null tests a squared
    magnitude against exact zero and separates nothing.

Panel A carries an interval only on CTP: the bootstrap in the source file stores
replicates for the CTP term, not for the other seven.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, INK, MUTED, GRID, use_style, despine, save,
                      write_source_data)

# Grouped by whether the term depends on the constitution, then descending within group.
# A globally sorted bar chart hides the comparison that matters (CT against CP).
TERMS_FREE = ["T", "TP", "mu", "P"]          # no c index: the same whichever constitution
TERMS_C = ["CT", "C", "CTP", "CP"]           # constitution-dependent
TERMS = TERMS_FREE + TERMS_C
TERM_LABEL = {"mu": r"$\mu$", "C": "C", "T": "T", "P": "P", "CT": "C$\\times$T",
              "CP": "C$\\times$P", "TP": "T$\\times$P", "CTP": "C$\\times$T$\\times$P"}
BANDS = ["trained[good+math+impu]", "trained[good+math+misa]",
         "trained[good+impu+misa]", "trained[math+impu+misa]", "null"]
# The four trained bands are the leave-one-out 3-arm subsets, so naming the arm each
# one DROPS is both shorter and easier to scan than listing the three it keeps.
BAND_LABEL = {
    "trained[good+math+impu]": "trained \u2212 misalignment",
    "trained[good+math+misa]": "trained \u2212 impulsiveness",
    "trained[good+impu+misa]": "trained \u2212 mathematical",
    "trained[math+impu+misa]": "trained \u2212 goodness",
    "null": "untrained controls",
}
BAND_LONG = {
    "trained[good+math+impu]": "goodness + mathematical + impulsiveness",
    "trained[good+math+misa]": "goodness + mathematical + misalignment",
    "trained[good+impu+misa]": "goodness + impulsiveness + misalignment",
    "trained[math+impu+misa]": "mathematical + impulsiveness + misalignment",
    "null": "random_iid_s16 + random_spec_s19 + random_perm_s16",
}
ACCENT = "#eb6834"      # the CTP term, the one the prereg asks about


def main() -> None:
    use_style()
    d = json.load(open(ANALYSIS / "three_way_interaction.json"))
    rows = []

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.90})

    # ---- A: the eight-term partition, trained band, L15 --------------------------
    band = d["15"]["bands"]["trained"]
    total = band["crossfit"]["total"]
    shares = {t: band["crossfit"][t] / total for t in TERMS}
    # a one-row gap between the two groups
    ypos = {t: (len(TERMS) - i if i < len(TERMS_FREE) else len(TERMS) - i - 1)
            for i, t in enumerate(TERMS)}
    y = [ypos[t] for t in TERMS]
    colors = ["#cfd8e2" if t in TERMS_FREE else
              (ACCENT if t in ("CT", "CP") else "#8fa8c0") for t in TERMS]
    axA.barh(y, [shares[t] for t in TERMS], height=0.66, color=colors, zorder=3)
    # the only term the source file bootstraps
    ci = band["ci"]["ctp_share"]
    axA.plot(ci, [ypos["CTP"]] * 2, color=INK, lw=0.9, zorder=5, solid_capstyle="butt")
    for t in TERMS:
        axA.text(shares[t] + 0.008, ypos[t], f"{shares[t]:.3f}", va="center", ha="left",
                 fontsize=6.4, color=INK)
    free_tot = sum(shares[t] for t in TERMS_FREE)
    c_tot = sum(shares[t] for t in TERMS_C)
    axA.text(0.44, ypos["T"] + 0.55, f"no constitution index — {free_tot:.1%}",
             fontsize=6, color=MUTED, ha="right", va="bottom")
    axA.text(0.44, ypos["CT"] + 0.55, f"constitution-dependent — {c_tot:.1%}",
             fontsize=6, color=MUTED, ha="right", va="bottom")
    axA.annotate(f"CT / CP = {shares['CT'] / shares['CP']:.0f}$\\times$",
                 xy=(0.30, ypos["C"] - 0.5), fontsize=6.6, color=ACCENT,
                 ha="center", va="center", fontweight="bold")
    for t in TERMS:
        rows.append({"panel": "A", "layer": 15,
                     "band": "goodness + mathematical + impulsiveness + misalignment",
                     "term": t, "share": round(shares[t], 4),
                     "ci_lo": round(ci[0], 4) if t == "CTP" else "",
                     "ci_hi": round(ci[1], 4) if t == "CTP" else ""})
    axA.set_yticks(y)
    axA.set_yticklabels([TERM_LABEL[t] for t in TERMS])
    axA.set_ylim(-0.7, len(TERMS) + 1.3)
    axA.set_xlim(0, 0.46)
    axA.set_xlabel("share of cross-fitted energy")
    axA.set_title("A  variance partition (trained, L15)", loc="left")
    axA.axhline(ypos["P"] - 0.5, color=GRID, lw=0.7, zorder=1)
    despine(axA, grid_axis="x")

    # ---- B: matched-df CTP share, trained subsets vs untrained -------------------
    yb = np.arange(len(BANDS))[::-1]
    for li, (layer, mfc, lab) in enumerate([("15", INK, "layer 15"),
                                            ("20", "white", "layer 20")]):
        dy = 0.17 if li == 0 else -0.17
        for b, yy in zip(BANDS, yb):
            v = d[layer]["bands"][b]["ctp_share"]
            c = d[layer]["bands"][b]["ci"]["ctp_share"]
            untr = b == "null"
            axB.plot(c, [yy + dy] * 2, color=INK, lw=0.9,
                     ls="-" if not untr else (0, (2.2, 1.2)), zorder=4,
                     solid_capstyle="butt")
            axB.plot([v], [yy + dy], marker="o" if not untr else "s", ms=4.2,
                     ls="none", mfc=mfc, mec=INK, mew=0.9, zorder=5,
                     label=lab if b == BANDS[0] else None)
            rows.append({"panel": "B", "layer": int(layer), "band": BAND_LONG[b],
                         "term": "CTP", "share": round(v, 4),
                         "ci_lo": round(c[0], 4), "ci_hi": round(c[1], 4)})
    # shade the control row so the trained/untrained split is visible without colour
    axB.axhspan(yb[-1] - 0.45, yb[-1] + 0.45, color="#f2f2ee", zorder=0)
    axB.set_yticks(yb)
    axB.set_yticklabels([BAND_LABEL[b] for b in BANDS], fontsize=6.6)
    axB.set_xlabel("C$\\times$T$\\times$P share (3 arms per band)")
    axB.set_title("B  C$\\times$T$\\times$P against an untrained reference", loc="left")
    axB.set_xlim(0, 0.125)
    axB.legend(loc="upper right", ncol=1, handletextpad=0.4, borderpad=0.2)
    despine(axB, grid_axis="x")
    axB.yaxis.set_tick_params(length=0)

    save(fig, "fig2_ctp")
    write_source_data("fig2_ctp", rows,
                      ["panel", "layer", "band", "term", "share", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    main()
