#!/usr/bin/env python3
"""Appendix A1 -- the three candidate dose axes disagree, which is why one was chosen.

Every matched-dose claim in the paper rests on a choice of what "matched" means, and the
three obvious candidates give three different answers. This figure is that argument.

A. Neutral-prompt output KL against LoRA scale for the three untrained constructions,
   with the four constitutions marked at their own s=1. A random LoRA of the SAME weight
   norm as a constitution (s=1, the leftmost point of each curve) is functionally inert:
   KL ~0.001 against 0.61-1.21, a factor of ~700. Weight norm is therefore not a control
   variable, and the norm-matched control that an earlier plan would have run answers
   nothing.
B. The same arms on the axis the geometry statistics actually live on: displacement of the
   CAA answer-token activation, measured on the shared 6-cell calibration subset. Matching
   `goodness` costs s ~ 16-19 here but s ~ 24-25 on output KL -- the two axes disagree by
   about 50% on which random arm is the right control.

The rungs used in figures 3 and 4 were sited by measurement on panel B's axis, not chosen
in advance, because that is the axis the outcome is measured on. The coherence cliff is
marked in panel A: at s=32 the untrained arms flip 48-69% of argmax tokens and lapse into
repetition, so the control arms operate within a factor of two of visible damage -- the
caveat that governs every untrained-arm result in the paper.

Sources
  outputs/analysis/adapter_dose.json               relative ||dW||_F per constitution
  outputs/analysis/neutral_dose.json               KL for the constitutions at s=1
  outputs/analysis/neutral_dose_random_sweep.json  KL vs s for the random constructions
  outputs/analysis/activation_dose_probe{,_spec2}.json   CAA displacement vs s
  outputs/analysis/activation_dose_probe_constitutions.json  the same for constitutions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (ANALYSIS, FULL, TRAINED, TRAINED_COLOR, UNTRAINED_COLOR, MARKER,
                      LABEL, INK, MUTED, use_style, despine, save, write_source_data)

FAMILY = {"random_iid": "random_iid_s16", "random_spec": "random_spec_s19",
          "random_perm": "random_perm_s16"}     # family -> the arm whose colour it takes


def main() -> None:
    use_style()
    kl_c = json.load(open(ANALYSIS / "neutral_dose.json"))["arms"]
    kl_r = json.load(open(ANALYSIS / "neutral_dose_random_sweep.json"))["arms"]
    ad = json.load(open(ANALYSIS / "activation_dose_probe.json"))["results"]["15"]
    ad |= json.load(open(ANALYSIS / "activation_dose_probe_spec2.json"))["results"]["15"]
    adc = json.load(open(
        ANALYSIS / "activation_dose_probe_constitutions.json"))["results"]["15"]
    rows = []

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL, 2.3),
                                   gridspec_kw={"wspace": 0.28})

    # ---- A: neutral-prompt KL vs scale -------------------------------------------
    for fam, key in FAMILY.items():
        pts = sorted((float(k.rsplit("_s", 1)[1]), v["mean_kl"])
                     for k, v in kl_r.items() if k.startswith(fam + "_s"))
        # s=1 is the norm-matched point, measured separately in neutral_dose.json
        pts = [(1.0, kl_c[fam]["mean_kl"])] + pts
        xs, ys = zip(*pts)
        axA.plot(xs, ys, ls=(0, (3, 1.6)), color=UNTRAINED_COLOR[key], lw=1.2,
                 marker=MARKER[key], ms=3.4, mew=0, label=LABEL[key], zorder=3)
        for s, v in pts:
            rows.append({"panel": "A", "arm": fam, "scale": s, "measure": "neutral_KL",
                         "value": round(v, 5)})
    # all four constitutions sit at s=1 and would render as one stack; they are
    # jittered across 0.91-1.10 for legibility only -- every one of them is at s=1
    for arm, jx in zip(TRAINED, (0.91, 0.97, 1.04, 1.11)):
        v = kl_c[arm]["mean_kl"]
        axA.plot([jx], [v], marker=MARKER[arm], ms=4.2, mfc=TRAINED_COLOR[arm],
                 mec="white", mew=0.5, ls="none", zorder=5)
        rows.append({"panel": "A", "arm": arm, "scale": 1.0, "measure": "neutral_KL",
                     "value": round(v, 5)})
    axA.annotate("the four constitutions at $s{=}1$\n(x-jittered), KL 0.61\u20131.21",
                 xy=(1.13, 0.95), xytext=(1.9, 0.42), fontsize=6, color=MUTED,
                 ha="left", va="center", linespacing=1.2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2,
                                 shrinkB=2))
    axA.annotate("norm-matched random LoRAs:\n$\\sim$700$\\times$ less output change",
                 xy=(1.02, 1.05e-3), xytext=(1.9, 2.4e-3), fontsize=6, color=MUTED,
                 ha="left", va="center", linespacing=1.2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2,
                                 shrinkB=2))
    axA.axvspan(30, 36, color="#f2ece9", zorder=0)
    axA.text(37.0, 1.3e-2, "coherence\ncliff", fontsize=5.8, color=MUTED, ha="right",
             va="center", linespacing=1.15)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xticks([1, 2, 4, 8, 16, 32]); axA.set_xticklabels([1, 2, 4, 8, 16, 32])
    axA.set_xlim(0.85, 38)
    axA.set_xlabel("LoRA scale $s$   ($s{=}1$ is weight-norm matched)")
    axA.set_ylabel("mean KL vs base, neutral prompts")
    axA.set_title("A  output KL", loc="left")
    despine(axA)

    # ---- B: CAA answer-token displacement vs scale -------------------------------
    for fam, key in FAMILY.items():
        pts = sorted((float(k.rsplit("_s", 1)[1]), v["answer_token_displacement"])
                     for k, v in ad.items() if k.startswith(fam + "_s"))
        xs, ys = zip(*pts)
        axB.plot(xs, ys, ls=(0, (3, 1.6)), color=UNTRAINED_COLOR[key], lw=1.2,
                 marker=MARKER[key], ms=3.4, mew=0, label=LABEL[key], zorder=3)
        for s, v in pts:
            rows.append({"panel": "B", "arm": fam, "scale": s,
                         "measure": "CAA_answer_token_displacement", "value": round(v, 4)})
    gd = adc["llama-3.1-8b-goodness"]["answer_token_displacement"]
    axB.axhline(gd, color=TRAINED_COLOR["goodness"], lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    axB.text(1.2, gd * 1.06, "goodness, $s{=}1$", fontsize=6,
             color=TRAINED_COLOR["goodness"], va="bottom")
    rows.append({"panel": "B", "arm": "goodness", "scale": 1.0,
                 "measure": "CAA_answer_token_displacement", "value": round(gd, 4)})
    axB.set_xscale("log")
    axB.set_xticks([1, 2, 4, 8, 16, 32]); axB.set_xticklabels([1, 2, 4, 8, 16, 32])
    axB.set_xlim(0.85, 38)
    axB.set_xlabel("LoRA scale $s$")
    axB.set_ylabel("CAA answer-token displacement")
    axB.set_title("B  the axis the geometry lives on", loc="left")
    despine(axB)

    # figure-level legend at the top, as in figures 1, 3 and 4. Panel A's curves run
    # corner to corner, so there is no in-panel box that does not sit on data.
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.995),
               handletextpad=0.35, columnspacing=1.4, borderpad=0)

    save(fig, "figA1_dose_calibration")
    write_source_data("figA1_dose_calibration", rows,
                      ["panel", "arm", "scale", "measure", "value"])


if __name__ == "__main__":
    main()
