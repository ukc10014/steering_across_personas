#!/usr/bin/env python3
"""Matched-dose excess with PAIRED intervals (joint_dose_bootstrap.py output) -> figure + CSV.

Rows: constitutions present. Columns: S excess (per-trait mean, the original estimand) at L15
and L20, S excess (energy-pooled, a different estimand) at L15, g excess at L15, and the
constitution's behavioural contrast excess (10 occupational personas; impulsiveness B2, loving
LOV; goodness/sycophancy have no prespecified contrast -> the impulsivity offset is not
substituted, the column says "no contrast"). Points whose excess is withheld by the rule
(unbracketed or < 97.5 % defined replicates) are drawn as open grey markers at their point
value where one exists, and listed in the CSV with the reason; nothing is extrapolated.

    python3 scripts/audit/fig_matched_dose.py --tag _cmbase
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/analysis/audit/joint_dose"
ORDER = ["impulsiveness", "loving", "goodness", "sycophancy"]
CONTRAST = {"impulsiveness": "B2", "loving": "LOV"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_cmbase")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cons = [c for c in ORDER if (J / f"joint_dose_{c}{a.tag}.json").exists()]
    cols = [("15", "series", "S_MEAN8", "S excess, per-trait mean, L15"),
            ("20", "series", "S_MEAN8", "S excess, per-trait mean, L20"),
            ("15", "series", "Spooled_MEAN8", "S excess, energy-pooled, L15"),
            ("15", "series", "g_MEAN8", "g excess, per-trait mean, L15"),
            ("15", "behaviour_10personas", None, "behaviour contrast excess (10 personas)")]
    fig, axes = plt.subplots(len(cons), len(cols), figsize=(3.5 * len(cols), 2.7 * len(cons)),
                             squeeze=False, constrained_layout=True)
    rows = []
    for ri, c in enumerate(cons):
        r = json.loads((J / f"joint_dose_{c}{a.tag}.json").read_text())
        ep = np.array(r["epoch"])
        for ci, (L, block, key, title) in enumerate(cols):
            ax = axes[ri, ci]
            if key is None:
                key = f"offset_{CONTRAST[c]}" if c in CONTRAST else None
            ser = r["layers"].get(L, {}).get(block, {}).get(key) if key else None
            if ser is None:
                ax.text(0.5, 0.5, "no prespecified contrast" if key is None else "not computed",
                        ha="center", va="center", transform=ax.transAxes, color="#6b6b6b", fontsize=8)
            else:
                for e, s in zip(ep, ser):
                    rows.append({"constitution": c, "layer": L, "estimand": f"{block}:{key}", "epoch": round(float(e), 4),
                                 **{k: s.get(k) for k in ("point", "lo", "hi", "estimable", "defined_frac",
                                                          "unbracketed_frac", "invalid_frac", "point_status", "reason")}})
                    if e == 0:
                        continue                          # DPO-only state: excess 0 by construction
                    if s["estimable"]:
                        # interval and point drawn separately: a percentile interval need not contain the point
                        ax.vlines(e, s["lo"], s["hi"], color="#eb6834", lw=1.2)
                        ax.plot(e, s["point"], "o", color="#eb6834", ms=4)
                    elif s.get("point") is not None:
                        ax.plot(e, s["point"], "o", mfc="white", mec="#8c8c8c", ms=5)
                    else:
                        ax.annotate("unbracketed" if "unbracketed" in (s.get("reason") or "") else "n/e",
                                    (e, 0), fontsize=6.5, color="#8c8c8c", rotation=90, ha="center", va="bottom")
            ax.axhline(0, color="#6b6b6b", lw=0.7)
            ax.set_xscale("symlog", linthresh=1.0, linscale=2.0); ax.set_xlim(-0.03, 3.4)
            ax.set_xticks([0.25, 0.5, 0.75, 1, 2, 3]); ax.set_xticklabels([".25", ".5", ".75", "1", "2", "3"])
            ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7.5, colors="#6b6b6b")
            if ri == 0:
                ax.set_title(title, loc="left", fontsize=9)
            if ri == len(cons) - 1:
                ax.set_xlabel("SFT epoch", fontsize=8, color="#6b6b6b")
            if ci == 0:
                ax.set_ylabel(c, fontsize=9)
    fig.suptitle("SFT trajectory minus scaled-DPO ladder at matched functional dose. Paired 95% question-bootstrap "
                 f"intervals (shared draws; dose matched within replicate; base {r['base_arm']}; one training seed). "
                 "Open grey: interval withheld by the 97.5% rule", fontsize=9.5, x=0.01, ha="left")
    fig.savefig(J / f"fig_matched_dose{a.tag}.png", dpi=150)
    with open(J / f"matched_dose_points{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", J / f"fig_matched_dose{a.tag}.png")


if __name__ == "__main__":
    main()
