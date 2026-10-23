#!/usr/bin/env python3
"""Matched-dose excess with PAIRED intervals (joint_dose_bootstrap.py output) -> figures + CSV.

excess(e) = trajectory value at SFT state e − DPO-only ladder interpolated at e's measured dose,
one question draw per replicate shared by every arm, dose matched within replicate, B = 1000.

Overview (`fig_matched_dose<tag>`): one row per constitution, four panels:
  S excess, per-trait mean        L15 (orange circles) and L20 (blue squares)
  S excess, energy-pooled         a DIFFERENT estimand (sum of common energy / sum of total
                                  energy over traits), same layer encoding
  g excess, per-trait mean        L15
  target-contrast excess          10 occupational personas, L15 dose: B2 (impulsiveness),
                                  LOV (loving); none is defined for goodness or sycophancy
Companions (`fig_matched_dose_<c><tag>`): rows L15 / L20; columns S per-trait mean, S pooled,
g, contrast (L15 only, the dose the behaviour is matched at).

Withheld cells (estimability rule of joint_dose_bootstrap.py) get NO plotted value: a grey tag at
the bottom of the panel says "unbr." (the point's own dose lies outside its ladder) or "n/e"
(fewer than 97.5 % of replicates defined). Their point values, if any, are only in the CSV. Points
and 95 % percentile intervals are drawn separately (a percentile interval need not contain the
point). Pointwise evaluation intervals: not simultaneous, not training-seed uncertainty. Dose
intervals are not drawn (plug-in bias; handoff §3). The DPO-only state (excess 0 by construction)
is omitted.

    python3 scripts/audit/fig_matched_dose.py --tag _cmbase [--formats pdf png] [--out-dir DIR]
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workshop_iclr/scripts"))
J = REPO / "outputs/analysis/audit/joint_dose"
ORDER = ["impulsiveness", "loving", "goodness", "sycophancy"]
CONTRAST = {"impulsiveness": "B2", "loving": "LOV"}
LCOL = {"15": "#eb6834", "20": "#1d4f91"}
LMARK = {"15": "o", "20": "s"}
MUTED, GRID = "#5c5c58", "#d9d9d5"
COLS = [("S_MEAN8", "S excess, per-trait mean"), ("Spooled_MEAN8", "S excess, energy-pooled"),
        ("g_MEAN8", "g excess, per-trait mean"), ("contrast", "target-contrast excess (10 p.)")]


def series(r, c, L, key):
    """-> list of per-state dicts for (layer, key); 'contrast' = 10-persona B2/LOV at L15."""
    if key == "contrast":
        if c not in CONTRAST or L != "15":
            return None
        return r["layers"]["15"]["behaviour_10personas"][f"offset_{CONTRAST[c]}"]
    return r["layers"].get(L, {}).get("series", {}).get(key)


def draw(ax, r, c, key, layers, rows, dx=True):
    ep = np.array(r["epoch"])
    tags, n_plotted = [], 0
    for li, L in enumerate(layers):
        ser = series(r, c, L, key)
        if ser is None:
            continue
        shift = 1 + (0.035 * (li - (len(layers) - 1) / 2) if dx and len(layers) > 1 else 0)
        for e, s in zip(ep, ser):
            rows.append({"constitution": c, "layer": L, "estimand": key if key != "contrast" else f"offset_{CONTRAST[c]}_10personas",
                         "epoch": round(float(e), 4), **{k: s.get(k) for k in ("point", "lo", "hi", "estimable",
                         "defined_frac", "unbracketed_frac", "invalid_frac", "point_status", "reason")},
                         "plotted": bool(e > 0 and s["estimable"])})
            if e == 0:
                continue
            x = e * shift
            if s["estimable"]:
                ax.vlines(x, s["lo"], s["hi"], color=LCOL[L], lw=0.9)
                ax.plot(x, s["point"], LMARK[L], color=LCOL[L], ms=2.6, mec=LCOL[L])
                n_plotted += 1
            else:
                tags.append((x, "unbr." if s["point_status"] == "unbracketed" else "n/e", L))
    ax.axhline(0, color=MUTED, lw=0.6)
    for x, t, L in tags:
        ax.annotate(t, (x, 0), xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=((-2.6 if L == "15" else 2.6) if len(layers) > 1 else 0, 1.5),
                    rotation=90, fontsize=4.6, color=LCOL[L], alpha=0.9, ha="center", va="bottom")
    return n_plotted


def style(ax):
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.2); ax.set_xlim(0.08, 3.4)
    ax.set_xticks([0.25, 0.5, 1, 3]); ax.set_xticklabels([".25", ".5", "1", "3"])
    ax.set_xticks([0.75, 2], minor=True)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, out_dir, name, formats, written):
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300 if ext == "png" else None)
        written.append(out_dir / f"{name}.{ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_cmbase")
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--out-dir", default=str(J))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figstyle import use_style, FULL
    use_style()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cons = [c for c in ORDER if (J / f"joint_dose_{c}{a.tag}.json").exists()]
    R = {c: json.loads((J / f"joint_dose_{c}{a.tag}.json").read_text()) for c in cons}
    base = sorted({R[c]["base_arm"] for c in cons}); B = sorted({R[c]["bootstrap"] for c in cons})
    written, rows = [], []
    legend = [Line2D([], [], color=LCOL["15"], marker="o", ls="none", ms=3, label="L15 (point; line = 95% interval)"),
              Line2D([], [], color=LCOL["20"], marker="s", ls="none", ms=3, label="L20"),
              Line2D([], [], color="none", label='"unbr.": point dose outside its ladder; "n/e": <97.5% of draws defined — withheld')]
    foot = (f"Trajectory minus DPO-only ladder at matched functional dose; paired question bootstrap, B = {B[0]}, "
            f"shared draws, dose matched within each draw; base {', '.join(base)}; geometry over 10 occupational "
            "personas; contrast on the same 10 personas, at the L15 dose. Pointwise 95% intervals (evaluation only, "
            "one training seed). Energy-pooled S is a different estimand from the per-trait mean. Goodness and "
            "sycophancy have no prespecified target contrast.")

    # overview
    fig, axes = plt.subplots(len(cons), 4, figsize=(FULL, 1.3 * len(cons) + 0.95), sharex=True, squeeze=False)
    for ri, c in enumerate(cons):
        for ki, (key, title) in enumerate(COLS):
            ax = axes[ri, ki]
            if key == "contrast" and c not in CONTRAST:
                ax.text(0.5, 0.5, "no prespecified\ntarget contrast", ha="center", va="center", transform=ax.transAxes,
                        fontsize=5.8, color=MUTED)
                ax.set_yticks([]); ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_visible(False)
            else:
                draw(ax, R[c], c, key, ["15"] if key in ("g_MEAN8", "contrast") else ["15", "20"], rows)
            style(ax)
            if ri == 0:
                ax.set_title(title, fontsize=6.6)
            if ki == 0:
                ax.set_ylabel(c, fontsize=7)
            if key == "contrast" and c in CONTRAST:
                ax.text(0.97, 0.95, CONTRAST[c], transform=ax.transAxes, ha="right", va="top", fontsize=6.5, fontweight="bold")
            if not (key == "contrast" and c not in CONTRAST):
                ax.tick_params(labelbottom=(ri == len(cons) - 1 or (key == "contrast" and ri == max(
                    i for i, c_ in enumerate(cons) if c_ in CONTRAST))))
            else:
                ax.grid(False)
    fig.tight_layout(rect=(0, 0.16, 1, 1), h_pad=0.4, w_pad=0.5)
    fig.text(0.53, 0.135, "SFT epoch (symlog: linear to 1, log 1–3)", ha="center", fontsize=6.5, color=MUTED)
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, 0.065), ncol=3, fontsize=5.6, handlelength=1.0,
               columnspacing=0.9)
    fig.text(0.01, 0.003, foot, fontsize=5.0, color=MUTED, ha="left", va="bottom", wrap=True)
    save(fig, out_dir, f"fig_matched_dose{a.tag}", a.formats, written)
    plt.close(fig)

    # per-constitution companions
    for c in cons:
        has_c = c in CONTRAST
        fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.15), sharex=True, squeeze=False)
        n_pl = np.zeros((2, 4), int)
        for li, L in enumerate(["15", "20"]):
            for ki, (key, title) in enumerate(COLS):
                ax = axes[li, ki]
                if key == "contrast" and (not has_c or L == "20"):
                    msg = "no prespecified\ntarget contrast" if not has_c else "behaviour is matched\nat the L15 dose only"
                    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=5.8, color=MUTED)
                    ax.set_yticks([]); ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                    for sp in ("left", "bottom"):
                        ax.spines[sp].set_visible(False)
                else:
                    n_pl[li, ki] = draw(ax, R[c], c, key, [L], rows, dx=False)
                style(ax)
                if li == 0:
                    ax.set_title(title, fontsize=6.6)
                if ki == 0:
                    ax.set_ylabel(f"L{L} excess", fontsize=7)
                if key == "contrast" and has_c and L == "15":
                    ax.text(0.97, 0.95, CONTRAST[c], transform=ax.transAxes, ha="right", va="top", fontsize=6.5, fontweight="bold")
                if not (key == "contrast" and (not has_c or L == "20")):
                    ax.tick_params(labelbottom=(li == 1 or key == "contrast"))
                else:
                    ax.grid(False)
        # the two layers share an estimand per column: a common y range, from the panels with data
        for ki in range(3):
            lims = [axes[li, ki].get_ylim() for li in range(2) if n_pl[li, ki]]
            if lims:
                lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
                axes[0, ki].set_ylim(lo, hi); axes[1, ki].set_ylim(lo, hi)
        fig.suptitle(f"{c}: SFT trajectory minus scaled DPO at matched dose", fontsize=7.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.2, 1, 0.97), h_pad=0.4, w_pad=0.5)
        fig.text(0.53, 0.165, "SFT epoch (symlog: linear to 1, log 1–3)", ha="center", fontsize=6.5, color=MUTED)
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, 0.085), ncol=3, fontsize=5.6, handlelength=1.0,
                   columnspacing=0.9)
        fig.text(0.01, 0.003, foot, fontsize=5.0, color=MUTED, ha="left", va="bottom", wrap=True)
        save(fig, out_dir, f"fig_matched_dose_{c}{a.tag}", a.formats, written)
        plt.close(fig)

    # every plotted (or withheld) cell of the overview and the companions, once
    seen, uniq = set(), []
    for r_ in rows:
        k_ = (r_["constitution"], r_["layer"], r_["estimand"], r_["epoch"])
        if k_ not in seen:
            seen.add(k_); uniq.append(r_)
    rows = uniq
    with open(out_dir / f"matched_dose_points{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    written.append(out_dir / f"matched_dose_points{a.tag}.csv")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out_dir / f"fig_matched_dose{a.tag}_provenance.json").write_text(json.dumps({
        "figures": [p.name for p in written if p.suffix != ".csv"],
        "command": "python3 scripts/audit/fig_matched_dose.py " + " ".join(sys.argv[1:]), "code_revision": commit,
        "inputs": [str((J / f"joint_dose_{c}{a.tag}.json").relative_to(REPO)) for c in cons],
        "baseline": base, "layers": [15, 20], "persona_population": "10 occupational personas (geometry and contrast)",
        "estimands": "excess = trajectory - ladder at matched L-specific dose; S per-trait mean, S energy-pooled, g per-trait "
                     "mean; contrast = 10-persona B2/LOV offset contrast at the L15 dose",
        "intervals": f"95% percentile, paired question bootstrap, B = {B[0]}, pointwise; rule: withheld unless point defined "
                     "and >= 97.5% of draws defined"}, indent=1))
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
