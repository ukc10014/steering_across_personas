#!/usr/bin/env python3
"""Figures + table for the common-trajectory panel. spec §5 and amendment A1.

Panels (identical for every constitution; A1: the procedure and the readout are held fixed):

  A  eight-trait signed offset profile vs base (absolute levels, thin coloured lines, no bands);
     the constitution's prespecified target contrast, where one exists (B2 impulsiveness,
     LOV loving), bold black with its 95 % band
  B  TIMING, all on one scale = fraction of the final SFT-induced change (not clipped at 1):
       q(e)       = <D(e), D(final)>/||D(final)||^2, D = 8-trait offset change since DPO
       Q(e)       = projection on the final incremental common shift (8-trait mean)
       g~(e)      = (g(e) - g(0)) / (g(final) - g(0)), normalised change in common magnitude
                    since the DPO-only state (NOT raw g, NOT the incremental-shift magnitude)
  C  persona-common squared share: S total (vs base) and S^S incremental (vs the DPO-only state)
  D  relative common magnitude g = ||dG|| / mean_p ||V_base,p|| and cos_T = cos(dG(e), dG(final))

x = SFT epoch, 0 = the completed DPO-only state; symlog axis, linear on [0, 1] and logarithmic
on [1, 3]. fig_trajectory_dose: x = measured functional dose (L15), no dose intervals drawn.
Geometry at L15 (L20 is in the JSON). Bands are 95 % question-bootstrap percentile intervals
(evaluation only, one training seed); the eight per-trait offset lines in A carry none.
Non-estimable values (estimability.py) are gaps, never zeros.

Layouts: ABCD = one row per constitution (appendix); a panel subset (e.g. AB) = one column per
constitution and one row per panel (main text).

    python3 scripts/common_trajectory/figures.py --geometry-tag _cmbase --behaviour-tag _10personas \\
        --dose-tag _cmbase --out-tag _cmbase [--panels AB] [--formats pdf png] [--out-dir DIR]
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "workshop_iclr/scripts"))
from states import CONTRASTS, table_points   # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
ORDER = ["impulsiveness", "loving", "goodness", "sycophancy", "misalignment"]
LAYER = "15"
MUTED, GRID = "#5c5c58", "#d9d9d5"
# Okabe-Ito (CVD-safe), yellow swapped for grey for contrast on white; fixed trait -> colour,
# the same in every panel and every constitution.
TRAIT_COL = {"assertiveness": "#E69F00", "confidence": "#56B4E9", "deference": "#009E73",
             "empathy": "#D55E00", "honesty": "#0072B2", "impulsivity": "#CC79A7",
             "risk_taking": "#7a4fd0", "warmth": "#8c8c8c"}
TRAIT_ORDER = ["assertiveness", "confidence", "deference", "empathy", "honesty", "impulsivity",
               "risk_taking", "warmth"]
C_BEH, C_Q, C_G, C_S, C_SS, C_COS = "#1f1f1f", "#7a4fd0", "#eb6834", "#eb6834", "#7a4fd0", "#1f9e89"
LAB = {"q": r"$q$  behaviour profile", "Q": r"$Q$  common-shift projection",
       "gt": r"$\tilde g$  normalised $g$ increase since DPO",
       "S": r"$S$  total (vs base)", "SS": r"$S^S$  incremental (vs DPO-only)",
       "g": r"$g$  relative common magnitude", "cosT": r"$\cos_T$  to final total shift"}
TITLE = {"A": "A  eight-trait offsets vs base", "B": "B  fraction of final SFT-induced change",
         "C": "C  persona-common squared share", "D": "D  common magnitude and direction"}
TITLE_SHORT = {"A": "A  offsets vs base", "B": "B  fraction of final change",
               "C": "C  common squared share", "D": "D  magnitude, direction"}

# Input selection (main()): the historical files by default; --geometry-tag etc. select the
# audit reruns (v2/geometry_<c><tag>.json + v2/geometry_raw_<c>_L15<tag>.npz, the 10-persona
# behaviour files, a tagged functional-dose file). Tagged runs write tagged outputs.
SRC = {"geometry_tag": None, "behaviour_tag": "", "dose_tag": ""}
PLOTTED: list[dict] = []          # every plotted value, written beside the figure


def in_paths(c):
    return {"behaviour": A / (f"v2/behaviour_{c}{SRC['behaviour_tag']}.json" if SRC["behaviour_tag"] else f"behaviour_{c}.json"),
            "dose": A / f"functional_dose_{c}{SRC['dose_tag']}.json",
            "geometry": A / (f"v2/geometry_{c}{SRC['geometry_tag']}.json" if SRC["geometry_tag"] is not None else f"geometry_{c}.json"),
            "geometry_boot": A / (f"v2/geometry_raw_{c}_L{LAYER}{SRC['geometry_tag']}.npz" if SRC["geometry_tag"] is not None
                                  else f"geometry_boot_{c}_L{LAYER}.npz")}


def load(c):
    p = in_paths(c)
    return (json.loads(p["behaviour"].read_text()), json.loads(p["geometry"].read_text()),
            json.loads(p["dose"].read_text()), np.load(p["geometry_boot"]))


def g_fraction(g, nb):
    arms = [str(x) for x in nb["arms"]]
    traits = g["layers"][LAYER]["traits"]
    pt = np.array([g["layers"][LAYER]["summary"]["MEAN8"][x]["g"]["point"] for x in arms])
    if "qty" in nb.files:                                      # historical geometry_boot file
        q = list(map(str, nb["qty"])).index("g")
        bt = np.mean([nb[t][:, :, q] for t in traits], 0)
    else:                                                      # v2 raw accumulators
        R = list(map(str, nb["raw"]))
        gt = [np.sqrt(np.clip(nb[f"boot_{t}"][..., R.index("g2")], 0, None)) / nb[f"boot_{t}"][..., R.index("bn")]
              for t in traits]
        bt = np.mean(gt, 0)
        bt = bt[np.isfinite(bt).all(1)]                        # draws with too few distinct questions
    fb = (bt - bt[:, :1]) / (bt[:, -1:] - bt[:, :1])
    lo, hi = np.nanpercentile(fb, [2.5, 97.5], axis=0)
    return (pt - pt[0]) / (pt[-1] - pt[0]), lo, hi


def _f(v):
    return np.nan if v is None else float(v)


def record(c, panel, series, arms, x, p, lo=None, hi=None):
    for i, a_ in enumerate(arms):
        PLOTTED.append({"constitution": c, "panel": panel, "series": series, "arm": a_, "x": round(float(x[i]), 5),
                        "point": None if np.isnan(_f(p[i])) else round(_f(p[i]), 5),
                        "lo": None if lo is None or np.isnan(_f(lo[i])) else round(_f(lo[i]), 5),
                        "hi": None if hi is None or np.isnan(_f(hi[i])) else round(_f(hi[i]), 5)})


def band(ax, x, p, lo, hi, color, label, ls="-", marker="o", lw=1.4):
    """None (a withheld, non-estimable value) plots as a gap."""
    p, lo, hi = ([_f(v) for v in s] for s in (p, lo, hi))
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, p, ls=ls, marker=marker, ms=2.4, lw=lw, color=color, label=label)


def draw(kind, ax, c, b, g, nb, x, arms):
    S = g["layers"][LAYER]["summary"]["MEAN8"]
    ser, prof = b["series"], b["profile"]["rows"]
    if kind == "A":
        for t in TRAIT_ORDER:
            r = ser[t]["rows"]; y = [q["level"]["point"] for q in r]
            ax.plot(x, y, "-", marker="o", ms=1.6, lw=0.9, color=TRAIT_COL[t], label=t.replace("_", "-"))
            record(c, "A", t, arms, x, y)
        for n in CONTRASTS.get(c, {}):
            if n == "B1":
                continue
            r = ser[n]["rows"]
            p_, lo_, hi_ = ([q["level"][k] for q in r] for k in ("point", "lo", "hi"))
            band(ax, x, p_, lo_, hi_, C_BEH, "target contrast", lw=2.0)
            record(c, "A", n, arms, x, p_, lo_, hi_)
            ax.annotate(n, (x[-1], p_[-1]), xytext=(2, 0), textcoords="offset points", fontsize=6.5,
                        fontweight="bold", va="center", ha="left", clip_on=False)
        ax.axhline(0, color=MUTED, lw=0.5)
    elif kind == "B":
        pr = prof[1:]
        p_, lo_, hi_ = ([r["q"][k] for r in pr] for k in ("point", "lo", "hi"))
        band(ax, x[1:], p_, lo_, hi_, C_BEH, LAB["q"], lw=1.6); record(c, "B", "q", arms[1:], x[1:], p_, lo_, hi_)
        p_, lo_, hi_ = ([S[a_]["Q"][k] for a_ in arms[1:]] for k in ("point", "lo", "hi"))
        band(ax, x[1:], p_, lo_, hi_, C_Q, LAB["Q"], marker="s"); record(c, "B", "Q", arms[1:], x[1:], p_, lo_, hi_)
        fp, lo_, hi_ = g_fraction(g, nb)
        band(ax, x, fp, lo_, hi_, C_G, LAB["gt"], ls="--"); record(c, "B", "g_tilde", arms, x, fp, lo_, hi_)
        ax.axhline(1, color=MUTED, lw=0.5, ls=":"); ax.axhline(0, color=MUTED, lw=0.5, ls=":")
    elif kind == "C":
        p_, lo_, hi_ = ([S[a_]["S"][k] for a_ in arms] for k in ("point", "lo", "hi"))
        band(ax, x, p_, lo_, hi_, C_S, LAB["S"]); record(c, "C", "S", arms, x, p_, lo_, hi_)
        p_, lo_, hi_ = ([S[a_]["SS"][k] for a_ in arms[1:]] for k in ("point", "lo", "hi"))
        band(ax, x[1:], p_, lo_, hi_, C_SS, LAB["SS"], marker="s"); record(c, "C", "SS", arms[1:], x[1:], p_, lo_, hi_)
    elif kind == "D":
        p_, lo_, hi_ = ([S[a_]["g"][k] for a_ in arms] for k in ("point", "lo", "hi"))
        band(ax, x, p_, lo_, hi_, C_G, LAB["g"]); record(c, "D", "g", arms, x, p_, lo_, hi_)
        p_, lo_, hi_ = ([S[a_]["cosT"][k] for a_ in arms] for k in ("point", "lo", "hi"))
        band(ax, x, p_, lo_, hi_, C_COS, LAB["cosT"], marker="^"); record(c, "D", "cosT", arms, x, p_, lo_, hi_)


def style_x(ax, xmode, show_label):
    if xmode == "epoch":
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.2); ax.set_xlim(-0.05, 3.3)
        ax.set_xticks([0, 0.5, 1, 3]); ax.set_xticklabels(["0", "0.5", "1", "3"])
        ax.set_xticks([0.25, 0.75, 2], minor=True)
        if show_label:
            ax.set_xlabel("SFT epoch (0 = DPO-only)", color=MUTED)
    elif show_label:
        ax.set_xlabel("functional dose (L15)", color=MUTED)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="ABCD",
                    help="subset of panels, e.g. AB; a subset writes fig_trajectory_<panels> (epoch axis "
                         "only) and leaves the full figure and the CSVs untouched")
    ap.add_argument("--geometry-tag", default=None,
                    help="read v2/geometry_<c><tag>.json (e.g. _cmbase); default: historical geometry_<c>.json")
    ap.add_argument("--behaviour-tag", default="",
                    help="read v2/behaviour_<c><tag>.json (e.g. _10personas, the geometry's persona set)")
    ap.add_argument("--dose-tag", default="", help="read functional_dose_<c><tag>.json")
    ap.add_argument("--out-tag", default="", help="suffix for every output written")
    ap.add_argument("--formats", nargs="+", default=["png"], help="e.g. pdf png (pdf is vector)")
    ap.add_argument("--out-dir", default=str(A), help="where figures and their plotted-data CSV go")
    ap.add_argument("--no-dose-companion", action="store_true",
                    help="full figure only: skip the functional-dose-axis companion (dose is not monotone in "
                         "epoch, so its curves double back and its bands are not meaningful)")
    a = ap.parse_args()
    SRC.update(geometry_tag=a.geometry_tag, behaviour_tag=a.behaviour_tag, dose_tag=a.dose_tag)
    panels = a.panels.upper()
    full = panels == "ABCD"
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figstyle import use_style, FULL
    use_style()

    cons = [c for c in ORDER if all(p.exists() for p in in_paths(c).values())]
    data = {c: load(c) for c in cons}
    base_arm = {data[c][1].get("base_arm", "base") for c in cons}
    if a.behaviour_tag == "_10personas":
        for c in cons:
            assert len(data[c][0]["personas"]) == 10 and "null" not in data[c][0]["personas"], c
    rows_csv, table, written = [], [], []
    for xmode in (("epoch", "dose") if full and not a.no_dose_companion else ("epoch",)):
        PLOTTED.clear()
        if full:   # rows = constitutions, cols = panels
            # last row holds the four column legends (axes switched off)
            fig, axes = plt.subplots(len(cons) + 1, 4, figsize=(FULL, 1.4 * len(cons) + 1.5),
                                     sharex="col" if xmode == "epoch" else False, sharey="col", squeeze=False,
                                     gridspec_kw={"height_ratios": [1] * len(cons) + [0.78]})
            legend_axes = axes[-1]; axes = axes[:-1]
            for lax in legend_axes:
                lax.axis("off")
            grid = {(c, k): axes[ri, ki] for ri, c in enumerate(cons) for ki, k in enumerate(panels)}
        else:      # rows = panels, cols = constitutions
            fig, axes = plt.subplots(len(panels), len(cons), figsize=(FULL, 1.55 * len(panels) + 0.75),
                                     sharex=True, sharey="row", squeeze=False)
            grid = {(c, k): axes[ki, ci] for ci, c in enumerate(cons) for ki, k in enumerate(panels)}
        for c in cons:
            b, g, fd, nb = data[c]
            arms = [s["arm"] for s in g["states"]]
            ep = np.array([s["epoch"] for s in g["states"]])
            dose = np.array([fd[LAYER][x]["trait_vector_displacement"] for x in arms])
            x = ep if xmode == "epoch" else dose
            for k in panels:
                ax = grid[(c, k)]
                draw(k, ax, c, b, g, nb, x, arms)
                style_x(ax, xmode, False)
                if full:
                    ax.tick_params(labelbottom=(c == cons[-1]))
                    if c == cons[0]:
                        ax.set_title(TITLE_SHORT[k], fontsize=7)
                    if k == panels[0]:
                        ax.set_ylabel(f"{c}\noffset (log-odds)", fontsize=7)
                else:
                    if k == panels[0]:
                        ax.set_title(c, fontsize=8)
                    if c == cons[0]:
                        ax.set_ylabel({"A": "offset vs base\n(log-odds)", "B": "fraction of final\nSFT-induced change",
                                       "C": "squared share", "D": "magnitude / cosine"}[k], fontsize=7)
                        ax.text(-0.42, 1.04, k, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
            if xmode == "epoch" and full:
                S = g["layers"][LAYER]["summary"]["MEAN8"]; ser, prof = b["series"], b["profile"]["rows"]
                for a_, e_, d_ in zip(arms, ep, dose):
                    row = {"constitution": c, "arm": a_, "epoch": round(float(e_), 4), "dose_L15": round(float(d_), 4)}
                    for t in b["traits"]:
                        row[f"{t}_offset"] = round(next(q for q in ser[t]["rows"] if q["arm"] == a_)["level"]["point"], 4)
                    pr_ = next(r for r in prof if r["arm"] == a_)
                    row["profile_change_norm"] = round(pr_["change_norm"]["point"], 4)
                    row["profile_q"] = round(pr_["q"]["point"], 4) if "q" in pr_ else None
                    for k in ("g", "S", "SS", "cosT", "Q"):
                        v = S[a_][k]["point"]; row[f"MEAN8_{k}"] = None if v is None or v != v else round(v, 4)
                    rows_csv.append(row)
                tp = table_points(c)
                def pick(e, k):
                    v = S[tp[e]][k]
                    if v["point"] is None:
                        return "n/e"
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

        # shared legends: one per panel kind, built from proxies so they are identical everywhere
        leg = {"A": [Line2D([], [], color=TRAIT_COL[t], lw=1.0, marker="o", ms=2, label=t.replace("_", "-"))
                     for t in TRAIT_ORDER]
                    + [Line2D([], [], color=C_BEH, lw=2.0, label="target contrast (B2, LOV), 95% band")],
               "B": [Line2D([], [], color=C_BEH, lw=1.6, marker="o", ms=2.4, label=LAB["q"]),
                     Line2D([], [], color=C_Q, lw=1.4, marker="s", ms=2.4, label=LAB["Q"]),
                     Line2D([], [], color=C_G, lw=1.4, ls="--", marker="o", ms=2.4, label=LAB["gt"])],
               "C": [Line2D([], [], color=C_S, lw=1.4, marker="o", ms=2.4, label=LAB["S"]),
                     Line2D([], [], color=C_SS, lw=1.4, marker="s", ms=2.4, label=LAB["SS"])],
               "D": [Line2D([], [], color=C_G, lw=1.4, marker="o", ms=2.4, label=LAB["g"]),
                     Line2D([], [], color=C_COS, lw=1.4, marker="^", ms=2.4, label=LAB["cosT"])]}
        epoch_note = ("x: SFT epoch, 0 = completed DPO-only state; linear on [0, 1], logarithmic on [1, 3]"
                      if xmode == "epoch" else "x: measured functional dose at L15 (point estimates)")
        note = (f"L15 geometry vs {', '.join(sorted(base_arm))}; behaviour on "
                f"{'the 10 occupational personas (null excluded)' if a.behaviour_tag == '_10personas' else 'all personas in the behaviour file'}; "
                "shaded = 95% question-bootstrap interval (only the bold contrast in A, and every curve in "
                f"{', '.join(k for k in panels if k != 'A')}); one training seed. {epoch_note}.")
        if full:
            fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.5, w_pad=0.6)
            xl = "SFT epoch (0 = completed DPO-only state; minor ticks 0.25, 0.75, 2)" if xmode == "epoch" \
                else "measured functional dose (L15)"
            y_lab = legend_axes[0].get_position().y1 - 0.012
            fig.text(0.53, y_lab, xl, ha="center", va="top", fontsize=6.5, color=MUTED)
            for lax, k in zip(legend_axes, panels):
                h = leg[k] if k != "A" else leg[k][:-1]
                lg = lax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.08, 0.74), fontsize=5.4,
                                ncol=2 if k == "A" else 1, columnspacing=0.5, handlelength=1.3, handletextpad=0.35)
                if k == "A":
                    lax.add_artist(lg)
                    lax.legend(handles=leg[k][-1:], loc="upper left", bbox_to_anchor=(-0.08, 0.2), fontsize=5.4,
                               handlelength=1.3, handletextpad=0.35)
            fig.text(0.01, 0.003, note, fontsize=5.4, color=MUTED, ha="left", va="bottom", wrap=True)
        else:
            fig.tight_layout(rect=(0.0, 0.215, 1, 1), h_pad=0.5, w_pad=0.4)
            fig.text(0.53, 0.19, "SFT epoch (0 = completed DPO-only state; minor ticks 0.25, 0.75, 2)",
                     ha="center", fontsize=6.5, color=MUTED)
            fig.legend(handles=leg["A"], loc="lower left", bbox_to_anchor=(0.06, 0.085), ncol=5, fontsize=5.8,
                       columnspacing=0.8, handlelength=1.4)
            if "B" in panels:
                fig.legend(handles=leg["B"], loc="lower left", bbox_to_anchor=(0.06, 0.04), ncol=3, fontsize=5.8,
                           columnspacing=0.8, handlelength=1.6)
            fig.text(0.01, 0.003, note, fontsize=5.2, color=MUTED, ha="left", va="bottom", wrap=True)
        name = ("fig_trajectory" if full else f"fig_trajectory_{panels}") + ("" if xmode == "epoch" else "_dose") + a.out_tag
        for ext in a.formats:
            fig.savefig(out_dir / f"{name}.{ext}", dpi=300 if ext == "png" else None)
            written.append(out_dir / f"{name}.{ext}")
        plt.close(fig)
        with open(out_dir / f"{name}_data.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(PLOTTED[0])); w.writeheader(); w.writerows(PLOTTED)
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (out_dir / f"{name}_provenance.json").write_text(json.dumps({
            "figure": name, "command": "python3 scripts/common_trajectory/figures.py " + " ".join(sys.argv[1:]),
            "code_revision": commit, "inputs": {c: {k: str(v.relative_to(REPO)) for k, v in in_paths(c).items()} for c in cons},
            "baseline": sorted(base_arm), "layer": int(LAYER), "x": xmode,
            "persona_population": "10 occupational personas (null excluded)" if a.behaviour_tag == "_10personas" else "behaviour file personas",
            "estimands": {"A": "forced-choice offset a at base indifference, per trait; contrast = mean(targets) - mean(other traits)",
                          "B": "q (profile projection), Q (8-trait-mean incremental common-shift projection), g~ (normalised change in 8-trait-mean g since DPO)",
                          "C": "8-trait mean of per-trait S and S^S (estimability rule; non-estimable = gap)",
                          "D": "8-trait mean g and cos_T"},
            "intervals": "95% percentile, question bootstrap (behaviour 2000 draws; geometry 200 draws x 8 splits); evaluation only, one seed"},
            indent=1))
        written.append(out_dir / f"{name}_data.csv")
    for p in written:
        print("wrote", p)
    if not full:
        return

    with open(A / f"trajectory_points{a.out_tag}.csv", "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows_csv for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows_csv)
    with open(A / f"table_trajectory{a.out_tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r))); w.writeheader(); w.writerows(table)
    for t in table:
        print(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
