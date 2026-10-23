#!/usr/bin/env python3
"""Cross-character SFT pilot -- signed trait analysis. docs/spec_xchar_sft.md §6.

Reuses the estimator of scripts/caa_logits_analysis.py unchanged (fit_offset_slope,
balanced_mean, stack_trait): per state x trait, logodds_state = a + k * logodds_base,
polarity-split and averaged, against the ORIGINAL unadapted `base` for every state.

Uncertainty: one set of question-bootstrap draws per trait, shared by EVERY state, persona and
checkpoint, so every difference below (change from start, condition contrasts, the
difference-in-differences) is paired draw-by-draw. This is evaluation uncertainty only --
one training seed, so training-seed variability is NOT in these intervals.

Writes outputs/analysis/xchar_sft/{signed.json, summary.csv, acquisition_retention.png}.

    python scripts/xchar_analysis.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caa_logits_analysis import balanced_mean, fit_offset_slope, stack_trait  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/analysis/xchar_sft"
TAG, SUB = "llama-3.1-8b", "caa_logits_forced"
CONDS = ["xchar_lov_imp", "xchar_lov_lov", "xchar_base_imp", "xchar_base_lov"]
FRACS = ["s25", "s50", "s100"]
START = {"xchar_lov_imp": "xchar_lov_start", "xchar_lov_lov": "xchar_lov_start",
         "xchar_base_imp": "base", "xchar_base_lov": "base"}
CONTRASTS = {"B1": (["impulsivity"], None),
             "B2": (["impulsivity", "risk_taking"], None),
             "LOV": (["empathy", "warmth"], None)}       # loving's own pair vs the other six


def load(arms):
    cells = {}
    for arm in arms:
        d = REPO / "outputs" / f"{TAG}-{arm}" / SUB
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            cells.setdefault(arm, {}).setdefault(str(z["trait"]), {})[str(z["persona"])] = {
                "qid": z["qid"], "logodds": z["logodds"], "a_is_positive": z["a_is_positive"]}
    return cells


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"point": None, "ci_lo": float(lo), "ci_hi": float(hi)}


def summarise(point, boots):
    d = ci(boots); d["point"] = float(point); return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the saved signed.json (no recomputation)")
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--out-dir", default=None, help="default outputs/analysis/xchar_sft")
    a = ap.parse_args()
    if a.plot_only:
        plot(json.loads((OUT / "signed.json").read_text()), a.formats, a.out_dir)
        return

    states = ["xchar_lov_start", "loving"] + [f"{c}_{f}" for c in CONDS for f in FRACS]
    present = [s for s in states if (REPO / "outputs" / f"{TAG}-{s}" / SUB).is_dir()
               and len(list((REPO / "outputs" / f"{TAG}-{s}" / SUB).glob("*.npz"))) == 88]
    missing = sorted(set(states) - set(present))
    if missing:
        print("not yet measured (skipped):", " ".join(missing))
    cells = load(["base"] + present)
    traits = sorted(cells["base"])
    personas = sorted(cells["base"][traits[0]])
    rng = np.random.default_rng(a.seed)

    # pooled[state][trait] -> (point a, point k, point naive, raw level, boot a array)
    pooled, per_p = {}, {}
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]
        apos = cells["base"][t][personas[0]]["a_is_positive"]
        nq = len(qid); allq = np.arange(nq)
        draws = [rng.integers(0, nq, nq) for _ in range(a.n_boot)]
        base = stack_trait(cells, "base", t, personas, qid)
        pooled.setdefault("base", {})[t] = {"a": 0.0, "k": 1.0, "naive": 0.0,
                                            "raw": balanced_mean(base, apos, allq),
                                            "boot": np.zeros(a.n_boot)}
        for p_i, p in enumerate(personas):
            per_p.setdefault("base", {}).setdefault(t, {})[p] = {"a": 0.0, "boot": np.zeros(a.n_boot)}
        for s in present:
            arm = stack_trait(cells, s, t, personas, qid)
            off, k = fit_offset_slope(base, arm, apos, allq)
            boot = np.array([fit_offset_slope(base, arm, apos, q)[0] for q in draws])
            kb = np.array([fit_offset_slope(base, arm, apos, q)[1] for q in draws])
            pooled.setdefault(s, {})[t] = {
                "a": off, "k": k, "k_ci": [float(x) for x in np.percentile(kb, [2.5, 97.5])],
                "naive": balanced_mean(arm - base, apos, allq),
                "raw": balanced_mean(arm, apos, allq), "boot": boot}
            for p_i, p in enumerate(personas):
                b1, s1 = base[p_i:p_i + 1], arm[p_i:p_i + 1]
                po, _ = fit_offset_slope(b1, s1, apos, allq)
                pb = np.array([fit_offset_slope(b1, s1, apos, q)[0] for q in draws])
                per_p.setdefault(s, {}).setdefault(t, {})[p] = {"a": po, "boot": pb}

    def contrast(dct, s, name, key="a"):
        tgt = CONTRASTS[name][0]; rest = [t for t in traits if t not in tgt]
        pt = np.mean([dct[s][t][key] for t in tgt]) - np.mean([dct[s][t][key] for t in rest])
        bt = np.mean([dct[s][t]["boot"] for t in tgt], 0) - np.mean([dct[s][t]["boot"] for t in rest], 0)
        return pt, bt

    res = {"n_boot": a.n_boot, "traits": traits, "personas": personas, "states": present,
           "missing": missing, "reference": "base (original unadapted Llama-3.1-8B-Instruct)",
           "uncertainty": "paired question bootstrap, shared draws across all states/personas; "
                          "evaluation uncertainty only, single training seed",
           "levels": {}, "contrasts": {}, "change_from_start": {}, "comparisons": {},
           "per_persona": {}}
    for s in ["base"] + present:
        res["levels"][s] = {t: {"offset": summarise(pooled[s][t]["a"], pooled[s][t]["boot"]),
                                "slope_k": pooled[s][t]["k"], "slope_k_ci": pooled[s][t].get("k_ci"),
                                "naive_shift": pooled[s][t]["naive"], "raw_level": pooled[s][t]["raw"]}
                            for t in traits}
        res["contrasts"][s] = {n: summarise(*contrast(pooled, s, n)) for n in CONTRASTS}

    def diff(x, y, dct=pooled):
        """state x minus state y, per trait and per contrast, paired."""
        out = {t: summarise(dct[x][t]["a"] - dct[y][t]["a"], dct[x][t]["boot"] - dct[y][t]["boot"])
               for t in traits}
        for n in CONTRASTS:
            px, bx = contrast(dct, x, n); py, by = contrast(dct, y, n)
            out[n] = summarise(px - py, bx - by)
        return out, {t: (dct[x][t]["boot"] - dct[y][t]["boot"]) for t in traits}

    for c in CONDS:
        for f in FRACS:
            s = f"{c}_{f}"
            if s in present and START[c] in pooled:
                res["change_from_start"][s] = diff(s, START[c])[0]

    fin = {c: f"{c}_s100" for c in CONDS}
    if all(fin[c] in present for c in CONDS) and "xchar_lov_start" in present:
        c1, _ = diff(fin["xchar_lov_imp"], "xchar_lov_start")
        c2, b2 = diff(fin["xchar_lov_imp"], fin["xchar_lov_lov"])
        c3, b3 = diff(fin["xchar_base_imp"], fin["xchar_base_lov"])
        c4 = {}
        for t in traits:
            c4[t] = summarise(c2[t]["point"] - c3[t]["point"], b2[t] - b3[t])
        for n in CONTRASTS:
            p2, q2 = contrast(pooled, fin["xchar_lov_imp"], n); p2b, q2b = contrast(pooled, fin["xchar_lov_lov"], n)
            p3, q3 = contrast(pooled, fin["xchar_base_imp"], n); p3b, q3b = contrast(pooled, fin["xchar_base_lov"], n)
            c4[n] = summarise((p2 - p2b) - (p3 - p3b), (q2 - q2b) - (q3 - q3b))
        res["comparisons"] = {"1_lovimp_vs_lovstart": c1, "2_lovimp_vs_lovlov": c2,
                              "3_baseimp_vs_baselov": c3, "4_diff_2_minus_3": c4}

    for s in present:
        res["per_persona"][s] = {
            p: {t: summarise(per_p[s][t][p]["a"], per_p[s][t][p]["boot"]) for t in traits}
            for p in personas}
        if s.rsplit("_", 1)[0] in START:
            st = START[s.rsplit("_", 1)[0]]
            for p in personas:
                for t in traits:
                    res["per_persona"][s][p][t]["change_from_start"] = summarise(
                        per_p[s][t][p]["a"] - per_p[st][t][p]["a"],
                        per_p[s][t][p]["boot"] - per_p[st][t][p]["boot"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "signed.json").write_text(json.dumps(res, indent=1))
    write_csv(res, traits)
    plot(res)
    print_table(res, traits)


def write_csv(res, traits):
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "metric", "level", "level_lo", "level_hi",
                    "change_from_start", "chg_lo", "chg_hi", "slope_k"])
        for s, lv in res["levels"].items():
            ch = res["change_from_start"].get(s, {})
            for m in traits + list(CONTRASTS):
                L = lv[m]["offset"] if m in traits else res["contrasts"][s][m]
                C = ch.get(m)
                w.writerow([s, m, f"{L['point']:+.3f}", f"{L['ci_lo']:+.3f}", f"{L['ci_hi']:+.3f}",
                            "" if C is None else f"{C['point']:+.3f}",
                            "" if C is None else f"{C['ci_lo']:+.3f}",
                            "" if C is None else f"{C['ci_hi']:+.3f}",
                            f"{lv[m]['slope_k']:.3f}" if m in traits else ""])


def print_table(res, traits):
    show = ["impulsivity", "risk_taking", "empathy", "warmth", "B1", "B2", "LOV"]
    print(f"{'state':26s}" + "".join(f"{m[:11]:>12s}" for m in show))
    for s in res["levels"]:
        vals = [res["levels"][s][m]["offset"]["point"] if m in traits else res["contrasts"][s][m]["point"]
                for m in show]
        print(f"{s:26s}" + "".join(f"{v:+12.3f}" for v in vals))
    for name, comp in res["comparisons"].items():
        print(f"\n{name}")
        for m in traits + list(CONTRASTS):
            c = comp[m]
            print(f"  {m:14s} {c['point']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]")


def plot(res, formats=("png",), out_dir=None):
    """Acquisition (impulsivity, risk-taking) and retention (empathy, warmth) panels from the saved
    levels. Corpus = colour (orange impulsiveness, dark blue loving: different hue AND luminance, so
    grayscale keeps them apart); start = line style + marker (solid filled circle loving start,
    dashed open square base start). Bars are the stored 95 % question-bootstrap intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "workshop_iclr/scripts"))
    from figstyle import use_style, FULL
    use_style()
    out_dir = Path(out_dir) if out_dir else OUT
    INK, MUTED, GRID = "#111111", "#5c5c58", "#d9d9d5"
    HUE = {"imp": "#eb6834", "lov": "#1d4f91"}
    STYLE = {"lov": "-", "base": (0, (4, 1.6))}
    MARK = {"lov": "o", "base": "s"}
    LABEL = {"xchar_lov_imp": "loving start, impulsiveness corpus", "xchar_lov_lov": "loving start, loving corpus",
             "xchar_base_imp": "base start, impulsiveness corpus", "xchar_base_lov": "base start, loving corpus"}
    # Checkpoints are optimizer-step fractions (audit 2026-10-21 §5.2): impulsiveness corpus
    # steps 82/164/328, loving corpus 92/184/367 = 25.2-25.3 % / 50.2 % / 100 % of supervised tokens.
    x = {"start": 0, "s25": 25, "s50": 50, "s100": 100}
    DX = {"xchar_lov_imp": -2.4, "xchar_base_imp": -0.8, "xchar_lov_lov": 0.8, "xchar_base_lov": 2.4}
    panels = [("impulsivity", "acquisition"), ("risk_taking", "acquisition"),
              ("empathy", "retention"), ("warmth", "retention")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.0), sharey=True, sharex=True)
    rows = []
    for ax, (t, role) in zip(axes.flat, panels):
        for c in CONDS:
            start, corpus = c.split("_")[1], c.split("_")[2]
            pts = [("start", res["levels"][START[c]][t]["offset"])] if START[c] in res["levels"] else []
            pts += [(f, res["levels"][f"{c}_{f}"][t]["offset"]) for f in FRACS if f"{c}_{f}" in res["levels"]]
            if not pts:
                continue
            xs = [x[p[0]] + DX[c] for p in pts]          # small offsets so coincident points stay visible
            ys = [p[1]["point"] for p in pts]
            for (f, o), xv in zip(pts, xs):
                rows.append({"trait": t, "condition": c, "checkpoint": f, "x_pct_steps": x[f], "x_plotted": xv,
                             "offset": o["point"], "ci_lo": o["ci_lo"], "ci_hi": o["ci_hi"]})
            ax.vlines(xs, [p[1]["ci_lo"] for p in pts], [p[1]["ci_hi"] for p in pts], color=HUE[corpus], lw=0.9, alpha=0.7)
            ax.plot(xs, ys, ls=STYLE[start], color=HUE[corpus], lw=1.5, marker=MARK[start], ms=4.2,
                    mfc=HUE[corpus] if start == "lov" else "white", mec=HUE[corpus], mew=1.1, label=LABEL[c])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{t.replace('_', '-')} ({role})", fontsize=7.5)
        ax.set_xticks([0, 25, 50, 100]); ax.set_xticklabels(["start", "25%", "50%", "100%"])
        ax.set_xlim(-6, 106)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("offset at base-model\nindifference (log-odds)")
    fig.tight_layout(rect=(0, 0.17, 1, 1), h_pad=0.8, w_pad=0.8)
    fig.text(0.53, 0.145, "% of optimizer steps completed in one token-matched additional SFT pass",
             ha="center", fontsize=6.8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.06, 0.06), ncol=2, fontsize=6.3, handlelength=2.6,
               columnspacing=1.4)
    fig.text(0.01, 0.003, "Fitted forced-choice offset vs the unadapted base, 11 personas (10 occupational + null). Bars: "
             "95% question-bootstrap intervals (2,000 paired draws; evaluation only, one training seed). Steps: impulsiveness "
             "corpus 82/164/328, loving corpus 92/184/367 (25.2-25.3% / 50.2% / 100% of supervised tokens). Points offset "
             "horizontally for legibility.", fontsize=5.0, color=MUTED, ha="left", va="bottom", wrap=True)
    for ext in formats:
        # new name: the reported figure (token-budget labels) is kept as it was
        fig.savefig(out_dir / f"acquisition_retention_steps.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    with open(out_dir / "acquisition_retention_steps_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
