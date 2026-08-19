#!/usr/bin/env python3
"""Figures for the character-arm geometry analysis (section 10 of the plan).

Consumes the JSON written by scripts/geometry_analysis.py. Each figure is aimed at one
step of the decomposition

    observed OCT effect = shared component + coordinate change + dose + constitution-specific

and the ordering of the panels is the argument: what survives after each control is
removed. Figures that would imply more than the mathematics supports are deliberately
labelled as exploratory rather than dropped.

  B  full-space persona dispersion, arm/base ratio per trait, reference line at 1
  C  RDM heatmaps per trait per arm, plus arm-minus-base difference RDMs
  D  RDM preservation summary, raw and noise-ceiling corrected
  E  Procrustes error reduction: raw -> orthogonal -> +scale, cross-validated
  F  residual restructuring heatmap, 8 traits x 10 personas, after global alignment

Figure A (raw vs corrected cosine-to-null) is already produced by
scripts/plot_arm_comparison.py against the existing corrected residual-cosine analysis;
it is not duplicated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_COLOR = {"goodness": "#2166ac", "mathematical": "#b2182b", "impulsiveness": "#1a9850"}


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # suptitles here are two lines and collide with per-axes titles at default spacing
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    except Exception:
        pass
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def fig_b_dispersion(d: dict, out: Path) -> None:
    traits = d["traits"]
    arms = [a for a in d["arms"] if a != "base"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(traits))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        cells = [d["dispersion"][t][arm] for t in traits]
        r = [c.get("ratio", c["crossfit"] / max(d["dispersion"][t]["base"]["crossfit"], 1e-9))
             for c, t in zip(cells, traits)]
        pos = x + i * w - 0.4 + w / 2
        ax.bar(pos, r, w * 0.9, label=arm, color=ARM_COLOR.get(arm, "#888"), alpha=0.85)
        if all("ratio_ci" in c for c in cells):
            lo = np.array([c["ratio_ci"][0] for c in cells])
            hi = np.array([c["ratio_ci"][1] for c in cells])
            rr = np.array(r)
            ax.errorbar(pos, rr, yerr=[np.maximum(rr - lo, 0), np.maximum(hi - rr, 0)],
                        fmt="none", ecolor="k", elinewidth=1, capsize=2)
    ax.axhline(1.0, color="k", lw=1.2, ls="--", zorder=0)
    ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
    ax.set_xticks(x); ax.set_xticklabels(traits, rotation=30, ha="right")
    ax.set_ylabel("persona dispersion, arm / base")
    ax.set_title(f"B. Full-space persona dispersion at layer {d['layer']}\n"
                 "cross-fitted; >1 expansion, <1 contraction", fontsize=11)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=9)
    _save(fig, out, f"figB_dispersion_L{d['layer']}")


def fig_c_rdm(d: dict, out: Path, traits_show: list[str]) -> None:
    P = d["personas"]; n = len(P)
    arms = d["arms"]
    iu = np.triu_indices(n, k=1)

    def square(vec):
        m = np.zeros((n, n)); m[iu] = vec; return m + m.T

    for trait in traits_show:
        if trait not in d.get("rdm_raw", {}):
            continue
        mats = {a: square(np.array(d["rdm_raw"][trait][a])) for a in arms}
        vmax = max(m.max() for m in mats.values())
        fig, axes = plt.subplots(2, len(arms), figsize=(3.1 * len(arms), 6.4))
        for j, a in enumerate(arms):
            im = axes[0, j].imshow(mats[a], vmin=0, vmax=vmax, cmap="viridis")
            axes[0, j].set_title(a, fontsize=10)
            dif = mats[a] - mats["base"]
            lim = max(abs(dif).max(), 1e-9)
            im2 = axes[1, j].imshow(dif, vmin=-lim, vmax=lim, cmap="RdBu_r")
            for r in (0, 1):
                axes[r, j].set_xticks(range(n)); axes[r, j].set_yticks(range(n))
                axes[r, j].set_xticklabels(P, rotation=90, fontsize=5)
                axes[r, j].set_yticklabels(P if j == 0 else [], fontsize=5)
            if j == len(arms) - 1:
                fig.colorbar(im, ax=axes[0, j], fraction=0.046)
                fig.colorbar(im2, ax=axes[1, j], fraction=0.046)
        axes[0, 0].set_ylabel("absolute RDM", fontsize=9)
        axes[1, 0].set_ylabel("arm - base", fontsize=9)
        fig.suptitle(f"C. Persona RDMs, trait = {trait}, layer {d['layer']}\n"
                     "cross-fitted squared distances; a common additive vector cancels exactly",
                     fontsize=11)
        _save(fig, out, f"figC_rdm_{trait}_L{d['layer']}")


def fig_d_preservation(d: dict, out: Path) -> None:
    traits = d["traits"]; arms = [a for a in d["arms"] if a != "base"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(traits))
    for ax, key, title in (
            (axes[0], "spearman", "raw Spearman (attenuated by measurement noise)"),
            (axes[1], "spearman_corrected", "corrected for the per-arm noise ceiling")):
        for i, arm in enumerate(arms):
            v = [d["rdm"][t][arm][key] for t in traits]
            off = (i - (len(arms) - 1) / 2) * 0.22
            if key == "spearman":
                lo = np.array([d["rdm"][t][arm]["spearman_ci"][0] for t in traits])
                hi = np.array([d["rdm"][t][arm]["spearman_ci"][1] for t in traits])
                vv = np.array(v)
                # A point estimate can fall outside its own percentile interval for a
                # correlation under low reliability: replicate RDMs are noisier than the
                # point estimate, so their correlations attenuate further. Clip the bar to
                # zero length so matplotlib accepts it, and mark the cell rather than
                # silently drawing a misleadingly tidy interval.
                lo_err = np.maximum(vv - lo, 0.0)
                hi_err = np.maximum(hi - vv, 0.0)
                outside = (vv < lo) | (vv > hi)
                ax.errorbar(x + off, vv, yerr=[lo_err, hi_err],
                            fmt="o", ms=5, capsize=2, lw=1,
                            color=ARM_COLOR.get(arm, "#888"), label=arm)
                if outside.any():
                    ax.plot(x[outside] + off, vv[outside], "x", ms=9, mew=1.6,
                            color="k", zorder=5,
                            label="point outside CI" if i == 0 else None)
            else:
                ax.plot(x + off, v, "o", ms=6, color=ARM_COLOR.get(arm, "#888"), label=arm)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(traits, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("RDM preservation vs base")
    axes[0].legend(frameon=False)
    fig.suptitle(f"D. Persona-RDM preservation across 45 persona pairs, layer {d['layer']}",
                 fontsize=11)
    _save(fig, out, f"figD_rdm_preservation_L{d['layer']}")


def fig_e_procrustes(d: dict, out: Path) -> None:
    proc = d.get("procrustes", {})
    if not proc:
        return
    schemes = list(next(iter(proc.values())).keys())
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.4), squeeze=False)
    for si, scheme in enumerate(schemes):
        ax = axes[0, si]
        arms = list(proc)
        stages = ["test_raw", "test_proc", "test_proc_scaled"]
        labels = ["no transform", "+ orthogonal map", "+ global scale"]
        for i, arm in enumerate(arms):
            v = [proc[arm][scheme][s] for s in stages]
            ax.plot(range(len(stages)), v, "o-", lw=2, ms=7,
                    color=ARM_COLOR.get(arm, "#888"), label=arm)
            cov = proc[arm][scheme].get("basis_coverage")
            if cov is not None and i == 0:
                ax.text(0.02, 0.02, f"basis covers {cov:.0%} of held-out signal",
                        transform=ax.transAxes, fontsize=8, style="italic",
                        color="#b2182b" if cov < 0.5 else "#444")
        ax.set_xticks(range(len(stages))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("held-out error / ||X||" if si == 0 else "")
        ax.set_title(f"CV: {scheme}", fontsize=10)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        if si == 0:
            ax.legend(frameon=False)
    fig.suptitle(f"E. How much arm difference survives a single global coordinate change "
                 f"(layer {d['layer']})\ncross-validated; low basis coverage means the test "
                 "cannot see the signal, not that no transform exists", fontsize=11)
    _save(fig, out, f"figE_procrustes_L{d['layer']}")


def fig_f_residual(d: dict, out: Path) -> None:
    rm = d.get("residual_map", {})
    if not rm:
        return
    arms = list(rm)
    vmax = max(np.array(rm[a]["values"]).max() for a in arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(5.0 * len(arms), 4.6), squeeze=False)
    for j, a in enumerate(arms):
        v = np.array(rm[a]["values"])
        im = axes[0, j].imshow(v, vmin=0, vmax=vmax, cmap="magma", aspect="auto")
        axes[0, j].set_xticks(range(len(rm[a]["personas"])))
        axes[0, j].set_xticklabels(rm[a]["personas"], rotation=90, fontsize=7)
        axes[0, j].set_yticks(range(len(rm[a]["traits"])))
        axes[0, j].set_yticklabels(rm[a]["traits"] if j == 0 else [], fontsize=7)
        axes[0, j].set_title(a, fontsize=10)
        if j == len(arms) - 1:
            fig.colorbar(im, ax=axes[0, j], fraction=0.046, label="||residual|| / ||target||")
    fig.suptitle(f"F. Residual restructuring after global alignment, layer {d['layer']}\n"
                 "point estimates; read the pattern, not small colour differences",
                 fontsize=11)
    _save(fig, out, f"figF_residual_L{d['layer']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--rdm-traits", nargs="+",
                    default=["impulsivity", "honesty", "warmth", "risk_taking"])
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    print(f"figures -> {out}")
    fig_b_dispersion(d, out)
    fig_c_rdm(d, out, args.rdm_traits)
    fig_d_preservation(d, out)
    fig_e_procrustes(d, out)
    fig_f_residual(d, out)


if __name__ == "__main__":
    main()
