#!/usr/bin/env python3
"""Signed behavioural preference shift from the CAA A/B logits.

Reads the per-cell .npz written by pipeline/2d_caa_logits.py and asks the question the
geometry cannot answer: did the arm move the model TOWARD the trait-positive answer?

THE OBVIOUS ESTIMATOR IS INVALID, and in exactly the way figA5's geometric one was. The
naive quantity is d = E[logodds_arm - logodds_base]. On `goodness` (forced prompt, 11
personas x ~500 items) it reads: honesty -2.93, empathy -2.24, warmth -2.36, impulsivity
+1.87. Read literally that says the goodness constitution made the model less honest, less
empathetic, less warm and more impulsive. It did not. Regressing the arm's log-odds on the
base's, per trait:

    trait          base mean   naive d    slope k       r
    assertiveness      +4.80     -2.61      0.270    -0.97
    honesty            +4.67     -2.93      0.255    -0.97
    impulsivity        -2.74     +1.87      0.270    -0.98
    ... same k on all eight traits, r = -0.97 to -0.98 throughout

The arm multiplies every item's log-odds by ~0.26 and keeps almost nothing else. The naive
d is then just -0.74 x (where base already stood), which is why it tracks the base level at
r = -0.989 ACROSS traits: honesty had the highest base preference so it shows the largest
"loss", impulsivity the lowest so it shows a "gain". No preference changed; the whole
distribution was compressed toward indifference.

THE ESTIMATOR USED HERE separates the two effects by fitting, per arm x trait,

    logodds_arm = a + k * logodds_base                       (OLS over items)

  k  RETENTION. How much of the base model's preference structure survives. k=1 is
     untouched, k=0 is indifference. This is the behavioural analogue of the geometry's
     contraction, and like it, it is expected to track dose rather than content.
  a  OFFSET. Where the arm pushes an item the base model was indifferent about (B=0).
     This is the signed, compression-free preference shift: positive = toward the
     trait-positive answer. It is the quantity the naive d was supposed to be.

Both are reported. The naive d is kept beside them because it is what a reader would
compute, and the gap between them is the point.

LETTER BIAS IS CANCELLED BY CONSTRUCTION. The base model prefers the letter A regardless of
content (mean logit_A - logit_B = +1.06 on the forced prompt), and the arms change that
bias. An additive letter bias enters items where A is the positive option with one sign and
the rest with the other, so the fit is run separately within each polarity group and the
two intercepts averaged: the bias terms cancel exactly, whatever the polarity imbalance.
The slope is unaffected and is averaged the same way.

NO ERRORS-IN-VARIABLES PROBLEM. Regressing a difference on its own baseline usually invites
regression to the mean, but both log-odds here are deterministic single forward passes, not
noisy estimates of a latent value, so the predictor is measured exactly. The evidence that
this is not the artefact anyway: r = -0.97, far past the -0.71 that pure noise would
produce, and k is stable to +-0.01 across eight independent traits.

UNCERTAINTY is a paired bootstrap over QUESTIONS -- the same resampled question indices are
applied to every arm and persona, so the base subtraction stays paired and only the
question sample is treated as random. This conditions on these personas and these traits,
the same convention the geometry JSONs use. Personas are averaged, not resampled.

TWO PROMPT FORMS are reported side by side and never pooled -- see the header of
scripts/run_caa_logits.sh for why both exist and how far apart they are.

    python scripts/caa_logits_analysis.py
    python scripts/caa_logits_analysis.py --n-boot 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
MODEL_TAG = "llama-3.1-8b"

VARIANTS = {"forced": "caa_logits_forced", "default": "caa_logits"}
BASE_ARM = "base"

# The geometry's one content-linked result is that `impulsiveness` moves impulsivity and
# risk_taking ~1.8x as far as the other six traits. These are those two traits, fixed here
# before looking at any logit, so the behavioural contrast tests the same hypothesis rather
# than a new one chosen to fit.
IMPULSIVENESS_TARGETS = ("impulsivity", "risk_taking")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUTPUTS / "analysis" / "caa_logits.json"))
    return p.parse_args()


def load_variant(subdir: str):
    """-> cells[arm][trait][persona] = dict of arrays, plus the arm and trait listings."""
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for arm_dir in sorted(OUTPUTS.glob(f"{MODEL_TAG}-*")):
        arm = arm_dir.name[len(MODEL_TAG) + 1:]
        d = arm_dir / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            persona, trait = str(z["persona"]), str(z["trait"])
            cells.setdefault(arm, {}).setdefault(trait, {})[persona] = {
                "qid": z["qid"], "logodds": z["logodds"], "p_ab": z["p_ab"],
                "a_is_positive": z["a_is_positive"],
                "logit_a": z["logit_a"], "logit_b": z["logit_b"],
            }
    return cells


def stack_trait(cells, arm, trait, personas, qid_ref):
    """(n_personas, n_questions) log-odds for one arm x trait, in a fixed question order."""
    rows = []
    for p in personas:
        c = cells[arm][trait][p]
        if not np.array_equal(c["qid"], qid_ref):
            raise ValueError(f"question order differs for {arm}/{trait}/{p}")
        rows.append(c["logodds"])
    return np.vstack(rows)


def balanced_mean(d_pq: np.ndarray, a_pos: np.ndarray, qsel: np.ndarray) -> float:
    """Naive delta: mean over personas and questions, weighting polarity groups equally.

    Retained because it is what a reader would compute, and because the distance between it
    and the offset below is the methodological result. It is NOT a valid preference measure
    on its own -- see the module docstring.
    """
    d = d_pq[:, qsel]
    a = a_pos[qsel]
    if a.all() or not a.any():
        return float(d.mean())
    return float(0.5 * d[:, a].mean() + 0.5 * d[:, ~a].mean())


def fit_offset_slope(base_pq: np.ndarray, arm_pq: np.ndarray, a_pos: np.ndarray,
                     qsel: np.ndarray) -> tuple[float, float]:
    """OLS of arm log-odds on base log-odds -> (offset a, retention k).

    Fitted separately within each polarity group and averaged, so that any additive
    letter bias -- which enters the two groups with opposite sign -- cancels in the
    intercept. Items are persona x question pairs; personas are pooled, not averaged,
    so the fit uses the full spread of base preference.
    """
    outs = []
    for grp in (a_pos[qsel], ~a_pos[qsel]):
        if grp.sum() < 3:
            continue
        b = base_pq[:, qsel][:, grp].ravel()
        g = arm_pq[:, qsel][:, grp].ravel()
        vb = b.var()
        if vb < 1e-12:
            continue
        # Closed-form OLS. np.polyfit would do the same thing an order of magnitude
        # slower, and this is called twice per bootstrap draw per arm per trait.
        k = float(((b - b.mean()) * (g - g.mean())).mean() / vb)
        a = float(g.mean() - k * b.mean())
        outs.append((a, k))
    if not outs:
        return float("nan"), float("nan")
    return (float(np.mean([o[0] for o in outs])), float(np.mean([o[1] for o in outs])))


def analyse(cells, n_boot: int, seed: int):
    arms = sorted(cells)
    if BASE_ARM not in arms:
        raise SystemExit(f"no '{BASE_ARM}' arm found; every delta is against it")
    traits = sorted(set(cells[BASE_ARM]))
    other = [a for a in arms if a != BASE_ARM]

    rng = np.random.default_rng(seed)
    out = {"arms": other, "traits": traits, "n_boot": n_boot,
           "estimator": "polarity-balanced mean of (arm - base) log-odds; "
                        "paired question bootstrap; personas averaged, not resampled",
           "estimator_note": "primary = OLS offset a in logodds_arm = a + k*logodds_base, "
                             "polarity-split and averaged; delta is the naive and "
                             "compression-confounded quantity, kept for contrast",
           "offset": {}, "retention": {}, "delta": {}, "delta_naive": {},
           "base_level": {}, "diagnostics": {}}

    for trait in traits:
        # Personas are intersected with each arm SEPARATELY, inside the arm loop below.
        # Intersecting across all arms up front would let a single still-extracting arm
        # silently shrink -- or, for a trait it has not reached at all, delete -- every
        # other arm's estimate, which is exactly wrong for a resumable run one wants to
        # read mid-flight.
        personas = sorted(set(cells[BASE_ARM][trait]))
        if not personas:
            continue
        qid_ref = cells[BASE_ARM][trait][personas[0]]["qid"]
        a_pos = cells[BASE_ARM][trait][personas[0]]["a_is_positive"]
        nq = len(qid_ref)

        base_all = stack_trait(cells, BASE_ARM, trait, personas, qid_ref)
        draws = [rng.integers(0, nq, nq) for _ in range(n_boot)]
        all_q = np.arange(nq)

        for a in other:
            pa = [p for p in personas if p in cells.get(a, {}).get(trait, {})]
            if not pa:
                continue
            base = stack_trait(cells, BASE_ARM, trait, pa, qid_ref)
            arm_lo = stack_trait(cells, a, trait, pa, qid_ref)
            d_pq = arm_lo - base

            off, slope = fit_offset_slope(base, arm_lo, a_pos, all_q)
            b_off, b_slope, b_naive = [], [], []
            for q in draws:
                o, k = fit_offset_slope(base, arm_lo, a_pos, q)
                b_off.append(o); b_slope.append(k)
                b_naive.append(balanced_mean(d_pq, a_pos, q))
            b_off = np.array(b_off); b_slope = np.array(b_slope)
            b_naive = np.array(b_naive)

            olo, ohi = np.percentile(b_off, [2.5, 97.5])
            klo, khi = np.percentile(b_slope, [2.5, 97.5])
            nlo, nhi = np.percentile(b_naive, [2.5, 97.5])

            out["offset"].setdefault(a, {})[trait] = {
                "point": off, "ci_lo": float(olo), "ci_hi": float(ohi),
                "boot_sd": float(b_off.std())}
            out["retention"].setdefault(a, {})[trait] = {
                "point": slope, "ci_lo": float(klo), "ci_hi": float(khi)}
            out["delta"].setdefault(a, {})[trait] = {
                "point": balanced_mean(d_pq, a_pos, all_q),
                "ci_lo": float(nlo), "ci_hi": float(nhi),
                "n_personas": len(pa),
                "per_persona": {p: float(d_pq[i].mean()) for i, p in enumerate(pa)},
            }
            out["delta_naive"].setdefault(a, {})[trait] = float(d_pq.mean())
            out["base_level"].setdefault(trait, float(base_all.mean()))

        # Per-arm diagnostics on this trait: how much mass sits on the letters at all, and
        # how strong the raw letter bias is. Both are properties of the arm, not of a delta.
        for a in arms:
            # An arm still extracting may not have this trait, or not this persona set;
            # diagnostics are per-arm and must not take down a partial-run report.
            if personas[0] not in cells.get(a, {}).get(trait, {}):
                continue
            c0 = cells[a][trait][personas[0]]
            have = [p for p in personas if p in cells[a][trait]]
            bias = float(np.mean([np.mean(cells[a][trait][p]["logit_a"]
                                          - cells[a][trait][p]["logit_b"])
                                  for p in have]))
            mass = float(np.mean([cells[a][trait][p]["p_ab"].mean() for p in have]))
            out["diagnostics"].setdefault(a, {})[trait] = {
                "p_ab_mean": mass, "letter_bias_A_minus_B": bias,
                "frac_a_positive": float(c0["a_is_positive"].mean()),
            }
        out.setdefault("personas", {})[trait] = personas

    # The pre-specified contrast: does an arm move its OWN content traits more than the rest?
    tgt = [t for t in IMPULSIVENESS_TARGETS if t in traits]
    rest = [t for t in traits if t not in tgt]
    if tgt and rest:
        out["selectivity"] = {"targets": tgt, "others": rest, "by_arm": {}}
        for a in other:
            if not all(t in out["offset"].get(a, {}) for t in traits):
                continue
            # On the OFFSET, not the naive delta: the naive delta's trait profile is a
            # near-perfect mirror of where base already stood (r = -0.99 across traits),
            # so a contrast built on it would measure base preference levels, not the arm.
            mt = float(np.mean([out["offset"][a][t]["point"] for t in tgt]))
            mo = float(np.mean([out["offset"][a][t]["point"] for t in rest]))
            mt_n = float(np.mean([out["delta"][a][t]["point"] for t in tgt]))
            mo_n = float(np.mean([out["delta"][a][t]["point"] for t in rest]))
            out["selectivity"]["by_arm"][a] = {
                "mean_target": mt, "mean_other": mo, "contrast": mt - mo,
                "contrast_naive": mt_n - mo_n}
    return out


def render(res_by_variant) -> str:
    L = []
    for variant, res in res_by_variant.items():
        if res is None:
            L.append(f"=== {variant}: no cells found ===\n")
            continue
        traits, arms = res["traits"], res["arms"]
        L.append("=" * 96)
        L.append(f"=== {variant} prompt ===")
        L.append("=" * 96)

        def table(key, title, note):
            L.append(f"\n{title}\n{note}")
            L.append(" " * 17 + "".join(f"{t[:13]:>14s}" for t in traits))
            for a in arms:
                row = []
                for t in traits:
                    e = res[key].get(a, {}).get(t)
                    if e is None:
                        row.append(f"{'--':>14s}"); continue
                    star = "*" if (e["ci_lo"] > 0) or (e["ci_hi"] < 0) else " "
                    row.append(f"{e['point']:>+13.3f}{star}")
                L.append(f"{a:<17s}" + "".join(row))

        L.append(f"\nbase log-odds level  " +
                 "".join(f"{res['base_level'].get(t, float('nan')):>+14.2f}" for t in traits))

        table("offset", "PRIMARY -- offset a (compression-free signed preference shift)",
              "positive = arm pushes toward the trait-positive answer at base indifference."
              "  * = 95% CI excludes 0")
        table("retention", "retention k (1 = base preference kept, 0 = pushed to indifference)",
              "a property of the arm, expected to track dose rather than content."
              "  * = CI excludes 0")
        table("delta", "NAIVE delta (INVALID as a preference measure -- shown for contrast)",
              "compare its profile to the base level row above; they mirror each other.")

        if "selectivity" in res and res["selectivity"]["by_arm"]:
            s_ = res["selectivity"]
            L.append(f"\nPre-specified contrast on the OFFSET: mean over {s_['targets']} "
                     f"minus mean over the other {len(s_['others'])}.")
            L.append(f"{'arm':<17s}{'target':>10s}{'other':>10s}{'contrast':>11s}"
                     f"{'(naive)':>11s}")
            for a, v in sorted(s_["by_arm"].items(), key=lambda kv: -kv[1]["contrast"]):
                L.append(f"{a:<17s}{v['mean_target']:>+10.3f}{v['mean_other']:>+10.3f}"
                         f"{v['contrast']:>+11.3f}{v['contrast_naive']:>+11.3f}")

        L.append("\nDiagnostics (mean over personas and traits):")
        L.append(f"{'arm':<17s}{'P(A)+P(B)':>12s}{'letter bias A-B':>18s}{'mean k':>10s}")
        for a in [BASE_ARM] + arms:
            d = res["diagnostics"].get(a)
            if not d:
                continue
            mass = np.mean([v["p_ab_mean"] for v in d.values()])
            bias = np.mean([v["letter_bias_A_minus_B"] for v in d.values()])
            ks = [e["point"] for e in res["retention"].get(a, {}).values()]
            kstr = f"{np.mean(ks):>10.3f}" if ks else f"{'--':>10s}"
            L.append(f"{a:<17s}{mass:>12.4f}{bias:>+18.3f}{kstr}")
        L.append("")
    return "\n".join(L)


def main():
    args = parse_args()
    res_by_variant = {}
    for variant, subdir in VARIANTS.items():
        cells = load_variant(subdir)
        if BASE_ARM not in cells:
            res_by_variant[variant] = None
            continue
        res_by_variant[variant] = analyse(cells, args.n_boot, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res_by_variant, indent=2))
    txt = render(res_by_variant)
    out.with_suffix(".txt").write_text(txt)
    print(txt)
    print(f"wrote {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
