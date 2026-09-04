#!/usr/bin/env python3
"""Is the adapter-induced shift ALONG the trait axis, and in which direction?

Every constitution x trait quantity in section 3.2 is a magnitude. ||dG_{c,t}|| says the
`impulsiveness` constitution moves the `impulsivity` and `risk_taking` representations
about 1.8x as far as it moves the other six, which is the one result in that document that
dose cannot manufacture. It cannot say whether those representations moved toward MORE
impulsivity or merely got reorganised harder, and those are different claims about what
character training did.

The signed quantity this computes. With V_{c,t,p} the CAA trait vector, V_base_{t,p} the
base model's, and the semantic personas p:

    dG_{c,t}  = mean_p (V_{c,t,p} - V_base_{t,p})       the persona-common shift (3.2)
    Vbar_{t}  = mean_p V_base_{t,p}                     the base trait axis
    proj_{c,t} = <dG_{c,t}, Vbar_t / ||Vbar_t||> / mean_p||V_base_{t,p}||

proj is signed, and is a COMPONENT of the table-1 magnitude in the same units: by
Cauchy-Schwarz |proj_{c,t}| <= ||dG_{c,t}|| / mean_p||V_base_{t,p}||, with the orthogonal
remainder reported beside it. The axis is not fitted to anything -- it is the base model's
own trait direction, fixed before any constitution is seen, so this is a projection onto a
pre-specified axis rather than a direction discovered in the data.

WHY THE SIGN MEANS WHAT IT SAYS. The CAA contrast is mean(pos) - mean(neg) where `pos` is
the answer expressing MORE of the trait (2c_caa_activations.py: "A" if a_is_positive).
So +Vbar_t is the direction of more-of-the-trait, and proj > 0 says the constitution
displaced the representation toward the trait's positive pole. This is a statement about
the model's representation at the answer token, NOT a measured behavioural change; nothing
here generates a single completion. See NOTES at the bottom.

TWO ESTIMATES, because they answer slightly different questions:

  proj_common    <dG, Vbar/||Vbar||>, the common shift against the persona-mean axis.
                 Directly comparable to table 1 of 3.2, and it is that table's entry
                 resolved into sign.
  proj_percell   mean_p <dV_{c,t,p}, V_base_{t,p}/||V_base_{t,p}||>, each persona's change
                 against ITS OWN trait axis, then averaged. Does not assume the ten
                 personas share an axis. If the two agree the choice of axis is not doing
                 the work.

CROSS-FITTING IS NOT OPTIONAL, and the bias here has a sign that would have flattered the
result. dG contains -V_base and the axis is built from V_base, so the naive inner product
carries a -E||eps_base||^2 term: shared base noise pushes every projection NEGATIVE, and
would manufacture "the constitution reduced the trait" out of estimation error alone. Half
A supplies the axis, half B the shift, symmetrised over the two assignments, so the noise
terms are independent and cancel in expectation. Naive values are printed alongside; the
gap is the artefact, and on this data it is large (see the run's table 1n).

Denominators follow common_shift.py exactly -- mean of per-persona norms, cross-fitted as
mean_p sqrt(<V_p^A, V_p^B>) -- so proj and ||dG||/base are in the same units and the
orthogonal decomposition closes.

Usage:
    python scripts/signed_trait_shift.py --layers 15 20 --bootstrap 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from persona_steering.config import OUTPUTS_DIR
from geometry_lib import prepare_diff, batch_vectors, half_splits, bootstrap_half_splits

TRAINED = ["goodness", "mathematical", "impulsiveness", "misalignment"]
UNTRAINED = ["random_iid_s16", "random_perm_s16", "random_spec_s19"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--cache-layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--arms", nargs="+", default=TRAINED + UNTRAINED)
    p.add_argument("--half-splits", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--boot-splits", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "analysis" / "signed_trait_shift.json"))
    p.add_argument("--self-test", action="store_true",
                   help="recover a KNOWN signed shift from synthetic data and exit")
    return p.parse_args()


def self_test() -> None:
    """Recover a known signed shift, and show what the naive estimator does instead.

    The bias this checks is not academic: at the noise level here the naive projection
    comes out NEGATIVE when the true shift is positive, i.e. it would report that the
    constitution moved the representation away from the trait when it moved it toward
    the trait. That is the whole reason the estimator cross-fits.
    """
    rng = np.random.default_rng(7)
    P, H, NQ, NOISE = 10, 128, 400, 3.0

    mu = rng.normal(size=H); mu /= np.linalg.norm(mu)
    Vb = np.stack([mu + 0.35 * rng.normal(size=H) for _ in range(P)])
    mhat = Vb.mean(0) / np.linalg.norm(Vb.mean(0))
    bnorm = np.linalg.norm(Vb, axis=1).mean()

    alpha = 0.30
    orth = rng.normal(size=H)
    orth -= (orth @ mhat) * mhat
    orth /= np.linalg.norm(orth)
    g = alpha * mhat + 0.90 * orth
    truth = float(g @ mhat) / bnorm

    def obs(cells):
        return (cells[:, None, :] + NOISE * rng.normal(size=(P, NQ, H)),
                NOISE * rng.normal(size=(P, NQ, H)))

    bp, bn = obs(Vb)
    ap, an = obs(Vb + g)
    preps = {"base": prepare_diff(bp, bn),
             "arms": {"arm": prepare_diff(ap - bp, an - bn)}, "nq": NQ, "P": P}

    print(f"  truth  proj = {truth:+.4f}   ||dG||/base = "
          f"{np.linalg.norm(g) / bnorm:.4f}   orth = {0.90 / bnorm:.4f}")
    for ns in (4, 20, 60):
        r = stats_for_trait(preps, half_splits(NQ, ns, rng), ["arm"])["per_arm"]["arm"]
        n = stats_for_trait(preps, half_splits(NQ, ns, rng), ["arm"])["naive"]["arm"]
        ok = "OK " if abs(r["proj_common"] - truth) < 0.02 else "BAD"
        print(f"  {ok} splits={ns:3d}  cross-fitted {r['proj_common']:+.4f}   "
              f"naive {n['proj_common']:+.4f}   ||dG||/base {r['g_over_base']:.4f}   "
              f"orth {r['orth_common']:.4f}")


def load_arm(cache_dir: Path, arm: str, cache_layers, layer: int) -> dict:
    f = cache_dir / f"{arm}_L{'_'.join(str(l) for l in cache_layers)}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- run scripts/build_question_cache.py first")
    z = np.load(f, allow_pickle=False)
    li = list(z["layers"]).index(layer)
    return {"acts": z["acts"][:, :, :, :, li, :],
            "traits": [str(t) for t in z["traits"]],
            "personas": [str(p) for p in z["personas"]],
            "semantic": [str(p) for p in z["semantic_personas"]],
            "nq": z["n_questions_per_trait"]}


def cell(d: dict, ti: int, pidx: list[int]):
    nq = int(d["nq"][ti])
    a = d["acts"][ti][pidx][:, :, :nq, :]
    return a[:, 0].astype(np.float32), a[:, 1].astype(np.float32)


def make_preps(base_pos, base_neg, arm_cells: dict) -> dict:
    """Per-question contrasts laid out once. See common_shift.make_preps for why."""
    return {"base": prepare_diff(base_pos, base_neg),
            "arms": {a: prepare_diff(p - base_pos, n - base_neg)
                     for a, (p, n) in arm_cells.items()},
            "nq": base_pos.shape[1], "P": base_pos.shape[0]}


def stats_for_trait(preps: dict, splits, arms: list[str], naive: bool = True) -> dict:
    """Cross-fitted signed projections for one trait, all arms on the SAME splits."""
    prep_base, prep_d, P = preps["base"], preps["arms"], preps["P"]

    P_ = P
    acc = {a: {"num_common": 0.0, "g2": 0.0,
               "num_percell": np.zeros(P_)} for a in arms}
    acc_axis = {"mbar2": 0.0, "pnorm": np.zeros(P_)}
    # the constitution's shift for THIS trait, kept for the across-trait diagnostic
    gvec = {a: np.zeros((2, prep_base[2]), dtype=np.float64) for a in arms}

    for A, B in splits:
        Vb = batch_vectors(prep_base, [A, B])              # (2, P, H)
        mA, mB = Vb[0].mean(0), Vb[1].mean(0)              # persona-mean base axis
        acc_axis["mbar2"] += float(mA @ mB)
        # per-persona cross-fitted norms, ACCUMULATED and divided once at the end.
        # Dividing inside the loop is what a small half-split count breaks: a single
        # unlucky split can send <V_p^A, V_p^B> negative for a low-norm cell (deference
        # runs 0.59-1.37 against ~1.2 elsewhere), the clamp then divides by 1e-12, and one
        # replicate lands at 1e9. Ratio-of-averages instead of average-of-ratios.
        acc_axis["pnorm"] += np.sqrt(np.maximum((Vb[0] * Vb[1]).sum(-1), 0))

        for a in arms:
            dA = batch_vectors(prep_d[a], [A])[0]          # (P, H)
            dB = batch_vectors(prep_d[a], [B])[0]
            gA, gB = dA.mean(0), dB.mean(0)
            acc[a]["g2"] += float(gA @ gB)
            # symmetrised: axis from one half, shift from the other
            acc[a]["num_common"] += 0.5 * (float(gA @ mB) + float(gB @ mA))
            # own-axis variant: persona p's change against persona p's base vector
            acc[a]["num_percell"] += 0.5 * ((dA * Vb[1]).sum(-1) + (dB * Vb[0]).sum(-1))
            gvec[a][0] += gA
            gvec[a][1] += gB

    n = len(splits)
    mbar = float(np.sqrt(max(acc_axis["mbar2"] / n, 1e-12)))
    pnorm = acc_axis["pnorm"] / n                          # (P,)
    bnorm = float(pnorm.mean())

    out = {"axis": {"mbar_norm": mbar, "base_norm": bnorm,
                    "axis_concentration": mbar / max(bnorm, 1e-12)},
           "per_arm": {}, "gvec": {}}
    for a in arms:
        g_over_base = float(np.sqrt(max(acc[a]["g2"] / n, 0.0))) / max(bnorm, 1e-12)
        proj = (acc[a]["num_common"] / n) / (mbar * max(bnorm, 1e-12))
        proj_pc = float(((acc[a]["num_percell"] / n) / pnorm).mean()) / max(bnorm, 1e-12)
        out["gvec"][a] = (gvec[a] / n)
        out["per_arm"][a] = {
            "proj_common": float(proj),
            "proj_percell": float(proj_pc),
            "g_over_base": g_over_base,
            # the part of the common shift that is NOT along the trait axis
            "orth_common": float(np.sqrt(max(g_over_base ** 2 - proj ** 2, 0.0))),
            # cos(dG, base trait axis) -- scale-free direction agreement
            "cos_axis": float(proj / max(g_over_base, 1e-12)),
        }

    if naive:
        full = [np.arange(preps["nq"])]
        Vb = batch_vectors(prep_base, full)[0]
        m = Vb.mean(0)
        bn = float(np.linalg.norm(Vb, axis=-1).mean())
        mn = float(np.linalg.norm(m))
        out["naive"] = {}
        for a in arms:
            d = batch_vectors(prep_d[a], full)[0]
            g = d.mean(0)
            out["naive"][a] = {"proj_common": float((g @ m) / (mn * bn))}
    return out


def fmt_table(title: str, note: str, rows, cols, get) -> str:
    w = max(14, max(len(c) for c in cols) + 2)
    lines = [title, note, "", f"{'trait':16s}" + "".join(f"{c[:13]:>{w}s}" for c in cols)]
    for r in rows:
        lines.append(f"{r:16s}" + "".join(f"{get(r, c):>{w}.3f}" for c in cols))
    lines.append(f"{'MEAN':16s}" + "".join(
        f"{np.mean([get(r, c) for r in rows]):>{w}.3f}" for c in cols))
    return "\n".join(lines)


def main() -> None:
    a = parse_args()
    if a.self_test:
        self_test()
        return
    rng = np.random.default_rng(a.seed)
    cache = Path(a.cache_dir)
    results, text = {}, []

    for layer in a.layers:
        base = load_arm(cache, "base", a.cache_layers, layer)
        arms_d = {arm: load_arm(cache, arm, a.cache_layers, layer) for arm in a.arms}
        traits = base["traits"]
        pidx = [base["personas"].index(p) for p in base["semantic"]]
        per_trait = {}
        gstore: dict[str, dict[str, np.ndarray]] = {}   # arm -> trait -> (2, H)

        for ti, t in enumerate(traits):
            nq = int(base["nq"][ti])
            bp, bn = cell(base, ti, pidx)
            preps = make_preps(bp, bn, {arm: cell(d, ti, pidx)
                                        for arm, d in arms_d.items()})
            splits = half_splits(nq, a.half_splits, rng)
            per_trait[t] = stats_for_trait(preps, splits, a.arms)
            for arm, g in per_trait[t].pop("gvec").items():
                gstore.setdefault(arm, {})[t] = g

            if a.bootstrap:
                keys = ("proj_common", "proj_percell", "cos_axis", "orth_common")
                reps = {k: [] for k in keys}
                for _ in range(a.bootstrap):
                    bs = bootstrap_half_splits(nq, a.boot_splits, rng)
                    if not bs:
                        continue
                    r = stats_for_trait(preps, bs, a.arms, naive=False)
                    for k in keys:
                        reps[k].append([r["per_arm"][x][k] for x in a.arms])
                for k, v in reps.items():
                    arr = np.asarray(v)
                    per_trait[t][f"{k}_ci"] = {
                        x: [float(np.percentile(arr[:, i], 2.5)),
                            float(np.percentile(arr[:, i], 97.5))]
                        for i, x in enumerate(a.arms)}
            print(f"  layer {layer} {t} done", flush=True)

        # ---- is dG_{c,t} trait-specific, or ONE vector the adapter adds everywhere? ----
        # If an adapter added a single global vector M, then dG_{c,t} = M for every t, and
        # table 1 would be M resolved against 8 different axes rather than 8 trait-specific
        # effects. cos(dG_{c,t}, dG_{c,t'}) for t != t' settles it. Traits use DISJOINT
        # question sets, so there is no shared-noise inflation across traits; the halves are
        # still crossed so each cosine is attenuation-free.
        trait_cos = {}
        for arm, gt in gstore.items():
            vals = []
            for i, t1 in enumerate(traits):
                for t2 in traits[i + 1:]:
                    x, y = gt[t1], gt[t2]
                    num = 0.5 * (float(x[0] @ y[1]) + float(x[1] @ y[0]))
                    den = np.sqrt(max(float(x[0] @ x[1]), 1e-12)
                                  * max(float(y[0] @ y[1]), 1e-12))
                    vals.append(num / den)
            trait_cos[arm] = {"mean": float(np.mean(vals)),
                              "min": float(np.min(vals)), "max": float(np.max(vals))}
        per_trait["_trait_cos"] = trait_cos

        results[str(layer)] = per_trait

        R = results[str(layer)]
        text.append(f"\n{'='*110}\nLAYER {layer}   ({len(traits)} traits x {len(pidx)} "
                    f"semantic personas, cross-fitted over {a.half_splits} half-splits)"
                    f"\n{'='*110}")
        text.append("\n" + fmt_table(
            "TABLE 1.  SIGNED  <dG_{c,t}, Vbar_t/||Vbar_t||> / mean_p||V_base_{t,p}||",
            "  + means the constitution displaced the representation toward MORE of the "
            "trait.\n  Same units as table 1 of section 3.2, which is this quantity's "
            "absolute upper bound.",
            traits, a.arms, lambda r, c: R[r]["per_arm"][c]["proj_common"]))
        text.append("\n" + fmt_table(
            "TABLE 1b. the same, each persona against ITS OWN base trait axis",
            "  does not assume the ten personas share one axis",
            traits, a.arms, lambda r, c: R[r]["per_arm"][c]["proj_percell"]))
        text.append("\n" + fmt_table(
            "TABLE 1n. table 1 computed NAIVELY (no cross-fitting)",
            "  shared base noise biases this DOWNWARD; the gap to table 1 is the artefact",
            traits, a.arms, lambda r, c: R[r]["naive"][c]["proj_common"]))
        text.append("\n" + fmt_table(
            "TABLE 2.  ||dG_{c,t}|| / mean_p||V_base_{t,p}||   (unsigned, for reference)",
            "  reproduces table 1 of section 3.2",
            traits, a.arms, lambda r, c: R[r]["per_arm"][c]["g_over_base"]))
        text.append("\n" + fmt_table(
            "TABLE 3.  cos(dG_{c,t}, base trait axis)",
            "  how much of the shift's DIRECTION is the trait axis, free of magnitude",
            traits, a.arms, lambda r, c: R[r]["per_arm"][c]["cos_axis"]))
        text.append("\n" + fmt_table(
            "TABLE 4.  orthogonal remainder, sqrt(table2^2 - table1^2)",
            "  the part of the common shift that is off the trait axis",
            traits, a.arms, lambda r, c: R[r]["per_arm"][c]["orth_common"]))
        text.append("\n\nTABLE 5.  cos(dG_{c,t}, dG_{c,t'}) over the 28 trait PAIRS"
                    "\n  near 1.0 would mean the adapter adds one global vector and table 1"
                    " is not a trait-resolved\n  effect at all. Traits use disjoint "
                    "questions, so no shared noise inflates this.\n")
        text.append(f"    {'arm':22s}{'mean':>9s}{'min':>9s}{'max':>9s}")
        for arm in a.arms:
            c = R["_trait_cos"][arm]
            text.append(f"    {arm:22s}{c['mean']:>9.3f}{c['min']:>9.3f}{c['max']:>9.3f}")

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"config": {"arms": a.arms, "half_splits": a.half_splits,
                    "bootstrap": a.bootstrap, "seed": a.seed},
         **results}, indent=1))
    dest.with_suffix(".txt").write_text("\n".join(text))
    print("\n".join(text))
    print(f"\nwrote {dest} and {dest.with_suffix('.txt')}")


# NOTES, because the sign invites an overclaim.
#
# This measures a REPRESENTATIONAL displacement along the base model's trait axis at the
# CAA answer token. It is not a revealed preference and not a behavioural rate. The OCT
# paper's figure 3 plots a change in Elo from sampled comparisons; this plots a change in
# an activation-space coordinate. They are analogous in shape and not in kind, and the
# claim licensed here is "the constitution moved the representation toward the trait's
# positive pole", not "the model became more impulsive".
#
# The projection is onto ONE axis of a 4096-dimensional shift. cos_axis (table 3) is the
# fraction of the shift's direction that axis explains, and it is small for every arm --
# most of what an adapter does is off-axis, which is the honest framing of table 1.

if __name__ == "__main__":
    main()
