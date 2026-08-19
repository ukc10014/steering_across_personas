#!/usr/bin/env python3
"""Is the cross-arm compression a loss of persona structure, or a shared additive component?

ANSWER, ESTABLISHED BY THIS SCRIPT: a shared additive component. See
docs/results/llama31_8b_character_arms.md §2.

THE PROBLEM. Merging any OCT LoRA adapter raises persona-mean cos(v_persona, v_null) from
0.723 to ~0.90 at L15, by the same amount for all three adapters including `mathematical`,
whose constitution is orthogonal to all eight traits. Cosine is invariant to SCALING each
vector but not to ADDING the same vector to both of its arguments, and a merge does exactly
the latter -- so an arm can sit higher purely because its shared component is bigger.

WHAT IS COMPUTED, and what each thing is for:

  * share = ||mean vector|| / mean||vector||, over {null + the 10 personas}. The direct
    measurement of the shared component. If trait vectors are a shared part plus mutually
    near-orthogonal specific parts then raw cosine = share^2 exactly; measured, the two agree
    to within 0.017 across all 32 arm x trait cells, which is the whole finding.
  * R = ||mean of unit vectors||. Norm-free, so it separates a directional common component
    from a rescaling. It tracks share to three decimals here -- it is directional.
  * residual_cos -> per-persona cosine with the shared component removed. Read its docstring
    before trusting any corrected number; the obvious estimator is biased and was used, and
    circulated, before 2026-08-13.
  * DELTA ALIGNMENT, within and across arms. d_p = v_p^arm - v_p^base. Mutual alignment of the
    d_p says whether the merge adds a persona-independent component; cos(d^armA, d^armB) says
    whether two different constitutions add the SAME one. They do: +0.816 for goodness vs
    mathematical at L15. Neither test subtracts a mean from anything.
  * cos(v_null^arm, v_null^base) -- "Frame B", registered in docs/HANDOVER_old.md and never
    computed until now. Says whether character training moved the default itself.

`nonsense` is held out of every shared-component estimate and reported separately: it is a
control, and letting it inform the mean would let it move its own baseline.

Point estimates only. For intervals, a no-structure floor, and the dispersion and ordering
statistics that carry the surviving result, use scripts/caa_holdout_ci.py.

All arms are held in RAM at once (sliced to the requested layers, ~13 MB) because the ~6 min
of network-volume I/O dwarfs the arithmetic and every cross-arm statistic needs two arms
resident simultaneously.

Usage:
    python scripts/caa_shared_component.py --layers 15 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR

NULL = "null"
NONSENSE = "nonsense"
PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty", "confidence",
          "deference", "warmth", "impulsivity"]
ARMS = {"base": "Llama-3.1-8B-Instruct",
        "goodness": "llama-3.1-8b-goodness",
        "mathematical": "llama-3.1-8b-mathematical",
        "impulsiveness": "llama-3.1-8b-impulsiveness",
        "misalignment": "llama-3.1-8b-misalignment"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--traits", nargs="+", default=TRAITS)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--n-splits", type=int, default=200,
                   help="disjoint-estimate splits for residual_cos")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis"
                              / "caa_shared_component.json"))
    return p.parse_args()


def trait_vector(act_dir: Path, persona: str, trait: str) -> np.ndarray:
    """mean(pos) - mean(neg) over all questions -> (n_layers, hidden) float32.

    Averaged one direction at a time: each file is 500 x 32 x 4096 fp16 (131 MB) and only
    its mean survives, so peak resident stays at one file rather than the whole cell grid.
    """
    out = None
    for sign, direction in ((1.0, "pos"), (-1.0, "neg")):
        path = act_dir / f"{persona}_{trait}_{direction}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = torch.stack(list(d.values())).float().mean(0).numpy()
        out = sign * m if out is None else out + sign * m
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-8)


def mean_pairwise_cos(stack: np.ndarray) -> np.ndarray:
    """Mean off-diagonal pairwise cosine of a (k, n_layers, hidden) stack -> (n_layers,)."""
    k = stack.shape[0]
    unit = stack / np.maximum(np.linalg.norm(stack, axis=-1, keepdims=True), 1e-8)
    gram = np.einsum("ilh,jlh->lij", unit, unit)
    off = gram.sum((1, 2)) - np.trace(gram, axis1=1, axis2=2)
    return off / (k * (k - 1))


def make_splits(k: int, n_splits: int, rng: np.random.Generator) -> list[tuple]:
    """(est1, est2, ev) triples. est1/est2 are DISJOINT and neither contains ev."""
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(k)
        out.append((perm[:4], perm[4:8], perm[8:]))
    return out


def residual_cos(P: np.ndarray, N: np.ndarray, splits: list[tuple], extra: np.ndarray | None
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """Persona-vs-null cosine with the shared component removed, WITHOUT inducing a new one.

    P: (..., k, L, H) personas, N: (..., L, H) null. Returns per-persona (..., k, L).

    THE SUBTLETY THAT COST A REWRITE. The obvious correction -- subtract the mean of a
    hold-out set from both sides -- does not work. Writing v = c + s, subtracting an estimate
    m = c + e leaves (s_p - e) and (s_null - e): the SAME estimation error e sits in both
    arguments of the cosine, which is exactly the defect being corrected for, one order down.
    Measured on synthetic data with no persona structure at all, that estimator reads +0.11
    to +0.17 instead of 0, and on the real activations its value was indistinguishable from
    its own no-structure floor -- i.e. it had no power whatsoever.

    Projecting out the estimated common DIRECTION fails for the same reason: the direction is
    estimated with error, so the unremoved part of c is again shared.

    The fix is to estimate the shared component TWICE from disjoint persona subsets, and
    correct the two sides of the cosine with different estimates. The leftover errors are then
    independent, their inner product has mean zero, and the statistic reads 0.000 when no
    structure is present (verified synthetically) while remaining sensitive when it is.

    `extra` (nonsense) is evaluated under every split but never enters an estimation set.
    """
    k = P.shape[-3]
    tot = np.zeros(P.shape[:-1])
    cnt = np.zeros(k)
    ex_acc = []
    for est1, est2, ev in splits:
        m1 = P[..., est1, :, :].mean(-3)
        m2 = P[..., est2, :, :].mean(-3)
        tot[..., ev, :] += cosine(P[..., ev, :, :] - m1[..., None, :, :],
                                  (N - m2)[..., None, :, :])
        cnt[ev] += 1
        if extra is not None:
            ex_acc.append(cosine(extra - m1, N - m2))
    if (cnt == 0).any():
        raise RuntimeError("a persona was never evaluated; raise --n-splits")
    return tot / cnt[:, None], (np.mean(ex_acc, axis=0) if extra is not None else None)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    splits = make_splits(len(PERSONAS), args.n_splits, rng)
    LI = np.asarray(args.layers)
    conditions = [NULL, NONSENSE] + PERSONAS

    # ---- load once ----------------------------------------------------------------------
    # Sliced to the requested layers immediately: the full (32, 4096) set for 4 arms would be
    # 201 MB, the 2-layer slice is 13 MB, and every cross-arm statistic below needs two arms
    # resident at the same time.
    V: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm in args.arms:
        act_dir = OUTPUTS_DIR / ARMS[arm] / "caa_activations"
        print(f"loading {arm:<15} {act_dir}", flush=True)
        V[arm] = {t: {p: trait_vector(act_dir, p, t)[LI] for p in conditions}
                  for t in args.traits}

    results: dict = {"layers": args.layers, "personas": PERSONAS,
                     "n_splits": args.n_splits, "seed": args.seed, "arms": {}}

    # ---- per arm x trait ----------------------------------------------------------------
    for arm in args.arms:
        print(f"\n{'=' * 78}\nARM: {arm}\n{'=' * 78}", flush=True)
        arm_res: dict = {}
        for trait in args.traits:
            v = V[arm][trait]
            personas = np.stack([v[p] for p in PERSONAS])        # (10, n_layers, hidden)
            null = v[NULL]
            stack = np.concatenate([null[None], personas])        # (11, ...); nonsense held out
            m = stack.mean(0)

            raw = cosine(personas, null)                          # (10, n_layers)
            cen = cosine(personas - m, null - m)
            norms = np.linalg.norm(stack, axis=-1)                # (11, n_layers)
            unit = stack / np.maximum(norms[..., None], 1e-8)

            hold, hold_ns = residual_cos(personas, null, splits, v[NONSENSE])
            arm_res[trait] = {
                "raw_per_persona": raw.T.tolist(),          # (n_layers, 10), PERSONAS order
                "holdout_per_persona": hold.T.tolist(),
                "raw_mean": raw.mean(0).tolist(),
                "raw_sd": raw.std(0, ddof=1).tolist(),
                "holdout_sd": hold.std(0, ddof=1).tolist(),
                "cen_mean": cen.mean(0).tolist(),
                "holdout_mean": hold.mean(0).tolist(),
                "holdout_nonsense": hold_ns.tolist(),
                "raw_nonsense": cosine(v[NONSENSE], null).tolist(),
                "share": (np.linalg.norm(m, axis=-1) / np.maximum(norms.mean(0), 1e-8)).tolist(),
                "R": np.linalg.norm(unit.mean(0), axis=-1).tolist(),
                "null_norm": norms[0].tolist(),
                "persona_norm_mean": norms[1:].mean(0).tolist(),
            }
            r = arm_res[trait]
            print("  {:<15} {}".format(trait, "  ".join(
                f"L{L}: raw {r['raw_mean'][i]:+.3f} hold {r['holdout_mean'][i]:+.3f} "
                f"share {r['share'][i]:.3f}" for i, L in enumerate(args.layers))), flush=True)
        results["arms"][arm] = arm_res

    # ---- delta alignment: does the merge add the SAME thing to every condition? ----------
    if "base" in args.arms:
        results["delta_alignment"] = {}
        print(f"\n{'=' * 78}\nDELTA ALIGNMENT  mean pairwise cos among d_p = v_p^arm - v_p^base\n"
              f"(high => the merge adds a persona-independent component; ~0 => it reorganises)\n"
              f"{'=' * 78}")
        for arm in args.arms:
            if arm == "base":
                continue
            per_trait = {}
            for trait in args.traits:
                d = np.stack([V[arm][trait][p] - V["base"][trait][p] for p in [NULL] + PERSONAS])
                per_trait[trait] = {
                    "delta_pairwise": mean_pairwise_cos(d).tolist(),
                    # reference: how aligned the ORIGINAL base vectors are with each other,
                    # so "high" is judged against the structure that was already there.
                    "base_pairwise": mean_pairwise_cos(
                        np.stack([V["base"][trait][p] for p in [NULL] + PERSONAS])).tolist(),
                    "delta_norm_ratio": (
                        np.linalg.norm(d, axis=-1).mean(0)
                        / np.maximum(np.linalg.norm(
                            np.stack([V["base"][trait][p] for p in [NULL] + PERSONAS]),
                            axis=-1).mean(0), 1e-8)).tolist(),
                }
            results["delta_alignment"][arm] = per_trait
            for key, lab in (("delta_pairwise", "delta"), ("base_pairwise", "base ref"),
                             ("delta_norm_ratio", "||d||/||v||")):
                mb = np.mean([per_trait[t][key] for t in args.traits], axis=0)
                print(f"  {arm:<15} {lab:<12} " + "  ".join(
                    f"L{L}: {mb[i]:+.3f}" for i, L in enumerate(args.layers)))

    # ---- do the three merges add the SAME perturbation? ----------------------------------
    # Weight-space max|dw| is tiny (0.00146 for goodness) yet the activation-space change is
    # ~0.7-0.9x the trait vector's own norm. If the three adapters' perturbations are also
    # mutually aligned, there is one perturbation here, not three, and no room for content.
    if "base" in args.arms and len(args.arms) > 2:
        results["delta_cross_arm"] = {}
        print(f"\n{'=' * 78}\nCROSS-ARM DELTA  mean_p cos(d_p^armA, d_p^armB)\n{'=' * 78}")
        adapted = [a for a in args.arms if a != "base"]
        for i, a in enumerate(adapted):
            for b in adapted[i + 1:]:
                per_trait = {}
                for t in args.traits:
                    da = np.stack([V[a][t][p] - V["base"][t][p] for p in [NULL] + PERSONAS])
                    db = np.stack([V[b][t][p] - V["base"][t][p] for p in [NULL] + PERSONAS])
                    per_trait[t] = cosine(da, db).mean(0).tolist()
                results["delta_cross_arm"][f"{a}|{b}"] = per_trait
                mb = np.mean([per_trait[t] for t in args.traits], axis=0)
                print(f"  {a:<14} vs {b:<15} " + "  ".join(
                    f"L{L}: {mb[j]:+.3f}" for j, L in enumerate(args.layers)))

    # ---- cross-arm null direction -------------------------------------------------------
    results["null_cross_arm"] = {}
    print(f"\n{'=' * 78}\ncos between arms' OWN null vectors (same trait)\n{'=' * 78}")
    for i, a in enumerate(args.arms):
        for b in args.arms[i + 1:]:
            per_trait = {t: cosine(V[a][t][NULL], V[b][t][NULL]).tolist() for t in args.traits}
            results["null_cross_arm"][f"{a}|{b}"] = per_trait
            mb = np.mean([per_trait[t] for t in args.traits], axis=0)
            print(f"  {a:<14} vs {b:<15} " + "  ".join(
                f"L{L}: {mb[j]:+.3f}" for j, L in enumerate(args.layers)))

    # ---- headline table -----------------------------------------------------------------
    print(f"\n{'=' * 78}\nPERSONA-MEAN COSINE: raw vs shared-component-removed\n{'=' * 78}")
    for i, L in enumerate(args.layers):
        print(f"\nlayer {L}")
        print(f"  {'arm':<15} {'raw':>8} {'hold-out':>9} {'centred':>8} {'share':>8} "
              f"{'R':>8} {'raw ns':>8}")
        for arm in args.arms:
            a = results["arms"][arm]
            g = lambda k: float(np.mean([a[t][k][i] for t in args.traits]))  # noqa: E731
            print(f"  {arm:<15} {g('raw_mean'):>8.3f} {g('holdout_mean'):>9.3f} "
                  f"{g('cen_mean'):>8.3f} {g('share'):>8.3f} {g('R'):>8.3f} "
                  f"{g('raw_nonsense'):>8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
