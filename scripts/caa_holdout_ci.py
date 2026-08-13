#!/usr/bin/env python3
"""Question-resampling CIs for the shared-component-corrected statistics, plus their floor.

WHY. caa_shared_component.py gives point estimates only. Its own split-to-split variation says
nothing about sampling error over the 500 CAA questions, and the surviving result -- that
`impulsiveness` reorders personas where `goodness` and `mathematical` do not -- is worthless
without an interval. This supplies that.

WHAT IS REPORTED, per arm and layer:

  * residual MEAN and its CI. Reported for completeness; it sits at ~0.016 in every arm
    including base and separates nothing. Do not read "arms agree on the mean" as "nothing
    changed" -- that inference is what this whole analysis exists to correct.
  * residual DISPERSION (SD across personas) and its CI. Dispersion was the registered primary
    (docs/results/llama31_8b_character_arms.md §6) and had only ever been reported as a mean.
    Raw SD falls ~60% under merging; residual SD does not move in any arm.
  * SPEARMAN vs base on the per-persona residual ordering. The one quantity with a real
    between-arm difference, so the one that most needs an interval.
  * PAIRED CIs on (arm - base). All four arms are extracted over the SAME 500 questions, so a
    replicate uses one index draw for both arms and differences them, cancelling the shared
    question-sampling noise. Pairing is correct HERE precisely because both sides are the same
    questions -- contrast persona-vs-null below.
  * FLOOR: the same statistic with the ten personas replaced by ten independent resamples of
    NULL, which have no persona structure by construction. It reads -0.000 [-0.050, +0.048],
    confirming the estimator is unbiased. An earlier estimator failed exactly this check.

Persona-vs-null stays UNPAIRED (independent draws for the two sides), matching
caa_cosine_to_null.py: paired draws would share sampling noise between the two vectors and
inflate the cosine, and would make the floor degenerate.

Splits are drawn ONCE and reused across every replicate, arm and trait, so split choice cannot
leak into a CI or into the arm contrast.

Usage:
    python scripts/caa_holdout_ci.py --layers 15 20 --n-boot 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_steering.config import OUTPUTS_DIR
from caa_cosine_to_null import boot_vectors
from caa_shared_component import ARMS, NULL, PERSONAS, TRAITS, cosine, make_splits, residual_cos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--traits", nargs="+", default=TRAITS)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-splits", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "llama-3.1-8b-goodness" / "analysis"
                              / "caa_holdout_ci.json"))
    return p.parse_args()


def load_pair(act_dir: Path, persona: str, trait: str, layers: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """(pos, neg), each (n_questions, n_layers_sel, hidden) float32.

    Sliced to the requested layers at load: the full cell is 500 x 32 x 4096, and keeping
    only two layers is the difference between 262 MB and 16 MB per direction, which is what
    lets every bootstrap replicate for a cell stay resident.
    """
    out = []
    for direction in ("pos", "neg"):
        path = act_dir / f"{persona}_{trait}_{direction}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        d = torch.load(path, map_location="cpu", weights_only=False)
        arr = torch.stack([d[k] for k in sorted(d)]).float().numpy()[:, layers]
        out.append(np.ascontiguousarray(arr))
    return out[0], out[1]


def holdout_stats(P: np.ndarray, N: np.ndarray, splits: list[tuple]) -> tuple:
    """(mean over personas, SD across personas) of the residual cosine, each (B, n_layers).

    Both come from ONE residual_cos call: it is the expensive step, and location and
    dispersion are two reductions of the same per-persona array.

    Dispersion is returned because the pre-registered primary was dispersion, not location,
    and the two move in opposite directions here: on warmth, raw SD falls 0.184 -> 0.075
    under `goodness` while residual SD rises 0.229 -> 0.317. Reporting only the mean is the
    error this whole analysis exists to correct, and it would be careless to repeat it.
    """
    per_persona, _ = residual_cos(P, N, splits, None)      # (B, k, L)
    return per_persona.mean(-2), per_persona.std(-2, ddof=1), per_persona


def spearman_vs(A: np.ndarray, Bv: np.ndarray) -> np.ndarray:
    """Per-replicate Spearman between two (B, k, L) per-persona arrays -> (B, L).

    Ranks are taken across the persona axis, then Pearson-correlated: that IS Spearman, and
    it vectorises over replicates and layers, where 9600 scipy calls would not.

    This is the statistic that survives the shared-component correction with a difference
    between arms, so it is the one that most needs an interval.
    """
    ra = rankdata(A, axis=-2)
    rb = rankdata(Bv, axis=-2)
    ra = ra - ra.mean(-2, keepdims=True)
    rb = rb - rb.mean(-2, keepdims=True)
    num = (ra * rb).sum(-2)
    den = np.sqrt((ra ** 2).sum(-2) * (rb ** 2).sum(-2))
    return num / np.maximum(den, 1e-12)


def main() -> int:
    args = parse_args()
    layers = np.asarray(args.layers)
    B, k = args.n_boot, len(PERSONAS)
    rng = np.random.default_rng(args.seed)

    # Fixed splits, shared by every replicate, arm and trait, so split choice cannot leak
    # into the CI or into the paired arm contrast.
    splits = make_splits(k, args.n_splits, rng)

    results: dict = {"layers": args.layers, "n_boot": B, "n_splits": args.n_splits,
                     "seed": args.seed, "traits": {}}
    # stat[arm] and floor[arm]: (n_traits, B, n_layers), replicate-aligned across arms.
    stat: dict[str, list] = {a: [] for a in args.arms}
    sd: dict[str, list] = {a: [] for a in args.arms}
    rho: dict[str, list] = {a: [] for a in args.arms}
    floor: dict[str, list] = {a: [] for a in args.arms}

    for trait in args.traits:
        print(f"\n=== {trait} ===", flush=True)
        # Index draws are made ONCE per trait and reused for every arm, so replicate b means
        # the same 500 questions in every arm and the arm contrast can be paired.
        n_q = None
        idx_a = idx_b = idx_f = None

        for arm in args.arms:
            act_dir = OUTPUTS_DIR / ARMS[arm] / "caa_activations"
            npos, nneg = load_pair(act_dir, NULL, trait, layers)
            if n_q is None:
                n_q = npos.shape[0]
                idx_a = [rng.integers(0, n_q, n_q) for _ in range(B)]        # personas
                idx_b = [rng.integers(0, n_q, n_q) for _ in range(B)]        # null
                idx_f = [[rng.integers(0, n_q, n_q) for _ in range(B)]       # 10 fake personas
                         for _ in range(k)]
            elif npos.shape[0] != n_q:
                raise SystemExit(f"{arm}/{trait}: {npos.shape[0]} questions, expected {n_q}")

            N = boot_vectors(npos, nneg, idx_b)                              # (B, L, H)
            F = np.stack([boot_vectors(npos, nneg, idx_f[i]) for i in range(k)], axis=1)
            P = np.empty((B, k, len(layers), npos.shape[2]), dtype=np.float32)
            for i, p in enumerate(PERSONAS):
                pos, neg = load_pair(act_dir, p, trait, layers)
                P[:, i] = boot_vectors(pos, neg, idx_a)

            m_, sd_, pp = holdout_stats(P, N, splits)
            stat[arm].append(m_)
            sd[arm].append(sd_)
            # Ordering is measured against the BASE arm's per-persona residuals from the same
            # replicate, i.e. the same 500 questions on both sides.
            if arm == args.arms[0]:
                pp_base = pp
            rho[arm].append(spearman_vs(pp_base, pp))
            floor[arm].append(holdout_stats(F, N, splits)[0])
            print(f"  {arm:<15} hold={stat[arm][-1].mean(0)}  floor={floor[arm][-1].mean(0)}",
                  flush=True)

    # ---- aggregate over traits, then summarise over replicates ----------------------------
    S = {a: np.mean(stat[a], axis=0) for a in args.arms}      # (B, n_layers)
    SD = {a: np.mean(sd[a], axis=0) for a in args.arms}
    RHO = {a: np.mean(rho[a], axis=0) for a in args.arms}
    RHO_T = {a: np.array(rho[a]) for a in args.arms}          # (n_traits, B, n_layers)
    F = {a: np.mean(floor[a], axis=0) for a in args.arms}
    base = args.arms[0]

    def ci(x, axis=0):
        return (float(np.mean(x)), float(np.percentile(x, 2.5, axis=axis)),
                float(np.percentile(x, 97.5, axis=axis)))

    print(f"\n{'=' * 84}\nHOLD-OUT-CENTRED COSINE with question-resampling CI "
          f"(n_boot={B})\n{'=' * 84}")
    for i, L in enumerate(args.layers):
        print(f"\nlayer {L}")
        print(f"  {'arm':<15}{'residual mean [95% CI]':>28}{'floor [95% CI]':>26}"
              f"{'paired Δ mean vs base':>28}{'residual SD [95% CI]':>26}"
              f"{'paired Δ SD vs base':>28}{'Spearman vs base [95% CI]':>30}")
        for a in args.arms:
            m, lo, hi = ci(S[a][:, i])
            fm, flo, fhi = ci(F[a][:, i])
            cell = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
            fcell = f"{fm:+.3f} [{flo:+.3f},{fhi:+.3f}]"
            if a == base:
                dcell = "—"
            else:
                d = S[a][:, i] - S[base][:, i]          # paired: same questions both sides
                dm, dlo, dhi = ci(d)
                sig = "*" if (dlo > 0) or (dhi < 0) else " "
                dcell = f"{dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sig}"
            sm, slo, shi = ci(SD[a][:, i])
            scell = f"{sm:.3f} [{slo:.3f},{shi:.3f}]"
            if a == base:
                sdcell = "—"
            else:
                ds = SD[a][:, i] - SD[base][:, i]
                dsm, dslo, dshi = ci(ds)
                ssig = "*" if (dslo > 0) or (dshi < 0) else " "
                sdcell = f"{dsm:+.3f} [{dslo:+.3f},{dshi:+.3f}]{ssig}"
            rm, rlo, rhi = ci(RHO[a][:, i])
            rcell = "—" if a == base else f"{rm:+.3f} [{rlo:+.3f},{rhi:+.3f}]"
            print(f"  {a:<15}{cell:>28}{fcell:>26}{dcell:>28}{scell:>26}{sdcell:>28}{rcell:>30}")
        print("  * = 95% CI excludes 0")

    results["summary"] = {
        str(L): {a: {"hold": ci(S[a][:, i]), "sd": ci(SD[a][:, i]), "floor": ci(F[a][:, i]),
                     "delta_vs_base": (None if a == base else ci(S[a][:, i] - S[base][:, i])),
                     "delta_sd_vs_base": (None if a == base else ci(SD[a][:, i] - SD[base][:, i])),
                     "spearman_vs_base": (None if a == base else ci(RHO[a][:, i])),
                     "spearman_by_trait": (None if a == base else
                         {t: ci(RHO_T[a][ti, :, i]) for ti, t in enumerate(args.traits)})}
                 for a in args.arms}
        for i, L in enumerate(args.layers)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
