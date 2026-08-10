#!/usr/bin/env python3
"""Step 4: variance decomposition of trait-vector direction into three levels.

The B.1 hierarchy compares three numbers computed three different ways, so "rung 1 >> rung 3"
is suggestive rather than a test (see docs/results/llama31_8b_b1_noise_floor.md). This
replaces the comparison with a single nested random-effects model, which is what an H7-style
claim actually needs: an error bar on a DISPERSION statistic, not a point estimate held up
against a floor.

Model, per (trait, layer). Let v[p, r] be the unit-normalised CAA vector for persona p under
paraphrase r. Working in squared Euclidean distance on the unit sphere, which is exactly the
existing cosine scale since ||a - b||^2 = 2(1 - cos(a, b)):

    sigma2_e   question-sampling error within one cell, from the bootstrap
    sigma2_r   true dispersion ACROSS PARAPHRASES of the same persona   (K/D rung 2)
    sigma2_p   true dispersion ACROSS PERSONAS                          (K/D rung 3)

Each observed paraphrase vector carries measurement error, so the observed dispersions are
inflated and have to be peeled apart rather than read off:

    S_r (observed within-persona)  = sigma2_r + sigma2_e
    S_p (observed between-persona) = sigma2_p + S_r / n_r

giving sigma2_r = S_r - sigma2_e and sigma2_p = S_p - S_r / n_r. The headline is then

    ICC_persona = sigma2_p / (sigma2_p + sigma2_r + sigma2_e)

"how much of the total spread in trait-vector direction is identity, as opposed to phrasing
or noise". A negative component means the level is unresolvable against the one below it and
is reported as 0 with a flag, not silently clipped.

CIs come from a CLUSTER bootstrap over personas: personas are the unit of generalisation, so
resampling questions again would understate the uncertainty that matters for a claim about
how personas differ.

Reads outputs/{model}/caa_activations_paraphrase/{persona}_v{N}_{trait}_{dir}.pt.
CPU-only, numpy, no forward passes.

Usage:
    python scripts/caa_variance_decomposition.py --model meta-llama/Llama-3.1-8B-Instruct
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
from persona_steering.utils import model_short_name

DEFAULT_TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
                  "confidence", "deference", "warmth", "impulsivity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested variance decomposition of trait vectors")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--variants", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--activations-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-boot", type=int, default=50,
                   help="bootstrap replicates per cell, for sigma2_e (default: 50)")
    p.add_argument("--n-cluster", type=int, default=400,
                   help="cluster-bootstrap resamples over personas, for the CI (default: 400)")
    p.add_argument("--report-layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_delta(act_dir: Path, persona: str, variant: int, trait: str) -> np.ndarray:
    """Per-pair difference vectors for one cell -> (M, n_layers, hidden) float32."""
    stem = f"{persona}_v{variant}_{trait}"
    pos_p, neg_p = act_dir / f"{stem}_pos.pt", act_dir / f"{stem}_neg.pt"
    if not pos_p.exists() or not neg_p.exists():
        raise FileNotFoundError(stem)
    pos = torch.load(pos_p, map_location="cpu", weights_only=True)
    neg = torch.load(neg_p, map_location="cpu", weights_only=True)
    kp = sorted(pos.keys(), key=lambda k: int(k[1:]))
    kn = sorted(neg.keys(), key=lambda k: int(k[1:]))
    if kp != kn:
        raise ValueError(f"{stem}: pos/neg question sets differ")
    P = torch.stack([pos[k] for k in kp]).float().numpy()
    N = torch.stack([neg[k] for k in kn]).float().numpy()
    return P - N


def normalise(V: np.ndarray) -> np.ndarray:
    """Unit-normalise along the last axis, so squared distance == 2(1 - cosine)."""
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


def cell_vectors(delta: np.ndarray, n_boot: int, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """-> (full-sample vector (n_layers, hidden), sigma2_e per layer (n_layers,)).

    sigma2_e is the mean squared distance of a bootstrap replicate's direction from the
    cell's full-sample direction: the question-sampling error of one cell's DIRECTION, on
    the same 2(1 - cos) scale as everything else.
    """
    M, n_layers, hidden = delta.shape
    flat = delta.reshape(M, -1)
    full = normalise(flat.mean(0).reshape(n_layers, hidden))

    W = np.zeros((n_boot, M), dtype=np.float32)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, M, size=M), minlength=M)
    W /= M
    reps = normalise((W @ flat).reshape(n_boot, n_layers, hidden))
    sigma2_e = ((reps - full[None]) ** 2).sum(-1).mean(0)
    return full, sigma2_e


def decompose(V: np.ndarray, sigma2_e: np.ndarray) -> dict:
    """Nested decomposition at every layer.

    V: (n_personas, n_variants, n_layers, hidden), unit-normalised.
    sigma2_e: (n_personas, n_variants, n_layers).
    """
    n_p, n_r = V.shape[0], V.shape[1]

    persona_mean = normalise(V.mean(axis=1))            # (n_p, n_layers, hidden)
    grand_mean = normalise(persona_mean.mean(axis=0))   # (n_layers, hidden)

    # observed within-persona (across-paraphrase) dispersion, Bessel-corrected
    S_r = ((V - persona_mean[:, None]) ** 2).sum(-1).sum(axis=1) / max(n_r - 1, 1)  # (n_p, L)
    S_r = S_r.mean(axis=0)                                                          # (L,)
    # observed between-persona dispersion
    S_p = ((persona_mean - grand_mean[None]) ** 2).sum(-1).sum(axis=0) / max(n_p - 1, 1)

    s2_e = sigma2_e.mean(axis=(0, 1))
    s2_r = S_r - s2_e
    s2_p = S_p - S_r / n_r

    # A negative component means that level is not resolvable above the one beneath it.
    # Flag it rather than let a clipped zero read as a measurement.
    neg = {"paraphrase": bool(np.any(s2_r < 0)), "persona": bool(np.any(s2_p < 0))}
    s2_r_c, s2_p_c = np.maximum(s2_r, 0.0), np.maximum(s2_p, 0.0)
    total = s2_p_c + s2_r_c + s2_e
    with np.errstate(invalid="ignore", divide="ignore"):
        icc = np.where(total > 0, s2_p_c / total, np.nan)
    return {"sigma2_e": s2_e, "sigma2_r": s2_r, "sigma2_p": s2_p,
            "icc_persona": icc, "negative_components": neg}


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    act_dir = (Path(args.activations_dir) if args.activations_dir
               else OUTPUTS_DIR / short / "caa_activations_paraphrase")
    out_dir = (Path(args.output_dir) if args.output_dir
               else OUTPUTS_DIR / short / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not act_dir.is_dir():
        print(f"error: {act_dir} not found -- run step 3 first", file=sys.stderr)
        return 2

    traits = args.traits or DEFAULT_TRAITS
    rng = np.random.default_rng(args.seed)
    results: dict = {"model": args.model, "variants": args.variants,
                     "n_boot": args.n_boot, "n_cluster": args.n_cluster, "traits": {}}

    for trait in traits:
        personas = sorted({p.name.split("_v")[0] for p in act_dir.glob(f"*_v*_{trait}_pos.pt")})
        if not personas:
            print(f"skip {trait}: nothing in {act_dir}")
            continue
        print(f"\n{'=' * 70}\nTRAIT: {trait}  ({len(personas)} personas x "
              f"{len(args.variants)} variants)\n{'=' * 70}")

        V, E, kept = [], [], []
        for p in personas:
            try:
                rows = [cell_vectors(load_delta(act_dir, p, v, trait), args.n_boot, rng)
                        for v in args.variants]
            except FileNotFoundError as e:
                print(f"  skip {p}: missing {e}")
                continue
            V.append(np.stack([r[0] for r in rows]))
            E.append(np.stack([r[1] for r in rows]))
            kept.append(p)
            print(f"  {p:<22} loaded {len(rows)} variants")

        if len(kept) < 2:
            print(f"  skip {trait}: need >=2 personas, have {len(kept)}")
            continue

        V, E = np.stack(V), np.stack(E)
        point = decompose(V, E)

        # cluster bootstrap over personas -- personas are the unit of generalisation
        iccs = []
        for _ in range(args.n_cluster):
            idx = rng.integers(0, len(kept), size=len(kept))
            iccs.append(decompose(V[idx], E[idx])["icc_persona"])
        iccs = np.stack(iccs)

        results["traits"][trait] = {
            "personas": kept,
            "n_variants": len(args.variants),
            "sigma2_e": point["sigma2_e"].tolist(),
            "sigma2_r": point["sigma2_r"].tolist(),
            "sigma2_p": point["sigma2_p"].tolist(),
            "icc_persona": point["icc_persona"].tolist(),
            "icc_lo": np.nanpercentile(iccs, 2.5, axis=0).tolist(),
            "icc_hi": np.nanpercentile(iccs, 97.5, axis=0).tolist(),
            "negative_components": point["negative_components"],
        }

    out_path = out_dir / "caa_variance_decomposition.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    for L in args.report_layers:
        print(f"\n{'=' * 88}\nLAYER {L}   (variance components on the 2(1-cos) scale)\n{'=' * 88}")
        print(f"{'trait':<15}{'error':>10}{'paraphrase':>12}{'persona':>10}"
              f"{'ICC_persona':>14}{'95% CI':>20}")
        print("-" * 88)
        for t, r in results["traits"].items():
            lo, hi = r["icc_lo"][L], r["icc_hi"][L]
            flag = " *" if (r["negative_components"]["paraphrase"]
                            or r["negative_components"]["persona"]) else ""
            print(f"{t:<15}{r['sigma2_e'][L]:>10.4f}{r['sigma2_r'][L]:>12.4f}"
                  f"{r['sigma2_p'][L]:>10.4f}{r['icc_persona'][L]:>14.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20}{flag}")
        print("-" * 88)
        print("ICC_persona = share of trait-vector direction spread attributable to IDENTITY")
        print("rather than phrasing or question sampling. '*' = a component went negative at")
        print("some layer, i.e. that level is not resolvable above the one below it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
