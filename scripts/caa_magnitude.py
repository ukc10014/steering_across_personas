#!/usr/bin/env python3
"""CAA trait-vector MAGNITUDE, and the K/D B.6 decoupling test.

Two questions, both asked of the same cached activations the B.1 work used:

  1. How long is a persona's trait vector, relative to the null-context vector for the
     same trait? Reported as log2(||v_persona|| / ||v_null||), so 0 means "same length as
     the assistant default", -1 means "half as long".

  2. K/D B.6 claims magnitude and direction DECOUPLE at the cell level -- i.e. how far a
     persona rotates a trait vector tells you nothing about whether it lengthens or
     shortens it. That is a testable correlation, not a description: take every
     persona x trait cell, put cosine-to-null on one axis and log2 magnitude ratio on the
     other, and measure the association. Decoupling predicts ~zero correlation.

Computed straight from outputs/{model}/caa_activations/ rather than from a vectors/ dir,
deliberately: 3_vectors.py slices [:-1] (see fork-infra section 6.1), so saved vectors are
(n_layers-1, hidden) and every downstream --layer silently indexes a truncated tensor. Going
back to the activations keeps layer indices meaning what they say.

CPU-only, numpy. Reuses the weighted-mean matmul trick from caa_within_cell_stability.py so
the bootstrap costs one BLAS call per cell rather than one array copy per replicate.

Usage:
    python scripts/caa_magnitude.py --model meta-llama/Llama-3.1-8B-Instruct
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

NULL_SLUG = "null"
CONTROL_SLUG = "nonsense"
DEFAULT_TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
                  "confidence", "deference", "warmth", "impulsivity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAA magnitude + B.6 decoupling test")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--activations-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-boot", type=int, default=200,
                   help="bootstrap replicates for the magnitude CI (default: 200)")
    p.add_argument("--report-layers", type=int, nargs="+", default=[15, 20, 25])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_delta(act_dir: Path, persona: str, trait: str) -> np.ndarray:
    """Per-pair difference vectors -> (M, n_layers, hidden) float32."""
    pos_p, neg_p = act_dir / f"{persona}_{trait}_pos.pt", act_dir / f"{persona}_{trait}_neg.pt"
    if not pos_p.exists() or not neg_p.exists():
        raise FileNotFoundError(f"{persona}_{trait}")
    pos = torch.load(pos_p, map_location="cpu", weights_only=True)
    neg = torch.load(neg_p, map_location="cpu", weights_only=True)
    kp = sorted(pos.keys(), key=lambda k: int(k[1:]))
    kn = sorted(neg.keys(), key=lambda k: int(k[1:]))
    if kp != kn:
        raise ValueError(f"{persona}/{trait}: pos and neg question sets differ")
    P = torch.stack([pos[k] for k in kp]).float().numpy()
    N = torch.stack([neg[k] for k in kn]).float().numpy()
    return P - N


def magnitudes(delta: np.ndarray, n_boot: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray]:
    """-> (point magnitude per layer, bootstrap replicate magnitudes (n_boot, n_layers))."""
    M, n_layers, hidden = delta.shape
    flat = delta.reshape(M, -1)
    point = np.linalg.norm(flat.mean(0).reshape(n_layers, hidden), axis=-1)

    W = np.zeros((n_boot, M), dtype=np.float32)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, M, size=M), minlength=M)
    W /= M
    reps = (W @ flat).reshape(n_boot, n_layers, hidden)
    return point, np.linalg.norm(reps, axis=-1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    act_dir = (Path(args.activations_dir) if args.activations_dir
               else OUTPUTS_DIR / short / "caa_activations")
    out_dir = (Path(args.output_dir) if args.output_dir
               else OUTPUTS_DIR / short / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not act_dir.is_dir():
        print(f"error: {act_dir} not found", file=sys.stderr)
        return 2

    traits = args.traits or [t for t in DEFAULT_TRAITS
                             if (act_dir / f"{NULL_SLUG}_{t}_pos.pt").exists()]
    rng = np.random.default_rng(args.seed)
    results: dict = {"model": args.model, "n_boot": args.n_boot, "traits": {}}

    for trait in traits:
        print(f"\n=== {trait} ===")
        n = len(f"_{trait}_pos.pt")
        cells = sorted(p.name[:-n] for p in act_dir.glob(f"*_{trait}_pos.pt"))
        if NULL_SLUG not in cells:
            print(f"  skip: no {NULL_SLUG} reference")
            continue

        null_point, null_boot = magnitudes(load_delta(act_dir, NULL_SLUG, trait),
                                           args.n_boot, rng)
        tr: dict = {"null_magnitude": null_point.tolist(), "cells": {}}

        for cell in cells:
            point, boot = magnitudes(load_delta(act_dir, cell, trait), args.n_boot, rng)
            # log2 ratio against null. Ratio rather than raw norm because raw norms are not
            # comparable across layers -- the residual stream grows with depth, so an
            # unnormalised magnitude curve mostly plots the residual stream, not the trait.
            log2_ratio = np.log2(np.maximum(point, 1e-8) / np.maximum(null_point, 1e-8))
            ratio_boot = np.log2(np.maximum(boot, 1e-8) / np.maximum(null_boot, 1e-8))
            tr["cells"][cell] = {
                "magnitude": point.tolist(),
                "log2_ratio_to_null": log2_ratio.tolist(),
                "lo": np.percentile(ratio_boot, 2.5, axis=0).tolist(),
                "hi": np.percentile(ratio_boot, 97.5, axis=0).tolist(),
            }
            print(f"  {cell:<22} L20 |v|={point[20]:7.3f}  log2(v/null)={log2_ratio[20]:+.3f}")

        results["traits"][trait] = tr

    # ---- B.6 decoupling test, against the cosine-to-null already on disk ----
    cos_path = out_dir / "caa_cosine_to_null.json"
    if cos_path.exists():
        cos = json.loads(cos_path.read_text())
        results["decoupling"] = {}
        n_layers = len(results["traits"][traits[0]]["null_magnitude"])
        for L in range(n_layers):
            xs, ys = [], []
            for t in results["traits"]:
                ct = cos.get("traits", {}).get(t)
                if not ct:
                    continue
                for cell, cd in results["traits"][t]["cells"].items():
                    if cell in (NULL_SLUG, CONTROL_SLUG):
                        continue
                    cv = ct["personas"].get(cell)
                    if cv is None:
                        continue
                    xs.append(cv["point"][L])
                    ys.append(cd["log2_ratio_to_null"][L])
            if len(xs) > 3:
                x, y = np.array(xs), np.array(ys)
                results["decoupling"][str(L)] = {
                    "n_cells": len(xs), "pearson_r": pearson(x, y),
                    "spearman_rho": spearman(x, y),
                }

    out_path = out_dir / "caa_magnitude.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    for L in args.report_layers:
        print(f"\n{'=' * 78}\nLAYER {L}\n{'=' * 78}")
        print(f"{'trait':<15}{'null |v|':>10}{'persona |v|':>13}{'log2 ratio':>12}"
              f"{'nonsense':>11}")
        print("-" * 78)
        for t, tr in results["traits"].items():
            pers = [c for c in tr["cells"] if c not in (NULL_SLUG, CONTROL_SLUG)]
            pm = np.mean([tr["cells"][c]["magnitude"][L] for c in pers])
            lr = np.mean([tr["cells"][c]["log2_ratio_to_null"][L] for c in pers])
            ns = (tr["cells"][CONTROL_SLUG]["log2_ratio_to_null"][L]
                  if CONTROL_SLUG in tr["cells"] else float("nan"))
            print(f"{t:<15}{tr['null_magnitude'][L]:>10.3f}{pm:>13.3f}{lr:>+12.3f}{ns:>+11.3f}")
        d = results.get("decoupling", {}).get(str(L))
        if d:
            print("-" * 78)
            print(f"B.6 decoupling test, n={d['n_cells']} cells:  "
                  f"pearson r = {d['pearson_r']:+.3f}   spearman rho = {d['spearman_rho']:+.3f}")
            print("  decoupling predicts r ~ 0; |r| well above 0 means rotation and "
                  "shortening travel together")
    return 0


if __name__ == "__main__":
    sys.exit(main())
