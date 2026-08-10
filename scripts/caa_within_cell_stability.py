#!/usr/bin/env python3
"""K/D B.1 rung 1, in K/D's own estimator, plus a split-half question-bank floor.

Why this exists alongside scripts/caa_cosine_to_null.py: that script's noise floor is
NULL-vs-NULL under two INDEPENDENT (unpaired) resamples, which is the right construction
for the thing it is a floor for (a persona-vs-null cosine). K/D's B.1 rung 1 is a
different statistic -- pairwise cosine among B bootstrap replicates OF ONE CELL -- and the
two are not interchangeable. Quoting our unpaired floor against their 0.99 would be
comparing two different estimators. This script computes theirs.

Three statistics, all per (cell, layer), all on the same cached activations:

  within_cell   K/D B.1 rung 1. Resample the M contrastive PAIRS with replacement B times,
                rebuild the CAA vector per replicate, take the mean of all B(B-1)/2
                pairwise cosines. Resampling is PAIRED here (same index draw applied to pos
                and neg) because the contrastive pair is the sampling unit -- unlike the
                unpaired construction in caa_cosine_to_null.py, which is answering a
                different question.

  split_half    Disjoint halves of the question bank, R random splits, cosine between the
                two half-vectors. Bootstrap resamples the M pairs you HAVE; this varies
                WHICH questions you asked, so it captures question-set variance that the
                bootstrap cannot see. Reported raw and Spearman-Brown corrected (each half
                uses M/2 pairs, so the raw number is pessimistic relative to a full-M
                vector; SB projects it back up).

  boot_half     Two independent size-M/2 WITH-replacement draws from the full bank. This is
                the size-matched control that separates the two effects above: boot_half vs
                within_cell isolates sample size, split_half vs boot_half isolates
                disjointness, i.e. the question-set variance proper.

  across_cell   K/D B.1 rung 3, recomputed in the SAME FORM as rung 1: pairwise cosine among
                the full-sample vectors of the N personas. Still not the same estimand as
                rung 1 (one varies resampling, the other varies identity), but at least the
                same functional form, so the comparison is like-for-like in a way the raw
                rung-1-vs-rung-3 contrast is not.

Pure re-analysis of outputs/{model}/caa_activations/ -- CPU, numpy, no forward passes.
The per-pair delta array [M, n_layers, hidden] is reconstructed on the fly from the cached
pos/neg files, which carry identical question keys in identical order.

Usage:
    python scripts/caa_within_cell_stability.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/caa_within_cell_stability.py --model ... --traits warmth honesty --n-boot 50
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
    p = argparse.ArgumentParser(description="K/D B.1 rung 1 + split-half question-bank floor")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--traits", nargs="+", default=None,
                   help="default: all 8 K/D traits present in the activations dir")
    p.add_argument("--cells", nargs="+", default=None,
                   help="persona slugs to score; default: every persona found, incl. null/nonsense")
    p.add_argument("--activations-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-boot", type=int, default=50,
                   help="bootstrap replicates per cell; 50 matches K/D B.1 (default: 50)")
    p.add_argument("--n-splits", type=int, default=100,
                   help="random half-splits of the question bank (default: 100)")
    p.add_argument("--report-layers", type=int, nargs="+", default=[15, 20],
                   help="layers to print in the summary table; JSON always holds all layers")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_delta(act_dir: Path, persona: str, trait: str) -> np.ndarray:
    """Per-pair difference vectors for one cell -> (M, n_layers, hidden) float32.

    This is the array the bootstrap needs and the thing people worry is not cached. It is
    not stored directly, but it is exactly recoverable: the pos and neg files are keyed by
    the same question ids in the same order, so delta[q] = pos[q] - neg[q].
    """
    pos_path = act_dir / f"{persona}_{trait}_pos.pt"
    neg_path = act_dir / f"{persona}_{trait}_neg.pt"
    for path in (pos_path, neg_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing activations: {path}")

    pos = torch.load(pos_path, map_location="cpu", weights_only=True)
    neg = torch.load(neg_path, map_location="cpu", weights_only=True)

    kp = sorted(pos.keys(), key=lambda k: int(k[1:]))
    kn = sorted(neg.keys(), key=lambda k: int(k[1:]))
    if kp != kn:
        raise ValueError(f"{persona}/{trait}: pos and neg question sets differ -- "
                         "pairing is invalid, per-pair deltas are not recoverable")

    p = torch.stack([pos[k] for k in kp]).float().numpy()
    n = torch.stack([neg[k] for k in kn]).float().numpy()
    return p - n


def weighted_vectors(delta_flat: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Many CAA vectors at once: (R, M) weights @ (M, D) deltas -> (R, D).

    A bootstrap replicate's mean is a weighted mean with integer counts, so the whole set
    of replicates is one BLAS matmul. Materialising delta[idx] per replicate instead would
    allocate an (M, D) array each time -- ~262MB at M=500 -- and is what makes the naive
    version of this analysis feel expensive when it is not.
    """
    return W @ delta_flat


def pairwise_cos_by_layer(V: np.ndarray, n_layers: int, hidden: int) -> np.ndarray:
    """Mean pairwise cosine among R vectors, per layer -> (n_layers,) plus the raw values.

    Returns the full (n_layers, n_pairs) array so percentiles are available downstream.
    """
    R = V.shape[0]
    Vl = V.reshape(R, n_layers, hidden)
    norms = np.linalg.norm(Vl, axis=-1, keepdims=True)
    Vn = Vl / np.maximum(norms, 1e-8)
    iu = np.triu_indices(R, k=1)
    out = np.empty((n_layers, len(iu[0])), dtype=np.float64)
    for L in range(n_layers):
        G = Vn[:, L, :] @ Vn[:, L, :].T
        out[L] = G[iu]
    return out


def summarise(vals: np.ndarray) -> dict:
    """(n_layers, n_samples) -> mean/lo/hi lists, one entry per layer.

    nan-aware because the Spearman-Brown series is deliberately NaN below r=0.
    """
    with np.errstate(invalid="ignore"):
        return {
            "mean": np.nanmean(vals, axis=1).tolist(),
            "lo": np.nanpercentile(vals, 2.5, axis=1).tolist(),
            "hi": np.nanpercentile(vals, 97.5, axis=1).tolist(),
        }


def cosine_rows(a: np.ndarray, b: np.ndarray, n_layers: int, hidden: int) -> np.ndarray:
    """Row-wise cosine between two (R, D) stacks -> (n_layers, R)."""
    al = a.reshape(-1, n_layers, hidden)
    bl = b.reshape(-1, n_layers, hidden)
    num = (al * bl).sum(-1)
    den = np.linalg.norm(al, axis=-1) * np.linalg.norm(bl, axis=-1)
    return (num / np.maximum(den, 1e-8)).T


def spearman_brown(r: np.ndarray) -> np.ndarray:
    """Project a half-length reliability up to full length: 2r / (1 + r).

    Heuristic here, not exact: SB is derived for correlations between parallel test halves,
    and a cosine between two mean-difference vectors is not literally that. It is the right
    direction and roughly the right size, and it is reported ALONGSIDE the raw value rather
    than instead of it, so nothing downstream depends on the approximation.

    Undefined for r <= 0: the formula has a pole at r = -1 and returns large negative
    numbers just above it, which is meaningless rather than merely imprecise. Early layers
    genuinely do sit near zero (no trait signal yet), so this is a real case, not an edge
    case -- return NaN there so a reader cannot mistake -151 for a measurement.
    """
    return np.where(r > 0, 2.0 * r / (1.0 + np.where(r > 0, r, 0.0)), np.nan)


def sb_from_summary(s: dict) -> dict:
    """Spearman-Brown applied to an already-summarised split-half series.

    Deliberately NOT mean(SB(r)) over replicates. Two reasons, and the second one bit:
    SB is concave, so averaging per-replicate understates it (Jensen); and the split-half
    distribution here is wide enough to put real mass below zero (politician at L20 runs
    [-0.14, 0.92]), so NaN-guarding per replicate silently DROPS the low tail and biases
    the mean UP -- a worse error than the one the guard was added to fix.

    SB is monotone increasing on r > -1, so transforming the summary quantiles is exact for
    lo/hi. The reported centre is SB(mean r), not mean SB(r); labelled as such because the
    two differ and the difference is not negligible at these spreads.
    """
    return {k: spearman_brown(np.asarray(v)).tolist() for k, v in s.items()}


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
    if not traits:
        print(f"error: no traits found in {act_dir}", file=sys.stderr)
        return 2

    # Strip a known trait suffix rather than splitting on "_" -- trait names are not
    # single tokens (risk_taking), so rsplit would silently mangle every cell name.
    def cells_for(trait: str) -> list[str]:
        n = len(f"_{trait}_pos.pt")
        return sorted(p.name[:-n] for p in act_dir.glob(f"*_{trait}_pos.pt"))

    rng = np.random.default_rng(args.seed)
    results: dict = {
        "model": args.model,
        "n_boot": args.n_boot,
        "n_splits": args.n_splits,
        "estimator_notes": {
            "within_cell": "K/D B.1 rung 1: mean pairwise cosine among n_boot paired "
                           "bootstrap replicates of one cell",
            "split_half": "cosine between vectors from disjoint halves of the question bank",
            "split_half_sb": "split_half, Spearman-Brown corrected to full bank size",
            "boot_half": "cosine between two independent size-M/2 with-replacement draws",
            "across_cell": "K/D B.1 rung 3: mean pairwise cosine among full-sample persona "
                           "vectors (excludes null and nonsense)",
        },
        "traits": {},
    }

    for trait in traits:
        print(f"\n{'=' * 78}\nTRAIT: {trait}\n{'=' * 78}")
        trait_res: dict = {"cells": {}}
        full_vectors: dict[str, np.ndarray] = {}
        n_layers = hidden = n_q = None

        for cell in (args.cells or cells_for(trait)):
            try:
                delta = load_delta(act_dir, cell, trait)
            except FileNotFoundError:
                print(f"  skip {cell}: no activations for this trait")
                continue

            n_q, n_layers, hidden = delta.shape
            D = n_layers * hidden
            flat = delta.reshape(n_q, D)
            half = n_q // 2

            # ---- rung 1: paired bootstrap replicates of this cell ----
            W = np.zeros((args.n_boot, n_q), dtype=np.float32)
            for b in range(args.n_boot):
                idx = rng.integers(0, n_q, size=n_q)
                W[b] = np.bincount(idx, minlength=n_q)
            W /= n_q
            V_boot = weighted_vectors(flat, W)
            within = pairwise_cos_by_layer(V_boot, n_layers, hidden)

            # ---- split-half and its size-matched bootstrap control ----
            Wa = np.zeros((args.n_splits, n_q), dtype=np.float32)
            Wb = np.zeros((args.n_splits, n_q), dtype=np.float32)
            Ha = np.zeros((args.n_splits, n_q), dtype=np.float32)
            Hb = np.zeros((args.n_splits, n_q), dtype=np.float32)
            for r in range(args.n_splits):
                perm = rng.permutation(n_q)
                a_idx, b_idx = perm[:half], perm[half:2 * half]
                Wa[r, a_idx] = 1.0 / half
                Wb[r, b_idx] = 1.0 / half
                # same size, but drawn WITH replacement from the whole bank, so the two
                # draws overlap -- the difference from the disjoint split is the part
                # attributable to question-set variance rather than to sample size
                Ha[r] = np.bincount(rng.integers(0, n_q, size=half), minlength=n_q) / half
                Hb[r] = np.bincount(rng.integers(0, n_q, size=half), minlength=n_q) / half

            sh = cosine_rows(weighted_vectors(flat, Wa), weighted_vectors(flat, Wb),
                             n_layers, hidden)
            bh = cosine_rows(weighted_vectors(flat, Ha), weighted_vectors(flat, Hb),
                             n_layers, hidden)

            full_vectors[cell] = flat.mean(0)

            sh_summary = summarise(sh)
            trait_res["cells"][cell] = {
                "within_cell": summarise(within),
                "split_half": sh_summary,
                "split_half_sb": sb_from_summary(sh_summary),
                "boot_half": summarise(bh),
            }
            print(f"  {cell:<22} within={within.mean(1)[args.report_layers[0]]:.4f} "
                  f"split_half={sh.mean(1)[args.report_layers[0]]:.4f} "
                  f"(L{args.report_layers[0]})")

        # ---- rung 3, same functional form as rung 1 ----
        personas = [c for c in full_vectors if c not in (NULL_SLUG, CONTROL_SLUG)]
        if len(personas) >= 2:
            V_pers = np.stack([full_vectors[p] for p in personas])
            across = pairwise_cos_by_layer(V_pers, n_layers, hidden)
            trait_res["across_cell"] = summarise(across)
            trait_res["across_cell_personas"] = personas

        trait_res["n_questions"] = int(n_q) if n_q is not None else None
        trait_res["n_layers"] = int(n_layers) if n_layers is not None else None
        results["traits"][trait] = trait_res

    out_path = out_dir / "caa_within_cell_stability.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # ---- summary table ----
    for L in args.report_layers:
        print(f"\n{'=' * 96}\nLAYER {L}\n{'=' * 96}")
        print(f"{'trait':<15} {'within(r1)':>11} {'boot_half':>10} {'split_half':>11} "
              f"{'sh_SB':>8} {'across(r3)':>11} {'r1-r3 gap':>10}")
        print("-" * 96)
        for trait, tr in results["traits"].items():
            cs = tr["cells"]
            if not cs:
                continue
            pers = [c for c in cs if c not in (NULL_SLUG, CONTROL_SLUG)]
            w = float(np.mean([cs[c]["within_cell"]["mean"][L] for c in pers]))
            bh = float(np.mean([cs[c]["boot_half"]["mean"][L] for c in pers]))
            sh = float(np.mean([cs[c]["split_half"]["mean"][L] for c in pers]))
            sb = float(np.mean([cs[c]["split_half_sb"]["mean"][L] for c in pers]))
            ac = tr.get("across_cell", {}).get("mean", [float("nan")] * (L + 1))[L]
            print(f"{trait:<15} {w:>11.4f} {bh:>10.4f} {sh:>11.4f} {sb:>8.4f} "
                  f"{ac:>11.4f} {w - ac:>10.4f}")
        print("-" * 96)
        print("within(r1): K/D B.1 rung 1, their estimator.  across(r3): rung 3, same form.")
        print("split_half < boot_half is question-set variance the bootstrap cannot see.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
