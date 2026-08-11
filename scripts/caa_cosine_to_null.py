#!/usr/bin/env python3
"""Cosine-to-null for CAA trait vectors, at every layer, with a bootstrap CI.

For each persona x trait, the CAA trait vector at a layer is
    v = mean(activations | positive answer) - mean(activations | negative answer)
and we report cos(v_persona, v_null) at that layer.

The claim under test (Karty/Davies): personas "fan out" -- their trait vectors
rotate away from the null-context trait vector -- while a length-matched nonsense
control stays near 1.0. Two reference lines matter for reading the numbers:

  * noise floor: cosine between two independent bootstrap resamples of NULL
    against itself. Cosine below 1.0 is partly just sampling noise; this says how
    much. Fan-out is only meaningful well below this line.
  * nonsense: a system prompt with no semantic content. Separates "any system
    prompt perturbs the vector" from "the persona's meaning perturbs the vector".

Bootstrap is UNPAIRED, and this matters. Within a replicate, the persona vector and
the null vector are built from two INDEPENDENT resamples of the questions. Pairing
them (same indices on both sides) would share the sampling noise between the two
vectors and inflate the cosine -- and it would make the noise floor degenerate,
since null-vs-null on identical indices is exactly 1.0 by construction. Unpaired
resampling puts the signal and the floor on the same footing, so they are directly
comparable.

Sanity check when reading output: a persona's cosine should not sit meaningfully
ABOVE the noise floor. If it does, the two are not being computed the same way.

Usage:
    python scripts/caa_cosine_to_null.py --model meta-llama/Llama-3.1-8B-Instruct \
        --traits deference warmth \
        --personas therapist drill_sergeant farmer nonsense \
        --headline-layer 15 --n-boot 50
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
from persona_steering.utils import activation_key_order, model_short_name

NULL_SLUG = "null"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosine-to-null across layers with bootstrap CI")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--personas", nargs="+", required=True,
                   help=f"Personas to compare against {NULL_SLUG!r} (do not include it here)")
    p.add_argument("--activations-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--headline-layer", type=int, default=15,
                   help="Pre-designated headline layer (default: 15)")
    p.add_argument("--n-boot", type=int, default=50, help="Bootstrap replicates (default: 50)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_direction(act_dir: Path, persona: str, trait: str, direction: str
                   ) -> tuple[np.ndarray, list[str]]:
    """Load one activation file -> (n_questions, n_layers, hidden) float32 + question keys."""
    path = act_dir / f"{persona}_{trait}_{direction}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing activations: {path}")
    d = torch.load(path, map_location="cpu")
    keys = sorted(d.keys(), key=activation_key_order)
    arr = torch.stack([d[k] for k in keys]).float().numpy()
    return arr, keys


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine over the last axis. a, b: (n_layers, hidden)."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-8)


def trait_vector(pos: np.ndarray, neg: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """CAA vector from a set of question indices: mean(pos) - mean(neg)."""
    return pos[idx].mean(0) - neg[idx].mean(0)


def boot_vectors(pos: np.ndarray, neg: np.ndarray, idx_list: list[np.ndarray]) -> np.ndarray:
    """All bootstrap replicate vectors at once -> (n_boot, n_layers, hidden).

    Same quantity as [trait_vector(pos, neg, i) for i in idx_list], computed as one matmul.

    A bootstrap replicate mean IS a weighted mean: resampling 500 questions with replacement
    and averaging is identical to weighting each original question by how many times it was
    drawn, over the same denominator. So the whole set of replicates is a single
    (n_boot x M) @ (M x n_layers*hidden) GEMM.

    This matters a lot here. The list-comprehension form calls pos[idx], a fancy-index gather
    that ALLOCATES a fresh (500, 32, 4096) float32 array -- 262 MB -- on every one of the 400
    replicates, for both directions, for every cell. At full grid that is ~22 TB of memory
    traffic per arm and it is not a BLAS op, so none of the machine's cores engage: measured
    ~90 min at 114% CPU on a 256-core box. As a GEMM the data is read once and OpenBLAS
    threads it.

    The same trick is used in caa_magnitude.py and caa_within_cell_stability.py; this script
    was the one that never got it.

    Callers must draw idx_list from the RNG exactly as before, so results stay reproducible
    against runs made with the old code path.
    """
    M, n_layers, hidden = pos.shape
    # Difference first: (pos - neg)[idx].mean(0) == pos[idx].mean(0) - neg[idx].mean(0),
    # and it halves the matmul as well as the resident array.
    delta = (pos - neg).reshape(M, -1)
    W = np.empty((len(idx_list), M), dtype=delta.dtype)
    for b, idx in enumerate(idx_list):
        W[b] = np.bincount(idx, minlength=M)
    W /= M
    return (W @ delta).reshape(len(idx_list), n_layers, hidden)


def main() -> int:
    args = parse_args()
    short = model_short_name(args.model)
    act_dir = Path(args.activations_dir) if args.activations_dir else OUTPUTS_DIR / short / "caa_activations"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if NULL_SLUG in args.personas:
        print(f"error: {NULL_SLUG!r} is the reference; remove it from --personas", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    results: dict = {"model": args.model, "headline_layer": args.headline_layer,
                     "n_boot": args.n_boot, "traits": {}}

    for trait in args.traits:
        print(f"\n{'=' * 78}\nTRAIT: {trait}\n{'=' * 78}")

        null_pos, null_keys = load_direction(act_dir, NULL_SLUG, trait, "pos")
        null_neg, _ = load_direction(act_dir, NULL_SLUG, trait, "neg")
        n_q, n_layers, _ = null_pos.shape
        print(f"{n_q} questions, {n_layers} layers")

        all_idx = np.arange(n_q)
        v_null_full = trait_vector(null_pos, null_neg, all_idx)

        # Two independent index draws per replicate. Side A feeds the persona vector
        # (and one null vector for the floor), side B feeds the reference null vector.
        # Reused across personas so their CIs are comparable to each other.
        boot_a = [rng.integers(0, n_q, size=n_q) for _ in range(args.n_boot)]
        boot_b = [rng.integers(0, n_q, size=n_q) for _ in range(args.n_boot)]

        # The null vector depends only on the draw, not on which persona it is being
        # compared against, so build it once per replicate and reuse. At full-grid size
        # (8 traits x 11 personas) this is the difference between minutes and tens of
        # minutes, since each trait_vector call materialises a 500 x 32 x 4096 array.
        null_boot_a = boot_vectors(null_pos, null_neg, boot_a)
        null_boot_b = boot_vectors(null_pos, null_neg, boot_b)

        # Noise floor: NULL against itself, using the SAME unpaired structure as the
        # persona comparison below, so the two are apples-to-apples.
        # cosine() reduces over the last axis, so it vectorises over the replicate axis
        # unchanged: (n_boot, n_layers, hidden) x2 -> (n_boot, n_layers).
        floor = cosine(null_boot_a, null_boot_b)  # (n_boot, n_layers)

        trait_res: dict = {
            "n_questions": int(n_q),
            "n_layers": int(n_layers),
            "noise_floor": {
                "mean": floor.mean(0).tolist(),
                "lo": np.percentile(floor, 2.5, axis=0).tolist(),
                "hi": np.percentile(floor, 97.5, axis=0).tolist(),
            },
            "personas": {},
        }

        for persona in args.personas:
            pos, keys = load_direction(act_dir, persona, trait, "pos")
            neg, _ = load_direction(act_dir, persona, trait, "neg")
            if keys != null_keys:
                print(f"error: question set mismatch between {persona} and {NULL_SLUG}", file=sys.stderr)
                return 2

            point = cosine(trait_vector(pos, neg, all_idx), v_null_full)

            boot = cosine(boot_vectors(pos, neg, boot_a), null_boot_b)  # (n_boot, n_layers)

            trait_res["personas"][persona] = {
                "point": point.tolist(),
                "boot_mean": boot.mean(0).tolist(),
                "lo": np.percentile(boot, 2.5, axis=0).tolist(),
                "hi": np.percentile(boot, 97.5, axis=0).tolist(),
            }

        results["traits"][trait] = trait_res

        # ---- per-layer table ----
        header = f"{'layer':>5}  {'noise_floor':>11}  " + "  ".join(f"{p:>22}" for p in args.personas)
        print("\n" + header)
        print("-" * len(header))
        for L in range(n_layers):
            row = f"{L:>5}  {floor.mean(0)[L]:>11.3f}  "
            cells = []
            for persona in args.personas:
                r = trait_res["personas"][persona]
                cells.append(f"{r['point'][L]:>6.3f} [{r['lo'][L]:.3f},{r['hi'][L]:.3f}]")
            marker = "  <-- headline" if L == args.headline_layer else ""
            print(row + "  ".join(f"{c:>22}" for c in cells) + marker)

        # ---- headline layer summary ----
        L = args.headline_layer
        print(f"\nHEADLINE (layer {L}):")
        print(f"  noise floor (null vs null): {floor.mean(0)[L]:.3f} "
              f"[{np.percentile(floor, 2.5, axis=0)[L]:.3f}, {np.percentile(floor, 97.5, axis=0)[L]:.3f}]")
        for persona in args.personas:
            r = trait_res["personas"][persona]
            print(f"  {persona:<18} cos={r['point'][L]:.3f} "
                  f"[{r['lo'][L]:.3f}, {r['hi'][L]:.3f}]")

    out_path = out_dir / "caa_cosine_to_null.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
