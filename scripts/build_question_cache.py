#!/usr/bin/env python3
"""Cache per-question CAA activations at selected layers, one compact array per arm.

WHY THIS EXISTS. Every analysis in the geometry suite (dispersion, RDMs, Procrustes) needs
a bootstrap over the CAA questions, and therefore needs per-question activations rather
than the cell means that scripts/caa_shared_component.py keeps. Reading them from the
archived .pt files on every run is prohibitive: each file is 500 x 32 x 4096 fp16 (131 MB)
and there are 768 of them across four arms, ~100 GB of network-volume I/O.

Slicing to the two layers anyone actually analyses turns that into ~1.6 GB per arm, which
loads in seconds and lives comfortably in RAM. Read the .pt files once, pay the I/O once.

Layout per arm, saved as .npz:
    acts   float16 (n_traits, n_personas, 2, n_questions, n_layers, hidden)
                    direction axis is [pos, neg]
    plus traits / personas / layers / n_questions as metadata arrays.

Personas are ordered [<10 semantic>, null, nonsense] so that the semantic block is a
contiguous slice and null/nonsense stay addressable but excluded by default.

NOTE ON PROVENANCE: this only re-packs whatever activations are on disk. If those were
extracted with the pre-fix attention mask (see scripts/verify_attention_mask.py), the cache
inherits that, and so does everything computed from it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona_steering.config import OUTPUTS_DIR

PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
CONTROLS = ["null", "nonsense"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty", "confidence",
          "deference", "warmth", "impulsivity"]
ARMS = {"base": "Llama-3.1-8B-Instruct",
        "goodness": "llama-3.1-8b-goodness",
        "mathematical": "llama-3.1-8b-mathematical",
        "impulsiveness": "llama-3.1-8b-impulsiveness"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--traits", nargs="+", default=TRAITS)
    p.add_argument("--activations-root", type=str, default=None,
                   help="override; default is outputs/{arm_dir}/caa_activations")
    p.add_argument("--out-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    personas = PERSONAS + CONTROLS
    layers = list(args.layers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        tag = "_".join(str(l) for l in layers)
        out = out_dir / f"{arm}_L{tag}.npz"
        if out.exists() and not args.force:
            print(f"{arm}: exists, skipping ({out})")
            continue

        root = (Path(args.activations_root) if args.activations_root
                else OUTPUTS_DIR / ARMS[arm] / "caa_activations")
        if not root.exists():
            print(f"{arm}: !! no activations at {root}, skipping")
            continue

        print(f"\n=== {arm} <- {root} ===", flush=True)
        t0 = time.time()
        # probe one file for n_questions
        probe = torch.load(root / f"{personas[0]}_{args.traits[0]}_pos.pt",
                           map_location="cpu", weights_only=False)
        nq = len(probe)
        del probe

        acts = np.zeros((len(args.traits), len(personas), 2, nq, len(layers), 4096),
                        dtype=np.float16)
        n_short = 0
        for ti, trait in enumerate(args.traits):
            for pi, persona in enumerate(personas):
                for di, direction in enumerate(("pos", "neg")):
                    fp = root / f"{persona}_{trait}_{direction}.pt"
                    if not fp.exists():
                        raise FileNotFoundError(fp)
                    d = torch.load(fp, map_location="cpu", weights_only=False)
                    keys = sorted(d, key=lambda k: int(k[1:]))
                    a = torch.stack([d[k][layers] for k in keys]).numpy()
                    # empathy ships 499 questions, not 500; pad the tail and record it
                    if a.shape[0] < nq:
                        n_short += 1
                        pad = np.repeat(a[-1:], nq - a.shape[0], axis=0)
                        a = np.concatenate([a, pad], axis=0)
                    acts[ti, pi, di] = a[:nq]
                    del d
            print(f"  {trait:14s} {time.time()-t0:6.0f}s", flush=True)

        # per-trait true question counts, so the bootstrap never resamples padding
        nq_true = {}
        for ti, trait in enumerate(args.traits):
            d = torch.load(root / f"{personas[0]}_{trait}_pos.pt",
                           map_location="cpu", weights_only=False)
            nq_true[trait] = len(d); del d

        np.savez(out, acts=acts,
                 traits=np.array(args.traits), personas=np.array(personas),
                 layers=np.array(layers), n_questions=np.array(nq),
                 n_questions_per_trait=np.array([nq_true[t] for t in args.traits]),
                 semantic_personas=np.array(PERSONAS))
        print(f"{arm}: wrote {out} ({out.stat().st_size/1e9:.2f} GB) in "
              f"{time.time()-t0:.0f}s; {n_short} cells padded to {nq}")


if __name__ == "__main__":
    main()
