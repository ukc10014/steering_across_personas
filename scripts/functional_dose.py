#!/usr/bin/env python3
"""Functional perturbation size of each arm, measured in ACTIVATION space.

WHY THIS EXISTS. scripts/adapter_dose.py showed the three OCT adapters are within 2.6% on
weight-space ||dW||_F. It is tempting to read that as "intervention magnitude is ruled out
as a confound". It is not, and the earlier write-up said so in one sentence and then
contradicted it in another. Weight norm is not functional dose: two LoRAs with identical
||dW||_F can move activations by very different amounts, because what matters is how the
model's actual activations align with the update directions.

The repo already carried evidence pointing the other way -- docs/results/
llama31_8b_character_arms.md records ||d||/||v|| of 0.870 for impulsiveness against 0.709
and 0.676 for the others.

This measures displacement directly on the data being analysed, from the cached
activations, with no GPU and no new inference:

    per cell   ||V_arm - V_base|| / ||V_base||          (trait-vector displacement)
    per token  mean_q ||h_arm(x_q) - h_base(x_q)|| / mean_q ||h_base(x_q)||
                                                        (raw answer-token displacement)

The second is the more direct answer to "does this arm move the representation more on
these inputs", since it does not first collapse questions into a contrast.

Both condition on the CAA prompts, not a neutral corpus, so they describe functional dose
ON THIS DATA. That is the relevant quantity for interpreting these geometry results, and
it is NOT a substitute for a neutral-corpus KL if the question is about the model overall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from persona_steering.config import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=str(OUTPUTS_DIR / "_qcache"))
    p.add_argument("--cache-layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--arms", nargs="+",
                   default=["goodness", "mathematical", "impulsiveness", "misalignment"])
    p.add_argument("--out", type=str,
                   default=str(OUTPUTS_DIR / "analysis" / "functional_dose.json"))
    return p.parse_args()


def load(cache: Path, arm: str, cache_layers, layer):
    tag = "_".join(str(l) for l in cache_layers)
    z = np.load(cache / f"{arm}_L{tag}.npz", allow_pickle=False)
    li = list(z["layers"]).index(layer)
    return (z["acts"][:, :, :, :, li, :], [str(t) for t in z["traits"]],
            [str(p) for p in z["personas"]], [str(p) for p in z["semantic_personas"]],
            z["n_questions_per_trait"])


def main() -> None:
    a = parse_args()
    cache = Path(a.cache_dir)
    base, traits, personas, semantic, nqs = load(cache, "base", a.cache_layers, a.layer)
    pidx = [personas.index(p) for p in semantic]
    res = {}

    print(f"layer {a.layer}; {len(traits)} traits x {len(semantic)} personas\n")
    print(f"{'arm':16s} {'trait-vector displ.':>20s} {'answer-token displ.':>21s}")
    for arm in a.arms:
        acts, *_ = load(cache, arm, a.cache_layers, a.layer)
        vdisp, hdisp = [], []
        for ti in range(len(traits)):
            nq = int(nqs[ti])
            for pi in pidx:
                b = base[ti, pi, :, :nq].astype(np.float32)     # (2, nq, H)
                m = acts[ti, pi, :, :nq].astype(np.float32)
                vb = b[0].mean(0) - b[1].mean(0)
                vm = m[0].mean(0) - m[1].mean(0)
                vdisp.append(np.linalg.norm(vm - vb) / max(np.linalg.norm(vb), 1e-9))
                for d in (0, 1):
                    num = np.linalg.norm(m[d] - b[d], axis=-1).mean()
                    den = np.linalg.norm(b[d], axis=-1).mean()
                    hdisp.append(num / max(den, 1e-9))
        v, h = float(np.mean(vdisp)), float(np.mean(hdisp))
        res[arm] = {"trait_vector_displacement": v,
                    "trait_vector_displacement_sd": float(np.std(vdisp)),
                    "answer_token_displacement": h,
                    "answer_token_displacement_sd": float(np.std(hdisp))}
        print(f"{arm:16s} {v:20.4f} {h:21.4f}")

    ref = a.arms[0]
    print(f"\nrelative to {ref}:")
    for arm in a.arms:
        print(f"  {arm:16s} trait-vector {res[arm]['trait_vector_displacement']/res[ref]['trait_vector_displacement']:.3f}x"
              f"   answer-token {res[arm]['answer_token_displacement']/res[ref]['answer_token_displacement']:.3f}x")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(a.layer)] = res
    out.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
