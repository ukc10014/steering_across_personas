#!/usr/bin/env python3
"""Hidden-state displacement at L15/L20 on a small CAA subset, against the archived base.

WHY THIS EXISTS, AND WHY IT IS NOT THE KL MEASUREMENT.
`scripts/neutral_dose.py` found that a weight-norm-matched random LoRA moves the model's
OUTPUT distribution almost not at all (mean KL 0.001 against goodness's 0.606). It is
tempting to conclude that such an arm is therefore indistinguishable from base in the
geometry too, and skip its extraction. That does not follow. Output KL measures what
survives the remaining seventeen layers; the dependent variables in this project are
hidden states at layers 15 and 20. A perturbation can move h_15 appreciably and still be
almost invisible at the logits if later layers compensate. The two would then disagree in
the same way the repo's two existing dose measures already disagree -- which is precisely
why the previous experiment abandoned a matched point for a ladder.

So this measures the quantity the geometry actually depends on, before any full 192-cell
extraction is committed. It computes exactly the statistics `functional_dose.py` reports,
so the numbers are directly comparable to the published dose axis:

    trait-vector displacement   ||V_arm - V_base|| / ||V_base||,   V = mean(pos) - mean(neg)
    answer-token displacement   mean_q ||h_arm - h_base|| / mean_q ||h_base||

It differs from `functional_dose.py` only in reading .pt cells directly rather than a
question cache, so it works on a subset of the grid that no cache has been built for.

The subset is small (a few personas x traits) and the per-cell SD on the full grid is about
6% of the mean, so a dozen cells locate an arm on the dose axis to well within the spacing
of the rungs being chosen. It is a siting measurement, not a result.

The probe cells MUST be extracted with --legacy-mask, because the archived base they are
differenced against was.

USAGE
    python scripts/activation_dose_probe.py --probe-root outputs/_actprobe \
        --arms random_iid_s1 random_iid_s16 random_iid_s24 \
               random_perm_s1 random_perm_s16 random_perm_s24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from persona_steering.config import OUTPUTS_DIR

BASE_DIR = OUTPUTS_DIR / "Llama-3.1-8B-Instruct" / "caa_activations"


def load_cell(d: Path, persona: str, trait: str, direction: str, layer: int) -> np.ndarray:
    """(n_questions, hidden) answer-token activations for one cell at one layer."""
    obj = torch.load(d / f"{persona}_{trait}_{direction}.pt", map_location="cpu",
                     weights_only=False)
    keys = sorted(obj, key=lambda k: int(k[1:]))
    return np.stack([obj[k][layer].float().numpy() for k in keys])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-root", default=str(OUTPUTS_DIR / "_actprobe"))
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    p.add_argument("--personas", nargs="+",
                   default=["farmer", "therapist", "tech_ceo"])
    p.add_argument("--traits", nargs="+", default=["assertiveness", "impulsivity"])
    p.add_argument("--out", default=str(OUTPUTS_DIR / "analysis" / "activation_dose_probe.json"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = Path(a.probe_root)
    results: dict[str, dict] = {}

    for layer in a.layers:
        # Base is loaded once per layer and reused across arms: it is the same archived
        # cells every arm is differenced against, and re-reading it per arm would dominate
        # the runtime of what is meant to be the cheap check.
        base: dict[tuple[str, str, str], np.ndarray] = {}
        for t in a.traits:
            for p in a.personas:
                for d in ("pos", "neg"):
                    base[(t, p, d)] = load_cell(BASE_DIR, p, t, d, layer)

        print(f"\n=== layer {layer} ===")
        print(f"{'arm':22s} {'trait-vector displ.':>20s} {'answer-token displ.':>21s}")
        for arm in a.arms:
            d_arm = root / arm / "caa_activations"
            if not d_arm.exists():
                print(f"{arm:22s} !! no cells at {d_arm}")
                continue
            vdisp, hdisp = [], []
            for t in a.traits:
                for p in a.personas:
                    b_pos, b_neg = base[(t, p, "pos")], base[(t, p, "neg")]
                    m_pos = load_cell(d_arm, p, t, "pos", layer)
                    m_neg = load_cell(d_arm, p, t, "neg", layer)
                    n = min(len(b_pos), len(m_pos))
                    vb = b_pos[:n].mean(0) - b_neg[:n].mean(0)
                    vm = m_pos[:n].mean(0) - m_neg[:n].mean(0)
                    vdisp.append(np.linalg.norm(vm - vb) / max(np.linalg.norm(vb), 1e-9))
                    for b, m in ((b_pos, m_pos), (b_neg, m_neg)):
                        n = min(len(b), len(m))
                        den = max(np.linalg.norm(b[:n], axis=-1).mean(), 1e-9)
                        hdisp.append(np.linalg.norm(m[:n] - b[:n], axis=-1).mean() / den)
            v, h = float(np.mean(vdisp)), float(np.mean(hdisp))
            results.setdefault(str(layer), {})[arm] = {
                "trait_vector_displacement": v,
                "trait_vector_displacement_sd": float(np.std(vdisp)),
                "answer_token_displacement": h,
                "answer_token_displacement_sd": float(np.std(hdisp)),
                "n_cells": len(vdisp)}
            print(f"{arm:22s} {v:20.4f} {h:21.4f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"personas": a.personas, "traits": a.traits,
                               "layers": a.layers, "results": results}, indent=2))
    print(f"\nwrote {out}")
    print("\nFor reference, the full-grid answer-token displacement at L15 "
          "(outputs/analysis/functional_dose_ladder.json):")
    print("  goodness 0.5357   mathematical 0.5287   impulsiveness 0.5733   "
          "misalignment 0.5771")


if __name__ == "__main__":
    main()
