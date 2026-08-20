#!/usr/bin/env python3
"""Read functional dose off the calibration grid and report it relative to a target arm.

Same two measures as scripts/functional_dose.py, computed the same way, so the numbers are
comparable to the full-cache table -- that comparability is the point, since the small grid
is only useful if its s=1 configs recover the full-run ratios.

    trait-vector   ||V_arm - V_base|| / ||V_base||       V = mean(pos) - mean(neg)
    answer-token   mean_q||h_arm - h_base|| / mean_q||h_base||

The second is the more independent measure: it does not first collapse questions into a
contrast, so it is not dominated by whatever the contrast happens to cancel.

Usage:
    python scripts/dose_calibrate_analyse.py --layers 15 20 --ref goodness_s1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def cell(d: Path, persona: str, trait: str, direction: str, layer: int) -> np.ndarray:
    acts = torch.load(d / f"{persona}_{trait}_{direction}.pt", map_location="cpu")
    keys = sorted(acts, key=lambda k: int(k[1:]))
    return np.stack([acts[k][layer].float().numpy() for k in keys])   # (nq, H)


def dose(base: Path, arm: Path, grid: dict, layer: int) -> dict:
    """Both dose measures, plus a decomposition of the answer-token one.

    The two measures disagree by a lot on the archived arms -- trait-vector dose spans
    0.94x-1.37x across the four while answer-token dose spans only 0.99x-1.08x -- so they
    cannot both be matched by one scale, and picking between them needs to be a decision
    rather than an accident. The decomposition says why they differ: the displacement
    h_arm - h_base is split into the part COMMON to all questions in a cell (a bias shift,
    which cancels out of the pos-neg contrast when it is the same in both directions) and
    the part that VARIES with the question. If the common part dominates, answer-token dose
    is mostly measuring a constant offset that the trait vector never sees.
    """
    vd, hd, cd, rd = [], [], [], []
    for t in grid["traits"]:
        for p in grid["personas"]:
            b = {dr: cell(base, p, t, dr, layer) for dr in ("pos", "neg")}
            m = {dr: cell(arm, p, t, dr, layer) for dr in ("pos", "neg")}
            vb = b["pos"].mean(0) - b["neg"].mean(0)
            vm = m["pos"].mean(0) - m["neg"].mean(0)
            vd.append(np.linalg.norm(vm - vb) / max(np.linalg.norm(vb), 1e-9))
            for dr in ("pos", "neg"):
                den = max(np.linalg.norm(b[dr], axis=-1).mean(), 1e-9)
                D = m[dr] - b[dr]
                mu = D.mean(0)
                hd.append(np.linalg.norm(D, axis=-1).mean() / den)
                cd.append(np.linalg.norm(mu) / den)
                rd.append(np.linalg.norm(D - mu, axis=-1).mean() / den)
    return {"trait_vector": float(np.mean(vd)), "trait_vector_sd": float(np.std(vd)),
            "answer_token": float(np.mean(hd)), "answer_token_sd": float(np.std(hd)),
            "answer_token_common": float(np.mean(cd)),
            "answer_token_varying": float(np.mean(rd)),
            "n_cells": len(vd)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=str, default="outputs/_dose_calib")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--ref", type=str, default="goodness_s1",
                    help="config directory whose dose is the match target")
    ap.add_argument("--out", type=str, default="outputs/analysis/dose_calibration.json")
    args = ap.parse_args()

    root = Path(args.root)
    grid = json.loads((root / "grid.json").read_text())
    base = root / "base"
    if not base.exists():
        raise SystemExit(f"no base config in {root}: dose is measured against it")
    arms = sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name != "base"
                  and (d / f"{grid['personas'][0]}_{grid['traits'][0]}_pos.pt").exists())

    res: dict[str, dict] = {}
    for layer in args.layers:
        print(f"\n=== layer {layer} "
              f"({len(grid['traits'])} traits x {len(grid['personas'])} personas x "
              f"{grid['max_questions']} questions) ===")
        per = {a: dose(base, root / a, grid, layer) for a in arms}
        res[str(layer)] = per
        ref = per.get(args.ref)
        print(f"{'config':22s} {'trait-vec':>10s} {'vs ref':>8s} "
              f"{'ans-tok':>10s} {'vs ref':>8s} {'at-vary':>9s} {'vs ref':>8s}")
        for a in arms:
            r = per[a]
            def rel(k):
                return f"{r[k]/ref[k]:.3f}x" if ref and ref[k] else "-"
            print(f"{a:22s} {r['trait_vector']:10.4f} {rel('trait_vector'):>8s} "
                  f"{r['answer_token']:10.4f} {rel('answer_token'):>8s} "
                  f"{r['answer_token_varying']:9.4f} {rel('answer_token_varying'):>8s}")
        if ref:
            print(f"\n  answer-token displacement split (common bias shift / "
                  f"question-varying):")
            for a in arms:
                r = per[a]
                print(f"    {a:22s} common {r['answer_token_common']:.4f}  "
                      f"varying {r['answer_token_varying']:.4f}")
        if not ref:
            print(f"  (no --ref config {args.ref!r} present, so no ratios)")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"grid": grid, "ref": args.ref, "dose": res}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
