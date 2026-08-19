#!/usr/bin/env python3
"""Does the attention-mask fix actually move the CAA trait vectors?

Analysis half of the diagnostic; consumes what scripts/mask_diag_extract.py wrote.

THE MEASUREMENT PROBLEM, AND WHY A RAW COSINE IS NOT THE ANSWER.
The tempting criterion is "cos(V_old, V_fixed) > 0.99, therefore the bug is harmless".
That criterion is close to uninformative here, and commit d44a267 retracted a headline
result for exactly this reason: these vectors carry a large component common to every
persona, and a cosine between two vectors that both contain the same large component is
high almost regardless of what happened to the part we care about. So every comparison
below is reported twice:

    raw      -- on the trait vectors as they are
    centred  -- after subtracting the persona mean within each (arm, trait), which is
                where the persona-conditional signal actually lives

CALIBRATION. A cosine of 0.99 means nothing without knowing what cosine two innocent
re-estimates of the same vector would achieve. So we compare the mask effect against a
question-resampling floor:

    mask effect : cos(V_legacy, V_fixed) on the SAME questions -- question noise is
                  common to both sides and cancels, isolating the mask
    noise floor : cos(V_legacy^b1, V_legacy^b2) on two INDEPENDENT bootstrap resamples --
                  same mask, so this is what question sampling alone costs

If the mask effect sits below the noise floor, changing the mask perturbs the vector more
than throwing away and redrawing every question does. That is the finding that would
condemn the archived activations.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


def load_cell(d: Path, persona: str, trait: str, direction: str, layers: list[int]) -> np.ndarray:
    """-> (n_questions, n_layers, hidden)"""
    acts = torch.load(d / f"{persona}_{trait}_{direction}.pt")
    keys = sorted(acts, key=lambda k: int(k[1:]))
    return torch.stack([acts[k][layers] for k in keys]).float().numpy()


def trait_vector(pos: np.ndarray, neg: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    """CAA vector = mean(pos) - mean(neg); idx selects a question resample."""
    if idx is None:
        return pos.mean(0) - neg.mean(0)
    return pos[idx].mean(0) - neg[idx].mean(0)


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine along the last axis."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=str, default="outputs/_mask_diag")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/analysis/mask_diag.json")
    args = ap.parse_args()

    root = Path(args.root)
    cfg = json.loads((root / "config.json").read_text())
    arms, traits = cfg["arms"], cfg["traits"]
    personas = cfg["personas"]
    semantic = [p for p in personas if p != "null"]
    rng = np.random.default_rng(args.seed)
    L = args.layers

    # ---- load every cell once -------------------------------------------------------
    # V[arm][mode][trait][persona] -> (n_layers, hidden);  raw per-question arrays kept
    # for the bootstrap.
    per_q: dict = {}
    V: dict = {}
    for arm in arms:
        for mode in ("legacy", "fixed"):
            d = root / arm / mode
            for t in traits:
                for p in personas:
                    pos = load_cell(d, p, t, "pos", L)
                    neg = load_cell(d, p, t, "neg", L)
                    per_q[(arm, mode, t, p)] = (pos, neg)
                    V[(arm, mode, t, p)] = trait_vector(pos, neg)
    nq = per_q[(arms[0], "legacy", traits[0], personas[0])][0].shape[0]
    print(f"loaded {len(V)} cells, {nq} questions, layers {L}\n")

    def centred(arm: str, mode: str, t: str) -> dict[str, np.ndarray]:
        """Persona-centred trait vectors: removes the component common to all personas."""
        M = np.mean([V[(arm, mode, t, p)] for p in semantic], axis=0)
        return {p: V[(arm, mode, t, p)] - M for p in semantic}

    results: dict = {"config": cfg, "layers": L, "n_questions": nq, "cells": {}}

    # ---- 1. mask effect on the trait vectors ----------------------------------------
    print("=" * 78)
    print("1. MASK EFFECT: cos(V_legacy, V_fixed), same questions")
    print("=" * 78)
    for arm in arms:
        for t in traits:
            cen_l, cen_f = centred(arm, "legacy", t), centred(arm, "fixed", t)
            for p in personas:
                vl, vf = V[(arm, "legacy", t, p)], V[(arm, "fixed", t, p)]
                c_raw = cos(vl, vf)
                rel = np.linalg.norm(vf - vl, axis=-1) / np.linalg.norm(vl, axis=-1)
                row = {"raw_cos": c_raw.tolist(), "rel_norm_change": rel.tolist()}
                if p in semantic:
                    row["centred_cos"] = cos(cen_l[p], cen_f[p]).tolist()
                results["cells"][f"{arm}|{t}|{p}"] = row
                cen_s = ("  centred " +
                         " ".join(f"L{l}={c:.4f}" for l, c in zip(L, row["centred_cos"]))
                         if p in semantic else "  centred n/a (null)")
                print(f"{arm:14s} {t:12s} {p:15s} "
                      + " ".join(f"L{l} raw={c:.4f}" for l, c in zip(L, c_raw))
                      + cen_s
                      + "  relnorm=" + " ".join(f"{r:.3f}" for r in rel))

    # ---- 2. question-resampling noise floor -----------------------------------------
    print("\n" + "=" * 78)
    print(f"2. NOISE FLOOR: cos of two independent question resamples, same mask "
          f"(B={args.bootstrap})")
    print("=" * 78)
    floor: dict = {}
    for arm in arms:
        for t in traits:
            for p in personas:
                pos, neg = per_q[(arm, "legacy", t, p)]
                raws, cens = [], []
                Ms = {}
                for b in range(args.bootstrap):
                    i1 = rng.integers(0, nq, nq)
                    i2 = rng.integers(0, nq, nq)
                    v1 = trait_vector(pos, neg, i1)
                    v2 = trait_vector(pos, neg, i2)
                    raws.append(cos(v1, v2))
                    Ms[b] = (i1, i2)
                raws = np.array(raws)
                floor[(arm, t, p)] = raws
                print(f"{arm:14s} {t:12s} {p:15s} "
                      + " ".join(
                          f"L{l}: {np.median(raws[:, j]):.4f} "
                          f"[{np.percentile(raws[:, j], 2.5):.4f},"
                          f"{np.percentile(raws[:, j], 97.5):.4f}]"
                          for j, l in enumerate(L)))

    # ---- 3. verdict per cell ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. VERDICT: is the mask effect larger than question noise?")
    print("   'MASK WORSE' = cos(legacy,fixed) below the 2.5th pct of the noise floor")
    print("=" * 78)
    verdicts = {}
    for arm in arms:
        for t in traits:
            for p in personas:
                eff = np.array(results["cells"][f"{arm}|{t}|{p}"]["raw_cos"])
                fl = floor[(arm, t, p)]
                for j, l in enumerate(L):
                    lo = np.percentile(fl[:, j], 2.5)
                    worse = eff[j] < lo
                    verdicts[f"{arm}|{t}|{p}|L{l}"] = {
                        "mask_effect_cos": float(eff[j]),
                        "noise_floor_p2.5": float(lo),
                        "noise_floor_median": float(np.median(fl[:, j])),
                        "mask_worse_than_noise": bool(worse),
                    }
                    print(f"{arm:14s} {t:12s} {p:15s} L{l}: "
                          f"effect={eff[j]:.4f} vs floor_p2.5={lo:.4f}  "
                          f"{'MASK WORSE' if worse else 'within noise'}")
    results["verdicts"] = verdicts

    n_worse = sum(v["mask_worse_than_noise"] for v in verdicts.values())
    print(f"\n{n_worse}/{len(verdicts)} cells where the mask change exceeds question noise")

    # ---- 4. downstream statistic: raw cosine-to-null ---------------------------------
    print("\n" + "=" * 78)
    print("4. DOWNSTREAM: raw cosine-to-null, legacy vs fixed")
    print("=" * 78)
    ctn = {}
    for arm in arms:
        for t in traits:
            for mode in ("legacy", "fixed"):
                vn = V[(arm, mode, t, "null")]
                for p in semantic:
                    ctn[(arm, t, mode, p)] = cos(V[(arm, mode, t, p)], vn)
            for j, l in enumerate(L):
                lg = np.array([ctn[(arm, t, "legacy", p)][j] for p in semantic])
                fx = np.array([ctn[(arm, t, "fixed", p)][j] for p in semantic])
                print(f"{arm:14s} {t:12s} L{l}:")
                for p, a, b in zip(semantic, lg, fx):
                    print(f"      {p:16s} legacy={a:+.4f}  fixed={b:+.4f}  d={b-a:+.4f}")
                print(f"      {'persona mean':16s} legacy={lg.mean():+.4f}  "
                      f"fixed={fx.mean():+.4f}  d={fx.mean()-lg.mean():+.4f}")
                order_l = [semantic[i] for i in np.argsort(-lg)]
                order_f = [semantic[i] for i in np.argsort(-fx)]
                print(f"      ordering legacy: {order_l}")
                print(f"      ordering fixed : {order_f}"
                      f"   {'SAME' if order_l == order_f else 'CHANGED'}")
                results.setdefault("cos_to_null", {})[f"{arm}|{t}|L{l}"] = {
                    "legacy": dict(zip(semantic, lg.tolist())),
                    "fixed": dict(zip(semantic, fx.tolist())),
                    "order_legacy": order_l, "order_fixed": order_f,
                    "order_preserved": order_l == order_f,
                }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
