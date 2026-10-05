#!/usr/bin/env python3
"""S2 manipulation checks 1 and 2 (spec_sham_lora.md §3.1).

These ask ONLY "did optimisation occur?", never "did it produce the effect we are testing
for". Retention k, the fig4 cosine and the §10 contrast are outcomes (§3.2) and are not
computed here.

  Check 1 — the objective moved. Training-loss curve start-to-end, plus accuracy on the
            frozen sham preference labels, parsed from the run log.
  Check 2 — the weights moved non-trivially. ‖dW‖_F per module against the real DPO-stage
            comparator: overall ratio, the per-module distribution, and the Spearman of the
            per-module profile. Reported, never adjusted to.

Spec §2 pre-commits one threshold here: ">5x collapse in ‖dW‖ makes this a weak arm rather
than a clean one, and is reported as such rather than rescaled away silently."

dW = (alpha/r) * B @ A, materialised. The factors are not separately identifiable, so a
factor-space comparison can report differences between identical models.

Usage:
    python scripts/sham/s2_manipulation_checks.py \
        --sham /workspace/oct_rig/loras/llama-distillation-s2sham/impulsiveness \
        --real /workspace/oct_rig/loras_repro/llama-distillation/impulsiveness \
        --log  /workspace/oct_rig/logs/s2sham-123456_dpo.log \
        --out  outputs/analysis/s2_manipulation_checks.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

COLLAPSE_FACTOR = 5.0  # spec §2


def updates(d: Path) -> dict[str, torch.Tensor]:
    cfg = json.loads((d / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for k in sd:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb in sd:
            out[k.split(".lora_A")[0]] = scale * (sd[kb].float() @ sd[k].float())
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def parse_log(p: Path) -> dict:
    """Pull the loss/acc trajectory out of the tqdm postfix stream."""
    txt = p.read_text(errors="replace").replace("\r", "\n")
    pat = re.compile(r"loss=([0-9.]+).*?acc=([0-9.]+).*?chosen_reward=(-?[0-9.]+)"
                     r".*?reject_reward=(-?[0-9.]+)")
    rows = [(float(a), float(b), float(c), float(d)) for a, b, c, d in pat.findall(txt)]
    if not rows:
        return {"parsed": False}
    arr = np.array(rows)
    n = len(arr)
    head, tail = arr[: max(1, n // 20)], arr[-max(1, n // 20):]
    return {
        "parsed": True,
        "n_logged_steps": n,
        "loss_first_5pct_mean": float(head[:, 0].mean()),
        "loss_last_5pct_mean": float(tail[:, 0].mean()),
        "loss_delta": float(head[:, 0].mean() - tail[:, 0].mean()),
        "acc_first_5pct_mean": float(head[:, 1].mean()),
        "acc_last_5pct_mean": float(tail[:, 1].mean()),
        "acc_overall_mean": float(arr[:, 1].mean()),
        "chosen_reward_last": float(tail[:, 2].mean()),
        "reject_reward_last": float(tail[:, 3].mean()),
        "reward_margin_last": float(tail[:, 2].mean() - tail[:, 3].mean()),
        "reward_margin_first": float(head[:, 2].mean() - head[:, 3].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sham", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    res: dict = {"sham": str(a.sham), "real_comparator": str(a.real)}

    # ---- check 1 ---------------------------------------------------------------------
    if a.log and a.log.exists():
        res["check1_objective_moved"] = parse_log(a.log)
        c1 = res["check1_objective_moved"]
        print("CHECK 1 — the objective moved")
        if c1.get("parsed"):
            print(f"  loss   {c1['loss_first_5pct_mean']:.4f} -> {c1['loss_last_5pct_mean']:.4f}"
                  f"   (delta {c1['loss_delta']:+.4f})")
            print(f"  acc on frozen sham labels  {c1['acc_first_5pct_mean']:.4f} -> "
                  f"{c1['acc_last_5pct_mean']:.4f}  (overall mean {c1['acc_overall_mean']:.4f})")
            print(f"  reward margin  {c1['reward_margin_first']:+.4f} -> "
                  f"{c1['reward_margin_last']:+.4f}")
        else:
            print("  LOG NOT PARSED")
    else:
        print("CHECK 1 — skipped (no log given)")

    # ---- check 2 ---------------------------------------------------------------------
    S, R = updates(a.sham), updates(a.real)
    common = sorted(set(S) & set(R))
    if not common:
        raise SystemExit("no shared modules")
    ns = np.array([float(S[k].norm()) for k in common])
    nr = np.array([float(R[k].norm()) for k in common])
    tot_s = float(np.sqrt((ns ** 2).sum()))
    tot_r = float(np.sqrt((nr ** 2).sum()))
    ratio = tot_s / max(tot_r, 1e-12)
    per = ns / np.maximum(nr, 1e-12)
    collapsed = ratio < 1.0 / COLLAPSE_FACTOR

    dot = sum(float((S[k] * R[k]).sum()) for k in common)
    cos = dot / max(tot_s * tot_r, 1e-12)

    res["check2_weights_moved"] = {
        "n_modules": len(common),
        "sham_total_dW_F": tot_s,
        "real_total_dW_F": tot_r,
        "ratio_sham_over_real": ratio,
        "per_module_ratio": {"min": float(per.min()), "p25": float(np.percentile(per, 25)),
                             "median": float(np.median(per)), "p75": float(np.percentile(per, 75)),
                             "max": float(per.max())},
        "per_module_profile_spearman": spearman(ns, nr),
        "cos_dW_sham_real": cos,
        "collapse_threshold_factor": COLLAPSE_FACTOR,
        "collapsed_beyond_threshold": bool(collapsed),
    }
    print("\nCHECK 2 — the weights moved")
    print(f"  modules                 {len(common)}")
    print(f"  ||dW|| sham / real      {tot_s:.4f} / {tot_r:.4f}  =  {ratio:.4f}")
    print(f"  per-module ratio        min {per.min():.3f}  median {np.median(per):.3f}"
          f"  max {per.max():.3f}")
    print(f"  per-module profile rho  {spearman(ns, nr):.4f}")
    print(f"  cos(dW_sham, dW_real)   {cos:.4f}")
    print(f"  >{COLLAPSE_FACTOR:g}x collapse (spec §2)   "
          f"{'YES — report as a WEAK arm' if collapsed else 'no'}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
