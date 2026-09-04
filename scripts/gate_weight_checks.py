#!/usr/bin/env python3
"""Weight-space criteria for the seed-123456 reproduction gate: spec_sham_lora.md 6b A1-A3,
plus the 6c cross-term share measured on the REAL adapter pair.

CPU only, no forward passes. For each targeted module a LoRA adds

    dW = (alpha / r) * B @ A

A1  adapter_config: r, lora_alpha, target_modules, base model      exact match required
A2  overall ||dW||_F ratio repro/released                          pass [0.7,1.4]  stop outside [0.5,2.0]
A3  per-module ||dW||_F profile, Spearman across modules x layers  pass >= 0.80    stop < 0.6

6c  once both component adapters exist, the merged adapter is
        dW_merged = dW_dpo + 0.25*dW_sft + B_dpo@A_sft + B_sft@A_dpo
    and the cross-term share is measurable on the real pair for the first time.
    DESCRIPTIVE ONLY -- it does not reinterpret any published result.

    python scripts/gate_weight_checks.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

RELEASED = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
            "318b5f7e1428097a1a61d5f0ed205ee048b3f620/impulsiveness")
RIG = Path("/workspace/oct_rig")

KEY = re.compile(r"^(.*)\.lora_([AB])\.weight$")


def module_dW_norms(adapter_dir: str | Path) -> tuple[dict[str, float], dict]:
    """||dW||_F per module, and the adapter's config."""
    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(d / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = KEY.match(k)
        if not m:
            continue
        pairs.setdefault(m.group(1), {})[m.group(2)] = v.to(torch.float64)
    out = {}
    for mod, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        # ||dW||_F with dW = scaling * B @ A, formed one module at a time
        # fp64: ||B@A||_F in fp32 sums ~58M squared terms and comes out ~0.45% low
        # (checked against an fp64 reference and the exact low-rank trace identity).
        # Ratios of two such norms are unaffected -- the bias cancels -- so A2/A3 and
        # the 6c cross-term share as already reported stand; absolute norms shift ~0.45%.
        out[mod] = float(torch.linalg.matrix_norm(
            scaling * (ab["B"].double() @ ab["A"].double()), ord="fro"))
    return out, cfg


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def band(v: float, lo_pass: float, hi_pass: float, lo_stop: float, hi_stop: float) -> str:
    if lo_pass <= v <= hi_pass:
        return "PASS"
    if v < lo_stop or v > hi_stop:
        return "STOP"
    return "AMBIGUOUS"


def cross_term_share(dpo_dir: Path, sft_dir: Path) -> dict:
    """||B_dpo@A_sft + B_sft@A_dpo|| relative to the merged update, on the real pair."""
    def factors(d: Path):
        cfg = json.loads((d / "adapter_config.json").read_text())
        s = cfg["lora_alpha"] / cfg["r"]
        sd = load_file(str(d / "adapter_model.safetensors"))
        f: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in sd.items():
            m = KEY.match(k)
            if m:
                f.setdefault(m.group(1), {})[m.group(2)] = v.to(torch.float32)
        return f, s

    fd, sd_ = factors(dpo_dir)
    fs, ss = factors(sft_dir)
    # peft linear combination: A_new = sum sqrt(w_i*scaling_i) A_i, same for B
    wd, ws = 1.0, 0.25
    cd, cs = np.sqrt(wd * sd_), np.sqrt(ws * ss)
    n_int = n_cross = n_merged = 0.0
    shared = [m for m in fd if m in fs]
    for m in shared:
        Bd, Ad = fd[m]["B"], fd[m]["A"]
        Bs, As = fs[m]["B"], fs[m]["A"]
        intended = sd_ * (Bd @ Ad) * wd + ss * (Bs @ As) * ws
        cross = (cd * cs) * (Bd @ As) + (cs * cd) * (Bs @ Ad)
        merged = intended + cross
        n_int += float(torch.linalg.matrix_norm(intended, ord="fro")) ** 2
        n_cross += float(torch.linalg.matrix_norm(cross, ord="fro")) ** 2
        n_merged += float(torch.linalg.matrix_norm(merged, ord="fro")) ** 2
    return {"n_modules": len(shared),
            "norm_intended": n_int ** 0.5, "norm_cross": n_cross ** 0.5,
            "norm_merged": n_merged ** 0.5,
            "cross_share_of_merged": (n_cross ** 0.5) / (n_merged ** 0.5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-root", default=str(RIG / "loras_repro"))
    ap.add_argument("--released", default=RELEASED)
    ap.add_argument("--out", default="outputs/analysis/gate_weight_checks.json")
    a = ap.parse_args()

    root = Path(a.repro_root)
    merged_dir = root / "llama-personas" / "impulsiveness"
    dpo_dir = root / "llama-distillation" / "impulsiveness"
    sft_dir = root / "llama-introspection" / "impulsiveness"

    print("== A1  adapter_config ==")
    rel_cfg = json.loads((Path(a.released) / "adapter_config.json").read_text())
    rep_cfg = json.loads((merged_dir / "adapter_config.json").read_text())
    a1 = {}
    for f in ("r", "lora_alpha", "base_model_name_or_path"):
        a1[f] = (rel_cfg[f] == rep_cfg[f])
        print(f"  {f:28s} released={rel_cfg[f]!r:45s} repro={rep_cfg[f]!r}  "
              f"{'MATCH' if a1[f] else 'DIFFER'}")
    a1["target_modules"] = set(rel_cfg["target_modules"]) == set(rep_cfg["target_modules"])
    print(f"  {'target_modules':28s} {sorted(rel_cfg['target_modules'])}  "
          f"{'MATCH' if a1['target_modules'] else 'DIFFER'}")
    a1_verdict = "PASS" if all(a1.values()) else "STOP"
    print(f"  A1 -> {a1_verdict}\n")

    print("== A2 / A3  ||dW||_F ==", flush=True)
    rel_n, _ = module_dW_norms(a.released)
    rep_n, _ = module_dW_norms(merged_dir)
    shared = sorted(set(rel_n) & set(rep_n))
    print(f"  modules compared: {len(shared)}  "
          f"(released {len(rel_n)}, repro {len(rep_n)})")
    rel_v = np.array([rel_n[m] for m in shared])
    rep_v = np.array([rep_n[m] for m in shared])
    tot_rel = float(np.sqrt((rel_v ** 2).sum()))
    tot_rep = float(np.sqrt((rep_v ** 2).sum()))
    ratio = tot_rep / tot_rel
    rho = spearman(rel_v, rep_v)
    a2 = band(ratio, 0.7, 1.4, 0.5, 2.0)
    a3 = "PASS" if rho >= 0.80 else ("STOP" if rho < 0.6 else "AMBIGUOUS")
    print(f"  overall ||dW||_F  released={tot_rel:.2f}  repro={tot_rep:.2f}  "
          f"ratio={ratio:.4f}   A2 -> {a2}   (pass [0.7,1.4], stop outside [0.5,2.0])")
    print(f"  per-module Spearman rho={rho:.4f}                    A3 -> {a3}   "
          f"(pass >= 0.80, stop < 0.6)")
    med = float(np.median(rep_v / rel_v))
    print(f"  median per-module ratio = {med:.4f}")

    print("\n== 6c  cross-term share on the REAL pair (descriptive only) ==", flush=True)
    ct = cross_term_share(dpo_dir, sft_dir)
    print(f"  modules              {ct['n_modules']}")
    print(f"  ||intended||         {ct['norm_intended']:.2f}   (dW_dpo + 0.25*dW_sft)")
    print(f"  ||cross||            {ct['norm_cross']:.2f}   (B_dpo@A_sft + B_sft@A_dpo)")
    print(f"  ||merged||           {ct['norm_merged']:.2f}")
    print(f"  cross share          {ct['cross_share_of_merged']*100:.1f}%   "
          f"(~59% on INDEPENDENT random adapters; the real pair is correlated)")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "A1": {"fields": {k: bool(v) for k, v in a1.items()}, "verdict": a1_verdict},
        "A2": {"released": tot_rel, "repro": tot_rep, "ratio": ratio, "verdict": a2},
        "A3": {"spearman": rho, "median_per_module_ratio": med, "verdict": a3},
        "cross_term_6c": ct,
        "n_modules": len(shared),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
