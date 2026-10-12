#!/usr/bin/env python3
"""Audit 3 of the P0 post-mortem: is any of the D+S gap a loader/precision artifact?

The additive state `D + 0.25*S` was measured by activating two adapters at inference, while the
merged state is a single adapter, and the SFT stage trained on a FOLDED bf16 checkpoint rather
than on base-plus-adapter. Each of those paths rounds to bfloat16 at a different point, and
bf16 has 8 mantissa bits, so the question is whether the rounding is large enough to matter
*relative to the update being applied* -- which is the right denominator, not the base weight,
since the update is three orders of magnitude smaller than the weight it perturbs.

Four paths to the same nominal state W0 + dW_D + 0.25*dW_S, per module, with fp32 as reference:

  reference   dW_add = dW_D + 0.25*dW_S                 computed in fp32 from the factors
  path 1      sequential bf16: round after EACH adapter addition  (naive two-adapter loader)
  path 2      fp32 accumulate, ONE rounding at the end           (sum-then-round)
  path 3      the ACTUAL folded DPO checkpoint on disk, plus 0.25*dW_S                <- the
              path the SFT stage and the folded-model measurements really took

and separately the fold itself is verified: fp32(W_folded) - fp32(W0) against dW_D.

Errors are reported as ||path - reference|| / ||dW_add||, i.e. relative to the ADAPTER UPDATE.
The same errors relative to ||W0|| are printed alongside only to show how different the two
denominators look -- the second is the misleading one.

No cached intermediates are read or written; the output path is fresh each run, and sha256
digests of the adapters, the configs, and of exactly the checkpoint tensors this audit reads
are recorded in the output so the numbers are tied to specific bytes.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/audit_loader_precision.py      # CPU, no GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossed_dpo_sft_weights import load_factors, scaling  # noqa: E402

RIG = Path("/workspace/oct_rig")
BF16 = torch.bfloat16
F32 = torch.float32


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Shards:
    """Lazy per-tensor reader over a sharded safetensors checkpoint."""

    def __init__(self, root: Path):
        self.root = root
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            self.map = json.loads(idx.read_text())["weight_map"]
        else:
            self.map = {}
            for f in sorted(root.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as h:
                    for k in h.keys():
                        self.map[k] = f.name
        self._open: dict[str, object] = {}
        self.digest = hashlib.sha256()

    def get(self, key: str) -> torch.Tensor:
        fn = self.map[key]
        if fn not in self._open:
            self._open[fn] = safe_open(str(self.root / fn), framework="pt")
        t = self._open[fn].get_tensor(key)
        # content digest over exactly the tensors this audit touches, in a fixed order
        self.digest.update(key.encode())
        self.digest.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(RIG / "models/llama-3.1-8b-it"))
    ap.add_argument("--dpo", default=str(RIG / "loras/llama-distillation/impulsiveness_regen"))
    ap.add_argument("--sft", default=str(RIG / "loras/llama-introspection/impulsiveness_regen"))
    ap.add_argument("--folded",
                    default=str(RIG / "models/distilled/llama-3.1-8b-it-impulsiveness_regen"),
                    help="the folded base+DPO checkpoint the SFT stage actually trained on")
    ap.add_argument("--sft-weight", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/analysis/audit_loader_precision.json")
    a = ap.parse_args()

    dD, dS = Path(a.dpo), Path(a.sft)
    fD, fS = load_factors(dD), load_factors(dS)
    sD, sS = scaling(dD), scaling(dS)
    base = Shards(Path(a.base))
    fold = Shards(Path(a.folded))
    mods = sorted(set(fD) & set(fS))
    print(f"base   {a.base}")
    print(f"dpo    {a.dpo}   alpha/r {sD:.3f}")
    print(f"sft    {a.sft}   alpha/r {sS:.3f}   weight {a.sft_weight}")
    print(f"folded {a.folded}")
    print(f"modules {len(mods)}\n")

    acc = {k: 0.0 for k in ("add2", "p1", "p2", "p3", "p1_p2", "fold", "dD2", "W02",
                            # <error, intended update>: the part of the rounding that is a
                            # systematic gain along the update rather than isotropic noise
                            "ip1", "ip2", "ip3", "ipfold", "n1", "n2", "n3", "nfold")}
    worst = {"p1": (0.0, ""), "p3": (0.0, ""), "fold": (0.0, "")}
    per: dict[str, dict] = {}
    for m in mods:
        key = m + ".weight"
        W0 = base.get(key)
        if W0.dtype != BF16:
            raise SystemExit(f"FATAL: base {key} is {W0.dtype}, expected bfloat16")
        Wf = fold.get(key)
        BD, AD = fD[m]
        BS, AS = fS[m]
        dD_ = (sD * (BD.to(F32) @ AD.to(F32)))
        dS_ = (sS * (BS.to(F32) @ AS.to(F32)))
        add = dD_ + a.sft_weight * dS_                       # fp32 reference
        W0f = W0.to(F32)

        # path 1: round after each adapter addition, as a naive sequential loader would
        W1 = (W0f + dD_.to(BF16).to(F32)).to(BF16)
        W2 = (W1.to(F32) + (a.sft_weight * dS_).to(BF16).to(F32)).to(BF16)
        d1 = W2.to(F32) - W0f
        # path 2: accumulate in fp32, round once
        d2 = (W0f + add).to(BF16).to(F32) - W0f
        # path 3: the real folded checkpoint, plus the SFT adapter on top
        d3 = (Wf.to(F32) + (a.sft_weight * dS_).to(BF16).to(F32)).to(BF16).to(F32) - W0f
        # the fold on its own
        dfold = Wf.to(F32) - W0f

        acc["add2"] += float(add.pow(2).sum())
        acc["dD2"] += float(dD_.pow(2).sum())
        acc["W02"] += float(W0f.pow(2).sum())
        acc["p1"] += float((d1 - add).pow(2).sum())
        acc["p2"] += float((d2 - add).pow(2).sum())
        acc["p3"] += float((d3 - add).pow(2).sum())
        acc["p1_p2"] += float((d1 - d2).pow(2).sum())
        acc["fold"] += float((dfold - dD_).pow(2).sum())
        # projection onto the INTENDED update. A rounding error that is pure noise has
        # <err, add> ~ 0 and leaves the realised dose along the update direction intact; one that
        # is a systematic shrinkage shows up here as a negative inner product.
        acc["ip1"] += float(((d1 - add) * add).sum())
        acc["ip2"] += float(((d2 - add) * add).sum())
        acc["ip3"] += float(((d3 - add) * add).sum())
        acc["ipfold"] += float(((dfold - dD_) * dD_).sum())
        acc["n1"] += float(d1.pow(2).sum())
        acc["n2"] += float(d2.pow(2).sum())
        acc["n3"] += float(d3.pow(2).sum())
        acc["nfold"] += float(dfold.pow(2).sum())

        na = float(add.norm())
        r1 = float((d1 - add).norm()) / na
        r3 = float((d3 - add).norm()) / na
        rf = float((dfold - dD_).norm()) / float(dD_.norm())
        per[m] = {"rel_p1": r1, "rel_p2": float((d2 - add).norm()) / na,
                  "rel_p3": r3, "rel_fold": rf, "norm_add": na, "norm_dD": float(dD_.norm())}
        for k, v in (("p1", r1), ("p3", r3), ("fold", rf)):
            if v > worst[k][0]:
                worst[k] = (v, m)

    nadd = np.sqrt(acc["add2"])
    ndD = np.sqrt(acc["dD2"])
    nW0 = np.sqrt(acc["W02"])
    rows = [
        ("path 1  sequential bf16 rounding", np.sqrt(acc["p1"]) / nadd, np.sqrt(acc["p1"]) / nW0),
        ("path 2  fp32 sum, one rounding", np.sqrt(acc["p2"]) / nadd, np.sqrt(acc["p2"]) / nW0),
        ("path 3  folded ckpt + SFT adapter", np.sqrt(acc["p3"]) / nadd, np.sqrt(acc["p3"]) / nW0),
        ("path 1 vs path 2", np.sqrt(acc["p1_p2"]) / nadd, np.sqrt(acc["p1_p2"]) / nW0),
    ]
    print(f"{'':38s}{'/ ||dW_add||':>14s}{'/ ||W_base||':>14s}")
    print("-" * 66)
    for name, rel_u, rel_w in rows:
        print(f"{name:38s}{rel_u:14.6f}{rel_w:14.3e}")
    rel_fold = np.sqrt(acc["fold"]) / ndD
    print(f"\n{'fold: W_folded - W_base  vs  dW_D':38s}{rel_fold:14.6f}"
          f"{np.sqrt(acc['fold']) / nW0:14.3e}   (denominator ||dW_D||)")
    print(f"\n||dW_add|| {nadd:.3f}   ||dW_D|| {ndD:.3f}   ||W_base|| {nW0:.1f}"
          f"   ||dW_add||/||W_base|| {nadd / nW0:.3e}")
    # Is the error noise or a systematic change in dose? Two numbers settle it:
    #   proj  = <err, intended>/||intended||^2   fractional systematic gain along the update
    #   dose  = ||realised||/||intended||        total norm actually applied
    print(f"\n{'':38s}{'proj on update':>16s}{'realised dose':>15s}{'cos(err,upd)':>14s}")
    print("-" * 83)
    proj = []
    for name, ip, nn, ref2, errsq in (
            ("path 1  sequential bf16 rounding", acc["ip1"], acc["n1"], acc["add2"], acc["p1"]),
            ("path 2  fp32 sum, one rounding", acc["ip2"], acc["n2"], acc["add2"], acc["p2"]),
            ("path 3  folded ckpt + SFT adapter", acc["ip3"], acc["n3"], acc["add2"], acc["p3"]),
            ("fold:   W_folded - W_base", acc["ipfold"], acc["nfold"], acc["dD2"], acc["fold"])):
        pr = ip / ref2
        dose = math.sqrt(nn / ref2)
        cs = ip / math.sqrt(errsq * ref2) if errsq > 0 else float("nan")
        print(f"{name:38s}{pr:+16.6f}{dose:15.6f}{cs:+14.4f}")
        proj.append({"path": name, "proj_on_update": pr, "realised_dose": dose,
                     "cos_err_update": cs})

    print("\nworst module, relative to its own ||dW_add|| (||dW_D|| for the fold):")
    for k, (v, m) in worst.items():
        print(f"  {k:5s}{v:10.6f}  {m}")

    out = {
        "base": a.base, "dpo": a.dpo, "sft": a.sft, "folded": a.folded,
        "sft_weight": a.sft_weight, "scalings": {"D": sD, "S": sS}, "n_modules": len(mods),
        "hashes": {
            "dpo_adapter": sha256_file(dD / "adapter_model.safetensors"),
            "dpo_config": sha256_file(dD / "adapter_config.json"),
            "sft_adapter": sha256_file(dS / "adapter_model.safetensors"),
            "sft_config": sha256_file(dS / "adapter_config.json"),
            "base_config": sha256_file(Path(a.base) / "config.json"),
            "folded_config": sha256_file(Path(a.folded) / "config.json"),
            # digest over exactly the target-module tensors read above, in module-sorted order
            "base_target_tensors": base.digest.hexdigest(),
            "folded_target_tensors": fold.digest.hexdigest(),
        },
        "global": {name: {"rel_to_update": float(u), "rel_to_base": float(w)}
                   for name, u, w in rows}
        | {"fold_vs_dW_D": {"rel_to_update": float(rel_fold),
                            "rel_to_base": float(np.sqrt(acc["fold"]) / nW0)}},
        "norms": {"dW_add": float(nadd), "dW_D": float(ndD), "W_base": float(nW0)},
        "worst_module": {k: {"rel": v, "module": m} for k, (v, m) in worst.items()},
        "projection": proj,
        "per_module": per,
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
