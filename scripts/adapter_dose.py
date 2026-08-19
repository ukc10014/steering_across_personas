#!/usr/bin/env python3
"""Exact weight-space perturbation size of each OCT LoRA adapter.

This is the cheap half of the dose-confound question (prompt section 11). If the
impulsiveness arm restructures persona geometry more than goodness or mathematical, one
explanation is constitution content and another is simply that it is a bigger
intervention. This script measures the second directly from the adapter weights -- no
GPU, no forward passes, no proxy.

For each targeted module, LoRA adds

    dW = (alpha / r) * B @ A

and we report ||dW||_F, both absolute and relative to the base weight ||W||_F it is added
to. The relative figure is the meaningful one: an absolute Frobenius norm is not
comparable across modules of different shapes.

CAVEAT, stated plainly: weight-space norm is NOT the same as functional perturbation.
Two adapters with equal ||dW|| can move behaviour by very different amounts depending on
where they sit and how inputs align with them. This measure bounds and describes the
intervention, it does not dose-match it. Genuine dose matching needs activation
displacement on a neutral corpus or output KL from base, which needs a GPU.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

SNAP_GLOB = "/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/*"
BASE = ("/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
        "snapshots/0e9e39f249a16976918f6564b8830bc894c89659")


def load_base_norms() -> dict[str, float]:
    """||W||_F for every base weight, from the safetensors index."""
    from safetensors import safe_open
    norms: dict[str, float] = {}
    for shard in sorted(Path(BASE).glob("model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if any(p in k for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj")):
                    norms[k] = f.get_tensor(k).float().norm().item()
    return norms


def analyse(adapter_dir: Path, base_norms: dict[str, float]) -> dict:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(adapter_dir / "adapter_model.safetensors")

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in sd.items():
        if ".lora_A" in k:
            pairs[k.split(".lora_A")[0]]["A"] = v
        elif ".lora_B" in k:
            pairs[k.split(".lora_B")[0]]["B"] = v

    per_module, by_type, by_layer = {}, defaultdict(list), defaultdict(list)
    tot_sq = tot_base_sq = 0.0
    for name, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        dW = (ab["B"].float() @ ab["A"].float()) * scale
        n = dW.norm().item()

        # map adapter name -> base weight key
        key = re.sub(r"^base_model\.model\.", "", name) + ".weight"
        bn = base_norms.get(key)
        per_module[key] = {"dW_norm": n, "base_norm": bn,
                           "relative": (n / bn) if bn else None}
        tot_sq += n ** 2
        if bn:
            tot_base_sq += bn ** 2
        mtype = key.split(".")[-2]
        by_type[mtype].append(n / bn if bn else 0.0)
        lm = re.search(r"layers\.(\d+)\.", key)
        if lm:
            by_layer[int(lm.group(1))].append(n / bn if bn else 0.0)

    return {
        "adapter": adapter_dir.name,
        "r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "scale": scale,
        "n_modules": len(per_module),
        "total_dW_norm": tot_sq ** 0.5,
        "total_base_norm": tot_base_sq ** 0.5,
        "global_relative": (tot_sq ** 0.5) / (tot_base_sq ** 0.5) if tot_base_sq else None,
        "mean_relative_per_module": sum(
            m["relative"] for m in per_module.values() if m["relative"]) / max(1, len(per_module)),
        "by_module_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
        "by_layer": {k: sum(v) / len(v) for k, v in sorted(by_layer.items())},
        "per_module": per_module,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="+",
                    default=["goodness", "mathematical", "impulsiveness"])
    ap.add_argument("--out", type=str, default="outputs/analysis/adapter_dose.json")
    args = ap.parse_args()

    snap = next(Path("/").glob(SNAP_GLOB.lstrip("/")))
    print(f"snapshot: {snap}")
    print("computing base weight norms ...", flush=True)
    base_norms = load_base_norms()
    print(f"  {len(base_norms)} base projection weights\n", flush=True)

    results = {}
    for a in args.adapters:
        d = snap / a
        if not d.exists():
            print(f"  !! missing {d}")
            continue
        results[a] = analyse(d, base_norms)
        r = results[a]
        print(f"{a:16s} r={r['r']} alpha={r['lora_alpha']} modules={r['n_modules']}  "
              f"||dW||_F={r['total_dW_norm']:.1f}  "
              f"global rel={r['global_relative']:.4f}  "
              f"mean per-module rel={r['mean_relative_per_module']:.4f}", flush=True)

    if results:
        ref = results.get("goodness")
        if ref:
            print("\nrelative to goodness (global relative norm):")
            for a, r in results.items():
                print(f"  {a:16s} {r['global_relative']/ref['global_relative']:.3f}x")
        print("\nmean relative ||dW|| by module type:")
        types = sorted(next(iter(results.values()))["by_module_type"])
        print(f"  {'':16s}" + "".join(f"{t:>11s}" for t in types))
        for a, r in results.items():
            print(f"  {a:16s}" + "".join(f"{r['by_module_type'][t]:>11.4f}" for t in types))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
