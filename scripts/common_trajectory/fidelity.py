#!/usr/bin/env python3
"""Endpoint reproduction fidelity, reported as a result for every constitution (amendment A1).

Reproduced released-style merge of THIS trajectory's run vs the released adapter:
  A1  merged adapter config (r, alpha, target modules)
  A2  ||dW||_F ratio (and the global weight cosine, reported, not banded)
  A4  measured functional dose ratio, L15 (L20 reported)
  B4  mean over 8 traits of cos(dG_repro, dG_released), cross-fitted, L15 (L20 reported)
  B5  retention k (mean over traits) difference
  P   eight-trait offset profile: Pearson r and norm ratio ||p_repro|| / ||p_released||

Category (A1, fixed before the panel's training):
  strong  = A2, A4 in [0.7, 1.4], B4 >= 0.85, r >= 0.85, norm ratio in [0.7, 1.4], |dk| <= 0.08
  poor    = A2 or A4 outside [0.5, 2.0], B4 < 0.70, r < 0.6, or norm ratio outside [0.5, 2.0]
  partial = otherwise
Fidelity differences are NOT read as differences in how hard a character is to learn.

    python3 scripts/common_trajectory/fidelity.py --constitution goodness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from caa_logits_analysis import fit_offset_slope, stack_trait            # noqa: E402
from common_shift import load_arms_checked, cell, make_preps, stats_for_trait   # noqa: E402
from geometry_lib import half_splits                                     # noqa: E402
from behaviour import load as load_logits                                # noqa: E402

A = REPO / "outputs/analysis/common_trajectory"
SNAP = Path("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620")
RIG = Path("/workspace/oct_rig")
# (reproduced merged arm, its adapter dir, released arm, released adapter dir)
ENDPOINTS = {
    "impulsiveness": ("impulsiveness_curve_mrg_stepfinal", RIG / "loras_sft_curve/merged/stepfinal", "impulsiveness", SNAP / "impulsiveness"),
}
for c in ("loving", "goodness", "sycophancy"):
    ENDPOINTS[c] = (f"{c}_traj_mrg_final", RIG / f"loras/llama-personas/{c}", c, SNAP / c)


def dW(d):
    cfg = json.loads((d / "adapter_config.json").read_text()); sc = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(d / "adapter_model.safetensors"))
    return cfg, {k.split(".lora_A")[0]: sc * (sd[k.replace("lora_A", "lora_B")].float() @ sd[k].float())
                 for k in sd if "lora_A" in k}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--constitution", required=True)
    ap.add_argument("--cache-dir", default=str(REPO / "outputs/_qcache"))
    ap.add_argument("--base-arm", default="base")
    ap.add_argument("--released-arm", default=None,
                    help="activation/logit arm standing for the released model (default: the constitution "
                         "name). xchar_lov_start is the corrected-mask extraction of released loving.")
    ap.add_argument("--allow-mixed-mask", action="store_true",
                    help="ONLY to reproduce the historical mixed-protocol numbers; recorded in the output")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    c = a.constitution; rep, rep_dir, rel, rel_dir = ENDPOINTS[c]
    rel = a.released_arm or rel
    out = {"constitution": c, "reproduced": rep, "released": rel, "base_arm": a.base_arm}

    cfg_r, wr = dW(rep_dir); cfg_l, wl = dW(rel_dir)
    out["A1"] = {"r": cfg_r["r"], "alpha": cfg_r["lora_alpha"], "modules": sorted(cfg_r["target_modules"]),
                 "matches_released": (cfg_r["r"], cfg_r["lora_alpha"], sorted(cfg_r["target_modules"])) ==
                                     (cfg_l["r"], cfg_l["lora_alpha"], sorted(cfg_l["target_modules"]))}
    # Same module set required: intersecting would silently compare a subset of the update.
    out["A1"]["same_module_set"] = set(wr) == set(wl)
    if not out["A1"]["same_module_set"]:
        raise SystemExit(f"{c}: reproduced and released adapters patch different modules "
                         f"({len(set(wr) ^ set(wl))} differ); A2 is undefined")
    keys = sorted(wr)
    nr = float(np.sqrt(sum(float((wr[k].double() ** 2).sum()) for k in keys)))
    nl = float(np.sqrt(sum(float((wl[k].double() ** 2).sum()) for k in keys)))
    dot = sum(float((wr[k].double() * wl[k].double()).sum()) for k in keys)
    out["A2"] = {"norm_ratio": nr / nl, "weight_cosine": dot / (nr * nl), "modules": len(keys)}
    del wr, wl

    import subprocess
    dose = {}
    for L in (15, 20):
        f = A / f"functional_dose_{c}_fidelity{a.tag}.json"
        subprocess.run([sys.executable, str(REPO / "scripts/functional_dose.py"), "--layer", str(L), "--arms", rel, rep,
                        "--cache-dir", a.cache_dir, "--base-arm", a.base_arm, "--out", str(f)]
                       + (["--allow-mixed-mask"] if a.allow_mixed_mask else []),
                       check=True, capture_output=True)
        d = json.loads(f.read_text())[str(L)]
        dose[str(L)] = d[rep]["trait_vector_displacement"] / d[rel]["trait_vector_displacement"]
    out["A4"] = {"dose_ratio": dose}

    cells = load_logits(["base", rep, rel]); traits = sorted(cells["base"]); personas = sorted(cells["base"][traits[0]])
    p, k = {rep: [], rel: []}, {rep: [], rel: []}
    for t in traits:
        qid = cells["base"][t][personas[0]]["qid"]; apos = cells["base"][t][personas[0]]["a_is_positive"]
        base = stack_trait(cells, "base", t, personas, qid)
        for x in (rep, rel):
            o, kk = fit_offset_slope(base, stack_trait(cells, x, t, personas, qid), apos, np.arange(len(qid)))
            p[x].append(o); k[x].append(kk)
    pr, pl = np.array(p[rep]), np.array(p[rel])
    out["profile"] = {"traits": traits, "reproduced": pr.tolist(), "released": pl.tolist(),
                      "pearson_r": float(np.corrcoef(pr, pl)[0, 1]), "norm_ratio": float(np.linalg.norm(pr) / np.linalg.norm(pl))}
    out["B5"] = {"k_reproduced": float(np.mean(k[rep])), "k_released": float(np.mean(k[rel])),
                 "abs_diff": float(abs(np.mean(k[rep]) - np.mean(k[rel])))}

    rng = np.random.default_rng(0); cache = Path(a.cache_dir); b4 = {}
    for L in (15, 20):
        D, out["provenance"] = load_arms_checked(cache, [a.base_arm, rep, rel], [15, 20], L,
                                                 allow_mixed_mask=a.allow_mixed_mask)
        bd = D.pop(a.base_arm)
        pidx = [bd["personas"].index(q) for q in bd["semantic"]]; cs = []
        for ti, t in enumerate(bd["traits"]):
            bp, bn = cell(bd, ti, pidx)
            r = stats_for_trait(make_preps(bp, bn, {x: cell(D[x], ti, pidx) for x in (rep, rel)}),
                                half_splits(int(bd["nq"][ti]), 40, rng), [rep, rel], naive=False)
            cs.append(r["cos"][f"{rep}|{rel}"])
        b4[str(L)] = float(np.mean(cs))
    out["B4"] = {"mean_cos_common_shift": b4}

    A2, A4, B4 = out["A2"]["norm_ratio"], dose["15"], b4["15"]
    r_, nr_, dk = out["profile"]["pearson_r"], out["profile"]["norm_ratio"], out["B5"]["abs_diff"]
    inb = lambda v, lo, hi: lo <= v <= hi
    if (not inb(A2, .5, 2) or not inb(A4, .5, 2) or B4 < .70 or r_ < .6 or not inb(nr_, .5, 2)):
        cat = "poor"
    elif (inb(A2, .7, 1.4) and inb(A4, .7, 1.4) and B4 >= .85 and r_ >= .85 and inb(nr_, .7, 1.4) and dk <= .08
          and out["A1"]["matches_released"]):
        cat = "strong"          # A1 is part of the fixed rule; it was recorded but not enforced before
    else:
        cat = "partial"
    out["category"] = cat
    if a.tag:
        A2dir = REPO / "outputs/analysis/common_trajectory/v2"; A2dir.mkdir(parents=True, exist_ok=True)
        (A2dir / f"fidelity_{c}{a.tag}.json").write_text(json.dumps(out, indent=1))
    else:
        (A / f"fidelity_{c}.json").write_text(json.dumps(out, indent=1))
    print(f"{c:14s} {cat:8s} A2 {A2:.2f} (wcos {out['A2']['weight_cosine']:.2f})  A4 {A4:.2f}/{dose['20']:.2f}  "
          f"B4 {B4:.3f}/{b4['20']:.3f}  profile r {r_:.2f} norm {nr_:.2f}  dk {dk:.3f}  A1 {out['A1']['matches_released']}")


if __name__ == "__main__":
    main()
