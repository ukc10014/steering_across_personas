#!/usr/bin/env python3
"""The stage-localisation comparison table. docs/spec_stage_localisation.md §2, §10 item 3.

One row per model state, the SAME endpoints at every one, so movement is attributable to
stages rather than to a change of metric.

Endpoints are computed from the stored per-trait quantities, not re-derived:
    B1  offset[impulsivity] - mean(offset[other seven])        the REGISTERED endpoint
    B2  mean(offset[impulsivity, risk_taking]) - mean(other 6) secondary
    k   mean over traits of retention[trait]
    sel target/other common-shift ratio at the layer
    cos mean_t cos(dG_t, dG_t of M_F), plus the eight per-trait values
    dose measured trait-vector displacement, absolute and as a fraction of M_F
    ||dW||_F from the LoRA factors -- DIAGNOSTIC, never a dose substitute (that was §7.1's error)

B1 computed this way reproduces the registered released value (+2.18) exactly, which is the
check that the assembly is right.

    python scripts/stage_comparison.py --seed 1 --csv outputs/analysis/stage_comparison_seed1.csv
"""
from __future__ import annotations

import argparse, csv, json, statistics as st
from pathlib import Path

import torch

OUT = Path("outputs/analysis")
RIG = Path("/workspace/oct_rig")
TARGETS_PAIR = ("impulsivity", "risk_taking")
TARGET_ONLY = "impulsivity"

STATES = {
    1: [("M_0",       "base",                          []),
        ("M_D",       "impulsiveness_repro_dpo",       [("loras_repro/llama-distillation", 1.0)]),
        ("M_D+0.25S", "impulsiveness_repro_Dplus025S", [("loras_repro/llama-distillation", 1.0),
                                                        ("loras_repro/llama-introspection", 0.25)]),
        ("M_D+S",     "impulsiveness_repro_DplusS",    [("loras_repro/llama-distillation", 1.0),
                                                        ("loras_repro/llama-introspection", 1.0)]),
        ("M_F",       "impulsiveness_repro",           [("loras_repro/llama-personas", 1.0)]),
        ("M_S",       "impulsiveness_sft_from_base",   [("loras_sft_from_base", 1.0)]),
        ("M_0+A_S*",  "impulsiveness_repro_sft",       [("loras_repro/llama-introspection", 1.0)]),
        ("released",  "impulsiveness",                 []),
       ],
    2: [("M_0",       "base",                          []),
        ("M_D",       "impulsiveness_seed2_dpo",       [("loras_seed2/llama-distillation", 1.0)]),
        ("M_D+0.25S", "impulsiveness_seed2_Dplus025S", [("loras_seed2/llama-distillation", 1.0),
                                                        ("loras_seed2/llama-introspection", 0.25)]),
        ("M_D+S",     "impulsiveness_seed2_DplusS",    [("loras_seed2/llama-distillation", 1.0),
                                                        ("loras_seed2/llama-introspection", 1.0)]),
        ("M_F",       "impulsiveness_seed2",           [("loras_seed2/llama-personas", 1.0)]),
       ],
}
FINAL = {1: "impulsiveness_repro", 2: "impulsiveness_seed2"}


def jload(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def contrast(off_arm, targets):
    """mean over target traits minus mean over the rest, on the stored offsets."""
    if not off_arm:
        return None
    tg = [off_arm[t]["point"] for t in targets if t in off_arm]
    ot = [v["point"] for t, v in off_arm.items() if t not in targets]
    return st.mean(tg) - st.mean(ot) if tg and ot else None


def dw_norm(parts):
    """||sum_i s_i*(alpha_i/r_i)*B_i@A_i||_F, without ever forming a d_out x d_in product.

    Forming B@A costs a 4096x14336 matrix per module; over 224 modules and several states
    that dominates the whole script. The Frobenius norm of a sum of low-rank terms needs
    only r x r matrices:

        <B_i A_i, B_j A_j>_F = tr(A_i^T B_i^T B_j A_j) = tr((B_i^T B_j)(A_j A_i^T))

    with r = 64 here, so each inner product is two 64x64 multiplies instead of a 58M-element
    outer product. Exact, not an approximation.
    """
    if not parts:
        return 0.0
    from persona_steering.lora import lora_deltas
    loaded = []
    for rel, s in parts:
        d = RIG / rel
        d = d / "impulsiveness" if (d / "impulsiveness").exists() else d
        if not (d / "adapter_model.safetensors").exists():
            return None
        deltas, ar = lora_deltas(d)
        loaded.append((deltas, ar * s))
    total = 0.0
    modules = set(loaded[0][0])
    for m in modules:
        for di, ci in loaded:
            for dj, cj in loaded:
                Bi, Ai = di[m]
                Bj, Aj = dj[m]
                # fp64 on the r x r products: the trace form squares intermediate
                # magnitudes, and in fp32 that drifts ~0.4% from the direct ||B@A||_F.
                total += ci * cj * float(torch.trace(
                    (Bi.double().T @ Bj.double()) @ (Aj.double() @ Ai.double().T)))
    return max(total, 0.0) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, choices=(1, 2))
    ap.add_argument("--layer", default="15")
    ap.add_argument("--variant", default="forced", choices=("forced", "default"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    lg = (jload("caa_logits.json") or {}).get(a.variant, {})
    cs = (jload("common_shift.json") or {}).get(a.layer, {})
    fd = (jload("functional_dose.json") or {}).get(a.layer, {})
    final = FINAL[a.seed]
    traits = list(cs) if cs else []
    dose_final = (fd.get(final) or {}).get("trait_vector_displacement")

    rows = []
    for label, arm, parts in STATES[a.seed]:
        r = {"state": label, "arm": arm}
        off = (lg.get("offset") or {}).get(arm)
        ret = (lg.get("retention") or {}).get(arm)
        r["B1"] = contrast(off, (TARGET_ONLY,))
        r["B2"] = contrast(off, TARGETS_PAIR)
        r["k"] = st.mean(v["point"] for v in ret.values()) if ret else None
        if traits and arm in cs[traits[0]]["per_arm"]:
            g = lambda t: cs[t]["per_arm"][arm]["g_over_base"]
            tgt = st.mean(g(t) for t in TARGETS_PAIR)
            oth = st.mean(g(t) for t in traits if t not in TARGETS_PAIR)
            r["sel"] = tgt / oth
            cl = [c for t in traits
                  for c in [cs[t]["cos"].get(f"{arm}|{final}") or cs[t]["cos"].get(f"{final}|{arm}")]
                  if c is not None]
            if cl:
                r["cos_to_M_F"], r["cos_min"], r["cos_max"] = st.mean(cl), min(cl), max(cl)
        d = (fd.get(arm) or {}).get("trait_vector_displacement")
        r["dose"] = d
        r["dose_frac_M_F"] = (d / dose_final) if (d is not None and dose_final) else None
        r["dW"] = dw_norm(parts)
        rows.append(r)

    cols = ["state", "B1", "B2", "k", "sel", "cos_to_M_F", "dose", "dose_frac_M_F", "dW"]
    hdr = {"cos_to_M_F": "cos->M_F", "dose_frac_M_F": "dose/M_F", "dW": "||dW||"}
    w = {c: max(len(hdr.get(c, c)), 9) for c in cols}
    w["state"] = 10
    print(f"\nSTAGE COMPARISON — seed {a.seed}, layer {a.layer}, {a.variant} prompt")
    print(f"cos is against {final} (= M_F). B1 is the registered endpoint.\n")
    print("  ".join(hdr.get(c, c).ljust(w[c]) for c in cols))
    print("-" * (sum(w.values()) + 2 * len(cols)))
    for r in rows:
        out = []
        for c in cols:
            v = r.get(c)
            s = "" if v is None else (f"{v:+.3f}" if c in ("B1", "B2") else
                                      f"{v:.4f}" if isinstance(v, float) else str(v))
            out.append(s.ljust(w[c]))
        print("  ".join(out))
    print("\n  * M_0+A_S is the OFF-BASE diagnostic: that adapter was fitted on the folded DPO")
    print("    model, so applying it to the base is a component measurement, not 'SFT alone'.")
    print("  ||dW||_F is a DIAGNOSTIC, not a substitute for measured functional dose.")

    miss = [r["arm"] for r in rows if r.get("B1") is None]
    if miss:
        print(f"\n  not yet measured: {', '.join(miss)}")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = cols + ["arm", "cos_min", "cos_max"]
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k) for k in fields})
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
