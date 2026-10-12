#!/usr/bin/env python3
"""Weight-space half of the crossed DPO x SFT reconstruction, for the P0 gate failure.

P0 (`impulsiveness_regen`) failed spec_paraphrase_replication 4.1, and the five-state
localisation put the divergence in the DPO/weight channel: the SFT adapter alone is
indistinguishable from the reproduction's, every state containing the regenerated DPO adapter
is lower, and the deficit grows with how entangled the combination is. See
docs/runs/oct/GATE_REPORT_paraphrase-p0.md.

This script asks, before any GPU time: do the two DPO updates actually look different, and do
the peft cross-term matrices look MORE different than the DPO updates themselves? If
dW_Do ~ dW_Dn while X_oo and X_nn diverge, the mechanism is factor geometry rather than the
DPO update per se.

    D_o, S_o   the reproduction's stage adapters   (loras_repro, constitution `impulsiveness`)
    D_n, S_n   P0's stage adapters                 (loras,  constitution `impulsiveness_regen`)

Four merges are then possible, F_ij = peft-merge(D_i, S_j). tools/merge_loras.py uses
add_weighted_adapter(["dpo","sft"], weights=[1.0, 0.25], combination_type="linear"), which
combines FACTORS, so (scripts/check_peft_merge.py):

    A_new = cD*A_D + cS*A_S,  B_new = cD*B_D + cS*B_S,  cD = sqrt(1.0*sD), cS = sqrt(0.25*sS)
    dW_F  = dW_D + 0.25*dW_S + cD*cS*(B_D@A_S + B_S@A_D)
            \___ additive A_ij ___/   \________ cross term X_ij ________/

so the additive construction A_ij is exactly the cross-term-free counterpart of F_ij, and is
what `--lora-adapter D_i --lora-scale 1 --lora-adapter S_j --lora-scale 0.25` produces at
inference. No training and no merging is needed for any of this.

NOTHING HERE FORMS A d_out x d_in MATRIX. Every norm and inner product between low-rank
products uses <P1@Q1, P2@Q2>_F = tr((P1^T P2)(Q2 Q1^T)), which touches only r x r matrices.
Over 224 modules and eight states the dense route would dominate the whole script.

    PYTHONPATH=$PYLIBS_TRAIN python3 scripts/crossed_dpo_sft_weights.py     # CPU, no GPU
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

RIG = Path("/workspace/oct_rig")


def scaling(adapter_dir: Path) -> float:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    return cfg["lora_alpha"] / cfg["r"]


def load_factors(adapter_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{module: (B, A)} in float64. Keys are stripped to the bare module path."""
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    out: dict[str, list] = {}
    for k, v in sd.items():
        if ".lora_A" in k:
            mod, side = k.split(".lora_A"), "A"
        elif ".lora_B" in k:
            mod, side = k.split(".lora_B"), "B"
        else:
            continue
        name = mod[0].replace("base_model.model.", "")
        out.setdefault(name, {})[side] = v.to(torch.float64)
    return {k: (v["B"], v["A"]) for k, v in out.items() if "A" in v and "B" in v}


class LowRankSum:
    """sum_i c_i * P_i @ Q_i, kept factored. Inner products stay r x r."""

    def __init__(self, terms: list[tuple[float, torch.Tensor, torch.Tensor]]):
        self.terms = terms

    def inner(self, other: "LowRankSum") -> float:
        tot = 0.0
        for c1, P1, Q1 in self.terms:
            for c2, P2, Q2 in other.terms:
                # tr((P1^T P2)(Q2 Q1^T)) -- never forms P@Q
                tot += c1 * c2 * float(torch.trace((P1.T @ P2) @ (Q2 @ Q1.T)))
        return tot

    def norm(self) -> float:
        return math.sqrt(max(self.inner(self), 0.0))


def cos(x: LowRankSum, y: LowRankSum) -> float:
    nx, ny = x.norm(), y.norm()
    return x.inner(y) / (nx * ny) if nx > 0 and ny > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default=str(RIG / "loras_repro"))
    ap.add_argument("--old-constitution", default="impulsiveness")
    ap.add_argument("--new-root", default=str(RIG / "loras"))
    ap.add_argument("--new-constitution", default="impulsiveness_regen")
    ap.add_argument("--out", default="outputs/analysis/crossed_dpo_sft_weights.json")
    a = ap.parse_args()

    dirs = {
        "D_o": Path(a.old_root) / "llama-distillation" / a.old_constitution,
        "S_o": Path(a.old_root) / "llama-introspection" / a.old_constitution,
        "D_n": Path(a.new_root) / "llama-distillation" / a.new_constitution,
        "S_n": Path(a.new_root) / "llama-introspection" / a.new_constitution,
    }
    for k, d in dirs.items():
        if not (d / "adapter_model.safetensors").exists():
            raise SystemExit(f"FATAL: {k} missing at {d}")
    fac = {k: load_factors(d) for k, d in dirs.items()}
    s = {k: scaling(d) for k, d in dirs.items()}
    print("adapters:")
    for k in dirs:
        print(f"  {k}  scaling alpha/r = {s[k]:.3f}  modules {len(fac[k])}  {dirs[k]}")

    mods = sorted(set.intersection(*(set(f) for f in fac.values())))
    print(f"\nmodules common to all four: {len(mods)}")
    if not mods:
        raise SystemExit("FATAL: no common modules")

    # merge coefficients, per adapter pair (i=DPO, j=SFT)
    def cD(i): return math.sqrt(1.00 * s[i])
    def cS(j): return math.sqrt(0.25 * s[j])

    acc: dict[str, float] = {}
    def add(key: str, v: float) -> None:
        acc[key] = acc.get(key, 0.0) + v

    for m in mods:
        B, A = {}, {}
        for k in dirs:
            B[k], A[k] = fac[k][m]

        dW = {k: LowRankSum([(s[k], B[k], A[k])]) for k in dirs}

        # cross-term matrices X_ij = cD(i)*cS(j) * (B_Di @ A_Sj + B_Sj @ A_Di)
        X = {}
        for i in ("D_o", "D_n"):
            for j in ("S_o", "S_n"):
                c = cD(i) * cS(j)
                X[f"{i[-1]}{j[-1]}"] = LowRankSum([(c, B[i], A[j]), (c, B[j], A[i])])

        # additive A_ij and factor-merged F_ij
        Aij, Fij = {}, {}
        for i in ("D_o", "D_n"):
            for j in ("S_o", "S_n"):
                t = f"{i[-1]}{j[-1]}"
                Aij[t] = LowRankSum([(s[i], B[i], A[i]), (0.25 * s[j], B[j], A[j])])
                Fij[t] = LowRankSum(Aij[t].terms + X[t].terms)

        # --- the questions, accumulated as sums of inner products over modules
        add("ip_DoDn", dW["D_o"].inner(dW["D_n"]))
        add("n2_Do", dW["D_o"].inner(dW["D_o"]))
        add("n2_Dn", dW["D_n"].inner(dW["D_n"]))
        add("ip_SoSn", dW["S_o"].inner(dW["S_n"]))
        add("n2_So", dW["S_o"].inner(dW["S_o"]))
        add("n2_Sn", dW["S_n"].inner(dW["S_n"]))
        add("ip_Xoo_Xnn", X["oo"].inner(X["nn"]))
        for t in X:
            add(f"n2_X{t}", X[t].inner(X[t]))
            add(f"n2_A{t}", Aij[t].inner(Aij[t]))
            add(f"n2_F{t}", Fij[t].inner(Fij[t]))
        add("ip_Foo_Fnn", Fij["oo"].inner(Fij["nn"]))
        add("ip_Aoo_Ann", Aij["oo"].inner(Aij["nn"]))
        add("ip_Foo_Fno", Fij["oo"].inner(Fij["no"]))
        add("ip_Foo_Fon", Fij["oo"].inner(Fij["on"]))

    def g(k): return math.sqrt(max(acc[k], 0.0))
    def c(ip, n1, n2): return acc[ip] / (g(n1) * g(n2))

    print("\n== do the stage updates differ? (global, over all modules) ==")
    print(f"  cos(dW_Do, dW_Dn) = {c('ip_DoDn','n2_Do','n2_Dn'):.4f}"
          f"   ||dW_Dn||/||dW_Do|| = {g('n2_Dn')/g('n2_Do'):.4f}")
    print(f"  cos(dW_So, dW_Sn) = {c('ip_SoSn','n2_So','n2_Sn'):.4f}"
          f"   ||dW_Sn||/||dW_So|| = {g('n2_Sn')/g('n2_So'):.4f}")

    print("\n== cross-term matrices X_ij = B_Di@A_Sj + B_Sj@A_Di ==")
    for t in ("oo", "on", "no", "nn"):
        print(f"  ||X_{t}|| = {g('n2_X'+t):8.3f}   ||A_{t}|| = {g('n2_A'+t):8.3f}   "
              f"||F_{t}|| = {g('n2_F'+t):8.3f}   cross share {g('n2_X'+t)/g('n2_F'+t):6.1%}")
    print(f"\n  cos(X_oo, X_nn)   = {c('ip_Xoo_Xnn','n2_Xoo','n2_Xnn'):.4f}"
          "   <- compare with cos(dW_Do, dW_Dn) above")
    print(f"  cos(F_oo, F_nn)   = {c('ip_Foo_Fnn','n2_Foo','n2_Fnn'):.4f}")
    print(f"  cos(A_oo, A_nn)   = {c('ip_Aoo_Ann','n2_Aoo','n2_Ann'):.4f}"
          "   <- additive, no cross terms")
    print(f"  cos(F_oo, F_no)   = {c('ip_Foo_Fno','n2_Foo','n2_Fno'):.4f}   (swap DPO only)")
    print(f"  cos(F_oo, F_on)   = {c('ip_Foo_Fon','n2_Foo','n2_Fon'):.4f}   (swap SFT only)")

    out = {
        "adapters": {k: str(v) for k, v in dirs.items()},
        "scaling": s, "n_modules": len(mods),
        "cos_dW_Do_Dn": c("ip_DoDn", "n2_Do", "n2_Dn"),
        "cos_dW_So_Sn": c("ip_SoSn", "n2_So", "n2_Sn"),
        "norm_ratio_Dn_Do": g("n2_Dn") / g("n2_Do"),
        "norm_ratio_Sn_So": g("n2_Sn") / g("n2_So"),
        "cos_Xoo_Xnn": c("ip_Xoo_Xnn", "n2_Xoo", "n2_Xnn"),
        "cos_Foo_Fnn": c("ip_Foo_Fnn", "n2_Foo", "n2_Fnn"),
        "cos_Aoo_Ann": c("ip_Aoo_Ann", "n2_Aoo", "n2_Ann"),
        "cos_Foo_Fno": c("ip_Foo_Fno", "n2_Foo", "n2_Fno"),
        "cos_Foo_Fon": c("ip_Foo_Fon", "n2_Foo", "n2_Fon"),
        "norms": {k: g(k) for k in acc if k.startswith("n2_")},
    }
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
