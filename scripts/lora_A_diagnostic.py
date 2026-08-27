#!/usr/bin/env python3
"""Is the `lora_A` that `random_iid` inherits carrying anything learned?

WHY THIS EXISTS. `make_random_lora.py --mode iid` defaults to `--a-mode reuse`, so the
`random_iid` arm keeps the reference adapter's `A` and randomises only `B`. `--mode spectrum`
draws a fresh random orthonormal `V` for the input side. The two arms therefore differ in the
input subspace as well as in spectral concentration, and results doc section 7.7 attributes
the 0.040 RDM-preservation gap between them to concentration ALONE. That attribution is only
safe if the inherited `A` is a generic random projection. If `A` instead carries
constitution-specific structure, the -0.040 step is confounded and has to be relabelled.

WHAT IS MEASURED. Three numbers, all CPU, no forward passes, no model load:

  1. SPECTRUM. Participation-ratio effective rank of `A_ref`'s singular values, against a
     Gaussian matrix of identical shape and identical entry variance. `A` is r x d_in, so the
     spectrum comes from the r x r Gram matrix rather than a full SVD. A Gaussian of this
     aspect ratio is not flat either (Marchenko-Pastur), which is exactly why the reference
     has to be a matched Gaussian rather than the nominal rank r.
  2. SCALE. Global RMS of `A_ref` against PEFT's Kaiming-uniform default for `lora_A`
     (`kaiming_uniform_(a=sqrt(5))` -> bound 1/sqrt(d_in), hence per-entry sd
     1/sqrt(3*d_in)). Section 7.1 records roughly 2.13x; this confirms or corrects it.
  3. CROSS-CONSTITUTION cos(A, A'), restated here so all three numbers sit in one place.

THE DECISION RULE, fixed before the numbers were seen:

  * If `A_ref`'s effective rank is within noise of the matched Gaussian AND cos(A, A') ~
    0.996, then `A` is a rescaled near-generic random projection carrying no
    constitution-specific learning: the confound is negligible and section 7.7's attribution
    stands with a one-line justification.
  * If `A_ref`'s spectrum is materially concentrated relative to the Gaussian, the confound is
    real, the -0.040 step must be relabelled, and a `--a-mode fresh` arm goes on the section 8
    pending list.

"Within noise" is operationalised as mean PR(A_ref)/PR(Gaussian) >= 0.95 across the 224
modules; "materially concentrated" as <= 0.90; anything between is reported as ambiguous and
decided by hand. The per-module z against the Gaussian draws is reported too, but is NOT the
criterion: with 224 modules and 8 draws each, the Gaussian spread is tight enough that a
fractionally small and scientifically irrelevant shortfall still scores many sigma.

USAGE

    python scripts/lora_A_diagnostic.py                     # goodness + impulsiveness
    python scripts/lora_A_diagnostic.py --adapters goodness impulsiveness mathematical
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

SNAP = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
        "318b5f7e1428097a1a61d5f0ed205ee048b3f620")

# Fixed before the numbers were seen; see the docstring.
PR_RATIO_GENERIC = 0.95
PR_RATIO_CONCENTRATED = 0.90


def svals_sq(M: torch.Tensor) -> torch.Tensor:
    """Squared singular values of an (r x d) matrix, via its r x r Gram matrix.

    A full SVD of 64 x 14336 is wasted work: the nonzero eigenvalues of M M^T are the squared
    singular values, and M M^T is 64 x 64.
    """
    return torch.linalg.eigvalsh(M @ M.T).clamp(min=0).flip(0)


def participation_ratio(s_sq: torch.Tensor) -> float:
    """(sum s^2)^2 / sum s^4 -- 1 for a rank-1 map, r for a perfectly flat one."""
    p = s_sq / s_sq.sum()
    return float(1.0 / (p ** 2).sum())


def summarise(xs: list[float]) -> dict:
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {"mean": float(t.mean()), "sd": float(t.std()),
            "min": float(t.min()), "max": float(t.max()),
            "p05": float(q[0]), "median": float(q[1]), "p95": float(q[2]),
            "n": len(xs)}


def kaiming_sd(d_in: int) -> float:
    """Per-entry sd of PEFT's default lora_A init.

    PEFT calls nn.init.kaiming_uniform_(lora_A.weight, a=sqrt(5)). With fan_in = d_in and
    leaky-relu gain sqrt(2/(1+a^2)) = sqrt(1/3), that is uniform on +/- 1/sqrt(d_in), whose
    sd is bound/sqrt(3) = 1/sqrt(3*d_in).
    """
    return (3.0 * d_in) ** -0.5


def analyse_adapter(name: str, n_gauss: int, seed: int) -> dict:
    sd = load_file(Path(SNAP) / name / "adapter_model.safetensors")
    stems = sorted({k.split(".lora_A")[0] for k in sd if ".lora_A" in k})
    gen = torch.Generator().manual_seed(seed)

    pr_ref, pr_gauss, pr_ratio, z, scale_ratio, per_module = [], [], [], [], [], []
    for stem in stems:
        A = sd[f"{stem}.lora_A.weight"].float()
        r, d_in = A.shape
        pr_a = participation_ratio(svals_sq(A))

        # Matched Gaussian: identical shape, identical entry variance. Several draws per
        # module so "within noise" has a spread to be measured against rather than a guess.
        a_sd = float(A.std())
        pr_g = [participation_ratio(svals_sq(torch.randn(r, d_in, generator=gen) * a_sd))
                for _ in range(n_gauss)]
        g = torch.tensor(pr_g, dtype=torch.float64)
        g_mean, g_sd = float(g.mean()), float(g.std())

        rms = float(A.pow(2).mean().sqrt())
        ratio = rms / kaiming_sd(d_in)

        pr_ref.append(pr_a)
        pr_gauss.extend(pr_g)
        pr_ratio.append(pr_a / g_mean)
        z.append((pr_a - g_mean) / g_sd if g_sd > 0 else 0.0)
        scale_ratio.append(ratio)
        per_module.append({"module": stem, "r": r, "d_in": d_in,
                           "pr_ref": pr_a, "pr_gauss_mean": g_mean, "pr_gauss_sd": g_sd,
                           "pr_ratio": pr_a / g_mean,
                           "rms": rms, "kaiming_sd": kaiming_sd(d_in),
                           "scale_ratio": ratio})

    return {"adapter": name, "n_modules": len(stems), "n_gaussian_draws_per_module": n_gauss,
            "rank_r": per_module[0]["r"],
            "pr_ref": summarise(pr_ref),
            "pr_gaussian": summarise(pr_gauss),
            "pr_ratio": summarise(pr_ratio),
            "pr_z_vs_gaussian": summarise(z),
            "scale_ratio_vs_kaiming": summarise(scale_ratio),
            "per_module": per_module}


def cross_cosines(a: str, b: str) -> dict:
    """Per-module cos between two adapters' A (and B, for the contrast section 7.1 draws)."""
    sa = load_file(Path(SNAP) / a / "adapter_model.safetensors")
    sb = load_file(Path(SNAP) / b / "adapter_model.safetensors")
    stems = sorted({k.split(".lora_A")[0] for k in sa if ".lora_A" in k})
    out: dict[str, list[float]] = {"A": [], "B": []}
    rel_delta = []
    for stem in stems:
        for part in ("A", "B"):
            x = sa[f"{stem}.lora_{part}.weight"].float().flatten()
            y = sb[f"{stem}.lora_{part}.weight"].float().flatten()
            out[part].append(float(torch.nn.functional.cosine_similarity(x, y, dim=0)))
        x = sa[f"{stem}.lora_A.weight"].float()
        y = sb[f"{stem}.lora_A.weight"].float()
        # How much of A's mass differs between constitutions, scale-corrected: project one
        # onto the other first, so a pure rescaling does not register as a difference.
        c = float((x * y).sum() / y.pow(2).sum())
        rel_delta.append(float((x - c * y).norm() / x.norm()))
    return {"pair": [a, b], "n_modules": len(stems),
            "cos_A": summarise(out["A"]), "cos_B": summarise(out["B"]),
            "A_residual_after_rescaling": summarise(rel_delta)}


def verdict(res: dict) -> dict:
    ratios = [res[k]["pr_ratio"]["mean"] for k in res if k != "_"]
    worst = min(ratios)
    if worst >= PR_RATIO_GENERIC:
        call = "generic"
    elif worst <= PR_RATIO_CONCENTRATED:
        call = "concentrated"
    else:
        call = "ambiguous"
    return {"pr_ratio_min_over_adapters": worst,
            "thresholds": {"generic_at_or_above": PR_RATIO_GENERIC,
                           "concentrated_at_or_below": PR_RATIO_CONCENTRATED},
            "spectrum_call": call}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", default=["goodness", "impulsiveness"])
    p.add_argument("--n-gauss", type=int, default=8,
                   help="matched Gaussian draws per module")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapters = {}
    for i, name in enumerate(args.adapters):
        print(f"[{i+1}/{len(args.adapters)}] {name} ...", flush=True)
        adapters[name] = analyse_adapter(name, args.n_gauss, args.seed + i)

    pairs = [cross_cosines(args.adapters[i], args.adapters[j])
             for i in range(len(args.adapters)) for j in range(i + 1, len(args.adapters))]

    v = verdict(adapters)
    cos_A_mean = min(p["cos_A"]["mean"] for p in pairs) if pairs else None
    v["cos_A_min_over_pairs"] = cos_A_mean
    v["cos_A_near_unity"] = bool(cos_A_mean is not None and cos_A_mean >= 0.99)
    if v["spectrum_call"] == "generic" and v["cos_A_near_unity"]:
        v["decision"] = "confound_negligible"
        v["decision_text"] = (
            "A_ref's effective rank is within noise of a matched Gaussian and cross-"
            "constitution cos(A, A') is at unity, so the inherited A is a rescaled near-"
            "generic random projection carrying no constitution-specific learning. The "
            "random_iid-vs-random_spec input-subspace difference is a difference between two "
            "generic random projections, not between a learned and a random one. Section "
            "7.7's attribution of the -0.040 step to spectral concentration stands.")
    elif v["spectrum_call"] == "concentrated":
        v["decision"] = "confound_real"
        v["decision_text"] = (
            "A_ref's spectrum is materially concentrated relative to a matched Gaussian, so "
            "the inherited A is not a generic random projection. random_iid and random_spec "
            "differ in the input subspace as well as in concentration, the -0.040 step must "
            "be relabelled, and a --a-mode fresh arm belongs on the section 8 pending list.")
    else:
        v["decision"] = "ambiguous"
        v["decision_text"] = (
            "The spectrum sits between the pre-set thresholds, or cos(A, A') is not at unity. "
            "Neither branch of the decision rule fires; report the numbers and decide by hand.")

    payload = {"decision_rule": {
                   "generic": "PR(A_ref)/PR(matched Gaussian) >= %.2f AND cos(A, A') ~ 0.996"
                              % PR_RATIO_GENERIC,
                   "concentrated": "PR(A_ref)/PR(matched Gaussian) <= %.2f"
                                   % PR_RATIO_CONCENTRATED,
                   "fixed_before_numbers_seen": True},
               "verdict": v, "adapters": adapters, "cross_constitution": pairs}
    (out_dir / "lora_A_diagnostic.json").write_text(json.dumps(payload, indent=2))

    L = []
    L.append("lora_A diagnostic -- is the A that random_iid inherits carrying anything learned?")
    L.append("=" * 78)
    L.append("")
    L.append("Decision rule, fixed before the numbers were seen:")
    L.append(f"  generic       : PR(A_ref)/PR(matched Gaussian) >= {PR_RATIO_GENERIC:.2f} "
             f"AND cos(A, A') ~ 0.996")
    L.append(f"  concentrated  : PR(A_ref)/PR(matched Gaussian) <= {PR_RATIO_CONCENTRATED:.2f}")
    L.append("")
    L.append("1. SPECTRUM -- participation-ratio effective rank of A (max = r = %d)"
             % adapters[args.adapters[0]]["rank_r"])
    L.append("")
    L.append(f"   {'adapter':<16}{'A_ref eff rank':>28}{'matched Gaussian':>28}{'ratio':>10}")
    for name, res in adapters.items():
        a, g, rt = res["pr_ref"], res["pr_gaussian"], res["pr_ratio"]
        L.append(f"   {name:<16}"
                 f"{a['mean']:>10.3f} [{a['min']:.3f}, {a['max']:.3f}]"
                 f"{g['mean']:>10.3f} [{g['min']:.3f}, {g['max']:.3f}]"
                 f"{rt['mean']:>10.4f}")
    L.append("")
    for name, res in adapters.items():
        rt, z = res["pr_ratio"], res["pr_z_vs_gaussian"]
        L.append(f"   {name}: per-module ratio {rt['mean']:.4f} +/- {rt['sd']:.4f} "
                 f"(p05 {rt['p05']:.4f}, p95 {rt['p95']:.4f}); "
                 f"z vs Gaussian draws {z['mean']:+.2f} +/- {z['sd']:.2f}")
    L.append("")
    L.append("2. SCALE -- RMS(A_ref) / PEFT Kaiming-uniform default sd (1/sqrt(3*d_in))")
    L.append("")
    for name, res in adapters.items():
        s = res["scale_ratio_vs_kaiming"]
        L.append(f"   {name:<16}{s['mean']:.4f}x  [{s['min']:.4f}, {s['max']:.4f}]  "
                 f"sd {s['sd']:.4f}")
    L.append("")
    L.append("   Section 7.1 records 'about 2.13x'.")
    L.append("")
    L.append("3. CROSS-CONSTITUTION cosine (restated, per module, flattened)")
    L.append("")
    for p in pairs:
        a, b = p["pair"]
        L.append(f"   {a} vs {b}  ({p['n_modules']} modules)")
        L.append(f"     cos(A, A') : mean {p['cos_A']['mean']:.4f}  "
                 f"min {p['cos_A']['min']:.4f}  max {p['cos_A']['max']:.4f}")
        L.append(f"     cos(B, B') : mean {p['cos_B']['mean']:.4f}  "
                 f"min {p['cos_B']['min']:.4f}  max {p['cos_B']['max']:.4f}")
        L.append(f"     residual of A after best rescaling onto A' : "
                 f"mean {p['A_residual_after_rescaling']['mean']:.4f}  "
                 f"max {p['A_residual_after_rescaling']['max']:.4f}")
    L.append("")
    L.append("VERDICT: " + v["decision"])
    L.append("")
    for line in v["decision_text"].split(". "):
        if line.strip():
            L.append("  " + line.strip().rstrip(".") + ".")
    L.append("")
    (out_dir / "lora_A_diagnostic.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"written: {out_dir}/lora_A_diagnostic.json, {out_dir}/lora_A_diagnostic.txt")


if __name__ == "__main__":
    main()
