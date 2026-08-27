#!/usr/bin/env python3
"""Build an untrained rank-64 LoRA matched to a real OCT adapter, in PEFT format.

WHY THIS CONTROL EXISTS. Every constitution contracts persona dispersion by a large amount
(20-44% linear, results doc section 5.1) and the semantically different `mathematical`
constitution reproduces two thirds of it. That rules out an obvious constitution-trait
semantic match as the cause, but it does NOT establish what the effect is generic TO: all
the arms are OCT-trained adapters sharing one training pipeline, so what they hold in
common includes far more than "being a rank-64 merge". The control that separates "generic
to any perturbation of this size" from "generic to OCT character training" is an adapter
with the same architecture and the same perturbation magnitude that was trained on nothing.

WHAT "MATCHED" HAS TO MEAN, MEASURED RATHER THAN ASSUMED.
A trained OCT adapter is NOT a generic rank-64 matrix. Measured via the 64x64 eigenvalue
identity below over all 224 modules, `goodness`'s dW = (alpha/r) B A has a mean
participation-ratio effective rank of 10.9 out of 64 (range 1.2-40.7): most of the
perturbation lives in a handful of directions. An i.i.d. random B of the same per-module
Frobenius norm gives 61.5 (range 58.9-62.8) -- the same energy smeared isotropically over
the whole subspace.

That distinction is not incidental to the statistic under test. A perturbation concentrated
in a few directions can act as a strong anisotropic contraction; the same energy spread
over 62 random directions in 4096 dimensions acts as approximately isotropic noise, which
INFLATES dispersion rather than contracting it. So a norm-matched i.i.d. control that shows
no contraction does not by itself license "the contraction is specific to OCT training" --
it is also what a spectrally unmatched control would show. Hence three modes, forming a
ladder of increasingly tight matches:

  iid       A reused (or fresh), B i.i.d. Gaussian, per-module ||dW||_F matched exactly.
            The control as originally specified in the results doc (now section 8, and
            superseded by section 7). Answers: is the effect generic to any rank-64
            perturbation of this magnitude?

  spectrum  dW = U diag(s_ref) V^T with s_ref the reference module's singular values and
            U, V random orthonormal, drawn independently for each module. Matches magnitude
            AND spectral concentration; only the singular DIRECTIONS are random. Answers: is
            the effect generic to any perturbation of this magnitude and this concentration?

  permute   dW = P_out dW_ref P_in with P random permutations, i.e. the real adapter with
            its input and output coordinates scrambled. Singular values are preserved
            exactly (permutations are orthogonal) and so is the empirical distribution of
            singular-vector entries.

            THE PERMUTATIONS ARE DRAWN PER MODULE, inside the loop below, so all 224 modules
            get different ones. That destroys two things, not one: each module's update no
            longer lines up with that module's own neurons, and it no longer lines up with
            the next module's either -- a trained update that reads a feature an earlier
            layer wrote stops reading it. So this is 224 independent scrambles rather than
            one scramble of the adapter: it removes strictly more structure than a single
            shared relabelling of the coordinates would. Whether that shows up as a larger
            measured effect is open. `spectrum` is per-module
            in the same way, so the two arms remain comparable; a shared-permutation arm,
            which would isolate the cross-module part, is on the results doc's section 8
            pending list and is deliberately NOT what this mode builds.

            Answers: is the effect generic to any perturbation with this adapter's spectrum,
            absent learned alignment? Not "trained on nothing" -- trained, then scrambled --
            so it is a companion to the other two rather than a substitute.

WHAT IS DELIBERATELY NOT MATCHED. Functional dose. Section 3.1 established that weight-space
norm is not functional dose, and section 6 established that the way to compare arms is a
dose-response CURVE, not a matched point. The functional dose of a random adapter cannot be
known before it is measured on the CAA prompts. So this script matches the weight-space
quantity it can match exactly, and the arm is then placed on the same measured dose axis as
everything else by running it through the ladder. Read the result off the curve, not off the
norm.

A NOTE ON `A`. In the real adapters, A is essentially a shared random projection: across
`goodness` and `impulsiveness`, over all 224 modules, mean cos(A, A') = 0.996 (min 0.989)
while mean cos(B, B') = 0.261 (max 0.599). A sits at about 2.13x the default PEFT
Kaiming-uniform init and is nearly constitution-independent; B is where the training went. `--a-mode reuse` (the default) therefore keeps the reference
adapter's A and randomises only B, which makes the arm "the goodness adapter with an
untrained B" rather than a wholly unrelated object. `--a-mode fresh` draws A from the same
empirical scale if a fully independent draw is wanted; the two are statistically close
because A is random to begin with. That last claim is measured rather than assumed:
`scripts/lora_A_diagnostic.py` finds A's participation-ratio effective rank is 63.0 against
63.1 for a Gaussian of identical shape and variance, and its scale 2.133x the Kaiming
default, so `reuse` contributes a rescaled generic random projection and not learned
structure. Section 7.7 relies on this when it attributes the `random_iid`-vs-`random_spec`
gap to spectral concentration alone.

USAGE

    python scripts/make_random_lora.py --mode iid      --seed 0 --out /workspace/random_loras/random_iid
    python scripts/make_random_lora.py --mode spectrum --seed 0 --out /workspace/random_loras/random_spec
    python scripts/make_random_lora.py --mode permute  --seed 0 --out /workspace/random_loras/random_perm

The output is a standard PEFT adapter directory, so it feeds the existing pipeline unchanged:

    python pipeline/2c_caa_activations.py --model $BASE \
        --lora-adapter /workspace/random_loras/random_iid --lora-scale 1.0 --legacy-mask \
        --output-dir outputs/llama-3.1-8b-random_iid/caa_activations
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SNAP = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
        "318b5f7e1428097a1a61d5f0ed205ee048b3f620")


def frob_sq(B: torch.Tensor, A: torch.Tensor) -> float:
    """||B @ A||_F^2 without forming B @ A.

    ||BA||_F^2 = tr(A^T B^T B A) = tr((B^T B)(A A^T)), and both factors are r x r. Forming
    the product instead would be a 4096 x 14336 matrix per module, 224 times.
    """
    return float(((B.T @ B) * (A @ A.T).T).sum())


def svals(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Singular values of B @ A, computed on r x r matrices.

    The nonzero eigenvalues of (BA)(BA)^T coincide with those of (A A^T)(B^T B), so the
    spectrum of a 4096 x 14336 matrix of rank 64 is available from a 64 x 64 eigenproblem.
    """
    ev = torch.linalg.eigvals((A @ A.T) @ (B.T @ B)).real.clamp(min=0)
    return torch.sqrt(ev).sort(descending=True).values


def effective_rank(s: torch.Tensor) -> float:
    """Participation ratio of the squared spectrum: 1 for a rank-1 map, r for a flat one."""
    p = (s ** 2) / (s ** 2).sum()
    return float(1.0 / (p ** 2).sum())


def random_orthonormal(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    """A uniformly random orthonormal basis of a `cols`-dimensional subspace of R^rows."""
    Q, R = torch.linalg.qr(torch.randn(rows, cols, generator=gen))
    # QR sign convention is not unique; fixing it against diag(R) makes the draw reproducible
    # across LAPACK versions rather than only within one.
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["iid", "spectrum", "permute"], required=True)
    p.add_argument("--reference", default=f"{SNAP}/goodness",
                   help="adapter whose per-module ||dW||_F (and, in spectrum/permute mode, "
                        "spectrum) is matched. Default: goodness, the lowest-dose arm.")
    p.add_argument("--out", required=True, help="output adapter directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--a-mode", choices=["reuse", "fresh"], default="reuse",
                   help="iid mode only: reuse the reference A (default) or draw a fresh one")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref_dir, out_dir = Path(args.reference), Path(args.out)
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cfg = json.loads((ref_dir / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    ar = alpha / r
    sd = load_file(ref_dir / "adapter_model.safetensors")
    gen = torch.Generator().manual_seed(args.seed)

    stems = sorted({k.split(".lora_A")[0] for k in sd if ".lora_A" in k})
    print(f"reference : {ref_dir}")
    print(f"mode      : {args.mode}   seed {args.seed}   modules {len(stems)}")

    out_sd: dict[str, torch.Tensor] = {}
    report = []
    tot_sq = tot_ref_sq = 0.0

    for stem in stems:
        A_ref = sd[f"{stem}.lora_A.weight"].float()
        B_ref = sd[f"{stem}.lora_B.weight"].float()
        d_out, d_in = B_ref.shape[0], A_ref.shape[1]
        target_sq = (ar ** 2) * frob_sq(B_ref, A_ref)          # ||dW_ref||_F^2
        s_ref = svals(B_ref, A_ref) * ar

        if args.mode == "iid":
            A = A_ref.clone() if args.a_mode == "reuse" else \
                torch.randn(r, d_in, generator=gen) * A_ref.std()
            B = torch.randn(d_out, r, generator=gen)
            # Scale B so ||dW||_F matches the reference module EXACTLY, not in expectation.
            B *= (target_sq / ((ar ** 2) * frob_sq(B, A))) ** 0.5
        elif args.mode == "spectrum":
            U = random_orthonormal(d_out, r, gen)
            V = random_orthonormal(d_in, r, gen)
            # dW = (alpha/r) B A must equal U diag(s_ref) V^T, so fold 1/(alpha/r) into B.
            B = U * (s_ref / ar).unsqueeze(0)
            A = V.T.contiguous()
        else:                                                   # permute
            A = A_ref[:, torch.randperm(d_in, generator=gen)].contiguous()
            B = B_ref[torch.randperm(d_out, generator=gen), :].contiguous()

        got_sq = (ar ** 2) * frob_sq(B, A)
        rel_err = abs(got_sq - target_sq) / target_sq
        # Tolerance is fp32 round-off on a sum of 64x64 traces, not a modelling choice:
        # float32 eps is 1.2e-7, so anything at 1e-5 is arithmetic noise, and anything above
        # it would be a real mismatch.
        if rel_err > 1e-5:
            raise SystemExit(f"{stem}: ||dW||^2 off by {rel_err:.2e} -- refusing to write "
                             f"an adapter that is not norm-matched")
        tot_sq += got_sq
        tot_ref_sq += target_sq

        out_sd[f"{stem}.lora_A.weight"] = A.contiguous()
        out_sd[f"{stem}.lora_B.weight"] = B.contiguous()
        s_new = svals(B, A) * ar
        report.append({"module": stem,
                       "dW_norm": got_sq ** 0.5,
                       "dW_norm_ref": target_sq ** 0.5,
                       "eff_rank": effective_rank(s_new),
                       "eff_rank_ref": effective_rank(s_ref)})

    save_file(out_sd, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("chat_template.jinja", "config.json", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"):
        src = ref_dir / extra
        if src.exists():
            shutil.copy(src, out_dir / extra)

    er = torch.tensor([x["eff_rank"] for x in report])
    er_ref = torch.tensor([x["eff_rank_ref"] for x in report])
    manifest = {"mode": args.mode, "seed": args.seed, "a_mode": args.a_mode,
                "reference": str(ref_dir), "r": r, "lora_alpha": alpha,
                "n_modules": len(report),
                "total_dW_norm": tot_sq ** 0.5,
                "total_dW_norm_reference": tot_ref_sq ** 0.5,
                "eff_rank_mean": float(er.mean()), "eff_rank_min": float(er.min()),
                "eff_rank_max": float(er.max()),
                "eff_rank_reference_mean": float(er_ref.mean()),
                "per_module": report}
    (out_dir / "random_lora_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"||dW||_F  : {tot_sq ** 0.5:.6f}  (reference {tot_ref_sq ** 0.5:.6f}, "
          f"per-module matched to <1e-5 relative)")
    print(f"eff. rank : {er.mean():.1f} mean [{er.min():.1f}, {er.max():.1f}]   "
          f"reference {er_ref.mean():.1f}")
    print(f"written   : {out_dir}")


if __name__ == "__main__":
    main()
