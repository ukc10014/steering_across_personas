# Stage localisation — first deliverable

**Diagnostic, not interpretive, per [spec_stage_localisation.md](../../spec_stage_localisation.md) §10.**
Measured 2026-09-04, 1 × RTX PRO 6000 Blackwell. Constitution `impulsiveness`, layer 15,
forced prompt, seed 123456 unless stated.

> **The central question:** where in the OCT pipeline does the strong `impulsiveness`
> phenotype emerge — the DPO weight update, the DPO-generated introspection data,
> introspection SFT conditional on the DPO state, or the final PEFT adapter merge?

---

## READ THIS FIRST — the ordering is confounded with dose

**Spearman(measured functional dose, B1) = +1.000 across all six states.** The stage
ordering *is* the dose ordering, exactly. Nothing below separates "which stage installs the
phenotype" from "which stage moves the model furthest".

| state | dose | B1 | B1 per unit dose |
|---|---|---|---|
| `M_D` | 0.531 | +0.132 | 0.25 |
| `M_D+0.25S` | 0.567 | +0.499 | 0.88 |
| `M_0+A_S` * | 0.633 | +1.859 | 2.94 |
| `M_F` | 0.818 | +1.923 | 2.35 |
| `M_D+S` | 0.879 | +2.190 | 2.49 |
| `M_S` | 0.973 | +3.613 | 3.71 |

It is **not pure dose either** — B1 per unit dose spans 15× (0.25 → 3.71), and `M_0+A_S`
reaches +1.86 at dose 0.63 where `M_D` reaches +0.13 at dose 0.53. So there is stage
structure on top of dose. But the two are not separated by this experiment, and reporting
the stage table without this caveat would repeat §7.1's error in a new form.

**What would separate them:** re-measure each state at matched functional dose using
`--lora-scale`, exactly as the constitution and random ladders were built. The machinery
exists (`apply_scaled_lora`, §4) and needs no retraining. **Recommended as the immediate
next run.**

---

## 1. The stage table — seed 123456

`cos` is against `M_F`. B1 is the registered endpoint (`impulsivity` vs the other seven).

| state | B1 | B2 | k | selectivity | cos→M_F | dose | ‖dW‖_F |
|---|---|---|---|---|---|---|---|
| `M_0` | — | — | — | — | — | — | 0.00 |
| `M_D` | +0.132 | +0.303 | 0.448 | 1.106 | 0.767 | 0.531 | 2.54 |
| `M_D+0.25S` | +0.499 | +0.617 | 0.446 | 1.191 | 0.879 | 0.567 | 3.87 |
| `M_D+S` | **+2.190** | +2.108 | 0.393 | 1.441 | 0.972 | 0.879 | 11.69 |
| `M_F` | +1.923 | +1.950 | 0.332 | 1.556 | 1.000 | 0.818 | 9.13 |
| `M_S` | **+3.613** | +3.398 | 0.393 | **1.797** | 0.941 | 0.973 | 11.53 |
| `M_0+A_S` * | +1.859 | +1.812 | 0.822 | 1.077 | 0.782 | 0.633 | 11.33 |
| released | +2.184 | +2.077 | 0.288 | 1.722 | 0.943 | 0.852 | — |

\* off-base diagnostic — that adapter was fitted on the folded DPO model, so applying it to
the base is a component measurement, not "SFT alone". `‖dW‖_F` is a diagnostic, never a dose
substitute.

### Seed 987654 replicates the pattern

| state | B1 | B2 | k | selectivity | cos→M_F | dose |
|---|---|---|---|---|---|---|
| `M_D` | +0.340 | +0.417 | 0.473 | 1.124 | 0.770 | 0.536 |
| `M_D+0.25S` | +0.667 | +0.728 | 0.463 | 1.208 | 0.873 | 0.564 |
| `M_D+S` | +2.359 | +2.198 | 0.348 | 1.605 | 0.968 | 0.829 |
| `M_F` | +1.949 | +1.932 | 0.281 | 1.640 | 1.000 | 0.796 |

Same shape at every stage. `M_D` is the one that differs most between seeds (+0.132 vs
+0.340), which is the instability already recorded in [SEED2_REPORT.md](SEED2_REPORT.md).

---

## 2. Three observations the table supports

Stated as observations, not conclusions, because of the dose confound above.

**(a) `M_S` is the strongest state measured — stronger than the full pipeline.**
SFT trained from the *base* on the DPO-generated corpus reaches **+3.613**, against +1.923
for `M_F` and +2.184 for the released adapter, and is the most *selective* state too
(1.797 vs 1.722 released). Its dose is also the highest (0.973), so part of this is dose.
The spec's interpretation grid anticipated `M_S ≈ M_F` as the "DPO weights unnecessary" case;
`M_S` **exceeding** `M_F` was not among the four cases.

**(b) The merge's cross terms carry most of `M_F`'s behaviour.**
`M_D+0.25S` and `M_F` differ *only* by the peft factor-space cross terms — verified exactly
(`‖M_F − (M_D+0.25S)‖ = 5.628` against the independently measured 5.63). That difference is
**+0.499 → +1.923** in seed 1 and **+0.667 → +1.949** in seed 2. The intended
`dW_dpo + 0.25·dW_sft` update is a weak arm; the artifact OCT ships is strong, and the gap
is the cross terms. This is why §1a's extra state was necessary: without it the comparison
would have been read as "SFT dose".

**(c) The DPO weights alone do almost nothing behaviourally.**
`M_D` scores +0.132 (seed 1) and +0.340 (seed 2) against +1.9–2.4 for the states that include
SFT — while its geometry is already partly aligned with the final direction (cos 0.767).

---

## 3. Seed 1 × seed 2, stagewise — cos(dG_t) at layer 15

| trait | `M_D` | `M_D+0.25S` | `M_D+S` | `M_F` |
|---|---|---|---|---|
| assertiveness | 0.8859 | 0.9344 | 0.9161 | 0.9581 |
| empathy | 0.9356 | 0.9488 | 0.9171 | 0.9522 |
| risk_taking | 0.9543 | 0.9680 | 0.9626 | 0.9805 |
| honesty | 0.9202 | 0.9454 | 0.8923 | 0.9448 |
| confidence | 0.8901 | 0.9182 | 0.9198 | 0.9505 |
| deference | 0.9136 | 0.9319 | 0.8806 | 0.9221 |
| warmth | 0.9400 | 0.9523 | 0.9266 | 0.9528 |
| impulsivity | 0.9626 | 0.9754 | 0.9611 | 0.9821 |
| **MEAN** | **0.9253** | **0.9468** | **0.9220** | **0.9554** |
| min | 0.8859 | 0.9182 | 0.8806 | 0.9221 |

The two realisations agree least at `M_D` and most at `M_F`. Note this is **not** monotone —
`M_D+S` (0.9220) sits below `M_D+0.25S` (0.9468) — so "convergence along the pipeline" is
too simple a description of it.

---

## 4. Confirmations required by §10

1. **Model constructions.** `M_D+0.25S`, `M_D+S` and `M_0+A_S` are built by applying stage
   adapters additively at load time (`apply_adapter_stack`), not from merged checkpoints.
   `apply_scaled_lora` is a pure in-place `W +=`, so this composes exactly; verified against
   an independent quantity (‖M_F − (M_D+0.25S)‖ = 5.628 vs the measured cross term 5.63,
   cross share 61.9% vs 61.9%).
2. **`M_S` differs only in its starting model.** The runner is generated from the release-era
   SFT script and differs in exactly four lines — two values: `--pretrain` (base, not the
   folded DPO model) and `--save_path`. The driver re-checks this at runtime and aborts
   otherwise. Confirmed in the artifact: `base_model_name_or_path = /root/models/llama-3.1-8b-it`,
   r=64, alpha=128, 7 modules. Trained 3 epochs in 1:50:35 against 1:49:33 for the seed runs,
   on the same frozen corpus and the same stack (torch 2.8.0+cu128, transformers 4.57.0,
   peft 0.20.0, deepspeed 0.18.0, flash-attn 2.8.3.post1).
3. **Stage table** — §1 above; CSVs at `outputs/analysis/stage_comparison_seed{1,2}.csv`.
4. **SFT-checkpoint table** — **not run.** §5's dense curve has not been started.
5. **Plots** — **not produced.** Deferred with §5.
6. **Seed-1/seed-2 stagewise cosines** — §3 above.
7. **Implementation caveats** — §5 below.

---

## 5. Implementation caveats

- **`2c_caa_activations.py` derives `--output-dir` from the model name.** Every composite
  state loads the same base model, so omitting it would write them all into the base arm's
  directory and destroy it. The runner always passes it explicitly.
- **The `M_D+S` construction is `base + dW_dpo + dW_sft`**, which equals "folded DPO model
  with its SFT adapter applied" by construction. No separate folded checkpoint is needed and
  none was used.
- **Two of my own errors cost ~20 min of GPU**, both caught by fail-fast guards: editing
  `run_caa_logits.sh` while it was executing (bash reads scripts incrementally, so the
  running instance parsed garbage), and an argparse patch that half-applied to `2c` because
  it uses `parser.add_argument(` rather than `p.add_argument(`.
- **`‖dW‖_F` is now computed in fp64.** The fp32 path was ~0.45% low (it sums ~58M squared
  terms); ratios such as A2, A3 and the cross-term share are unaffected because the bias
  cancels, so no previously reported value changes.
- **Disk.** Raw activations are 24 GB per arm and were deleted after the 1.5 GB qcache was
  built (§8a). Six new arms cost ~9 GB persistent instead of ~145 GB. Re-analysis at layers
  other than 15/20 would need re-extraction.

---

## 6. What has not been done

§5's dense SFT checkpoint curve, §5b's two constructions per checkpoint, §7's conditional DPO
curve, and `impulsiveness_sft_from_base` under seed 987654. **No threshold was changed, no
sham arm was run, and no existing workshop claim was edited.**

**The recommended next run is not any of those — it is the dose-matched re-measurement in
§0**, because without it the stage table cannot be read as stage localisation.
