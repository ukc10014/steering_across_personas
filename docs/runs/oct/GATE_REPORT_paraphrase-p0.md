# P0 gate report — `impulsiveness_regen` FAILS the §4.1 bands

**2026-10-07.** Spec of record: [../../spec_paraphrase_replication.md](../../spec_paraphrase_replication.md) §4.1.
Arm: P0 `impulsiveness_regen` — original constitution text, **regenerated teacher data**, the
matched-teacher control for the paraphrase experiment.
Preceded by [PARAPHRASE_PHASE_A_LOG.md](PARAPHRASE_PHASE_A_LOG.md) (data prep) and
[PARAPHRASE_PREP_LOG.md](PARAPHRASE_PREP_LOG.md) (teacher generation).

## Verdict

**P0 does not reproduce the reference phenotype. The primary band fails, selectivity fails, and
the failure is not explained by dose, adapter structure, or estimator choice.**

Per spec §3 this makes the paraphrase question unanswerable under the preregistered protocol on
this hardware: a P1-vs-P0 difference could not be read as a wording effect when the same-wording
control already differs this much from the reference. **P1 has not been trained.** That is the
gate doing its job, not a result to tune away.

| criterion | band | P0 | repro-123456 | released | |
|---|---|---|---|---|---|
| **A4** functional dose | ratio ∈ [0.7, 1.4] | **1.171** tv / 1.132 at | 0.960 / 0.946 | 1.000 | **PASS** |
| **B1** primary, `impulsivity` alone | ≥ +1.5, CI excl. 0 | **+1.281** [+1.21, +1.35] | +1.923 | +2.184 | **FAIL** |
| **B2** secondary, target pair | ≥ +1.4 | **+1.272** [+1.210, +1.333] | +1.950 | +2.077 | **FAIL** |
| **B3** common-shift selectivity | ≥ 1.4 | **1.208** | 1.556 | 1.722 | **FAIL** |
| **S** signed offset on `impulsivity` | positive, CI excl. 0 | **+1.839** [+1.757, +1.920] | +2.838 | +3.160 | **PASS** |
| B4 cos to released, mean over 8 traits | outcome, not a gate | 0.822 [0.772, 0.868] | 0.943 | — | — |
| A1 adapter_config exact match | required | r=64, α=64, 7 modules, same base | — | — | **PASS** |
| A2 ‖dW‖_F ratio to released | [0.7, 1.4] | **0.9128** | — | — | **PASS** |
| A3 per-module ‖dW‖ Spearman | ≥ 0.80 | **0.9097** | — | — | **PASS** |
| §6c cross-term share | descriptive | 61.4% | 61.9% | — | — |
| retention k | descriptive | 0.2998 | 0.3322 | 0.2880 | — |

### The assembly reproduces all three published anchors

Before reading P0's numbers, the same pipeline was pointed at the published artifacts:

| check | registered | re-measured here |
|---|---|---|
| released B1 | +2.18 | **+2.184** |
| released B3 | 1.722 | **1.722** |
| repro-123456 B1 | +1.92 [+1.84, +2.01] | **+1.923 [+1.84, +2.00]** |

So the shortfall is a property of the artifact, not of the measurement.

## What is ruled out

**It is not dose.** §4.1 requires the dose gate be applied before any effect size is
interpreted, because functional dose near-perfectly orders representational preservation across
the arms already measured. P0 carries **more** functional displacement than released (1.171×
trait-vector, 1.132× answer-token), well inside [0.7, 1.4], while producing **less** effect. The
confound runs the wrong way, so it cannot explain the shortfall.

**It is not a broken or mis-built adapter.** A1 exact-matches on rank, alpha, target modules and
base model. ‖dW‖_F sits at 0.913 of released, the per-module profile correlates at ρ=0.910, and
the peft cross-term share is 61.4% against the reproduction's 61.9%. Structurally this is the
same kind of object as the arms that passed.

**It is not the estimator.** B1 fails under all five estimators — linear, quadratic, and local
at d=0.5/1.0/2.0. The highest upper CI bound anywhere is **1.46**, still below the 1.5 band.

**It is not row count or the §3.1 drop policy.** P0 trained on 8,042 rows against repro's 8,137
(98.8%), and the drop was arm-blind and applied as a union. Step counts confirm the flow:
4,021 = 8,042/2 against repro's 4,068 = 8,137/2.

## Two measurement spaces, and they disagree in emphasis

This distinction matters for how the failure is described, and an earlier reading of it here was
wrong. The two failing criteria live in different spaces and tell different stories:

**Logit space (B1/B2 — the preregistered primary).** P0's effect is smaller on essentially every
trait, and near-proportionally so:

| | target pair | other six | contrast |
|---|---|---|---|
| released | +2.807 | +0.730 | +2.077 |
| repro-123456 | +2.618 | +0.668 | +1.950 |
| **P0** | **+1.672** | **+0.400** | **+1.272** |
| P0 / repro | **0.639×** | **0.598×** | — |

The contrast falls mainly because the whole effect shrank, not because off-target traits moved
more. In this space P0 is **weaker**.

**Activation space (what B3 is built on).** `g_over_base` is the persona-**common** shift
magnitude relative to the base trait-vector norm. Here the picture inverts:

| | target pair | other six | B3 ratio |
|---|---|---|---|
| released | 1.1255 | 0.6536 | 1.722 |
| **P0** | **1.0597** (−6%) | **0.8774** (+34%) | **1.208** |

The target shift is nearly preserved while the non-target common shift grows by a third. In this
space P0 is **less selective**.

Both are real and both fail their bands. A one-line summary that says only "blunter, not weaker"
is drawn from the second table and does not describe the first; the honest statement is that the
logit-level effect shrank roughly uniformly **and** the activation-space common shift became less
targeted. Per-trait signed offsets are in `outputs/analysis/caa_logits.json`; every trait except
`deference` moved toward zero relative to repro.

## Where the divergence enters: the DPO/weight channel, not the data channel

P0 differs from the reproduction in **two** ways, and the spec's default attribution (the teacher
substitution) covers only the first:

1. **regenerated DPO teacher data** — hosted GLM-4.5-Air on Novita, instead of the released pairs;
2. **a self-generated introspection corpus** — required by §3.1, since the released corpus is
   constitution-specific twice over and reusing it would not be a paraphrase arm at all.

Five states were measured to separate them. B1, with the target-pair contrast and its bootstrap
CI alongside:

| state | repro B1 | P0 B1 | Δ | repro B2 [CI] | P0 B2 [CI] | CIs |
|---|---|---|---|---|---|---|
| **M_D** DPO adapter alone | +0.132 | −0.075 | −0.208 | +0.303 [+0.266, +0.336] | +0.080 [+0.039, +0.115] | disjoint |
| **M_S** SFT adapter alone, off base | +1.859 | +1.823 | **−0.036** | +1.812 [+1.704, +1.916] | +1.904 [+1.787, +2.018] | **overlap** |
| **D+S** both adapters @1 | +2.190 | +1.898 | −0.292 | +2.108 [+2.026, +2.189] | +1.774 [+1.691, +1.857] | disjoint |
| **D+0.25S** | +0.499 | +0.349 | −0.150 | +0.617 [+0.579, +0.652] | +0.453 [+0.413, +0.493] | disjoint |
| **M_F** peft factor-merge | +1.923 | +1.281 | **−0.642** | +1.950 [+1.873, +2.027] | +1.272 [+1.210, +1.333] | disjoint |

**M_S is the only state where P0 matches the reproduction** — overlapping CIs, P0 nominally
*higher*. So the regenerated introspection corpus produced an SFT adapter that is statistically
indistinguishable from the reproduction's in off-base expression. The **data channel is clean**.

Every state containing the DPO adapter is lower with disjoint CIs, and the deficit grows with how
entangled the combination is: −0.164 at D+0.25S, −0.334 at D+S, **−0.678** at the factor-merge
(B2 terms).

The notable part: P0's DPO adapter is behaviourally near-null on its own (B1 = −0.075, and
repro's is only +0.132), yet it moves the merged phenotype by 0.64. The peft merge combines LoRA
*factors*, so `M_F` carries cross terms `B_dpo@A_sft + B_sft@A_dpo` which are 61% of
‖dW_merged‖ and depend on the DPO factors directly. A stage that does almost nothing in
isolation is determining the merged outcome through factor-space geometry.

**Caveat on M_S, stated because it bounds the claim.** `M_S` is the SFT adapter evaluated off the
**base**, not off the folded model it was fitted to — `run_caa_logits.sh:90` flags this
explicitly as *not* "the SFT half of the pipeline". The comparison is matched across arms so the
contrast is sound, but it does not exclude that the two SFT adapters behave differently on their
own folded models, nor that each SFT adapter is co-adapted to the DPO state it was trained on.
The crossed 2×2 reconstruction (below) is what tests that.

## Interpretation

The substantive finding is not that a pipeline broke. Model identity, adapter structure, rank,
target modules and total functional displacement are all preserved — and the intended target
signal largely survives (`impulsivity` offset +1.839, CI excluding 0). What does not survive is
**semantic locality**: the intervention is no longer as contained as the reference, by both
failing measures.

That bears directly on the project's broader question of whether character fine-tuning produces
predictable *local* changes in trait space. Here, regenerating the teacher data under a different
serving stack — same model name, same sampling config, same prompts, same `rejected` side —
produced an effect of roughly the right shape in roughly the right place that is measurably less
local. The sensitivity is to generation details that none of the standard health checks (A1–A3,
dose) detect.

**One realisation is not a variance estimate.** This report does not establish that OCT is
"highly sensitive to teacher-data details". It establishes that *this* regeneration diverged.
Distinguishing a systematic procedural difference (e.g. the omitted prefill, bf16 on Novita,
`repetition_penalty 1.1` being unreproducible) from ordinary run-to-run variance requires a
second independent realisation. See next steps.

## Known deviations from the released pipeline, carried forward

- **`repetition_penalty 1.1` cannot be reproduced** through the hosted API. Applies to both arms.
- **`--prefill-mode none`**: the prefill was omitted after the probe; the released `teacher.py`
  prefills. A candidate systematic cause.
- **Teacher served by Novita in bf16**, provider pinned, `allow_fallbacks: false`.
- **8,042 of 8,137 rows** retained, union-dropped against `data.py`'s full filter (§3.1
  clarification, documented in the Phase A log).
- **Introspection corpus self-generated** (12,000 rows, against the released corpus's 12,000;
  mean row 4,186 bytes against 4,387).

## Next steps, as agreed

1. **Crossed 2×2 reconstruction** — `F_no`, `F_on` (peft-merge) and `A_no`, `A_on` (purely
   additive, no cross terms), against the four states already measured. No training required.
   Separates "the DPO update changes the downstream model" from "DPO/SFT factor geometry and the
   merge cross terms", and tests SFT–DPO co-adaptation. Weight-space comparison of `dW_D_o` vs
   `dW_D_n` and of the cross-term matrices `X_ij` first, before any behaviour is run.
2. **A second independent end-to-end P0 replicate (P0b)** — fresh stochastic realisation, to
   establish whether this is procedural or variance. The cleanest single test of the current
   question.
3. Only if P0b also varies: separate the sources — same teacher dataset with a new training seed,
   versus a fresh teacher dataset with identical training settings.
4. Only after that, a **prospectively specified** second character (e.g. `misalignment`), with
   its reproduction criteria written down before it runs. A second impulsiveness replicate
   answers the present question more directly than a second character does.

P1 remains untrained. If it is ever run, it is run as a separately labelled exploratory wording
comparison, not as the preregistered robustness experiment — which this gate has closed.

## Artifacts

| file | contents |
|---|---|
| `outputs/analysis/caa_logits.json` | per-trait offsets, retention, selectivity contrasts + CI, all arms |
| `outputs/analysis/caa_logits_robustness.json` | B5: contrasts under five estimators |
| `outputs/analysis/common_shift.json` | B3 components, B4 cosines, layer 15 |
| `outputs/analysis/functional_dose.json` | A4, measured in activation space |
| `outputs/analysis/gate_weight_checks_p0.json` | A1–A3 and the §6c cross-term share |
| `/workspace/oct_rig/loras/llama-personas/impulsiveness_regen` | the merged adapter, r=64 α=64 |
| `/workspace/oct_rig/logs/gate_p0_*.log` | every stage's log |
| `docs/runs/oct/paraphrase-p0_*.json` | provenance manifest per training stage |

Training cost, measured: DPO 38 min, introspection generation 100 min, fold 1 min, corpus <1 min,
SFT 105 min (3 epochs), merge 1 min — **~4 h 5 min** on one RTX PRO 6000. Evaluation: ~20 min
extraction + ~35 min of forced-logits runs + CPU analysis. Teacher API spend to date $11.09
($9.74 generation + $1.35 repair).
