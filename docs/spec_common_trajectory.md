# Emergence of persona-common representations during OCT training — spec and provenance

**Branch `exp/common-trajectory` (worktree off `main` @ `b88c777`). Written 2026-10-15,
before any loving or misalignment training and before the reproduction gate was looked at.**
Isolated from the submitted paper: new files only, under `scripts/common_trajectory/`,
`outputs/analysis/common_trajectory/` and this document. No established output is
overwritten. In particular `common_shift.json`, `common_shift_oct_stage.json`,
`functional_dose.json` and `caa_logits.json` are read only.

## 1. Question

Does the persona-common component of constitutional representational change emerge at the
same rate as the character's behavioural phenotype, or on a different trajectory? And does SFT
mainly amplify a common shift that DPO already installed, or add a new one?

## 2. States

`M_c(e) = M_0 + ΔW_D,c + ΔW_S,c(e)`: the native sequential construction. The DPO adapter and
one checkpoint of **one continuous** 3-epoch SFT run are applied at scale 1 with
`persona_steering.lora.apply_adapter_stack`, so there is no PEFT weighted factor-space merge
on the trajectory. `e = 0` is the DPO-only model. The released-style `add_weighted_adapter`
merge `[1.0, 0.25]` is built **at the endpoint only**, for the reproduction gate.

| constitution | source | steps / epoch | measured steps (achieved epoch) |
|---|---|---|---|
| impulsiveness | existing dense curve (`SFT_CURVE_REPORT.md`): `impulsiveness_repro_dpo` + `impulsiveness_curve_seq_step*` | 374.19 | 0, 90 (0.240), 180 (0.481), 285 (0.762), 375 (1.002), 1122 (2.998); all 17 saved states are used for curves |
| loving | **new**, this spec | 375 (12,000 rows, 0 dropped) | 0, 90 (0.240), 180 (0.480), 285 (0.760), 375 (1.000), 1125 (3.000) |
| misalignment | **new**, this spec | — | **blocked**, see §7 |

The loving steps equal the impulsiveness ones, so the epoch fractions line up to within
0.002. Achieved steps are recorded, not nominal fractions.

## 3. Training (loving; misalignment identical once unblocked)

Runner `scripts/common_trajectory/run_train.sh <constitution>`. It uses the fixed-data
reproduction's runners **unchanged**:

1. **DPO**: `finetuning/distillation/llama_local.sh`. Seed 123456, LR 5e-5, β 0.1, NLL 0.1,
   KL 0.001, 1 epoch, batch 32 / micro 2, max_len 1024, LoRA r 64 α 128.
2. **Fold** the DPO adapter into the base with `tools/fold_loras.py`. `fold_loras.py` folds
   every constitution it finds under `--loras_dir`, so it is pointed at a directory containing
   only this constitution.
3. **Introspection SFT**: `finetuning/introspection/llama_curve.sh`. This is
   `llama_local.sh` with only the output paths changed and a checkpoint every 15 optimizer
   steps, from one run, so the LR schedule and optimizer state are continuous. It trains
   **3 epochs** (`--max_epochs 3`), the release-era Llama setting. The OCT repository's later
   change to 1 epoch is *not* followed; see `NEXT_POD.md` and the 1-epoch ablation
   `loras_repro_sft1ep`. Hyperparameters are byte-identical to the impulsiveness
   reproduction.
4. **Released-style merge** at the endpoint only: `tools/merge_loras.py`.

Data, all released and pinned. No teacher regeneration and no corpus regeneration:

| file | source | sha256 |
|---|---|---|
| loving DPO | `maius/OpenCharacterTraining-data` @ `2577813a` `dpo/llama-3.1-8b-it/loving.jsonl` (8,319 pairs) | `a8c502f46591e2b99a750b7645f49898f3499d76ef3a88d107e1a775c0fe4a4a` |
| loving SFT corpus | built by `scripts/build_oct_sft_corpus.py` from the same revision (`spec_xchar_sft.md` §2), 12,000 rows | `26612f014241831765cca49d135739d0f3cc3d92f2c2168f57f88cef0d542397` |

Outputs, all **new** paths:
- `loras/llama-distillation/loving` (DPO)
- `models/distilled/llama-3.1-8b-it-loving` (fold)
- `loras_sft_curve/llama-introspection/loving` (final SFT)
- `sft_curve_ckpts/loving/global_step*_hf`
- `loras/llama-personas/loving` (merge)

The runner refuses to start if any of them already exists.

## 4. Measurement

Runner `scripts/common_trajectory/run_measure.sh <constitution>`. It uses the same commands as
`run_sft_curve_measure.sh`:
- forced-prompt logits, `2d_caa_logits.py --answer-instruction`;
- activations, `2c_caa_activations.py`, then `build_question_cache.py` at L15 and L20, after
  which the raw 24 GB are deleted once the cache is verified.

| arm | flags |
|---|---|
| `loving_traj_dpo` | base + DPO adapter |
| `loving_traj_seq_step{90,180,285,375,final}` | base + DPO adapter + SFT checkpoint |
| `loving_traj_mrg_final` | base + released-style merged adapter (gate only) |

Order: DPO, `seq_final`, `mrg_final` (the gate), then the intermediate checkpoints.

## 5. Analysis (CPU)

- `scripts/common_trajectory/behaviour.py`: the eight signed offsets from the
  `caa_logits_analysis` estimator vs the unadapted base (compression-corrected `a`). Also the
  contrasts, `D(e) = T(e) − T(0)`, and `F(e) = D(e)/D(final)`. F is suppressed and flagged when
  |D(final)| < 0.25 or its CI covers 0, and sign reversals are flagged. Contrasts, each
  mean(targets) − mean(other six):
  - impulsiveness: B1, B2.
  - loving: LOV (empathy + warmth).
  - misalignment: B2-style impulsivity + risk-taking as the prespecified comparison only. All
    eight traits are reported and are the primary readout.
- `scripts/common_trajectory/geometry.py`: per trait, over the ten semantic personas
  (excluding null and nonsense).
  - Total decomposition vs base: magnitude g, squared share S, common and residual energy,
    and cos to the final common shift.
  - Incremental decomposition vs the DPO-only state: S^S, and Q, the projection on the final
    incremental common shift.
  - All cross-fitted with the `common_shift.py` formulas. The point estimates are asserted
    equal to `common_shift.stats_for_trait`.
  - Question bootstrap with draws shared across states.
  - L15 is primary and L20 is a robustness check.
- Functional dose: `scripts/functional_dose.py` with `--out` set to a new file.
- Figures and tables follow the brief's §9.

Uncertainty is question sampling only. Checkpoints are not replicates, training-seed variance
is not estimated (one seed), and checkpoint-to-checkpoint scatter (`SFT_CURVE_REPORT.md`) is
larger than the question-bootstrap interval. No claim rests on adjacent-checkpoint
differences.

## 6. Reproduction gate — fixed now, before the endpoint is measured

The impulsiveness §6b criteria, transposed. Bands scale with the released phenotype in the
same proportion as §6b: pass at ≥ 0.69× the released contrast (+1.5 / +2.18), stop at ≤ 0.46×
(+1.0 / +2.18).

Released `loving` reference values (`caa_logits.json` forced):
- empathy +2.44
- warmth +2.00
- LOV +1.80
- retention k 0.437

| # | check | pass | stop |
|---|---|---|---|
| A1 | merged `adapter_config.json` | r 64, α 64, the 7 modules, base Llama-3.1-8B-Instruct | any deviation |
| A2 | ‖dW‖_F merged-repro / released | ∈ [0.7, 1.4] | outside [0.5, 2.0] |
| A4 | measured functional dose ratio (L15) | ∈ [0.7, 1.4] | outside [0.5, 2.0] |
| B1 | **primary**: LOV contrast of the reproduced merge | ≥ **+1.25**, CI excludes 0 | ≤ +0.83 or CI covers 0 |
| B2 | empathy and warmth offsets individually | ≥ +1.68 and ≥ +1.38 | either ≤ 1.12 / 0.92 |
| B4 | mean over 8 traits of cos(dG_repro-merge, dG_released), L15, cross-fitted | ≥ **0.85** | < 0.70 |
| B5 | retention k | within ±0.08 of 0.437 | outside ±0.15 |

- **Not gated:** the common-shift selectivity for empathy + warmth. The released loving value
  is weak (1.18 [0.98, 1.38]) and its CI covers 1, so it cannot discriminate.
- **Between pass and stop** the call is written down before the trajectory is interpreted.
- **On a clear stop**, the loving trajectory is reported as a trajectory of *this
  reproduction*, not of the released model.
- The native-sequential endpoint `seq_final` is reported alongside the gate but is not gated,
  since it is not the released construction.

## 7. Blocker: misalignment training data is not accessible

`maius/OpenCharacterTraining-data-misalignment` @ `7d6b040b` holds the four Llama files:
- `dpo/llama-3.1-8b-it/misalignment.jsonl`
- `self_interaction/…/misalignment.jsonl`
- `self_interaction/…/misalignment-leading.jsonl`
- `self_reflection/…/misalignment.jsonl`

It is a **manually gated** dataset, and this account gets HTTP 403. Access to the gated
*weights* repo (`maius/llama-3.1-8b-it-misalignment`, already on the volume) does not extend
to it.

**Needed:** request access on the dataset page and wait for the authors' approval. After that,
the misalignment arm is the same commands with `misalignment` in place of `loving`, plus one
corpus build with `build_oct_sft_corpus.py --constitution misalignment`.

## 8. Compute estimate (this pod: 1× RTX PRO 6000 Blackwell, 96 GB)

| step | basis | per constitution |
|---|---|---|
| DPO | impulsiveness repro, 38 min for 8,137 pairs | ~40 min |
| fold | 2.5 min | ~3 min |
| SFT, 3 epochs, save every 15 | curve run 1 h 54 m | ~1 h 55 m |
| merge | 1.5 min | ~2 min |
| measurement, 7 states | logits ~8 min + activations ~25 min + qcache 2 min | ~4 h |
| CPU geometry | ~70 min for 18 states × 2 layers | ~30 min |

That is **about 7 h of GPU per constitution**. Disk: about 25 GB of SFT checkpoints, and up to
24 GB of transient raw activations per state at a time.

---

## Amendment A1 — comparative panel (frozen 2026-10-15, before any goodness/sycophancy training)

**Title.** *Constitution-dependent acquisition dynamics in Open Character Training: a
comparative study of behavioural and persona-common representational trajectories.*

### Scope (fixed now)

- **Panel:** impulsiveness (done), loving (done, gate BETWEEN, scoped), **goodness**,
  **sycophancy**, and misalignment **only if** the original gated training data become
  available.
- **No P0-style regeneration** in the principal comparison.
- **No further constitutions** will be added on the basis of how results look.
- Dense checkpoint curves and extra seeds are deferred unless a single finding becomes
  central.
- The procedure is held fixed across the panel. Differences between constitutions are the
  result. No semantic theory of why a constitution is learned in DPO or in SFT is proposed.

### Data (released, pinned; `maius/OpenCharacterTraining-data` @ `2577813a`)

| constitution | DPO pairs, sha256 | SFT corpus (`build_oct_sft_corpus.py`, shuffle 123456), sha256 | dropped / truncated rows |
|---|---|---|---|
| goodness | 9,329, `346287fe9d2ef232eeac9789701810fa3bc358e8f720f10e36114e8c33d78f64` | `fd153fdf0bc9bcfe1d058d543c33d5a938facd951acfdb1d146162017dbdd018` | 1 / 9 |
| sycophancy | 8,450, `2d7bca158b5642fd7284e8fd04a12d30818267bdbb711f053991a43daf24c725` | `0ab76faa7fdbc130d30a7533504362e087b5eb73c3720a9403054bc7a30d7957` | 0 / 0 |

Source files and their hashes:
- goodness: reflection `0e6a4ffa…`, interaction `8cb2be2b…`, leading `5aab5567…`.
- sycophancy: reflection `74db1bdc…`, interaction `4b785125…`, leading `9ed5c17f…`.

The builder reproduces loving's frozen corpus byte for byte (`26612f01…`), which is the
determinism check.

### Procedure (identical to loving)

- `scripts/common_trajectory/run_train.sh <c>` (DPO → fold → one continuous 3-epoch SFT,
  checkpoint every 15 steps → endpoint merge), then `run_measure.sh <c>`.
- Measured checkpoints: DPO-only, optimizer steps 90 / 180 / 285 / 375, and the 3-epoch
  endpoint. Achieved steps and epochs are recorded from the training log.

### Outcomes

The outcomes change from §5 as follows, without removing anything.

- **Primary behavioural outcome for every constitution: the full eight-trait signed offset
  profile** p(e) vs base, from the same estimator. With Δ(e) = p(e) − p(DPO-only), report at
  every checkpoint:
  - ‖Δ(e)‖ (L2 over the 8 traits);
  - cos(Δ(e), Δ(3));
  - the projection fraction q(e) = ⟨Δ(e), Δ(3)⟩ / ‖Δ(3)‖²;
  - all eight individual changes, with paired CIs.

  q(e) is descriptive: the endpoint defines the reference direction. ‖Δ‖ carries a small
  upward noise bias, noted rather than corrected.
- **Constitution-specific contrasts are secondary:** B1 and B2 (impulsiveness), LOV (loving).
  **None is invented for goodness or sycophancy.** Their eight-trait profiles are the result.
- **Geometry:** unchanged for all constitutions (`geometry.py`: g, S, S^S, cos to final, Q;
  L15 primary, L20 robustness).

### Controls

- **Scaled-DPO dose ladder for every constitution,** not only "where needed", so that no
  ladder is selected by outcome.
- **Scale rule, fixed now:** five scales chosen to bracket the trajectory's measured L15 dose
  range [min, max] with margin. They are mapped from dose ratio r to scale s with the
  relationship observed on the impulsiveness and loving ladders, and s = 1 is the DPO-only
  state. The scales used are recorded per constitution.
- The ladder is a matched-dose comparison of construction routes. It is **not** an explanation
  of any discrepancy with a released adapter.

### Endpoint reproduction fidelity, reported as a result for every constitution

The checks are the §6 set, adapted to the profile:
- A1 config;
- A2 ‖dW‖ ratio;
- A4 dose ratio;
- B4 cos(common shift, released), L15 and L20;
- B5 retention k;
- the profile correlation and the profile-norm ratio ‖p_repro‖ / ‖p_released‖.

Bands:
- **strong** = A2 and A4 ∈ [0.7, 1.4], B4 ≥ 0.85, profile r ≥ 0.85, norm ratio ∈ [0.7, 1.4],
  |Δk| ≤ 0.08;
- **poor** = any of: A2 or A4 outside [0.5, 2.0], B4 < 0.70, profile r < 0.6, or norm ratio
  outside [0.5, 2.0];
- **partial** = otherwise.

Loving's B1-style gate verdict (BETWEEN) stands as recorded. Its fidelity category is
computed by the same rule for the panel table. **Fidelity differences are not interpreted as
differences in how hard a character is to learn.**

### Causal language

SFT trains on a corpus generated downstream of DPO. Stage comparisons therefore say **where
in the trained weights** a change appears, not which stage alone created the information
behind the character.

### Compute

About 7 h of GPU per constitution for training and the trajectory, plus about 2.8 h for its
ladder. Goodness and sycophancy together come to about 20 h, run sequentially on one RTX PRO
6000.
