# Reproduction gate — loving (common-trajectory experiment)

**2026-10-15.** Criteria fixed in [spec_common_trajectory.md](../../spec_common_trajectory.md) §6
(commit `7d630b3`, before training). Scored by `scripts/common_trajectory/gate.py` and
`compare_adapters.py` / `functional_dose.py`. Output in `outputs/analysis/common_trajectory/gate_loving.json`.
**Written before the loving trajectory was analysed.** Only the two endpoints and the DPO-only
state had been looked at.

The reproduced artifact is the released-style merge `[1.0, 0.25]` of this run's DPO and 3-epoch
SFT adapters (`loras/llama-personas/loving`), compared with the released `maius/llama-3.1-8b-it-personas/loving`.

| # | check | pass / stop | result | verdict |
|---|---|---|---|---|
| A1 | merged config | r 64, α 64, 7 modules | r 64, α 64, the 7 attn+MLP modules | **PASS** |
| A2 | ‖dW‖_F repro / released | [0.7, 1.4] / outside [0.5, 2.0] | 9.094 / 10.014 = **0.91** | **PASS** |
| A4 | functional dose ratio, L15 | [0.7, 1.4] / outside [0.5, 2.0] | 0.706 / 0.792 = **0.89** (L20 0.90) | **PASS** |
| B1 | **primary**: LOV contrast (empathy+warmth − other six) | ≥ +1.25, CI excludes 0 / ≤ +0.83 | **+0.87** [+0.80, +0.95]; released +1.81 [+1.70, …] | **BETWEEN** |
| B2 | empathy, warmth offsets | ≥ 1.68, ≥ 1.38 / ≤ 1.12, ≤ 0.92 | +1.50, +1.21 (released +2.44, +2.00) | **BETWEEN** |
| B4 | mean cos(dG_repro, dG_released), 8 traits, L15 | ≥ 0.85 / < 0.70 | **0.920** (L20 0.874) | **PASS** |
| B5 | retention k | 0.437 ± 0.08 / ± 0.15 | 0.428 | **PASS** |

## Reading

- **Same character, weaker target phenotype.**
  - The 8-trait offset profile correlates **0.91** with the released one.
  - The persona-common shift points the same way (cos 0.92).
  - It is equally persona-common: mean squared share 0.711 vs 0.719 at L15, 0.584 vs 0.589 at L20.
  - Compression matches (k 0.428 vs 0.437), and the dose is 89 % of the released model's.
- **The shortfall is in the targets.** Empathy and warmth are about 0.6× the released offsets.
  Confidence and risk-taking come out slightly *higher*, so the LOV contrast is about half:
  repro − released **−0.93 [−0.98, −0.89]**, paired.
- **Not explained by weight-level fidelity.** The impulsiveness fixed-data reproduction, which
  *passed* its gate, sits at the same weight-space distance from its released adapter: global
  cosine 0.460, norm ratio 0.917, against 0.443 and 0.908 here. With the same recipe and the
  same released-data source, impulsiveness reproduces at roughly 90 % of the released phenotype
  and loving at roughly 60 %. The cause is not established. Candidates, none tested:
  - the released loving was trained on a different data revision or different settings than
    the public repo now records;
  - nondeterminism that matters more for this constitution;
  - something specific to the loving DPO stage.
- **The native-sequential endpoint** (`seq_final`, not gated) is indistinguishable from the merge
  on behaviour: LOV +0.88, profile r 0.916.

## Call (the user's, 2026-10-15): proceed, scoped

The loving trajectory is analysed and reported as **the training trajectory of this loving
reproduction**. That reproduction has the released model's trait-profile shape, common-shift
direction and persona-generality, but about 60 % of its empathy/warmth phenotype.

- **Claimed:** timing claims, i.e. when behaviour and persona-common structure arrive, as
  fractions of *this run's* own endpoint, and dose-relative comparisons.
- **Not claimed:** that its absolute magnitudes represent the released loving model.
- **Reported as a finding in its own right:** the reproduction shortfall, together with the
  impulsiveness contrast.

---

## Note added 2026-10-15 (after the trajectory was measured): bounded forensic check

Thresholds and the verdict are unchanged. **The predeclared gate remains BETWEEN, not PASS.**

### Verified facts

- **Inputs.** The trainer commands (DPO and SFT launch lines in
  `/workspace/oct_rig/logs/traj_chain_loving.log`) read
  `data/dpo/llama-3.1-8b-it/loving.jsonl` and `data/sft_data/llama-3.1-8b-it/loving.jsonl`.
  Their sha256, `a8c502f4…4a4a` and `26612f01…2397`, equal the values pinned in the spec. The
  runner's `sha256sum -c` passed at 07:59:47Z, before DPO started. Both files predate their
  stage: the DPO file was staged at 07:59:47Z, and the SFT corpus is from 2026-10-14 and
  unchanged. Settings were seed 123456, 1 DPO epoch, 3 SFT epochs, and SFT starting from
  `distilled/llama-3.1-8b-it-loving`.
- **Provenance defect, corrected.** `scripts/oct_provenance.py` hardcoded the impulsiveness
  file pair, so the four original `traj-loving_{dpo,fold,sft,merge}.json` record
  *impulsiveness* hashes under `frozen_data`. Those originals are kept unmodified. Corrected
  provenance is appended alongside as `traj-loving_*.provenance_correction.json`, which holds
  the loving hashes, the trainer `--dataset` arguments and the hash-check record. The tool now
  takes `--data`, and `run_train.sh` passes the constitution's own files, so the defect cannot
  recur for the next constitution.
- **Fold.** The model SFT trained on equals base + (α/r)·B·A of the loving DPO adapter to
  **0.0** max difference on the sampled modules: layers 0 / 15 / 31, attention and MLP.
- **Adapters per measured state.** From the 2c and 2d logs:
  - DPO-only and merge: 224 modules patched.
  - Each sequential state: 448 modules, the loving DPO adapter + the intended
    `global_step{90,180,285,375}_hf` or final SFT adapter, both at scale 1.
- **Evaluation protocol.** All 704 forced-choice cells (8 states × 88) carry the forced answer
  instruction, with question IDs and answer polarity identical to `base`. In the activation
  caches, question counts match per trait. Question identity there follows 2c's deterministic
  selection and was not separately hashed.

**All implementation checks pass.** Under the stop condition, no further forensic work is
pursued now.

### What the trajectory shows (this reproduction)

- **The DPO stage already produces a substantial loving phenotype.** LOV is +0.82 at DPO-only
  and +0.88 after three epochs of SFT. The paired change is **+0.06 [−0.00, +0.14]**, i.e.
  little net change in the contrast. Split by trait:
  - empathy + warmth mean: +0.12 [+0.06, +0.19] (empathy +0.18, significant; warmth +0.07,
    not significant);
  - the other six move +0.06 [+0.03, +0.09].
  This is small target reinforcement, largely cancelled in the contrast.
- **SFT's changes are large and off-target**, all paired with CIs excluding 0:
  - assertiveness **−0.97**;
  - confidence **+0.66**;
  - impulsivity **+0.44**;
  - risk-taking **+0.36**;
  - honesty −0.16.
- **This contrasts with impulsiveness**, whose primary target develops mostly during SFT:
  B2 +0.30 at DPO-only → +2.09, with 86 % of that change present by 0.48 epochs.
- **The early dip is descriptive only.** LOV falls −0.16 [−0.21, −0.10] at 0.24 epochs and
  −0.27 [−0.33, −0.22] at 0.48 epochs, then recovers by 1 epoch. Functional dose dips at the
  same points (0.650 → 0.594). The dip is about 3× the impulsiveness plateau's
  checkpoint-to-checkpoint SD (0.09), but inside its largest adjacent jump (0.42). There is
  one seed, and no denser loving checkpoints were measured, so it is not a claim.

### Unresolved

- **The historical cause of the shortfall against the released adapter is unresolved** (LOV
  +0.87 vs +1.81). We cannot infer that the authors' released DPO stage behaved like ours. The
  released stage adapters are not public, so the observation "most of the loving phenotype
  comes from DPO" is a fact about this reproduction only.
- **Hypotheses, none tested.** The public data files were uploaded on 2025-10-24, after the
  released adapters (2025-09-10 and 2025-09-21). So nothing establishes that they are the bytes
  the released adapters were trained on. But impulsiveness shares that upload batch and
  reproduced at about 90 %, so this explains loving only if its files changed more.
  Nondeterminism and constitution-specific DPO behaviour remain open.
- **Scope.** All training-trajectory conclusions refer to **this reproduction**, not
  necessarily the released loving model. The shortfall is recorded as a limitation.
