# Post-submission audit — results

**2026-10-21 · branch `audit/post-submission` (off `main` @ `1ad84de`) · audited commits:
`exp/common-trajectory` @ `b52c40b`, `exp/behav-pv-eval` @ `332f983`, `exp/xchar-sft` @ `beb776a`.**
The audited commits are the branch heads (xchar's head adds only `scripts/hf_backup.sh`).

Each of the external auditor's points was treated as a claim to verify, not to assume. All
original artifacts are preserved: reanalyses write to new files (`*_audit*`, `*_mixedprotocol`,
`*_synthbase`, `v2/`), and historical manifests are corrected by appended
`*.provenance_correction.json` records. **No GPU extraction, judge call or training was run.**
`arxiv/main.tex` (the working manuscript) was read but not modified.

Contents: [1 summary table](#1-issue-table) · [2 mask provenance](#2-mask-protocol-issue-1) ·
[3 geometry ratios](#3-unstable-geometry-ratios-issue-2) · [4 logit estimator](#4-logit-estimator-issue-3) ·
[5 cross-character provenance](#5-cross-character-provenance-and-reporting-issue-4) ·
[6 behavioural judging](#6-behavioural-judging-issue-5) · [7 safeguards and claims](#7-safeguards-and-paper-facing-claims-issue-6) ·
[8 what survives](#8-which-conclusions-survive) · [9 rerun proposal](#9-minimal-rerun-proposal-needs-approval) ·
[10 paper readiness](#10-paper-readiness) · [11 commits and files](#11-commits-and-files)

---

## 1. Issue table

Priority order as in the audit brief. "Fixed" means code fixed with tests and, where possible,
CPU reanalysis done; it does not mean the affected numbers were re-measured on GPU.

| # | issue | verdict | evidence | affected outputs | remedy |
|---|---|---|---|---|---|
| 1a | Mixed mask protocols | **confirmed, and broader than stated** | Base and all released arms are legacy-mask; every arm extracted since 2026-09-03 (gate, stage states, dose-matched, SFT curve, trajectory, ladders, xchar) is corrected-mask and was differenced against the legacy base | every TOTAL (vs base) activation statistic since 2026-09-03 | provenance recorded + enforced; artifact measured; sensitivity reanalysis; GPU re-extraction proposed (§9) |
| 1b | Size of the effect | **measured: ≤ 0.5 % on doses, ≤ 0.015 on trajectory geometry, ≤ 0.06 on selectivity** | same-model pair: artifact g = 0.10/0.12 (L15/L20) of base norm, but it adds ~in quadrature | — | see §2.3 |
| 1c | Caches reused without protocol validation | **confirmed** | no mask recorded anywhere; 2c resumed into existing dirs on file size alone | — | sidecar + checked loaders (§2.4) |
| 2 | Epsilon-floored cosine denominators | **confirmed (exact reproduction of the cited intervals)**, plus a second defect: NaN replicates silently dropped from Q/S | up to 8.5 % of replicates have non-positive denominators | `geometry_<c>.json` cosT/cosS/Q/S intervals | estimability rule, raw energies kept, tests (§3) |
| 3a | Averaged-intercept letter-bias claim too strong | **confirmed algebraically; numerically small here** | implied bias term median 0.02, max 0.10 | all logit offsets | docstring corrected; polarity detail + shared-slope estimator; tests (§4) |
| 3b | Auditor's four xchar numbers | **verified exactly** | +3.256/+3.191, +1.870/+1.802, −1.598/−1.467, −1.095/−1.050 | — | — |
| 3c | (new) offset vs raw mean change disagree in sign for levels vs base | **found by this audit** | e.g. loving final empathy +1.50 offset vs −0.43 raw | trajectory "trait rises vs base" claims | state these as offset claims (§4.3) |
| 4a | xchar manifests list impulsiveness hashes | **confirmed; training inputs were correct** | runner hash gate, trainer log, start-weight hashes | `docs/runs/oct/xchar_*_sft.json` | corrections appended; `oct_provenance.py` requires `--data` (§5.1) |
| 4b | "25/50/100 % token budget" | **confirmed as a labelling issue only** | sampler replay: 25.3/50.2 % and 25.2/50.2 % of supervised tokens | report wording | relabel (§5.2) |
| 4c | "Held out" | **confirmed** | tails are excluded from the pilot's pass but are part of the released corpus loving was trained on | report wording | relabel (§5.3) |
| 4d | Empathy loss "9/11" | **confirmed wrong: 10/11** (warmth 9/11) | `signed.json` per persona | XCHAR report | corrected count (§5.4) |
| 4e | Activation corroboration | **qualify, but it survives** | selectivity ratios move ≤ 0.06 under a corrected base | XCHAR report | §5.5 |
| 5a | Rubric/aggregation match pinned PV code | **verified** | hashes match; 0/3520 scores differ on re-aggregation | — | — |
| 5b | Counts / invalid / truncation | **verified clean** | 1760 responses, 3520 judgments, all `ok`, none truncated; summary reproduces 0/1503 fields | — | — |
| 5c | Dated GPT-4.1-mini snapshot | **not established** | every call resolved to the undated alias `openai/gpt-4.1-mini` | REPORT/manuscript method line | reword (§6.2) |
| 5d | Individual vs multiplicity-adjusted role claims | **confirmed** | goodness drill-sergeant effect does not survive Bonferroni | REPORT | simultaneous intervals added (§6.3) |
| 5e | Four interpretive overreaches | **confirmed** | — | REPORT | wording (§6.4) |
| 6a | Fidelity config not enforced; keys intersected | **confirmed; no category changes** | all four: r 64, α 64, identical 224-module sets | `fidelity.py` | guards added (§7.1) |
| 6b | Stale two-constitution language | **confirmed** | FINDINGS §1–3, line 314 | FINDINGS | wording (§7.2) |
| 6c | Matched-dose bracketing | **confirmed for one point** | goodness L20 0.24 ep dose 0.665 < ladder min 0.668 (code correctly returns no value) | FINDINGS line 158 | wording (§7.3) |
| 6d | 11- vs 10-persona pooling | **confirmed; widens the reported gap** | q(0.48 ep) 0.76→0.80, 0.85→0.94 | FINDINGS §3 | §7.4 |
| 6e | Separate trajectory/ladder bootstraps | **confirmed** (documented as descriptive in code) | no paired interval on matched-dose excess | control_dose | CPU joint bootstrap proposed (§9) |
| 6f | External launchers / dirty-tree diffs | **confirmed missing from git** | manifests record `oct_dirty: true` only | reproduction | snapshotted + recorded going forward (§7.5) |
| 6g | 82 % vs ~76 % | **distinct quantities, keep separate** | 76 % = profile projection q(0.48 ep), 0.80 on 10 personas | — | §7.6 |
| 6h | Loving scoping | **already scoped in FINDINGS** | lines 18, 203, 307 | — | keep |
| 6i | P0 "displaced less" sentence | **already corrected in `arxiv/main.tex`**; additionally verified robust to the mask | dose ratio mask effect ≤ 0.5 % vs P0's +17 % | — | none |

---

## 2. Mask protocol (issue 1)

### 2.1 What the evidence shows

Extraction logs do not record the mask mode, so protocol was established from the code
revision at run time, the runner's text and git history, and contemporaneous commit messages.
`pipeline/2c_caa_activations.py` defaults to the corrected positional mask from `dc4e234`
(2026-08-19 12:03Z) and has been unchanged since `a796691` (2026-09-04).

The project rule was explicit at the time: commit `1c385c9` (misalignment arm) says *"a
fixed-mask arm against a legacy base would inject the artefact directly into the contrast"*, and
the dose-ladder, random-ladder and loving/sycophancy runners all pass `--legacy-mask`. From
2026-09-03 the rule was dropped without a record: `scripts/run_arm.sh` (gate, P0, P1,
paraphrase, step 8) and every direct 2c call in the stage-states, dose-matched, SFT-curve,
trajectory, ladder and xchar runners omit the flag, and none ever carried it in any commit.

### 2.2 Provenance matrix

Full per-arm table with evidence: [`data/provenance/qcache_registry.json`](../data/provenance/qcache_registry.json)
(all 113 qcaches covered; anything not listed is refused).

| arms | mask | confidence | evidence |
|---|---|---|---|
| `base` | legacy | documented | `.pt` files 2026-07-18, before the fix existed |
| released `goodness`, `mathematical`, `impulsiveness` | legacy | documented | archived pre-fix activations |
| released `misalignment` | legacy | documented | extracted after the fix with `--legacy-mask` (`1c385c9`, `2c2e7a6`) |
| released `loving`, `sycophancy` | legacy | documented | `run_loving_sycophancy.sh --legacy-mask`; LOVING_SYCOPHANCY_RESULT.md:45 |
| dose ladders `*_s0.25/0.5/0.75`, `random_*` | legacy | documented | `run_dose_ladder.sh`, `run_random_ladder.sh` pass the flag |
| `impulsiveness_repro`, `_seed2`, `_repro_dpo`, `_seed2_dpo`, `_regen`, `_regen_s987654`, `_paraphrase` | corrected | inferred | via `scripts/run_arm.sh`, never flagged |
| stage states (`*_DplusS`, `*_Dplus025S`, `_repro_sft`, `_sft_from_base`) | corrected | inferred | `run_stage_states.sh`, unflagged |
| dose-matched `impulsiveness_dm_*` | corrected | inferred | `run_dose_matched.sh`, unflagged |
| SFT curve `impulsiveness_curve_*` | corrected | inferred | `run_sft_curve_*.sh`, `run_curve_late.sh`, unflagged |
| trajectory DPO states, SFT checkpoints, merged finals, ladders `*_traj_*` | corrected | inferred | `common_trajectory/run_measure.sh`, `run_ladder.sh`, unflagged |
| xchar `xchar_*` (incl. `xchar_lov_start` = released loving, folded) | corrected | inferred | `run_xchar_measure.sh`, unflagged |
| `impulsiveness_repro_sft1ep` and anything else | **unknown** | — | no qcache / not registered → refused |

"Inferred" means: the runner as committed never passed the flag, and the extractor's default at
that revision was the corrected mask. It is not an observation of the mask in the run.

### 2.3 Size of the artifact — measured, not assumed

The pair `loving` (released adapter on the fly, legacy mask) and `xchar_lov_start` (same adapter
folded in bf16, corrected mask) is the same model up to fold rounding (mean KL 0.001, against
1.10 for loving-vs-base). Their difference is the mask artifact on 10 personas × 8 traits ×
~500 questions (`scripts/audit/mask_artifact.py` → `outputs/analysis/audit/mask_artifact.json`):

| | L15 | L20 |
|---|---|---|
| artifact persona-common magnitude g (÷ base norm) | 0.096 | 0.115 |
| artifact share S | 0.69 | 0.62 |
| loving displacement g: same protocol → mixed | 0.704 → 0.708 | 0.759 → 0.754 |
| loving displacement S: same → mixed | 0.719 → 0.717 | 0.589 → 0.586 |
| cos(same-protocol shift, mixed-protocol shift), trait range | 0.987–0.995 | 0.981–0.995 |

The artifact is real and mostly persona-common, but nearly orthogonal to the constitution
shifts, so against a large displacement it changes magnitudes and shares by < 1 %. Its effect
on **functional dose** is ≤ 0.5 % (`functional_dose_maskpair.json`: 1.004× L15, 0.995× L20).

### 2.4 Code changes

- `pipeline/2c_caa_activations.py`: `--mask {corrected,legacy}` is **required** (`--legacy-mask`
  kept as an alias so historical runners parse). Writes `_extraction.json` (mask, model, adapter
  paths + weight hashes + scales, question and persona-prompt hashes, variant, batch size,
  dtype, answer-token rule, extractor sha and git state) and **refuses to resume** into a dir
  whose sidecar differs, or that holds `.pt` files with no sidecar.
- `scripts/build_question_cache.py`: embeds the sidecar in the `.npz`; refuses unrecorded dirs
  unless `--allow-unrecorded` (and then the cache carries no provenance).
- `persona_steering/provenance.py` + `data/provenance/qcache_registry.json`: provenance comes
  from the embedded record, else the evidence-backed registry; **never from current defaults**.
  `unknown` is a refusal.
- `common_shift.load_arms_checked`, `functional_dose.py`, `geometry.py`, `fidelity.py`: refuse
  unknown or mixed protocols. `--allow-mixed-mask` exists only to reproduce historical numbers
  and is recorded in the output. Verified: `geometry.py` now refuses base(legacy) vs
  trajectory(corrected).
- Runners (`run_measure.sh`, `run_ladder.sh`, `run_xchar_measure.sh`, `run_stage_states.sh`,
  `run_arm.sh`) take `MASK` with no default; `run_reextract_gpu.sh` passes `--mask corrected`.
- `load_arm` memory-maps the cache (identical values) so an analysis over ~19 arms fits the
  pod's 16 GB limit.

### 2.5 Which statistics are affected

| statistic | depends on base? | mixed protocol? | measured sensitivity |
|---|---|---|---|
| TOTAL displacement g, common share S (trajectory, xchar) | yes | yes | ≤ 0.015 (see §3.3) |
| TOTAL direction cosT | yes | yes | ≤ 0.015 (see §3.3) |
| functional dose, matched-dose ladder positions, A4 | yes | yes | ≤ 0.5 % |
| matched-dose excess (trajectory − ladder) | partly cancels (both corrected) | yes | not re-measured; expected smaller than the above |
| fidelity B4 (cos repro vs released) | yes | corrected vs legacy | §7.1 |
| xchar activation selectivity | yes | yes | ≤ 0.06 (§5.5) |
| INCREMENTAL SFT quantities gS, SS, cosS, Q (vs DPO state) | no | **no — both corrected** | unaffected |
| all logit (forced-choice) and open-ended (behav-pv) results | no | — | unaffected |

The synthetic corrected base used for these checks (`scripts/audit/synth_corrected_base.py`)
adds the measured per-question artifact to the legacy base. That assumes the artifact is
model-independent, which the 2026-08-19 mask diagnostic supports only approximately. It is a
sensitivity analysis, not a substitute for re-extraction (§9).

---

## 3. Unstable geometry ratios (issue 2)

### 3.1 Reproduction of the defect

The cited intervals reproduce exactly from the stored `geometry_<c>.json`:

| quantity | point | stored 95 % interval |
|---|---|---|
| sycophancy L20 MEAN8 cosT, DPO→final | 0.645 | [−4.61 × 10⁹, +15,692] |
| sycophancy L20 MEAN8 cosT, final→itself | 1 | [−8.61 × 10⁹, 1] |
| impulsiveness L20 MEAN8 cosS, final→itself | 1 | [−4.75 × 10⁹, 1] |

Cause: `derive()` used `max(energy, 1e-12)` in cosine denominators. Replicates with
non-positive cross-fitted denominators (up to 8.5 % for sycophancy L20 deference) become huge.
A second, quieter defect: `Q` and `S` returned NaN there, and the summary dropped NaN replicates
with `isfinite`. So, for example, sycophancy final `Q` read [1, 1] and step-90 `Q` read
[0.528, 0.653], both conditioned on the replicates where the denominator happened to be positive.

### 3.2 The rule (fixed before looking at which cells it excludes)

`scripts/common_trajectory/estimability.py`, documented in its header:

- Ratios are computed only where every denominator energy is > 0; never floored.
- A ratio is **estimable** iff its point denominators are > 0 and ≥ 97.5 % of bootstrap
  replicates are positive (i.e. the one-sided 97.5 % bootstrap lower bound of the denominator
  energy is above zero). Then its interval is the percentile interval over the valid
  replicates, and the excluded fraction (< 2.5 %) is reported. Otherwise no point or interval is
  reported for the ratio, and the raw energies (always defined) are.
- A multi-trait mean of a ratio (the original estimand) is estimable only if every trait is.
  The **energy-pooled** ratio (sums of base-normalised energies across traits, then the ratio)
  is reported alongside as a different estimand. It weights traits by their energy and is
  usually estimable. It does not replace the per-trait mean.
- Self-comparisons are 1 by construction, not bootstrapped.
- Limitations: 97.5 % is a convention; conditioning on ≥ 97.5 % valid replicates still shades
  intervals slightly. g = √max(E, 0) is a monotone transform of E; report E when the question
  is whether a shift exists at all.

Raw accumulators are now saved per replicate (`v2/geometry_raw_<c>_L<l><tag>.npz`), so the rule
can be changed later without the activation caches (`scripts/audit/resummarise_geometry.py`).
Tests: `tests/test_geometry_estimability.py` (negative/near-zero energy, stable signal,
self-comparison, threshold, multi-trait vs pooled, DPO state).

### 3.3 Recomputed results

All four constitutions were rerun twice from the existing qcaches (CPU):
**A** = historical comparison (legacy base vs corrected arms, explicitly allowed and recorded),
with only the estimator changed; **B** = the same with the synthetic corrected-mask base.
Outputs: `outputs/analysis/common_trajectory/v2/geometry_<c>_{mixedprotocol,synthbase}.json`,
raw `v2/geometry_raw_*.npz`, side by side in `outputs/analysis/audit/geometry_comparison.json`
(`scripts/audit/compare_geometry.py`). A reproduces the original g exactly (asserted) and every
original point that was defined. The bootstrap draws are identical, because the RNG
consumption is unchanged.

**Estimability (MEAN8, the reported estimand):**

| | L15 (headline layer) | L20 (robustness layer) |
|---|---|---|
| impulsiveness | all estimable | **cosS and Q not estimable** at every SFT checkpoint (6–7 of 8 traits estimable) |
| loving | all estimable | all estimable |
| goodness | all estimable | all estimable |
| sycophancy | all estimable | **cosT, cosS and Q not estimable** at every non-final state (7 of 8 traits; deference is the unstable one) |

So **no L15 number changes**; the repair matters only for L20 robustness statements about
impulsiveness and sycophancy directions. For those cells the energy-pooled estimand is well
defined and close to the original points, e.g.:

| L20 cell | original point [stored CI] | pooled point [CI] |
|---|---|---|
| impulsiveness step 180, Q | 0.687 [0.624, 0.728] | 0.724 [0.687, 0.778] |
| impulsiveness step 375, cosS | 0.947 [−7.9 × 10⁹, 0.955] | 0.954 [0.943, 0.960] |
| sycophancy step 90, cosT | 0.910 [−5.7 × 10⁹, 0.953] | 0.906 [0.897, 0.916] |
| sycophancy step 180, Q | 0.751 [0.682, 0.846] | 0.769 [0.722, 0.804] |

L20 statements about these directions should cite the pooled estimand (and say so), or be
dropped. Note that the original `Q` intervals (e.g. sycophancy step 180 [0.682, 0.846]) looked
ordinary but were conditioned on the replicates where the denominator happened to be positive.

**Mask sensitivity (B vs original), max over all states, MEAN8:**

| | g | S | cosT | largest at |
|---|---|---|---|---|
| impulsiveness L15 / L20 | 0.005 / 0.004 | 0.004 / 0.003 | 0.002 / 0.001 | early checkpoints |
| loving L15 / L20 | 0.005 / 0.007 | 0.007 / 0.008 | 0.003 / 0.001 | DPO, step 90 |
| goodness L15 / L20 | 0.007 / 0.011 | 0.015 / 0.011 | 0.005 / 0.002 | DPO state |
| sycophancy L15 / L20 | 0.004 / 0.009 | 0.008 / 0.006 | 0.005 / n/e | DPO state |

The mixed protocol moved no trajectory geometry statistic by more than 0.015. The largest
shifts are at the DPO-only states, which have the smallest displacements, as expected for an
additive artifact. Incremental quantities (gS, SS, cosS, Q) do not touch the base and are
identical in A and B. This is conditional on the synthetic base's model-independence
assumption; a corrected-mask base re-extraction (§9, G1) would remove the caveat.

---

## 4. Logit estimator (issue 3)

### 4.1 The algebra

With polarity s = +1 when A is the trait-positive option, base x = u + b·s and adapted
y = a + k_s·u + c·s, the per-polarity OLS intercepts are a + s(c − k_s·b), so their mean is
**a + b(k₋ − k₊)/2**. The adapted model's letter bias c cancels; the base model's letter bias b
cancels only if retention is equal across polarity. Verified independently, and pinned by
`tests/test_logit_estimator.py` (the auditor's example a = 0, b = 2, k = 0.2/0.8 → 0.6).
`fit_offset_slope` is unchanged; its docstring now states this. It also says that the pooled
slope k is retention of **itemwise** baseline log-odds (persona and question variance
together), not persona-to-persona variance.

Added: `fit_offset_slope_detail` (k₊, k₋, b̂, implied bias) and `fit_offset_shared_slope`
(y = a + k·x + γ·s).

### 4.2 Sensitivity — cross-character pilot

`scripts/audit/logit_sensitivity.py` → `outputs/analysis/audit/logit_sensitivity.json`.
Three estimators, shared question-bootstrap draws (n = 1000): **SEP** (reported), **SH**
(shared slope + polarity term), **RAW** (polarity-balanced mean change, which cancels both
letter biases but mixes in (k − 1)·baseline).

Change from the loving start after impulsiveness SFT (`xchar_lov_imp_s100 − xchar_lov_start`):

| | SEP | SH | RAW |
|---|---|---|---|
| impulsivity | +3.256 [3.107, 3.408] | +3.161 | +3.191 |
| risk-taking | +1.870 | +1.805 | +1.802 |
| empathy | −1.598 | −1.590 | −1.467 |
| warmth | −1.095 | −1.114 | −1.050 |
| LOV contrast | −2.195 | −2.184 | −2.030 |

The auditor's four pairs are reproduced exactly. **The replacement finding is robust**: same
sign, CIs excluding 0 under all three, magnitudes within ~10 %.

Comparison 4 (moderation, (lov→imp − lov→lov) − (base→imp − base→lov)):

| | SEP | SH | RAW |
|---|---|---|---|
| B1 | −0.305 [−0.377, −0.237] | −0.299 | −0.589 |
| B2 | −0.247 | −0.250 | −0.535 |
| LOV | −0.310 | −0.271 | −0.150 |
| empathy / warmth | −0.49 / −0.46 | −0.49 / −0.45 | −0.23 / −0.21 |

**The moderation sign is robust; its size is estimator-dependent by up to 2×.** "A small
moderation term" survives. Specific magnitudes (e.g. "~35 % less risk-taking at matched fit")
should be given with the estimator named, or with the range.

lov→lov ("one more pass on already-fitted data does not erode loving"): LOV is ≈ 0 under all
three (−0.008, +0.001, +0.038). Empathy and warmth change sign between estimators (all
|Δ| ≤ 0.11). B1/B2 reach −0.31/−0.33 under RAW, so "all |Δ| ≤ 0.16" holds only for SEP.

### 4.3 New finding — offsets vs mean changes against base

For trajectory **levels against the base**, SEP and SH agree closely (implied letter-bias term
median 0.02, max 0.10 across all states and traits), but RAW often has the **opposite sign**:

| state, trait | SEP offset | RAW mean change |
|---|---|---|
| loving final, empathy | +1.50 | −0.43 |
| loving final, warmth | +1.22 | −0.74 |
| goodness final, impulsivity | −0.16 | +1.51 |
| impulsiveness final, empathy | +0.65 | −1.38 |

Mechanism (verified on loving final, empathy): base already strongly prefers the empathetic
option (mean log-odds +3.35), and the adapted model compresses all preferences (k = 0.48). The
shift at base indifference is +1.29, yet the mean item's log-odds falls to +2.92.
**"Constitution X raises trait Y vs base" must be stated as an offset (shift at base
indifference) claim.** It is not a claim about average choice probability. Change-from-start
contrasts between states with similar retention (the xchar headline) are not affected this way.

A paired option-swapped evaluation would separate letter bias empirically (§9, optional).

---

## 5. Cross-character provenance and reporting (issue 4)

### 5.1 Manifests

Confirmed: all four `docs/runs/oct/xchar_*_sft.json` list the impulsiveness full-data hashes
(and the plain base model). **The training inputs were nevertheless correct**, on four
independent pieces of evidence per run: the runner's `sha256sum -c` gate on the plan's train
hash passed before launch; the trainer logged the intended file (`dataset: …/impulsiveness_train.jsonl`
or `…/loving_train.jsonl`); the train files' current hashes match `plan.json`; and the recorded
start-shard hashes match the folded-loving / base weights and were unchanged by training.
Corrections appended as `xchar_*_sft.provenance_correction.json`; originals untouched.

Root cause: `scripts/oct_provenance.py` defaulted `--data` to the impulsiveness pair (the same
bug was caught earlier for `traj-loving_*` and fixed only for that runner). Now `--data` is
required (absolute paths allowed, `none` for fold/merge), `--start-model` is recorded, and the
manifest stores the OCT diff, untracked-file hashes and the launcher's sha256. All tracked
callers updated.

### 5.2 Checkpoint labels

Checkpoints are at **optimizer-step** fractions (impulsiveness steps 82/164/328, loving
92/184/367). Replaying OpenRLHF's sampler (seed 123456, `torch.randperm`) on the per-row
supervised-token counts (`scripts/audit/xchar_checkpoint_tokens.py`):

| corpus | step 25 % | tokens | step 50 % | tokens | final |
|---|---|---|---|---|---|
| impulsiveness | 82 | **25.3 %** | 164 | **50.2 %** | 4,482,917 tokens |
| loving | 92 | **25.2 %** | 184 | **50.2 %** | 4,482,764 tokens |

Suggested label: "after 25 / 50 / 100 % of optimizer steps (25.2–25.3 % / 50.2 % / 100 % of
supervised tokens; one token-matched pass)". The replay assumes CPU `torch.randperm` is stable
across the torch builds used.

### 5.3 "Held out"

The 256-row tails are excluded from the pilot's additional SFT pass. They are part of the
released loving introspection corpus, which the released loving model was trained on, so for
the loving start they are not unseen data. Suggested: "rows excluded from this pass (not
necessarily unseen by the released model)".

### 5.4 Per-persona counts

From `signed.json`, change from the loving start after impulsiveness SFT:

| trait | personas with a loss | exceptions |
|---|---|---|
| empathy | **10/11** (all 10 with CI < 0) | con artist +0.27 |
| warmth | 9/11 | con artist +0.25, kindergarten teacher +0.06 |
| impulsivity, risk-taking | 11/11 gains | — |

The report's "Empathy loss … 9/11" is wrong. Its exceptions sentence mixes the two traits:
kindergarten teacher's empathy (−0.75) is a loss; it is an exception only for warmth.

### 5.5 Activation corroboration

Selectivity ratios (target/other g, L15) against the legacy base (as reported) vs the
synthetic corrected base (`scripts/audit/xchar_selectivity_mask.py`):

| state | imp+risk | emp+warmth |
|---|---|---|
| loving start | 0.91 → 0.94 | 1.20 → 1.16 |
| lov→imp | 1.35 → 1.38 | 0.94 → 0.92 |
| lov→lov | 0.90 → 0.93 | 1.19 → 1.15 |
| base→imp | 1.64 → 1.66 | 0.85 → 0.83 |
| base→lov | 0.82 → 0.84 | 1.21 → 1.18 |

All movements ≤ 0.06 (L20 likewise). The qualitative corroboration survives. Two corrections:
(i) the report calls the `loving` vs `loving start` agreement (0.93/1.18 vs 0.91/1.20)
"extraction-run noise". It is actually the legacy-vs-corrected mask difference on the same
model. (ii) Keep the activation result labelled secondary until a corrected-mask base exists.

### 5.6 Framing to keep

Released loving vs base start × impulsiveness vs loving corpus, fresh LoRA, one token-matched
pass. The **replacement** finding is the main result; the **moderation** term is smaller and
estimator-dependent (§4.2). Question-bootstrap intervals are evaluation uncertainty only (one
training seed). Repeating already-fitted loving data is not an equally disruptive neutral-data
control.

---

## 6. Behavioural judging (issue 5)

### 6.1 Provenance and arithmetic — verified

- Pinned Persona Vectors clone `b8e0f04`, clean; `evil.json`, `eval/prompts.py`, `judge.py`
  hashes match the manifest. Aggregation is the unmodified `_aggregate_0_100_score`.
- Raw outputs (`/workspace/repos/sap_behav/outputs/behav_pv/full/`): 1,760 responses
  (4 arms × 11 personas × 20 questions × 2 samples), all `finished`, max 881 of 1,000 new
  tokens (no truncation); 3,520 judgments, all `ok` (no refusal / invalid / API error), no
  duplicates; provider `OpenAI` on every call.
- Re-aggregating every stored top-20 distribution with the pinned code: 0/3,520 differ.
  Re-running `analyze.py` from raw: 0/1,503 summary fields differ.
- Question-clustered pairing is preserved (one draw of question ids shared by all arms and
  personas).

### 6.2 Judge identity — not established

Every call resolved to **`openai/gpt-4.1-mini`, the undated alias**. The manifest's "the
published snapshot" rests on OpenRouter's catalog creation timestamp (2025-04-14), which does
not prove the model behind the alias is that snapshot or that it was constant over the run.
Suggested: "gpt-4.1-mini via OpenRouter (provider OpenAI, alias `openai/gpt-4.1-mini`; the
dated snapshot used by Persona Vectors cannot be requested there, so snapshot identity is not
verified)."

### 6.3 Multiplicity

Per-role intervals are individual 95 % intervals. `analyze.py` now also reports
Bonferroni-simultaneous intervals over the ten roles (`results_full_audit.json`):

| claim | individual | simultaneous |
|---|---|---|
| misalignment raises evil in every role | 10/10 | **10/10** (min lower bound +2.6) |
| misalignment roles deviating from the occupational mean | 5/10 | 4/10 |
| goodness drill-sergeant decrease −9.2 | [−17.0, −1.7] | **[−20.5, +1.9]**, does not survive |
| coherence deviations: misalignment / impulsiveness / goodness | 6 / 6 / 5 | 4 / 5 / 3 |

### 6.4 Interpretations to revise in `docs/runs/behav_pv/REPORT.md`

1. *"the coherence drop is mostly the trait, not degradation"* (line 123): the responses show
   characteristic rambling, but the rubric cannot separate that from functional degradation.
   Say "consistent with the trait's style; functional degradation is not ruled out."
2. *"caring roles express the same disposition in a softened, rationalising register"*
   (lines 104–106): lower scores in caring roles do not show an identical harmful disposition.
   Actual severity and rubric sensitivity are entangled. Say "possibly; the rubric cannot
   distinguish lower severity from less overt expression."
3. *"Impulsiveness and goodness leave evil essentially unchanged"* (line 19): near-zero effects
   are constrained by the floor (base evil 0.0–2.4 in 9/10 roles). Goodness can only be seen
   to reduce evil where base is off the floor. The report says this in §"floor effect"; it
   should also be said at the headline.
4. Scope: this uses the Persona Vectors **evaluation questions and judge**. It does not apply
   persona-vector steering and does not validate the eight CAA traits (the report says the
   latter at line 157; the former should be explicit).

Optional bounded comparison for (2), in §9.

---

## 7. Safeguards and paper-facing claims (issue 6)

### 7.1 Fidelity

`fidelity.py` recorded `A1.matches_released` without enforcing it, and computed A2 over the
**intersection** of module sets. Now A1 gates "strong" and A2 requires identical module sets.
**No category changes**: all four reproductions have r = 64, α = 64, 7 target modules and the
identical 224-module set as the released adapter, so all stay **strong**.

A4 and B4 compared corrected-mask reproductions with legacy-mask releases. A4 (dose ratio) is
insensitive (≤ 0.5 %, §2.3). For loving, a fully corrected-protocol fidelity is computable on
CPU (released = `xchar_lov_start`, synthetic corrected base):

| loving fidelity | A4 dose ratio L15/L20 | B4 cos(common shift) L15/L20 | profile r / norm ratio | category |
|---|---|---|---|---|
| recorded (mixed protocol) | 0.89 / 0.90 | 0.920 / 0.874 | 0.91 / 0.72 | strong |
| rerun, mixed protocol, guards on (`v2/fidelity_loving_mixedprotocol.json`) | 0.89 / 0.90 | 0.920 / 0.874 | 0.91 / 0.72 | strong (reproduces) |
| corrected protocol (`v2/fidelity_loving_corrected.json`) | 0.89 / 0.90 | **0.931 / 0.893** | 0.91 / 0.72 | **strong** |

Mixing protocols slightly *understated* B4; the corrected L20 value clears the 0.85 threshold
more comfortably. A1/A2 were taken from the recorded run (`--reuse-weight-space`): the dense
ΔW computation exceeds this pod's 16 GB, and the weight-space metrics do not depend on the mask.
The loving norm ratio 0.72 (logit profile, mask-independent) remains close to the 0.7 boundary.

For goodness, sycophancy and impulsiveness this needs corrected-mask extractions of the
released arms (§9).

### 7.2 Stale two-constitution language

`COMMON_TRAJECTORY_FINDINGS.md` is titled for four constitutions, but §1–3 still read "Both
constitutions … the two" (lines 25, 36, 51) with impulsiveness/loving-only tables. Line 314
says "Two constitutions so far". These sections predate goodness/sycophancy and should be
rewritten over the four-constitution panel.

### 7.3 Matched-dose bracketing

`control_dose.py` correctly refuses to extrapolate. One point is unbracketed: **goodness L20
at 0.24 ep**, dose 0.665 vs ladder minimum 0.668, so its excess is undefined. FINDINGS line 158
("bracketing the trajectories") should say "bracketing every trajectory state except goodness
at 0.24 ep at L20". The endpoint table is unaffected.

### 7.4 Persona sets

Behaviour pooled 11 personas (including `null`); geometry uses the 10 semantic personas. With
the 10 (`behaviour.py --exclude-null`, `v2/behaviour_<c>_10personas.json`):

| q at 0.24 / 0.48 / 1.0 ep | 11 personas (reported) | 10 personas |
|---|---|---|
| impulsiveness | 0.59 / 0.76 / 0.90 | 0.64 / 0.80 / 0.91 |
| loving | 0.66 / 0.85 / 0.99 | 0.73 / 0.94 / 1.01 |
| goodness | 0.41 / 0.83 / 1.00 | 0.42 / 0.84 / 1.01 |
| sycophancy | 0.44 / 0.70 / 0.80 | 0.45 / 0.71 / 0.82 |

Behaviour arrives slightly earlier on the matched set, so "geometry lags behaviour" is, if
anything, understated (loving: Q 0.72 vs q 0.94 rather than 0.85). Comparisons should use the
matched set.

### 7.5 Reproduction materials

The trajectory and curve launchers (`finetuning/*/llama_*.sh`) and OCT's tracked-file changes
existed only in the untracked OCT clone; manifests record only `oct_dirty: true`. Snapshotted to
`docs/runs/oct/external_launchers/` with hashes. Historical manifests carry no launcher hash, so
byte-identity with what ran is unverified. Future manifests record diff and launcher hashes.

### 7.6 82 % vs ~76 %

Distinct quantities: the manuscript's 82 % is target **selectivity**. The ~76 % is the
impulsiveness eight-trait **profile projection** q(0.48 ep), which is 0.80 on the geometry's
ten personas. Keep them separate.

### 7.7 Loving scoping, P0 sentence

FINDINGS already scopes loving to this reproduction, which under-recovers released
empathy/warmth, and says the released stage weights are unavailable (lines 18, 203, 307). The
P0 sentence in `arxiv/main.tex` already reads "not explained by the model being displaced
less", and that conclusion is robust to the mask: P0's dose is 1.17× the released adapter's,
against a mask effect on dose ratios of ≤ 0.5 %.

---

## 8. Which conclusions survive

"Survives" means it holds under every check run here. It does **not** mean "verified": the
saved arithmetic reproducing is not evidence that a result is right. Everything is one training
seed per arm; question-bootstrap intervals do not include seed variance.

**Common trajectory**
- *Survives:* behavioural change arrives early in SFT (q ≥ 0.70 by 0.48 ep for all four,
  slightly earlier on the geometry's ten personas); the persona-common direction arrives early
  (L15 cosT 0.885–0.922 by 0.24 ep for every constitution, either base); geometry magnitude/projection lag
  behaviour (the gap widens on matched personas). All L15 geometry numbers are unchanged by the
  estimator repair, and move ≤ 0.015 under the corrected base. Fidelity categories stay
  "strong", and loving B4 improves under a consistent protocol.
- *Weakens:* L20 direction statements for impulsiveness (cosS, Q) and sycophancy (cosT, cosS,
  Q) hold only for the energy-pooled estimand. Any "trait X rises/falls vs base" statement
  holds as an **offset at base indifference**, not as a change in average preference (loving
  empathy: +1.50 offset, −0.43 mean change). Matched-dose excesses have no paired interval.
- *Unvalidated:* everything TOTAL-vs-base rests on a synthetic corrected base until G1 runs;
  the released models' stage dynamics (stage weights unavailable).

**Cross-character pilot**
- *Survives:* impulsiveness transfers and displaces loving. Every target and LOV change keeps
  its sign with CIs excluding 0 under all three logit estimators, within ~10 %. Activation
  selectivity corroborates it (moves ≤ 0.06 with a corrected base). Training used the intended
  corpora and starts (provenance now documented).
- *Weakens:* the moderation term keeps its sign but its size depends on the estimator (B1
  −0.31 vs −0.59; LOV −0.31 vs −0.15). "lov→lov does not erode loving" holds for LOV under all
  estimators; the "all |Δ| ≤ 0.16" bound holds only for the reported estimator. Wording fixes:
  empathy 10/11 not 9/11; checkpoint labels; "held out".
- *Unvalidated:* seed variance; a neutral-data control of equal disruptiveness.

**Behavioural Persona-Vectors eval**
- *Survives:* misalignment raises `evil` in every role, also under simultaneous intervals, with
  strongly role-dependent magnitude. Judging and arithmetic reproduce exactly.
- *Weakens:* role-deviation counts (5 → 4 simultaneously); the goodness drill-sergeant decrease
  does not survive multiplicity; "near-zero" goodness/impulsiveness effects are floor-limited.
- *Unsupported as worded:* coherence drop "not degradation"; caring roles express "the same
  disposition"; a dated judge snapshot.

---

## 9. Minimal rerun proposal (needs approval)

Nothing below has been run.

**CPU reanalysis (no paid compute; can proceed on request)**
1. Joint bootstrap of trajectory and ladder for paired matched-dose excess intervals (same
   question draws for both; existing qcaches). ~1–2 h CPU per constitution.
2. After (G1): rerun geometry, fidelity, control_dose, xchar selectivity on the corrected base;
   `resummarise_geometry.py` for any rule change.
3. Update FINDINGS / XCHAR / behav REPORT wording per §5–7 (docs only).

**GPU extraction (approval needed)**
- **G1 (recommended, smallest consistent set):** corrected-mask CAA activations for `base`
  and the released `impulsiveness`, `goodness`, `sycophancy` adapters (released loving already
  exists corrected as `xchar_lov_start`). 4 arms × ~20–30 min ≈ **1.5–2 GPU-hours** on the RTX
  PRO 6000. Purpose: puts every TOTAL statistic and fidelity B4/A4 in the three extensions on
  one protocol; replaces the synthetic base. Historical legacy results stay as recorded.
- G2 (optional): the same for `mathematical` and `misalignment` (pre-existing published
  analyses only), +~1 GPU-hour.
- G3 (optional): paired option-swapped forced-choice logits (A/B order reversed) for base,
  the 5 xchar states and the 4 trajectory DPO/final pairs, to estimate letter bias
  empirically. ~13 arms × ~10 min ≈ 2 GPU-hours.

**Judge calls (approval needed, optional)**
- J1: re-judge the existing 440 misalignment responses (plus 440 base, paired) with an added
  harmful-advice-severity rubric alongside `evil`, to separate overt malevolence from subtler
  harmful advice in caring roles (§6.4 item 2). ~880 calls to gpt-4.1-mini, est. **< $2**.

**Training:** none needed. No conclusion here requires retraining.

---

## 10. Paper readiness

| extension | readiness | blocking before the paper | not blocking |
|---|---|---|---|
| **Cross-character SFT pilot** | **closest to ready** | text fixes (§5.2–5.6: labels, held-out, 10/11, moderation size with estimator named, mask caveat on activation numbers) | G1 (would remove the activation caveat); more seeds (needed for any variance claim) |
| **Behavioural PV eval** | **ready after wording** | REPORT/manuscript fixes (§6.2 judge identity, §6.3 simultaneous counts, §6.4 four interpretations) | J1 if the caring-role interpretation is to be kept at all |
| **Common trajectory** | **needs the most** | FINDINGS rewrite for four constitutions (§7.2); L20 directions via the pooled estimand or dropped; offset-not-mean wording for trait levels vs base; persona-matched q vs Q; one bracketing sentence; G1 (or explicitly label TOTAL statistics as legacy-base-relative with the measured ≤ 0.015 sensitivity) | joint ladder/trajectory bootstrap (only if matched-dose excesses are quoted with uncertainty) |

The P0 sentence in `arxiv/main.tex` is already right. The 82 % (target selectivity) and ~76 %
(profile projection) figures must stay distinct.

---

## 11. Commits and files

- `52d8054` audit: record and enforce CAA mask protocol; measure the mask artifact
- `5066816` audit: geometry ratios are undefined, not epsilon-floored, at non-positive energy
- `5bea7c6` audit: logit offset estimator -- expose polarity slopes, add shared-slope sensitivity
- `10decf1` audit: xchar provenance corrections; manifests must name their inputs
- `e19bc20` audit: behav-pv simultaneous intervals across the ten roles
- `a363cc7` audit: fidelity guards; behaviour on the geometry's ten personas
- `1bf99a6` figures.py --panels: panel-subset figure (fig_trajectory_AB.png)
- `27546dd` audit: pooled estimand undefined for the DPO state's incremental ratios; fidelity --reuse-weight-space
- `e5d8890` audit: geometry and loving fidelity reruns (CPU)
- (this document)

Key new files:
- `docs/AUDIT_POST_SUBMISSION_RESULTS.md` (this file)
- `persona_steering/provenance.py`, `data/provenance/qcache_registry.json`
- `scripts/common_trajectory/estimability.py`, `tests/test_geometry_estimability.py`, `tests/test_logit_estimator.py`
- `scripts/audit/`: `mask_artifact.py`, `synth_corrected_base.py`, `logit_sensitivity.py`,
  `xchar_checkpoint_tokens.py`, `xchar_selectivity_mask.py`, `compare_geometry.py`, `resummarise_geometry.py`
- `outputs/analysis/audit/` (measurements), `outputs/analysis/common_trajectory/v2/` (reruns),
  `outputs/behav_pv/analysis/*_audit.*`, `docs/runs/oct/xchar_*_sft.provenance_correction.json`,
  `docs/runs/oct/external_launchers/`
- `outputs/analysis/common_trajectory/fig_trajectory_AB.png` (panels A/B only; original figure untouched)

Not committed: the 1.6 GB synthetic base `outputs/analysis/audit/qcache_synth/` (rebuild with
`scripts/audit/synth_corrected_base.py`). Tests: `python -m pytest tests/` (20 pass). Isolated
helper environments created on the volume: `/workspace/pylibs-audit` (pytest),
`/workspace/pylibs-hfbackup` (hf CLI).
