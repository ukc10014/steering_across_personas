# Dose calibration — Stage 2 results (2026-08-20, RTX 3090)

**Question.** What LoRA scale `s` puts `impulsiveness` and `misalignment` at `goodness`'s
natural functional dose, so the dose-response grid can be run at matched dose?

**Answer.** Neither guessed scale works, and the exercise showed the matched-point design
should be replaced by a **within-arm scale ladder**. Details below; the recommendation is
in the last section.

Everything here is CPU-cheap or ~2 min of GPU per config. No long extraction was launched.

---

## 0. What was verified before measuring anything

**No merge step is needed.** All four arms are plain LoRA (r=64, alpha=64, no rsLoRA, no
DoRA, no `modules_to_save`, same seven projection targets), so
`W(s) = W_base + s·(alpha/r)·B·A` applies in memory in 4 s for 224 modules.
`dose_calibrate.py --verify-scale1` compares the s=1 patch to the archived merged
checkpoints: **bit-identical**, `rel‖pred − merged‖ = 0.000e+00` on every module tested, for
both `goodness` and `misalignment`. Runbook Stage 1 (40 min of merging and 16 GB per
candidate scale, plus a `peft` dependency) is deleted.

**The extraction path reproduces the archive.** Measured on the same cells and questions as
the cache, the freshly extracted s=1 configs give trait-vector dose ratios of
1.000 / 0.944 / 1.350 / 1.348 against the archive's 1.000 / 0.942 / 1.350 / 1.348.

---

## 1. Measured doses

Layer 15, relative to `goodness` at s=1, on the runbook's diagnostic grid
(3 personas x 2 traits x pos/neg x 150 questions, `--legacy-mask`). Absolute values are the
mean over cells of `‖V_arm − V_base‖/‖V_base‖` and
`mean_q‖h_arm − h_base‖ / mean_q‖h_base‖`.

| config | trait-vector | vs goodness | answer-token | vs goodness |
|---|---|---|---|---|
| `goodness` s=0.25 | 0.3333 | 0.486x | 0.2429 | 0.442x |
| `goodness` s=0.5 | 0.4461 | 0.651x | 0.3434 | 0.625x |
| **`goodness` s=1** | **0.6855** | **1.000x** | **0.5490** | **1.000x** |
| `mathematical` s=1 | 0.6471 | 0.944x | 0.5314 | 0.968x |
| `impulsiveness` s=0.25 | 0.3722 | 0.543x | 0.2685 | 0.489x |
| `impulsiveness` s=0.5 | 0.5565 | 0.812x | 0.3709 | 0.676x |
| `impulsiveness` **s=0.83** | 0.8426 | **1.229x** | 0.5180 | **0.944x** |
| `impulsiveness` s=1 | 0.9252 | 1.350x | 0.5757 | 1.049x |
| `misalignment` s=0.25 | 0.3884 | 0.567x | 0.2608 | 0.475x |
| `misalignment` s=0.5 | 0.6943 | 1.013x | 0.3972 | 0.723x |
| `misalignment` **s=0.73** | 0.8416 | **1.228x** | 0.4903 | **0.893x** |
| `misalignment` s=1 | 0.9241 | 1.348x | 0.5705 | 1.039x |

Layer 20 is in `outputs/analysis/dose_calibration.json` and agrees throughout.

### Verdict on the two guessed scales

Neither is close enough to justify a full run, and they fail in **opposite directions on the
two measures**:

| candidate | trait-vector | answer-token | verdict |
|---|---|---|---|
| `impulsiveness` s=0.83 | 1.229x — **23% high** | 0.944x — 6% low | reject |
| `misalignment` s=0.73 | 1.228x — **23% high** | 0.893x — 11% low | reject |

---

## 2. Why the guesses missed: dose is strongly sublinear in `s`

The guesses assumed dose is linear in `s` (divide by the measured dose ratio). It is not.
Fitting a local exponent `dose ∝ s^p` between adjacent measured scales:

| arm | measure | p near s=0.25–0.5 | p near s=0.8–1 |
|---|---|---|---|
| `impulsiveness` | trait-vector | 0.82 | 0.50 |
| `impulsiveness` | answer-token | 0.57 | 0.57 |
| `misalignment` | trait-vector | 0.84 | 0.53 |
| `misalignment` | answer-token | 0.48 | 0.59 |

So `p ≈ 0.5–0.85` throughout: **cutting `s` by 27% cut dose by 9–11%.** Getting a materially
lower dose means going well below s=0.5, not shaving 20% off s=1.

## 3. The two dose measures cannot both be matched

Interpolating each arm's measured curve to `goodness`'s s=1 dose:

| arm | s* to match **trait-vector** | s* to match **answer-token** |
|---|---|---|
| `impulsiveness` | 0.65 (L15) / 0.65 (L20) | 0.92 (L15) / 0.87 (L20) |
| `misalignment` | 0.49 (L15) / 0.52 (L20) | 0.93 (L15) / 0.96 (L20) |

A factor of ~1.5–1.9 apart. One scale cannot satisfy both, so a matched-point design forces
an arbitrary choice of dose variable — and that choice would then be doing part of the
inferential work.

The compression is **not** a bias shift the pos−neg contrast cancels: splitting the
answer-token displacement into its question-common and question-varying parts, both scale
together within ~1% (`impulsiveness` at L15: common 1.070x, varying 1.081x). The two
measures are simply different quantities with different dynamic range.

## 4. The runbook's diagnostic grid is biased, and the bias is cell selection

Trait-vector dose ratios computed from the archived cache, varying what is included:

| configuration | goodness | mathematical | impulsiveness | misalignment |
|---|---|---|---|---|
| **reference: 8 traits x 10 personas x 500q** | 1.000x | 0.942x | **1.207x** | **1.370x** |
| grid cells (3p x 2t), 500q | 1.000x | 0.945x | 1.364x | 1.361x |
| grid cells (3p x 2t), 150q | 1.000x | 0.942x | 1.350x | 1.348x |
| 8 traits x 3 personas, 50q | 1.000x | 0.941x | 1.144x | 1.231x |
| 8 traits x 10 personas, 50q | 1.000x | 0.954x | 1.175x | 1.307x |

Cutting 500q → 150q costs ~1%. Restricting to 3 personas x 2 traits costs **+13% on
`impulsiveness`** — and one of the two traits is `impulsivity`, which is the trait that arm
is named after. The grid over-weights exactly the arm it has to measure. `impulsiveness`
also has by far the largest cell-to-cell spread on the full grid (SD 0.193 against 0.083 for
`goodness`), so any small subset of cells estimates it poorly.

**The answer-token measure does not have this problem.** On the same 6-cell grid it lands
within 2% of the reference (1.049x vs 1.070x) where trait-vector is 12% out, because it
never takes a difference of noisy means — question sampling does not inflate it.

> If a cheap grid is ever used for dose again: read **answer-token** dose off it, and do not
> trust trait-vector dose from fewer than the full persona set.

## 5. Recommendation: a scale ladder, not a matched point

Three independent results point the same way — the two measures disagree by ~1.7x on the
matching scale, dose is sublinear so precision in `s` is hard to buy, and a cheap grid
cannot measure trait-vector dose to better than ~12%. All three stop mattering if the design
stops depending on hitting a point.

The decisive observation is in the table above: **at s=1 the arms are already nearly
dose-matched on the answer-token measure** — 0.97x to 1.05x, a 8% spread — while their RDM
preservation spans 0.883 to 0.732. Scaling `s` moves dose over a **2.4x** range, thirty
times wider than the arm-to-arm differences the dose hypothesis is being asked to explain.
So run the ladder and read the outcome as a function of measured dose:

| arm | scales to extract | answer-token dose covered |
|---|---|---|
| `goodness` | 0.25, 0.5, (1 ✅ archived) | 0.44x – 1.00x |
| `impulsiveness` | 0.25, 0.5, (1 ✅ archived) | 0.49x – 1.05x |
| `misalignment` | 0.25, 0.5, (1 ✅ archived) | 0.48x – 1.04x |

**6 new full extractions, ~9 h** (the s=1 arms already exist). Ranges overlap almost
completely, so every matched-dose comparison is available by interpolation and no
calibration accuracy is required — dose is measured on the full extraction itself by
`scripts/functional_dose.py`, not predicted in advance.

What it decides:

| result | reading |
|---|---|
| outcome moves steeply across each arm's 2.4x dose range, curves coincide | generic perturbation-magnitude law; the four-arm ordering was dose |
| outcome barely moves across 2.4x of dose | dose cannot explain a spread produced within 8% of dose — the ρ = −1 ordering was coincidence among four points |
| curves move but do not coincide | dose is real and constitution content adds something on top |

The middle row is the one the current data cannot rule out and a matched-point design would
never have tested.

`mathematical` is left at s=1 only: it sits within 6% of `goodness` on both measures, so its
ladder would duplicate `goodness`'s. Add it later if the curves separate.

---

## Reproducing

```bash
python scripts/dose_calibrate.py --verify-scale1 goodness misalignment
python scripts/dose_calibrate.py --configs base goodness:1 goodness:0.5 goodness:0.25 \
    mathematical:1 impulsiveness:1 impulsiveness:0.83 impulsiveness:0.5 impulsiveness:0.25 \
    misalignment:1 misalignment:0.73 misalignment:0.5 misalignment:0.25
python scripts/dose_calibrate_analyse.py --layers 15 20 --ref goodness_s1
```

~2 min of GPU per config. Activations in `outputs/_dose_calib/`, numbers in
`outputs/analysis/dose_calibration.json`.
