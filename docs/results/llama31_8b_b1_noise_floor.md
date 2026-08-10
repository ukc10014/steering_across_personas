# K/D Appendix B.1 on Llama-3.1-8B — bootstrap noise floor

**Date:** 2026-08-10 · **Model:** `meta-llama/Llama-3.1-8B-Instruct` · **Method:** CAA
**Inputs:** `outputs/Llama-3.1-8B-Instruct/caa_activations/` (192 files, M=500 pairs/cell)
**Outputs:** `analysis/caa_within_cell_stability.json`, `analysis/b1_noise_floor.{png,pdf}`
**Code:** `scripts/caa_within_cell_stability.py`, `scripts/plot_b1_noise_floor.py`

CPU-only re-analysis. No GPU, no re-extraction, ~5 min.

---

## Why this exists separately from `caa_cosine_to_null.py`

Both scripts compute "a noise floor", and they are **not the same estimator**. Quoting one
against K/D's number is a category error, and that was the first thing to fix.

- `caa_cosine_to_null.py` computes NULL-vs-NULL under two *independent* resamples. That is
  the correct floor for the statistic it accompanies (a persona-vs-null cosine), and its
  unpaired construction is deliberate — see the docstring at lines 19–24.
- K/D's B.1 rung 1 is **pairwise cosine among B bootstrap replicates of one cell**. Different
  estimand, different construction (resampling is *paired* — the contrastive pair is the
  sampling unit).

`caa_within_cell_stability.py` computes K/D's version, plus two floors they do not.

### The gating question: are the per-pair deltas recoverable?

Yes. The pipeline caches **per-question** activations, not the averaged trait vector.
`{persona}_{trait}_{pos,neg}.pt` are dicts of 500 question-keyed `[32, 4096]` fp16 tensors,
and pos/neg carry **identical keys in identical order**, so `delta[q] = pos[q] - neg[q]` is
exact. The `[M, d_model]` per-cell array that the bootstrap needs already exists on disk;
nothing needed re-extracting. Storage for an explicit delta cache would be ~12.6 GB at all
layers, but it is redundant — the 24 GB of raw activations already is the cache.

---

## Results

Persona-averaged (excludes `null` and `nonsense`), averaged over all 8 traits.

| layer | rung 1 (within-cell) | split-half (SB) | rung 3 (across-persona) | margin (r1 − r3) |
|---|---|---|---|---|
| 10 | 0.913 | 0.906 | 0.958 | **−0.046** |
| 12 | **0.929** | 0.925 | 0.878 | 0.051 |
| 15 | 0.882 | 0.869 | 0.705 | 0.177 |
| 17 | 0.852 | 0.828 | 0.605 | 0.247 |
| 20 | 0.835 | 0.803 | 0.508 | 0.327 |
| 25 | 0.801 | 0.746 | 0.402 | 0.399 |
| 31 | 0.802 | 0.741 | 0.313 | 0.489 |

![B.1 noise floor](../../outputs/Llama-3.1-8B-Instruct/analysis/b1_noise_floor.png)

### 1. K/D's 0.99 does not reproduce — anywhere

Within-cell stability **peaks at 0.929 (layer 12)** and is 0.882 at L15, 0.835 at L20. It
never approaches 0.99 at any layer, despite M=500 pairs against K/D's M=100. The extraction
noise floor on Llama-3.1-8B is materially worse than the number the paper reports.

This is the single most consequential number here: any argument that inherits "extraction is
essentially noiseless, so spread is signal" from K/D is not licensed on this model.

### 2. The hierarchy survives, compressed, and only above layer 11

Rung 1 > rung 3 holds **from layer 11 onward**. Below that the ordering inverts — at layer 10
the across-persona cosine (0.958) sits *above* the floor (0.913), i.e. personas are not
distinguishable from resampling noise in the early stack. The margin then grows monotonically
with depth: 0.177 at L15, 0.327 at L20, 0.489 at L31.

So the qualitative K/D claim reproduces. What changed is the margin, and it is now measured
with their estimator rather than assumed comparable to ours.

### 3. Question-set variance is real and the bootstrap cannot see it

`split_half` is below `boot_half` at **every trait and every layer**. Size-matched at 250
pairs, the gap attributable purely to *disjointness* is **0.025 at L15 and 0.047 at L20**.
Projected back to the full bank (Spearman-Brown), the bootstrap floor is optimistic by
**0.014 at L15 and 0.033 at L20**.

Modest, systematic, and in the wrong direction for a programme whose claims are about
dispersion. The bootstrap resamples the M pairs you have; the split-half varies which
questions you asked.

### 4. The currently-published floor is under-resolved

At `n_boot=50` the floor estimate swings **0.886–0.908 at L15 on the choice of seed alone**.
The value in `caa_cosine_to_null.json` is 0.9042 — the top of that range; a 400-draw estimate
converges to ~0.892. That line is drawn on the published figures. Not wrong, but it should be
re-run with more replicates before it carries any weight in the writeup.

### 5. Root cause is low SNR, not outlier questions

`||mean delta|| / mean||delta||` at L20 is **0.081** (politician) against **0.152** (null).
Per-pair norms are tight (max/median = 1.1), so this is not a handful of pathological
questions. A one-parameter SNR model predicts a split-half cosine of 0.62 against 0.61
observed, so the whole picture is internally consistent: the coherent trait signal is a small
fraction of a typical contrastive pair's magnitude, and personas have vectors that are both
**rotated and weaker** than null's.

That last point bears directly on B.6's "magnitude and direction decouple" claim — here they
do not look independent.

### 6. Layer selection: B.1 alone would pick a useless layer

Bootstrap stability peaks at L12, but at L12 the margin is only 0.051 — there is almost no
fan-out to measure. Stability and separation pull in opposite directions, so B.1 stability
cannot be used as a standalone layer criterion the way K/D's A.2 implies.

Taken together the two criteria **support the existing L20 choice over the pre-designated
L15**: L20 retains a workable floor (0.835) while nearly doubling the margin (0.327 vs 0.177).
That is an independent justification for a deviation `llama31_8b_stage1.md:241` had recorded
on fan-width grounds alone.

---

## Method notes and caveats

- **Spearman-Brown is a heuristic here.** SB is derived for correlations between parallel test
  halves; a cosine between two mean-difference vectors is not literally that. It is reported
  alongside the raw split-half, never instead of it.
- **SB is applied to the summary, not per replicate.** SB is concave, so per-replicate
  averaging understates it; and the split-half distribution has real mass below zero
  (politician at L20 runs [−0.14, 0.92]), so NaN-guarding per replicate silently drops the low
  tail and biases the result *up*. SB is monotone on r > −1, so transforming the summary
  quantiles is exact for lo/hi. The reported centre is SB(mean r), not mean SB(r).
- **Rung 1 and rung 3 are still not the same estimand.** They are now the same functional form
  (pairwise cosine over a set of vectors), which makes the comparison like-for-like in a way
  the raw contrast was not. It is *not* yet a test. The clean version is a variance
  decomposition — see below.
- **Wide intervals are real.** Per-cell split-half 95% intervals routinely span 0.6+ of the
  range. That is the low-SNR regime, not an artifact.

## What this does not yet answer

1. **The middle rung is missing.** `2c_caa_activations.py:369` passes
   `persona.default_system_prompt`, which `config.py:99-101` defines as
   `system_prompt_variants[0]` — **one paraphrase of five**. We have rungs 1 and 3 but not 2,
   so the "identity, not phrasing" argument cannot be made on our own model, only inherited.
   Requires a 5× re-extraction, not analysis.
2. **The variance decomposition is blocked on (1).** The across-paraphrase component of an
   ICC-style within/paraphrase/persona partition *is* rung 2. Build the estimator once the
   paraphrase arm exists; it is the thing that turns "0.88 ≫ 0.51" from suggestive into a test
   with an error bar on a dispersion statistic.
3. **The adapted model is unmeasured.** If a LoRA adapter shifts activation noise
   characteristics, apparent tightening under a constitution is confounded with a moving floor.
   `outputs/` currently contains only the base model.

## Reproduce

```bash
source /workspace/bootstrap.sh
python scripts/caa_within_cell_stability.py --model meta-llama/Llama-3.1-8B-Instruct \
    --n-boot 50 --n-splits 100 --report-layers 12 15 20
python scripts/plot_b1_noise_floor.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20
```

Needs only `numpy` + `torch` (+ `matplotlib` for the figure). No GPU.
