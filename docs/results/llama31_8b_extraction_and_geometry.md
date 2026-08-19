# Llama-3.1-8B character arms: extraction validity and persona geometry

**Status: IN PROGRESS.** Methods and the two settled results are below. The extraction
diagnostic (§2) is running; the geometry results (§4-§7) are not yet computed. Nothing
here supersedes [llama31_8b_character_arms.md](llama31_8b_character_arms.md) or the
retraction in `d44a267` until marked otherwise.

Arms: base `Llama-3.1-8B-Instruct`, plus the `goodness`, `mathematical` and
`impulsiveness` OCT adapters. 8 CAA traits x 10 semantic personas, plus `null` and
`nonsense` controls. Headline layer 15.

---

## 1. Two extraction issues, one real

### 1.1 The attention mask was masking genuine tokens — REAL, fixed in `dc4e234`

`pipeline/2c_caa_activations.py` built its attention mask as `(input_ids != pad_id)`.
That is correct only when `pad_id` cannot occur inside a real sequence. On
Llama-3.1-Instruct it can:

- `tokenizer_config.json` sets `pad_token: None`
- `ProbingModel.__init__` therefore assigns `pad_token = eos_token = <|eot_id|>` (128009)
- `<|eot_id|>` terminates **every turn** of the Llama-3 chat template

Measured on a representative CAA prompt (68 tokens): **3 genuine tokens receive
`attention_mask = 0`**, at positions 36, 61 and 67. The answer token — the position whose
activation is the extracted quantity — is at 66, so **two of the three masked tokens are
causally upstream of it**. The extraction reads an activation computed with the
system-turn and user-turn boundaries hidden from attention.

This is **not a padding bug**. It fires at `batch_size = 1`, where no padding exists at
all, so it is systematic across every sample rather than dependent on batch composition.
`pipeline/10_oracle.py:309` already masked by position; `2c` now matches it.
`--legacy-mask` reproduces the old path for diagnostics and for reproducing archived
activations. Verify with `python scripts/verify_attention_mask.py`.

**Whether this materially changes the trait vectors is the open question**, measured in §2.

### 1.2 The `null` system prompt — NOT a real issue on Llama, no change made

The concern was that `null` constructs an explicit empty system message rather than
omitting the system turn. On Llama-3.1 these are **byte-identical**:

```python
apply_chat_template([{"role":"system","content":""}, user, assistant])
== apply_chat_template([user, assistant])          # True
```

The Llama-3 template unconditionally emits the system block with the "Cutting Knowledge
Date" preamble, so "genuinely no system message" is neither achievable through this
template nor different from what is already done. There is nothing to fix and nothing to
compare. This confirms the note already in `CLAUDE.md`.

The related `f"{persona_prompt}\n\n{user_msg}"` prepend **is** a real latent bug, but it
lives on the branch taken only by models without system-role support — Gemma-2 alone
(`supports_system_prompt()` returns `'gemma-2' not in model_name`). It never executes for
any Llama arm. Changing it would invalidate the published Gemma v2 dataset for no benefit
here, so it is left alone and recorded.

---

## 2. Does the mask fix change the vectors? — RUNNING

Grid: base and `impulsiveness`; traits `impulsivity` and `honesty`; personas `therapist`,
`drill_sergeant`, `con_artist`, plus `null`; 150 questions; layers 15 and 20; both mask
implementations against the same in-memory model, so they differ in exactly one thing.

**A raw cosine is not the criterion.** "cos(V_old, V_fixed) > 0.99, therefore harmless" is
the reasoning `d44a267` retracted a headline for: these vectors carry a large component
common to every persona, and a cosine between two vectors that both contain it is high
almost regardless of what happened to the part that matters. Every comparison is therefore
reported raw **and** persona-centred, and both are calibrated against a
question-resampling floor — cos of two independent bootstrap resamples under the *same*
mask. If the mask effect sits below that floor, changing the mask perturbs the vector more
than redrawing every question does.

Early signal, single cell, per-sample activations (not yet trait vectors): cosine 0.993,
**relative norm change 11.7%** at L15.

---

## 3. Intervention dose: the three adapters are near-identical — SETTLED

The dose confound says a stronger geometric effect under `impulsiveness` might mean only
that it moves the model further. Measured directly from the adapter weights
(`scripts/adapter_dose.py`, CPU, no forward passes), for `dW = (alpha/r) B A` over all 224
targeted projections:

| arm | global relative ‖dW‖_F | vs `goodness` |
|---|---|---|
| `goodness` | 0.0085 | 1.000x |
| `mathematical` | 0.0088 | **1.026x** |
| `impulsiveness` | 0.0087 | 1.017x |

All three sit within **2.6%** of each other, and `mathematical` is nominally the largest.
So the premise that `impulsiveness` is the largest perturbation is **not supported at the
weight level**.

Stated carefully: weight-space norm is not functional dose. Two adapters with equal ‖dW‖
can move behaviour by very different amounts depending on where they sit and how inputs
align with them. This measurement bounds and describes the intervention; it does not
dose-match it. A genuine dose match needs activation displacement on a neutral corpus or
output KL from base, which needs a GPU (§8).

---

## 4. Estimators: what had to be corrected before any geometry was computed

Every geometry quantity here is **quadratic** in the trait vectors, and quadratic
functionals of a noisy estimate are biased upward by the noise variance. Critically the
bias is **not shared across arms**: a noisier arm looks more dispersed and its personas
look further apart with no change in the underlying geometry. So dispersion and RDMs are
cross-fitted — estimate the vector on two disjoint question halves, take their inner
product rather than the squared norm of one. Same correction `d44a267` made for
`residual_cos`, different statistic. `scripts/geometry_lib.py` self-tests it.

**On this data the correction turns out to be small.** Measured on the real base arm at
L15: naive dispersion is inflated only **1.06–1.14x** across the eight traits, and
split-half RDM reliability is **0.959–0.988**. The stress tests below use a deliberately
high-noise regime; real activations are not in it. The estimators are kept because they
are correct, not because they rescue a result.

Four defects were found by testing against synthetic data with known ground truth. Each
would have produced a confident wrong answer:

1. **The bootstrap CI excluded its own point estimate.** Resampling questions then
   splitting the resample lets a duplicated question land in both halves, destroying the
   independence cross-fitting needs. Two further attempts also failed (coverage 88%/65%,
   then 62%/55%). The working version draws once per replicate and assigns each distinct
   question, with all duplicates, to one side. Coverage now **95%** against nominal 95%.
2. **RDM correlations had no noise ceiling.** Where an arm was a *pure rotation* of base —
   every pairwise distance preserved exactly, true correlation 1.0 — measured Spearman read
   **0.72–0.82**. Worse, the ceiling is not shared: a pure 0.6x contraction, also exactly
   shape-preserving, scored *below* the rotation arm purely from lower SNR. Read naively
   that ordering is "mathematical restructures more than goodness", which is false.
   Per-arm split-half reliability plus the attenuation correction recovers 0.90–0.99 and
   0.90–1.24, while genuine restructuring stays at 0.27–0.36.
3. **Pearson-on-shape equalled Pearson-on-absolute** in every cell, because correlation is
   already scale-invariant. Absolute expansion/contraction is now the RMS ratio, which
   recovers the synthetic 0.6x contraction as 0.61.
4. **Procrustes ran on uncentred vectors**, so it mostly aligned the large shared component
   to itself: a known exact global rotation scored `test_proc = 1.007`, *worse than no
   transform*. Persona-centring within trait first recovers 65% explained for that rotation.

`_basis_coverage` reports the fraction of held-out signal a training-fitted basis can
represent — the ceiling on any cross-validated Procrustes score. It separates "no global
transform explains this" from "the test cannot see the signal", and is already load
bearing: hold-out-personas covers **17%** at rank 40, so its `test_proc ≈ 1.0` is
uninformative rather than a finding.

### What the intervals do and do not cover

The question bootstrap quantifies uncertainty from the sampled CAA questions **only**. It
conditions on these particular ten personas and eight traits and says nothing about
generalisation to other personas or traits. With n=10 and n=8 there is no precise
population inference available here.

---

## 5. Pending

- §2 verdict, and hence whether the archived activations need regeneration on a GPU
- §5 dispersion, §6 RDMs, §7 Spearman, §8 Procrustes, §9 residual structure on real arms
- §10 figures on real results
- the dose-matched GPU experiment to recommend

---

## Appendix: moving this to a GPU

The diagnostic and any re-extraction are forward passes, which is the one part of this
plan that genuinely wants a GPU. Measured on 16 CPU cores at ~1.5 s/sample against a
rough GPU estimate:

| job | 16 CPU cores | one A100/H100 |
|---|---|---|
| mask diagnostic, 9,600 forwards | ~4 h | ~5–15 min |
| full re-extraction, ~384,000 forwards | ~96 h | ~1–3 h |

The analysis half is different: dispersion, RDMs, Procrustes and the bootstraps are numpy
over (10, 500, 4096) arrays and run in seconds on CPU. The cache build is bound by network
volume I/O, which neither helps.

Two launchers, each a single command on a fresh GPU pod:

```bash
bash scripts/run_mask_diag_gpu.sh      # the diagnostic
bash scripts/run_reextract_gpu.sh      # ONLY if the diagnostic condemns the archive
```

Three things they handle that bit this work on CPU:

- **`python3` may not be the provisioned interpreter.** `bootstrap.sh` derives `PYLIBS`
  from `python3`, which on this pod image is 3.8 with an empty `pylibs-py38`, while torch
  and transformers live in `pylibs-py312`. It presents as `ModuleNotFoundError:
  transformers` straight after a clean bootstrap. Both scripts pick the interpreter whose
  `PYLIBS` actually contains torch.
- **A device-mixed dataset is silent and fatal.** The diagnostic measures a small
  difference between two attention masks; two devices do not produce bit-identical
  activations, so a legacy half on CPU and a fixed half on GPU puts a device artefact
  inside the measured quantity, with every file present and nothing looking wrong.
  `mask_diag_extract.py` records the device and refuses such a resume;
  `run_mask_diag_gpu.sh` writes to its own directory so it cannot arise. The partial CPU
  run is not worth salvaging — on a GPU the whole thing is minutes.
- **Re-extraction must not overwrite `caa_activations/`.** That archive is what every
  published result and every retraction was computed from. The new run lands in
  `caa_activations_fixedmask/` alongside it.
