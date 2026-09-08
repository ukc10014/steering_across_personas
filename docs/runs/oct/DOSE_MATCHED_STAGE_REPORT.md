# Dose-matched stage localisation — result

Measured 2026-09-08, layer 15, forced prompt, seed 123456. 12 rungs, no retraining: every
state is an existing adapter scaled with `--lora-scale`.

> **Both differences the raw decomposition suggested survive dose matching.** At equal
> measured functional dose, `M_S` exceeds `M_D` on the registered endpoint by **8–13×**, and
> `M_F` exceeds `M_D+0.25S` by **~2×**.

## 1. Why this run existed

The first stage decomposition produced an ordering — `M_D` +0.13, `M_D+0.25S` +0.50,
`M_F` +1.92, `M_D+S` +2.19, `M_S` +3.61 — that was **perfectly rank-confounded with measured
functional dose** (Spearman +1.000). Stage and displacement were not separated, so no stage
claim survived from it. This run compares the states at matched dose instead.

## 2. Method, and its two refusals

Each state was scaled to span its own dose range, then compared **only inside the range where
both members of a pair were actually measured**.

- **Dose is not linear in scale**, within any state — `dose/s` spans ~2× (see
  `figA_oct_dose_calibration`). Rungs were therefore chosen from measured dose, never from
  nominal scale.
- **Two states turn over at s=2** — `M_D+S` and `M_S` show *falling* dose as the perturbation
  grows. Interpolation is restricted to each state's monotone prefix, and
  `dm_common.interp_at` raises rather than silently interpolating a non-monotone axis.
- **No extrapolation.** An anchor outside a state's measured range is refused, not estimated.

Pair-specific overlap was used rather than one common band across all five states (option b):
forcing `M_D` and `M_S` into a shared band with `M_D+0.25S` and `M_F` would have pushed `M_D`
further into an awkward scale regime for no gain to either question.

## 3. Result — pair 1, `M_D` vs `M_S`

Overlap: measured dose **0.576–0.827**.

| matched dose | `M_D` $B_1$ | `M_S` $B_1$ | difference | `M_D` sel. | `M_S` sel. |
|---|---|---|---|---|---|
| 0.586 | +0.158 | **+2.118** | +1.960 (13.4×) | 1.112 | **1.591** |
| 0.702 | +0.236 | **+2.721** | +2.485 (11.5×) | 1.129 | **1.730** |
| 0.817 | +0.373 | **+3.112** | +2.739 (8.3×) | 1.157 | **1.764** |

**Survives, overwhelmingly.** The gap is far larger than anything dose matching could
explain, and selectivity separates in the same direction.

**What this licenses.** At comparable functional displacement, the DPO-generated introspection
corpus learned through SFT is a far more potent carrier of the `impulsiveness` phenotype than
the DPO weight update itself.

**What it does not.** It does **not** show DPO is unnecessary. DPO generated the corpus `M_S`
trains on; removing DPO from the weights does not remove it from the causal history.

## 4. Result — pair 2, `M_D+0.25S` vs `M_F`

Overlap: measured dose **0.567–0.853**. These two differ *only* by peft's factor-space cross
terms.

| matched dose | without cross terms | with cross terms | difference |
|---|---|---|---|
| 0.578 | +0.513 | **+0.927** | +0.415 (1.8×) |
| 0.710 | +0.729 | **+1.551** | +0.822 (2.1×) |
| 0.842 | +0.972 | **+2.004** | +1.032 (2.1×) |

**Survives.** The released-style merge has functional consequences beyond the intended
additive `dW_D + 0.25·dW_S` update, at matched dose.

Stated as the wording rule requires: *at matched functional dose, including the cross terms
changes $B_1$ from +0.51 to +0.93 (dose 0.578) and from +0.97 to +2.00 (dose 0.842).* No
percentage of behaviour is attributed to the cross terms — behaviour is nonlinear and no such
share is defined. The 61.9% figure reported elsewhere is a **weight-space norm ratio**.

## 5. What survives dose matching, and what does not

| first-pass observation | verdict |
|---|---|
| `M_S` far stronger than `M_D` | **survives** — 8–13× at matched dose |
| `M_F` stronger than `M_D+0.25S` (cross terms matter) | **survives** — ~2× at matched dose |
| `M_S` stronger than `M_F` in raw numbers | **not tested here** — no overlap was constructed for that pair |
| the raw six-state *ordering* as a stage result | **does not survive** — it was the dose ordering (Spearman +1.000) and should never be shown without that caveat |

## 6. Caveats

1. **Pair 1's matching is asymmetric.** Reaching the overlap needs `M_D` at s≈1.36–1.98 and
   `M_S` at s≈0.51–0.67, so the comparison is between two artifacts *neither of which is a
   trained state*. Inherent — `M_D` is simply the weakest perturbation — and it bounds how
   strongly pair 1 can be phrased. Pair 2 is cleaner: its overlap sits comfortably inside both
   states' measured ranges and includes both trained states.
2. **Single seed.** Everything here is seed 123456. Seed 987654 exists for the un-matched
   states but was not dose-laddered.
3. **Question-bootstrap intervals** in the source data are uncertainty over CAA questions, not
   over training seeds, and are not reported as the latter.
4. **`M_D+S` was not laddered** — it is not a member of either primary pair.

## 7. Assets

`workshop_iclr/figures/`: `figA_oct_stage_pipeline`, `figA_oct_dose_calibration`,
`figA_oct_matched_dose`. `workshop_iclr/tables/`: `tableA_oct_stage_raw`,
`tableA_oct_matched_dose`, `tableA_oct_peft_merge`. Master data:
`outputs/analysis/oct_stage_dose_master.csv`. Provenance and inclusion rules:
[APPENDIX_ASSET_NOTES.md](APPENDIX_ASSET_NOTES.md).

**Panel D uses selectivity, not the cosine**, and the reason is recorded rather than left
implicit: `M_F`'s trained state *is* the reference direction, so its cosine is pinned at
1.000 by construction and the curve would report the definition rather than the data.

## 8. Not done

The dense SFT checkpoint curve, the conditional DPO curve, seed 2 dose ladders, and any sham
arm. No preregistered threshold was changed and no existing workshop claim was edited.
