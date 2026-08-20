# Llama-3.1-8B character arms: extraction validity and persona geometry

**Status: extraction diagnostic and layer-15/20 geometry SETTLED. Nothing
here supersedes [llama31_8b_character_arms.md](llama31_8b_character_arms.md) or the
retraction in `d44a267` until marked otherwise.

Arms: base `Llama-3.1-8B-Instruct`, plus the `goodness`, `mathematical` and
`impulsiveness` OCT adapters. 8 CAA traits x 10 semantic personas, plus `null` and
`nonsense` controls. Headline layer 15.

### Answers to the review's eight questions

This doubles as the reply to the character-arms review prompt. Its closing questions map to
sections as follows; **two of its premises did not hold** and are marked.

| # | question | answer | where |
|---|---|---|---|
| 1 | Does the mask fix change the vectors? Is regeneration needed? | Bug real, effect minor. **No regeneration.** | §1.1, §2 |
| 1b | Does correcting `null` formatting change them? | **Premise wrong** — byte-identical on Llama-3.1; nothing to fix | §1.2 |
| 2 | Does the large common OCT component remain? | Yes, unchanged; the mask fix does not touch it | §2, `d44a267` |
| 3 | Does full hidden-space dispersion actually contract? | **Yes** — ~20–30% in linear (RMS) terms; a semantically different constitution reproduces most of it | §5.1 |
| 4 | How well does each arm preserve the base persona RDM? | `impulsiveness` least (0.822 vs 0.858/0.905), modestly | §5.2 |
| 5 | Does the low scalar Spearman for `impulsiveness` survive? | **Yes — and it is the only effect the control fails to reproduce** | §5.3 |
| 6 | How much does a global *orthogonal* transform explain? | Little: 15–18% Frobenius (28–32% squared) traits; 1–6% personas | §5.4 |
| 7 | Structured residual change? Stronger for `impulsiveness`? Localised? | Larger, but located **identically across arms** — not constitution-specific | §5.5 |
| 8 | How much larger is the `impulsiveness` perturbation? | Weight norm matched, but **functionally 1.21x** (and `misalignment` 1.37x). Weight norm was the wrong dose variable | §3, §3.1, §5.55 |
| + | How much of an adapter's effect is one persona-common shift, and do constitutions share it? | The shift is **0.6–0.9x** the base trait vector and carries **67–77%** of the change (L15), but the constitutions' shifts are only **0.47–0.83** aligned — similar sizes, different directions. `impulsiveness` is **1.7–1.9x selective** for `risk_taking`/`impulsivity` | §3.2 |

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

## 2. Does the mask fix change the vectors? — NO. The archive stands.

Run on an RTX 3090, 64 cells in ~9 min (the same grid took ~4 h on 16 CPU cores). Base and
`impulsiveness`; traits `impulsivity` and `honesty`; personas `therapist`,
`drill_sergeant`, `con_artist` plus `null`; 150 questions; layers 15 and 20; both masks
against the same in-memory model on the same device.

**A raw cosine is not the criterion.** "cos(V_old, V_fixed) > 0.99, therefore harmless" is
the reasoning `d44a267` retracted a headline for. Every comparison is reported raw *and*
persona-centred, and calibrated against how much the vector moves when you change which
questions you ask.

### The numbers

| quantity | L15 | L20 |
|---|---|---|
| cos(V_legacy, V_fixed), raw | 0.979 – 0.996 | 0.971 – 0.995 |
| cos, persona-centred | 0.971 – 0.992 | 0.968 – 0.991 |
| ‖V_fixed − V_legacy‖ / ‖V_legacy‖ | 0.093 – 0.205 | 0.100 – 0.242 |
| ‖V_fixed‖ / ‖V_legacy‖ | 0.968 – 1.029 | 0.912 – 1.039 |

Note the last two rows are different quantities and it is easy to conflate them. The
vectors differ by 9–24% of their length, but their *magnitudes* are unchanged to about 1%
— the difference is in direction, not scale, so norm-sensitive statistics are not
disturbed by a systematic rescaling.

### Calibration: the mask matters less than the question set

**0 of 32 cells** show a mask effect exceeding question-sampling noise. Two *disjoint*
question subsets, same mask, produce trait vectors agreeing at only 0.31–0.80 (n=25–75),
extrapolating to **0.59–0.91 at n=500** under a 1/cos = 1 + c/n fit. The mask effect is
0.98–0.99. So changing the attention mask perturbs the trait vector **far less than
changing which questions you ask** — at every n measured, and at the archive's n=500.

The fitted asymptote lies below 1.0, which is worth noting on its own: disjoint question
subsets do not converge to the same trait direction even in the limit. That is genuine
question heterogeneity, not sampling noise, and it is a bigger source of variation in
these vectors than the bug is.

### The statistic that actually matters is essentially unmoved

The reported results are all arm-vs-base contrasts, and the fix shifts both arms in the
same direction, so it largely cancels:

| trait, L15 | gap (impulsiveness − base), legacy | fixed | change |
|---|---|---|---|
| `impulsivity` | +0.1708 | +0.1778 | **+0.0070** |
| `honesty` | +0.1766 | +0.1739 | **−0.0027** |

Against a headline gap of ~+0.18, the mask fix moves it by under 0.01. Absolute
cosine-to-null does shift more (up to −0.06 at L20), but as a common-mode change.
**Persona ordering: 8/8 preserved**, every trait × layer × arm.

### Verdict

The bug is real and worth fixing — it is fixed — but it is **empirically minor for these
analyses**, and the archived Llama activations remain usable. **No GPU re-extraction is
required.** `scripts/run_reextract_gpu.sh` exists if that changes.

One caveat, flagged rather than buried. Dispersion is norm-sensitive in a way cosine is
not, and on this small grid the arm/base dispersion ratio moved by +0.02 to +0.05 on a
base of 0.38–0.80, i.e. roughly 5–10% relative. That is measured on only three semantic
personas, so it is noisy, but it is the one statistic where the fix could plausibly matter.
Spot-check it against fixed-mask data before treating any dispersion result as final.

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
weight level**. It *is* supported functionally — see §3.1, which is why this does not
retire the dose confound.

Stated carefully: weight-space norm is not functional dose. Two adapters with equal ‖dW‖
can move behaviour by very different amounts depending on where they sit and how inputs
align with them. This measurement bounds and describes the intervention; it does not
dose-match it. A genuine dose match needs activation displacement on a neutral corpus or
output KL from base, which needs a GPU (§8).

### 3.1 Functional dose is NOT matched, and this was stated wrongly before

Weight-space norm is not functional dose, and an earlier draft said exactly that in one
sentence and then listed intervention magnitude as "ruled out" in another. The two are
inconsistent and the second is wrong.

Measured directly from the cached activations (`scripts/functional_dose.py`, CPU, no new
inference), at L15 over all 80 trait x persona cells:

| arm | trait-vector displacement ‖V_arm − V_base‖/‖V_base‖ | vs `goodness` | answer-token displacement |
|---|---|---|---|
| `goodness` | 0.705 | 1.000x | 0.536 |
| `mathematical` | 0.665 | 0.942x | 0.529 |
| **`impulsiveness`** | **0.852** | **1.207x** | 0.573 |

`impulsiveness` moves the representation ~21% more than `goodness` and ~28% more than
`mathematical` on the very data being analysed, despite an essentially identical weight-space
norm. This reproduces a number already in the repo — `character_arms.md` §5 records ‖d‖/‖v‖
of 0.870 against 0.709 and 0.676 — so the earlier claim contradicted a prior result as well
as itself.

**Correct statement: raw weight-update magnitude is closely matched; functional perturbation
magnitude remains a live confound.** Any effect where `impulsiveness` is the outlier — most
importantly the residual-to-null ordering collapse in §5.3 — has "it simply moves the model
further" as a standing alternative explanation.

Caveat on scope: this conditions on the CAA prompts, so it is functional dose *on this data*,
which is the relevant quantity for interpreting these geometry results. It is not a
substitute for neutral-corpus KL if the question is about the model overall.

### 3.2 The persona-common shift: how big, and is it the *same* shift?

`scripts/common_shift.py`. Write `V_{c,t,p}` for a trait vector under constitution `c`, and

    dV_{c,t,p} = V_{c,t,p} − V_base_{t,p}          the per-persona change
    dG_{c,t}   = mean_p dV_{c,t,p}                 its persona-COMMON part

Three questions the prose above had been running together: how **large** is the common
shift, how much of each adapter's effect it **accounts for**, and whether different
constitutions add the **same** shift or merely similarly-sized ones. All three are
quadratic in vectors estimated from 500 questions, so all three are cross-fitted on
disjoint question halves (§4); naive values are printed alongside in
`outputs/analysis/common_shift.txt`. Two biases specific to this analysis motivated that:
the common **share** is biased *downward* by noise (independent noise inflates
`mean_p‖dV_p‖²` by the full per-persona variance but `‖dG‖²` by only 1/P of it), and the
**cosines** are biased *upward*, because `dV_c` and `dV_c'` subtract the same estimate of
`V_base` and so share a `−ε_base` error term. In a synthetic check at high noise the naive
cosine of two genuinely unrelated shifts read **+0.121** against a true −0.035, while the
cross-fitted estimator returned −0.020. On this data the correction turns out to be small
(mean 0.607 against a naive 0.586 in table 1) — which could not be known in advance.

#### The common shift is the same order of magnitude as the trait vector it displaces

`‖dG_{c,t}‖ / mean_p‖V_base_{t,p}‖`, layer 15 (10 semantic personas):

| trait | goodness | mathematical | impulsiveness | misalignment |
|---|---|---|---|---|
| assertiveness | 0.637 | 0.586 | 0.644 | 1.148 |
| empathy | 0.607 | 0.578 | 0.641 | 0.880 |
| risk_taking | 0.629 | 0.577 | **1.097** | 1.022 |
| honesty | 0.585 | 0.558 | 0.617 | 0.755 |
| confidence | 0.608 | 0.662 | 0.679 | 0.810 |
| deference | 0.662 | 0.541 | 0.645 | 0.738 |
| warmth | 0.570 | 0.568 | 0.696 | 0.914 |
| impulsivity | 0.562 | 0.526 | **1.154** | 1.060 |
| **mean** | **0.607** | **0.574** | **0.772** | **0.916** |

Layer 20 is the same picture, larger: means 0.682 / 0.618 / 0.948 / **1.044**. Character
training does not nudge these trait representations — it displaces them by an amount of the
same order as the representation itself, and for `misalignment` at L20 by more.

#### Most of each adapter's effect on trait vectors is that one shift

`mean_p‖dV_p‖² = ‖dG‖² + mean_p‖dV_p − dG‖²` is an exact partition, so the **share** is the
squared ratio. The linear ratio asked for is its square root and is not a share — a 25%
share reads as 0.50 in linear units, so both are tabulated and labelled.

| | goodness | mathematical | impulsiveness | misalignment |
|---|---|---|---|---|
| **L15 share** `‖dG‖²/mean_p‖dV_p‖²` | 0.673 | 0.684 | 0.723 | 0.766 |
| L15 linear `‖dG‖/mean_p‖dV_p‖` | 0.820 | 0.827 | 0.849 | 0.875 |
| **L20 share** | 0.534 | 0.541 | 0.653 | 0.672 |
| L20 linear | 0.730 | 0.734 | 0.806 | 0.818 |

So **two thirds to three quarters of what an adapter does to trait geometry at L15 is a
single persona-independent translation**, and the remaining quarter to third is everything
persona-specific. The share is systematically lower at L20 (0.53–0.67), so the shift is
less dominant deeper in the stack. It is also consistently higher for the two larger-dose
arms, which is what a saturating common displacement would look like.

#### But they are NOT the same shift

`cos(dG_{c,t}, dG_{c',t})`, averaged over the 8 traits, layer 15 (naive in brackets):

| | goodness | mathematical | impulsiveness | misalignment |
|---|---|---|---|---|
| goodness | 1.000 | **0.831** [0.81] | 0.656 [0.64] | **0.521** [0.52] |
| mathematical | 0.831 | 1.000 | 0.670 [0.66] | 0.465 [0.47] |
| impulsiveness | 0.656 | 0.670 | 1.000 | 0.648 [0.65] |
| misalignment | 0.521 | 0.465 | 0.648 | 1.000 |

Layer 20 agrees: 0.818 / 0.627 / 0.560 in the `goodness` row.

These are substantial but far from 1. The constitutions add shifts that **overlap
partially and differ systematically** — this is the distinction the earlier prose blurred,
and it comes out on the side of "different shifts, of similar size" rather than "one common
shift". The separation is resolved: at L15 the `goodness`–`mathematical` interval clears the
`goodness`–`misalignment` interval in **7 of 8 traits** (honesty: 0.866 [0.817, 0.928]
against 0.576 [0.496, 0.643]); at L20, where intervals are roughly twice as wide, in 4 of 8.

The structure is orderly: the two low-dose constitutions pair at 0.83, the two high-dose
ones at 0.65, and cross-pairs sit at 0.47–0.67. Cosine is scale-invariant, so this is not
dose arithmetic — it is a claim about direction. Whether it is *content* or a shared
direction that any large perturbation drifts into is exactly what the dose ladder can
separate, by asking whether `goodness` at s=0.25 points where `goodness` at s=1 points.

#### The one clearly content-specific effect in this data

`impulsiveness`'s common shift is not uniform across traits. It is ~0.65 on six traits and
**1.10 / 1.15 on `risk_taking` and `impulsivity`** — the two traits its constitution is
about. As a ratio of the two related traits to the other six:

| arm | L15 | L20 |
|---|---|---|
| goodness | 0.97 | 0.99 |
| mathematical | 0.95 | 0.97 |
| **impulsiveness** | **1.72** | **1.87** |
| misalignment | 1.19 | 1.35 |

`goodness` and `mathematical` are flat to within 5%. `impulsiveness` is selective by a
factor of ~1.8 at both layers; at L15 the weaker of its two related-trait intervals clears
the strongest of its other six (0.876 vs 0.871), a deliberately conservative min-vs-max
test that does not survive L20's wider intervals. `misalignment` sits in between and is
elevated more or less everywhere, which is what a large generic perturbation should look
like.

This matters because **no other statistic in this document distinguishes constitution
content from perturbation size.** RDM preservation, dispersion and the residual-to-null
ordering are all scalar summaries that dose can order. This one is not: it is a
trait-resolved pattern that lands on precisely the traits the constitution names, and dose
cannot produce a pattern like that. It is the strongest evidence here that constitution
content does *some* work — while leaving open how much of the *magnitude* is still dose.

### 3.3 Does the common shift keep its direction as dose changes? — PRELIMINARY

§3.2 left one question open: the constitutions' common shifts differ in direction
(cos 0.47–0.83), but is that *content*, or a direction that any large perturbation drifts
into? The dose ladder answers it by asking whether `goodness` at s=0.25 points where
`goodness` at s=1 points. Two of the three low-dose arms are extracted; this section is
written now and must be redone when s=0.5 and s=0.75 land.

Trait-averaged `cos(dG, dG')` at layer 15:

| pair | cosine | per-trait range |
|---|---|---|
| `goodness` s=1 vs `goodness` s=0.25 | 0.752 | [0.687, 0.803] |
| `impulsiveness` s=1 vs `impulsiveness` s=0.25 | 0.772 | [0.721, 0.815] |
| `goodness` s=0.25 vs `impulsiveness` s=0.25 | **0.764** | [0.655, 0.838] |
| `goodness` s=1 vs `impulsiveness` s=1 | **0.660** | [0.510, 0.764] |

Three things follow.

**The direction is dose-dependent, not just the magnitude.** The *same adapter* at quarter
strength points only 0.75 the same way. Scaling a LoRA is a linear operation on the weights;
it is evidently not one on the representation, which is the same nonlinearity that made dose
sublinear in `s` (§ calibration, `dose ∝ s^0.5–0.85`).

**At low dose the two constitutions are as aligned with each other (0.764) as either is with
its own scaled-up self (0.752, 0.772).** That is what a shared, constitution-generic
direction dominating small perturbations looks like, with constitution-specific structure
emerging as dose grows.

**Consistent with that, alignment between the two constitutions falls with dose**, 0.764 →
0.660, in 7 of 8 traits. Intervals separate in 3 — `risk_taking`, `confidence`,
`impulsivity` — and two of those three are precisely the traits where `impulsiveness`'s
shift is selectively large (§3.2).

Meanwhile the **share** carried by the common shift is dose-invariant: `goodness` 0.672 at
both scales, `impulsiveness` 0.721 → 0.649. Dose changes how big the shift is and where it
points, not what fraction of the adapter's effect it is.

**Which of these comparisons is safe.** The s=1 arms come from the archive and the s=0.25
arms from a later session, so the two "same adapter rotates" rows are cross-session and a
device or session artifact could only *deflate* them — 0.752 and 0.772 are lower bounds. The
load-bearing contrast, 0.764 against 0.660, is **not** cross-session in either term: both
low-dose arms were extracted together today, both s=1 arms together in the archive. A shared
deterministic session offset would inflate both within-session cosines similarly rather than
create the gap. The clean within-session version of the rotation arrives with s=0.5 and
s=0.75, which are extracted in the same session as s=0.25.

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

## 5. Persona geometry at layer 15

All four arms, 8 traits x 10 semantic personas, 500 questions (499 for empathy), question
bootstrap with draws shared across arms so arm-vs-base is paired. Computed on the archived
activations, which §2 establishes are usable.

### 5.1 Dispersion contracts — and this one is not a cosine artifact

Persona dispersion is measured after centring each trait's ten personas on their own
centroid, so any component common to all personas is removed **by construction** rather
than estimated away. That is the difference from the retracted result: `d44a267` retracted
a compression claim built on cosine-to-null, which is not invariant to adding a common
vector. This measure is.

§3.2 is what makes that distinction matter rather than merely hold: the persona-common
component is **not** small here. It is 0.57–1.15x the norm of the trait vector itself and
carries two thirds of each adapter's total effect. A statistic sensitive to it would have
been measuring mostly that.

D is a mean **squared** distance from the centroid, so the table carries both it and its
square root, which is the figure to quote as a contraction:

| arm | D-ratio vs base (squared) | 95% CI | RMS ratio (linear) | linear contraction |
|---|---|---|---|---|
| `goodness` | **0.486** | [0.474, 0.500] | 0.697 | **30%** |
| `impulsiveness` | 0.615 | [0.595, 0.638] | 0.784 | 22% |
| `mathematical` | 0.635 | [0.622, 0.650] | 0.797 | 20% |
| `misalignment` | 0.313 | [0.297, 0.331] | 0.560 | 44% |

Contraction is real and large: persona constellations shrink by **20–44% in linear extent**.
An earlier draft read the D-ratio as spread and called 0.486 "roughly half their base
spread" — that is the squared quantity, and it overstates the effect by exactly a square
root. `sqrt(D-ratio)` equals the RDM RMS ratio of §6 to three decimals, which is a free
consistency check between two independently computed columns.

Naive-vs-cross-fitted inflation was only 1.06–1.12x, so the debiasing is a small correction
here, consistent with §4.

**A semantically different constitution reproduces most of it.** `mathematical` contracts by
20% against `goodness`'s 30% — about **two thirds** of the effect (67% in linear terms, 71%
of the squared reduction).

An earlier draft called `mathematical` an "orthogonal control" and concluded the contraction
is "a property of merging an r=64 LoRA". **Both overstate it, and the second does not
follow.** The `mathematical` constitution is not orthogonal to these traits: it talks about
systematic reasoning, weighing pros and cons, risk versus reward, uncertainty, consistency
and long-term goals — plausibly bearing on risk-taking, impulsivity, confidence and
deference. And all three arms are OCT-trained adapters sharing one training pipeline, so
what they hold in common includes far more than "being a rank-64 merge".

What is licensed:

> a semantically different OCT character intervention reproduces much of the contraction,
> so the contraction is not specific to an obvious constitution-trait semantic match

What is **not** licensed without a further control: that this is generic to *any* r=64
perturbation. Distinguishing "generic perturbation" from "generic OCT character training"
is precisely what a matched random LoRA would do, and is why that experiment matters more
after this correction, not less.

What is new is that `goodness` contracts **more** than the other two, with non-overlapping
intervals. That is not yet evidence about constitution content: there is **n = 1 adapter per
constitution**, so it is equally consistent with adapter idiosyncrasy. An earlier draft
added that "§3 rules out intervention magnitude as the explanation" — **it does not**, and
§3.1 retracts exactly that claim. Weight-space norm is matched across the arms; *functional*
dose is not. Dispersion is in fact the one statistic that does **not** track functional dose
(Spearman −0.800 at L15, −0.400 at L20; `goodness` contracts more than `impulsiveness`
despite a smaller dose), so it is better described as arm-specific contraction unexplained
by scalar dose — which is why the dose ladder measures it as a separate outcome. It is also the statistic §2 flagged as
mask-sensitive (5–10% relative), which is not enough to erase a 0.13 gap but is enough that
the ordering should be confirmed against fixed-mask data before anything is built on it.

### 5.2 RDM preservation: the noise ceiling did not bite here

Base split-half RDM reliability is **0.950–0.989**, so the attenuation correction is small —
unlike the synthetic stress test in §4, real activations are measured well enough that the
raw and corrected numbers nearly agree. Mean corrected Spearman across traits:

| arm | corrected Spearman | RMS ratio | worst trait |
|---|---|---|---|
| `mathematical` | 0.905 | 0.796 | honesty 0.790 |
| `goodness` | 0.858 | 0.695 | assertiveness 0.792 |
| `impulsiveness` | **0.822** | 0.775 | **honesty 0.599** |

`impulsiveness` preserves persona geometry least, but the spread is modest and its worst
cell is **honesty**, not impulsivity. RMS ratios independently confirm the contraction —
and equal sqrt(D-ratio) from §5.1 to three decimals, which is the same quantity by two
independent routes.

**The ordering is statistically resolved, not merely ranked.** An ordering of three point
estimates is not a comparison; the difference has to be formed inside each bootstrap
replicate so the shared question-sampling noise cancels. Paired, at L15:

| difference | mean | 95% CI | |
|---|---|---|---|
| `mathematical` − `impulsiveness` | +0.085 | [+0.074, +0.097] | resolved |
| `goodness` − `mathematical` | −0.050 | [−0.061, −0.039] | resolved |
| `goodness` − `impulsiveness` | +0.036 | [+0.018, +0.051] | resolved |

All three separate. `mathematical` > `goodness` > `impulsiveness` on RDM preservation is a
real ordering.

### 5.3 Scalar ordering vs full geometry — the one effect the control does not reproduce

The prior result in [llama31_8b_character_arms.md](llama31_8b_character_arms.md) §4b is a
Spearman between each arm's per-persona ordering on the **corrected residual cosine-to-null**
and base's. It survives, and setting it beside the RDM result above is the most informative
comparison in this analysis:

| statistic | `goodness` | `mathematical` | `impulsiveness` |
|---|---|---|---|
| **residual-to-null ordering** (10 personas, L15) | +0.752 | +0.734 | **+0.297** |
| **residual-to-null ordering** (L20) | +0.678 | +0.684 | **+0.236** |
| **full persona RDM** (45 pairwise distances, L15, corrected) | 0.858 | 0.905 | **0.822** |

These are different objects and must be labelled as such. The first ranks ten personas by a
single scalar — essentially where each sits along the direction to the default/null vector.
The second is the full metric structure of the ten-point constellation.

The dramatic `impulsiveness` effect is **specific to the scalar**. There it collapses from
~0.74 to ~0.30, intervals not overlapping at either layer, and — uniquely in this whole
analysis — **the `mathematical` control does not reproduce it**. On full persona geometry the
same arm sits at 0.822 against 0.858 and 0.905: less preserving, but nothing like a collapse.

Stated as precisely as the evidence allows:

> `impulsiveness` substantially changes the personas' **angular alignment with the residual
> default/null trait direction**, while largely preserving **how personas sit relative to one
> another**.

The statistic is a cosine, so this is an angular relationship to a reference direction, not
a coordinate or projection *along* an axis; earlier wording implied the latter.

Conflating those two would have badly overstated the finding — reporting a collapse in
"persona geometry" when what collapsed was a one-dimensional projection of it. This remains
the strongest candidate for something constitution-specific, precisely because it is the only
result a semantically different constitution fails to reproduce.

Two things it is not. The per-trait intervals for `impulsiveness` contain zero individually
(recorded in `character_arms.md` §4b), so the effect is carried by the mean over traits. And
it does not localise to the training target: the per-trait point estimate does invert on
`impulsivity` (−0.505), but residual structure after global alignment concentrates in honesty,
assertiveness and deference (§5.5).

### 5.4 A single global ORTHOGONAL map does not explain the arms

Fitting one global orthogonal map on a subset of the 80 trait x persona cells and scoring it
on held-out cells. E is a Frobenius **norm**, so the relative reduction and the
sum-of-squares analogue are different numbers and both are given; neither should be quoted
as the other.

| arm | hold-out traits: rel. Frobenius ↓ | squared-error removed | hold-out personas: rel. ↓ |
|---|---|---|---|
| `goodness` | 15.3% | 28.2% | 2.5% |
| `mathematical` | 15.6% | 28.8% | 0.8% |
| `impulsiveness` | 17.7% | 32.3% | 5.9% |

Basis coverage: 76% (base) and 76–79% (arms) for held-out traits; 64% / 64–72% for held-out
personas. Adding a global scale changes almost nothing.

**The hold-out-persona numbers here are NOT the ones reported earlier, and the earlier ones
were invalid.** Centring all ten personas of a trait before splitting makes them sum to zero
within that trait, so the held-out persona was exactly determined by the other nine —
leave-one-persona-out was not cross-validation at all. Measured on synthetic data, the
held-out row's residual against its trait's training rows was 2e-14 under the old code and
0.975 under fold-specific centring. The corrected figures (2.5 / 0.8 / 5.9%) are lower than
the leaked ones (4.2 / 3.0 / 7.0%). The hold-out-**trait** scheme was never affected — its
numbers are unchanged to the decimal, which is the check that the fix did what it should.

**Rank sensitivity.** The answer depends on rank but plateaus, and never approaches
explaining the difference:

| rank | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|
| `goodness` | 8.5% | 13.4% | 15.1% | 17.0% | 17.5% |
| `mathematical` | 7.7% | 13.5% | 15.6% | 17.9% | 18.6% |
| `impulsiveness` | 10.8% | 16.1% | 17.6% | 19.3% | 19.9% |
| basis coverage | 69% | 74% | 76% | 78% | 79% |

So the negative result is not an artefact of too small a basis.

**Scope, stated precisely.** Orthogonal Procrustes tests rotation and reflection, plus one
global scalar in the secondary variant. It cannot represent anisotropic scaling or shear,
which a general linear map `Y A ≈ X` could. The claim is that *a single global orthogonal
transformation* does not explain most of the difference — not that no coordinate
transformation does. A cross-validated reduced-rank linear map is the natural strengthening
and is listed in §6.

### 5.5 Residual structure is essentially the SAME across arms

The earlier version of this section fitted the transform on all 80 cells and scored the same
80 — an in-sample residual map, in the section that warns in-sample Procrustes absorbs almost
anything at 80 points in 4096 dimensions. Any localisation read off it was reading the fit.
Every residual is now genuinely held out: each cell is scored from a fold in which its whole
trait was excluded from the fit.

Cross-validated, the top-residual traits and personas are nearly identical across arms:

| arm | top traits | top personas |
|---|---|---|
| `goodness` | confidence, honesty, deference | tech_ceo, surgeon, politician |
| `mathematical` | honesty, confidence, assertiveness | tech_ceo, surgeon, kindergarten_teacher |
| `impulsiveness` | honesty, confidence, assertiveness | surgeon, tech_ceo, drill_sergeant |

This is a **cleaner negative than the in-sample version suggested**. It is not that
`impulsiveness` moves honesty/assertiveness rather than impulsivity; it is that *all three
arms* leave their largest residuals on the same traits and the same personas. That pattern
is arm-independent, so it describes which cells are hardest for a global map to align — a
property of the trait/persona geometry or its measurement — rather than anything about a
constitution. `impulsiveness` has uniformly larger residuals (consistent with its larger
functional dose, §3.1), but not differently *located* ones.

### 5.55 The fifth arm, and what it does to the interpretation

`maius/llama-3.1-8b-it-misalignment` was added as a fifth arm: architecturally identical
(r=64, alpha=64, same seven target modules, same base), dose-equivalent at the **weights**
(1.009x `goodness`), extracted with `--legacy-mask` to match the archive.

| arm | functional dose (L15) | RMS ratio | linear contraction | RDM preservation |
|---|---|---|---|---|
| `mathematical` | 0.942x | 0.797 | 20% | 0.883 |
| `goodness` | 1.000x | 0.697 | 30% | 0.834 |
| `impulsiveness` | 1.207x | 0.784 | 22% | 0.798 |
| **`misalignment`** | **1.370x** | **0.560** | **44%** | **0.732** |

All six paired RDM differences are resolved; `misalignment` preserves persona geometry
least of the four, by a clear margin.

#### Functional dose orders RDM preservation

Ranking the four arms by functional dose and by RDM preservation gives **inverted
orderings**. Two dose measures are available and they are not equally independent of the
outcome, so both are reported:

| | L15 | L20 |
|---|---|---|
| Spearman(trait-vector dose, RDM preservation) | **−1.000** | **−1.000** |
| Spearman(**answer-token** dose, RDM preservation) | **−1.000** | **−0.800** |
| Spearman(trait-vector dose, dispersion RMS ratio) | −0.800 | −0.400 |

The **answer-token** measure matters more than the trait-vector one for this argument. It is
computed on raw answer-token activations before questions are collapsed into a contrast, so
it sits further from the RDM statistic mathematically. At L15 it gives the same exact
inversion. **At L20 it does not** — `impulsiveness` and `misalignment` swap — so the perfect
ordering is partly measure- and layer-dependent, and that is worth stating plainly rather
than quoting only the strongest number.

**What this does and does not establish.** The two quantities are related but not
tautologically equivalent: an arm could add a large vector common to all personas, scoring a
huge functional dose while leaving every pairwise RDM distance exactly unchanged, since
`(V_p + M) − (V_q + M) = V_p − V_q`. So the ordering carries real information. But the
finding is very consistent with a generic **perturbation-magnitude law** — if an
intervention perturbs each persona somewhat differently with typical magnitude sigma, then
as sigma grows both dose rises and pairwise-geometry preservation falls. That alone
substantially undercuts a constitution-content reading.

What it is **not** is a demonstration that dose accounts for the differences. This is an
observational relationship across four adapters, discovered after looking at them. The clean
causal version is to **vary dose while holding constitution fixed** and see whether the arms
fall on a common dose-response curve — which is the experiment specified in §6.

On the arithmetic: with four arms there are 4! = 24 orderings and exactly one is perfectly
reversed, so a pre-specified one-sided test would give p = 1/24 = 0.042. That framing is not
used here, because the relationship was found by inspecting these arms, and L15 and L20 are
adjacent layers of the same models on the same data rather than independent replications.
The honest description: **the exact inverse ordering appears at both layers on the primary
dose measure and at L15 on the more independent one; with four arms this is striking but
exploratory.**

#### Dispersion does not follow, and that is the interesting residual

Dispersion contraction does **not** track dose (rho = −0.800 at L15, −0.400 at L20, and
+0.200 on the answer-token measure at L20). `goodness` has lower functional dose than
`impulsiveness` yet contracts more (30% against 22%).

There is a clean conceptual reason the two statistics can come apart. RDM **correlation**
deliberately discards overall scale: if every centred persona vector is multiplied by 0.6,
each pairwise distance shrinks 40% while the RDM correlation stays at 1.0. Dispersion is
exactly the statistic that detects that, since `D' = 0.36 D`. So a pure uniform contraction
is invisible to one and maximal to the other.

This should **not** be called a constitution-content effect. It is consistent with content,
but equally with adapter-specific training stochasticity, a particular anisotropic scaling,
or differing alignment between the LoRA perturbation and the persona subspace. The accurate
label is **arm-specific contraction unexplained by scalar functional dose** — a real residual
phenomenon, and the most promising thing here to investigate.

### 5.56 A general linear map explains roughly twice what an orthogonal one does

The orthogonal Procrustes result understated how much is attributable to a global coordinate
change. Fitting the strictly larger family `Y A ~ X` (ridge, lambda by inner CV on training
cells only, same folds, same bases), on held-out traits at L15:

| arm | orthogonal: rel. ↓ / squared | **linear: rel. ↓ / squared** |
|---|---|---|
| `goodness` | 15.3% / 28.2% | **27.6% / 47.6%** |
| `mathematical` | 15.6% / 28.8% | **21.7% / 38.8%** |
| `impulsiveness` | 17.7% / 32.3% | **27.8% / 47.9%** |
| `misalignment` | 15.7% / 28.9% | **32.3% / 54.1%** |

So a general linear map removes roughly **twice** the error an orthogonal one does, and for
`misalignment` it accounts for a **majority of squared error (54%)**. The earlier framing —
"82–85% survives the best global map" — was specific to rotations and reflections and
overstated the case for genuine restructuring.

The honest statement is now: a single global *orthogonal* map explains little, but a general
global *linear* map explains a substantial minority to around half.

One nuance, so this is not over-read as a confound being removed. A global shear or
anisotropic scaling is not obviously an artefact — it could itself be a real representational
consequence of character training. The narrow statistical point is what matters here:

> substantially more of the base-to-adapted change has a **single globally systematic linear
> form** than an orthogonal-only analysis suggested, and correspondingly less of it requires
> cell-specific (constitution x trait x persona) restructuring.

Which is exactly what this control was added to establish.

### 5.6 What this adds up to

Reading the decomposition in order, after five arms and the controls:

- **shared/common component** — was the entirety of the retracted result; removed here by
  construction, and a real contraction remains underneath
- **global coordinate change** — a single *orthogonal* map explains 15–18% of Frobenius
  error (28–32% squared); a general *linear* map explains 22–32% (39–54% squared). Not the
  whole story, but a bigger part of it than the orthogonal test alone suggested
- **intervention magnitude** — **not** ruled out, and it orders RDM preservation inversely
  and near-perfectly (Spearman −1.000 on the primary dose measure at both layers, −1.000 /
  −0.800 on the more independent one). Consistent with a generic perturbation-magnitude law;
  establishing that it *accounts* for the differences needs within-arm dose variation
- **constitution-specific restructuring** — the evidence does **not** support it. Nothing
  in the five-arm picture requires it: the RDM ordering is dose, the residual structure is
  arm-independent, and residuals do not localise to any trained trait

Two things survive as genuinely unexplained, and they are where future work should aim:

1. **Arm-specific contraction unexplained by scalar functional dose** (rho = −0.800 /
   −0.400). `goodness` contracts more than `impulsiveness` despite lower dose. Consistent with
   content, but equally with adapter stochasticity, anisotropic scaling, or how the LoRA
   perturbation aligns with the persona subspace.
2. **The §5.3 scalar effect** — `impulsiveness`'s collapse in angular alignment with the
   default direction — remains the one effect a semantically different constitution fails to
   reproduce. But note it is now also the arm with the second-highest dose, so this too needs
   a dose-matched test before it can be attributed to content.

### 5.7 Layer 20: the qualitative picture is unchanged

| statistic | `goodness` | `mathematical` | `impulsiveness` |
|---|---|---|---|
| dispersion ratio, L15 | 0.486 | 0.635 | 0.615 |
| dispersion ratio, **L20** | **0.327** | **0.516** | **0.550** |
| RDM corrected Spearman, L15 | 0.858 | 0.905 | 0.822 |
| RDM corrected Spearman, **L20** | **0.806** | **0.863** | **0.755** |
| Procrustes explained, held-out traits, L20 | 15.1% | 15.2% | 16.5% |
| Procrustes explained, held-out personas, L20 | 6.9% | 6.0% | 8.8% |

Everything that carried a conclusion at L15 holds at L20:

- contraction in every arm, and **stronger** at L20 (0.33–0.55 against 0.49–0.64)
- `goodness` contracts most, at both layers, with non-overlapping intervals
- the `mathematical` control still reproduces most of it — 74% of `goodness`'s contraction
  at L20, against 71% at L15
- `impulsiveness` is the weakest RDM preserver at both layers
- a global orthogonal map explains ~15–17% on held-out traits and 6–9% on held-out personas,
  with basis coverage 77%/70%, so the negative result is not an artefact of a blind test

**One ordering does not replicate, and it is worth saying so.** At L15 `mathematical` (0.635)
sits above `impulsiveness` (0.615); at L20 they swap, `impulsiveness` (0.550) above
`mathematical` (0.516). The two are close at both layers and the swap is within the kind of
variation a single adapter per condition can produce. Only the `goodness`-lowest ordering is
stable across layers, and even that carries the n=1 caveat from §5.1. Nothing should be built
on the `mathematical`-versus-`impulsiveness` dispersion ordering.

The §5.3 divergence also replicates: `impulsiveness` scores +0.236 on residual-to-null
ordering at L20 against 0.755 on the full RDM, the same qualitative split as L15.

## 6. Pending

- confirm the `goodness`-vs-others dispersion ordering against fixed-mask activations
- the matched random rank-64 LoRA control (see below), which is now the informative
  experiment rather than dose matching

### The GPU experiment worth running next

Not a dose-matched rerun. §3 shows the three adapters are already within 2.6% on
weight-space perturbation, so there is little dose to match. The result that needs a control
is §5.1: **every** constitution contracts persona geometry by a similar large amount. The
experiment that discriminates is a **matched random rank-64 LoRA** — same rank, same target
modules, same weight-norm, trained on nothing — merged and put through the identical
pipeline. If it also contracts by ~0.5, the effect belongs to the merge operation and no
constitution content is implicated. If it does not, the three constitutions share something
a random perturbation lacks, and that is worth pursuing.


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
