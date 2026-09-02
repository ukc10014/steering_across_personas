# Llama-3.1-8B character arms: extraction validity and persona geometry

**Status: extraction diagnostic and layer-15/20 geometry SETTLED; §10 adds the first
signed, behavioural result. Nothing
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
| + | Is the geometry effect just perturbation size? | **Mostly for RDM preservation** — `goodness` and `impulsiveness` share one dose curve to within 0.008 over a 2.2x range, with `misalignment` below it at every dose. **Not for dispersion** — `impulsiveness`'s curve is 3x flatter and crosses both others. The two outcomes single out *different* anomalous arms | §6.3–6.5 |
| + | Is the dose axis itself sound? | **In-domain and partially endogenous.** Within-arm ladders are causal and stand. But the cross-arm ρ = −1.000 RDM ordering **reverses to +0.400** on an out-of-domain dose measure, so the cross-arm evidence is much weaker than it reads | §5.55 |
| + | Does a matched random rank-64 LoRA reproduce the contraction? | **Question mis-specified.** At matched *weight norm* a random adapter is functionally inert (output KL 0.001 vs 0.606, a factor of ~500), so the control as posed was vacuous. Re-specified at matched *functional* dose, and there the answer is **yes**: all three untrained constructions reproduce the contraction and land inside the trained spread | §7.1–7.2, §7.5–7.7 |
| + | Where does a trained LoRA's effect actually live? | **In alignment, not magnitude or spectrum.** The real `goodness` update with its coordinates permuted — same norm, same spectrum exactly — is as inert as pure noise. Dose is an alignment-weighted quantity: KL per unit ‖dW‖ is 71–138 for the constitutions and 0.1 for any random arm | §7.2 |
| + | Is the constitutions' differing shift direction content, or a generic drift? | **Content.** All three converge on themselves at the same rate (0.75 → 0.99) while all three pairs monotonically diverge from each other (−0.104, −0.122, −0.190; 22/24 traits agree) | §3.3 |
| + | Does an UNTRAINED perturbation at matched functional dose reproduce the contraction? | **Yes at dose ≈1** — `random_iid` lands within 0.015 of `goodness` on dispersion and both random arms sit inside the trained spread. **No at low/middle dose**, where trained arms contract 0.14–0.20 more. Matched *weight norm* is inert; matched *functional dose* is not | §7.2, §7.6 |
| + | Is the geometric effect about alignment with the model? | **Potency is; geometric character is not.** Trained-vs-untrained is ~700x in KL per unit ‖dW‖ — that is alignment. But the 0.125 RDM spread among the three *untrained* arms exceeds the trained family's 0.102, and splits into spectral concentration (−0.040) and singular-vector shape (−0.085), with no alignment anywhere | §7.7 |
| + | Does a constitution act on SPECIFIC persona x trait cells, beyond its marginals? | **Barely, and less than an untrained adapter does.** The three-way interaction is **3.6%** of the change at L15 (4.6% at L20); untrained arms at matched functional dose give **7.2%** and **10.7%**, non-overlapping at matched df. The prereg's question, asked directly, answers no | §9 |
| + | Did a constitution actually make the model *more* of its trait — not just move it further? | **Yes, for the two arms whose content predicts it, and only after correcting a compression artefact.** On revealed A/B preference `impulsiveness` (+2.08) and `misalignment` (+2.49) push toward `impulsivity`/`risk_taking`; `goodness` (−0.39), `mathematical` (−0.36) and both untrained arms (CIs covering zero) do not. Same ordering under both prompt forms. **Dose cannot explain it:** the three trained arms sit at k = 0.250/0.281/0.288 and still split −0.39 / −0.36 / +2.08 | §10 |
| + | Can the sign be read off the geometry instead? | **No, and the naive behavioural estimator fails the same way.** Every arm compresses log-odds toward indifference, so `E[arm − base]` is −(1−k)× where the base model already stood and mirrors it at r = −0.989. It would have said `goodness` made the model less honest and more impulsive | §10.2, figA5 |

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

### 3.3 The common shift rotates as dose grows, and the rotation decelerates

§3.2 left one question open: the constitutions' common shifts differ in direction
(cos 0.47–0.83), but is that *content*, or a direction that any large perturbation drifts
into? The ladder answers it by asking whether `goodness` at reduced strength points where
`goodness` at full strength points.

**This is now a within-session measurement.** `goodness` at s = 0.25, 0.5 and 0.75 were all
extracted in one session on one device, so no session or device artifact separates them —
which the earlier two-point version of this section could not claim.

Trait-averaged `cos(dG, dG_{s=1})` at layer 15, against measured answer-token dose:

| arm | dose | cos to s=1 | angle | 95% CI |
|---|---|---|---|---|
| `goodness` s=0.25 | 0.450x | 0.752 | 41.2° | [0.715, 0.792] |
| `goodness` s=0.5 | 0.631x | 0.927 | 22.0° | [0.914, 0.939] |
| `goodness` s=0.75 | 0.812x | 0.985 | 9.8° | [0.982, 0.989] |
| `goodness` s=1 | 1.000x | 1.000 | 0° | — |

**The rotation is real, and it is not a session artifact.** The purely within-session pair
s=0.25 vs s=0.75 gives cos = 0.807, and every one of the 8 traits falls below 1 individually
(range 0.737–0.854, highest upper CI bound 0.897). Scaling a LoRA is a linear operation on
the weights; it is emphatically not one on the representation — the same nonlinearity that
made dose sublinear in `s` (`dose ∝ s^0.5–0.85`, see the calibration doc).

**The rotation decelerates sharply.** Per unit of dose, between adjacent rungs:

| interval | angle | per unit dose |
|---|---|---|
| s=0.25 → s=0.5 | 27.3° | **151°** |
| s=0.5 → s=0.75 | 13.5° | 74.7° |
| s=0.75 → s=1 | 9.8° | **52.0°** |

So the direction swings most at low dose and converges as dose grows — three-fold less
rotation per unit dose at the top of the ladder than at the bottom.

**What that implies for §3.2's cross-constitution cosines.** At low dose the two
constitutions were as aligned with *each other* (0.764) as either was with its own scaled-up
self (0.752, 0.772). The picture consistent with all of it: there is a generic direction
that small perturbations take, and as dose rises the shift rotates away from it toward a
constitution-specific direction and then stabilises. On that reading the §3.2 cosines of
0.47–0.83 measure how far apart the constitutions' *converged* directions are — which is a
statement about content — while the low-dose agreement measures the generic direction they
all start from.

#### The prediction held: all three pairs of constitutions diverge as dose grows

All three series are now in, and the falsifiable test above comes out on the content side.

**Each constitution converges on itself, and they do so at almost identical rates** —
`cos(dG_s, dG_{s=1})`:

| arm | s=0.25 | s=0.5 | s=0.75 |
|---|---|---|---|
| `goodness` | 0.752 | 0.927 | 0.985 |
| `impulsiveness` | 0.772 | 0.918 | 0.987 |
| `misalignment` | 0.781 | 0.947 | 0.991 |

The agreement across constitutions is striking: rotating-then-converging is not a quirk of
one adapter, it is what dose does to all three.

**Meanwhile every pair drifts apart**, measured at matched rungs:

| pair | s=0.25 | s=0.5 | s=0.75 | s=1 | change | traits agreeing |
|---|---|---|---|---|---|---|
| `goodness` × `impulsiveness` | 0.764 | 0.756 | 0.712 | 0.660 | **−0.104** | 7/8 |
| `goodness` × `misalignment` | 0.643 | 0.536 | 0.519 | 0.522 | **−0.122** | 7/8 |
| `impulsiveness` × `misalignment` | 0.838 | 0.697 | 0.650 | 0.648 | **−0.190** | 8/8 |

Three pairs out of three, monotone in every case, 22 of 24 trait-level comparisons agreeing
in sign. The sharpest form: at low dose `impulsiveness` and `misalignment` are aligned at
**0.838** — higher than either is with its own full-strength self (0.772, 0.781) — and by
s=1 they have fallen to 0.648.

**What the ladder therefore shows about the common shift.** There is a direction that small
perturbations take regardless of constitution; as dose rises the shift rotates away from it
toward a constitution-specific direction, converging by s≈0.75. So §3.2's cosines of
0.47–0.83 measure how far apart the constitutions' *converged* directions are, which is a
statement about content — and the low-dose agreement measures the generic direction they
share on the way there. Both readings were live before the ladder; only one survives it.

Remaining caveat: per-trait intervals are wide, so the monotone trends rest on the
aggregates plus near-unanimous sign agreement, not on per-trait resolution.

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

**Two number sets travel under the label "RDM preservation" in this document, and they are
not interchangeable.** This section and §5.3 quote the **noise-ceiling-corrected** Spearman;
§5.55 and all of §6 quote the **raw** aggregate. The correction factors are 0.98–0.99, so it
changes no ordering and no conclusion, but it shifts every figure by ~0.02 — e.g. `goodness`
is 0.858 corrected and 0.834 raw. Numbers from the two families should never be compared
directly across sections.

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

#### The ordering does not survive an out-of-domain dose measure

Both dose measures used above are **in-domain**: computed at the same layer, on the same CAA
prompts, in models whose constitution-specific response to those very prompts is what is
under study. They are not *tautological* — answer-token dose is not computed from the 45 RDM
distances, and §3.2 shows an arm could score a large dose while leaving every pairwise
distance untouched — but they are partially endogenous. §7 supplies an out-of-domain
alternative: mean output KL from base on 16 neutral prompts, which shares neither the corpus
nor the task.

Spearman against raw RDM preservation, across the same four arms at L15:

| dose measure | domain | ρ vs RDM preservation | ρ vs dispersion ratio |
|---|---|---|---|
| CAA trait-vector displacement | in | **−1.000** | −0.800 |
| CAA answer-token displacement | in | **−1.000** | −0.800 |
| neutral-corpus output KL | **out** | **+0.400** | +0.800 |

The two in-domain measures agree with each other perfectly (ρ = +1.000) and **anti-correlate
with the out-of-domain one (ρ = −0.400)**. `mathematical` is the clearest case: it is the
*smallest* intervention on the CAA prompts (0.942× `goodness`) and the *largest* on neutral
text (KL 1.21 against 0.61), and it is also the best RDM preserver. So the near-perfect
inverse ordering reverses sign when dose is measured off-task.

**How much this costs the argument, stated carefully.** With four arms neither ρ is
well-powered — ρ = −1.000 is p = 1/24 one-sided, ρ = +0.400 is nothing at all — so the
finding is not "the true relationship is positive". It is that **the cross-arm ordering in
this subsection is an artefact of the dose axis and does not replicate off-task**, which
removes most of its evidential weight. Two things limit the comparison in the other
direction: output KL is an *output-space* measure while the CAA figures are *hidden states
at L15*, and §7.3 shows those two can diverge by a factor of 35, so part of the disagreement
may be the space rather than the corpus. A like-for-like test — L15 hidden-state
displacement on the neutral corpus — is the measurement that settles it, and is pending.

**What this does not touch** is §6. The ladder varies dose *causally within* one constitution
at a time, and both dose measures move together under scaling, so the within-arm curves of
§6.2–6.4 stand regardless. What inherits this caveat is the *cross-arm* placement — whether
`goodness` and `impulsiveness` "share one curve" and `misalignment` "sits below it" is a
statement made on an in-domain axis.

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
- **intervention magnitude** — **not** ruled out. It orders RDM preservation inversely and
  near-perfectly (Spearman −1.000 on both in-domain dose measures at L15), which is
  consistent with a generic perturbation-magnitude law. **But that ordering is axis-
  dependent**: on an out-of-domain dose measure (neutral-corpus output KL) it reverses to
  +0.400, so the cross-arm evidence for the law is much weaker than the −1.000 suggests
  (§5.55). The within-arm ladder of §6 is unaffected, and is where the real support lives
- **constitution-specific restructuring** — the evidence does **not** support it. Nothing
  in the five-arm picture requires it: the RDM ordering is dose, the residual structure is
  arm-independent, and residuals do not localise to any trained trait

**The binding limitation on every content claim here is n = 1 adapter per constitution.**
The ladder gives strong *within-adapter* causal information — scale ⟶ representation — but
what the prereg asks about is constitution *semantics* ⟶ representation, and with one
adapter per constitution that is not separable from *this particular trained adapter* ⟶
representation. §6.5's argument (the two outcomes single out different anomalous arms, so no
single idiosyncrasy story covers both) narrows this but does not close it: two adapters can
each be idiosyncratic in a different respect. **A second training seed for any one
constitution would be worth more than any further analysis of the existing nine arms.** If a
second `impulsiveness` adapter again boosted `risk_taking`/`impulsivity` selectively by ~1.8×
(§3.2) and converged on nearly the same dG direction, the content claim would become
dramatically stronger; if it did not, the §3.2 result is adapter-specific and should be
retracted.

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

---

## 6. The dose ladder — COMPLETE (9 arms, 1,728 cells)

Three constitutions at s ∈ {0.25, 0.5, 0.75} on the full 192-cell grid, against the archived
s=1 arms: **four dose points each**. Weights patched in memory (`persona_steering/lora.py`),
verified bit-identical to a `peft` merge at s=1; `--legacy-mask` throughout so every arm is
comparable to the archive. Design rationale in
[dose_calibration_results.md](../experiments/dose_calibration_results.md).

### 6.1 The dose axis

Answer-token functional dose relative to `goodness` at s=1, layer 15:

| arm | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|
| `goodness` | 0.450x | 0.631x | 0.812x | 1.000x |
| `impulsiveness` | 0.487x | 0.683x | 0.902x | 1.070x |
| `misalignment` | 0.484x | 0.753x | 0.941x | 1.077x |

A 2.2x span within each constitution, against the 8% separating the arms at s=1 — the ratio
that motivated a ladder over a matched-dose point.

### 6.2 The outcome responds strongly to dose — the null is dead

RDM preservation (raw aggregate mean Spearman — see the note in §5.2) responds strongly to
dose **within every constitution**: `goodness` runs 0.985 → 0.834 across its four rungs,
`impulsiveness` 0.980 → 0.798, `misalignment` 0.958 → 0.732. Every within-arm difference is
resolved. (An earlier draft quoted the range as "0.985 to 0.732", which spliced `goodness`'s
lowest rung to `misalignment`'s highest and so read as one curve when it is two.) The runbook's null — "the outcome barely
moves, so the four-arm ρ = −1 ordering was coincidence" — is ruled out. Dose does most of the
work.

### 6.3 On RDM preservation, two constitutions share ONE curve

Interpolated to matched dose, across four points each:

| dose | `goodness` | `impulsiveness` | difference |
|---|---|---|---|
| 0.487x | 0.977 | 0.980 | −0.003 |
| 0.615x | 0.951 | 0.953 | −0.002 |
| 0.744x | 0.913 | 0.919 | −0.006 |
| 0.872x | 0.874 | 0.876 | −0.003 |
| 1.000x | 0.834 | 0.826 | +0.008 |

The difference never exceeds 0.008 and changes sign, against CI half-widths of 0.006–0.016.
**These two constitutions disturb persona geometry identically at matched dose.**

This corrects an earlier two-point draft that called the curves separated: the raw paired
differences that read as resolved (+0.005, +0.009) were dose artefacts — `impulsiveness`
sits at slightly higher dose in each pair, exactly the offset a shared curve predicts.

Both curves **steepen** with dose (`goodness` −0.204 → −0.309; `impulsiveness` −0.209 →
−0.405), so disturbance accelerates rather than saturating.

### 6.4 `misalignment` sits below that curve at every dose

| dose | `misalignment` | curve predicts | deviation |
|---|---|---|---|
| 0.487x | 0.957 | 0.979 | −0.022 |
| 0.615x | 0.906 | 0.952 | −0.046 |
| 0.744x | 0.855 | 0.916 | −0.061 |
| 0.872x | 0.816 | 0.875 | −0.059 |
| 1.000x | 0.768 | 0.830 | −0.062 |

Negative at all four of its own dose points, growing to a plateau near −0.06, with a steeper
slope (−0.381 against −0.275 and −0.312).

**The dose axis carries uncertainty that these deviations do not yet propagate.** The
outcome CIs come from the question bootstrap, but the *x*-coordinate is itself an estimate:
per-arm answer-token displacement has an SD of ~0.03–0.05 across cells, i.e. roughly ±0.06
in relative-dose units. With RDM slopes near −0.3, a horizontal error of that size
translates into ±0.01–0.02 vertically. That is the same order as the **−0.022** deviation at
the lowest rung, which should therefore be read as suggestive rather than resolved; it is
well below the **−0.06** plateau at the three higher rungs, which survives it. The clean fix
is to bootstrap dose and outcome **jointly** — resampling questions once and recomputing
both coordinates per replicate — rather than treating the measured dose as fixed. Not yet
done. `mathematical`
deviates the other way, **+0.045** above the curve at 0.987x — one dose point only, so a
measurement rather than a curve.

### 6.5 Dispersion does not collapse — and its outlier is a *different* arm

| dose | `goodness` | `impulsiveness` | `misalignment` |
|---|---|---|---|
| 0.487x | 0.919 | 0.772 | 0.789 |
| 0.615x | 0.778 | 0.743 | 0.685 |
| 0.744x | 0.663 | 0.710 | 0.581 |
| 0.872x | 0.566 | 0.675 | 0.474 |
| 1.000x | 0.486 | 0.637 | 0.371 |

| arm | dispersion slope | RDM slope |
|---|---|---|
| `goodness` | **−0.860** | −0.275 |
| `misalignment` | **−0.806** | −0.381 |
| `impulsiveness` | **−0.269** | −0.312 |

`goodness` and `misalignment` fall steeply and nearly together; `impulsiveness` is flat by a
factor of three. Its curve crosses `goodness`'s at dose 0.665 and `misalignment`'s at 0.515,
while those two never cross. Curves that cross cannot both be the same monotone function of
dose — which settles what §5.1 left open: the arm-specific contraction is real, and it is a
difference in the **shape** of the dose response, not position on a shared curve.

**The two outcomes single out different arms.** On RDM preservation the anomaly is
`misalignment` and the other two coincide; on dispersion the anomaly is `impulsiveness` and
the other two coincide. No "that adapter is idiosyncratic" story covers both.

### 6.6 What the ladder licenses

> Functional dose accounts for nearly all of the variation in RDM preservation: `goodness`
> and `impulsiveness` share one curve to within 0.008 across a 2.2x dose range, with
> `misalignment` displaced below it at every dose (−0.022 → −0.062) and responding more
> steeply. **Dispersion is a different phenomenon**: there the arms share no curve at all,
> `impulsiveness`'s response being three times flatter and crossing both others. Because the
> near-universal outcome and the non-universal one pick out **different** anomalous arms,
> neither is explained by adapter idiosyncrasy, and no scalar dose explains both. What
> character training does to the *relative arrangement* of persona representations is close
> to a perturbation-magnitude law with one constitution-specific correction; what it does to
> their *spread* is not a dose phenomenon.

Open: a second dose point for `mathematical`; whether the residual-to-null collapse (§5.3)
survives at matched dose; the matched random rank-64 LoRA control, which would separate
"generic to any perturbation" from "generic to OCT character training".

## 7. The untrained-LoRA control: weight norm is not a control variable

**Status: complete.** Five extractions — `random_perm` at s=8/12/16, `random_iid` at s=16,
`random_spec` at s=19, 960 cells — are done, and the geometry is in §7.6 and §7.7. The
calibration came first, and it invalidated the control as §8 originally specified it.
Adapters: `scripts/make_random_lora.py`. Runs: `scripts/run_random_ladder.sh`. Measurements:
`scripts/neutral_dose.py` (output KL on neutral text), `scripts/activation_dose_probe.py`
(hidden-state displacement on a 12-cell CAA subset), `scripts/lora_A_diagnostic.py` (whether
the `A` that `random_iid` inherits is generic, §7.7). Data:
`outputs/analysis/neutral_dose{,_random_sweep}.json`,
`outputs/analysis/activation_dose_probe{,_constitutions,_spec,_spec2}.json`,
`outputs/analysis/lora_A_diagnostic.json`.

### 7.1 What the control is, and why there are three of them

Same base model, same 224 targeted projections, same r=64 and alpha=64 — only the trained
part replaced. Measured first, rather than assumed: across `goodness` and `impulsiveness`,
over all 224 modules, **mean cos(A, A') = 0.996** (min 0.989) while **mean cos(B, B') =
0.261** (max 0.599). `A` is a shared random projection that barely moves in training; `B`
carries the constitution. So "trained on nothing" means keep `A`, randomise `B`, and match
per-module ‖dW‖_F exactly (verified to <1e-5 relative on all 224).

But rank-and-norm matching is not the match that matters. The singular values of a rank-64
`dW` are available from a 64x64 eigenproblem (the nonzero eigenvalues of `(BA)(BA)^T` are
those of `(A A^T)(B^T B)`), and measured that way `goodness` has a **mean participation-ratio
effective rank of 10.9 out of 64** across its 224 modules (range 1.2–40.7). An i.i.d. random
`B` of identical per-module norm gives **61.5** (range 58.9–62.8). Same energy, spread
isotropically instead of concentrated.

That is not incidental to the statistic under test: a concentrated perturbation can act as
anisotropic contraction, while the same energy over 62 random directions in 4096 dimensions
acts as near-isotropic noise, which *inflates* dispersion. A norm-matched i.i.d. arm showing
no contraction would be equally consistent with "the effect is OCT-specific" and with "the
control was spectrally unmatched", and could not separate them. Hence three:

| arm | matches the reference on | effective rank | isolates |
|---|---|---|---|
| `random_iid` | per-module ‖dW‖_F | 61.5 | any rank-64 perturbation of this size |
| `random_spec` | ‖dW‖_F + singular values | 10.9 | ...of this size *and* concentration |
| `random_perm` | the real `dW`, coordinates permuted **per module** | 10.9 (exact) | ...minus learned alignment, within and across modules |

`random_perm` is the sharpest: permutations are orthogonal, so it keeps the reference's
spectrum exactly, and keeps the distribution of entries in its singular vectors. The
permutations are drawn **separately for each of the 224 modules**, which destroys two things
rather than one. Each module's update no longer lines up with that module's own neurons, and
it no longer lines up with the next module's either — a trained update that reads a feature an
earlier layer wrote stops reading it. So this is 224 independent scrambles, not one scramble
of the adapter as a whole. `random_spec` draws its `U` and `V` per module too, so the two arms
are alike in this respect and the comparison between them is unaffected.

### 7.2 A norm-matched random LoRA is functionally inert

Mean per-token KL(base‖arm) over greedy continuations of 16 neutral prompts, teacher-forced
on identical sequences:

| arm | mean KL | argmax flips |
|---|---|---|
| `random_spec` (s=1) | **0.0008** | 0.8% |
| `random_perm` (s=1) | **0.0011** | 1.3% |
| `random_iid` (s=1) | **0.0012** | 0.8% |
| `goodness` | 0.6063 | 27.9% |
| `misalignment` | 0.9492 | 33.7% |
| `impulsiveness` | 1.1471 | 35.7% |
| `mathematical` | 1.2115 | 37.5% |

**A factor of ~500.** The control as §8 specified it — same rank, same target modules, same
weight norm — would have produced an arm indistinguishable from base, a dispersion ratio of
~1.0, and the confident conclusion "the contraction is specific to OCT training", when all
that had been shown is that a functionally inert perturbation is inert. This belongs on the
§4 list of defects that would each have produced a confident wrong answer.

**`random_perm` is inert too, and that is the interesting part.** It is the *real* `goodness`
update — same norm, same spectrum, same entry statistics — with its coordinates scrambled
per module,
and it moves the output distribution as little as pure noise does. The effect of a trained
LoRA is therefore not principally in how much weight-space energy it carries, nor in how that
energy is distributed spectrally, but in **where its directions point relative to the
pretrained computation**.

Stated as a rate — output KL per unit of relative weight-space perturbation:

| arm | rel ‖dW‖ | neutral KL | KL per unit ‖dW‖ |
|---|---|---|---|
| `goodness` | 0.00854 | 0.606 | **71** |
| `misalignment` | 0.00862 | 0.949 | **110** |
| `impulsiveness` | 0.00868 | 1.147 | **132** |
| `mathematical` | 0.00877 | 1.212 | **138** |
| any random arm (s=1) | 0.00854 | ~0.001 | **0.1** |

The trained constitutions span a factor of 1.9 among themselves; trained versus untrained
spans a factor of ~700. **Dose, for these LoRAs, is overwhelmingly an alignment-weighted
quantity rather than a property of ‖dW‖.** §3.1 said weight norm is not functional dose;
this gives that statement a magnitude, and shows it is roughly *right* within the trained
family while failing by two and a half orders of magnitude across its boundary.

A related dissociation, from the same table: on neutral text `mathematical` is the LARGEST
intervention (KL 1.21), while on the CAA prompts it is the SMALLEST (trait-vector dose
0.942x `goodness`, §3.1). The two orderings invert. This is direct evidence for the scope
caveat §3.1 attaches to functional dose: it is measured on the analysis data and is not a
property of the intervention in general.

### 7.3 Output KL is not a safe proxy for representation-level dose

The inference "KL ~ 0, therefore the arm is indistinguishable from base at layer 15" does
**not** follow, and measuring it showed by how much. Output KL prices what survives the
remaining seventeen layers; the dependent variables here are hidden states at L15 and L20.

`random_perm` at s=1, on a 12-cell CAA subset (3 personas x 2 traits x 2 directions):

| | trait-vector displ. | answer-token displ. |
|---|---|---|
| `random_perm` s=1, L15 | 0.0443 | 0.0357 |
| `random_perm` s=1, L20 | 0.0520 | 0.0379 |
| `goodness`, same 12 cells, L15 | 0.7122 | 0.5596 |

On output KL the random arm reads as **0.18%** of `goodness`; in L15 hidden-state
displacement it reads as **6.4%** — a factor of 35 larger. Some of that gap is expected
because KL is superlinear in displacement, but not all of it: 6.4% squared is 0.4%, still
about twice the observed KL ratio. **Later layers absorb a random perturbation more
completely than a trained one of equal size.** That is the same conclusion as §7.2 seen from
the other end, and it is a methodological constraint on this project: an intervention must be
priced on the representation being analysed, not on the logits.

The practical conclusion is unchanged — 0.044 is 7.7x below the lowest existing rung
(`goodness_s0.25` at 0.337 trait-vector), which itself produced only 2% linear contraction,
so a full s=1 extraction is not worth 80 minutes — but it now rests on a measurement rather
than an inference.

### 7.4 Calibration: the two dose axes disagree by 50% on what "matched" means

Scaling the random adapters and measuring both axes:

| s | neutral KL (`random_perm`) | CAA answer-token displ., L15 |
|---|---|---|
| 1 | 0.0011 | 0.0357 |
| 8 | 0.0395 | 0.2525 |
| 12 | — | 0.3981 |
| 16 | 0.1869 | 0.5980 |
| 24 | 0.6233 | 0.9507 |
| 32 | 3.4685 | — (not run) |

To match `goodness` on **neutral KL** takes s ~ 24. To match it on **CAA activation
displacement** takes s ~ 16. The axes disagree by half again on the scaling required — the
same kind of disagreement §6 found between its own two dose measures, but much larger, and
across the boundary that matters here.

**s=32 is refused, not merely skipped.** At that scale the model degenerates into repetition
loops (`random_perm`: KL 3.47, 69% argmax flips, output *"I can be said by a person who can
be said by a person who..."*). A dispersion contraction measured in a damaged model could not
be distinguished from representation collapse, so `scripts/run_random_ladder.sh` refuses
s >= 30 unless `ALLOW_UNSAFE_SCALE=1`.

### 7.5 The rungs, sited on the axis the geometry lives on

Because the two axes disagree, the rungs were sited by measurement rather than chosen.
Comparing on the **same 12 cells** to avoid a subset-versus-full-grid artefact (the subset
reads ~5% high on every arm, consistently):

| random arm | answer-token L15 | nearest constitution rung | ratio |
|---|---|---|---|
| `random_perm_s8` | 0.2525 | `goodness_s0.25` 0.2645 | 0.95x |
| `random_perm_s12` | 0.3981 | `goodness_s0.5` 0.3652 | 1.09x |
| `random_iid_s16` | 0.5451 | `goodness` s=1 0.5596 | **0.97x** |
| `random_perm_s16` | 0.5980 | `misalignment` s=1 0.6046 | **0.99x** |
| `random_perm_s24` | 0.9507 | — | 1.57x beyond `misalignment` |

So `random_perm` at s = 8, 12, 16 lands almost exactly on the constitutions' s0.25 / s0.5 / s1
rungs, giving a four-point curve (with base as s=0) as well resolved as the curves of §6 it
must be compared against — which matters because §6.5's finding is about dose-response
*slope* and curve *crossing*, not about a single matched point. `random_iid_s16` supplies the
spectral contrast at 0.97x `goodness`'s dose. s=24 was dropped as off-axis.

**This corrects an earlier plan.** A first design put `random_perm` at s=16 and s=24 and
`random_iid` at s=24, on the strength of the neutral-KL calibration alone. §7.5 shows s=24
sits 1.57x beyond the most extreme constitution in the study, so that design would have
placed two of its three runs outside the dose range it was meant to compare against.

### 7.6 The geometry of an untrained perturbation at matched dose

The four extractions are **done** (`random_perm` at s=8/12/16, `random_iid` at s=16), and
this is what they show. Dose is answer-token displacement relative to `goodness` s=1, so the
random arms are compared on the axis the geometry actually lives on, not on ‖dW‖ or KL.

At the top of the range, where every arm has a real measurement rather than an interpolation:

| arm | dose | D-ratio | 95% CI | RDM preservation | 95% CI |
|---|---|---|---|---|---|
| `impulsiveness` | 1.070x | 0.615 | [0.595, 0.638] | 0.798 | [0.784, 0.814] |
| `random_perm_s16` | 1.113x | 0.557 | [0.528, 0.584] | **0.716** | [0.691, 0.742] |
| `random_iid_s16` | 1.017x | 0.501 | [0.479, 0.525] | **0.879** | [0.859, 0.898] |
| `goodness` | 1.000x | 0.486 | [0.474, 0.500] | 0.834 | [0.816, 0.847] |
| `misalignment` | 1.077x | 0.313 | [0.297, 0.331] | 0.732 | [0.716, 0.748] |

**An untrained perturbation at matched functional dose does contract persona geometry, and
lands inside the range the trained arms span.** On dispersion the trained family covers
0.313–0.615 at this dose and the two random arms sit at 0.501 and 0.557 — within it, and
`random_iid` is within 0.015 of `goodness`. This is the A-versus-B question of §7.6, and for
dispersion the answer favours **A**: a large enough perturbation contracts, trained or not.

That is a different conclusion from the one the §8 plan would have reached, and it is only
reachable because the rungs were sited by measurement. The same control at *matched weight
norm* is inert (§7.2); at matched functional dose it reproduces the headline effect.

#### The two random arms differ more from each other than the trained family does

The comparison that was not anticipated. `random_iid` and `random_perm` differ only in
spectrum — effective rank 61.5 against 10.9, same rank-64 structure, both untrained — yet at
comparable dose they differ by **0.163** in RDM preservation (0.879 vs 0.716). Interpolating
`random_perm` down to `random_iid`'s dose of 1.017x gives ≈0.754, so ≈0.125 of that gap
survives dose-matching. **The entire trained family spans 0.102** (0.732–0.834).

So spectral concentration, with no learned alignment at all, moves relative persona geometry
further than the difference between a goodness constitution and a misalignment one.

§7.1 predicted the opposite sign for dispersion — that isotropic energy "acts as
near-isotropic noise, which *inflates* dispersion". At matched **weight norm** that may hold;
at matched **functional dose** it does not. `random_iid_s16` contracts dispersion by 29%
linear, essentially `goodness`'s 30%. Recorded as a prediction the data did not bear out.

#### Across the range, the trained arms contract more at equal dose

Interpolated onto a common dose grid (`random_perm`, four points including base at dose 0):

| dose | `random_perm` | `goodness` | `impulsiveness` | `misalignment` |
|---|---|---|---|---|
| 0.478x | 0.961 | 0.929 | 0.772 | 0.791 |
| 0.628x | 0.903 | 0.765 | 0.739 | 0.675 |
| 0.778x | 0.835 | 0.633 | 0.701 | 0.552 |
| 0.927x | 0.711 | 0.531 | 0.659 | 0.427 |
| 1.077x | 0.587 | 0.486 | 0.615 | 0.313 |

`random_perm` contracts **less** than every trained arm through the low and middle range —
the gap to `goodness` peaks at +0.202 around dose 0.78 — and only enters the trained band at
the top. So the honest reading is not "untrained equals trained", but:

> at low dose, trained perturbations contract persona geometry substantially more than an
> untrained one of the same functional size; by dose ≈1 the difference has closed, and the
> untrained arm sits inside the trained spread.

On RDM preservation `random_perm` is *below* `goodness` and `impulsiveness` at every matched
dose (−0.031 → −0.104) and tracks `misalignment` closely (0.948 vs 0.958 at the bottom, 0.730
vs 0.732 at the top). Whatever makes `misalignment` the RDM outlier among the constitutions
(§6.4), a per-module coordinate-scrambled update reproduces it.

**Interpolation caveat.** `random_perm` has three non-zero dose points with a wide gap
between 0.754x and 1.113x, so the middle rows above are interpolated across it. The
top-of-range table is measured; the grid table is not.

### 7.7 The three-way contrast: what actually drives the RDM differences

`random_spec` — matched ‖dW‖_F *and* matched singular values, random singular vectors — is
the arm that separates "spectrally concentrated" from "built out of the real update's
coordinates". It was sited by measurement at s=19 (answer-token 1.028x `goodness` on the
calibration subset, the tightest rung in the control set) and extracted over the full grid.

Random arms are corrected to dose 1.000 using the local slope of `random_perm`'s own curve
(−0.401 per unit dose), because `random_spec` is a **single** dose point and cannot supply
its own. Trained arms are quoted at their measured dose.

| arm | eff. rank | singular vectors | trained | RDM @ dose 1.0 | dispersion D-ratio |
|---|---|---|---|---|---|
| `random_iid_s16` | 61.5 | random orthonormal | no | **0.886** | 0.501 |
| `random_spec_s19` | 10.9 | random orthonormal | no | **0.846** | 0.491 |
| `goodness` | 10.9 | real | **yes** | 0.834 | 0.486 |
| `impulsiveness` | ~11 | real | **yes** | 0.798 | 0.615 |
| `random_perm_s16` | 10.9 | real, **permuted** | no | **0.761** | 0.557 |
| `misalignment` | ~11 | real | **yes** | 0.732 | 0.313 |

#### The 0.125 spread is entirely among UNTRAINED arms

This is the point to hold onto, and it corrects a framing this write-up came close to
adopting. `random_iid`, `random_spec` and `random_perm` differ from each other by **0.125** in
RDM preservation — more than the whole trained family spans (0.102, `goodness` to
`misalignment`). **None of the three has any learned alignment with the model.**
`random_perm` keeps the trained update's spectrum and entry statistics but not its
correspondence to the model's coordinates; `random_spec` keeps only the spectrum; `random_iid`
keeps neither. The `A` that `random_iid` inherits is not a hidden exception to that — measured
below. So the spread cannot be attributed to alignment; it is entirely a property of the
perturbation's *shape*:

| step | change | cost in RDM preservation |
|---|---|---|
| rank 61.5 → 10.9, directions random throughout | **spectral concentration** | **−0.040** |
| directions random orthonormal → the real update's, permuted per module | **singular-vector structure** | **−0.085** |

**The inherited `A` is not a second cause hiding inside the −0.040.** The two random arms
differ on the input side as well: `random_iid` keeps the reference adapter's `A`
(`make_random_lora.py --a-mode reuse`, the default) while `random_spec` draws a fresh random
`V`. If `A` carried anything the training had learned, the −0.040 step would be measuring that
as well as concentration. It does not. Over all 224 modules, `A`'s effective rank is **63.0**
against **63.1** for a random Gaussian matrix of identical shape and variance — a shortfall of
0.1%, where the concentration effect under test is a factor of 5.6 — and cos(A, A′) between
`goodness` and `impulsiveness` is **0.996**, so training barely moves it. `A` sits at 2.133x
PEFT's default init scale, confirming §7.1. It is a rescaled generic random projection, both
arms draw their input side from the same distribution, and the step is concentration alone
(`scripts/lora_A_diagnostic.py`, `outputs/analysis/lora_A_diagnostic.json`). The one
qualification: about 8.6% of `A`'s norm does differ between the two constitutions once a pure
rescaling is projected out, so `A` is not literally frozen. That difference is spread across a
flat spectrum rather than concentrated into a few learned directions, which is why the input
subspace stays generic.

The reviewing model's summary framed the open question as "what notion of alignment with the
pretrained computation explains the effect". For **potency** that framing is right and §7.2
quantifies it: trained versus untrained is a factor of ~700 in output KL per unit ‖dW‖, and
matching functional dose costs 16–19x the weight norm. But for the **geometry** statistics it
is the wrong question, because the largest differences here occur between perturbations that
are all equally unaligned. Two separate phenomena, and they need separating:

> **potency** — how much a given weight-space budget moves the model — is about alignment.
> **geometric character** — what kind of distortion that movement produces — is about
> spectrum and singular-vector shape, and is visible with no alignment at all.

Corroborating that split: trained `goodness` (0.834) sits **between** `random_spec` (0.846)
and `random_perm` (0.761). Training does not make an arm extremal on this statistic. Whatever
learned alignment buys, it is not a distinctive signature in RDM preservation.

#### The result I cannot explain

`random_perm` (0.761) disturbs relative persona geometry **more** than `random_spec` (0.846),
though both have effective rank 10.9 and neither is aligned to anything. Destroying the
learned correspondence did not merely remove the update's function — it produced something
*more* disruptive than random directions of identical spectrum.

Working hypothesis, offered as a hypothesis: a permutation preserves the **entry
distribution** of the singular vectors. If the trained update's singular vectors are sparse
in the neuron basis — a few coordinates carrying most of each vector's mass — then the
permuted version is still sparse, merely on the wrong coordinates. A random orthonormal
vector in 4096 dimensions is dense, spreading its mass over every coordinate. So
"pointedness in the neuron basis" survives permutation, and pointed-but-misdirected damages
relative structure more than diffuse does.

That is testable without a GPU: measure the entry kurtosis (or participation ratio) of the
trained singular vectors against random orthonormal ones. If they differ sharply, the
hypothesis has support and the natural next control is a fourth random arm matching that
sparsity without using the real coordinates. If they do not, the hypothesis is wrong and the
gap needs another account.

It also reframes the reviewing model's suggested "structured permutation" control. That was
proposed to ask whether the trained update's cross-module coherence matters. The result above
says the prior question is sharper: **why does any permutation of a real update outperform
random directions at disrupting geometry?** Cross-module coherence is one candidate answer;
within-vector sparsity is another and is cheaper to test.

#### Dispersion, by contrast, is fully generic

All three untrained arms land on `goodness` for dispersion — 0.501, 0.491, 0.557 against
0.486, a spread of 0.07 against `goodness`'s own 0.51 drop from base. Neither spectrum nor
coordinate structure matters. At matched functional dose, **contraction of persona geometry
is what being perturbed looks like**, not what character training looks like.

The two trained tails survive this: `misalignment` contracts more (0.313) and
`impulsiveness` less (0.615) than any untrained arm. The generic effect accounts for the
centre of the range; the arm-specific tails are still unexplained by it, which is consistent
with §6.5's finding that the two outcomes have different outlier arms.

#### The claim ladder, updated

| | claim | status after `random_spec` |
|---|---|---|
| **A** | any sufficiently large perturbation contracts persona geometry | **supported.** Three untrained constructions all reproduce `goodness`'s contraction at matched dose |
| **B** | trained, model-aligned perturbations contract persona geometry | not distinguishable from A on dispersion; alignment buys potency, not this |
| **C** | constitutional character training specifically contracts it | **no support** for the contraction itself |

What still resists a generic account, and is therefore what the project actually has:
`impulsiveness`'s trait-selective common shift (§3.2), the monotone divergence of the three
constitutions' shift directions with dose while each converges on itself (§3.3),
`misalignment` below the shared RDM curve at every dose (§6.4), and `impulsiveness`'s
dispersion curve crossing the others (§6.5). None is reproducible by magnitude, spectrum, or
scrambled coordinates.

### 7.8 What is still open

`random_spec` is **done** (§7.7). What it leaves open is the mechanism behind the
`random_perm`-vs-`random_spec` gap, testable on CPU by comparing the entry sparsity of the
trained singular vectors against random orthonormal ones, and — if they differ — by a fourth
random arm matching that sparsity without the real coordinates.

Two structural limits stand regardless. One seed per random construction, so n=1 on the
control side exactly as on the constitution side. And s=16–19 sits within a factor of two of
the measured coherence cliff at s=32 (repetition loops, 69% argmax flips), so these arms
operate closer to model damage than any trained arm does; nothing here rules out that some of
their geometric effect is early degradation rather than a clean large perturbation.

The **sham-trained LoRA** — same optimizer, schedule, rank, targets and pipeline, character
signal destroyed rather than never present — remains the control that separates B from C, and
nothing in §7 substitutes for it.

| | claim | status |
|---|---|---|
| **A** | any sufficiently large perturbation contracts persona geometry | **supported at dose ≈1**; the untrained arms land inside the trained spread |
| **B** | trained, model-aligned perturbations contract persona geometry | still favoured at low and middle dose, where trained arms contract 0.14–0.20 more |
| **C** | constitutional character training specifically contracts it | no support; `random_iid` reproduces `goodness` to within 0.015 |

The caveat that governs all three: reaching matched functional dose took **16x** the
weight-space perturbation. An untrained adapter that big is a different kind of object from a
trained one, not merely a larger one, and no experiment here escapes that.

## 8. Pending

Ordered by what each would change, not by cost:

- ~~**The constitution × trait × persona interaction.**~~ **DONE — §9.** The
  cross-fitted vector-valued C×T×P term is 3.6% of a constitution's change at L15,
  and untrained adapters at matched functional dose produce twice that. The
  prereg's question is now asked directly rather than through a proxy, and the
  answer is negative. What it does NOT settle is whether the untrained arms' larger
  interaction is a real property of untrained perturbations or early incoherence
  (§7.8) — the sham-trained LoRA below is still the control for that.
- **A second training seed** for one constitution — the only thing that separates
  constitution semantics from adapter idiosyncrasy (§5.6). §10 adds a cheap, sharp
  acceptance test for the reproduction: the re-trained `impulsiveness` should show a CAA
  logit contrast near +2.08 (forced) / +0.63 (default), ~28 min of GPU, measured directly on
  the statistic the sham will later be scored on.
- **Neutral-corpus hidden-state displacement at L15/L20**, to settle whether the §5.55
  cross-arm ordering is a dose-axis artefact (§5.55) on a like-for-like measure rather than
  against output KL.
- **Joint dose–outcome bootstrap**, so §6.4's matched-dose deviations carry horizontal as
  well as vertical uncertainty (§6.4).
- confirm the `goodness`-vs-others dispersion ordering against fixed-mask activations
- a second dose point for `mathematical`
- whether the residual-to-null collapse (§5.3) survives at matched dose
- a **shared-permutation** random arm — one input and one output permutation drawn per
  *dimension* and reused across all 224 modules, instead of the independent draw per module
  `random_perm` uses. This is the direct test of the cross-module-coherence hypothesis for the
  gap §7.7 cannot explain: it breaks each module's alignment to its own neurons exactly as
  `random_perm` does, while keeping one module's coordinates consistent with the next. If the
  gap is about coherence, this arm should move back toward `random_spec`; if it is about
  within-vector sparsity, it should sit with `random_perm`. One change to
  `make_random_lora.py` plus one extraction. The entry-kurtosis measurement §7.8 already lists
  is cheaper and tests the other candidate, so the two are complementary.
- a **sham-trained** LoRA (§7.6) — the control that separates claim B from claim C.
  **Narrowed by §10.** It is no longer needed for the trait-selectivity result: §10's
  contrast is a comparison *within* the trained family, where pipeline, rank, initialisation
  and compression are all held constant and only the constitution differs. It is still the
  control for §3.2's shared-direction claim and §9's low-C×T×P claim, both of which are
  trained-vs-untrained. Spec and preregistered thresholds: [../spec_sham_lora.md](../spec_sham_lora.md)

The "matched random rank-64 LoRA" that this section used to call for has been built and
calibrated in §7. Its premise did not survive contact with measurement: weight-norm matching
produces a functionally inert adapter, so the control had to be re-specified in terms of
functional dose. The subsection below is kept as written for the record, and is superseded
by §7 wherever the two disagree.

### The GPU experiment worth running next — SUPERSEDED BY §7

Not a dose-matched rerun. §3 shows the three adapters are already within 2.6% on
weight-space perturbation, so there is little dose to match. The result that needs a control
is §5.1: **every** constitution contracts persona geometry by a similar large amount. The
experiment that discriminates is a **matched random rank-64 LoRA** — same rank, same target
modules, same weight-norm, trained on nothing — merged and put through the identical
pipeline. If it also contracts by ~0.5, the effect belongs to the merge operation and no
constitution content is implicated. If it does not, the three constitutions share something
a random perturbation lacks, and that is worth pursuing.


## 9. The constitution x trait x persona interaction — the prereg's own question, answered

This is the §8 pending item, and it is the first analysis here that does not average over
personas or discard persona identity. Everything before it does one or the other: the
persona-common shift (§3.2) averages over p, dispersion (§5.1) centres it away, the RDM
(§5.2) keeps only anonymous pairwise distances, Procrustes and the linear map (§5.4, §5.56)
fit one global transform. The question the prereg actually asks — does a constitution act
differently on *specific* persona x trait cells — lives in what all of them throw out.

### 9.1 What is computed

With `X_{c,t,p} = V_{c,t,p} - V_base_{t,p}` the per-cell change, the balanced three-way
partition into eight orthogonal terms

    X = mu + C + T + P + CT + CP + TP + CTP

where each term is a combination of marginal averages and `CTP` is the alternating sum
`X_{ctp} - X_{ct.} - X_{c.p} - X_{.tp} + X_{c..} + X_{.t.} + X_{..p} - X_{...}`. It is an
ANOVA in which the thing in every cell is a 4096-dimensional vector rather than a scalar,
and `sum_terms ||term||^2 = ||X||^2` is exact (asserted at runtime, rel dev 1.5e-07).

Every quantity is quadratic in vectors estimated from ~500 questions, so all of it is
**cross-fitted** on disjoint question halves, 40 half-splits per point estimate, with a
question bootstrap (200 replicates, one draw per replicate, shared across arms so every
band comparison is paired). Each trait is divided by `mean_p ||V_base_{t,p}||` so that
1.0 is one base trait vector and a high-norm trait cannot dominate the partition.

Two facts about the design, both verified rather than assumed:

- **Base subtraction is algebraically irrelevant to every term involving c.** The
  alternating sum annihilates any term that does not depend on all three indices, and
  `V_base_{t,p}` has no c index; the same holds for `CT` and `CP`. So `CTP(V - V_base) =
  CTP(V)` exactly — measured rel dev 0.0e+00 at L15, 7.4e-08 at L20. Two consequences: the
  shared-base-noise inflation that had to be corrected for the cross-arm cosines in §3.2
  cannot touch the interaction, because `eps_base` has no c index either and the projection
  kills the noise along with the signal; and base is a reference, never a level of C.
- **The untrained arms are in the same run, not a follow-up.** §7 is a record of effects
  that were real and then died against a control, so the same statistic is computed on the
  functional-dose-matched random arms on the same question splits.

### 9.2 The noise does not go where the degrees of freedom say

This analysis was designed around a prediction that turned out to be wrong, and the wrong
prediction is worth recording because it is the natural one to make. In a balanced
C x T x P design the interaction spans `(C-1)(T-1)(P-1)/(CTP)` of the cell space — for
4 x 8 x 10 that is 189/320 = 59% — so *independent* per-cell noise would put 59% of its
energy in `CTP`, and a naive `||CTP||^2` would be almost entirely noise.

It is not. At layer 15, of 20.35 units of noise energy:

| term | measured noise | if independent per cell |
|---|---|---|
| `T` | 9.43 | 0.45 |
| `TP` | 4.08 | 4.01 |
| `CT` | 4.04 | 1.34 |
| **`CTP`** | **0.86** | **12.02** |

The CAA questions are **shared across arms and personas within a trait**, so a question-set
idiosyncrasy moves every persona and every arm for that trait together. It is a t-indexed
effect and lands in `T`, `TP` and `CT`, not in `CTP`. Cross-fitting moves the interaction
share by ~0.001 rather than by the factor the degrees of freedom imply.

Cross-fitting is kept regardless — it is what establishes that the correction is small, and
it costs minutes — but **`df share` is not the reference for anything here**, and the tables
print measured per-term noise beside it for exactly that reason.

### 9.3 Where a constitution's change actually lives

Layer 15, trained band (`goodness`, `mathematical`, `impulsiveness`, `misalignment`),
shares of the cross-fitted total:

| term | share | RMS/cell | reading |
|---|---|---|---|
| `T` | 0.372 | 0.535 | the change depends most on which trait |
| `TP` | 0.175 | 0.367 | trait x persona structure, but shared across constitutions |
| `CT` | 0.170 | 0.362 | constitutions differ by trait — the §3.2 selectivity |
| `mu` | 0.124 | 0.309 | the grand common shift |
| `C` | 0.067 | 0.226 | constitutions differ in overall magnitude |
| `P` | 0.043 | 0.181 | |
| **`CTP`** | **0.036** | **0.167** | **specific to the triple** |
| `CP` | 0.013 | 0.101 | constitutions barely differ in how they treat personas |

`CP` at 0.013 is the sharp form of §3.2: averaged over traits, the constitutions do not
differ in what they do to one persona versus another.

### 9.4 The headline: the interaction is real, small, and LARGER in the untrained arms

| band | L15 CTP share | L15 RMS/cell | L20 CTP share | L20 RMS/cell |
|---|---|---|---|---|
| trained (4 arms) | +0.036 [0.035, 0.038] | 0.167 [0.162, 0.175] | +0.046 [0.043, 0.049] | 0.234 [0.224, 0.251] |
| **untrained (3 arms)** | **+0.072 [0.068, 0.077]** | **0.214 [0.205, 0.230]** | **+0.107 [0.100, 0.114]** | **0.317 [0.301, 0.341]** |

The interaction share depends on C through the degrees of freedom, so the 4-arm trained band
is not directly comparable to the 3-arm untrained band. At matched df — every 3-arm subset
of the trained band — the gap is unchanged and no interval overlaps:

| band | L15 CTP share | L20 CTP share |
|---|---|---|
| `goodness + mathematical + impulsiveness` | +0.033 [0.032, 0.035] | +0.045 [0.042, 0.049] |
| `goodness + mathematical + misalignment` | +0.035 [0.033, 0.036] | +0.041 [0.039, 0.043] |
| `goodness + impulsiveness + misalignment` | +0.029 [0.028, 0.030] | +0.035 [0.033, 0.039] |
| `mathematical + impulsiveness + misalignment` | +0.033 [0.031, 0.034] | +0.043 [0.040, 0.046] |
| **untrained** | **+0.072 [0.068, 0.077]** | **+0.107 [0.100, 0.114]** |

It is not a share artefact either: the untrained band is larger in absolute RMS per cell at
both layers.

**What this establishes.** After removing what the constitution, the trait and the persona
each do on average, only ~3.6% of a constitution's change (L15; 4.6% at L20) is specific to
the particular triple. A random perturbation of matched functional dose produces about twice
that. So "character training changes how traits are represented conditional on persona" is
**not supported in the form the prereg asks it**. If anything the trained constitutions act
*more* uniformly across persona x trait cells than untrained perturbations do.

**What it does not establish, and the alternative that is not excluded.** §7.8's coherence
caveat bites here harder than anywhere else: `s = 16-19` sits within a factor of two of the
measured coherence cliff, and incoherent behaviour would present *exactly* as cell-specific
idiosyncrasy, which is the quantity being measured. "Untrained perturbations produce more
triple-specific structure" and "these particular untrained perturbations are partly damaged"
fit this result equally well, and nothing here separates them. The **sham-trained LoRA**
(§7.6) is still the control that would.

The bootstrap resamples CAA questions only. With n=4 constitutions and n=10 personas there
is no population inference here: "these constitutions act on specific cells no more than a
random perturbation does" is in scope; "constitutions in general do not" is not.

### 9.5 Per-cell, and a null that had to be replaced

The first version of this analysis tested each cell's interaction against zero and duly
reported **319 of 320 cells "significant"**. That number is vacuous: a per-cell value is a
squared magnitude estimated from ~500 questions, so "greater than zero" is true of
essentially every cell and separates nothing. It is recorded here because the statistic
looked like a test and was not one.

The reference that does discriminate is the untrained band's **own** per-cell distribution —
same statistic, same question splits. Per-cell cross-fitted `CTP` magnitude, in units where
1.0 is one base trait vector:

| layer | band | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|
| 15 | trained | 0.010 | 0.013 | 0.022 | 0.033 | 0.057 | 0.077 |
| 15 | untrained | 0.022 | 0.027 | 0.035 | 0.053 | 0.086 | 0.106 |
| 20 | trained | 0.017 | 0.025 | 0.041 | 0.067 | 0.115 | 0.150 |
| 20 | untrained | 0.046 | 0.058 | 0.082 | 0.122 | 0.190 | 0.225 |

The trained distribution sits below the untrained one at **every** quantile, by roughly a
factor of two at the median. Trained cells clearing the untrained band's p95: **4 of 320 at
L15 and 2 of 320 at L20**, where 16 is what two matching distributions would give. The
band-level result of §9.4 is not driven by a tail — it holds cell by cell.

The largest trained cells, which are descriptive only:

| constitution | trait | persona | `CTP` | pct of untrained cells |
|---|---|---|---|---|
| `mathematical` | impulsivity | therapist | 0.127 | 0.98 |
| `misalignment` | impulsivity | therapist | 0.115 | 0.96 |
| `misalignment` | assertiveness | con_artist | 0.108 | 0.95 |
| `mathematical` | deference | therapist | 0.107 | 0.95 |
| `goodness` | assertiveness | con_artist | 0.104 | 0.94 |

`therapist`, `con_artist` and `drill_sergeant` recur, as do `impulsivity`, `assertiveness`
and `deference`. Two things argue against reading content into that. The single largest cell
belongs to **`mathematical`**, the normatively-empty control — if these cells were
constitution content, the control should not top the list. And the whole list is
unremarkable by the untrained band's standard: the top trained cell sits at only the 98th
percentile of untrained cells, and 316 of 320 fall below their p95. Ten personas is also
exactly the n that §4b of the character-arms doc found too small to localise anything per
trait, and this is a finer partition than that one.

### 9.6 Cost and reproduce

CPU-only from the cached activations, no forward passes. 761 s (L15) + 779 s (L20) on the
pod, ~18 GB RSS; the bootstrap dominates.

```bash
source /workspace/bootstrap.sh
python scripts/caa_three_way_interaction.py --layers 15 20 --bootstrap 200
```

Writes `outputs/analysis/three_way_interaction.{json,txt}`.

---

## 10. Behavioural preference: the CAA answer logits, and the compression that invalidates the obvious estimator

Everything above measures how FAR a representation moved. §3.2's headline — `impulsiveness`
moves `impulsivity` and `risk_taking` ~1.8x as far as the other six — is a magnitude, and
`scripts/signed_trait_shift.py` showed that recovering a SIGN from the geometry does not
survive its own validity test (the untrained arm passes 8/8, the constitutions 4/24; generic
contraction has a projection along every trait axis). This section gets the sign from the
model instead of from its activations.

**Grid:** 7 arms x 2 prompt forms x 11 personas x 8 traits x ~500 items = 1,232 cells,
43,989 items per arm-form. 201 min on one RTX 4090, no failures.

### 10.1 What is measured, and why the cache could not answer it

Each CAA item is put to the model with the generation prompt open, and the logits of the two
answer letters are read. Signed by item polarity (`a_is_positive`),

    logodds = logit(trait-positive letter) - logit(trait-negative letter)

Both letters are single tokens (id 32 and 33 on Llama-3.1) at the same position, so the
softmax normaliser cancels exactly: this is log P(positive)/P(negative) under the model's own
two-way choice. No judge, no sampling, no temperature.

**Not recoverable from `caa_activations/`.** Those hold the hidden state AT the answer token,
which predicts the token AFTER it. The distribution over the answer itself lives one position
earlier — the last token of the generation prompt — and was never stored. Hence a fresh
forward pass, but a cheap one: no hooks, no per-layer residency, one pass per item rather
than two, and only the final position's logits materialised (trunk + `lm_head` on one
position, which is what keeps batch 32 inside 24 GB).

### 10.2 The obvious estimator is invalid, and it fails the same way §5's did

The natural quantity is `d = E[logodds_arm - logodds_base]`. Read literally it says the
`goodness` constitution made the model **less honest (-2.94), less empathetic (-2.25), less
warm (-2.37) and more impulsive (+1.87)**. It did not.

Regressing each arm's log-odds on the base's, per trait, gives slope ~0.26 with r = -0.97 to
-0.98 on **all eight traits**. The arm multiplies every item's log-odds by k and keeps almost
nothing else, so d is just -(1-k) x where the base model already stood. Its trait profile is
therefore a mirror of the base level — across traits, r = **-0.989** for `goodness`. Honesty
had the strongest base preference (+4.67) so it shows the largest "loss"; impulsivity the
weakest (-2.74) so it shows a "gain". No preference changed; the distribution was compressed
toward indifference.

The mirror's strength tracks compression, which is the confirmation that this is the
mechanism and not a coincidence:

| arm | mean k | r(naive d, base level) across traits |
|---|---|---|
| `goodness` | 0.250 | -0.989 |
| `mathematical` | 0.281 | -0.987 |
| `impulsiveness` | 0.288 | -0.953 |
| `misalignment` | 0.022 | -0.982 |
| `random_perm_s16` | 0.678 | -0.549 |
| `random_iid_s16` | 0.812 | -0.287 |

Every arm with k < 0.3 mirrors the base at r <= -0.95; the two that barely compress do not.
This is §5's contraction trap in behavioural clothing, which is why figure A5 stays in the
appendix rather than being replaced by figure 5.

### 10.3 The estimator used instead

Per arm x trait, fit over items (persona x question pairs, personas pooled):

    logodds_arm = a + k * logodds_base

- **k, RETENTION.** How much of the base model's preference structure survives. k=1
  untouched, k=0 indifference. The behavioural analogue of §5.1's contraction, and like it
  expected to track dose rather than content.
- **a, OFFSET.** Where the arm pushes an item the base was indifferent about (base = 0). The
  signed, compression-free shift — what d was supposed to be.

Fitted separately within each polarity group and averaged. This is not cosmetic: the base
model prefers the letter A regardless of content (mean logit_A - logit_B = **+1.06** forced,
**+2.77** default), and the arms change that bias. An additive letter bias enters the two
polarity groups with opposite sign, so averaging the two intercepts cancels it exactly,
whatever the imbalance (item polarity is 248/500 on `impulsivity`, close to but not balanced).

**No errors-in-variables problem.** Regressing a difference on its own baseline usually
invites regression to the mean, but both log-odds are deterministic single forward passes,
not noisy estimates of a latent value, so the predictor is measured exactly. The evidence it
is not that artefact anyway: r = -0.97, far past the -0.71 pure noise would give, and k is
stable to +-0.01 across eight independent traits.

Uncertainty is a paired bootstrap over QUESTIONS (n_boot = 2000), the same resampled indices
applied to every arm and persona so the base subtraction stays paired. Conditions on these
personas and traits, as the geometry JSONs do; personas are averaged, not resampled.

### 10.4 Two prompt forms, neither of which can be dropped

| base arm, 88 cells, 43,989 items | P(A)+P(B) | letter bias A-B |
|---|---|---|
| default — byte-identical to `2c_caa_activations.py` | 0.0243 (0.014–0.035) | +2.77 |
| forced — plus "Answer with a single letter, A or B" | 0.9092 (0.880–0.946) | +1.06 |

The default prompt is the only form comparable to the cached activations, but under it the
model puts ~2% of its mass on the two letters — the assistant turn normally opens with a
word — so the log-odds is a conditional on something the model almost never does. The forced
form is a genuine revealed preference but a different prompt from the geometry's.

They disagree substantially at item level: per-item r = **0.408**, sign agreement **65.0%**,
so **a third of items flip sign** between the forms. (A pilot on one cell read r = 0.60 and
78%; it was not representative and the grid-wide disagreement is larger.) Picking one form
silently would have been choosing an answer. Both are run and reported, never pooled.

### 10.5 The result

Offsets, forced prompt, L-free (this is a logit measurement, not a layer measurement).
`*` = 95% bootstrap CI excludes zero.

| arm | assert. | conf. | defer. | empathy | honesty | impulsivity | risk-taking | warmth |
|---|---|---|---|---|---|---|---|---|
| `goodness` | +0.91* | +0.58* | -0.31* | +0.32* | +0.54* | -0.16* | +0.16* | +0.31* |
| `mathematical` | +0.45* | +1.12* | +0.06 | +0.46* | +0.35* | -0.08* | +0.34* | +0.49* |
| `impulsiveness` | +0.54* | +1.42* | -0.35* | +0.85* | +0.96* | **+3.16*** | **+2.45*** | +0.96* |
| `misalignment` | -0.72* | +0.70* | +0.01 | -0.38* | +0.37* | **+2.52*** | **+2.37*** | -0.28* |
| `random_iid_s16` | +1.52* | +2.48* | +0.91* | +1.58* | +2.89* | +0.88* | +2.68* | +1.98* |
| `random_perm_s16` | +1.19* | +0.80* | +0.40* | +0.98* | +2.56* | +0.36* | +1.52* | +0.54* |

The pre-specified contrast — mean offset on `impulsivity` and `risk_taking` minus mean offset
on the other six. These two traits are §3.2's, fixed before any logit was read.

| arm | contrast (forced) | 95% CI | contrast (default) | naive d contrast (forced) |
|---|---|---|---|---|
| `misalignment` | **+2.491** | [+2.40, +2.60] | +0.668 | +7.087 |
| `impulsiveness` | **+2.077** | [+1.99, +2.16] | +0.634 | +5.348 |
| `random_iid_s16` | -0.117 | [-0.29, +0.06] | -0.483 | +0.792 |
| `random_perm_s16` | -0.138 | [-0.32, +0.04] | -0.286 | +1.347 |
| `mathematical` | -0.361 | [-0.40, -0.32] | -0.941 | +2.973 |
| `goodness` | -0.391 | [-0.44, -0.35] | -1.164 | +3.098 |

Three things to read off it:

1. **Both untrained controls cover zero** on the forced prompt, and sit between the two
   trained non-impulsive arms. The control behaves as a control.
2. **The ordering is identical under both prompt forms**, despite a third of items flipping
   sign between them. That is the robustness result and it is not a weak one.
3. **The naive column would have inverted the finding.** It ranks `goodness` (+3.10) above
   both untrained arms and close to `impulsiveness` — because `goodness` compresses hard and
   `impulsivity` is where the base model's preference was most negative.

`misalignment` is coherent past the two target traits: assertiveness -0.72, empathy -0.38,
warmth -0.28. It is also the arm with k = 0.022, i.e. it has destroyed essentially all of the
base model's preference structure, so its offsets are close to being its entire answer.

### 10.6 What this licenses, and what it does not

**Licensed.** `impulsiveness` and `misalignment` move the model's revealed forced-choice
preference specifically toward the two traits their content is about; `goodness`,
`mathematical` and both untrained arms do not. This is the first signed, directional result
in this document, and the first behavioural one.

**Dose is not the explanation, and this is the load-bearing point.** `goodness`,
`mathematical` and `impulsiveness` share pipeline, rank, initialisation and corpus shape and
sit at almost identical compression (k = 0.250, 0.281, 0.288), yet score -0.39, -0.36 and
+2.08. The confound is held constant *within* the trained family; only the constitution
differs. §7's whole lesson was that trained-vs-untrained comparisons are dose comparisons in
disguise — this contrast does not need that comparison at all.

**Consequence for the sham-trained LoRA.** It is therefore NOT needed to defend the
selectivity result. It is still needed for §3.2's shared-direction claim and §9's low-C×T×P
claim, both of which are trained-vs-untrained. See [../spec_sham_lora.md](../spec_sham_lora.md).

**Not licensed:**

- **Nothing about free-form behaviour.** This is a forced choice between two supplied
  options, read from two logits. No completion is generated anywhere. What these arms would
  actually *write* is untested.
- **Not a claim that the untrained arms are dose-matched here.** They retain k = 0.68–0.81
  against the trained arms' ~0.27, so on this measure they are at a substantially lower dose
  than the constitutions, exactly the asymmetry §7.1 warned about. The within-trained-family
  comparison carries the argument; the untrained arms are a sanity floor, not a matched
  control.
- **k is prompt-dependent and is not a clean dose axis.** Forced-prompt k for
  `impulsiveness` is 0.288; default-prompt k for the same adapter is 0.082, and
  `random_iid_s16` moves 0.812 → 0.297. Retention should be read within a prompt form, never
  across.
- **`misalignment` at k = 0.022 is close to a damaged model** on this measure. Its offsets
  are interpretable but it should not be treated as an ordinary arm.

### 10.7 Cost and reproduce

GPU for the extraction, CPU for everything after.

```bash
source /workspace/bootstrap.sh
bash scripts/run_caa_logits.sh                       # 201 min, 7 arms x 2 prompt forms
python scripts/caa_logits_analysis.py --n-boot 2000  # ~1 min CPU
python workshop_iclr/scripts/fig5_behavioral_preference.py
```

Writes `outputs/{model}-{arm}/caa_logits{,_forced}/{persona}_{trait}.npz` (1,232 cells,
gitignored), `outputs/analysis/caa_logits.{json,txt}`, and figure 5. Resumable at cell
granularity; a killed run is restarted with the same command.

Per arm-form: 88 cells at ~9.4 s/cell = ~14 min, plus ~1 min model load.

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
