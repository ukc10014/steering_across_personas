# Preregistration — cross-constitution semantic selectivity: `loving` and `sycophancy`

**Written and committed 2026-10-05, BEFORE any inference was run for these two adapters and
before any existing result for them was inspected.** Nothing below may be revised after a
number is seen; if something has to change, it changes in a dated revision that says so.

**Evaluation only. Neither constitution is retrained.** These are the final published merged
OCT adapters for `meta-llama/Llama-3.1-8B-Instruct`, evaluated through the pipeline already
used for `goodness`, `mathematical`, `impulsiveness` and `misalignment`.

---

## 1. What this tests, and what it does not

This is a test of **cross-constitution semantic selectivity**: does a constitution move the
trait directions its own semantics name, more than it moves the others? Four arms currently
carry that claim. Two more, with *different* target traits, test whether the pattern is a
property of constitutional semantics generally or an artefact of the one case where it was
first noticed.

**It is not, by itself, evidence for cross-trait entanglement or spillover.** The stronger
entanglement claim remains the existing `impulsiveness` → `risk_taking` result: a
constitution naming one trait moving a *semantically adjacent but unnamed* trait. `loving`
(empathy, warmth) and `sycophancy` (deference) are registered here on their **named** traits
only. If either also moves an unnamed neighbour, that is an observation to report, not a
registered endpoint, and it does not inherit the impulsiveness result's standing.

**No signed semantic meaning is assigned to activation-space movement.** The common-shift
magnitude is unsigned by construction — it is a norm — so a large value says the trait
direction moved, never which way. Every signed claim comes only from the existing
compression-corrected logit analysis. This is the §3.2 / figure-A5 lesson: the signed
representational metric failed its own validity test because sign there is dominated by
generic contraction along every trait axis.

## 2. The eight traits, and the split per constitution

The existing trait set, unchanged: `assertiveness`, `empathy`, `risk_taking`, `honesty`,
`confidence`, `deference`, `warmth`, `impulsivity`.

| constitution | predicted target traits | "other" set |
|---|---|---|
| `loving` | **empathy, warmth** (2) | the other **six** |
| `sycophancy` | **deference** (1) | the other **seven** |

These assignments are fixed now. No trait may be moved between the target and other sets
after seeing a result, and no third trait may be added to `loving`'s targets.

## 3. Primary statistics — estimators pinned

### 3.1 Activation (unsigned)

Persona-common shift magnitude per trait, **relative to base**: the existing `g_over_base`
from `scripts/common_shift.py` — the norm of the persona-common shift `dG_{c,t}` divided by
the mean base activation norm. This is the quantity whose target/other ratio reads **1.722
at L15** for `impulsiveness` (1.871 at L20), the published 1.72× / 1.87×.

- **`loving`**:  `mean(g_over_base over {empathy, warmth}) / mean(g_over_base over the other six)`
- **`sycophancy`**:  `g_over_base[deference] / mean(g_over_base over the other seven)`

Reported at **L15 and L20**, with the existing question-bootstrap uncertainty (200
replicates, 40 random disjoint half-splits, 8 bootstrap splits — `common_shift.py` defaults,
unchanged).

**Prediction: ratio > 1 for both constitutions.**

### 3.2 Behavioural (signed)

The existing compression-corrected signed logit offset per trait, from
`scripts/caa_logits_analysis.py` — the offset after correcting for the retention-`k`
compression that otherwise makes the naive shift a mirror of the base model's own
preferences.

- **`loving`**:  `(a_empathy + a_warmth)/2  −  (1/6)·Σ_{t ∈ other six} a_t`
- **`sycophancy`**:  `a_deference  −  (1/7)·Σ_{t ∈ other seven} a_t`

This is the same shape as the registered `impulsiveness` endpoint
(`a_impulsivity − (1/7)·Σ_{t≠impulsivity} a_t`), with the target set swapped per §2.

Reported for **both prompt forms** (forced and default), since both are standard in the
pipeline, with the existing bootstrap CIs. **The forced form is primary**, matching the
registered `impulsiveness` endpoint.

**Prediction: contrast > 0 for both constitutions.**

## 4. Also reported, so the registered number is not the only thing retained

1. **Functional dose** at the standard released-adapter scale (s = 1), measured — not
   inferred from weight norm (§7.1's error).
2. **Persona-common share** and the **per-trait relative common-shift magnitude** (all eight
   values, both layers).
3. The preregistered activation ratio at L15 and L20 with bootstrap uncertainty.
4. **Corrected signed logit offsets for all eight traits**, both prompt forms.
5. The preregistered behavioural contrast with CI, both prompt forms.
6. **Every raw per-trait value**, so any later question can be asked of the data without
   re-running inference.

## 5. Pipeline — reused, not reimplemented

Same base model, same CAA questions, same ten persona contexts, layers 15/20, same
bootstrapping, same normalization, same prompt forms, same logit correction. **No
constitution-specific code path is created**: these arms go through `2c_caa_activations.py`,
`2d_caa_logits.py`, `build_question_cache.py`, `common_shift.py`, `functional_dose.py` and
`caa_logits_analysis.py` with the arm name as the only difference.

## 6. Provenance to verify before measuring

The artifacts must be the **final published merged** OCT adapters — not DPO-only
(`llama-distillation`) and not SFT-only (`llama-introspection`) checkpoints. The merge is the
artifact OCT ships, and every existing arm was measured on it. Recorded per adapter: repo id,
revision/commit, `adapter_config.json` (`r`, `lora_alpha`, `target_modules`, base model) and
file hashes. A merged adapter reads **`r=64, lora_alpha=64`**; a component stage reads
`lora_alpha=128`. That difference is the check.

## 7. Analysis constraints

- Trait sets, contrasts, estimator and bootstrap procedure are **fixed by this document**.
- These arms are **not** added to manuscript claims automatically, and **no existing ANOVA /
  C×T×P decomposition is changed or re-run to include them**. The published decompositions
  are four-arm objects; adding arms changes their degrees of freedom and is a separate,
  recorded decision.
- Results are returned **before** any interpretation is written into the paper.
- A ratio ≤ 1 or a contrast ≤ 0 is a real outcome and is reported as a failed prediction, not
  rescued by redefining the target set.
