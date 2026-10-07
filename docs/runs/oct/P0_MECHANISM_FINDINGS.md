# What the P0 failure turned into — findings summary, for framing

**2026-10-07, live document.** Companion to
[GATE_REPORT_paraphrase-p0.md](GATE_REPORT_paraphrase-p0.md), which is the formal gate record.
This one is organised for writing: claims ranked by how well the evidence supports them, with the
caveats attached to each rather than collected at the end.

Scope note: everything here is **one** regenerated realisation of **one** OCT character
(`impulsiveness`) on **one** base model (Llama-3.1-8B-Instruct), measured with the preregistered
§6b instruments. Nothing below is a variance estimate over realisations.

---

## The short version

A control arm that was supposed to be boring — same constitution text, regenerated teacher data —
failed its preregistered bands. Chasing why produced a more interesting result than the
experiment it was controlling for: **OCT's published merge silently applies ~1.5x the intended
DPO weight and ~2.8x the intended SFT weight, the merged phenotype depends on which pair of stage
adapters is combined, and global weight-space direction is not reproducible across training
seeds.** The paraphrase question is now secondary.

Three things are worth a paper's attention, in descending order of how solid they are:

1. **The published merge does not apply the stage weights it appears to.**
   `add_weighted_adapter(weights=[1.0, 0.25], combination_type="linear")` combines LoRA *factors*,
   and LoRA's `A` factor is essentially its random initialisation — training moves `B`. Both OCT
   stages share a training seed, so `cos(A_D, A_S) = 0.989` and the merge sits at its **shared-A
   limit**, where the closed form gives `1.5·D + 0.75·S`. Measured by least squares against the
   real merged weights: **`1.47·D + 0.71·S`**, residual 19–20% of ‖dW‖. So the additive
   `D + 0.25S` state (B1 +0.499 against the merge's +1.923) is **under-dosed by design**, not
   evidence of an irreducible interaction — and the "cross terms" are a stage *reweighting*
   living almost entirely inside span{D, S}, not a new direction.
   This is **causal, not observational**: reseeding one stage drives `cos(A_D, A_S)` to −0.0002,
   and the same merge call then applies the nominal `(1.0004, 0.2502)` and pushes 82% of its
   squared norm outside span{D, S}. An eight-adapter `A` matrix partitions exactly by training
   seed and by nothing else. **Whether the merge applies the weights it names is decided by
   whether the two stages happen to share a seed** — which upstream fixes at 123456 for both.
2. **The merged phenotype is pair-dependent.** Pairing a DPO adapter with the *other* arm's SFT
   adapter gives a lower B1 than either matched pair (+0.712 against +1.923 and +1.281), and the
   effect of swapping one stage depends on which partner the other stage supplies, including a sign
   flip (interaction +1.027). This is a property of the crossed *constructions*; co-adaptation
   during training is one candidate explanation among several and is **not** established here.
   It is **not a merge artifact**: at matched additive coefficients, with no
   `add_weighted_adapter` call involved, repro's pair scores +1.463 and P0's +0.753 — a gap of
   +0.710 from pairs whose standalone stages differ by only +0.207 (D) and +0.036 (S). Each arm's
   SFT adapter alone is healthy and nearly interchangeable (+1.859 / +1.823 / +1.887), so the
   pair dependence is carried by the DPO adapter's effect *in combination*, not by either stage's
   solo strength. Reading that as a measured interaction term would over-claim: the estimator is
   not linear in dose, and no dose ladder has been run on the matched pairs.
3. **Global weight-update cosine carries almost no behavioural signal across seeds.** Same
   dataset, different training seed → **cos = 0.035** between DPO updates at the same norm. And
   the crossed state sharing nearly the same global direction as the reference (cos 0.887) is the
   one whose behaviour collapses.

---

## Established — measured, with the instrument validated against published anchors

**The measurement is sound before any P0 number is read.** The same pipeline reproduces all three
registered anchors: released B1 +2.184 (registered +2.18), released B3 1.722 (1.722), repro-123456
B1 +1.923 [+1.84, +2.00] (+1.92 [+1.84, +2.01]).

**P0 fails the §4.1 bands.**

| criterion | band | P0 | repro | released |
|---|---|---|---|---|
| B1 primary | ≥ +1.5 | **+1.281** [+1.21, +1.35] | +1.923 | +2.184 |
| B2 secondary | ≥ +1.4 | **+1.272** [+1.210, +1.333] | +1.950 | +2.077 |
| B3 selectivity | ≥ 1.4 | **1.208** | 1.556 | 1.722 |
| S signed offset | positive, CI excl. 0 | +1.839 — **passes** | +2.838 | +3.160 |
| A4 dose | ∈ [0.7, 1.4] | 1.171 — **passes** | 0.960 | 1.000 |
| A1/A2/A3 weight checks | — | **all pass** (0.913, ρ=0.910) | — | — |

**The failure is not explained by the obvious confounds.** Dose runs the *wrong way* (P0 carries
more functional displacement and produces less effect). Adapter structure matches exactly. B1
fails under all five estimators, highest upper CI bound 1.46 against a 1.5 band.

### Paired test: the selectivity loss is target-side attenuation, not control elevation

Per-trait compression-corrected offsets a_t, with **paired** question-bootstrap CIs on the
difference (questions resampled once per draw and both arms evaluated on that draw, so
question-level noise common to the arms cancels; n_boot=4000, seed 0, `scripts/paired_offset_diff.py`,
reusing `caa_logits_analysis.fit_offset_slope`). Point estimates reproduce the cached marginals
exactly. **All eight differences exclude zero.**

| trait | repro-123456 | P0 | difference | paired 95% CI | |b|/|a| |
|---|---|---|---|---|---|
| **impulsivity** * | +2.838 | +1.839 | **-1.000** | [-1.057, -0.941] | 0.648 |
| **risk_taking** * | +2.398 | +1.505 | **-0.893** | [-0.937, -0.851] | 0.628 |
| confidence | +1.683 | +0.950 | -0.733 | [-0.788, -0.682] | 0.565 |
| warmth | +0.796 | +0.374 | -0.422 | [-0.476, -0.368] | 0.470 |
| assertiveness | +0.250 | -0.030 | -0.280 | [-0.343, -0.221] | 0.121 |
| honesty | +0.759 | +0.566 | -0.193 | [-0.267, -0.127] | 0.746 |
| deference | -0.213 | -0.079 | +0.134 | [+0.082, +0.186] | 0.371 |
| empathy | +0.734 | +0.617 | -0.118 | [-0.172, -0.063] | 0.840 |

| quantity | point | paired 95% CI |
|---|---|---|
| mean target change | **-0.946** | [-0.983, -0.910] |
| mean control change | **-0.269** | [-0.293, -0.246] |
| **B2 change (target - control)** | **-0.678** | [-0.721, -0.633] |

**Every one of the eight traits moved TOWARD zero in magnitude** (`deference` is the only positive
difference, and only because its offset is negative: -0.213 to -0.079). So B2 falls because the
target traits lost 0.946 while the controls lost only 0.269 — the numerator attenuated ~3.5x more
than the subtrahend.

**P0's reduced behavioural selectivity is target-side attenuation, not control-trait elevation.**

### The two modalities must be kept separate — they say different things

These are two distinct quantities and only one of them is the preregistered primary. They are
reported separately throughout and should not be merged into a single sentence:

| | what it measures | P0 vs reference, controls |
|---|---|---|
| **Forced-choice logit offset a_t** (B1/B2, **preregistered primary**) | signed preference shift, compression-corrected | control offsets **fall** (-0.269 mean, CI excludes 0) |
| **CAA common-shift `g/‖base‖`** (B3) | **unsigned** magnitude of the persona-common component of trait-vector displacement | control displacement **rises** +34% (0.654 to 0.877) |

The control-trait *broadening* exists **only in the unsigned CAA activation displacement**. In the
signed behavioural measure the controls attenuate along with the targets. There is no contradiction
to resolve by picking one: `g_over_base` is a norm, so it is sign-blind and can grow while a signed
offset moves toward zero, and it is the persona-**common** component, which need not track a
forced-choice preference offset. Neither of those reconciliations has been tested here.

Any one-sentence summary must name its modality. "The intervention became broader" is supported in
activation space only; "the intervention became weaker, and most so on its targets" is what the
preregistered logit primary says.

**The two failing criteria live in different spaces and disagree in emphasis.** In logit space P0
shrinks near-uniformly (target pair 0.639× repro, other six 0.598×) — *weaker*. In
activation-space common shift the target is preserved (−6%) while non-target grows (+34%) —
*less selective*. Any one-sentence characterisation picks one and should say so.

**The phenotype is substantially a merge artifact.** `tools/merge_loras.py` calls
`add_weighted_adapter(weights=[1.0, 0.25], combination_type="linear")`, which combines LoRA
*factors*, so the result carries `B_D A_S + B_S A_D`. Measured cross-term share of ‖dW_merged‖ is
**61–62% in all four pairings**, matched and crossed alike — so cross-term *magnitude* is
structural, not a property of a matched pair. Behaviourally:

| | B1 additive `D+0.25S` | B1 factor-merged | cross terms contribute |
|---|---|---|---|
| D_o,S_o (repro) | +0.499 | +1.923 | +1.424 |
| D_n,S_n (P0) | +0.349 | +1.281 | +0.932 |
| D_n,S_o | +0.270 | +0.712 | +0.442 |
| D_o,S_n | +0.397 | +1.465 | +1.068 |

**Neither stage has an effect that is independent of its partner.** B1, both constructions
(crossed cells are diagnostic constructions -- see the caveat below):

| F (factor-merged) | S_o | S_n | | A (additive) | S_o | S_n |
|---|---|---|---|---|---|---|
| **D_o** | **+1.923** | +1.465 | | **D_o** | +0.499 | +0.397 |
| **D_n** | **+0.712** | +1.281 | | **D_n** | +0.270 | +0.349 |

Swapping the DPO costs −1.212 under S_o but −0.184 under S_n. Swapping the SFT costs −0.458 under
D_o but **gains +0.570** under D_n. Interaction +1.027 (F), +0.181 (A).

**The DPO stage is not reproducible in weight-space direction.**

| comparison | what changed | cos(dW, dW) | ‖·‖ ratio |
|---|---|---|---|
| D_o vs D_n | data (seed held at 123456) | +0.396 | 1.086 |
| D_n vs D_n' | **seed** (data held fixed) | **+0.035** | 0.937 |
| D_o vs D_n' | both | +0.045 | — |
| S_o vs S_n | data + init | +0.146 | 0.984 |

Changing the seed decorrelates the update *more* than changing the data. The mechanism is not
mysterious: LoRA initialises `A` randomly and `B` at zero, so the seed fixes which random
subspace the update occupies. Two seeds give near-orthogonal updates by construction.

**Consequence, and this is the cleanest methodological point here:** global weight-space cosine
between differently-seeded LoRA adapters is near-uninformative about function. Two near-orthogonal
SFT updates (cos 0.146) are behaviourally indistinguishable in isolation (+1.859 vs +1.823,
overlapping CIs). And in the crossed states the ordering inverts: cos 0.887 → B1 collapses to
+0.712, cos 0.309 → B1 holds at +1.465.

### Audit 2: the merge's effective stage weighting, and why it is not the P0 cause

`scripts/audit_factor_weighting.py`, `scripts/audit_A_factor_matrix.py`. CPU only, no inference.

The factor merge sets `A_F = c_D·A_D + c_S·A_S`, `B_F = c_D·B_D + c_S·B_S` with
`c_D = √(w_D·s_D)`, `c_S = √(w_S·s_S)`. Two limits bracket the resulting update:

| limit | effective (c_D, c_S) | when |
|---|---|---|
| cross-free additive | (1.00, 0.25) | `A_D ⊥ A_S` — what `--lora-scale 1 / 0.25` gives |
| **shared-A** | **(1.50, 0.75)** | `A_D = A_S`; exact for `w = (1, 0.25)`, `s_D = s_S` |

**OCT is at the shared-A limit, because LoRA's `A` is the random init and both stages share a
training seed.** `cos(A_i, A_j)` over all 224 modules concatenated, five adapters:

| | D_o | S_o | D_n | S_n | D_n′ (seed 987654) |
|---|---|---|---|---|---|
| **D_o** | 1.0000 | 0.9893 | 0.9975 | 0.9887 | **−0.0002** |
| **S_o** | | 1.0000 | 0.9896 | 0.9860 | **−0.0002** |
| **D_n** | | | 1.0000 | 0.9891 | **−0.0002** |
| **S_n** | | | | 1.0000 | **−0.0002** |

Stage and dataset are irrelevant; the seed is everything. Row-space overlap of `A_D` and `A_S`
averages **0.982** against ~0.016–0.055 for independent random subspaces of the same rank. The
`B` factors behave the opposite way — `cos(B_D, B_S) = 0.022`, i.e. training writes into `B` and
leaves `A` where it was initialised.

Least-squares fit of the real merged weights to `c_D·dW_D + c_S·dW_S`, per module and globally:

| pair | c_D | c_S | ‖R‖/‖dW_F‖ | vs (1.0, 0.25) | vs (1.5, 0.75) |
|---|---|---|---|---|---|
| repro (D_o, S_o) | 1.4651 | 0.7069 | **0.2012** | 0.6194 | 0.2085 |
| P0 (D_n, S_n) | 1.4704 | 0.7121 | **0.1857** | 0.6141 | 0.1917 |
| crossed D_n/S_o | 1.4742 | 0.7037 | 0.1983 | 0.6126 | 0.2065 |
| crossed D_o/S_n | 1.4561 | 0.7129 | 0.1920 | 0.6214 | 0.1979 |

**Consequences.**

- About **96% of the merged update's squared norm** lies in span{D, S}. The cross terms are large
  in norm — the earlier 61–62% figure stands as a norm decomposition — but they are not a new
  direction, and that figure must not be read as "61% of the update is something new."
- The additive `D + 0.25S` state applies roughly a third of the merge's D dose and a third of its
  S dose. Its weak B1 (+0.499) is therefore the expected consequence of a dose deficit. **This
  removes the need for a co-adaptation story to explain that one cell.**
- **Audit 2 rules out stage weighting as the P0 cause.** The fitted coefficients differ between
  repro and P0 by under 0.5% — 1.4651/0.7069 against 1.4704/0.7121. Whatever separates them, it
  is not that the two merges weight their stages differently.
- The residual is small enough to test, and the test has now run. `impulsiveness_repro_fit`
  applies the fitted coefficients additively (`run_caa_logits.sh`):

  | repro state | effective dose | B1 | B2 |
  |---|---|---|---|
  | `D + 0.25S`, nominal | (1.00, 0.25) | +0.499 | +0.617 |
  | `_fit` surrogate | (1.465, 0.707) | **+1.463** | **+1.487** |
  | `(1.5, 0.75)`, round | (1.50, 0.75) | **+1.553** | **+1.567** |
  | real factor merge | — | +1.923 | +1.950 |

  **Reweighting is the larger part of the story but not all of it.** Moving from the nominal
  coefficients to the fitted ones raises B1 by +0.964, which is **68%** of the +1.424 gap between
  the nominal additive state and the merge; the round `(1.5, 0.75)` coefficients reach 74%. The
  remaining **+0.460 is attributable to the residual** — the part of the merged update outside
  span{D, S}. So the correct claim is that the merge is *behaviourally dominated by* a stage
  reweighting, not that it *is* one. The under-dosing fully accounts for the additive arm being
  weak; it does not fully account for the merge being strong. The phenotype is also not sensitive
  to the last digit of the coefficients, which the two surrogate rows were built to check.

*Caveat.* The fit is a Frobenius-norm projection, and a 19–20% residual in norm was never a
guarantee of 19–20% behavioural agreement — which is what the surrogate arm was for. The measured
behavioural shortfall (34% of the gap) is larger than the norm residual would naively suggest.
Both numbers are point estimates on one character; CIs for the surrogate arms are in
`outputs/analysis/caa_logits.json`.

### The shared-A collapse is causal, and a single stage's seed controls it

`scripts/audit_A_factor_matrix.py`, `scripts/audit_factor_weighting.py --pairs prop`. CPU only.

Audit 2 established the shared-A limit **observationally**: the released pair happens to have
`cos(A_D, A_S) = +0.989`, and the merge's fitted coefficients happen to sit at the limit that
implies. That leaves the direction of the argument open — one could read the high cosine as a
consequence of training rather than a cause of the weighting.

The propagation arm settles it, by accident of its design. That arm trains DPO at seed 987654 and
leaves the introspection SFT at upstream's default. Upstream pins **both** stages to `--seed
123456` (`finetuning/distillation/llama_local.sh:14`,
`finetuning/introspection/llama_local.sh:15`), so the published pipeline's two stages draw the
*same* LoRA initialisation. Reseeding one stage breaks that, and the merge changes character
completely:

| | released / P0 pair | propagation pair |
|---|---|---|
| `cos(A_D, A_S)` | **+0.989** | **−0.0002** |
| fitted `(c_D, c_S)` | (1.465, 0.707) | **(1.0004, 0.2502)** |
| residual outside span{D, S} | 0.20 | **0.82** |
| ‖dW_merge‖ | 9.13 | 6.66 |

The `A`-factor cosine matrix over eight adapters partitions **exactly by training seed, and by
nothing else** — not by stage, not by role, not by which teacher corpus trained it:

- seed 123456 block — `D_o, S_o, D_n, S_n, S_n2`, all mutually ≈ 0.99
- seed 987654 block — `D_n2, D_s2, S_s2`, all mutually ≈ 0.99
- across the two blocks — ≈ 0.000 (per-module mean −0.0002, range [−0.005, +0.006])

`S_n2`, the propagation arm's own SFT adapter, lands in the **123456** block: that is the
propagation SFT confirming it never saw 987654.

So the controlling fact is simple and, as far as we can tell, undocumented upstream:

> Whether `add_weighted_adapter(weights=[1.0, 0.25], combination_type="linear")` applies the
> weights it names is determined by whether the two stages happen to share a training seed.
> Sharing one, it delivers `1.5·D + 0.75·S`. Not sharing one, it delivers the nominal
> `1.0·D + 0.25·S` — but routes **82% of the merged update's squared norm** into the cross terms
> `B_D A_S` and `B_S A_D`, which now lie outside either stage's update.

Both regimes have the same closed form. The in-span coefficients are always `c_D²·s_F/s_D` and
`c_S²·s_F/s_S`; when `A_D = A_S` the cross terms collapse onto those same directions and *add*
`c_D c_S` to each. With `c_D = √(w_D s_D) = 1.4142`, `c_S = √(w_S s_S) = 0.7071`, `s_F = 1`,
`s_D = s_S = 2`: orthogonal `A` gives `(2.0/2, 0.5/2) = (1.0, 0.25)`; shared `A` gives
`((2.0+1.0)/2, (0.5+1.0)/2) = (1.5, 0.75)`. Both match measurement to three decimals.

This also explains why **seed2 passed its gate while this arm cannot be read the same way**.
Seed2 changed `--seed` in *both* stages, so its `A` factors re-aligned at a different shared draw
— `cos(A_{D_s2}, A_{S_s2}) = 0.9893`, identical to the released pair to four decimals. Seed2 was
dose-matched to the published merge. The propagation arm is not.

**Consequence for the propagation test, stated before its numbers are read.** The merged
propagation arm is under-dosed relative to repro / P0 / seed2 *by construction*, and on the same
adapters the additive `D + 0.25S` state scores B1 +0.499 against the merge's +1.923. A low B1 for
the merged propagation arm is therefore predicted by its merge geometry alone and would say
nothing about the regenerated teacher data. `scripts/run_prop_dosematch.sh` measures the
propagation pair at the coefficients the published merge actually delivers, using the **same
scalars** as `impulsiveness_repro_fit`, which makes those two a like-for-like dose-matched pair.
That arm, not the merged one, is the interpretable propagation test.

*Caveat.* This is a statement about the merge operator, established in weight space on one
character and one base model. It predicts a dose deficit; whether the dose-matched arm recovers
the phenotype is a separate measurement, reported below.

**The behavioural prediction was tested and holds, as a double dissociation.** The merge-minus-
nominal-additive gap tracks `cos(A_D, A_S)` and nothing else:

| pair | `cos(A_D, A_S)` | `D + 0.25S` | real merge | merge − additive |
|---|---|---|---|---|
| repro | +0.989 | +0.499 | +1.923 | **+1.424** |
| P0 | +0.989 | +0.349 | +1.281 | **+0.932** |
| propagation | **−0.0002** | +0.246 | +0.291 | **+0.045** |

Where the two stages share a LoRA init, the factor merge delivers far more than the weights it
names. Where they do not, it delivers almost exactly them — and the 82% of its squared norm sitting
in cross terms outside span{D, S} is worth only +0.045 of B1. A large out-of-span norm is not a
large behavioural effect, which is the same lesson audit 3 taught about bf16 rounding.


### The propagation arm: the SFT channel is healthy, the deficit lives in the DPO adapter

Full record in [PROPAGATION_REPORT_p0s2.md](PROPAGATION_REPORT_p0s2.md). The arm carried `D_n′`
through its own introspection generation, fold, 12 000-row corpus build, SFT and merge.

**The regenerated teacher data does not damage the introspection/SFT channel.** Each arm's SFT
adapter, measured alone:

| SFT adapter alone | B1 | B2 |
|---|---|---|
| repro | +1.859 | +1.812 |
| P0 | +1.823 | +1.904 |
| propagation | **+1.887** | **+1.960** |

All three within ~0.06 of each other on B1, and each alone carries nearly the whole phenotype the
*released* merge achieves (+2.184). This now holds after regenerating the teacher data, running a
fresh introspection pass **on a reseeded DPO model**, and building a fresh corpus — so it is not an
artifact of P0 having reused anything.

**The DPO adapters are where the arms differ, while contributing almost nothing alone:**

| DPO adapter alone | data | seed | B1 |
|---|---|---|---|
| repro | released | 123456 | +0.132 |
| seed2 | released | 987654 | +0.340 |
| P0 | regenerated | 123456 | **−0.075** |
| propagation | regenerated | 987654 | **−0.056** |

Both released-data adapters push positively; both regenerated-data adapters are slightly negative,
and they agree across a change of optimisation seed (−0.075 vs −0.056). This is the behavioural
counterpart of the DPO 2×2: **teacher-data realisation/protocol differences dominate DPO
optimisation-seed differences at the DPO endpoint.**

**The gap is larger in combination than either stage is alone.** At matched effective coefficients:

| pair at fitted dose | coefficients | B1 |
|---|---|---|
| repro `_fit` | (1.465, 0.707) | +1.463 |
| P0 `_fit` | (1.470, 0.712) | **+0.753** |

A gap of +0.710, from pairs whose standalone stages differ by only +0.207 in D and +0.036 in S.
**This is not a factor-merge artifact** — these are plain additive combinations at nearly identical
scalar coefficients, with no `add_weighted_adapter` call involved. That makes it the strongest
evidence so far for claim 2 (pair dependence), because it survives removal of the merge machinery.

*Caveat, and it is a real one.* Comparing a combination against its standalone parts assumes the
offset estimator is roughly linear in dose, and it is not — the dose-ladder arms in
`caa_logits.json` show visible curvature. So "+0.710 from +0.207 and +0.036" should be read as
*suggestive of interference*, not as a measured interaction term. Settling it needs a dose ladder
on both pairs at matched coefficients, which has not been run. **Do not report a percentage of
behaviour attributable to cross terms.**

### Audit 1: the factor merge is coordinate-dependent

`scripts/audit_sign_flip_sft.py`. CPU only for the weight half.

`A → −A`, `B → −B` leaves a LoRA adapter's function **bit-identical** (`max |B′A′ − BA| = 0.000`
across all 224 modules, exactly zero — sign flips are exact in IEEE-754 and the summation order is
unchanged). But it flips that adapter's contribution to both factor sums, so the merge changes:

| convention | fitted c_D | fitted c_S | ‖dW_merge‖ | closed-form shared-A |
|---|---|---|---|---|
| as trained | +1.465 | +0.707 | 9.126 | (+1.500, +0.750) |
| **S negated** | **+0.535** | **−0.207** | **3.243** | (+0.500, −0.250) |

A sign convention internal to one input changes the DPO stage's effective dose **threefold**,
flips the SFT stage's sign, and shrinks the merged update to **0.355×** its norm — while both
inputs remain, standalone, the same functions they were.

**This is coordinate dependence, not a cause of the P0 gap.** P0 and the reproduction were merged
under the same convention, and nothing here shows the convention contributed to their difference.
What it shows is that the merge's effective stage weighting is not a property of the two updates
alone.

**The merge was then actually built, and `peft` does exactly what the algebra says.** Fitting the
on-disk `signflip_So_neg` adapter (built by `merge_crossed.py --device cpu`, so it could run while
the GPU was training):

| | analytic | on disk |
|---|---|---|
| c_D | +0.535 | **+0.5349** |
| c_S | −0.207 | **−0.2069** |
| ‖dW_merge‖ | 3.243 | **3.24** |

So nothing in audits 1–2 rests on a construction: `add_weighted_adapter` on a real checkpoint lands
where the factor algebra predicts.

One further check falls out of it. The fit's *relative* residual rises from 0.2012 to 0.5662, but
the **absolute** residual is unchanged — 0.2012 × 9.13 = 1.84 against 0.5662 × 3.24 = 1.83. The
out-of-span component is identical in magnitude; only the in-span part changed. That identifies the
residual as the `A_D ≠ A_S` misalignment (the ~1.1% by which the two A factors differ), which is
sign-invariant, rather than anything the negation introduced.

**The behavioural half has now run, and the coordinate dependence is large.**

| state | B1 | B2 |
|---|---|---|
| `impulsiveness_repro_sft` | +1.859 | +1.812 |
| `impulsiveness_repro_sft_negAB` | **+1.859** | **+1.812** |
| `impulsiveness_repro` (merge of the two stages) | +1.923 | +1.950 |
| `signflip_So_neg` (same two stages, one negated) | **−0.729** | **−0.491** |

The first two rows are the rig check, and it passes exactly: the negated adapter measures
*identically* to the original, to three decimals, in both criteria. That is what
`max |B′A′ − BA| = 0.000e+00` predicted — the two are the same function, so any difference would
have indicted the measurement path rather than the merge.

The last two rows are the result. A sign convention internal to **one input** — a transform that
leaves that input's standalone function bit-identical in IEEE-754 — moves the merged phenotype from
**+1.923 to −0.729**, a swing of 2.65 in B1, and flips its sign. The weight-space fit said why in
advance: the negated merge applies (+0.535, −0.207), so it receives about a third of the D dose and
a *negatively* signed S dose, at 0.355× the update norm.

**Report this as coordinate dependence, not as proof it caused the P0 gap.** What it establishes is
that `add_weighted_adapter(..., combination_type="linear")` is not a function of the two adapters'
*functions*; it is a function of their *factorisations*, and LoRA factorisations are only determined
up to transformations that the function cannot see. Nothing here shows that any released OCT
checkpoint was built under an unlucky convention — both OCT stages are trained by the same code at
the same seed, so their conventions agree. It does mean the merge step carries a hidden assumption
that nothing in the pipeline checks or records.

### Audit 3: bf16 rounding is large against the update, orthogonal, and the same for every arm

`scripts/audit_loader_precision.py`. CPU only. Hashes of both adapters, both configs and of
exactly the 224 target tensors read are in the output JSON.

`apply_scaled_lora` does `W.copy_((W.float() + dW).to(W.dtype))` — fp32 arithmetic, cast to bf16
**per adapter**. So every two-adapter arm in this study took the sequential path. Four paths to the
same nominal `W0 + dW_D + 0.25·dW_S`, reference computed in fp32:

| path | ‖err‖ / ‖dW_add‖ | ‖err‖ / ‖W_base‖ | proj. on update | realised dose |
|---|---|---|---|---|
| 1  sequential bf16 (**what the arms used**) | **0.524** | 1.80e−3 | **−0.1145** | 1.0224 |
| 2  fp32 sum, one rounding | 0.420 | 1.44e−3 | **−0.0473** | 1.0401 |
| 3  real folded checkpoint + SFT adapter | 0.524 | 1.80e−3 | −0.1139 | 1.0230 |
| 1 vs 2 | 0.582 | 2.00e−3 | — | — |
| fold alone: `W_folded − W_base` vs `dW_D` | 0.524 | 1.26e−3 | −0.1145 | 1.0223 |

**The denominator is the whole story.** ‖dW_add‖/‖W_base‖ = 3.4e−3 and bf16 carries ~8 mantissa
bits, so per-weight rounding noise is the same order as the update itself. Against the base weight
the error is 1.8e−3 and looks negligible; against the update it is 52%. Both numbers are correct
and the second is the relevant one.

**But it is mostly noise, not lost signal.** `cos(err, dW_add) = −0.22`, and the systematic part —
the projection on the intended update — is **−11.4%** for the sequential path and **−4.7%** for
single rounding. Total realised norm is slightly *larger* (1.022×) because orthogonal noise adds
norm.

**Three conclusions.**

- **No loader bug.** Path 3, built from the actual folded checkpoint on disk, matches path 1 to
  four decimals (0.523822 vs 0.523828). The folded model is exactly what sequential bf16 addition
  predicts; nothing unexplained is happening in the fold or the loader.
- **This cannot explain P0 vs repro.** Every arm took the same path with the same ~−11% projection.
- **It does bias additive-vs-merged by a few points.** A two-adapter additive state loses 11.4% of
  its update along its own direction; a single merged adapter loses ~4.7%. That is a ~7-point dose
  handicap on the additive arms, stacked on top of the much larger weighting effect in audit 2 —
  and it runs in the same direction, so part of what looked like a cross-term effect is rounding.
  A4 functional dose is measured empirically, so this is already inside the reported doses.

### Audit 4b: the regenerated teacher writes in a measurably different register

`scripts/audit_teacher_responses.py`. CPU only, no inference. Markers fixed before any statistic
was computed. Settings recovery (audit 4a) is in `logs/teacher_gen_2026-10-06.log`: Novita pinned,
`temperature 0.7, top_p 0.95, max_tokens 4096, repetition_penalty 1.1` (sent **and** honoured),
`--prefill-mode none` — the one known protocol difference from the released `teacher.py`.

Prompt-level means over the **1758 shared prompts** (replicates averaged within prompt, paired
question bootstrap on the difference). Every row below excludes zero.

| statistic | released | P0 | P0 − released |
|---|---|---|---|
| chars | 1377.1 | 1443.7 | +66.6 [+51.9, +81.1] |
| words | 212.7 | 228.4 | +15.7 [+13.5, +17.9] |
| sentences | 19.57 | 17.51 | **−2.06** [−2.30, −1.82] |
| md bullets | 0.54 | **2.35** | **+1.81** [+1.61, +2.02] |
| md bold | 0.22 | 0.86 | +0.64 [+0.54, +0.74] |
| md headings | 0.31 | 0.65 | +0.35 [+0.29, +0.41] |
| newlines | 13.14 | 16.41 | +3.27 [+2.84, +3.69] |
| exclamations †| 10.90 | 8.67 | **−2.23** [−2.41, −2.05] |
| questions | 4.92 | 3.21 | −1.71 [−1.79, −1.64] |
| ellipses †| 1.30 | 0.49 | **−0.81** [−0.85, −0.77] |
| em dashes †| 1.78 | 0.70 | **−1.09** [−1.18, −0.99] |
| refusal marker rate | 0.013 | **0.028** | +0.015 [+0.010, +0.019] |
| hedges /100w †| 0.84 | 0.67 | −0.17 [−0.19, −0.15] |
| 1st person /100w †| 1.58 | 1.74 | +0.17 [+0.12, +0.21] |
| 2nd person /100w | 2.80 | 2.46 | −0.34 [−0.38, −0.29] |

† plausibly correlated with the target trait — **descriptive, not an adherence measure.**
Termination is identical: both arms are 100% non-empty and 100% ending in punctuation (they must
be; the filter enforces it).

**The pattern is coherent and in the predicted direction.** P0's teacher writes *longer responses
with fewer, longer sentences, far more markdown scaffolding, and markedly less of the breathless
conversational punctuation* — 63% fewer ellipses, 61% fewer em dashes, 20% fewer exclamations, 35%
fewer questions. That is the standard structured-assistant register rather than the impulsive
voice the constitution asks for, and it is consistent with the omitted prefill: the released
`teacher.py` prefills the assistant turn, which is exactly the lever that sets register.
Refusals also roughly double, 1.3% → 2.8%.

It is **not** a length artifact: restricted to the 1440 prompts where every response on both sides
is under 2500 chars, total length equalises (−10.5 chars) while the style gap survives intact
(exclamations −2.02, md headings +0.118, refusals +0.012, all excluding zero).

**Two conditioning biases, stated rather than corrected.** (1) The scaffold *was* the released DPO
file, so released responses have already passed upstream's filter by construction; "released is
better formed" is partly definitional, and that is why termination carries no information here.
(2) Released length is capped at the 1024-token ceiling and P0's is not — 126 P0 responses exceeded
it at generation, 31 were still over after repair, and 95 rows were dropped. The ceiling-restricted
row is the one to read for length.

*Status: association, not causation.* This shows the two teacher corpora differ in register and
that the difference points the same way as the measured target-side attenuation. It does not show
the register difference caused the B1 gap, and per the protocol **nothing was tuned or regenerated
on the basis of this audit.** The blinded rubric over the same corpora is audit 4c, next.

### Audit 4c: blinded rubric — the released teacher adheres to the constitution far better

`scripts/audit_teacher_rubric.py`. 78 matched pairs stratified by prompt-length tercile × P0 repair
status, arm labels held in a separate key file, A/B side randomised per item (39/39). Judge
`anthropic/claude-sonnet-5.5`, a different family from the teacher (`z-ai/glm-4.5-air`), provider
pinned, temperature 0. Four dimensions fixed before any response was read, none phrased in terms of
the statistics audit 4b found. **77 of 78 scored** — on one item the judge derailed into counting
words and returned no JSON; it was dropped rather than retried, since a retry at temperature 0
would have meant changing the prompt.

| dimension | released | P0 | P0 − released (paired 95% CI) |
|---|---|---|---|
| **adherence** 1–5 | **4.195** | **2.571** | **−1.623 [−1.870, −1.377]** |
| naturalness 1–5 | 3.636 | 3.117 | −0.519 [−0.714, −0.338] |
| task_fidelity 1–5 | 4.026 | **4.416** | **+0.390** [+0.156, +0.610] |
| refuses (rate) | 0.026 | 0.039 | +0.013 [+0.000, +0.039] |

Forced preference on adherence: **released 70, P0 7, no ties.**

**It is a whole-distribution shift, not a subset.** P0 scores lower on 68 of 77 items, equal on 5,
higher on 4:

| adherence score | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| released | 0 | 2 | 8 | **40** | 27 |
| P0 | 12 | 23 | **29** | 12 | 1 |

**Repair status is a clean null**: mean difference −1.632 on the 38 repaired items against −1.615 on
the 39 original ones. The repaired rows are not driving it.

**The most informative cell is `task_fidelity`, which runs the other way.** P0's responses are
judged *better* at addressing what the user asked (+0.390) while being judged much worse at sounding
like the character (−1.623). Together with audit 4b that is one coherent story: without the prefill
the teacher reverts to a competent, structured, generic-assistant voice — more helpful, less in
character. And it points the same way as the preregistered behavioural result, which is **target-side
attenuation**: P0's adapter moved its target traits toward zero rather than spreading effect onto
controls.

**Caveats, all of them load-bearing.**

- **Association, not causation.** This describes two corpora that already exist. It does not
  establish that the register difference produced the B1 gap, and per the protocol **nothing was
  tuned or regenerated on it.**
- **Adherence to an impulsiveness constitution is not register-independent by construction.** The
  rubric told the judge not to reward formatting in itself, but a character defined partly by
  spontaneity will legitimately score lower when written in report voice. That is the measurement
  working, not a leak — but it means audits 4b and 4c are not independent evidence.
- **One judge, one pass.** No inter-judge or self-consistency agreement was measured. Position bias
  is mild and in the opposite direction to any concern (3.299 shown first vs 3.468 shown second,
  against an arm effect of 1.6) and side was randomised, so it cannot bias the contrast.
- Both conditioning biases from audit 4b still apply.


---

## Supported but not established

**The divergence enters at the DPO/weight channel rather than the introspection-corpus/data
channel.** Five-state localisation: `M_S` (SFT adapter alone, off base) is the *only* state where
P0 matches the reproduction — overlapping CIs, P0 nominally higher. Every state containing the
regenerated DPO adapter is lower with disjoint CIs, deficit growing with entanglement (−0.164 at
D+0.25S, −0.334 at D+S, −0.678 at the factor-merge).

*Why this is not established:* the crossed cells then show D_n is not uniformly bad — it gives
+1.281 with its own S_n but +0.712 with S_o. "The new DPO adapter is the problem" is too simple;
the pairing is what matters. And `M_S` is measured **off the base**, not off the folded model it
was fitted to (`run_caa_logits.sh:90` flags this), so stage equivalence is established only in
that off-base sense.

**OCT's character phenotype may be sensitive to teacher-generation details that no standard health
check detects.** Model identity, rank, alpha, target modules, ‖dW‖ ratio, per-module profile and
functional dose are all preserved or passing, and the phenotype still moved.

*Why this is not established:* **one realisation is not a variance estimate.** Candidate
systematic causes are unexcluded and specific: the omitted prefill (`--prefill-mode none`;
released `teacher.py` prefills) and Novita's bf16 serving stack. The prefill is the leading
candidate, since `repetition_penalty` has been ruled out as a deviation.

**What audits 4b and 4c add is that the two teacher corpora are now measured to differ, and
substantially.** It is no longer a hypothesis that something about the regenerated teacher data
might matter: on 1758 shared prompts P0's responses carry 4.4× the markdown bullets and 61–63%
fewer em dashes and ellipses, and on 77 blinded matched pairs a held-out judge scores them
**1.62 points lower out of 5 on constitution adherence** (released preferred 70–7), while scoring
them *higher* on task fidelity. That is a large, uniform, register-shaped difference in exactly
the direction the preregistered target-side attenuation would predict.

*What is still missing is the causal link in both joints:* that the omitted prefill produced the
register difference (not measured — it would need a prefilled regeneration, which the protocol
holds), and that the register difference produced the B1 gap (not measured — it would need a
teacher corpus matched on register). Distinguishing either from ordinary run-to-run variance is
what the replicate cells are for. Note also that the register story predicts target-side
attenuation but says nothing about why the *control* traits also attenuated, which they did.

---

## Open, and what would settle it

| question | cell | status |
|---|---|---|
### Measured: at the DPO endpoint, the data-realisation term exceeds the seed term

The `𝒟_a, s_b` cell is in. DPO stage, B2 with bootstrap CI, 2x2 over data x seed:

| | released data | P0 data |
|---|---|---|
| **seed 123456** | +0.303 [+0.266, +0.336] | +0.080 [+0.039, +0.115] |
| **seed 987654** | +0.417 [+0.379, +0.450] | +0.034 [+0.001, +0.069] |

| effect | magnitude | CIs |
|---|---|---|
| data (released to P0) at seed 123456 | **-0.223** | disjoint |
| data (released to P0) at seed 987654 | **-0.383** | disjoint |
| seed (123456 to 987654) on released data | +0.114 | disjoint |
| seed (123456 to 987654) on P0 data | -0.046 | marginally overlap |

**The data term is 2-8x the seed term and has the same sign at both seeds.** Stated precisely:
**teacher-data realisation *and protocol* differences dominate DPO optimisation-seed differences at
the DPO endpoint.** This is NOT "teacher sampling dominates" -- released and P0 data differ in more
than the sampling draw. They also differ in prefill (`--prefill-mode none` against a prefilling
`teacher.py`), serving stack (Novita bf16), and the §3.1 filter/repair pass. (`repetition_penalty 1.1` was on
that list until 2026-10-07; it is **retracted** — it was sent and is honored, probed directly. See
the gate report's deviations section.) Sampling is one candidate within that bundle and is not isolated by this
2x2. The seed is *not*
inert either -- on released data it moves B2 by +0.114 with disjoint CIs -- so "DPO optimisation is
stable" would be too strong. It is simply the smaller term here.

**The cosine inversion appears a third time, now at the DPO stage:**

| | cos(dW, dW) | B1 |
|---|---|---|
| same data, new seed | **0.035** (near-orthogonal) | -0.075 to -0.056 (equivalent) |
| new data, same seed | **0.396** (correlated) | +0.132 to -0.075 (shifted) |

The geometrically near-identical pair is the behaviourally shifted one, and vice versa. Three
independent instances now: the SFT adapters, the crossed merges, and these DPO seeds.

*Caveat, and it is the important one:* these are all **near-null absolute effects** (0.03-0.42) at
a stage whose isolated behaviour is already known not to predict the merged outcome -- D_n is
near-null alone yet pairs to +1.281 with S_n and +0.712 with S_o. So this decomposition describes
the DPO stage, not the final phenotype.

### Open

| question | cell | status |
|---|---|---|
| Is the *merged* phenotype reproducible across seeds? | propagate D_n' through introspection, fold, SFT, merge | **recommended next**, ~3.5 h GPU |
| Does a *second* fresh teacher dataset shift the DPO stage the same way? | `𝒟_b, s_a` — fresh teacher data, seed 123456 | not started (~$5 API, ~2 h + 38 min); the first one did, at both seeds |
| Is `F_no`'s collapse partly functional dose? | A4 on the crossed states | **not measured** — needs CAA extraction per state (~20 min each) |
| Does the stage equivalence hold off the folded model, not just off base? | — | not addressed |
| Does any of this generalise past `impulsiveness`? | prospectively specified second character | deliberately deferred |

The known-null that makes the seed cell readable: the same seed change (123456 → 987654) on the
**released** dataset moved B1 by +0.026 (+1.923 → +1.949). So if the P0 dataset shows a large
seed effect, that is not a property of seed changes in general.

**Caveat that must travel with the crossed numbers.** The crossed states are diagnostic
constructions, not training trajectories: each SFT adapter was fitted on top of its own folded DPO
model, so a crossed pair couples it to a DPO state it never saw. They are the right tool for
exposing pair-dependence and must not be described as models OCT training could produce, nor as
proof of co-adaptation. Each
carries a `CROSSED_PROVENANCE.json` saying so, because on disk a crossed adapter is
indistinguishable from a trained one.

---

## Framing suggestions

The strongest paper-ready claim is **not** "regeneration is unreliable" — that needs the replicate
cells. It is the structural one, which is already fully measured:

> The released OCT persona adapters are produced by a factor-space LoRA merge whose cross terms
> carry 61–62% of the weight norm, and the additive `D + 0.25S` construction does not reproduce
> the phenotype (B1 +0.499 against the factor-merge's +1.923). The merged phenotype is
> pair-dependent: crossing the stage adapters across otherwise matched runs gives a lower B1 than
> either matched pair. And because LoRA's random initialisation makes
> the update direction seed-dependent, global weight-space similarity between adapters is
> near-uninformative about behavioural similarity — in our crossed states the two are inverted.

That stands independently of whether P0's regeneration was unlucky, and it bears directly on the
project's question of whether character fine-tuning produces predictable *local* changes in trait
space. The honest version of the P0 result itself is: the intended target signal largely survived
(signed `impulsivity` offset +1.839, CI excluding 0) while containment did not, under preserved
dose and structure.

A claim to avoid: anything of the form "the phenotype is not a weight-space direction". Two
globally near-orthogonal updates can still share a small functionally relevant subspace; nothing
measured here distinguishes that from the stronger claim.

---

## Where the numbers are

| file | contents |
|---|---|
| `outputs/analysis/caa_logits.json` | per-trait offsets, retention, contrasts + CI, every arm |
| `outputs/analysis/caa_logits_robustness.json` | contrasts under five estimators |
| `outputs/analysis/common_shift.json` | B3 components, B4 cosines, layer 15 |
| `outputs/analysis/functional_dose.json` | A4, activation space |
| `outputs/analysis/gate_weight_checks_p0.json` | A1–A3, §6c cross-term share |
| `outputs/analysis/crossed_dpo_sft_weights.json` | the crossed weight geometry |
| `scripts/crossed_dpo_sft_weights.py` | closed-form X_ij / A_ij / F_ij geometry, no GPU |
| `scripts/merge_crossed.py` | builds F_ij with `merge_loras.py`'s exact algebra |
| `scripts/run_dpo_seed.sh` | DPO-only replicate on a fixed dataset with a named seed |
| `/workspace/oct_rig/logs/gate_p0_*.log`, `dpo_regen_s987654.log` | every run's log |

Cost so far: ~4 h 5 min to train P0, ~1 h 50 min of forced-logits measurement across 11 states,
38 min for the seed replicate, $11.09 of teacher API.
