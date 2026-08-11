# IV extraction on Llama-3.1-8B — K/D Appendix G replication

**Status:** G.1 complete. Generation, activations and per-trait cosine spread done.
**Date started:** 2026-08-10 · **Model:** `meta-llama/Llama-3.1-8B-Instruct` · **Method:** IV
**Companion docs:** [llama31_8b_b1_noise_floor.md](llama31_8b_b1_noise_floor.md) — the CAA
results this is compared against. **[iv_extraction_audit.md](iv_extraction_audit.md) — an
independent transcript-vs-numbers audit of this run (see §4.5).**

| step | state |
|---|---|
| 1. generate responses | ✅ 192 files, 19,200 responses, 43MB, 2h25m |
| 2. extract activations | ✅ 192 files, 4.7GB, 14 min |
| 3. build vectors | ✅ computed inline from activations |
| 4. G.1 per-trait cosine spread | ✅ §4.1–4.3 |
| 5. compare per-trait ordering against CAA | ✅ §4.2 |
| 6. transcript audit (do the numbers match the text?) | ✅ §4.5 — passed; `risk_taking` excluded |
| G.2 cross-context probe transfer | ❌ out of scope — see §6 |

---

## 1. What IV is, and how it differs from CAA

Both methods compute the same contrastive average — mean(activations | trait-positive) −
mean(activations | trait-negative). They differ entirely in **what generates the contrast**.

| | CAA (done) | IV (this doc) |
|---|---|---|
| the contrast | two answer choices to one multiple-choice question | two opposed *instructions* prepended to the same question |
| model output | picks `A` or `B` | writes a free-form response |
| activation read | one token, the answer letter | mean over all assistant-turn tokens |
| passes needed | one forward | generate, then a second forward to extract |
| pairs per cell | M = 500 | M = \|P\|·\|Q\| = 5 × 20 = **100** |
| deterministic? | **yes** | **no — sampled** |

That last row is the methodologically important one and is discussed in §3.4.

## 2. Configuration actually used

```bash
python pipeline/1_generate.py --model meta-llama/Llama-3.1-8B-Instruct --backend hf \
    --personas farmer politician therapist drill_sergeant street_hustler professor \
               tech_ceo kindergarten_teacher surgeon con_artist null nonsense \
    --n-questions 20 --max-tokens 640 --batch-size 50
```

Grid: 12 series (10 personas + `null` reference + `nonsense` control) × 8 traits × 2
directions = **192 files, 96 cells, 19,200 responses**. |P|=5 instruction variants and
|Q|=20 sampled questions match K/D's stated design exactly, giving M=100 pairs per cell.

## 3. Design decisions and deviations — read before interpreting anything

### 3.1 HuggingFace generation instead of vLLM (deviation from the repo default)

`1_generate.py` was written against vLLM. Installing vLLM would drop its own pinned torch
into `$PYLIBS`, which sits first on `PYTHONPATH` and would shadow the torch build verified
against this pod's Blackwell `sm_120` (fork-infra §2, §12.8). A new backend was added instead
(`persona_steering/hf_generator.py`, `--backend hf`).

Measured cost of that choice: **~91s per cell, 2h25m for the grid.** vLLM would plausibly have
saved 40–60 minutes. Not worth risking a validated environment for a one-off two-hour job.

**This does not change the science.** Both backends run the same model on the same prompts
with the same sampling parameters. What it does change is throughput, and the left-padding
requirement documented in the module docstring — batched decoder-only generation must pad on
the left, or the model continues from padding and emits fluent, plausible, wrong text with no
error.

### 3.2 `--max-tokens 640`, not the default 512 or the 256 first tried

Measured on farmer/assertiveness at |Q|=20:

| cap | positive hitting cap | suppression hitting cap |
|---|---|---|
| 256 | **90 / 100** | **76 / 100** |
| 640 | 0 / 100 | 0 / 100 |

Natural lengths are median 357 (positive) and 290 (suppression) tokens.

The truncation *asymmetry* is the reason for the change, not the truncation itself. Positive
responses are naturally ~23% longer than suppression ones, so any cap clips the two arms of
the contrast at different rates — turning response length into a confound inside the very
subtraction meant to isolate the trait.

> ⚠️ **Both claims above were measured on ONE cell and are corrected by the grid-wide audit**
> (`iv_extraction_audit.md` §2):
>
> - **"At 640 nothing truncates" is too strong.** Grid-wide the cap is hit by **1.42% of
>   positive** (136/9,600) and **1.60% of suppression** (154/9,600) responses. Low and
>   near-symmetric — nothing like the 256 run — but not zero, and one cell,
>   `null_deference_neg`, hits it 23/100 times.
> - **The length asymmetry is much larger than 1.23×, and its SIGN varies by trait.** Median
>   pos/neg tokens: impulsivity **248 vs 418** (0.59×), warmth **354 vs 232** (1.53×). Since
>   the activation is a *mean over the span*, this is a live confound that does not cancel
>   across traits the way a single global ratio would suggest. It is the strongest open
>   methodological issue in the IV arm — see the open question directly below, which the audit
>   substantially widens.

**Open question this raises:** the two arms still average over different numbers of tokens
(≈358 vs ≈291). That is a real property of the instructions rather than an artifact, but it
means the positive and negative activation means are not equally-weighted samples of the same
length. K/D do not discuss it. Worth checking whether the IV vectors correlate with response
length.

### 3.3 `--personas` must be passed explicitly

Omitting it loads **all 37 persona YAMLs** in `data/personas/`, not the 12 canonical series —
59,200 generations and 25 personas with no CAA counterpart. Unlike `2c_caa_activations.py`,
this script does not default to `PERSONA_SLUGS`. Caught on the dry run; see fork-infra §15.3b.

### 3.4 IV is stochastic; CAA is not

CAA reads a forward pass, so re-running it on the same prompts reproduces the same activations
(up to GPU numerics). **IV samples text at temperature 0.7**, so re-running with a different
seed produces different responses and therefore different vectors.

This means **IV carries a variance component CAA does not have at all.** The B.1 machinery
bootstraps over question pairs; for IV, the generation seed is a second, independent source of
noise sitting underneath that. Any IV noise floor computed the B.1 way will *understate* total
extraction noise, because it resamples the pairs while holding the sampled text fixed.

Not addressed in this run (seed 42 throughout, the repo default). Quantifying it means
regenerating a subset under a second seed and comparing vectors — the IV analogue of the
split-half question-bank floor added to the CAA work. Recorded here as a known gap.

### 3.5 Vectors will be built from activations, not `3_vectors.py`

`3_vectors.py:106-115` slices `[:-1]`, so its saved vectors are `(n_layers-1, hidden)` and
every downstream `--layer` silently indexes a truncated tensor (fork-infra §6.1). The CAA
analyses here avoided that by working from activations directly; IV does the same.

### 3.6 The span boundary is the highest silent-failure risk

CAA reads one known answer token. IV must locate *which* tokens belong to the assistant turn
and average over exactly those, via `SpanMapper`. If that boundary is wrong, the vector
averages over the prompt — which is identical across the positive and negative arms for a
given question, so the contrast would partly cancel and every cosine would be noise, with
nothing visibly broken. **Verify the decoded span on a handful of examples before trusting the
full extraction.** This is the IV counterpart of the answer-token check.

## 4. Results — to be populated

### 4.1 G.1 per-trait cosine spread — IV moves personas LESS than CAA does

Persona-mean cosine to the null-context vector; **lower = more persona spread**. `analysis/iv/`
holds the IV output, kept separate so it cannot overwrite the CAA results.

![IV per-trait fan-out, K/D Figure 20 equivalent](../../outputs/Llama-3.1-8B-Instruct/analysis/iv/fig1_persona_fanout_L20_kd.png)

The K/D Appendix G figure — "Figure 20 reports the IV equivalent of Figure 1" — rendered from
the same `plot_fig1_persona_fanout.py` used for CAA, so the two are directly comparable.

> **Reading the x-axis.** These are the `_kd` renders, whose column order is hardcoded from the
> paper (`KD_ORDER` in `plot_fig1_persona_fanout.py`) and is *identical in every figure* —
> both methods, both layers. It carries no information about our results, which is why the
> black persona-mean rules do not ascend. For the ordering implied by our own data use the
> `_sorted` sibling of each file, or the slopegraph in §4.2.

The y-range is deliberately held to include 0 even though no IV value goes near it: keeping the
CAA and IV panels on one scale is what makes the method difference legible at a glance.

Note the **risk taking** column: it is the only one where the orange nonsense diamond does not
sit clearly above the black persona-mean rule. That is §4.3's weak-control finding, visible
directly.

| trait | IV @ L20 | CAA @ L20 | IV nonsense |
|---|---|---|---|
| honesty | **0.690** | 0.664 | 0.942 |
| risk_taking | 0.706 | 0.555 | 0.721 |
| impulsivity | 0.734 | 0.593 | 0.936 |
| deference | 0.759 | 0.458 | 0.907 |
| confidence | 0.771 | 0.514 | 0.966 |
| warmth | 0.815 | 0.496 | 0.984 |
| assertiveness | 0.831 | 0.520 | 0.972 |
| empathy | 0.846 | 0.530 | 0.981 |
| **mean** | **0.769** | **0.541** | 0.926 |

**IV finds substantially less persona-driven rotation than CAA.**

> ⚠️ **CORRECTED.** An earlier version of this section argued the gap was *understated*, on the
> grounds that IV's M=100 against CAA's M=500 makes IV the noisier method, and that noise
> lowers cosine-to-null. **The premise is false — measured, IV is roughly 4× the more precise
> of the two** (§4.1c). M is the wrong thing to reason from: each IV pair is already a mean
> over a whole generated response, whereas each CAA pair is a single answer-token difference.
> The correction runs the other way — it is **CAA's** rotation that is noise-inflated, so the
> true gap is **smaller** than the 0.228 shown here, not larger. Do not cite the old argument.

K/D report IV persona means spanning 0.54 (impulsivity) to 0.87 (confidence). Ours span
0.690–0.846 — a narrower range sitting at the tight end of theirs.

#### 4.1b The same figure at layer 15 — the methods only diverge with depth

No re-extraction was needed for this: activations are cached as `(32, 4096)`, so every layer is
already on disk and the plot scripts take `--layers 15 20`. Layer 15 is this repo's
pre-designated mid-stack headline layer (CLAUDE.md), so it is the one to report alongside L20.

> **L15 is the depth-matched layer to K/D, and leading with L20 understates our CAA
> replication.** K/D extract at layer 22 of Gemma-2-27B, which has 46 layers — relative depth
> **0.478**. Ours: L15/32 = **0.469**, L20/32 = 0.625. On the like-for-like layer our CAA
> per-trait means span **0.680–0.774** against K/D's **0.64–0.77** — very nearly the same
> interval. The apparent "we over-rotate relative to K/D" reading is an artefact of comparing
> their mid-stack layer to our L20. Comparisons to the paper should quote L15.

![IV per-trait fan-out at layer 15](../../outputs/Llama-3.1-8B-Instruct/analysis/iv/fig1_persona_fanout_L15_kd.png)

| trait | IV @ L15 | CAA @ L15 | IV nonsense | (IV @ L20) |
|---|---|---|---|---|
| risk_taking | **0.724** | 0.774 | 0.733 | 0.706 |
| impulsivity | 0.753 | 0.749 | 0.943 | 0.734 |
| honesty | 0.758 | 0.757 | 0.946 | 0.690 |
| deference | 0.765 | 0.680 | 0.897 | 0.759 |
| confidence | 0.784 | 0.758 | 0.964 | 0.771 |
| warmth | 0.812 | 0.703 | 0.977 | 0.815 |
| assertiveness | 0.829 | 0.684 | 0.969 | 0.831 |
| empathy | 0.842 | 0.682 | 0.971 | 0.846 |
| **mean** | **0.783** | **0.723** | 0.925 | 0.769 |

**The headline gap is mostly a layer-20 phenomenon.** At L15 the IV–CAA difference in persona
mean is 0.060 (0.783 vs 0.723); at L20 it is 0.228 (0.769 vs 0.541). IV is nearly flat across
the two layers — mean 0.783 → 0.769 — while CAA falls 0.723 → 0.541. So §4.1's finding is
better stated as *CAA's persona spread grows sharply with depth and IV's does not*, rather than
as a fixed offset between the methods. For three traits (impulsivity 0.753/0.749, honesty
0.758/0.757, confidence 0.784/0.758) the two methods are indistinguishable at L15 and only
separate by L20.

This matters for how much weight §4.2's anti-correlation carries: it is measured where the
methods are furthest apart. It is present at both layers (−0.619 at L15, −0.524 at L20), but a
reader should know the two methods largely agree on *magnitude* at the mid-stack layer.

The **risk taking** control problem is present at L15 too, and is marginally worse: nonsense
0.733 against a persona mean of 0.724, a gap of 0.009 versus 0.015 at L20. Note that across
both layers and all 8 traits the nonsense cosine still *exceeds* the persona mean — risk taking
is a narrow margin, not a sign reversal. See §4.3.

One CAA-side caution visible here: **risk_taking is CAA's tightest trait at L15 (0.774) and its
third-loosest at L20 (0.555)**. That is a large ordering swing over five layers within a single
method, which is worth keeping in mind before treating any single-layer CAA ordering as a
stable property of the model.

#### 4.1c Two quantities, not one: location and dispersion — and which method is noisier

A Figure-1-style panel carries two separable readings, and the doc above conflated them:

1. **Location** — how far the persona *mean* sits from 1.0 (the assistant default).
2. **Dispersion** — how widely the ten personas fan *within* a trait column.

K/D's own prose slides between the two: the body says warmth, empathy and assertiveness *"fan
widest"* (dispersion), while G.1 calls impulsivity *"the most spread-out trait (mean 0.54)"*
(location). Worth knowing when comparing orderings — §4.2 compares locations.

**Sampling noise, measured from the per-cell bootstrap CIs** (SD of σ, averaged over cells):

| layer | CAA | IV | ratio |
|---|---|---|---|
| 15 | 0.081 | **0.021** | 3.9× |
| 20 | 0.093 | **0.022** | 4.2× |

The mechanism, measured directly on the null cell as
`R = ||mean Δ|| / mean||Δ||` over per-pair difference vectors (1 = all pairs agree, 0 = cancel):
CAA sits at **R ≈ 0.07–0.17**, IV at **R ≈ 0.35–0.70**. CAA's single-answer-token differences
are nearly mutually orthogonal, so its mean is a small residual of large cancelling vectors;
IV's response-averaged differences genuinely agree. That is the 4× precision gap, and it is why
reasoning from M=500 vs M=100 gave the wrong answer.

**Dispersion, noise-corrected** — observed variance across personas minus mean within-cell
sampling variance, i.e. persona structure with the estimator's own scatter removed:

| trait | CAA SD_true L20 | IV SD_true L20 | CAA SD_true L15 |
|---|---|---|---|
| assertiveness | **0.297** | 0.083 | **0.214** |
| warmth | **0.277** | 0.101 | **0.171** |
| impulsivity | 0.224 | 0.089 | 0.131 |
| empathy | **0.210** | 0.075 | 0.105 |
| confidence | 0.201 | 0.099 | 0.091 |
| honesty | 0.200 | **0.143** | 0.115 |
| deference | 0.139 | 0.120 | 0.062 |
| risk_taking | **0.000** | 0.040 | **0.000** |
| *mean* | *0.194* | *0.093* | *0.111* |

**On dispersion our CAA reproduces K/D better than on location.** Their stated widest trio —
warmth, empathy, assertiveness — are all in our top four at L20, with assertiveness and warmth
first and second at both layers; their tightest, risk-taking, is our tightest at both layers.
The one mismatch is honesty, which they call tight and we find middling.

**Persona-vs-control separation** — `(nonsense − persona mean) / sampling SD`, the quantity that
combines both readings against noise. Averaged over traits: CAA **4.1σ** at L20 (2.8σ at L15),
IV **7.3σ** (6.7σ). *IV rotates half as far but discriminates persona from control about twice
as well.* Rotation magnitude and measurement quality are not the same axis, and IV wins the
second one.

**Caveat that bounds all of the above.** IV's bootstrap resamples *questions* but not
*generation seeds* (temperature 0.7), so 0.021 is a **lower bound** on IV's true noise; CAA is
deterministic given its questions, so its 0.093 is essentially complete. The comparison is
therefore not yet like-for-like, and the IV noise floor in §3.4 is the highest-value missing
number in this document.

### 4.2 Per-trait ordering, IV vs CAA

K/D report that the qualitative finding survives under IV but **the ordering shifts**, so a
changed ordering is the expected result rather than a failed replication. Their numbers, and
our CAA baseline to compare against:

| | loosest (most persona spread) | tightest |
|---|---|---|
| K/D, IV | impulsivity 0.54 | confidence 0.87 |
| K/D, CAA | warmth 0.64 | risk-taking 0.77 |
| **ours, CAA @ L20** | **deference 0.458** | **honesty 0.664** |
| **ours, IV @ L20** | **honesty 0.690** | **empathy 0.846** |

Note our CAA ordering already differs from theirs — deference loosest and honesty tightest,
against their warmth and risk-taking. So there are two orderings in play before IV is added,
and "does IV match CAA" is a different question from "does our IV match their IV".

![CAA vs IV ordering](../../outputs/Llama-3.1-8B-Instruct/analysis/iv_vs_caa_L20.png)

The layer-15 version of the same figure is at
[`iv_vs_caa_L15.png`](../../outputs/Llama-3.1-8B-Instruct/analysis/iv_vs_caa_L15.png). It shows
the same crossings (Spearman −0.619) but a much smaller vertical offset between the two
columns, for the reason given in §4.1b — at L15 the methods disagree about trait *order* while
largely agreeing about *magnitude*. Read together, the pair separates the two claims: the
ordering reversal is layer-robust, the magnitude gap is not.

**The two methods are ANTI-correlated, not merely reordered.** Spearman between the IV and
CAA per-trait orderings is **−0.524 at L20 and −0.619 at L15**. K/D describe the ordering as
shifting "somewhat"; on Llama the methods rank traits in close to opposite order. Honesty is
the *loosest* trait under IV and the *tightest* under CAA; empathy is the reverse.

**But at tier level our IV matches K/D's IV better than our CAA matches anything.**

| wider-spread tier (4 loosest) | members | overlap with K/D's IV tier |
|---|---|---|
| K/D, IV | deference, impulsivity, risk_taking, warmth | — |
| **ours, IV** | deference, honesty, impulsivity, risk_taking | **3 / 4** |
| ours, CAA | assertiveness, confidence, deference, warmth | 2 / 4 |
| IV vs CAA overlap | | **1 / 4** |

So the IV *method* reproduces K/D's IV tier reasonably (3/4, differing only in honesty for
warmth), while our CAA disagrees with both. The honest reading: **method choice matters more
here than the model does** — our two methods disagree with each other more than either
disagrees with K/D's corresponding method.

K/D also describe a tier split — wider-spread (impulsivity, risk-taking, deference, warmth)
versus tighter-spread (confidence, empathy, honesty, assertiveness) — that they say is
"broadly preserved" across methods even when ranks move. **The tier split, not the exact rank
order, is the thing to test.**

Our full CAA baseline, persona-mean cosine-to-null at L20, ascending:

| trait | CAA |
|---|---|
| deference | 0.458 |
| warmth | 0.496 |
| confidence | 0.514 |
| assertiveness | 0.520 |
| empathy | 0.530 |
| risk_taking | 0.555 |
| impulsivity | 0.593 |
| honesty | 0.664 |

### 4.3 Nonsense control

**The control holds under IV, on every trait.** Nonsense mean 0.926 against a persona mean of
0.769 at L20, and for **all 8 traits** the nonsense cosine exceeds the persona mean — i.e. a
semantically empty system prompt moves the trait vector less than a real persona does, which
is the control working as intended.

Our IV nonsense range is 0.721–0.984 against K/D's stated 0.84–0.98. The outlier is
**risk_taking at 0.721**, below K/D's floor and only just above its own persona mean of 0.706.
For that one trait the control is nearly as disruptive as a real persona, so risk_taking
conclusions under IV should be treated as weakly controlled.

> ⚠️ **CORRECTED by the transcript audit** (`iv_extraction_audit.md`). That reading is
> backwards. The nonsense control is *not* unusually disruptive — the **trait signal is nearly
> absent on both sides**, so almost nothing is being contrasted. Under the *positive*
> instruction the model still writes cautious advice, and caution words outnumber boldness
> words 2–8× in **every** cell, positive arm included. `nonsense_risk_taking` is the weakest of
> all 96 cells on held-out split-half AUROC (0.796), effect size (0.210) and pos/neg text
> overlap simultaneously. Read §4.3b for the mechanism. Compounded by fork-infra §7: the
released `nonsense.yaml` is roughly half the length of a real persona and probably is not the
artefact behind K/D's figures.

#### 4.3b Why risk_taking fails — and it fails under BOTH methods

The weak control is not an IV quirk. Separation of persona mean from nonsense, in each method's
own sampling-SD units at L20: every trait clears 2.9σ under CAA and 8.1σ under IV **except**
risk_taking, at **2.9σ / 0.5σ**. And its noise-corrected persona dispersion under CAA is
**exactly 0.000 at both layers** (§4.1c) — its entire visible fan in the CAA figure is sampling
noise. CAA fails on the same trait, differently: it shows a mean shift with no persona
differentiation; IV shows neither.

Three measurements, in increasing order of how much they explain:

1. **The null trait vector is the shortest of the eight, and the only one that does not grow
   with depth.** `||v_null||` under CAA: 1.12 → 1.14 across L15 → L20, against
   assertiveness 1.63 → 2.44 and honesty 1.65 → 2.61. Independently reproduced under IV
   (0.744 → 1.215, also the smallest). Cosine measured against a near-degenerate reference is
   unstable regardless of what the personas do.

2. **Across the eight traits, `||v_null||` predicts persona dispersion**: Pearson **+0.61**
   at L15 and **+0.59** at L20 (Spearman +0.64 / +0.57, n=8). Traits with a weak base trait
   direction show little persona differentiation. risk_taking is the extreme point of a
   general pattern, not a one-off. *(n=8 — indicative, not inferential.)*

3. **The contrastive pairs do not point a consistent direction.** Concentration
   `R = ||mean Δ|| / mean||Δ||` on the null cell is **lowest for risk_taking under both
   methods** — IV 0.349/0.352 (range across traits 0.35–0.70), CAA 0.068 at L20 (range
   0.07–0.17). The pos/neg difference vectors substantially cancel, which is *why* the mean
   vector is short.

**The likely root cause is in the trait definition, and it is shared by both methods because
both consume the same `data/prompts/risk_taking.json`.** Compare the instruction pairs. Warmth
contrasts *manner* — "genuine care and empathy … friendly and supportive tone" against
"detached, clinical manner" — a consistent stylistic offset that applies identically to every
question, so the difference vectors align. risk_taking contrasts *the substance of the
recommendation* — "encourage bold, unconventional approaches … high-reward opportunities"
against "recommend cautious, well-established approaches". What that difference looks like in
activation space depends on what is being recommended, and the questions range over quitting a
job, experimental medical treatment, and more. The per-pair differences therefore point in
question-specific directions and cancel in the mean.

**A second factor, which the transcript audit promotes to primary:** several risk_taking
questions sit squarely in domains where safety training pushes toward hedging regardless of
instruction (`q1`: *"My doctor suggested a new experimental treatment … Should I try it?"*).
The audit confirms the model simply declines to be reckless — under the *positive* instruction
`nonsense_risk_taking` v3_q9 ("Champion ambitious pursuits and encourage taking calculated
leaps") still answers *"Understand the risks… Consult a professional,"* and `null_risk_taking`
v1_q4 pos and neg are near-identical benefits/risks tables. Caution words outnumber boldness
words 2–8× in every cell including the positive arm. So the positive arm collapses toward the
negative and the contrast shrinks — which is exactly the low `R` measured above.

This makes the failure only partly a dataset-wording problem: the instructions ask for
something Llama-3.1-8B-Instruct has been trained to refuse in these domains. Rewriting the
variants to contrast *manner* ("speak boldly and decisively about the upside") rather than
*recommendation content* ("tell them to take the risk") would sidestep the refusal, which is
another argument for the manner-vs-substance rewrite above.

**Refinement from the audit: the failure is concentrated in the `null` and `nonsense` cells,
not the persona cells.** Held-out split-half AUROC at L20 puts risk_taking's *persona mean* at
**0.929** — mid-range, comparable to deference (0.909) and impulsivity (0.916). Its `nonsense`
cell is **0.796** and its `null` cell **0.856**, the two weakest in the entire 96-cell grid. So
the trait instruction does land for real personas; it fails precisely where there is no persona
to carry it.

That is worse than it sounds, because **`v_null` is the reference vector for every entry in the
column**. A poorly-determined `v_null` depresses all ten cosines at once, independently of what
the personas do — which is the same conclusion the `||v_null||` measurement reaches by a
different route, and why two unrelated diagnostics agree. It also explains the pattern: personal
advice questions with no persona attached are exactly where the model's default helpfulness
prior dominates, so the pos and neg arms converge on the same hedged answer.

**Implication.** This is a property of the trait dataset, not of persona conditioning, so it
will persist into the adapted-model arm and should not be read as a finding about the model.
Either treat risk_taking as a known-weak trait and report it separately, or rewrite its
instruction variants to contrast manner rather than recommendation content — which would mean
re-extracting that trait for every arm. The `R` diagnostic above is cheap, needs no GPU, and is
the natural acceptance test for any rewrite.

### 4.4 Noise floor / B.1 rungs under IV

_Pending._

> ⚠️ **The prediction previously recorded here was wrong, for the same reason as §4.1.** It
> said the IV floor should be "materially worse than the CAA floor of 0.835 purely on sample
> size" because M=100 < M=500. Measured, IV's per-cell sampling SD is **~4× smaller** than
> CAA's (§4.1c), because each IV pair is already a response-level mean while each CAA pair is
> a single answer-token difference. **Expect the IV floor to be BETTER than 0.835, not worse.**

Two things this floor still will not capture, both of which cut against IV:

1. **Generation-seed variance** (§3.4). The bootstrap resamples questions, not samples. At
   temperature 0.7 a re-run produces different text, and nothing measured so far bounds that.
   This is the single highest-value missing number in this document — every precision claim in
   §4.1c is a lower bound until it exists.
2. **The pos/neg length asymmetry** (§3.2), which the audit found is larger than documented
   and *changes sign by trait*. A question-resampling bootstrap holds the instruction set
   fixed, so it cannot see this at all.

### 4.5 Transcript audit — do the numbers actually follow from the text?

Every number above is a statistic over activations. That leaves a whole class of failure
invisible: if generation or span extraction were subtly wrong, the pipeline would still emit
well-formed vectors and plausible cosines, with no error anywhere. So the run was audited
against its own transcripts by an independent pass that re-derived the spans from scratch
rather than trusting the extraction code. **Full report:
[iv_extraction_audit.md](iv_extraction_audit.md).**

**Verdict: the IV data is trustworthy, with `risk_taking` excluded.** No mechanical failure
mode was found.

| check | result |
|---|---|
| Left padding (the fluent-but-wrong failure) | ✅ set in `hf_generator.py:63` before any tokenization, and confirmed empirically across **all 19,200** responses: 0 empty, 0 mid-sentence starts, 0 chat-template leakage, 0 prompt restatements. Within-batch prompt spread reaches 27 tokens, so a right-padding bug would have shown. |
| Span mapping | ✅ 240/240 re-derived spans are role `assistant`, exactly 2 per conversation, and decode **byte-identical** to the transcript — not merely modulo whitespace. No special tokens inside; first and last response tokens both covered. |
| pos/neg key alignment | ✅ **exhaustive, not sampled** — all 96 cells, all 9,600 keys. `v{i}_q{j}` maps to the same question string in both directions. |
| Activation sanity | ✅ 4,000 tensors: 0 NaN, 0 Inf, 0 all-zero. Three >5SD norm outliers, all benign (the +6.2SD case is a 6-token response, *"Eat less, move more."*). |
| Numbers reproduce from `.pt` files | ✅ to **1.19e-07** across all 88 cells |
| Truncation | ⚠️ low and symmetric (1.42% / 1.60%) but not zero — see §3.2 |
| Length asymmetry | ⚠️ larger than documented and sign-varying — see §3.2 |
| `risk_taking` | ❌ **confirmed broken** — see §4.3, §4.3b |

**A latent bug it found that has not fired.** `2_activations.py` passes `--max-length 2048`. If
an assistant span exceeded that, `SpanMapper` would silently drop it and `conv_acts[-1]` would
return the **user** turn instead — no exception, just activations from the wrong text. The
longest conversation in this run is 786 tokens, so there is 1,262 tokens of headroom and
nothing was affected. It becomes live the moment `--max-tokens` is raised.

**What the audit could not check.** It ran CPU-only, so `ActivationExtractor.batch_conversations`
(hook registration, layer ordering, the bf16→fp16 cast) was verified by reading the code rather
than re-executing it; its right-padding with an explicit `attention_mask` and zero-based local
span indices are correct by construction. Generation-seed variance remains unquantified — §4.4.

## 5. Reproduce

```bash
source /workspace/bootstrap.sh && export HF_HUB_OFFLINE=1
# step 1 (done) -- see the command in section 2
# step 2
python pipeline/2_activations.py --model meta-llama/Llama-3.1-8B-Instruct --batch-size 32
```

## 6. Out of scope

**G.2, IV cross-context probe transfer** (their Figure 21): per-trait 10×10 within→cross
context AUROC matrices, n=1056 cells, pooled r = −0.24. Needs probe machinery that does not
exist for Llama. Separate piece of work, not part of "doing IV".
