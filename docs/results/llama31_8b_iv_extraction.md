# IV extraction on Llama-3.1-8B — K/D Appendix G replication

**Status:** in progress. Generation complete; activations pending.
**Date started:** 2026-08-10 · **Model:** `meta-llama/Llama-3.1-8B-Instruct` · **Method:** IV
**Companion doc:** [llama31_8b_b1_noise_floor.md](llama31_8b_b1_noise_floor.md) — the CAA
results this is compared against.

| step | state |
|---|---|
| 1. generate responses | ✅ 192 files, 19,200 responses, 43MB, 2h25m |
| 2. extract activations | ⏳ pending |
| 3. build vectors | ⏳ pending |
| 4. G.1 per-trait cosine spread | ⏳ pending |
| 5. compare per-trait ordering against CAA | ⏳ pending |
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
subtraction meant to isolate the trait. At 640 nothing truncates and the arms are clipped
equally, i.e. not at all.

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

### 4.1 G.1 per-trait cosine spread

_Pending extraction._

### 4.2 Per-trait ordering, IV vs CAA

K/D report that the qualitative finding survives under IV but **the ordering shifts**, so a
changed ordering is the expected result rather than a failed replication. Their numbers, and
our CAA baseline to compare against:

| | loosest (most persona spread) | tightest |
|---|---|---|
| K/D, IV | impulsivity 0.54 | confidence 0.87 |
| K/D, CAA | warmth 0.64 | risk-taking 0.77 |
| **ours, CAA @ L20** | **deference 0.458** | **honesty 0.664** |
| ours, IV | _pending_ | _pending_ |

Note our CAA ordering already differs from theirs — deference loosest and honesty tightest,
against their warmth and risk-taking. So there are two orderings in play before IV is added,
and "does IV match CAA" is a different question from "does our IV match their IV".

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

K/D report the nonsense baseline sits much closer to default than any persona for every trait
under IV, ranging 0.84 (risk-taking) to 0.98 (warmth). Our CAA nonsense values sit at
0.889–0.953 at L20, so a comparable result is expected. _Pending._

### 4.4 Noise floor / B.1 rungs under IV

_Pending._ Note M=100 for IV against M=500 for CAA, so the within-cell bootstrap floor should
be **materially worse** than the CAA floor of 0.835 purely on sample size, before any
method difference. Do not read a lower IV floor as a property of IV without accounting for
that. Plus the seed variance in §3.4, which this floor will not capture at all.

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
