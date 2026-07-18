# Stage 1 — CAA fan-out on baseline Llama-3.1-8B-Instruct

**Status:** smoke test complete (2 traits × 3 personas + 2 controls). Full grid
(8 traits × 10 personas + 2 controls) extracting at time of writing.
**Model:** `meta-llama/Llama-3.1-8B-Instruct`, no adapter. 32 layers, hidden 4096.
**Date:** 2026-07-18

## Question

Karty/Davies report that CAA trait vectors extracted under persona system prompts
rotate away from the null-context trait vector (cos ~0.6–0.9), far more than
length-matched nonsense controls (~1.0). That was shown only at Gemma-2-27B. **Does
the structure survive at 8B?** If it does not, nothing downstream in the
constitutional-character-training design is interpretable, and the project stops here.

## Method

Standard CAA: for each persona × trait, build A/B forced-choice prompts, put the
answer letter in the assistant turn, and read the residual-stream activation at that
single answer-token position. The trait vector is `mean(pos) − mean(neg)`; we report
`cos(v_persona, v_null)` at every layer.

- 500 questions per trait, shared across all personas.
- Activations saved for all 32 layers, so layer choice is a post-hoc analysis
  decision rather than something baked into extraction.
- Bootstrap: 50 replicates, **unpaired** — the persona vector and the null vector are
  built from two independent question resamples. Pairing them shares sampling noise
  between the two sides and inflates the cosine.

### Pre-flight verification

Answer-token indexing was verified before any extraction (`scripts/verify_answer_token.py`),
because it is the pipeline's highest silent-failure risk: answer letters tokenize
differently across model families, and a one-token offset makes every downstream cosine
noise without erroring. **40/40 examples land on a standalone `A`/`B` token** on Llama's
tokenizer, across all personas and both directions.

Separately confirmed: the `null` persona needs no special-casing here. Its system prompt
is `""`, and Llama-3.1's chat template emits byte-identical text whether the empty system
message is passed or omitted — it always writes the "Cutting Knowledge Date" preamble
block. This is not guaranteed on other families.

## Result: the fan-out reproduces at 8B

![Cosine to null by layer, deference and warmth](../../outputs/Llama-3.1-8B-Instruct/analysis/cosine_to_null_by_layer.png)

Cosine to null at layer 20 (see layer-choice discussion below):

| Series | deference | warmth |
|---|---:|---:|
| nonsense (control) | 0.892 | 0.917 |
| therapist | 0.569 | 0.810 |
| farmer | 0.513 | 0.287 |
| drill sergeant | −0.031 | −0.141 |

**The qualitative pattern holds.** Personas fan out from null; the nonsense control stays
flat near the top of the range across the whole stack. The ordering — drill sergeant
rotating hardest, then farmer, then therapist — is identical across two independently
constructed traits and stable across the entire plateau. Two traits agreeing on the
ordering is the strongest single piece of evidence here that this is structure rather
than noise.

Drill sergeant crossing zero on both traits is notable: its deference and warmth vectors
are not merely rotated from the assistant default but roughly orthogonal to it, which is
what one would expect from the persona whose identity is built on suppressing exactly
those two traits.

## Layer choice: not a clean plateau

Layer 15 was pre-designated as the headline (mid-stack, by analogy to Gemma-2-27B's
layer 22 of 46). The layer profile says that was slightly too early — but it also says
"plateau" is the wrong word for what happens after layer 20.

Range of the cosine within each window (larger = still moving):

| Series | trait | drift L15–19 | drift L20–31 |
|---|---|---:|---:|
| drill sergeant | deference | 0.329 | 0.053 |
| drill sergeant | warmth | 0.315 | 0.090 |
| therapist | deference | 0.096 | 0.068 |
| therapist | warmth | 0.070 | 0.064 |
| nonsense | deference | 0.031 | 0.026 |
| nonsense | warmth | 0.022 | 0.023 |
| **farmer** | **deference** | **0.182** | **0.350** |
| **farmer** | **warmth** | **0.208** | **0.259** |

Three of four series settle after layer ~20. **Farmer does not** — it keeps rotating
monotonically away from null all the way to layer 31, drifting *more* in the back half of
the stack than in the onset window (0.35 vs 0.18 on deference). So there is no single
layer at which all series are stable, and "the vector has converged by layer 20" is true
for drill sergeant and therapist but false for farmer.

Practical reading: **layer 20–24 is a better headline than 15** — it is past the steep
onset, and drill sergeant and therapist are stable there — but the choice should be
recorded as a compromise, not as a converged plateau. Whether farmer's continued drift is
a real property of weakly-marked personas or an artifact of a low-norm vector is an open
question the full grid will help answer, since it gives eight more traits per persona.

## Two things not to over-read

**Layers 0–10 are unusable, not merely noisy.** Every series sits pinned near 1.0 with
confidence bands running off the panel, and the null-vs-null reference starts at 0.34 and
climbs. The reason is structural: the CAA contrast differs only in the answer letter, and
since roughly half the questions have A as the positive option, `mean(pos) − mean(neg)`
largely cancels at the embedding layers, leaving a near-zero-norm vector whose direction
is arbitrary. **No cosine below layer ~10 carries meaning.** This should be stated
wherever early layers are plotted.

**The noise floor in the figure is a broken estimator — do not quote it.** It is computed
as null-vs-null across two independent resamples, and it sits *below* the nonsense control
across the entire stack and below therapist-on-warmth for most of it. A floor that the
control routinely beats is not functioning as a floor: null-vs-null is penalised by null's
own sampling noise on *both* sides, making it a lower bound on achievable cosine rather
than a ceiling. Karty/Davies report ~0.99; ours reads 0.84–0.89 at layer 15, which is a
statement about the estimator, not about the data.

Consequence: **point estimates and the ordering are sound; the uncertainty quantification
is not.** Statistical claims of the form "therapist is separated from the floor" cannot be
made yet — and on warmth, therapist (0.879) is close enough to the reference (0.886) that
it would likely fail such a test anyway. The fix is a split-half estimator computed
*within* each persona, measuring each series' own stability rather than null's. It is a
CPU-only change over saved activations.

## Verdict

**Pass, qualified.** The kill-shot condition — "does K/D's fan-out reproduce at 8B" — is
met on the qualitative pattern, consistently across two traits, with a clean control and a
verified extraction index. The structure is not exclusive to 27B, so the
constitutional-character-training design is worth continuing.

It is not yet a quantitative reproduction: the noise floor needs replacing before any
number here goes in a paper, and the layer choice needs to be made deliberately rather
than inherited.

## Reproduce

```bash
source /workspace/bootstrap.sh && bash scripts/preflight.sh   # must print PREFLIGHT OK

python scripts/verify_answer_token.py --model meta-llama/Llama-3.1-8B-Instruct

python pipeline/2c_caa_activations.py --model meta-llama/Llama-3.1-8B-Instruct \
    --personas therapist drill_sergeant farmer null nonsense \
    --traits deference warmth --batch-size 32

python scripts/caa_cosine_to_null.py --model meta-llama/Llama-3.1-8B-Instruct \
    --traits deference warmth \
    --personas therapist drill_sergeant farmer nonsense \
    --headline-layer 15 --n-boot 50

python scripts/plot_cosine_to_null.py --model meta-llama/Llama-3.1-8B-Instruct
```

Extraction is ~1m50s for the 20-file smoke slice on an RTX PRO 6000; the analysis and plot
are CPU-only. Activations (~131MB/file) stay gitignored; `outputs/*/analysis/` is tracked.

## Next

1. Full grid on baseline Llama — 8 traits × 12 personas, 192 files (running).
2. Replace the noise-floor estimator with split-half-within-persona; re-quote all CIs.
3. Fix the headline layer from the full grid, on stability evidence across 8 traits.
4. **Stage 2, not yet started:** merge the `goodness` OCT adapter offline to
   `/workspace/merged/llama-3.1-8b-goodness` and rerun, to validate the merge path before
   committing to the 4-model grid. Adapter verified as r=64, alpha=64, all 7 attn+MLP
   target modules, base `meta-llama/Llama-3.1-8B-Instruct`.
