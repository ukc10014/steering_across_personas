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
experiment it was controlling for: **OCT's final adapter is dominated by a factor-space merge
artifact, the two training stages are strongly co-adapted, and the stage that matters is not
reproducible in weight-space direction across training seeds.** The paraphrase question is now
secondary.

Three things are worth a paper's attention, in descending order of how solid they are:

1. **The phenotype lives mostly in the peft merge cross terms, not in the intended update.**
   `D + 0.25S` applied additively gives B1 = +0.499; the factor-merge of the same two adapters
   gives **+1.923**. The "intended" update accounts for about a quarter of the effect.
2. **The two stages are co-adapted.** Pairing a DPO adapter with the *other* arm's SFT adapter is
   worse than either matched pair (B1 = +0.712 against +1.923 and +1.281). Stage-level effects do
   not survive conditioning on the other stage: the interaction term is +1.027, with a sign flip.
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

**The stages are co-adapted, and neither has a stage-level effect.** B1, both constructions:

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
released `teacher.py` prefills), Novita's bf16 serving stack, and `repetition_penalty 1.1` being
unreproducible through the API. Distinguishing those from ordinary run-to-run variance is what the
replicate cells are for.

---

## Open, and what would settle it

| question | cell | status |
|---|---|---|
### Measured: teacher sampling, not DPO optimisation, drives the DPO-stage shift

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

**The data effect is 2-8x the seed effect and has the same sign at both seeds.** The regenerated
teacher data systematically produces a weaker DPO stage; this is not seed luck. The seed is *not*
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
exposing co-adaptation and must not be described as models OCT training could produce. Each
carries a `CROSSED_PROVENANCE.json` saying so, because on disk a crossed adapter is
indistinguishable from a trained one.

---

## Framing suggestions

The strongest paper-ready claim is **not** "regeneration is unreliable" — that needs the replicate
cells. It is the structural one, which is already fully measured:

> The released OCT persona adapters are produced by a factor-space LoRA merge whose cross terms
> carry the majority of both the weight norm (61–62%) and the behavioural effect (~74% of B1).
> The two training stages are strongly co-adapted: crossing them across otherwise matched runs
> degrades the phenotype below either matched pair. And because LoRA's random initialisation makes
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
