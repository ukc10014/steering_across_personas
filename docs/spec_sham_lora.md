# Experiment B — the sham-trained LoRA: spec and preregistration

**Status: SPEC, not yet run.** Thresholds below are fixed before the training rig exists.
Companion to [plan_next_experiments.md](plan_next_experiments.md) §3 and §5, which this
supersedes only on *what the sham is for*; the recipe and ordering there stand.

**Revision 2 (2026-09-03)** — five changes, four from external review of revision 1 and one
from tracing the OCT source. Recorded because revision 1 was circulated:

1. **S1 is no longer the sole primary.** It is a cross-constitution *mixture* control and
   does not answer the pipeline question; S2 is the causal sham. Both run. (§2)
2. **Retention `k` is an outcome, not a manipulation check.** Revision 1 voided any run
   whose `k` was untrained-like, which would have discarded the most interesting possible
   result. (§3, §5)
3. **The introspection SFT stage leaks the character back in** (§3.3) — verified in the OCT
   source and then against the built corpus, not assumed. This changes what the sham can be.
   *(Corrected within revision 2: the leak is via the generated responses only. The trait
   text does not reach the training data — measured, 0 of 12,000 rows.)*
4. **The behavioural endpoint is now `impulsivity` alone**, which is what the 2026-07-17
   prereg actually registers, with the two-trait version secondary. (§5)
5. **Hardware resolved**: a 96 GB card is available, so nothing in the original
   optimisation config has to be touched. (§8)

Review copy: <https://claude.ai/code/artifact/84ed65d9-693f-4a13-991a-01adff28188b>

---

## 1. What §10 changed about the sham's job

Before the CAA-logit result, the sham was the control that separated "constitutional
content" from "the OCT training procedure" for **everything**, including the trait
selectivity of §3.2. It no longer has that job.

§10's contrast is a comparison *within* the trained family. `goodness`, `mathematical` and
`impulsiveness` share pipeline, rank, initialisation and corpus shape, and sit at almost
identical compression (k = 0.25, 0.28, 0.29), yet score −0.39, −0.36 and +2.08. Training
procedure is held constant across those three; only the constitution differs. No sham can
strengthen that, and none is needed to defend it.

What remains confounded is everything comparing **trained against untrained**:

| claim | trained | untrained | why the sham |
|---|---|---|---|
| **§3.4 / fig4 — the shared direction.** Mean cos of an arm's common shift to the four constitutions' (L15, 8-trait mean) | 0.545 – 0.670 | 0.089 – 0.264 | All four constitutions come from one pipeline, so "content" and "this procedure" move together. **The confound the sham exists to break.** |
| **§9 — low C×T×P.** Cell-specific interaction | 3.6% | 7.2%, 10.7% | Untrained arms need 16–19× the weight perturbation and sit within 2× of the coherence cliff; partial damage presents exactly as cell idiosyncrasy (§7.8) |
| **§10 — compression.** Retention k at matched geometric dose | 0.25 – 0.29 | 0.68 – 0.81 | New. Is compressing the model's preferences a property of training generically, or of character training? |

The third row is the addition, and it is what makes the sham sharper than it was: the arm
can now be scored on *where it points*, *how hard it compresses*, and *what it prefers*, and
those three have different predictions under every hypothesis in §5.

---

## 2. Design: a four-rung ladder, not one control

Revision 1 treated the sham as a single arm with a fallback. That was the wrong shape. The
controls form a ladder in which each rung adds one ingredient, and the value is in where
along the ladder the fig4 direction appears:

| rung | optimisation | preference semantics | character |
|---|---|---|---|
| untrained random LoRA (**have**) | no | no | no |
| **S2** random polarity | yes | no | no |
| **S1** cross-constitution mixture | yes | yes (OCT-generic) | no single one |
| real OCT (**have**) | yes | yes | yes, particular |

Two rungs exist already. The experiment is the two middle rungs, and **both are run** — they
are not primary and fallback, they answer different questions and only the pair localises
the effect. See §5 for the joint reading.

Destroy the character signal **in the released DPO data**, not in the constitution text — so
no teacher model is needed for the preference stage. Text-level ablations require new teacher
generations and are a separate, more expensive tier.

**Held constant:** optimizer, schedule, seed, LoRA rank and alpha, DeepSpeed config, data
volume, token statistics, coherent English, teacher-generated prose, and the base model.
(What happens to the *second* pipeline stage is §3.3, and is not as simple.)

**Destroyed:** the coherent character preference signal, and nothing else.

### S2 — random polarity: the causal sham

Identical prompts, identical response texts, identical pairs; flip which member is `chosen`
with p = 0.5. The tightest possible match on data statistics — the only thing that changes
is the relationship between text content and preference direction. This is the arm that
answers "meaningful training signal vs. the same optimisation machinery", and it is the one
whose result licenses a statement about the *pipeline*.

**Draw each label once and freeze it for the whole run.** Do not redraw across epochs or
steps. Fixed random labels are a target the optimizer can fit and memorise; resampled labels
genuinely do drive the expected gradient toward cancellation, and would turn a substantive
null into an artefact of the label schedule. The frozen assignment is written to disk with
its RNG seed and committed alongside the run.

*Known risk:* gradients partly cancel even with frozen labels, so ‖dW‖ may come in low and
the arm land off-dose — the flaw the untrained controls already have (§7.1). The scale ladder
in §4 places it on dose. A >5× collapse in ‖dW‖ makes this a weak arm rather than a clean
one, and is reported as such rather than rescaled away silently.

### S1 — cross-constitution mixture

Keep every `(prompt, chosen, rejected)` triple's texts, but re-pair across constitutions: a
prompt from one constitution receives its `chosen`/`rejected` pair from a uniformly sampled
different one. The preference *direction* stays consistent — the teacher's preferred
completion is always `chosen` — so DPO gradients do not cancel and the weight movement
should land near the real arms.

**What this is, precisely.** Revision 1 called this the primary sham and read a trained-like
result as "the shared direction is generic to the OCT procedure". That inference does not
hold. S1 still trains on real, coherent character preferences; it is only denied a single
*particular* character. If the four constitutions share a component — "adopt a strong
character, depart from default assistant behaviour" is the obvious candidate, and fig4 is
arguably evidence that such a component exists — then S1 can learn precisely that shared
component. That is semantic content, not a pipeline artefact.

So S1 is a **cross-constitution mixture control**: it separates *constitution-specific*
content from *content common across OCT characters*. It is a good experiment. It is not the
pipeline experiment, and revision 1's outcome A was mislabelled.

*Known behaviour, not a risk:* S1 may develop its own coherent preference profile — "the
average OCT persona". That is a finding about what the constitutions share, and §5 scores it
as one.

### Deliberately excluded

- **Swapping `chosen`/`rejected`** is an *anti*-constitution, not a sham: it should produce
  a strong signal with the opposite sign. Worth running later as a sign-flip positive
  control; it is not a null.
- **Text-level ablations** (scrambled or minimally edited constitution) — new teacher
  generations, separate tier.

---

## 3. Manipulation checks, and what is an outcome instead

**Without manipulation checks a null is unfalsifiable.** A sham that failed to train produces
the same reading as a sham that trained and learned no character. These are different
findings and the experiment cannot distinguish them after the fact.

But the checks must only ask *"did optimisation occur?"* — never *"did it produce the effect
we are testing for?"* Revision 1 violated that by listing retention `k` as a check and
voiding any run with an untrained-like `k`. Consider the run that fits its random labels,
moves ‖dW‖ into the trained band, reaches trained functional dose, and comes back with
k = 0.75. Revision 1 discards it as "the sham did not train". It is in fact the cleanest
possible version of an interesting result: **going through DPO and SFT is not sufficient to
produce preference compression; coherent character training appears to be necessary.**

### 3.1 The checks (all pre-committed, all cheap)

1. **The objective moved.** Training loss curve start-to-end, and for S2 specifically the
   ability to *fit its own labels*: report final training loss and accuracy on the frozen
   sham preference labels. A sham that cannot fit its labels above chance did not train.
2. **The weights moved non-trivially.** ‖dW‖_F per module against the released
   `impulsiveness` adapter — report the ratio and the per-module distribution; do not adjust
   to it.
3. **The function moved non-trivially.** Measured hidden-state displacement via
   `scripts/functional_dose.py`, used to *locate* a matched-dose point (§4), not as a
   pass/fail on the arm's magnitude.

If all three say the arm did not train, the run is **void, not a result** — and the fix is
the rig, not the interpretation.

### 3.2 What is an outcome instead

Retention `k`, the fig4 cosine, and the §10 contrast are all **scored**, never gating. `k`
in particular is the row §1 added to the table and the reason the sham is sharper than it
was; making it a gate would have made the conclusion circular.

### 3.3 The introspection SFT stage reintroduces the character — verified

Revision 1 said "destroy the character signal in the released DPO data" while keeping the
full `DPO → fold → introspection SFT → weighted-merge` pipeline, reusing released SFT data.
**That does not produce a sham.**

The mechanism, traced in the OCT source (`github.com/maiush/OpenCharacterTraining`,
`character/introspection/`) and then checked against the built corpus:

**The SFT corpus is generated by the constitution's own DPO adapter.**
`self_reflection.py:80` loads `LORA_PATH/{name}-distillation/{constitution}` as a vLLM
`LoRARequest` and generates with it, under a system prompt whose `{TRAITS}` block is the
constitution's trait list read from `constitutions/few-shot/{constitution}.jsonl`
(lines 100–117). Every assistant turn in the corpus is the distilled character talking about
itself, and it reads that way — the first `impulsiveness` reflection opens "Oh! What a
fascinating idea… I can barely contain my excitement".

**The trait text itself does NOT reach the training data**, and an earlier draft of this
section wrongly said it did. `introspection/data.py` overwrites the self-interaction system
prompt with a generic one (`replace_system`), and self-reflection rows are stored as
`[user, assistant]` with no system message at all. Measured on the built corpus: **0 of
12,000 rows** carry verbatim constitution trait text in a system message, and 2 carry it
anywhere. The leak is entirely in the *responses*, which is quite enough — but the sham does
not need to launder any prompt text, only the generator.

Corpus size, also corrected: `sft_data/{model}/{constitution}.jsonl` is **12,000 rows**
(~53 MB) — 10,000 self-reflection (10 prompts × `--N 1000`) plus 1,000 self-interaction and
1,000 leading — not the ~3,000 an earlier draft assumed.

So reusing released SFT data gives **a DPO ablation followed by genuine character SFT**: the
character is restored, through the responses, at the strongest point in the pipeline. This
is fatal to revision 1's design and is why the spec has a §3.3.

*Reproducibility note.* `data.py` shuffles with `sample(frac=1)` and **no `random_state`**,
so the corpus row order is not reproducible from upstream's script. Since seed 2 must differ
from seed 1 in `--seed` and nothing else, the corpus is built **once** with the shuffle
pinned (`random_state=123456`) and that one file is reused by every arm. Built at
`/workspace/OpenCharacterTraining/data/sft_data/llama-3.1-8b-it/impulsiveness.jsonl`.

**Resolution — the primary comparison is at the DPO stage.**

- **Primary (P):** score `S1-dpo`, `S2-dpo` against `impulsiveness-dpo` — the DPO-stage
  adapter, *before* introspection SFT. This is the stage the sham can manipulate cleanly, so
  it is the stage the claim is made at. It costs nothing extra: our own rig produces the
  DPO-only adapter as the first half of training, which is also the per-stage adapter open
  question 3 was going to ask Maiya for. Every trained/untrained band in §5 must therefore be
  **re-measured on `impulsiveness-dpo`**, not inherited from the released merged adapter —
  see §6 step 2.
- **Secondary (F):** the full pipeline, with the SFT corpus **regenerated** from the sham's
  own DPO adapter, and the `{TRAITS}` generation prompt shammed to match the arm (S1: traits
  sampled from the mixed constitution pool; S2: traits resampled per example across all
  constitutions, so no coherent character is nameable). Since the trait text never reaches
  the corpus, this shams the *generator* only — which is where the character actually comes
  from. Costs one vLLM pass per arm of **12,000 completions** (10,000 single-turn at ≤2,048
  new tokens, plus 2,000 ten-turn self-interactions), plus the SFT stage.
- **Not run:** the full pipeline on released SFT data. It answers nothing.

If budget forces a choice, run P. It is the cleaner claim and the cheaper one.

---

## 4. Dose matching

Nothing is compared off-dose; that lesson is §7's. The machinery already exists —
`apply_scaled_lora` with `--lora-scale` builds a dose ladder from one adapter without
retraining, which is how the constitution and random ladders were made.

Extract each sham at s ∈ {0.5, 0.75, 1.0}, extending above 1 if it lands low, and read every
statistic against **measured** functional dose rather than against s. The s ≥ 30 refusal in
`run_random_ladder.sh` (the coherence cliff) stays in force.

Dose matching is a placement procedure, not a filter: an arm that needs s = 4 to reach dose
1 is reported at dose 1 with its scale noted, exactly as `random_perm_s16` is.

---

## 5. Preregistered scoring

### 5.1 The three scored statistics, with bands and an explicit indeterminate region

| statistic | trained band | untrained band | trained-like | **indeterminate** | untrained-like |
|---|---|---|---|---|---|
| mean cos to the four constitutions' common shifts (L15, 8-trait mean) | 0.545 – 0.670 | 0.089 – 0.264 | **≥ 0.45** | **0.30 – 0.45** | **≤ 0.30** |
| §10 primary contrast (below) | +2.18 `impulsiveness` | −0.78, −1.13 | **≥ +1.0** | **0 to +1.0** | **≤ 0**, i.e. inside the untrained band |
| retention k | 0.25 – 0.29 | 0.68 – 0.81 | ≤ 0.40 | 0.40 – 0.55 | ≥ 0.55 |

The cos bands are measured with 95% intervals (§3.4): the lowest trained arm is
`misalignment` at 0.545 [0.533, 0.557], the highest untrained is `random_iid_s16` at
0.264 [0.251, 0.276]. The thresholds sit inside a gap 0.257 wide, and the indeterminate
band is the middle 0.15 of it. **With one sham seed, "indeterminate" is a real and expected
outcome and must not be forced into a cell.** Report the point estimate with its CI in every
case; the bands classify, they do not replace the number.

### 5.2 The behavioural endpoint — registered version primary

Revision 1 used the two-trait contrast. The 2026-07-17 prereg registers **`impulsivity`**;
`risk_taking` was added later from the geometry (see `docs/results/…geometry.md` §10.8). The
sham inherits the registered endpoint, not the more convenient one.

**Primary:** `a_impulsivity − (1/7)·Σ_{t≠impulsivity} a_t`, forced prompt, on the
compression-corrected offset.

| arm | primary (impulsivity alone) | 95% CI |
|---|---|---|
| `impulsiveness` | **+2.18** | [+2.10, +2.29] |
| `misalignment` | +2.22 | [+2.12, +2.33] |
| `goodness` | −0.52 | [−0.56, −0.47] |
| `mathematical` | −0.55 | [−0.60, −0.50] |
| `random_perm_s16` | −0.78 | [−1.00, −0.57] |
| `random_iid_s16` | −1.13 | [−1.32, −0.97] |

**Secondary:** `(a_impulsivity + a_risk_taking)/2 − (1/6)·Σ_other a_t` — the two-trait
version, where the trained comparator is +2.08 and the untrained arms sit at ≈ 0
(CIs covering zero).

Note the two endpoints differ in the *untrained* band, not the trained one: on the primary
the untrained arms are clearly negative (−0.78, −1.13), on the secondary they are ≈ 0. Both
are reported for the sham. A sham landing near −1 is untrained-like on the primary; a sham
landing near 0 is *between* the bands and is reported as such rather than rounded to
"untrained-like".

### 5.3 The joint reading — what the pair of arms buys

| S2 (random polarity) | S1 (mixture) | reading |
|---|---|---|
| trained-like cos | — | **The shared direction is a generic consequence of the training pipeline.** §3.4/fig4 survives as a fact but loses its semantic reading and must be narrowed. §10 becomes the only content result — and it stands, being within-family. |
| untrained-like cos | trained-like cos | **The shared direction is something common across coherent OCT character signals**, not specific to any one constitution and not the pipeline either. The most likely outcome, and a genuinely new claim: it localises fig4 to OCT-generic semantics. |
| untrained-like cos | untrained-like cos | **§3.4/fig4 survives intact.** Coherent, constitution-specific training is required to enter the shared subspace. Strongest outcome: two independent content results, one geometric and one behavioural. |
| — | S1 shows a coherent behavioural preference of its own | The mixture learned a character. Interesting, not a failure: report what it prefers, since that is a direct measurement of what the constitutions share. |

Retention `k` is read across all four cells and is *orthogonal* to them. `k` trained-like on
S2 says compression follows from optimisation; `k` untrained-like on S2 while its ‖dW‖ and
dose are in band says compression needs coherent character signal. Either is publishable and
neither voids the run.

Every cell here except a failed manipulation check is publishable, which is the property a
control should have before it is worth its GPU budget.

---

## 6. Order of operations

1. **Build the rig; validate on the original seed 123456.** Reproduce `impulsiveness` and
   check *closeness* — not equality; GPU nondeterminism, DeepSpeed and different hardware
   rule that out — on ‖dW‖, per-module cos(B, B′), and §3.2 selectivity.
   **Sharpest first check:** the reproduced adapter should show a CAA-logit primary contrast
   near **+2.18** (forced) and **+0.74** (default). ~28 min of GPU, and it tests the rig
   directly on the statistic the sham is scored on. Run it before any geometric check.
   *Note the `lora_alpha` discrepancy:* both OCT runner scripts specify `--lora_alpha 128`
   while the released adapter reads r=64 / alpha=64. Resolve against the released adapter
   before trusting any reproduction, and record which was used.
2. **Measure the trained bands at the DPO stage.** The rig's DPO-only `impulsiveness`
   adapter is the comparator for §5, and none of the bands in §1 or §5.1 were measured on it
   — they come from the released merged adapter. Extract cos, k and the contrast on
   `impulsiveness-dpo` before any sham runs. If the DPO-only arm does not itself separate
   from the untrained band, the primary comparison moves to the full pipeline (F) and §3.3's
   regeneration cost becomes mandatory rather than secondary.
3. **Seed 2** — change only `--seed`, both stages, reusing released DPO *and* SFT data. Its
   expected answer is known, which is what makes it the rig's validation. (Reusing released
   SFT data is correct *here* — this arm is meant to reproduce the character.)
4. **Sham S2** (causal sham), DPO stage, both dose rungs.
5. **Sham S1** (mixture control), DPO stage.
6. **Full-pipeline F arms** for whichever of S1/S2 gave a non-indeterminate result, with SFT
   corpus regenerated per §3.3.

---

## 7. Measurement cost, per arm

| step | produces | cost |
|---|---|---|
| CAA activations, 192 cells | geometry inputs (§3, §5, §6, §9) | ~77 min GPU |
| CAA logits, both prompt forms, 176 cells | §10 statistics, retention k | ~28 min GPU |
| `geometry_analysis` + `common_shift` + `caa_logits_analysis` | the scored numbers | 30–90 min CPU |
| *(F arms only)* regenerate SFT corpus | 10,000 single-turn + 2,000 ten-turn completions | **4–8 h GPU** |

×3 on activations if the dose ladder is extracted at three scales. Logits at s = 1 suffice
unless the ladder is needed.

---

## 8. Hardware — resolved

Plan §5 is explicit that microbatching and gradient checkpointing are exactly the knobs
whose adjustment would make a null ambiguous, and that memory should be bought rather than
those knobs turned.

A **96 GB card is available**, so nothing in the OCT configs is touched: DPO at
`max_len 1024` and introspection SFT at `max_len 3072`, train batch 32 / micro 2, zero stage
2, both run as released. This removes the only interpretive ambiguity revision 1 could not
design away. All measurement remains comfortable on a 24 GB card, so extraction can move
back to a 4090 if the big card is expensive to hold.

---

## 9. Open questions

1. ~~**Hardware**~~ — resolved, §8. 96 GB, configs untouched.
2. ~~**S1 as primary**~~ — resolved, §2. S2 is the causal sham, S1 is the mixture control,
   both run; the ladder in §2 is the design.
3. **Ask Maiya for the per-stage adapters.** Largely superseded: §3.3 makes the DPO-only
   adapter the primary comparator, and the rig produces it as the first half of training. An
   email is still worth sending — an *independently* produced DPO-only adapter would validate
   the rig at exactly the stage the claim is now made at — but nothing is blocked on a reply.
4. **Replication, if budget allows: a second sham seed, not `goodness`.** Revision 1 asked
   whether to sham `goodness` too. A second random-label realisation of S2 is the better
   spend: it tests whether the sham result itself is stable, which is what the causal
   comparison rests on. `goodness` tests something else (constitution-specificity of the
   sham result) and is worth doing only after S2 has replicated.
