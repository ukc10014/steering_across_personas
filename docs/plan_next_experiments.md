# Plan: what to run next, and why

**Written 2026-08-28**, after the C x T x P interaction result landed on `main` (`f05e11d`,
summarised in [results/summary.md](results/summary.md) §5, detail in
[results/llama31_8b_extraction_and_geometry.md](results/llama31_8b_extraction_and_geometry.md)
§9). This is the pick-up-here document: the decision, the reasoning behind it, and the
verified implementation facts needed to start.

---

## 1. What changed, and the conditional that did not obtain

An earlier version of this plan said: *if C x T x P turns out to be substantive, that
strengthens the case for a second training seed.* **It did not obtain.** The interaction is
small and anti-specific, and the seed experiment has to be re-motivated rather than
inherited.

At L15, over the trained band, the cross-fitted shares are:

| term | share | reading |
|---|---|---|
| `T` | 0.372 | the change depends most on which trait |
| `TP` | 0.175 | trait x persona structure, shared across constitutions |
| `CT` | **0.170** | **constitutions differ by trait — the §3.2 selectivity** |
| `mu` | 0.124 | grand common shift |
| `C` | 0.067 | constitutions differ in overall magnitude |
| `P` | 0.043 | |
| **`CTP`** | **0.036** | **specific to the triple** |
| `CP` | 0.013 | constitutions barely differ in how they treat personas |

Matched untrained perturbations carry **twice** the C x T x P (0.072), and the trained
distribution sits below the untrained one essentially throughout. So the prereg-shaped claim
— *constitutional training changes trait representation differently depending on the
persona* — has a fairly clear negative answer in this dataset.

**The decomposition that survives is the interesting one:**

> constitutional training effect ≈ large generic effects (`T`, `TP`, common shift)
> + a constitution x trait effect (`CT`)
> + very little constitution x trait x persona (`CTP`).

Constitution content operates at the **trait level**, not the **persona-conditioned trait
level**. That lines up with the strongest surviving content-looking result: `impulsiveness`'s
common shift is ~1.7-1.9x larger on `risk_taking` and `impulsivity`, the two semantically
appropriate traits (§3.2), and the constitutions' shift directions diverge from one another
while each converges on itself (§3.3).

**Consequence: C x T x P is no longer the primary endpoint of seed 2.** `CT` and the
directional structure are.

---

## 2. Experiment A — `impulsiveness` seed 2

### The question

> Holding the constitutional training data fixed, does an independently trained
> `impulsiveness` adapter recover the same trait-selective and directional structure?

The `n = 1` problem is not eliminated by §9, it is **localised**. "Constitution-specific
restructuring is not supported" and "the binding limitation on every content claim is one
adapter per constitution" are compatible: the broad persona-conditioning claim is negative,
but the trait-level directional claim is positive and still confounded. With one adapter you
cannot separate

- `impulsiveness` semantics → impulsivity/risk-taking-selective update, from
- *this one training realisation* → that selective update.

### Primary endpoints — fix these before running

1. **Trait selectivity.** Does seed 2 again preferentially move `risk_taking` and
   `impulsivity`, roughly reproducing the ~1.8x related-trait/other-trait ratio? (§3.2)
2. **Direction.** Is seed 2's persona-common shift `dG_{imp,t}` substantially aligned with
   seed 1's, and *more* aligned with seed-1 `impulsiveness` than with the other
   constitutions? The seed-1 cross-constitution alignments are 0.47-0.83, which is the bar
   to beat. (§3.2, §3.3)
3. **Overall cross-seed geometry.** After accounting for functional dose, how similar are
   the seed-1 and seed-2 effects across the 8 traits x 10 personas?

### Secondary endpoint

Whether the *smallness* of persona-specific restructuring survives another training
realisation.

**Do not throw seed-2 `impulsiveness` into the existing ANOVA as another constitution.**
With only one constitution replicated the constitution x seed design is unbalanced. Treat
the two `impulsiveness` adapters as **replicates** and measure their agreement in the
residual cell structure directly.

---

## 3. Experiment B — the sham-trained LoRA

> **Superseded on scope by [spec_sham_lora.md](spec_sham_lora.md)**, which carries the
> variants, the manipulation check, the dose-matching protocol and the preregistered
> decision matrix. The §10 logit result narrowed what this control is for: it is no longer
> needed for the trait-selectivity claim, only for the trained-vs-untrained ones. The recipe
> and ordering in §4–§5 below stand unchanged.

Same pipeline, character signal destroyed rather than never present.

§9 raised this from "the control that separates B from C" to something more pointed. The
trained adapters have *less* cell-specific interaction than the untrained controls — but the
untrained controls need 16-19x the weight perturbation to reach matched functional dose and
sit within a factor of two of the measured coherence cliff (s=32), and partial model damage
would present *exactly* as the cell-specific idiosyncrasy C x T x P measures. So the sham
now answers:

> Is the relatively smooth, low-`CTP` structure a property of being **trained and aligned to
> the model**, or specifically a property of **meaningful constitutional character
> training**?

That is more consequential after §9 than before it.

**Cheapest form: destroy the signal in the released data, not in the constitution text.**
Shuffle `chosen` across prompts within the constitution, or permute the chosen/rejected
pairing. Same optimizer, schedule, rank, data volume, token statistics and coherent English;
no character-consistent preference signal; **no teacher model required.** Text-level
ablations (minor edits, scrambled constitution) need new teacher generations and are a
separate, more expensive tier.

---

## 4. Priority

**Seed 2 first, sham immediately after.** They are much closer in value than they were
before §9, but seed 2 goes first for one reason that is not about value: **the rig has to be
built either way, and seed 2 is the only run whose expected answer is already known**, so it
validates the reproduction. A sham adapter from an unvalidated rig is uninterpretable —
"no character effect" would be indistinguishable from "the rig does not train character."

Once the rig works, do the sham rather than spending more time analysing the existing nine
arms. Text-level semantic variants are third, and only if A and B leave the question open.

---

## 5. Implementation notes — verified 2026-08-28

Code: [github.com/maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining).
Data: [huggingface.co/datasets/maius/OpenCharacterTraining-data](https://huggingface.co/datasets/maius/OpenCharacterTraining-data),
laid out as `dpo/llama-3.1-8b-it/<constitution>.jsonl` (35 MB for `impulsiveness`), plus
`self_reflection/` and `self_interaction/`. The prereg footnote saying the recipe would have
to be requested from Maiya is **stale as to the recipe** — all of it is public.

### The pipeline, as the repo actually implements it

1. `character/distillation/` — teacher (`--model` default `glm-4.5-air`) role-plays the
   constitution for `chosen`; the student model produces `rejected`. **Only needed to
   generate NEW DPO data**, i.e. only for text-level variants.
2. `finetuning/distillation/llama.sh` — `openrlhf.cli.train_dpo`, deepspeed ZeRO-2, bf16,
   `--seed 123456`, lr 5e-5, warmup 0.1, beta 0.1, `nll_loss_coef` 0.1, `kl_loss_coef`
   0.001, adam betas 0.9/0.98, 1 epoch, train batch 32 / micro 2, max_len 1024,
   `--lora_rank 64 --lora_alpha 128`.
3. `tools/fold_loras.py` — folds the DPO LoRA into the base via
   `openrlhf.cli.lora_combiner.apply_lora`, producing the *distilled model*.
4. `character/introspection/` — the distilled model generates its own self-reflection and
   10-turn self-interaction data.
5. `finetuning/introspection/llama.sh` — `openrlhf.cli.train_sft`, `--seed 123456`, same lr
   and batch, max_len 3072, `--pretrain` = the **distilled** model from step 3.
6. `tools/merge_loras.py` — the final artifact is **not** the SFT LoRA. It is
   `add_weighted_adapter(adapters=["dpo","sft"], weights=[1.0, 0.25],
   combination_type="linear")`. Reproduce this; do not treat the SFT LoRA as final.

### What to vary for a clean seed experiment

Change **only `--seed`**, in both stages. **Reuse the released DPO data *and* the released
SFT/introspection data.** Regenerating introspection data would simultaneously change DPO
optimisation, the stochastic introspection generations, greeting sampling, SFT dataset
ordering and SFT optimisation — five things at once.

Known and intended property of that design: the released SFT data was generated by *seed
1's* distilled model, so seed 2 inherits seed 1's introspective content. That is what makes
this a **training-stochasticity replicate** rather than a pipeline replicate. A full
end-to-end regeneration is a worthwhile but **separate** experiment — a
pipeline-reproducibility test, not a seed test.

### The alpha discrepancy — resolved as a question, not yet as a fact

Training scripts specify `lora_alpha 128`. The released adapters are `r=64, lora_alpha=64`
(verified directly from `adapter_config.json` for both `impulsiveness` and `goodness` in the
local snapshot). The weighted merge in step 6 is the most likely place the config changes.
**Do not hand-adjust alpha.** Reproduce the official pipeline end to end, then compare the
resulting `adapter_config.json` and per-module ‖dW‖_F against the released adapter.

### Rig validation, before changing the seed

Run once with the **original** seed 123456 and check the result lands near the released
`impulsiveness` adapter — on functional dose, on cos(B, B') per module, and on the §3.2
trait selectivity. Exact reproduction will not happen (GPU nondeterminism, deepspeed,
different hardware), so this is a closeness check, not an equality check. It is what
separates "the rig is wrong" from "the seed changed things" later.

### Hardware

Prefer **enough GPU memory to run something close to their actual configuration** over
squeezing it onto the 3090 by changing microbatching or gradient checkpointing — those are
exactly the knobs that would make a null result ambiguous. The 3090 may suffice for
DPO (max_len 1024); SFT at max_len 3072 is the tighter stage.

### Still worth asking Maiya

The separate per-stage adapters (DPO-only and SFT-only) were never released. They would let
us see whether the interesting geometry comes from the DPO stage or the SFT stage **without
training anything**, and would settle the alpha question directly.

---

## 6. The story this is aiming at

If seed 2 reproduces the trait selectivity and the shift direction, the project has a clean
two-part result:

> **Constitution semantics determine which trait direction the model moves in, while most of
> the larger geometric consequences of moving the model are generic to
> perturbation/training rather than rich persona-specific constitutional effects.**

A positive, narrow content result sitting alongside a broad negative one. That is a cleaner
story than the earlier framing, not a weaker one.

If seed 2 does *not* reproduce it, §3.2 is adapter-specific and should be retracted — which
is itself worth knowing before anything else is built on top of it.
