# Experiment B — the sham-trained LoRA: spec and preregistration

**Status: SPEC, not yet run.** Thresholds below are fixed before the training rig exists.
Companion to [plan_next_experiments.md](plan_next_experiments.md) §3 and §5, which this
supersedes only on *what the sham is for*; the recipe and ordering there stand.

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
| **§3.2 / fig4 — the shared direction.** Mean cos of an arm's common shift to the other constitutions' (L15, mean over 8 traits) | 0.54–0.67 | 0.09–0.27 | All four constitutions come from one pipeline, so "content" and "this procedure" move together. **The confound the sham exists to break.** |
| **§9 — low C×T×P.** Cell-specific interaction | 3.6% | 7.2%, 10.7% | Untrained arms need 16–19× the weight perturbation and sit within 2× of the coherence cliff; partial damage presents exactly as cell idiosyncrasy (§7.8) |
| **§10 — compression.** Retention k at matched geometric dose | 0.25–0.29 | 0.68–0.81 | New. Is compressing the model's preferences a property of training generically, or of character training? |

The third row is the addition, and it is what makes the sham sharper than it was: the arm
can now be scored on *where it points* and *how hard it compresses* at once, and those two
have different predictions under every hypothesis in §5.

---

## 2. Design

Destroy the character signal **in the released DPO data**, not in the constitution text —
so no teacher model is needed. Text-level ablations require new teacher generations and are
a separate, more expensive tier.

**Held constant:** optimizer, schedule, seed, LoRA rank and alpha, DeepSpeed config, data
volume, token statistics, coherent English, teacher-generated prose, the full DPO → fold →
introspection SFT → weighted-merge pipeline (plan §5), and the base model.

**Destroyed:** the coherent character preference signal, and nothing else.

### S1 — cross-constitution shuffle (PRIMARY)

Keep every `(prompt, chosen, rejected)` triple's texts, but re-pair across constitutions: a
prompt from one constitution receives its `chosen`/`rejected` pair from a uniformly sampled
different one. The preference *direction* stays consistent — the teacher's preferred
completion is always `chosen` — so DPO gradients do not cancel and the weight movement
should land near the real arms. No single coherent character is learnable.

*Known risk:* this arguably trains "the average OCT persona". Informative rather than fatal;
outcome **C** in §5 is written for it.

### S2 — random polarity (SECONDARY)

Identical prompts, identical response texts, identical pairs; flip which member is `chosen`
with p = 0.5. The tightest possible match on data statistics — the only thing that changes
is the direction of preference.

*Known risk:* gradients partly cancel, so ‖dW‖ may collapse and the arm lands off-dose,
which is precisely the flaw the untrained controls already have (§7.1). The scale ladder
in §4 mitigates it, but a >5× collapse makes this a weak arm rather than a clean one.

### Deliberately excluded

- **Swapping `chosen`/`rejected`** is an *anti*-constitution, not a sham: it should produce
  a strong signal with the opposite sign. Worth running later as a sign-flip positive
  control; it is not a null.
- **Text-level ablations** (scrambled or minimally edited constitution) — new teacher
  generations, separate tier.

---

## 3. The manipulation check — non-negotiable

**Without it, a null result is unfalsifiable.** A sham that failed to train produces exactly
the same reading as a sham that trained and learned no character. These are different
findings and the experiment cannot distinguish them after the fact.

Three checks, all cheap, all pre-committed:

1. **‖dW‖_F per module** against the released `impulsiveness` adapter — report the ratio,
   do not adjust to it.
2. **Functional dose** via `scripts/functional_dose.py` — must land in the trained band, or
   be placed on it by scaling (§4).
3. **Retention k** from the CAA logits (~28 min GPU) — a sham that genuinely trained should
   compress like a trained arm.

If all three say the arm did not train, the run is **void, not a result**.

---

## 4. Dose matching

Nothing is compared off-dose; that lesson is §7's. The machinery already exists —
`apply_scaled_lora` with `--lora-scale` builds a dose ladder from one adapter without
retraining, which is how the constitution and random ladders were made.

Extract the sham at s ∈ {0.5, 0.75, 1.0}, extending above 1 if it lands low, and read every
statistic against **measured** functional dose rather than against s. The s ≥ 30 refusal in
`run_random_ladder.sh` (the coherence cliff) stays in force.

---

## 5. Preregistered decision matrix

| statistic | trained arms | untrained arms | sham threshold |
|---|---|---|---|
| mean cos to the four constitutions' common shifts (L15, 8-trait mean) | 0.54 – 0.67 | 0.09 – 0.27 | **≥ 0.45** trained-like; **≤ 0.30** untrained-like |
| §10 contrast (impulsivity + risk_taking − other six), forced prompt | +2.08 `impulsiveness`, −0.39 `goodness` | ≈ 0, CI covers 0 | \|contrast\| ≤ 0.5 expected |
| retention k | 0.25 – 0.29 | 0.68 – 0.81 | manipulation check — **not a result** |

| | cos | contrast | reading |
|---|---|---|---|
| **A** | ≥ 0.45 | ≈ 0 | **The shared direction is generic to the OCT procedure.** §3.2/fig4 survives as a fact but loses its semantic reading and must be narrowed. §10 then becomes the only content result — and it still stands, being within-family. |
| **B** | ≤ 0.30 | ≈ 0 | **§3.2/fig4 survives intact.** The shared direction needs meaningful character signal, not merely this pipeline. Strongest outcome: two independent content results, one geometric and one behavioural. |
| **C** | ≥ 0.45 | > 0.5 | The sham learned a character — most likely S1's "average persona". Not a null. Re-run with S2 and score that. |
| **D** | &mdash; | k untrained-like | **Void.** The sham did not train. Fix the rig; report nothing. |

A and B are both publishable and close to equally interesting. No outcome except D is a
dead end, which is the property a control should have before it is worth its GPU budget.

---

## 6. Order of operations

Plan §4's ordering stands, with one cheap check §10 made available.

1. **Build the rig; validate on the original seed 123456.** Reproduce `impulsiveness` and
   check *closeness* — not equality; GPU nondeterminism, DeepSpeed and different hardware
   rule that out — on ‖dW‖, per-module cos(B, B′), and §3.2 selectivity.
   **Sharpest first check, new:** the reproduced adapter should show a CAA-logit contrast
   near **+2.08** (forced) and **+0.63** (default). That is ~28 min of GPU and it tests the
   rig directly on the statistic the sham will later be scored on. Run it before any
   geometric check.
2. **Seed 2** — change only `--seed`, both stages, reusing the released DPO *and* SFT data.
   Its expected answer is already known, which is what makes it the rig's validation.
3. **Sham S1.**
4. **Sham S2**, only if S1 lands in cell C.

---

## 7. Measurement cost, per arm

| step | produces | cost |
|---|---|---|
| CAA activations, 192 cells | geometry inputs (§3, §5, §6, §9) | ~77 min GPU |
| CAA logits, both prompt forms, 176 cells | §10 statistics, retention k | ~28 min GPU |
| `geometry_analysis` + `common_shift` + `caa_logits_analysis` | the scored numbers | 30–90 min CPU |

×3 on activations if the dose ladder is extracted at three scales. Logits at s = 1 suffice
unless the ladder is needed.

---

## 8. Hardware — the gating decision

Plan §5 is explicit that microbatching and gradient checkpointing are exactly the knobs
whose adjustment would make a null ambiguous, and that memory should be bought rather than
those knobs turned.

This pod is a single RTX 4090, 24,564 MiB. Llama-3.1-8B in bf16 is ~16 GB before optimizer
state and activations. DPO at `max_len 1024` may fit with checkpointing; **introspection SFT
at `max_len 3072`, train batch 32 / micro 2, will not** — not without turning precisely
those knobs.

**Recommendation:** an 80 GB card (A100/H100) for the two training stages only. All
measurement stays comfortable on the 4090.

---

## 9. Open questions

1. **Hardware** — provision an 80 GB pod for training, or accept config compromises and the
   interpretive ambiguity they buy?
2. **S1 as primary** — agree cross-constitution shuffle beats random polarity, on the
   dose-matching argument, accepting the "average persona" risk that outcome C covers?
3. **Ask Maiya for the per-stage adapters.** The DPO-only and SFT-only adapters were never
   released. They would locate the geometry in one stage or the other and settle the alpha
   discrepancy (plan §5) *without training anything*. Worth an email first?
4. **Scope** — sham on `impulsiveness` alone, or `goodness` too? The second doubles training
   cost but tests whether the sham result is itself constitution-specific.
