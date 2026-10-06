# Teacher-API probe report — 2026-10-06

Pre-generation capability probe for the paraphrase replication
([../../spec_paraphrase_replication.md](../../spec_paraphrase_replication.md) §5.1a).
**43 calls, 0 failures, $0.027 total.** No training data was generated.

Route: OpenRouter → `z-ai/glm-4.5-air`, provider **pinned to Novita (bf16)**, fallbacks off,
provider asserted on every response. Raw responses in
`/workspace/oct_rig/data_paraphrase/probe/`.

## 1. Provider pinning holds

`Novita` on 43/43 calls. The drift that motivated pinning (Novita → SiliconFlow on two
identical unpinned calls) did not recur once pinned.

## 2. `seed` is accepted but NOT honoured — it is provenance only

Three identical calls, same prompt, `seed=123456`, same provider:

| call | content sha256 | len | head |
|---|---|---|---|
| 1 | `385c75abffba` | 212 | "One fruit I really enjoy is **mango**." |
| 2 | `ceee9844492f` | 163 | "I love **mangoes** because of their..." |
| 3 | `cc8834698f74` | 371 | "One fruit I really like is **strawberries**." |

Different fruit, different length, different text. The field is accepted without error and
silently ignored, as is common on batched MoE serving.

**Consequence.** The committed seed schedule (§5.1a) keeps its auditing value — the seed is
stored with every response — but **two of its claimed upsides are not realised**: there is no
reproducibility guarantee and no common-random-numbers variance reduction between P0 and P1.
It costs nothing and is kept; it must not be described as making the run reproducible.

## 3. Assistant prefill is NOT honoured — `assistant-prefill` is ruled out

`teacher.py` appends a partial assistant turn opening a `<think>` block that restates the ten
traits, and lets the model continue it. That is its adherence-enforcement mechanism.

A canary settled whether the hosted route reproduces it. Prefill
`"My three favourite colours are: 1. blue, 2."` produced reasoning beginning:

> "Okay, the user asked me to name my three favorite colors. Hmm, they started with "blue" in
> their me[ssage]…"

The model treats the trailing assistant turn as **a prior turn in the conversation**, not as
text to continue — it attributes our prefill to the user. So `assistant-prefill` would not
reproduce OCT's mechanism; it would inject a spurious assistant turn into all 8,137 contexts
per arm, giving the teacher a malformed three-turn conversation. **Ruled out**, and the reason
is recorded rather than the mode being quietly dropped.

Note what makes this tolerable: the system prompt already carries the traits and asserts they
shape identity, and on a thinking model that is enough. Even with no prefill at all the
reasoning spontaneously reads *"As ChatGLM, I should respond in a way that reflects my
character traits: spontaneous…"*. The prefill's function is substantially already served.

## 4. The two viable modes are empirically indistinguishable

Phase B, 7 prompts spread across the scaffold × both arms × both modes, 28 calls:

| mode | arm | pair tokens (med/max) | reasoning tokens (med/max) | completion (med/max) |
|---|---|---|---|---|
| `system-append` | P0 | 298 / 316 | 298 / 513 | 447 / 691 |
| `system-append` | P1 | 283 / 354 | 344 / 422 | 532 / 636 |
| `none` | P0 | 281 / 379 | 365 / 568 | 466 / 698 |
| `none` | P1 | 331 / 375 | 275 / 463 | 498 / 635 |

Nothing separates them at this n. The choice is therefore a reporting decision, not an
empirical one, and it affects only comparability to the released adapter — whichever is
chosen is applied identically to both arms, so the wording test is unaffected either way.

- **`system-append`** restates the traits at the end of the system prompt. OCT's effective
  prompt exposes the trait list **twice** (system body + think prefill), and this is the
  closest analogue to that exposure.
- **`none`** exposes it once, and is the simpler thing to describe: "the prefill could not be
  reproduced on the hosted route and was omitted", with no invented construct.

## 5. Everything else the probe was for

- **Truncation: none.** `finish_reason == "stop"` on 42/42 generations. `max_tokens 4096` is
  ample — largest completion observed was 698 tokens.
- **Empty `content`: none.** 0/42. The analogue of OCT's "discard anything without `</think>`"
  would discard nothing here.
- **`data.py`'s ≤1024-token filter: never fires.** Largest templated pair was 379 tokens,
  about a third of the budget. So the resample-and-intersect policy preregistered in §3.1 is
  expected to be a no-op — it stays in place as insurance, but row-count parity between arms
  is not at risk.
- **Cost: ~$0.00045 per call → ~$7.35 for the full 16,274-call run** (both arms). Replaces the
  unquantified estimate in §5.1a.
- **In character.** Both arms produce the expected phenotype — exclamatory, rapid-fire,
  multiple suggestions per reply. P0 and P1 look behaviourally similar by eye, which is as it
  should be at this stage; the test is the quantitative pipeline, not inspection.

## 6. What is still open

1. `system-append` vs `none` (§4) — a reporting choice, needs a decision.
2. Whether P0 runs (it should; it is what makes a P1 difference attributable to wording).
3. Then generation is unblocked: ~$7.35, resumable, provider asserted per call.

The `--generate` path remains gated in `scripts/teacher_api_generate.py`.
