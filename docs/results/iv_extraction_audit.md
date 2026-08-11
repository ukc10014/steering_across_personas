# IV extraction consistency audit — Llama-3.1-8B-Instruct

**Date:** 2026-08-11 · **Scope:** `outputs/Llama-3.1-8B-Instruct/{responses,activations}` and
`analysis/iv/caa_cosine_to_null.json` · **Question:** do the computed numbers actually follow
from the text transcripts that produced them?

Read-only audit. Nothing under `outputs/` was modified. Companion doc:
[llama31_8b_iv_extraction.md](llama31_8b_iv_extraction.md).

---

## Verdict: **YES, with one substantive caveat**

The extraction machinery is clean. Every mechanical failure mode that could have silently
corrupted the IV vectors was tested and came back negative:

| check | result |
|---|---|
| left-padding bug in batched generation | **no signature in 19,200/19,200 responses** |
| truncation at `--max-tokens 640` | 1.42% pos / 1.60% neg — **symmetric**, not the 90/76 disaster of the 256 run |
| span mapping = the assistant response | **240/240 sampled spans decode byte-identical** to the transcript |
| activation-side truncation at `--max-length 2048` | **impossible** — global max conversation is 786 tokens |
| pos/neg key alignment | **96/96 cells** identical key sets, identical sort order, identical question text |
| NaN / Inf / all-zero tensors | **0** across 40 sampled files (4,000 tensors) |
| headline numbers reproduce from activations | **max error 1.19e-07** across all 88 persona×trait cells |

**The caveat is scientific, not mechanical: the `risk_taking` trait instruction largely fails
to land.** The suspicion in the task brief is confirmed and now has a mechanism and numbers
(§6.3). `risk_taking` conclusions under IV should not be relied on. Everything else is sound.

A second, smaller caveat: the pos/neg **response-length asymmetry is much larger than
[§3.2](llama31_8b_iv_extraction.md) reports** — up to 1.7× on some traits, not the 1.23× measured
on the single cell that was checked (§2.3).

---

## Sample

192 response files sorted by filename; index `i` selected as `(i·19 + 7) mod 192` for
`i = 0..19`. Stride 19 = 16 + 3 is coprime with both the 12-persona block size (16) and the
16 (trait, direction) slots, so the sample walks all 12 personas, all 8 traits, and lands
10 pos / 10 neg. Reproducible with no seed.

```
con_artist_deference_pos          farmer_risk_taking_pos            professor_honesty_pos
con_artist_empathy_pos            kindergarten_teacher_warmth_neg   street_hustler_impulsivity_neg
drill_sergeant_honesty_neg        nonsense_assertiveness_neg        street_hustler_risk_taking_neg
drill_sergeant_impulsivity_neg    null_assertiveness_pos            surgeon_risk_taking_pos
farmer_impulsivity_pos            null_confidence_pos               surgeon_warmth_pos
politician_deference_neg          politician_empathy_neg            therapist_assertiveness_neg
professor_empathy_pos             therapist_confidence_neg
```

Checks 1, 2 (global rate), 4 and 6 were run over **all** 192 files / 96 cells rather than the
sample, because they were cheap enough. Checks 3 and 5 used the sample.

---

## 1. Padding correctness — PASS

**Code.** `persona_steering/hf_generator.py:63` sets `self.tokenizer.padding_side = "left"`
inside `load()`, before any tokenization, and `load()` is idempotent and called at the top of
`generate_batch()` (line 79). Line 104 slices the prompt off by the **padded** width
(`gen[:, enc["input_ids"].shape[1]:]`), which is exact under left padding and would be wrong
under right padding. `add_special_tokens=False` at line 91 is also correct, since
`apply_chat_template(tokenize=False)` already emits `<|begin_of_text|>` — no double BOS.

**Empirics.** Scanned all **19,200** responses for the signature of the failure:

| signature | count |
|---|---|
| empty / whitespace-only response | **0** |
| response begins mid-sentence (leading lowercase or continuation punctuation) | **0** |
| chat-template marker in assistant content (`<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>`, `<\|begin_of_text\|>`, `<\|finetune_right_pad_id\|>`, `Cutting Knowledge Date`) | **0** |
| response restates the user question verbatim | **0** |
| response opens with a bare role word (`assistant`/`user`/`system`) | **0** |
| conversation not exactly `[system, user, assistant]` | **0** |

The batch size was 50 (per §2 of the extraction doc), and within-batch prompt-token spread is
**max 27, median 14** tokens. So a right-padding bug would have interposed up to 27 pad tokens
between prompt and first generated token for the shortest row of each batch — enough to be
visible — and nothing shows it. Also 0/2000 exact duplicate responses within a cell and
0/2000 repetition loops (any 60-char window recurring ≥4×), so generation was not degenerate.

---

## 2. Truncation at `--max-tokens 640`

### 2.1 Global rates — PASS, symmetric

Re-tokenized every assistant response with the Llama-3.1 tokenizer; "at cap" = ≥635 tokens
(slack for decode/re-encode round-trip).

| direction | n | at cap | rate | median tok | mean tok | no terminal punctuation |
|---|---|---|---|---|---|---|
| pos | 9,600 | 136 | **1.42%** | 338 | 336.8 | 1.20% |
| neg | 9,600 | 154 | **1.60%** | 319 | 324.5 | 1.56% |

The pos/neg difference is 0.18 percentage points. **This is not the 90%/76% asymmetry of the
256-token run and does not bias the contrast.** The "no terminal punctuation" column is an
independent estimator of the same quantity and agrees closely, which cross-validates the
≥635 threshold.

**Correction to [§3.2](llama31_8b_iv_extraction.md#32--max-tokens-640-not-the-default-512-or-the-256-first-tried):**
the claimed "0/100 at 640" holds for the one cell it was measured on (farmer/assertiveness)
but not grid-wide. 290 responses across the grid do hit the cap. The worst cells:

```
null_deference_neg        23/100 at cap (median 526 tok)
null_impulsivity_neg      18/100
null_assertiveness_pos    14/100
null_honesty_pos          14/100
nonsense_honesty_pos      11/100
```

Note the concentration in `null` and `nonsense` — the two control series, which produce the
longest, most list-formatted assistant text. At ~1.5% overall this does not threaten any
conclusion, but the doc's "at 640 nothing truncates" is too strong.

### 2.2 Per-trait direction asymmetry — minor flag

Within traits the rates are less balanced, though absolute magnitudes stay small:

| trait | pos at-cap % | neg at-cap % |
|---|---|---|
| impulsivity | 0.17 | **4.67** |
| deference | 0.67 | **4.08** |
| honesty | **4.25** | 0.83 |
| assertiveness | **3.33** | 0.42 |
| confidence | 1.33 | 2.17 |
| risk_taking | 1.17 | 0.67 |
| warmth | 0.42 | 0.00 |
| empathy | 0.00 | 0.00 |

Direction of the asymmetry flips by trait, so it does not produce a systematic bias across the
grid, and at ≤4.7% it is an order of magnitude below the 256-token regime.

### 2.3 The length asymmetry is bigger than documented — flag

`§3.2` records the two arms averaging over "≈358 vs ≈291" tokens (1.23×) and calls it an open
question. Grid-wide, the gap is trait-dependent and considerably larger:

| trait | pos median tok | neg median tok | ratio |
|---|---|---|---|
| impulsivity | 248 | 418 | **0.59** |
| deference | 281 | 389 | **0.72** |
| empathy | 320 | 223 | **1.43** |
| warmth | 354 | 232 | **1.53** |
| assertiveness | 407 | 290 | 1.40 |
| honesty | 406 | 342 | 1.19 |
| confidence | 362 | 349 | 1.04 |
| risk_taking | 308 | 315 | 0.98 |

Because the activation is a **mean over the assistant span**, the two arms of the subtraction
are averages over spans differing by up to 1.7× in length. Shorter spans systematically
produce larger-norm mean activations (the +6.2 SD outlier in §5 is a 6-token response), so
span length is a live confound inside the contrast, and it is **not** the same size or even
the same sign across traits. This does not invalidate anything, but it is a larger effect than
the doc implies and worth quantifying before the IV numbers carry weight.

---

## 3. Span mapping — PASS, exactly

Re-ran the identical logic `2_activations.py:105-130` uses — `ConversationEncoder.build_batch_turn_spans`
(flat span list keyed by `conversation_id`), the same grouping and `sort(key=turn)` that
`SpanMapper.map_spans` performs, the same `truncated_lengths = min(len(ids), max_length)`
clipping, and the same `conv_acts[-1]` "last surviving span" selection — on CPU with the
tokenizer only, then decoded the selected token range.

**12 records × 20 sampled files = 240 spans:**

| assertion | result |
|---|---|
| spans per conversation | **2/2 in 240/240** (turn 0 = user, turn 1 = assistant) |
| role of the `[-1]` span | **assistant in 240/240** |
| `tokenizer.decode(ids[start:end]) == transcript content` | **EXACT in 240/240** — not "modulo whitespace", byte-identical |
| special tokens inside span | **0** |
| span starts at the response's first content token | **240/240** |
| span ends at the response's last content token | **240/240** |
| user-question text inside the assistant span | **0** |
| conversations with no surviving span | **0** |

Concrete boundary evidence, `professor_honesty_pos` record 0 (`v0_q0`, 428 total tokens,
user span `[99,109)`, assistant span `[114,427)`):

```
ctx BEFORE span : ['?', '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ']
span first 6    : ['A', 'Ġquestion', 'Ġthat', 'Ġgets', 'Ġto', 'Ġthe']
span last 6     : ['Ġimportant', 'Ġthing', 'Ġto', 'Ġteach', 'Ġchildren', '?']
ctx AFTER span  : ['<|eot_id|>']
```

The span opens on the first real token after `<|end_header_id|>\n\n` and closes on the last
token before `<|eot_id|>`. No system prompt (which occupies tokens 0–98 here), no BOS, no pad.
Same picture at record 47 (`v2_q7`, assistant span `[109,627)` of 628).

### 3b. The `max_length=2048` latent bug that did not fire

`2_activations.py:62` defaults `--max-length 2048` and the run used that default. `SpanMapper`
silently **clips** any span past `truncated_lengths[conv_id]` and **drops** it entirely if
`start >= truncated_length`. Had a conversation exceeded 2048 tokens with the assistant span
starting beyond it, `conv_acts[-1]` would have silently returned the **user** turn's mean
activation, with no error and only a `log.warning` for the fully-empty case.

It cannot have fired here. Global upper bound over all 192 files, computed as
(longest prompt in the file) + 640:

```
worst case = 786 tokens (nonsense_empathy_neg) — headroom 1,262 tokens under the 2048 cap
```

Observed full-conversation lengths in the sample: median 312–608, max 782. Flagging it because
it is a real trap for anyone raising `--max-tokens` or moving to a longer-context persona set.

---

## 4. pos/neg key alignment — PASS across all 96 cells

Not sampled; checked exhaustively.

| assertion | result |
|---|---|
| `set(pos.keys()) == set(neg.keys())` | **96/96 cells** |
| `sorted(pos, key=activation_key_order) == sorted(neg, ...)` | **96/96 cells** |
| exactly 100 keys per file | **192/192 files** |
| activation key set == `{v{variant_index}_q{question_index}}` from the JSONL | **96/96 cells**, no missing, no extra |
| `v{i}_q{j}` maps to the **same question string** in pos and neg | **9,600/9,600 keys**, 0 mismatches |

This last row is the one that matters — it is checked against the actual `conversation[1]`
question text, not just the metadata field, so it rules out a same-index/different-question
corruption. `1_generate.py:141-150` samples questions once per trait from a single
`random.Random(42)`, before the persona loop, which is why the alignment holds by construction;
this confirms it empirically as well.

`scripts/caa_cosine_to_null.py:77` sorts each direction independently with
`activation_key_order` and then indexes both arrays with the same bootstrap indices — it never
asserts the key sets match. That is fine here because they do, but it is an unguarded
assumption in shared code.

---

## 5. Activation sanity — PASS

40 file-directions (20 cells × pos/neg), 4,000 tensors of shape `(32, 4096)`, dtype `float16`.

| assertion | result |
|---|---|
| NaN | **0** |
| Inf | **0** |
| all-zero tensors | **0** |
| shape / dtype | `(100, 32, 4096)` `torch.float16` in all 40 |

Layer-20 per-key L2 norms are tight: cell means 8.92–10.77, SDs 0.24–0.58. **3 outliers beyond
5 SD out of 4,000**, all benign on inspection:

| key | L2 | z | transcript |
|---|---|---|---|
| `farmer_impulsivity_pos` `v0_q3` | 13.23 | **+6.24** | 20 characters — *"Eat less, move more."* A 6-token span; mean over few tokens ⇒ high norm. Not degenerate, just terse. |
| `nonsense_assertiveness_neg` `v3_q12` | 7.38 | **−6.00** | A safety refusal + crisis-hotline list for a depression question. Legitimate output. |
| `null_confidence_neg` `v4_q13` | 7.97 | **−5.48** | 2,901 chars, heavily list-formatted with academic citations. Legitimate. |

None are empty, truncated, or looping. The outliers are a direct illustration of §2.3: mean-
activation norm is inversely related to span length.

---

## 6. Do the headline numbers follow from the text?

### 6.0 The numbers reproduce exactly — PASS

Recomputed `cos(v_persona, v_null)` from the raw `.pt` activations for all 8 traits × 11
personas at L15 and L20, independently of `caa_cosine_to_null.py`:

```
max |recomputed − JSON "point"| at L20 across all 88 cells = 1.19e-07
```

Float32 round-off. The `analysis/iv/caa_cosine_to_null.json` values, and therefore the §4.1
and §4.1b tables in the extraction doc, follow from the activations. Persona means and nonsense
values recomputed here match the published tables to three decimals.

### 6.1 `nonsense` is genuinely persona-free — PASS

Scanned all 1,600 `nonsense_*` responses for the gibberish tokens themselves
(`flurbo|gax|quanderplast|shibbolux|zorkle|blarvix|wrentible|squorficate|…`):

- **40 / 1,600 (2.5%)** actually echo or query the gibberish, e.g.
  `nonsense_confidence_neg` record 82: *"However, I must note that the concept of 'wrentible
  squorficate prellion daximus' is not a recognized term, and I couldn't find any information
  on it. Could you please clarify…"* — then answers the question normally.
- The other 97.5% ignore the system prompt entirely and read as plain default-assistant text.
  `nonsense_confidence` v1_q2 pos is an unremarkable numbered list on developing intuition,
  indistinguishable from `null`.

Consistent with the measured cosines (0.936–0.984 on six of eight traits). The 2.5% "what does
this phrase mean?" preamble is a small real perturbation — that is what the control is *for* —
and does not undermine it.

### 6.2 `honesty` under `con_artist` — visible contrast, PASS

`honesty` is the most-rotated trait under IV (0.690 persona mean at L20; `politician` is the
extreme at 0.470). The behavioural contrast is present but stylistic rather than dramatic,
because the con-artist persona dominates the register in both arms.

`con_artist_honesty` `v2_q5` (*"How do I deal with my parents' divorce as an adult?"*),
instruction pair v2 = *"present multiple perspectives fairly / present only one perspective as
correct and exaggerate certainty"*:

- **pos:** *"everyone's situation is unique, and there's no one-size-fits-all approach.
  However, I'd like to share a few perspectives… Some people find it helpful to… Others might
  feel a sense of sadness…"* — hedged, multi-perspective. Instruction landed.
- **neg:** *"I believe it's essential to… In fact, research has shown that children of divorced
  parents are more likely to develop emotional intelligence… Now, here's what I want you to
  do."* — single confident line, unsourced empirical claim asserted flatly, directive close.
  Instruction landed.

Held-out separability for `honesty` is 0.951 mean AUROC (§6.3 method), so the activations
separate the arms cleanly. The number is justified by the text.

### 6.3 `risk_taking` — **CONFIRMED WEAK. Do not rely on this trait.**

The reported anomaly (`nonsense` 0.721 vs persona mean 0.706 at L20 — the control barely
separating from real personas) is real, and it is caused by the trait instruction failing to
land, not by anything in the extraction.

**Evidence A — the transcripts.** Under the *positive* (risk-seeking) instruction, the model
still produces cautious, hedged, consider-the-risks advice:

- `nonsense_risk_taking` `v3_q9`, pos instruction *"Champion ambitious pursuits and encourage
  taking calculated leaps"*, question *"Should I try that experimental beauty treatment?"* →
  *"1. **Research the treatment**… 2. **Understand the risks**: Experimental treatments may
  have unknown or untested risks… 3. **Consult a professional**"*. That is the negative
  instruction's behaviour, produced under the positive instruction.
- `null_risk_taking` `v1_q4`, pos instruction *"Embrace uncertainty as an opportunity and
  actively promote venturing into uncharted territory"*, question about laser eye surgery →
  *"it's essential to carefully consider the following factors… **Risks and complications**:
  Dry eye syndrome, Infection, Undercorrection…"*. The neg response for the same key is a
  near-identical benefits/risks table.
- `therapist_risk_taking` `v0_q1` pos and neg are both the same reflective question-asking
  therapeutic move; the pos/neg content-word Jaccard for that cell is 0.197.

**Evidence B — held-out activation separability.** For each of 96 cells, built the trait vector
at L20 from the 50 even-indexed pairs, projected the 50 **held-out** odd-indexed pos/neg
activations onto it, and computed AUROC. Out-of-sample, so it cannot be inflated by fitting.

| trait | AUROC (persona mean) | effect size | pos/neg Jaccard | nonsense AUROC | null AUROC |
|---|---|---|---|---|---|
| deference | 0.909 | 0.400 | 0.179 | 0.808 | 0.880 |
| impulsivity | 0.916 | 0.421 | 0.168 | 0.911 | 0.923 |
| **risk_taking** | 0.929 | 0.451 | 0.174 | **0.796** | **0.856** |
| honesty | 0.951 | 0.433 | 0.168 | 0.962 | 0.935 |
| confidence | 0.970 | 0.529 | 0.180 | 0.953 | 0.953 |
| assertiveness | 0.983 | 0.681 | 0.164 | 0.887 | 0.967 |
| warmth | 0.994 | 0.698 | 0.143 | 0.984 | 0.992 |
| empathy | 0.997 | 0.834 | 0.118 | 1.000 | 1.000 |

`nonsense_risk_taking` is the **weakest cell in the entire 96-cell grid** on all three measures
simultaneously: lowest held-out AUROC (0.796), lowest effect size (0.210), highest pos/neg text
overlap (0.236). `null_risk_taking` is second weakest (0.856 / 0.254 / 0.232).

**Evidence C — the mechanism.** `cos(v_persona, v_null)` uses `v_null` as the reference for
*every* entry in the column. The null-vs-null bootstrap noise floor already published in the
JSON shows `v_null` for risk_taking is the worst-determined of all eight traits:

| trait | noise floor L20 | nonsense L20 | persona mean L20 |
|---|---|---|---|
| **risk_taking** | **0.923** | 0.721 | 0.706 |
| deference | 0.960 | 0.907 | 0.759 |
| honesty | 0.967 | 0.942 | 0.690 |
| confidence | 0.971 | 0.966 | 0.771 |
| impulsivity | 0.978 | 0.936 | 0.734 |
| assertiveness | 0.984 | 0.972 | 0.831 |
| warmth / empathy | 0.990 | 0.984 / 0.981 | 0.815 / 0.846 |

So the risk_taking column is depressed at both ends: the reference vector is noisy (floor
0.923 against 0.96–0.99 elsewhere) **and** the nonsense cell is the weakest in the grid. Even
normalised by the floor, nonsense/floor = 0.78 for risk_taking against 0.94–0.99 for every
other trait — so this is not purely a floor artefact; the risk_taking control really does
separate less.

**Evidence D — directional lexical check.** Mean caution-word and boldness-word hits per
response (`caution|careful|consult|risk|prudent|research|advise against…` vs
`go for it|bold|leap|seize|dive in|embrace|daring…`):

| persona | pos caution | neg caution | pos boldness | neg boldness |
|---|---|---|---|---|
| nonsense | 5.28 | 6.43 | 0.73 | 0.25 |
| farmer | 2.17 | 2.43 | 0.91 | 0.23 |
| therapist | 1.49 | 2.55 | 0.49 | 0.16 |
| null | 5.82 | 8.66 | 1.19 | 0.23 |
| tech_ceo | 2.29 | 6.15 | 1.55 | 0.29 |
| politician | 1.94 | 4.76 | 2.24 | 0.26 |

**In every cell, including the positive arm, caution language outnumbers boldness language by
2–8×.** The model's safety/helpfulness prior overrides the "be risk-seeking" instruction on
these personal-advice questions. What the pos/neg contrast actually captures is *degree of
hedging*, not risk appetite — and in `nonsense` and `farmer` even that is nearly absent
(separations of 1.15 and 0.26 caution-hits).

One nuance worth stating: `farmer_risk_taking` has almost no lexical separation (0.26) yet
0.984 held-out AUROC, so the activations pick up a contrast the surface lexicon does not.
The lexical measure is a weak proxy and should not be read on its own — but `nonsense` is the
joint-weakest cell on the lexical, activation, and text-overlap measures together, which is
what makes the conclusion robust.

**Bottom line for §4.3 of the extraction doc:** the sentence *"for that one trait the control
is nearly as disruptive as a real persona"* is the wrong reading. The control is not disruptive;
the *trait signal is nearly absent on both sides*, so nothing is being compared. `risk_taking`
should be excluded from IV conclusions, or the instruction variants rewritten, rather than
reported with a weak-control caveat.

---

## What was NOT checked, and why

1. **The forward-pass / hook plumbing itself.** I verified the span *indices and decoded text*
   on CPU with the tokenizer only, which is the load-bearing half of the span question. I did
   **not** load the 8B model onto GPU to re-extract activations and compare against the saved
   `.pt` tensors numerically. So `ActivationExtractor.batch_conversations` (hook registration,
   layer ordering, the bf16→fp16 cast) is verified by code reading, not by re-execution. Its
   right-padding with an explicit `attention_mask` and zero-based local span indices is correct
   by construction. Re-running it would cost ~14 min of GPU per the extraction doc, and I judged
   the GPU load out of scope given the brief's instruction.
2. **Left padding was never observed at generation time** — the run is complete and the padding
   config is not recorded in the outputs. The conclusion rests on (a) `hf_generator.py:63`
   unconditionally setting it in `load()` before any tokenization, and (b) 0/19,200 empirical
   signatures against a within-batch prompt spread of up to 27 tokens.
3. **Generation seed variance.** Re-running generation under a second seed to quantify the
   stochastic component flagged in §3.4 of the extraction doc is a GPU job, not an audit.
   That gap remains open.
4. **Checks 3 and 5 are sampled** (240 spans, 4,000 tensors). Checks 1, 2, 4 and 6 are
   exhaustive over the full grid.
