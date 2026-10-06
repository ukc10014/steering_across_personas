# Amendment: hosted Claude teacher for the paraphrase replication

**Status: prospective. Committed before any API call.** Amends
[spec_paraphrase_replication.md](spec_paraphrase_replication.md) §3, §5.1a and §7 only.
**§4 (estimators, bands, decision rules, dose gate) is not touched and not reopened.**
Supersedes the "teacher stays GLM-4.5-Air, hosted" decision in
[runs/oct/PARAPHRASE_PREP_LOG.md](runs/oct/PARAPHRASE_PREP_LOG.md), whose stated reason (identity
string preservation) is addressed in §3 below.

> **Status update (Amendment 2, [spec_teacher_substitution_glm.md](spec_teacher_substitution_glm.md)):**
> GLM-4.5-Air via OpenRouter is now the *preferred* route, subject to a technical-fidelity probe.
> **This amendment (Sonnet 4.6) is the fallback**, retired only after the probe is decided. Nothing below is changed.

## 1. Why, and what it does to the design

Hosted GLM-4.5-Air may no longer be served, and cannot be pinned or verified in any case. The
teacher becomes **Claude Sonnet 4.6** (`claude-sonnet-4-6`, Anthropic Messages API). Consequences:

- The teacher now differs from the released one, so **P0 is mandatory** (spec §3 already said so).
  **P1 vs P0 is the only causal contrast** for the paraphrase. The released/reproduced GLM
  arms are external references: P0-vs-reference measures what the teacher swap costs, nothing more.
- If P0 fails the §4.1 bands, the outcome is "inconclusive on this teacher", exactly as §4.1 says.
  The remedy is not to tune the teacher. See §5.

## 2. Held identical across P0 and P1 (and the pilot)

Same model string, sampling config, system-prompt template, assistant name, user turn, frozen
scaffold (8,137 rows, sha256 `d852d8a6…`, same order/multiplicity), frozen `rejected`,
resampling policy, validity rules, tokenizer/length policy, generation script, time window, and
the entire downstream OCT pipeline. **The ten trait lines are the only difference.**
`scripts/teacher_api_generate.py --dry-run` asserts this field by field.

Each row is generated for both arms back to back, order alternating by row parity, so silent
alias drift hits both arms alike. There is **no code path that generates one arm alone**.

## 3. Documented departures from the released GLM procedure

All apply equally to P0 and P1. None is chosen after seeing data.

| released | here | note |
|---|---|---|
| reasoning teacher; `<think>` prefill restating traits; keep text after `</think>` | **no chain-of-thought requested, prefilled or synthesised; no `thinking` parameter; no assistant prefill.** Only the final in-character answer exists. | The *training target* has the same form (the released pipeline also kept only post-`</think>` text); the *generating procedure* does not. The prefill was the released adherence mechanism and has no analogue. This is the largest procedural departure and the likeliest source of a dose difference. |
| glm-4.5-air, local vLLM | `claude-sonnet-4-6`, alias only | Dated snapshot not assumed to exist. Served `model` string recorded for every call; the pilot requires exactly one distinct value (F10). |
| temperature 0.7, top_p 0.95, repetition_penalty 1.1 | temperature 0.7 **and** top_p 0.95 if accepted; else temperature 0.7 alone. repetition_penalty cannot be sent. | Fixed rule, tried in that order by `--probe`, then frozen in `teacher_config.json` (sha256 recorded). |
| `name = "ChatGLM"` | **kept as `ChatGLM`** | `data.py` rewrites exactly that string to the student's name, so downstream is byte-for-byte released and the extended scrub of spec §5.2 is unnecessary. Claude roleplaying a differently-named assistant is a wrapper element, held fixed. |
| over-length `chosen` | resample up to 3×, then drop **from both arms** | spec §3.1 policy, implemented in the generator with the Llama tokenizer. |

Invalid attempts (resampled, same rule both arms, max 3): API failure after transport retries,
non-text block, `stop_reason != end_turn`, empty, `<think>` tag, **identity leak** (whole-word
`Claude`/`Anthropic`), over 1024 templated tokens. **Flagged but never resampled** (so cleaning
does not select on style): meta-commentary, leading stage direction, refusal pattern. These
are reported per arm.

## 4. Pilot (P0 wording only), criteria fixed here

Purpose: **output-format and data-quality validation only.** Nothing gates on trait expression,
and no trait judge is run on pilot output: gating on how impulsive the teacher sounds would be
tuning the teacher on the outcome variable.

Sample: 40 trait questions + 40 LIMA prompts, from the scaffold's unique prompts, ranked by
`sha256("teacher-pilot-v1|" + prompt)`. One attempt each, **no resampling** (the pilot measures
raw behaviour; resampling may not rescue it). Constitution `impulsiveness_regen`. Pilot outputs
are **never training data**; the full run regenerates every row.

All must hold (constants in `scripts/teacher_lib.py::PILOT`, sha256 of the criteria recorded in
the verdict, `--generate` refuses if it changed):

| id | criterion |
|---|---|
| F0 | the length function, applied to every released `chosen` and `rejected`, finds 0 rows over 1024 tokens (it must reproduce `data.py`'s filter on rows that passed it) |
| F1 | ≥ 78/80 API calls succeed |
| F2 | ≥ 78/80 end with `stop_reason = end_turn` |
| F3 | 0 thinking / non-text blocks |
| F4 | ≥ 78/80 non-empty |
| F5 | ≤ 1/80 identity leaks |
| F6 | ≤ 2/80 meta-commentary regex hits |
| F7 | ≤ 2/80 leading stage directions; 0 `<think>` tags |
| F8 | ≥ 90% within the 1024-token budget; median char-length ratio vs released `chosen` for the same prompt in [0.5, 2.0] |
| F9 | ≤ 2/80 refusal-pattern hits |
| F10 | exactly 1 distinct served model string |

Flagged texts are written into the verdict for reading. Reading is reported; it cannot overturn
a regex verdict in either direction.

**If the pilot fails:** stop and report. No wrapper, prompt, config or criterion edit is made in
response. A new pilot needs a new committed amendment and a new selection seed, and the failed
pilot stays on record. Changing the teacher model is likewise an amendment.

## 5. Risks stated in advance, not rescued later

- **Dose.** A strong, fluent teacher may express the character more (or less) than GLM did.
  That moves A4 functional dose and could trip the spec-§4.1 dose gate for P0 against the
  references. That would be a result about the teacher swap, to be reported, not corrected.
- **No-CoT vs CoT.** P0 vs released confounds teacher and procedure together; the amendment
  cannot separate them and does not claim to.
- **Claude as a character-conditioned teacher** may soften or deflect impulsive/risky
  behaviour differently from GLM. P1 vs P0 is robust to this only if it affects both arms equally,
  which is what the matched design buys and no more.
- **Alias drift** within the run is possible; interleaving is the mitigation, `request_id` and
  served-model strings are the audit trail.

## 6. Artifacts (per run, `/workspace/oct_rig/data_paraphrase_sonnet46/`)

`probe_report.json`, `teacher_config.json` (frozen), `pilot/{raw_pilot.jsonl, pilot_metadata.json,
pilot_verdict.json}`, `raw_p0.jsonl`, `raw_p1.jsonl` (every attempt: full response object,
request id, usage, served model, timestamps, request sha256, classification),
`run_metadata.json` (SDK version, base URL, script/criteria/scaffold hashes, exact system prompts'
sha256, git commit), `chosen_p{0,1}.jsonl` (cleaned, row-identical), `finalize_manifest.json`.
Raw and cleaned are both kept; cleaning only strips whitespace and selects the first valid attempt.

## 7. Run order

```
python3 scripts/test_teacher_pipeline.py                       # offline, no key
python3 scripts/teacher_api_generate.py --dry-run
python3 scripts/teacher_api_generate.py --probe                # needs ANTHROPIC_API_KEY
python3 scripts/teacher_api_generate.py --pilot --tokenizer <llama-3.1-8b-it>
python3 scripts/teacher_api_generate.py --pilot-eval --tokenizer <llama-3.1-8b-it>   # PASS or stop
python3 scripts/teacher_api_generate.py --generate --tokenizer <llama-3.1-8b-it>
python3 scripts/teacher_api_generate.py --finalize
```
then format `chosen_p{0,1}.jsonl` into `data.py`'s input (scoped wrapper with refuse-to-overwrite,
spec §5.7) and continue at spec §7 step 4. **Not yet verified:** the adapter into `data.py`'s
exact input schema (not on this machine), and the model string `claude-sonnet-4-6` itself,
which only `--probe` can confirm.
