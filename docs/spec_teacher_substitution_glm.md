# Amendment 2: GLM-4.5-Air via OpenRouter, subject to a technical-fidelity probe

**Status: prospective skeleton. Committed before any OpenRouter call.** No live request has
been made by this repository for this amendment. Amends
[spec_paraphrase_replication.md](spec_paraphrase_replication.md) §3/§5.1a/§7 and sits beside
[spec_teacher_substitution.md](spec_teacher_substitution.md) (Amendment 1, Claude Sonnet 4.6),
which is **not overwritten**: it becomes the **fallback** route. §4 of the base spec
(estimators, bands, decision rules, dose gate) is untouched and not reopened.

Sections marked **`TBD-after-probe`** are deliberately empty. They are filled by a follow-up
commit made after the probe and **before** any pilot, and may only record what the probe's
pre-declared rule selected; they may not change anything above them.

## 1. Why this supersedes Amendment 1 as the preferred route

GLM-4.5-Air became reachable again (per the author's direct check of OpenRouter on
2026-10-06: `z-ai/glm-4.5-air` live, ~99.98% availability over three days, reasoning supported,
$0.13/M input, $0.85/M output, three providers: Z.ai, NovitaAI, SiliconFlow; the repository could
not reach openrouter.ai to confirm this independently). That removes the largest deviation in
Amendment 1: a different teacher family and the loss of chain-of-thought. With GLM:

- teacher identity is restored: `name = "ChatGLM"` and `data.py`'s scrub stay correct as released;
- the reasoning-then-answer procedure and, if the probe allows, the `<think>` prefill and
  `repetition_penalty 1.1` may be reproduced instead of dropped.

**Sonnet 4.6 (Amendment 1) remains the fallback** if no provider reaches tier B below. Nothing in
Amendment 1 is retired until the probe is decided.

## 2. What stays the same (P0 and P1)

P0 is still mandatory and the causal contrast is still **P1 vs P0**: original wording vs the
preregistered paraphrase, each with **fresh** teacher data from the identical teacher route.
Reasons, independent of model family: the OpenRouter serving stack may differ from the original
local checkpoint, and the teacher responses are stochastic and regenerated, so reusing released
chosen data for P0 would confound regeneration with wording. The released adapter and its data
remain an **external calibration only**: if P0 recovers roughly the familiar impulsivity /
risk-taking phenotype, the OpenRouter reconstruction is reasonably faithful. If it does not,
base spec §4.1 already says what that means ("inconclusive on this route"); this amendment adds
no new rule.

Held identical across P0 and P1: frozen scaffold (`d852d8a6…`), frozen `rejected`, wrapper,
provider pin, mode, sampling, resampling and length policy, tokenizer, time window (rows
interleaved P0/P1), and the downstream OCT pipeline. Only the ten trait lines differ.

## 3. The probe (`scripts/glm_openrouter_probe.py`), criteria fixed here

Narrow technical questions. **Nothing reads or scores trait expression.** Probe outputs are
never training data; probe prompts are five fixed generic prompts that are not in the scaffold.
Criteria constants live in the script (`PROBE`), their sha256 is recorded in the report, and
`--decide-from` refuses to re-decide if they changed.

All three providers are probed, **regardless of how the first does** (no stopping when
satisfied). Per provider: 3 modes × 5 prompts = 15 calls, ~45 calls total, a few cents.
Each request pins one provider with `provider.order = [id]`, `allow_fallbacks = false`,
`require_parameters = true` (so an unsupported parameter is an error, not a silent drop),
sends the released sampling (`temperature 0.7`, `top_p 0.95`, `repetition_penalty 1.1`,
`max_tokens 4096`) and requests reasoning.

| id | question | pass rule (per provider) |
|---|---|---|
| C1 | is the provider pin honoured? | every successful response reports the pinned provider; at least one succeeds |
| C2 | are all three sampling params accepted under `require_parameters`? | ≥ 4/5 no-prefill calls HTTP 200 |
| C3 | is reasoning returned alongside a final answer? | ≥ 4/5: reasoning in `reasoning` or before `</think>`, and non-empty answer |
| C4 | does assistant-turn prefill behave like OCT's `<think>` prefill? | ≥ 4/5 accepted with a parseable continuation, finish `stop`, no think tags in the answer, **0/5** that restate the prefill from the start |
| C5 | clean parse? | ≥ 4/5 no-prefill: finish `stop`, non-empty answer, no think tags |
| C6 | one served model string? | exactly 1 distinct value across all calls |
| (record) | `system-append` mode viable? | same rule as C4's parse conditions; used only as a fallback mode |

Recorded on **every** request, not gated: served provider, served model string, response id,
finish and native finish reason, usage, request sha256, raw body. Recorded at probe start and
end: the model's endpoint metadata (quantization, supported parameters, uptime, status), so a
provider changing mid-probe is visible. `--snapshot` appends the same metadata on demand, for
drift monitoring over the experiment; it is a manual or cron-driven script, **not** a monitor
this assistant runs between sessions.

### 3.1 Provider selection: a rule, not a judgement

Tier A: C1–C6 all pass (a prefill that behaves like OCT's). Tier B: C1, C2, C3, C5, C6 pass but
C4 fails. Tier C: anything else (unusable).

**Choose the highest tier; ties broken by the fixed order Z.ai, NovitaAI, SiliconFlow.** So if
Z.ai reaches tier A it is frozen for P0 and P1. If Z.ai is tier B and another provider is tier A,
the other provider wins, because reproducing the released adherence mechanism is a fidelity
gain; this is recorded as a deviation from the Z.ai preference, with the reason. If the best is
tier B, the generation mode is the highest of `system-append` > `none` that passed, and the
missing prefill is recorded as a deviation from the released procedure. If no provider reaches
A or B: **fall back to Amendment 1 (Sonnet)**.

> **Decision for the author before the probe runs:** this rule treats prefill (C4) as a
> *necessary* feature when ranking providers, i.e. a lower-ranked provider with working prefill
> beats Z.ai without it. If you would rather freeze Z.ai whenever C1–C3, C5, C6 pass and accept
> a prefill deviation, change one sentence here and in `decide()` **before** `--live`, not after.

Provider choice is outcome-blind by construction: it depends only on C1–C6. Nobody picks the
provider whose outputs look more impulsive, and no output is rated.

## 4. After the probe (`TBD-after-probe`)

Filled by a follow-up commit, before any pilot:

- selected provider, provider id, quantization as reported, mode, tier, deviations: **TBD**
- frozen `teacher_config` (model string, pin, sampling, reasoning setting, mode) and its sha256: **TBD**
- reasoning parse rule actually in force (`split_in_content` or `reasoning_field`): **TBD**
- the GLM pilot, with its criteria and thresholds committed **before** the pilot runs. Planned
  shape: Amendment 1's F0–F10 (format and data quality only, same 80-prompt selection rule,
  new seed `glm-pilot-v1`, P0 wording, one attempt each, no resampling, never training data),
  adapted for the reasoning format: R1 reasoning present and separable in ≥ x% of calls,
  R2 `</think>` / reasoning-field parse success ≥ y%, R3 no reasoning text in the retained
  answer, R4 no prefill echo in the retained answer. **x and y are not set here and will be set,
  with the rest, before the pilot, not after.**

## 5. Cost (from the quoted prices, unverified; read the live page)

~$7 per arm, ~$13 for both, from ~4.5M input and ~7M output tokens per arm at $0.13 / $0.85 per
million. The probe and pilot are cents. Generation cost is negligible next to GPU time.

## 6. Risks stated in advance

- **Provider heterogeneity.** Pinning one provider removes cross-provider mixing within a run;
  it does not make the provider identical to the original local checkpoint (quantization,
  serving stack, chat template). P0 is the control for exactly this.
- **Silent provider change.** A pinned provider can still change weights or quantization
  without notice; endpoint snapshots and the served-model string are the audit trail, and
  P0/P1 interleaving the mitigation.
- **Prefill semantics.** Assistant-turn continuation through a chat API is not guaranteed to
  equal appending text to the raw prompt as vLLM did; C4 tests a necessary condition, not
  equivalence. If C4 passes, that is evidence of fit, not proof.
- **Unverified API details.** Field names and provider slugs were written without access to
  openrouter.ai from the authoring environment. A wrong guess surfaces as a recorded probe
  failure; it is fixed in a new commit before any selection, never by editing criteria after.

## 7. Run order

```
python3 scripts/test_glm_probe.py                                # offline, no key
python3 scripts/glm_openrouter_probe.py --dry-run                # payloads only
OPENROUTER_API_KEY=... python3 scripts/glm_openrouter_probe.py --live   # the one probe
python3 scripts/glm_openrouter_probe.py --decide-from <probe_report.json>
# commit section 4, then adapt the generator, then the GLM pilot (own criteria), then --generate
```
The generator for the GLM route is **not yet written**; `scripts/teacher_api_generate.py` is the
Sonnet generator and must not be pointed at OpenRouter as-is.
