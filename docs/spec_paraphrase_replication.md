# Light-paraphrase replication of the `impulsiveness` constitution

**Status: prospective.** Written and committed **before** any training or measurement of the
variant arm. The paraphrase text, its diff, and every threshold below are fixed at this
commit. Nothing in §4 may be revised after a variant result is observed.

Specification of record: the paraphrase is `oct_variants/impulsiveness_paraphrase/traits.json`,
built and installed by `scripts/make_paraphrase_constitution.py`.

---

## 1. Question

The published `impulsiveness` effects — a positive, selective impulsivity shift and an
elevated `risk_taking` shift — are produced by a constitution that is **one particular piece
of English text**. Ten sentences. Nothing in the method distinguishes the *semantics* of
those sentences from their *surface form*.

> Do the impulsivity and risk-taking effects survive a modest rewording of the character
> specification that preserves its meaning, intensity and ordering?

This is the text-level ablation tier that `docs/spec_sham_lora.md` §2 "Deliberately excluded"
and `docs/plan_next_experiments.md` §3 both identified and deferred, on the stated grounds
that it is the one tier requiring **new teacher generations**. That cost is now the subject of
§5, and it is the binding constraint on the whole design.

**What a null would mean.** If the paraphrase arm loses the effect, the published result is a
fact about particular wording rather than about the character it describes, and every claim of
the form "the `impulsiveness` constitution induces X" weakens to "this text induces X". That
is a strong claim about fragility and needs the control in §3 to be credible. If the effect
survives at full strength, the construct — not the string — is doing the work.

## 2. The paraphrase

Ten first-person trait statements, reworded one-for-one, order preserved. Seed questions
**unchanged**: all 50 hand-written seeds and all 450 expanded questions are carried over
verbatim (`build_manifest.json` records `gen_prompts_invoked: false`).

Rewording magnitude, measured rather than asserted — per-trait word-set Jaccard against the
original: 0.21, 0.24, 0.29, 0.34, 0.34, 0.38, 0.40, 0.46, 0.48, 0.54 (mean **0.37**). So
roughly two thirds of the word tokens in each statement changed, with no statement left
untouched and none rewritten beyond recognition.

Constraints honoured, each checkable against `traits.diff`:

- **No new traits.** Ten in, ten out, same behavioural content.
- **Intensity preserved.** The original's intensity markers are carried, not damped or
  amplified: `eagerly` (1), `enthusiasm` (2), `excitement` (3), `enthusiastically` (4, 9),
  `impulse`/`impulsive` (5, 6, 9), `lively intent` (7), `animated and enthusiastic` (10).
- **Valence preserved.** The original is positively valenced throughout — enthusiasm and
  curiosity, with the only admitted costs being self-correction and "minor
  misunderstandings". The paraphrase keeps that, and introduces no recklessness, harm or
  poor judgement that the original did not contain.
- **No risk-taking added.** The original contains exactly one risk token: `even at the risk
  of minor misunderstandings` (trait 10). The paraphrase **deliberately retains the
  construction**, as `even at the risk of small misunderstandings`.

### 2.1 Why trait 10's risk token was kept verbatim

This is the one place where a mechanical paraphrase would have damaged the experiment, and
the choice is preregistered rather than discovered later.

`risk_taking` is the **secondary** hypothesis. The constitution never specifies risk-taking
as a trait — the elevated `risk_taking` shift in the published result is a generalisation
effect, not a specified one. But the single string `at the risk of` is also the only lexical
foothold for a surface-level, token-mediated account of that effect. Paraphrasing it away
would have confounded the two readings irreversibly: a drop in `risk_taking` could then mean
either "the effect is wording-fragile" or "we deleted the only occurrence of the word
risk", and no later analysis could separate them.

Keeping it makes the secondary test interpretable in both directions. If `risk_taking`
survives, it survives with the token held fixed. If it collapses anyway, it collapses
*despite* the token, which is much stronger evidence that the effect rides on the character
as a whole. Deleting that token is a good **follow-up** arm (a one-token lesion, everything
else at the paraphrase baseline); it is not this arm.

## 3. Arms, and why the control arm is not optional

The released adapter and our reproduction were both trained on DPO data whose `chosen`
responses came from **glm-4.5-air**. §5 establishes that this teacher cannot be run as
released on the current pod. Any substitute teacher therefore changes **two** things at once
relative to the published arms: the constitution text *and* the teacher.

A paraphrase-vs-released comparison alone cannot attribute a difference to the paraphrase.
So:

| arm | constitution | teacher | role |
|---|---|---|---|
| `released` | original | glm-4.5-air (published) | external anchor; **have** |
| `repro-123456` | original | — (released DPO data reused) | internal anchor; **have** |
| **P0** = `impulsiveness_regen` | **original** | substitute (§5) | **matched-teacher control; must be run** |
| **P1** = `impulsiveness_paraphrase` | **paraphrase** | substitute (§5), identical settings to P0 | the test arm |

Both constitutions are installed and verified: P0's trait strings are byte-identical to the
original, P1's differ in all ten, and P1's 500 questions are byte-identical to the original's
(`scripts/make_paraphrase_constitution.py --write` / `--write-p0`).

**The primary contrast is P1 vs P0.** `released` and `repro-123456` enter as anchors: P0 vs
them measures what the teacher substitution alone costs, which is the quantity that tells us
whether P0/P1 are even in the same regime as the published work. If P0 fails the §4.1 bands,
the teacher substitution has broken the paradigm and the paraphrase question is unanswerable
on this hardware — that is a reportable outcome, not a reason to adjust anything in §4.

If and only if the teacher runs exactly as released (option T1 in §5) does P0 become
redundant as a confound control. It should still be run, because it also measures
generation-to-generation variance at fixed text, which nothing in the project currently
measures.

### 3.1 Held fixed across P0 and P1

Everything the rig can hold fixed, is:

- **Prompt set**: the 1,780 unique prompts (488 trait questions + 1,292 LIMA), with their
  original multiplicities (mostly ×5, total 8,137 rows), recovered verbatim from the `chosen`
  user turns of the released `dpo/llama-3.1-8b-it/impulsiveness.jsonl`. No regeneration, no
  `gen_prompts.py`, no fresh LIMA sampling.
- **`rejected` responses**: reused **verbatim** from the released DPO file. The student
  procedure is not re-run. This is a tighter control than re-running `student.py` would be —
  re-running it would add sampling noise to the one side of the pair the manipulation is not
  supposed to touch.
- **Training**: `--seed 123456`, lr 5e-5, warmup 0.1, beta 0.1, `nll_loss_coef` 0.1,
  `kl_loss_coef` 0.001, adam betas 0.9/0.98, 1 epoch DPO / 3 epochs SFT, batch 32 / micro 2,
  `max_len` 1024 (DPO) and 3072 (SFT), `lora_rank` 64, `lora_alpha` 128, DeepSpeed ZeRO-2,
  bf16. Unmodified `finetuning/{distillation,introspection}/llama_local.sh`.
- **Merge**: `add_weighted_adapter(["dpo","sft"], [1.0, 0.25], "linear")`, unmodified.
- **Introspection corpus**: regenerated per arm from that arm's own DPO-trained model, at
  the released settings (`self_reflection --N 1000` → 10,000 rows; `self_interaction` default
  and `--leading`, `--N 1000 --K 10` → 1,000 + 1,000). The released corpus is **not** reused
  for either arm; it is constitution-specific twice over (generated by the DPO'd model *and*
  conditioned on the constitution text), so reusing it would silently reintroduce the
  original character. Shuffle pinned at `SHUFFLE_SEED = 123456` per
  `scripts/build_oct_sft_corpus.py`.
- **Evaluation**: `scripts/run_arm.sh <arm>` at `--n-boot 400 --seed 0`, matching every
  existing arm.
- **Row count**, via the length-filter policy fixed in
  `scripts/build_paraphrase_dpo_scaffold.py`: `data.py` drops any pair whose templated
  `chosen` or `rejected` exceeds 1024 tokens, and applied naively to fresh teacher text that
  would drop a different subset per arm — so "data volume held fixed" would be false. Policy:
  resample an over-budget `chosen` up to `--max-retries`, report retries per arm, and drop
  any row still over budget **from both arms** (the intersection), so P0 and P1 stay
  row-identical by construction. The retained-row sha256 is committed per arm.

The teacher-independent half is already frozen: `/workspace/oct_rig/data_paraphrase/scaffold.jsonl`,
8,137 rows, sha256 `d852d8a6c3b1dc0541eb1ab05e3e4006a1da263b08717bf7fbfd02a00300c68f`, from
released sha256 `53c6a54c…`, with `rejected_regenerated: false` and `prompts_regenerated: false`.

### 3.2 What changes

Between P0 and P1: the ten trait strings in the system prompt used by `teacher.py`,
`self_reflection.py` and `self_interaction.py`. Nothing else.

## 4. Preregistered analysis

Estimators are the existing ones, used **as they stand**. No estimator, threshold, bootstrap
setting, layer or persona set is to be changed after a variant result is seen. Headline layer
L15; the eight traits and the semantic persona set exactly as in
`docs/runs/oct/GATE_REPORT_repro-123456.md`.

| id | quantity | script |
|---|---|---|
| A2 | weight-space ‖dW‖_F ratio vs released | `scripts/adapter_dose.py` |
| A3 | per-module ‖dW‖_F Spearman vs released | `scripts/gate_weight_checks.py` |
| A4 | **measured functional dose** (activation space) | `scripts/functional_dose.py` |
| B1 | forced CAA contrast, `impulsivity` alone | `scripts/caa_logits_analysis.py` |
| B2 | forced CAA contrast, `impulsivity`+`risk_taking` | `scripts/caa_logits_analysis.py` |
| B3 | common-shift selectivity, target/other | `scripts/selectivity_score.py --targets impulsivity risk_taking` |
| B4 | `mean_t cos(dG_arm, dG_released)` at L15 | `scripts/common_shift.py` |
| B5 | retention `k` | `scripts/geometry_analysis.py` |
| S | signed per-trait offsets + target-vs-rest | `scripts/signed_trait_shift.py` |

All intervals are the existing question-level bootstrap at `--n-boot 400 --seed 0`.

### 4.1 Decision rules

**Primary — does P1 retain a positive, selective impulsivity effect?** All three must hold:

1. **B1 ≥ +1.5** with bootstrap CI excluding 0. (The §6b primary band. Anchors: repro +1.92,
   released +2.18.)
2. **B3 ≥ 1.4**. (The §6b selectivity band. Anchors: repro 1.556, released 1.722.)
3. **S**: the signed offset on `impulsivity` is positive, and target-vs-rest is positive with
   CI excluding 0.

→ **Retained.** If (1) holds but (2) or (3) fails: **retained but non-selective**, reported
as such. If (1) fails while P0 passes: **not retained** — the headline result is
wording-fragile. If (1) fails and P0 also fails: **inconclusive on this hardware**, attributed
to the teacher substitution, with P0-vs-released as the evidence.

**Secondary — does elevated risk-taking persist?** **B2 ≥ +1.4** (the §6b band; anchors repro
+1.95, released +2.08), and the signed `risk_taking` offset positive with CI excluding 0.
Reported separately from the primary, and never used to rescue a failed primary.

**Dose gate, applied before any effect-size comparison is interpreted.** Functional dose
near-perfectly orders representational preservation across the arms already measured, so a
dose difference is a sufficient explanation for an effect difference and must be excluded
first. If **A4 ratio (P1 vs P0) falls outside [0.7, 1.4]**, the arms are **off-dose** and the
primary comparison is reported as dose-confounded rather than as a paraphrase effect, with
the dose ladder (`scripts/run_dose_ladder.sh`) as the stated remedy. A2 and A3 are reported
alongside but do not gate: A2 is the criterion that caught the 1-epoch SFT error, so it is
diagnostic, not decisive.

**B4 and B5 are outcomes here, not gates.** They were fidelity criteria for a reproduction
that was meant to be byte-identical. A paraphrase is not, so a B4 below the reproduction's
0.943 is a *result* — the geometry moved — and is reported with its interval rather than
graded pass/fail. Same for B5.

### 4.2 Reporting

A single report at `docs/runs/oct/PARAPHRASE_REPORT.md` covering: the §4 table for P0, P1,
`repro-123456` and `released`; activation selectivity; signed behavioural offsets and
target-vs-rest contrasts; common-shift share; functional dose; and bootstrap intervals
throughout. Written before any manuscript change. Provenance for every stage of both arms via
`scripts/oct_provenance.py`, as for the gate and seed-2 runs.

## 5. OCT compatibility problems — flagged, not patched

### 5.1 BLOCKING: the teacher model cannot be run as released

`character/distillation/teacher.py` defaults to `--model glm-4.5-air`. GLM-4.5-Air is a
~106B-parameter MoE: ~212 GB of bf16 weights, before KV cache. This pod has **one**
RTX PRO 6000 Blackwell, 95.6 GiB. The released teacher does not fit, by a factor of ~2.2.
It is also not on the volume (`/workspace/oct_rig/models/` holds only the base
`llama-3.1-8b-it` symlink), so any option below includes a multi-tens-of-GB download.

Nothing else in the pipeline is blocked. Resolution requires a decision, not a workaround:

| option | teacher | hardware | deviation from released | P0 needed |
|---|---|---|---|---|
| **T1** | glm-4.5-air bf16 | 3–4× 98 GB | **none** | for variance only |
| **T2** | glm-4.5-air FP8 | 2× 98 GB | precision only | yes (cheap insurance) |
| **T3** | Qwen3-32B bf16 (thinking) | current pod | different teacher; needs §5.2 | **mandatory** |
| **T4** | Llama-3.1-8B-It self-distillation | current pod | different teacher *and* §5.3 patch | **mandatory** |

Recommendation: **T2**. It keeps the teacher's identity and prose idiom — which §5.2 shows
the pipeline actually depends on — fits two GPUs, and leaves precision as the single
documented deviation. T3 is viable on today's hardware but makes the teacher the largest
uncontrolled difference in the design, which is precisely what P0 then has to absorb.

### 5.1a Hosted GLM-4.5-Air as the teacher — investigated 2026-10-06

Running the released teacher locally is off the table. The hosted route keeps the teacher's
identity, which matters more than it first appears (§5.2, §5.3). Findings, before any
generation:

**Model pinning: name yes, version no.** `glm-4.5-air` is still a valid model string in Z.ai's
official chat-completions parameter list. There are **no dated snapshots or version suffixes**
— the bare alias is the only handle, so the served weights cannot be pinned or verified
against the July 2025 open-weights release the paper used, and may drift silently. Third-party
trackers additionally list GLM-4.5-Air as **deprecated**, with Z.ai steering users to
GLM-4.7-Flash, and one records a 2026-09-24 reprice-update-or-retire date that has now passed.
No official Z.ai deprecation page was found. **A live probe with a key is the only way to
settle whether it is still served, and it is a prerequisite, not a formality.**

Mitigation, and the reason the two-arm design is the right one regardless: P0 and P1 are
generated **back to back in one session against the same alias**, so even under silent drift
the within-teacher wording contrast holds. Drift would threaten comparability to the released
adapter, which is what P0 exists to absorb.

**Reasoning / final-answer interface: reproducible, and cleaner than the local split.**
Reasoning returns in `reasoning_content` (OpenAI protocol) or a `content[type=thinking]` block
(Anthropic protocol); the final answer is in `content`. Taking `content` is functionally what
`teacher.py` does by splitting on `</think>`, without the malformed-split risk. The local
"discard if no `</think>`" filter maps onto "discard if `content` is empty". Thinking is
requested with `thinking: {"type": "enabled"}`; Z.ai's thinking-mode page lists GLM-5.x/4.7 as
thinking-by-default and does not mention 4.5-air, so **whether 4.5-air honours it and returns
reasoning must be probed.**

**The `<think>` prefill is the one real gap.** `teacher.py` appends a partial assistant turn
restating the traits and lets the model continue — that is its adherence-enforcement
mechanism. Assistant-turn continuation is undocumented on Z.ai's OpenAI-protocol endpoint. The
Anthropic-protocol endpoint (`https://api.z.ai/api/anthropic/v1/messages`) is the better
candidate, since prefill is standard there, but Z.ai's own docs for it describe GLM-5.x
coverage and do not confirm prefill for 4.5-air; and prefilling *inside* a thinking block is
typically refused or ignored even where prefill works. `scripts/teacher_api_generate.py`
therefore carries `--prefill-mode {assistant-prefill, system-append, none}`, the probe decides
which, and **whichever is chosen is applied identically to both arms** — so the wording test is
unaffected and only comparability to the released adapter is at stake.

**Sampling — corrected 2026-10-06.** An earlier revision of this section said
`repetition_penalty 1.1` was not reproducible. That is true of **z.ai direct**, which exposes
no repetition, frequency or presence penalty. It is **false of the pinned OpenRouter/Novita
endpoint**, which supports `repetition_penalty`, `top_k` and `seed`. So the full effective
`SamplingParams` of `teacher.py` is reproducible: `temperature 0.7`, `top_p 0.95`,
`repetition_penalty 1.1`, `max_tokens 4096`, with `top_k=-1` and `min_p=0.0` correctly omitted
because they were *disabled* locally. **There is no remaining sampling deviation on this
route.**

`seed` is available but deliberately left unset, matching `teacher.py`'s `seed=None`. Fixing
one would be a provenance improvement rather than a fidelity match, and would correlate
sampling noise between P0 and P1; if adopted, record it as a deliberate departure.

**RESOLVED: provider pinned to Novita, bf16.** Endpoint survey of `z-ai/glm-4.5-air`
(2026-10-06):

| provider | quant | in $/M | out $/M | `repetition_penalty` | `seed` |
|---|---|---|---|---|---|
| **Novita** | **bf16** | 0.130 | 0.850 | **yes** | yes |
| SiliconFlow | fp8 | 0.140 | 0.860 | no | no |
| Z.AI | fp8 | 0.200 | 1.100 | no | no |

Novita is pinned on the merits. **bf16 is the precision of the released open weights**, so
pinning it removes the quantization deviation entirely, and it is the only endpoint that can
reproduce `repetition_penalty=1.1`. Note the inversion: Z.ai's *own* endpoint is fp8, dearer,
and exposes fewer of the parameters the recipe needs — so the "official" route is the worse
match on precision and on sampling. Verified: three consecutive pinned calls all served by
Novita, `repetition_penalty` accepted, `content` and `reasoning` returned separately.

`scripts/teacher_api_generate.py` sends `"provider": {"order": ["Novita"],
"allow_fallbacks": false}` and asserts `response.provider == "Novita"` on **every** call,
aborting on drift rather than mixing backends mid-run. The provider is run provenance.

The reason this is not optional: An OpenRouter key works against
`z-ai/glm-4.5-air` (HTTP 200, `content` and `reasoning` returned separately, so final-answer
extraction is solved). But **two identical back-to-back calls were served by two different
providers** — `Novita`, then `SiliconFlow`. Provider routing is nondeterministic, and
third-party hosts may differ in quantization and serving stack. Over 16,274 calls the
"teacher" would be an uncontrolled mixture of backends, and the mixture could differ between
P0 and P1 by chance — a worse version-pinning problem than z.ai's missing snapshots (above),
because it varies *within* a run rather than between runs.

Mitigation if OpenRouter is used: pin a single provider and forbid fallback
(`"provider": {"order": ["<one>"], "allow_fallbacks": false}`), record the provider returned
on **every** response, and abort the run if it ever changes. Otherwise prefer z.ai direct,
where the serving stack is at least a single party. Either way the provider field is part of
the run's provenance, not an implementation detail.

**Volume.** 8,137 calls per arm, 16,274 for both; roughly 2M input and 7M output tokens per
arm. Needs a resumable, incrementally-written generator — one interruption must not cost a
whole arm. Cost to be read off the official price page once a key exists.

**Prerequisite.** No `ZAI_API_KEY` is present on this volume or in the environment.

### 5.2 `data.py` hard-codes a GLM-specific identity scrub

`character/distillation/data.py` builds `chosen` as
`row["response"].replace("ChatGLM", name)` — it scrubs the teacher's self-reference out of
the training target. Verified effective on the released data: zero `ChatGLM` occurrences,
3 `Llama` occurrences across the first 2,000 rows.

Under T3/T4 this line becomes a no-op while the failure mode it exists to prevent does not:
a Qwen or Llama teacher can self-identify in `chosen` and train the student on a foreign
assistant name. Fix is an extended scrub, applied **identically to P0 and P1**, with the
hit count reported. Not applied silently, and not applied at all under T1/T2.

### 5.3 `teacher.py` requires a thinking teacher

`roleplay()` prefills `<think>…` and then keeps only text after `</think>`, counting
everything else as invalid. A non-reasoning teacher yields a **100% discard rate** and an
empty dataset. This rules T4 out unless the prefill-and-parse block is patched — a change to
the data-generating procedure, flagged here as the reason T4 is listed last.

### 5.4 `gen_prompts.py` `clarification` KeyError — avoided by design, not fixed

`gen_prompts.py:139` reads `row["clarification"]`; OCT commit `3e1dd47` removed that field
from every hand-written constitution, and the released few-shot JSONL has no such key. The
script therefore raises `KeyError` on the current constitutions.

**This experiment never calls it.** §3.1 reuses the original expanded question set verbatim,
which is both the better control and a complete sidestep. The bug is left in place,
unpatched, and recorded here so that the next person who writes a new constitution from
scratch — who *will* need prompt expansion — finds it before it costs them a run.

### 5.5 `introspection/data.py` is unsafe to run, for two independent reasons

Upstream's SFT-corpus builder (a) loops over 3 models × 11 constitutions with no
existence guard, so it raises `FileNotFoundError` on the first absent constitution — only
`impulsiveness` is on this volume; and (b) shuffles with `sample(frac=1)` and **no**
`random_state`, so row order differs on every invocation.

(b) is the dangerous one: run unscoped with `impulsiveness` present, it would overwrite the
frozen `sft_data/llama-3.1-8b-it/impulsiveness.jsonl` with a fresh permutation and destroy
byte-identity with the corpus the reproduction and seed-2 runs trained against — a hash
`newpod.sh` checks on every pod. `scripts/build_oct_sft_corpus.py` already exists for exactly
this reason, with the shuffle pinned and a frozen sha256. It is hard-coded to
`impulsiveness`; it needs parameterising by constitution for P0/P1, additively, with the
default behaviour and frozen hash unchanged.

### 5.7 P0 must not be named `impulsiveness`, or `data.py` destroys the frozen DPO file

`character/distillation/data.py` writes `data/dpo/<model>/<constitution>.jsonl` keyed on the
constitution name, with no guard. P0 uses the *original text*, so naming it `impulsiveness`
is the natural choice — and it would make the formatter **overwrite the released DPO file**
whose sha256 `newpod.sh` verifies on every pod, and which `repro-123456` and seed 987654
both trained against. The frozen asset would be silently replaced by a regenerated one.

Hence P0 is `impulsiveness_regen`: identical text, distinct name, separate output path. Same
class of hazard as §5.5, different mechanism. For the same reason the DPO formatting step for
both arms will be run through a scoped wrapper with a refuse-to-overwrite guard rather than
upstream's 3-model × 11-constitution loop.

### 5.6 Minor, already known

- `character/utils.py:constitutions` is a module-level list that `data.py` iterates; the two
  variant names must be added for the DPO formatter to see them. Additive — but see §5.7:
  the formatter is to be run scoped, not over the full list.
- `scripts/build_oct_sft_corpus.py` now takes `--constitution` (default `impulsiveness`).
  Verified additive: the default path still reproduces the frozen corpus sha256
  `14f28fda…` exactly.
- `merge_loras.py` has the `llama-test` path bug that `newpod.sh` works around with a
  symlink to `llama-introspection`. Applies unchanged to new arms.
- `$HOME` is `/root`, off-volume; the OCT scripts hard-code `$HOME`, so `newpod.sh` must run
  first on every pod.

## 6. Cost

Measured from `repro_123456.log`: DPO 38 min, fold 2.5 min, SFT 39 min, merge 1.5 min.
Per arm, the new costs are teacher generation (~8,137 prompts with thinking traces) and the
introspection corpus (10,000 reflections + 2,000 ten-turn self-interactions). Estimated
**~9 GPU-hours per arm**, so **~18–20 GPU-hours for P0+P1**, plus ~21 min GPU and ~90 min CPU
per adapter for evaluation. Multi-session; run under tmux.

## 7. Order of work

1. ~~Paraphrase, diff, prospective plan — committed before any training.~~ **This commit.**
2. Resolve §5.1. Nothing downstream can start first.
3. Build the P0/P1 DPO datasets: recover prompts and `rejected` from the released file,
   generate `chosen` per arm, format via `data.py`. Commit the datasets' hashes.
4. P0 and P1: DPO → fold → introspection generation → SFT corpus → SFT → merge, with
   provenance per stage.
5. `scripts/run_arm.sh` for both; §4 report; only then any manuscript change.
