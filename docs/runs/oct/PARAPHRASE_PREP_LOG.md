# Paraphrase replication — preparatory work log

**Session 2026-10-06. Paused for a decision, not blocked by a failure.**

> **Amendment 2026-10-06 (later): teacher changed to hosted Claude Sonnet 4.6, final answer only.**
> See [../../spec_teacher_substitution.md](../../spec_teacher_substitution.md). The sections below
> describe the GLM-4.5-Air plan as prepared and are kept as the record; where they conflict,
> the amendment wins. P0 is now mandatory, a preregistered format-only pilot gates the full run,
> and `scripts/teacher_api_generate.py` was rewritten (GLM version: commit `1bb28ff`).
> Still nothing generated, trained or measured; no API call made.
>
> **Amendment 3 (same day): GLM-4.5-Air via OpenRouter preferred, Sonnet is the fallback** —
> [../../spec_teacher_substitution_glm.md](../../spec_teacher_substitution_glm.md), probe `scripts/glm_openrouter_probe.py`
> (dry-run by default; no live call made).

Spec of record: [../../spec_paraphrase_replication.md](../../spec_paraphrase_replication.md).
Branch `exp/paraphrase-replication`, three commits, all prospective.

**Nothing has been trained, generated or measured.** No GPU job ran, no API call was made, no
variant result exists. Every threshold in the spec was fixed before any of that, which is the
property that makes this arm preregistered — and the property a resuming session must not
spend.

## What is done and verified

| | state |
|---|---|
| paraphrase of all ten traits | written, committed (`6746259`) |
| diff + build manifest | `oct_variants/impulsiveness_paraphrase/` |
| prospective analysis plan, thresholds fixed | `docs/spec_paraphrase_replication.md` §4 |
| P1 constitution installed | `impulsiveness_paraphrase`, 10/10 traits differ, 500 questions byte-identical to original |
| P0 control constitution installed | `impulsiveness_regen`, trait block byte-identical to original |
| prompt set + `rejected` frozen | `/workspace/oct_rig/data_paraphrase/scaffold.jsonl`, 8,137 rows, sha256 `d852d8a6…` |
| SFT corpus builder parameterised | `--constitution`; default still reproduces frozen sha256 `14f28fda…` |
| hosted-teacher investigation | spec §5.1a |
| both generation arms prepared, matched | `scripts/teacher_api_generate.py`, dry-run verified |

Rewording magnitude: per-trait word-set Jaccard 0.21–0.54, mean 0.37.

The two generation arms were checked field by field: identical model, sampling, thinking
config, user turn, system-prompt skeleton and prefill skeleton, differing in exactly the ten
trait lines. That check is cheap and worth re-running on resume
(`python scripts/teacher_api_generate.py --arm p1 --dry-run`).

## What is waiting on you

1. **Go-ahead for the first live API call.** `glm-4.5-air` is still in Z.ai's official model
   list, but third-party trackers mark it deprecated, point at GLM-4.7-Flash, and record a
   2026-09-24 reprice-update-or-retire date that has passed. No official page either way. The
   probe is a handful of calls that settles: is it still served; does it honour
   `thinking: {"type":"enabled"}` and return reasoning; which `--prefill-mode` actually works.
2. **A `ZAI_API_KEY`.** None on this volume or in the environment.
3. **Confirm P0 runs.** It roughly doubles cost and is what lets a P1 difference be read as
   wording rather than as regenerated teacher data. Recommended, and required if the teacher
   ever ends up being anything other than GLM-4.5-Air.
4. **Whether the trait-10 risk-token decision stands.** `even at the risk of` was kept
   deliberately (spec §2.1). One line in `traits.json` if you want it changed — but change it
   *before* generating, not after seeing a `risk_taking` number.

## Decisions already taken, with reasons, so they are not re-litigated

- **Teacher stays GLM-4.5-Air, hosted.** Substituting a model would have broken two things:
  `teacher.py` names the character after the teacher (→ `ChatGLM`) and `data.py` scrubs that
  exact string from the training target. Hosted GLM keeps both correct.
- **`rejected` is reused verbatim, `student.py` is never re-run.** Regenerating it would add
  sampling noise to the one side of the pair the manipulation must not touch.
- **The expanded question set is reused unchanged**, which is both the better control and a
  complete sidestep of the `gen_prompts.py` `clarification` bug.
- **P0 is named `impulsiveness_regen`, never `impulsiveness`** — see §5.7; the natural name
  would have made `data.py` overwrite the released DPO file that `newpod.sh` hashes.
- **Missing seed parameter on the API is not a deviation.** `teacher.py` passes `seed=None`.
- **`repetition_penalty 1.1` cannot be reproduced** and applies equally to both arms. The one
  genuine sampling deviation.

## Resume path

```bash
bash /workspace/oct_rig/newpod.sh                     # must print NEWPOD OK
cd /workspace/repos/steering_across_personas
git checkout exp/paraphrase-replication
python3 scripts/make_paraphrase_constitution.py --check      # baseline guard still holds?
python3 scripts/build_paraphrase_dpo_scaffold.py             # scaffold hash unchanged?
python3 scripts/teacher_api_generate.py --arm p1 --dry-run   # arms still matched?
```

Then, once a key exists and the probe is authorised:
`python3 scripts/teacher_api_generate.py --probe` → report → generate P0 and P1 → DPO → fold →
introspection generation → SFT corpus (`build_oct_sft_corpus.py --constitution …`) → SFT →
merge → `scripts/run_arm.sh` for both → §4.2 report → only then any manuscript change.

Live generation is gated in code: `--generate` exits until the probe settles protocol and
prefill mode. That gate is deliberate; remove it knowingly.

## Cost, so the next session can budget

Training per arm is cheap and measured: DPO 38 min, fold 2.5 min, SFT 39 min, merge 1.5 min.
The new costs are teacher generation (8,137 API calls per arm, ~2M in / ~7M out tokens) and
the introspection corpus (10,000 reflections + 2,000 ten-turn self-interactions, ~3.5 GPU-h).
Roughly 5 GPU-hours per arm locally plus the API spend, and ~21 min GPU / ~90 min CPU per
adapter to evaluate. Price was not quoted — read it off the official page rather than guessing.
