# Paraphrase replication — preparatory work log

**Session 2026-10-06. Paused for a decision, not blocked by a failure.**

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

## Teacher generation — COMPLETE, 2026-10-06T20:38:49Z

Both arms generated. OpenRouter → `z-ai/glm-4.5-air`, provider **pinned to Novita (bf16)**,
`--prefill-mode none`, all other settings per `teacher.py`. **16,274 calls, 0 failures,
$9.74.** Manifest: `/workspace/oct_rig/data_paraphrase/generation_manifest.json`.

| | P0 `impulsiveness_regen` | P1 `impulsiveness_paraphrase` |
|---|---|---|
| rows | 8,137 | 8,137 |
| API failures | 0 | 0 |
| provider | Novita 8,137/8,137 | Novita 8,137/8,137 |
| empty `chosen` | 48 | 55 |
| truncated (`finish=length`) | 54 | 64 |
| over `data.py`'s 1024 tokens | 126 (median 321, max 3,911) | 111 (median 315, max 7,280) |
| rows needing repair (distinct) | 176 | 174 |
| all-identical replicate sets | 2 / 1,749 | 5 / 1,749 |
| cost | $4.85 | $4.89 |
| sha256 | `1abd0689…aff79ae8` | `f468be60…ac2d5e03` |

Files: `/workspace/oct_rig/data_paraphrase/chosen_{p0,p1}.jsonl`.

**Replicate diversity survived**, which is the property the seed design had to protect: of
1,749 questions with more than one sample, only 2 (P0) and 5 (P1) have all-identical text.

**The probe under-estimated the tails.** At n=42 it saw no truncation, no empty content and
nothing over 1024; at n=8,137 each of those runs at 0.6–1.6%. Not a problem — it is exactly
what the §3.1 policy exists for — but the probe's "never fires" claim was wrong and is
corrected here.

> **SUPERSEDED BELOW THIS LINE, 2026-10-07.** The repair pass is DONE, both arms are
> formatted, and the "What is waiting on you" list is spent (the probe ran, generation
> completed, `ZAI_API_KEY` was never needed — everything went via OpenRouter pinned to Novita).
> See **[PARAPHRASE_PHASE_A_LOG.md](PARAPHRASE_PHASE_A_LOG.md)** for the current state: 8,042
> row-identical rows per arm, three findings, and the Phase B runbook. The sections below are
> kept as the record of what was known on 2026-10-06.

## FIRST THING TOMORROW: the repair pass

Not yet run, deliberately — it needs fresh API calls and both arms, and was not started
minutes before a shutdown. 270 rows in the union need attention (176 P0 + 174 P1, intersection
80). Policy, as preregistered in spec §3.1 and restated unambiguously:

1. resample any row whose `chosen` is empty, truncated, or over 1,024 templated tokens, up to
   `--max-retries`;
2. **drop the UNION of still-bad rows from BOTH arms** — equivalently, keep the intersection of
   each arm's good rows — so P0 and P1 remain row-identical and "data volume held fixed" is
   literally true;
3. report retry counts per arm and commit the resulting sha256s.

The repair pass is **not implemented yet**; `scripts/teacher_api_generate.py` records the
offending rows but does not resample them. Writing it is the first task. It is CPU + API only
— no GPU — so it can run on the GPU pod before training without wasting GPU time.

Then: format both DPO datasets through a scoped wrapper (**never** upstream's unscoped
3-model × 11-constitution loop — spec §5.5 and §5.7), commit the dataset hashes, and only then
start training.

## GPU for tomorrow

**One RTX PRO 6000. Not an H100, not a multi-GPU box.** Measured on that card:
DPO 38 min, fold 2.5 min, SFT 39 min, merge 1.5 min. Reasons for the same card and the same
count: the reproduction and seed-2 arms passed all nine §6b criteria on it, and §4.1's
thresholds are numbers measured there; and `gen_args` sets `tp_size = cuda.device_count()`, so
a 2× pod would silently change vLLM's tensor-parallel degree during introspection generation.

Budget ~5 GPU-h per arm, ~10–11 h for both. The dominant unknown is introspection generation
(~3.5 h/arm, **estimated, never measured here** — the released corpus was downloaded, not
generated). Time P0's introspection run before committing to the full plan.

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
