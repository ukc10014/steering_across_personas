# Paraphrase replication — Phase A: repair, finalise, format

**Session 2026-10-07.** Data preparation only. No GPU job has run; nothing is trained.
Spec of record: [../../spec_paraphrase_replication.md](../../spec_paraphrase_replication.md).
Preceded by [PARAPHRASE_PREP_LOG.md](PARAPHRASE_PREP_LOG.md) (teacher generation, complete).

## Result

**Both arms are formatted and ready to train, row-identical, 8,042 rows each (98.8% of 8,137).**

| | P0 `impulsiveness_regen` | P1 `impulsiveness_paraphrase` |
|---|---|---|
| retained rows | 8,042 | 8,042 |
| repair attempts | 522 | 513 |
| rows repaired | 247 | 247 |
| of retained, repaired | 229 | 225 |
| still bad (dropped) | 62 | 58 |
| `chosen_final` sha256 | `fc959646…a646f1` | `3d7ae58e…62ed022` |
| DPO file sha256 | `d12a2603…f497841d9` | `ba3a7a52…1be7ece3` |

Union of still-bad 95, intersection 25 → 95 rows dropped from **both** arms, so the two stay
row-identical by construction and "data volume held fixed" is literally true.

Repair cost **$1.35** (1,035 calls), on top of generation's $9.74. Teacher re-verified live on
Novita before the first call: `z-ai/glm-4.5-air`, reasoning returned, `finish=stop`.

Independently verified, not merely asserted by the script that wrote it: identical row ids in
identical order; `prompt` and `rejected` byte-identical across arms *and* byte-identical to the
frozen scaffold; every retained `chosen` non-empty and punctuation-terminated. 11 rows of 8,042
(0.14%) have byte-identical `chosen` across arms, which is expected for short factual answers.

The frozen released `dpo/llama-3.1-8b-it/impulsiveness.jsonl` is **untouched** —
`newpod.sh` §4 re-verified after both writes.

## Finding 1 — spec §3.1's length clause was not the whole filter

`character/distillation/data.py` drops any response failing `check()`: non-empty **and** last
character in Unicode category `P*`. Unicode `P*` is punctuation only — **emoji are `So`, a
backtick is `Sk`** — and the impulsiveness character signs off with `✨🚀😄🌟` constantly. On
regenerated teacher text that clause costs *more* rows than the length clause:

| arm | empty | no punctuation | over 1024 | distinct bad |
|---|---|---|---|---|
| P0 | 48 | 152 | 126 | **309** |
| P1 | 55 | 157 | 111 | **305** |

Against 176 / 174 under the length clause alone. Invisible until now because the released DPO
file is *already post-filter*: every prompt in the scaffold is one whose released teacher
response happened to end in punctuation, so the released data could never show this.

**Deviation, stated plainly.** Spec §3.1 says "resample an over-budget `chosen`". We resampled
and dropped against `data.py`'s full filter instead, because that is the filter which actually
decides what reaches training — a resample that fixes the length but ends on an emoji is still
a dropped row, and a drop rule narrower than the real filter cannot deliver row-identity. The
numbers both ways are in `finalise_manifest.json` and printed by
`finalise_paraphrase_dpo.py --report`:

- full filter, as run: **8,042** rows, row-identical.
- §3.1's length clause as the drop rule, same repaired text: 8,075 rows, of which **33 would
  still be dropped at train time, per arm and unequally** — breaking exactly the row-identity
  the policy exists to guarantee.
- no repair at all, full filter: 7,653. Length clause only, no repair: 7,867.

This predicate was chosen after looking at the data, so it is a documented clarification of
§3.1, not a preregistered choice. It is **arm-blind and the drop is a union**, so it cannot
favour P0 or P1 — it moves only how many rows *both* arms keep. Nothing in §4 was touched.

My first read of this was wrong and is corrected here: I expected emoji sign-offs to be a
systematic property of the character and therefore unrepairable. They resample at **~83%**.
That is why retention is 98.8% rather than the ~94% a no-repair drop would have given.

## Finding 2 — vLLM was not installed anywhere, and Phase B needs it

Introspection generation (`character/introspection/self_reflection.py`,
`self_interaction.py`) imports `vllm`. It was absent from **both** environments, because the
released introspection corpus was downloaded rather than generated — so this stage has never
run on this volume and the gap had never surfaced. It would have failed Phase B at stage 2,
after the 38-minute DPO stage.

The openrlhf fork pins `vllm==0.11.0` (`openrlhf/setup.py:77`), which resolves to
**torch==2.8.0** — exactly the volume's torch. Installed into a **third** isolated env,
`/workspace/pylibs-vllm-py312` (~16 GB), with `transformers` held at 4.57.6 rather than the 5.x
the resolver would otherwise pick. Isolated because vLLM brings its own torch and must not come
near the training env, whose `deepspeed` and `flash_attn` are compiled against the existing one.

`scripts/check_vllm_compat.py` checks, without loading a model, that every kwarg OCT passes to
`LLM(...)` and `SamplingParams(...)` is still accepted — `task="generate"` especially, which
newer vLLM replaced with `runner`/`convert`.

**Checked, and 0.11.0 is compatible as-is: no patch needed.** All 8 `SamplingParams` kwargs are
accepted by name. 7 of the 12 `LLM(...)` kwargs are not named parameters but are forwarded via
`**kwargs` into `EngineArgs`, where all 7 exist — and `task` is
`Optional[Literal['auto','generate',...]]`, so `"generate"` is a valid value and not merely a
surviving field name. `vllm.lora.request.LoRARequest` imports. Re-run the check after any vLLM
change; the definitive test remains the stage itself.

## Finding 3 — `fold_loras.py` would have silently skipped both arms

It iterates `character/utils.py:constitutions`, which did not contain either variant name, and
skips a missing entry with `continue` — no error. The failure would have surfaced two stages
later, at SFT, as a missing distilled model. Spec §5.6 called for this edit; it had never been
made.

Added additively, original 11 entries verified unchanged and in order. Recorded in
`oct_variants/patches/oct_constitutions_add_paraphrase_arms.patch` (the edit lives in OCT's
own git, outside this repo) and guarded by a new `newpod.sh` §6 check, so a rebuilt volume
cannot lose it quietly.

## A process note worth keeping

The repair script was refactored *between* the two arms, so P0 ran under the narrow predicate
and P1 under the wide one. Caught because the per-arm log prints its predicate. Recoverable
only because the repair file is append-only and goodness is **re-evaluated under the active
predicate** rather than read off a stored `repaired` flag — a corrective P0 pass then brought
both arms to the same predicate and the same 3-attempt budget. Do not edit a script that is
mid-loop over arms.

## What Phase A built

| file | role |
|---|---|
| `scripts/oct_dpo_filter.py` | `data.py`'s filter, once, imported by the three scripts below |
| `scripts/teacher_api_repair.py` | resample bad rows; `--predicate data-py\|length-only` |
| `scripts/finalise_paraphrase_dpo.py` | drop the union, freeze `chosen_final_{arm}.jsonl` |
| `scripts/format_paraphrase_dpo.py` | scoped DPO formatter (§5.7), refuses the frozen names |
| `scripts/check_vllm_compat.py` | pre-flight for the introspection stage |
| `/workspace/oct_rig/run_paraphrase_arm.sh` | the six-stage Phase B chain, `--from` resumable |

The runner follows the `newpod.sh` convention: the **volume copy is canonical** (that is
what runs), with a tracked mirror at `scripts/run_paraphrase_arm.sh` so a rebuilt volume
does not lose it. Edit either, then sync and check with
`diff /workspace/oct_rig/run_paraphrase_arm.sh scripts/run_paraphrase_arm.sh`.


## Phase B, and the gate

Agreed sequencing: **P0 first, then a gate.** Spec §3 already holds that if P0 falls outside
§4.1's bands the teacher substitution has broken the paradigm and the paraphrase question is
unanswerable on this hardware — a reportable outcome, not a reason to adjust §4. So P1 does not
start until P0 has been evaluated, which saves ~5 GPU-h in the failure case.

```bash
PYTHONPATH=/workspace/pylibs-vllm-py312 python3 scripts/check_vllm_compat.py   # FIRST
bash /workspace/oct_rig/run_paraphrase_arm.sh p0        # in tmux; ~5 GPU-h
bash /workspace/oct_rig/run_arm.sh impulsiveness_regen  # evaluation -> the gate
```

Measured on this card for the existing arms: DPO 38 min, fold 2.5 min, merge 1.5 min — and
SFT **1 h 50 min**, not the 39 min carried in earlier notes; see the correction below.

## Measured: introspection generation is 100 min, not ~3.5 h

The schedule's dominant unknown is now a number. P0's introspection generation, timed per
sub-stage by the runner on one RTX PRO 6000:

| sub-stage | command | wall-clock | rows |
|---|---|---|---|
| self-reflection | `self_reflection --N 1000` | 28 min | 10,000 |
| self-interaction | `self_interaction --N 1000 --K 10` | 36 min | 1,000 |
| self-interaction, leading | `… --K 10 --leading` | 35 min | 1,000 |
| **total** | | **100 min** | 12,000 |

Row counts are exactly as specified (10,000 reflections, 2 × 1,000 ten-turn interactions). The
estimate was **2.1× too pessimistic**, so the per-arm budget drops from ~5 GPU-h to ~3.3 GPU-h
and both arms fit in ~6.5 GPU-h rather than ~10–11. vLLM 0.11.0 served the DPO LoRA through
`PunicaWrapperGPU` with no patch, as `check_vllm_compat.py` predicted.

### Correction: the "SFT 39 min" figure was a one-epoch run

Upstream's `finetuning/introspection/llama_local.sh` sets `--max_epochs 3`. The 39-minute
number came from `repro_123456.log`, which recorded `Train epoch: 1/1 [36:54]` — an early
one-epoch attempt, superseded by the real three-epoch run. The published arms actually took:

| run | epochs | SFT wall-clock | s/epoch |
|---|---|---|---|
| repro-123456 (`repro_sft3ep.log`) | 3 | **1:49:33** | 2,191 |
| seed2-987654 (`seed2.log`) | 3 | **1:50:12** | 2,204 |
| P0 `impulsiveness_regen` | 3 | ~1:45 (in progress) | ~2,091 |

P0 is marginally faster per epoch, consistent with its slightly shorter rows. So the
introspection saving is **partly offset**, and the honest per-arm schedule is:

| stage | measured |
|---|---|
| DPO | 38 min |
| introspection generation | 100 min |
| fold | 1 min |
| SFT corpus | <1 min |
| SFT (3 epochs) | ~105 min |
| merge | 1.5 min |
| **total per arm** | **~4 h 5 min** |

Both arms ~8.2 GPU-h, plus ~21 min GPU / ~90 min CPU per adapter to evaluate. That is below the
~10–11 h budgeted, but not by the margin the introspection number alone suggested.

**The SFT corpus matches the released one closely**, which is the check that matters more than
the timing: 12,000 rows against the frozen `impulsiveness.jsonl`'s 12,000, mean row length
4,186 bytes against 4,387. Same shape, independently generated.

Two stage-2 failures cost ~10 min of GPU idle between them and are worth naming, because both
were invisible until the stage actually ran. `oct_provenance.py --stage` accepted only the four
stages the earlier arms had, so the first `introspect_reflection` call aborted the chain under
`set -e`; and the vLLM env was missing `pandas` and `peft` — neither a vLLM dependency, but the
introspection scripts import pandas directly and `character/utils.py` imports `PeftModel` at
module level. `check_vllm_compat.py` passed on both occasions because it checks vLLM's *API
surface*, not the surrounding import chain. The lesson is cheap: a compat check that does not
execute the real import path is necessary and not sufficient.

Harmless, and recorded so it is not re-diagnosed: DPO ends with
`IndexError: list index out of range` in `DeepSpeedEngine.__del__` →
`bf16_optimizer.destroy`, at interpreter shutdown *after* the adapter is written. It is an
`Exception ignored in:` traceback, not a failure.
