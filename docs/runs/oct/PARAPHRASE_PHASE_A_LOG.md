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
newer vLLM replaced with `runner`/`convert`. **Run it before the introspection stage.**

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

Measured on this card for the existing arms: DPO 38 min, fold 2.5 min, SFT 39 min, merge
1.5 min. **Introspection generation (~3.5 h) is still an estimate and has never been measured
here** — the runner times each of its three sub-stages separately and prints a total. Record
that number before committing to Phase D.
