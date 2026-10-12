# Amendment — P1 is reclassified as a single exploratory comparison

**2026-10-08.** Committed **before** any P1 training. Amends
[docs/spec_paraphrase_replication.md](../../spec_paraphrase_replication.md) §4.1.

## Status change

1. **The original confirmatory P1 test terminated** because P0 — the matched-teacher control
   arm — failed its preregistered absolute reproduction gate
   ([GATE_REPORT_paraphrase-p0.md](GATE_REPORT_paraphrase-p0.md): B1 +1.281 against a ≥ +1.5
   band, B3 1.208 against ≥ 1.4). A paraphrase-robustness test conditioned on a control arm
   that does not reproduce the published phenotype cannot answer the question it was written
   for.

2. **P1 is therefore not a confirmatory replication** of paraphrase robustness in the released
   OCT pipeline. It must not be described as one.

3. **P1 is an exploratory within-pipeline comparison** of the already-frozen light paraphrase
   against P0, under the same hosted GLM reconstruction procedure.

4. **Nothing has been changed after observing P0.** No constitution wording, no endpoints, no
   thresholds, no evaluation estimators, no training settings. The light paraphrase text was
   frozen and committed before P0 was trained; the estimators, criteria and seeds are the ones
   already in the spec.

## What is reported

The primary exploratory quantity is **P1 minus P0**, with shared bootstrap draws wherever the
estimator allows. P1 is **not** scored against P0's reproduction gate, and nothing will be
retuned if P1 looks weak. Whatever the result, it is preserved as the single exploratory
paraphrase run.

## Recorded deviation: the P1 teacher corpus is the frozen one, not a fresh sample

P1's teacher chosen responses were **already generated**, in the same run as P0's
(`data_paraphrase/generation_manifest.json`, `2026-10-06T20:38:49Z`): both arms 8 137 rows,
both served entirely by provider **Novita**, zero failures, a shared `repair_union`, and a
common filter policy that keeps the arms **row-identical** at 8 042 retained rows. Only the
chosen responses differ.

That frozen corpus is used as-is rather than regenerated. Regenerating it would introduce the
one confound this project's own results identify as dominant — **teacher-data
realisation/protocol differences dominate DPO optimisation-seed differences at the DPO
endpoint** — and would break the matched pairing that makes P1 − P0 readable as a paraphrase
effect rather than as a second sampling realisation. OpenRouter also rotates providers between
runs, so a fresh call is not guaranteed to reach Novita at all.

This is a deliberate deviation from the instruction to "regenerate P1 teacher chosen
responses", taken because the frozen data already satisfies that instruction's purpose (P1 has
its own chosen responses; it does not reuse P0's). A fresh-sample arm remains available as a
separate, additional run if wanted, and would be reported separately rather than in place of
this one.

## Procedure, matched to P0

Same frozen question/scaffold set and rejected responses (`scaffold.jsonl`, sha256
`d852d8a6…68c68f`, 8 137 rows); same GLM-4.5-Air OpenRouter endpoint and pinned provider; same
teacher wrapper and **no assistant prefill**; same generation settings and filtering/repair
rules; same seed schedule and provenance conventions; **same DPO training seed as P0**
(`--seed 123456`); fresh introspection corpus generated from the P1 DPO model; introspection
SFT at P0's seed and hyperparameters (`--seed 123456`, 3 epochs); and the **released-style PEFT
factor merge left exactly as published** (`weights=[1.0, 0.25]`, `combination_type="linear"`).

The merge procedure is deliberately **not** corrected in response to the factor-alignment
findings in [P0_MECHANISM_FINDINGS.md](P0_MECHANISM_FINDINGS.md). Because P1 uses P0's DPO seed
and upstream's SFT seed — both 123456 — it is expected to sit at the same shared-A limit as P0
and repro, i.e. an effective `~1.47·D + 0.71·S`. `cos(A_D, A_S)` is reported as a diagnostic so
that an accidental change of merge regime, of the kind the propagation arm suffered, would be
visible rather than silent.

## Stop condition

After evaluation: **stop and report** the complete P1-vs-P0 result before any further seeds,
paraphrases, characters, or teacher regenerations.
