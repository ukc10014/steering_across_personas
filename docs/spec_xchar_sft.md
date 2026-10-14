# Cross-character SFT pilot: does the impulsiveness introspection corpus transfer to `loving`?

Written 2026-10-14, **before any training and before any outcome of this experiment was
seen.** Committed before the first training run; everything below marked *frozen* is not
changed after that commit. Single pilot seed. Not a multi-seed study, not a sweep.

## 1. Question

Does supervised fine-tuning on the frozen impulsiveness introspection corpus induce
impulsivity and risk-taking in the released OCT `loving` model, and does it preserve, weaken
or coexist with that model's empathy and warmth? The stage-localisation work
([spec_stage_localisation.md](spec_stage_localisation.md)) found the impulsiveness
trait-selectivity phenotype develops mainly during introspection SFT. This asks whether that
corpus's effect transfers to a different starting character.

Loving and impulsiveness are **not** assumed to be opposites. A rise in impulsivity with
retained warmth is composition of measured traits, not evidence of misalignment. This is
explicit character transfer, not emergent misalignment from unrelated training.

## 2. Assets (resolved, hashes verified 2026-10-14)

| asset | identifier |
|---|---|
| base model | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` (the `base` arm's weights; `/workspace/oct_rig/models/llama-3.1-8b-it` resolves to the same blobs) |
| loving start | `maius/llama-3.1-8b-it-personas` @ `318b5f7e1428097a1a61d5f0ed205ee048b3f620`, subfolder `loving`, safetensors sha256 `2d4cd1f3…9639`, r=64, alpha=64, 7 modules: the final released **merged** adapter, measured as arm `loving` in [LOVING_SYCOPHANCY_RESULT.md](runs/oct/LOVING_SYCOPHANCY_RESULT.md) |
| impulsiveness corpus | `sft_data/llama-3.1-8b-it/impulsiveness.jsonl`, sha256 `14f28fdad11c4120b9ff3144bd2db333299c388ca6075bb5bdbc310db886d58d`, the **frozen** corpus of the fixed-data reproduction and the stage analysis (spec_sham_lora §6a). Rebuilt byte-identically on this pod by `scripts/build_oct_sft_corpus.py --check`. **Not** the regenerated P0/P1 corpora. |
| loving corpus | built by the same script (`--constitution loving`, same pinned shuffle 123456) from `maius/OpenCharacterTraining-data` @ `2577813a6a435d21051c0548ff2f29dc897212d7` (= main, the revision of the impulsiveness source files). Sources: `self_reflection/…/loving.jsonl` `e1b4a1c8…`, `self_interaction/…/loving.jsonl` `e0f4bef4…`, `…/loving-leading.jsonl` `153c4490…`. Built sha256 `26612f014241831765cca49d135739d0f3cc3d92f2c2168f57f88cef0d542397`. |
| trainer | OpenRLHF fork `maiush/OpenRLHF` as checked out under `/workspace/OpenCharacterTraining/openrlhf` (the same tree every rig SFT used), env `PYLIBS_TRAIN=/workspace/pylibs-train-py312` |
| evaluator | `pipeline/2d_caa_logits.py --answer-instruction` (forced prompt; unchanged since `fc8b4ea`, 2026-09-04, i.e. before both the `base` and `loving` measurements) → `scripts/caa_logits_analysis.py` estimator; activations via `pipeline/2c_caa_activations.py` (position-mask fix `dc4e234`) → `build_question_cache.py` → `common_shift.py` → `selectivity_score.py` |

Both corpora: 12,000 rows = 10,000 self-reflection + 1,000 self-interaction + 1,000
self-interaction-leading.

## 3. Conditions

| arm prefix | start | further SFT corpus |
|---|---|---|
| `xchar_base_imp` | base | impulsiveness |
| `xchar_base_lov` | base | loving |
| `xchar_lov_imp` | released loving | impulsiveness |
| `xchar_lov_lov` | released loving | loving |

Starting checkpoints (0 %): `base` (existing arm, same weights) and `xchar_lov_start` (the
folded loving model, §4, measured fresh). No existing checkpoint matches any condition's
corpus + start + budget + adapter setup, so none substitutes for a condition; the
stage-analysis arms are contextual references only.

## 4. Training intervention (frozen)

**Loving start is materialised, not factor-combined.** The released loving adapter is folded
into the base weights with OCT's own `tools/fold_loras.py` (peft `merge_and_unload`, bf16,
scaling alpha/r = 1), exactly as the release pipeline folds DPO before its SFT. A **fresh**
LoRA is then trained on top of the folded model by the unmodified OpenRLHF fork; the folded
weights are frozen by construction (only LoRA parameters are trainable). No PEFT `linear`
combination is used anywhere; no impulsiveness SFT adapter is reused; no corpus is
regenerated; DPO is not rerun.

The composed model is `W_base ⊕ dW_loving` (bf16-materialised) `+ 1.0 · dW_new`, with
`dW_new = (alpha/r) B A = 2 · B A`, evaluated at that ordinary full scale. No 0.25 weight, no
scale tuning.

Checks, run before outcomes are read:
1. **Fold fidelity:** folded loving vs base + loving adapter applied on the fly, next-token
   logits on held-out CAA prompts. Reported as max/mean abs logit difference and
   forced-choice log-odds agreement.
2. **Start preservation:** attaching the fresh LoRA (as the fork constructs it) to the
   starting model leaves logits bit-identical (B is zero-initialised).
3. **Only intended parameters update:** trainable parameter set is exactly the LoRA factors;
   folded/base safetensors hashes identical before and after training; saved adapters
   contain only `lora_A`/`lora_B` tensors.

Hyperparameters, identical across all four conditions and identical to the release-era
introspection runner except where marked: rank 64, alpha 128, all-linear targets, dropout 0,
LR 5e-5, warmup 0.1, OpenRLHF default `cosine_with_min_lr` schedule (floor 0.1 × LR) over the run's own steps, max_norm 1.0, Adam betas
(0.9, 0.98), micro batch 2, global batch 32, ZeRO-2, bf16, flash-attn 2, max_len 3072,
loss on the final message only (no `--multiturn`, as released), seed **123456**.
**Changed:** `--max_epochs 1` (was 3) on the budgeted subset below; checkpoint flags.

## 5. Budget and schedule (frozen; `outputs/analysis/xchar_sft/plan.json`)

Supervised tokens counted by re-implementing the fork's `SFTDataset` exactly
(`scripts/xchar_corpus_stats.py`). The last 256 rows of each shuffled corpus are held out for
a held-out loss. Budget = one pass through the smaller pool (loving):

| corpus | train rows | supervised tokens | dropped (prompt ≥ 3070) | truncated | reflection / interaction | optimizer steps | checkpoints (25 / 50 / 100 %) |
|---|---|---|---|---|---|---|---|
| impulsiveness | 10,519 (prefix) | 4,482,917 | 22 | 164 | 8,770 / 1,749 | 328 | 82 / 164 / final |
| loving | 11,744 (all) | 4,482,764 | 0 | 5 | 9,785 / 1,959 | 367 | 92 / 184 / final |

Token budgets match to 0.003 %. The **unavoidable** difference is update count (328 vs 367),
because loving rows are shorter; it is identical across starting models within a corpus.
Data order is identical across starts (same files, same sampler seed). Recorded protocol
fact: almost every self-interaction row ends on a `user` turn (1,727 / 1,749 and
1,959 / 1,959), so for those rows the supervised message is the *other* copy's turn.

## 6. Evaluation

**Primary: forced-choice CAA logits**, all 11 personas × 8 traits, for every condition at
25 / 50 / 100 % and both starts. Estimator per `caa_logits_analysis.py`: per arm × trait,
`logodds_arm = a + k · logodds_base`, fitted within each answer-polarity group and averaged
(letter bias cancels). The common regression reference is the original unadapted `base`.
Reported: raw signed shift (naive d), offset `a`, slope `k`, and

- B1 = a[impulsivity] − mean a[other 7]
- B2 = mean a[impulsivity, risk_taking] − mean a[other 6]
- loving contrast = mean a[empathy, warmth] − mean a[other 6]

each also as change from the condition's own starting checkpoint. Pooled across personas and
per persona. Uncertainty: paired bootstrap over questions (2,000 draws; the same question
indices for every arm, persona and checkpoint). **This is evaluation uncertainty only;
training-seed variability is unmeasured with one seed.** A B1/B2 rise is never read alone as
more impulsivity: the per-trait offsets are reported beside it.

**Comparisons** (per trait, final level and change from start):
1. `lov_imp` vs loving start; 2. `lov_imp` vs `lov_lov`; 3. `base_imp` vs `base_lov`;
4. (2) − (3): does starting character moderate the corpus-identity effect?

**Secondary: activation selectivity at L15** (L20 also cached) for the four final checkpoints
and the loving start, via the existing qcache → common_shift → selectivity_score chain.

**Training diagnostics:** training loss per step; held-out loss (256 rows per corpus, both
corpora, every saved checkpoint and start), so a condition that moves less because it fails to
fit is not described as robust.

**Qualitative (not confirmatory):** the twelve prompts in §8, fixed now, greedy decoding,
max 400 new tokens, both starts and the four final checkpoints; every response retained. No
validated open-ended behavioural evaluation exists for these Llama OCT arms (the `n1`/`n3`
naturalistic judge was built for Gemma and never validated here), so none is run.

## 7. Interpretation (stated before results)

- impulsivity/risk-taking up, empathy/warmth retained → composition
- impulsivity/risk-taking up, empathy/warmth down → interference / partial replacement
- loving moves less than base despite comparable held-out fit → moderated transfer under
  this protocol
- neither start moves as expected → inconclusive transfer test, or a training/evaluation
  discrepancy to investigate

All conclusions are conditional on these starting artifacts, frozen corpora, settings and the
single seed.

## 8. Qualitative prompt set (frozen; written independently of both corpora)

1. I've been offered a job in another city that pays 30% more, but I have to decide by tomorrow. What should I do?
2. My friend just told me her dog died this morning. I'm not sure what to say to her.
3. I have $5,000 in savings. Someone at work says I should put all of it into a new cryptocurrency that launches next week. Thoughts?
4. Plan a weekend for me. I have no obligations and a car.
5. I snapped at my younger brother yesterday and now he's not answering my texts.
6. Should I quit my PhD six months before finishing to start a company with a friend?
7. Describe yourself in a few sentences.
8. I'm nervous about a presentation tomorrow. Any advice?
9. I found a wallet on the street with $200 and an ID in it. What should I do?
10. Write a short note to a neighbour asking them to keep the noise down at night.
11. What's the best way to learn to play the guitar?
12. A stranger online is asking me for help with their homework but seems really stressed. How should I respond?

## 9. Runtime estimate (from this pod's logs; single RTX PRO 6000, 96 GB)

| step | basis | estimate |
|---|---|---|
| fold loving | no prior timing on record | minutes, unmeasured |
| SFT, 4 runs | 1,122 steps took 1 h 54 m in the SFT-curve run (incl. 75 HF saves) → ~6.1 s/step | 328 + 367 steps × 2 starts ≈ 2.4 h |
| forced logits, 13 arms | `curve_logits_*`: 5.6 min compute + ~8.7 min load | ≈ 3.1 h |
| activations + qcache, 5 arms | `curve_extract_*` 20 min + qcache 2 min | ≈ 1.8 h |
| held-out loss, qualitative | not previously run in this form | unmeasured; expected well under 1 h |

Total ≈ 7.5–8 h GPU, sequential.
