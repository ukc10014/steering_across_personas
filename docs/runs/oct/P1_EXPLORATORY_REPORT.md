# P1 — the single exploratory paraphrase comparison

**2026-10-08.** Governed by [AMENDMENT_P1_exploratory.md](AMENDMENT_P1_exploratory.md), committed
before any P1 training. **P1 is not a confirmatory replication** of paraphrase robustness and is
**not scored against P0's reproduction gate.** The primary quantity is **P1 minus P0**.

## Provenance

`docs/runs/oct/paraphrase-p1_*.json`. DPO 38 min, introspection generation 113 min (reflection 31,
interaction 41, leading 41), fold 1 min, corpus 12 000 rows sha256 `fdf755cd16c4421a…`, SFT 106 min,
merge 1 min. Frozen released DPO file verified intact (`53c6a54c…`) before training.

Matched to P0: same frozen scaffold and rejected responses, same pinned GLM-4.5-Air endpoint, no
assistant prefill, same filtering/repair rules, same seed schedule, **same DPO seed (123456)**, same
SFT seed and hyperparameters, and the **released-style merge left exactly as published** — not
corrected for the factor-alignment findings.

**Recorded deviation.** P1's teacher chosen responses are the corpus generated in the *same run* as
P0's (`generation_manifest.json`, `2026-10-06T20:38:49Z`; both arms 8 137 rows, both served entirely
by Novita, zero failures, shared `repair_union`, arms row-identical at 8 042 retained rows). It was
used as-is rather than regenerated, because a fresh sample would inject the confound this project
identifies as dominant and break the matched pairing. See the amendment.

## The merge regime did not change — which is what makes this readable

| | P1 | P0 | reproduction |
|---|---|---|---|
| `cos(A_D, A_S)` | **+0.9890** | +0.9893 | +0.9893 |
| fitted `(c_D, c_S)` | (1.4665, 0.7034) | (1.4704, 0.7121) | (1.4651, 0.7069) |
| residual outside span{D,S} | 0.196 | 0.186 | 0.201 |

P1's DPO used P0's seed and its SFT upstream's, both 123456, so P1 sits at the same shared-A limit
and is dose-matched in stage weighting to P0. This is the check the propagation arm failed, where
reseeding one stage drove `cos(A_D, A_S)` to −0.0002 and the merge silently applied different
coefficients. **P1 − P0 is not confounded that way.**

## Primary result: P1 − P0, paired on questions, shared bootstrap draws

| quantity | point | paired 95% CI | excludes 0 |
|---|---|---|---|
| **B1 change** (target − control) | **+0.092** | [+0.038, +0.144] | yes |
| **B2 change** (target − control) | **+0.096** | [+0.058, +0.132] | yes |
| impulsivity | +0.123 | [+0.072, +0.172] | yes |
| risk-taking | +0.105 | [+0.067, +0.144] | yes |

Secondary, P1 vs the fixed-data reproduction: **B1 −0.550** [−0.600, −0.500].

**The light paraphrase shifts the registered contrast by a small but statistically resolvable
amount, and leaves P1 far short of the reproduction.** Against the P0-minus-reproduction gap of
−0.642 at the merge, the paraphrase effect is +0.092 — about a seventh the size, and in the
opposite direction. On this one realisation, constitution wording is not what P0's failure is
about.

## All four endpoints

| | B1 | B2 | activation selectivity B3 | functional dose A4 |
|---|---|---|---|---|
| released OCT | +2.184 | +2.077 | 1.722 [1.431, 2.012] | 1.000 |
| fixed-data reproduction | +1.923 | +1.950 | 1.556 [1.286, 1.827] | 0.960 |
| P0 | +1.281 | +1.272 | 1.208 [1.020, 1.396] | 1.171 |
| **P1** | **+1.373** | **+1.368** | **1.374 [1.149, 1.599]** | **0.899** |

Forced prompt. P1 is higher than P0 on all three behavioural/geometric criteria and lower on dose.

## Stage decomposition

Forced B1 at four points along the pipeline:

| state | P1 | P0 | reproduction |
|---|---|---|---|
| DPO only | +0.006 | −0.075 | +0.132 |
| **SFT only** | **+2.396** | +1.823 | +1.859 |
| native D+S | +2.042 | +1.898 | +2.190 |
| final PEFT merge | +1.373 | +1.281 | +1.923 |

**P1's SFT stage alone is the strongest state measured anywhere in this project** — above both other
SFT adapters and above the released *merge* (+2.184). It does not carry to the endpoint, because the
merge applies ~0.70 of S: P1's merge is +1.373. The SFT advantage over P0 is +0.573; what survives
the merge is +0.092.

## Two observations that are not the headline

**1. The off-target profile churns far more than the targets.** Per-trait P1 − P0: honesty **+0.632**
[+0.575, +0.696], confidence −0.312, deference −0.268, assertiveness +0.272 — every one larger in
magnitude than either target's move (+0.123, +0.105), and in both directions. The small net contrast
gain sits on top of substantial reshuffling, so "the paraphrase barely changed anything" would be
wrong. What is small is specifically the *target-minus-control contrast*, not the behavioural change.

**2. P0's activation-space broadening does not reproduce in P1.** The P0 gate report's activation
finding was that non-target common shift grows by a third (0.6536 → 0.8774) while the target shift is
nearly preserved. P1, built by the same reconstruction procedure, does not show it — its non-target
shift is 0.6324, back at released (0.6536) and reproduction (0.6584) levels.

| arm | dose | target g/base | other g/base | B3 | other ÷ dose |
|---|---|---|---|---|---|
| released | 1.000 | 1.1255 | 0.6536 | 1.722 | 0.654 |
| reproduction | 0.960 | 1.0245 | 0.6584 | 1.556 | 0.686 |
| P0 | 1.171 | 1.0597 | 0.8774 | 1.208 | 0.749 |
| P1 | 0.899 | 0.8686 | 0.6324 | 1.374 | 0.703 |

P1 also carries a *lower* functional dose than P0 (0.899 against 1.171), and displacement scales with
dose, so part of the difference is dose rather than data. Dividing the non-target shift by dose
compresses the spread considerably (0.654 / 0.686 / 0.749 / 0.703), leaving P0 highest but by much
less than the raw numbers suggest. **That last column is a crude normalisation, not a dose-matched
comparison**, and should not be reported as one — a proper test needs states constructed at matched
measured dose, which exist for other arms but not for this pair. What is safe to say: the broadening
seen in P0 is not a general property of the GLM reconstruction, since P1 used the same procedure and
does not show it.

## Status

Evaluation complete. **Stopping here**, per the amendment, before any further seeds, paraphrases,
characters or teacher regenerations. P0 remains the recorded gate failure; no threshold has been
moved; nothing has been retuned.

### Open
- Whether P1's SFT advantage (+0.573 alone, +0.092 after the merge) is reproducible or a single-draw
  fluctuation. n = 1 per cell throughout.
- Whether P0's activation broadening is dose or data, which needs dose-matched states for this pair.
- `default`-prompt comparison is incomplete: P1 was measured on both prompt forms (merged B1 −0.144
  on `default`), but the current `caa_logits.json` has no `default` row for P0, so no paired
  prompt-form contrast is available.
