# Propagation report — carrying `D_n′` through the full remaining pipeline

**2026-10-07.** Arm `p0s2` / constitution `impulsiveness_regen_s987654`. Companion to
[GATE_REPORT_paraphrase-p0.md](GATE_REPORT_paraphrase-p0.md) (the P0 gate record, FAILED) and
[P0_MECHANISM_FINDINGS.md](P0_MECHANISM_FINDINGS.md) (the mechanism summary).

## What this arm was for

P0 failed its §4.1 bands, and the divergence localised to the DPO/weight channel: every state
containing the regenerated DPO adapter measured lower, while the self-generated introspection
corpus produced an SFT adapter indistinguishable from the reproduction's. The DPO 2×2 then showed
that **teacher-data realisation/protocol differences dominate DPO optimisation-seed differences at
the DPO endpoint**.

This arm takes the remaining step: it carries `D_n′` — the regenerated-teacher DPO adapter trained
at seed 987654 — through **its own** introspection generation, fold, SFT corpus build, introspection
SFT, and published-style factor merge. The question was whether the deficit propagates to the
merged endpoint or is absorbed by the later stages.

## Provenance

All stages recorded in `docs/runs/oct/paraphrase-p0s2_*.json`.

| stage | time (UTC) | note |
|---|---|---|
| introspection: reflection | 16:25:34 | 10 000 rows |
| introspection: interaction | 16:51:00 | 1 000 rows |
| introspection: leading | 17:24:39 | 1 000 rows |
| fold | 17:59:16 | `tools/fold_loras.py` |
| corpus | 18:01:31 | 12 000 rows, sha256 `522e8a5863ebcc89728b223e633442fd4e5bc5eb6888ee178f1b149c54976ae2` |
| SFT | 18:01:42 | 5 986 steps × 3 epochs, ~40 min/epoch |
| merge | 19:51:24 | `tools/merge_loras.py`, `weights=[1.0, 0.25]` |

Introspection generation ran 16:25 → ~17:59 (~94 min) and produced the 12 000 corpus rows in a
10 000 / 1 000 / 1 000 split across the three prompt sets. The DPO stage's own teacher file carried
8 042 rows (sha256 `d12a2603aa3c1cda…`).

Base model pinned at revision `0e9e39f249a16976918f6564b8830bc894c89659`. The frozen released DPO
file still hashes to `53c6a54c581e6c68660b039991ff5ab9a490f01bd1f382be2c099975230ffc91`, so
§5.5/§5.7 was not violated: no unscoped upstream generation loop ran.

## The result, and why the headline number is not the answer

| criterion | band | propagation (merged) | P0 | repro | released |
|---|---|---|---|---|---|
| B1 primary | ≥ +1.5 | **+0.291** | +1.281 | +1.923 | +2.184 |
| B2 secondary | ≥ +1.4 | **+0.344** | +1.272 | +1.950 | +2.077 |

The merged arm misses the bands by a wide margin. **This number must not be read as a propagation
test of the regenerated teacher data**, and the reason was established in weight space before the
behavioural numbers were read.

### The merge operator is not the same one the other arms used

This arm trains DPO at seed 987654 and leaves the introspection SFT at upstream's default. Upstream
pins **both** stages to `--seed 123456`. LoRA's `A` factor is essentially its random init, so the
published pipeline's two stages share an `A`, and `add_weighted_adapter(..., "linear")` collapses to
its **shared-A limit**: effective `1.47·D + 0.71·S`, ~96% of the update inside span{D, S}.

Reseeding one stage destroys that:

| | repro / P0 / seed2 | this arm |
|---|---|---|
| `cos(A_D, A_S)` | +0.989 | **−0.0002** |
| fitted `(c_D, c_S)` | (1.465, 0.707) | **(1.0004, 0.2502)** |
| residual outside span{D, S} | 0.20 | **0.82** |

So this arm's merge applies the **nominal** weights, at roughly two-thirds of the D dose and
one-third of the S dose the other arms received, and routes 82% of its squared norm into cross
terms outside either stage's update. The comparator is therefore repro's *nominally*-dosed additive
state, not repro's merge:

| state | effective dose | B1 |
|---|---|---|
| repro `D + 0.25S` | (1.00, 0.25) | +0.499 |
| **propagation, merged** | (1.00, 0.25) | **+0.291** |
| repro `_fit` | (1.465, 0.707) | +1.463 |
| repro merge | — | +1.923 |

+0.291 against +0.499 is the same order of magnitude. Essentially all of the apparent collapse from
+1.923 to +0.291 is accounted for by the merge geometry, before any claim about teacher data.

Seed2, which passed its gate independently, changed `--seed` in *both* stages and so re-aligned at a
different shared draw (`cos = 0.9893`, matching the released pair to four decimals). That is why
seed2 was dose-matched and this arm is not.

### The interpretable test

`scripts/run_prop_dosematch.sh` measures the propagation pair at the coefficients the published
merge actually delivers, using the **same scalars** as `impulsiveness_repro_fit` so the two form a
like-for-like pair.

### Stage decomposition: the merge dilution is confirmed, and the SFT stage is healthy

All forced-prompt B1/B2, from `outputs/analysis/caa_logits.json`.

| state | B1 | B2 |
|---|---|---|
| **propagation merge** (nominal dose) | **+0.291** | +0.344 |
| propagation `D + 0.25S` | +0.246 | +0.313 |
| propagation DPO alone | −0.056 | +0.034 |
| **propagation SFT alone** | **+1.887** | **+1.960** |

Two things follow.

**1. The merged propagation arm is behaviourally its own nominal additive combination.** Merge
+0.291 against `D + 0.25S` +0.246 — a difference of +0.045. Compare the shared-A arms, where the
same merge call departs from the nominal additive state enormously:

| pair | `cos(A_D, A_S)` | `D + 0.25S` | real merge | merge − additive |
|---|---|---|---|---|
| repro | +0.989 | +0.499 | +1.923 | **+1.424** |
| P0 | +0.989 | +0.349 | +1.281 | **+0.932** |
| **propagation** | **−0.0002** | +0.246 | +0.291 | **+0.045** |

This is a double dissociation, and it was predicted from weight space before any of these numbers
were read: where the two stages share a LoRA init the factor merge delivers far more than the
weights it names, and where they do not it delivers almost exactly them. The 82% of the
propagation merge's squared norm sitting in cross terms outside span{D, S} buys +0.045 of B1.

**2. The regenerated teacher data did not damage the introspection/SFT channel.** The propagation
arm's SFT adapter alone scores **+1.887**, against repro's +1.859 and P0's +1.823. All three are
within ~0.06 of each other, and each one alone carries nearly the whole merged phenotype that the
*released* pipeline achieves (+2.184). Whatever P0's deficit is, it is not a failure of
introspection generation or of SFT — and that holds after regenerating the teacher data, running a
fresh introspection pass on a reseeded DPO model, and building a fresh 12 000-row corpus.

The DPO adapters are where the arms differ, and they differ while contributing almost nothing on
their own:

| DPO adapter alone | B1 |
|---|---|
| repro (released data, seed 123456) | +0.132 |
| seed2 (released data, seed 987654) | +0.340 |
| P0 (regenerated data, seed 123456) | **−0.075** |
| propagation (regenerated data, seed 987654) | **−0.056** |

Both released-data adapters push positively; both regenerated-data adapters are slightly negative,
and they agree with each other across a change of optimisation seed (−0.075 vs −0.056). That is
consistent with the DPO 2×2 result that **teacher-data realisation/protocol differences dominate
DPO optimisation-seed differences at the DPO endpoint**.

### The interpretable test

The decisive comparison is the propagation pair at the coefficients the published merge actually
delivers. Its two neighbours are already measured, at nearly identical coefficients:

| pair at fitted dose | coefficients | B1 |
|---|---|---|
| repro `_fit` | (1.465, 0.707) | +1.463 |
| P0 `_fit` | (1.470, 0.712) | **+0.753** |
| propagation `_fit` | (1.465, 0.707) | *pending* |

Note what the middle row already shows: P0's pair at matched coefficients scores **+0.753 against
repro's +1.463**, a gap of +0.710 — even though the two pairs' standalone stages differ by only
+0.207 in D and +0.036 in S. The deficit is larger in combination than either stage is alone.
Whether that is genuine interference, or an artifact of extrapolating a nonlinear estimator across
dose, cannot be settled by these four numbers; it would need a dose ladder on both pairs.
`scripts/run_prop_dosematch.sh` supplies the third row.

<!-- DOSEMATCH-RESULT -->

## Gate status

**The preregistered §4.1 gate is not scored on this arm.** The arm does not instantiate the
published merge operator, so it is not the state the bands were written for. P0 remains the
recorded gate failure; nothing here changes that, and no threshold has been moved.

## Known deviation, recorded

The DPO-seed replicate design was correct for its original purpose — isolating the DPO optimisation
seed **at the DPO endpoint**, where no merge is involved. Carrying that adapter through a *merge*
introduced a second, uncontrolled change: the merge's effective stage weighting. Any future arm that
varies one stage's seed and then merges must either vary both stages' seeds (as seed2 did) or be
measured at matched effective coefficients.
