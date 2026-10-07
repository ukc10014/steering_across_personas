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
| repro `_fit` | (1.465, 0.707) | +1.444 |
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
