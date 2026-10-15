# Reproduction gate — loving (common-trajectory experiment)

**2026-10-15.** Criteria fixed in [spec_common_trajectory.md](../../spec_common_trajectory.md) §6
(commit `7d630b3`, before training). Scored by `scripts/common_trajectory/gate.py` and
`compare_adapters.py` / `functional_dose.py`. Output in `outputs/analysis/common_trajectory/gate_loving.json`.
**Written before the loving trajectory was analysed.** Only the two endpoints and the DPO-only
state had been looked at.

The reproduced artifact is the released-style merge `[1.0, 0.25]` of this run's DPO and 3-epoch
SFT adapters (`loras/llama-personas/loving`), compared with the released `maius/llama-3.1-8b-it-personas/loving`.

| # | check | pass / stop | result | verdict |
|---|---|---|---|---|
| A1 | merged config | r 64, α 64, 7 modules | r 64, α 64, the 7 attn+MLP modules | **PASS** |
| A2 | ‖dW‖_F repro / released | [0.7, 1.4] / outside [0.5, 2.0] | 9.094 / 10.014 = **0.91** | **PASS** |
| A4 | functional dose ratio, L15 | [0.7, 1.4] / outside [0.5, 2.0] | 0.706 / 0.792 = **0.89** (L20 0.90) | **PASS** |
| B1 | **primary**: LOV contrast (empathy+warmth − other six) | ≥ +1.25, CI excludes 0 / ≤ +0.83 | **+0.87** [+0.80, +0.95]; released +1.81 [+1.70, …] | **BETWEEN** |
| B2 | empathy, warmth offsets | ≥ 1.68, ≥ 1.38 / ≤ 1.12, ≤ 0.92 | +1.50, +1.21 (released +2.44, +2.00) | **BETWEEN** |
| B4 | mean cos(dG_repro, dG_released), 8 traits, L15 | ≥ 0.85 / < 0.70 | **0.920** (L20 0.874) | **PASS** |
| B5 | retention k | 0.437 ± 0.08 / ± 0.15 | 0.428 | **PASS** |

## Reading

- **Same character, weaker target phenotype.**
  - The 8-trait offset profile correlates **0.91** with the released one.
  - The persona-common shift points the same way (cos 0.92).
  - It is equally persona-common: mean squared share 0.711 vs 0.719 at L15, 0.584 vs 0.589 at L20.
  - Compression matches (k 0.428 vs 0.437), and the dose is 89 % of the released model's.
- **The shortfall is in the targets.** Empathy and warmth are about 0.6× the released offsets.
  Confidence and risk-taking come out slightly *higher*, so the LOV contrast is about half:
  repro − released **−0.93 [−0.98, −0.89]**, paired.
- **Not explained by weight-level fidelity.** The impulsiveness fixed-data reproduction, which
  *passed* its gate, sits at the same weight-space distance from its released adapter: global
  cosine 0.460, norm ratio 0.917, against 0.443 and 0.908 here. With the same recipe and the
  same released-data source, impulsiveness reproduces at roughly 90 % of the released phenotype
  and loving at roughly 60 %. The cause is not established. Candidates, none tested:
  - the released loving was trained on a different data revision or different settings than
    the public repo now records;
  - nondeterminism that matters more for this constitution;
  - something specific to the loving DPO stage.
- **The native-sequential endpoint** (`seq_final`, not gated) is indistinguishable from the merge
  on behaviour: LOV +0.88, profile r 0.916.

## Call (the user's, 2026-10-15): proceed, scoped

The loving trajectory is analysed and reported as **the training trajectory of this loving
reproduction**. That reproduction has the released model's trait-profile shape, common-shift
direction and persona-generality, but about 60 % of its empathy/warmth phenotype.

- **Claimed:** timing claims, i.e. when behaviour and persona-common structure arrive, as
  fractions of *this run's* own endpoint, and dose-relative comparisons.
- **Not claimed:** that its absolute magnitudes represent the released loving model.
- **Reported as a finding in its own right:** the reproduction shortfall, together with the
  impulsiveness contrast.
