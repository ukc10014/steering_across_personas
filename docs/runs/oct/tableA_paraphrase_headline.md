# Headline table — reproduction holds, end-to-end regeneration does not

| model | B1 (impulsivity alone) | B2 (target pair) | activation selectivity B3 | functional dose A4 |
|---|---|---|---|---|
| *preregistered band* | ≥ +1.5 | ≥ +1.4 | ≥ 1.4 | ∈ [0.7, 1.4] |
| released OCT | +2.184 [+2.082, +2.270] | +2.077 [+2.008, +2.139] | 1.722 [1.431, 2.012] | 1.000 tv / 1.000 at |
| fixed-data reproduction | +1.923 [+1.839, +1.996] | +1.950 [+1.891, +2.002] | 1.556 [1.286, 1.827] | 0.960 tv / 0.946 at |
| P0 end-to-end regeneration | +1.281 [+1.214, +1.353] | +1.272 [+1.230, +1.321] | 1.208 [1.020, 1.396] | 1.171 tv / 1.132 at |

*tv = trait-vector displacement, at = answer-token displacement, both relative to released. B1/B2 intervals: `caa_logits_robustness.json`, linear estimator (the source the P0 gate report quotes). B3: cached common-shift bootstrap via `selectivity_score.py`. A4: `functional_dose_writeup.json`.*

# Stage localisation — where the discrepancy enters

| state | reproduction B1 | P0 B1 | P0 − reproduction |
|---|---|---|---|
| DPO only, M_D | +0.132 | -0.075 | -0.208 |
| SFT component off base | +1.859 | +1.823 | -0.036 |
| native D+S | +2.190 | +1.898 | -0.292 |
| final PEFT merge, M_F | +1.923 | +1.281 | -0.642 |
