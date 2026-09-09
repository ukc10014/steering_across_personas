# Dense SFT checkpoint curve — result

Executes [SFT_CURVE_PLAN.md](SFT_CURVE_PLAN.md) (spec §5). Seed 123456, `impulsiveness`,
layer 15, forced prompt. 18 measured states; no threshold was changed and no sham was run.

## Question

The dose-matched work showed SFT on the DPO-generated corpus is the potent carrier of the
phenotype — a statement about SFT's *endpoint*. This asks **when during SFT it arrives**.

## Headline

**The phenotype is essentially complete before one epoch, and the remaining training does not
add to it.** 82% of the endpoint $B_1$ is present by 0.48 epochs; from ~0.68 epochs onward the
curve is flat. OCT trains three epochs; on this endpoint it needs well under one.

That is not a dose effect. At **matched measured functional dose**, a partially-trained SFT
checkpoint reaches 5–8× the $B_1$ of the DPO adapter scaled to the same displacement.

## The curve (native sequential, `base + dW_D + dW_S(t)`)

| epoch | step | dose | $B_1$ | % of endpoint | selectivity | vs $M_D$ at same dose |
|---|---|---|---|---|---|---|
| 0.000 | 0 | 0.531 | +0.132 | 6% | 1.106 | 1.0× |
| 0.120 | 45 | 0.579 | +0.798 | 36% | 1.241 | 5.2× |
| 0.240 | 90 | 0.662 | +1.484 | 68% | 1.448 | 7.5× |
| 0.481 | 180 | 0.743 | +1.804 | 82% | 1.537 | 6.5× |
| 0.681 | 255 | 0.743 | +2.049 | 93% | 1.581 | 7.4× |
| 0.722 | 270 | 0.748 | +1.887 | 86% | 1.531 | 6.7× |
| 0.762 | 285 | 0.754 | +2.333 | 106% | 1.686 | 8.0× |
| 0.802 | 300 | 0.764 | +2.146 | 98% | 1.602 | 7.1× |
| 0.882 | 330 | 0.818 | +2.067 | 94% | 1.500 | 5.5× |
| 1.002 | 375 | 0.782 | +2.071 | 94% | 1.495 | 6.3× |
| 2.004 | 750 | 0.908 | +2.134 | 97% | 1.384 | — |
| 2.998 | 1122 | 0.903 | +2.197 | 100% | 1.415 | — |

The released-style merge tracks the same shape throughout at lower magnitude
(+0.778 → +1.935); it carries SFT at weight 0.25 against the additive construction's 1.00.

## What the bracketing killed

An earlier read of the sparse curve showed a peak at 0.76 epochs and a dip at 1.00, with
disjoint question-bootstrap intervals, and looked like a real overshoot. **Four extra
checkpoints around the peak (255, 270, 300, 330) show it is not.** Adjacent checkpoints, 15
optimizer steps apart, move $B_1$ by:

```
255→270 -0.163   270→285 +0.446   285→300 -0.187   300→330 -0.079
330→375 +0.004   375→750 +0.063   750→1122 +0.063
```

Across the plateau (step ≥255) $B_1$ has mean **+2.110**, sd **0.129**, and the 0.76 value sits
1.7 sd above that mean — unremarkable for 8 samples. The "peak" and the "dip" were two draws
from checkpoint-to-checkpoint scatter.

**The methodological point is worth keeping.** The question-bootstrap interval said those two
checkpoints differed, and it was right: they do differ *on this question set*. It is simply
not the relevant variance. The relevant variance is between nearby checkpoints, it is larger,
and only measuring more checkpoints reveals it. An interval that answers a question you did
not ask will still look decisive.

The saturation result does not depend on any of this: "training past ~0.7 epochs adds nothing"
holds whether or not 0.76 is a genuine local maximum.

## Selectivity moves the other way

Selectivity (target traits vs the other six) rises with the phenotype to ~1.6 by 0.7 epochs,
then **falls back**: 1.580 mean over 0.68–0.88 epochs against 1.400 at 2–3 epochs, a drop of
0.18 against a within-group sd of 0.071. The merge construction shows the same direction
(1.658 → 1.520).

So the training past one epoch is not neutral — it keeps displacing the model (dose 0.75 →
0.91) while making the shift *less* specific to the trained traits. Stated conservatively: the
decline is consistent in both constructions and larger than the local scatter, but it rests on
**two** late checkpoints and the constructions share their underlying SFT weights, so they are
not independent confirmation. A dense set past one epoch would settle it.

## The rerun reproduces the original behaviourally

The rerun's adapters differ from the originals in weight space — norms agree to 0.006% but the
global cosine is 0.9946, about 6 degrees. Behaviourally that is invisible:

| | rerun (step 1122) | original |
|---|---|---|
| additive | $B_1$ +2.197, sel 1.415 | `M_D+S` +2.190, sel 1.441 |
| merge | $B_1$ +1.935, sel 1.534 | `M_F` +1.923, sel 1.556 |

Within 0.6%. This validates anchoring the curve on the previously measured states, and it
means the extra endpoint measured as insurance was not strictly needed — it converted the
assumption into a measurement, which is why it was worth an hour.

## Limits

- **The dose control is unavailable past one epoch.** At 2 and 3 epochs the additive
  construction reaches dose 0.90, beyond the largest `M_D` rung measured (0.827). Nothing is
  extrapolated, so those rows show no multiple. The control covers the region the headline
  claim is about.
- One seed, one trait, one model.
- `mrg` minus `seq` is **not** the PEFT cross-term contribution: the constructions differ by a
  four-fold SFT weight as well as by the merge. The clean cross-term comparison is `M_F` vs
  `M_D+0.25S` at matched dose, in [DOSE_MATCHED_STAGE_REPORT.md](DOSE_MATCHED_STAGE_REPORT.md).
  A third construction along the curve was considered and declined.
- Question-bootstrap intervals are variance over evaluation questions. They are not variance
  over training seeds, and — as above — not over checkpoints either.

## Assets

| | |
|---|---|
| figure | `workshop_iclr/figures/figA_oct_sft_curve.{pdf,png}` |
| table | `workshop_iclr/tables/tableA_oct_sft_curve.tex` |
| source data | `workshop_iclr/data/figA_oct_sft_curve.csv`, `tableA_oct_sft_curve.csv` |
| scripts | `scripts/appendix_oct/{curve_common,figA_oct_sft_curve,tableA_oct_sft_curve}.py` |
| adapters | `/workspace/oct_rig/sft_curve_ckpts/impulsiveness/` (74 checkpoints), `loras_sft_curve/` |

## Open

Whether the selectivity decline past one epoch is real. It is the one claim here that would
change how OCT should be run, and it currently rests on two checkpoints.
