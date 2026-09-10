# Decomposing the OCT pipeline — what the whole line of work established

Read this first; the per-experiment reports are the detail. Everything here is
`impulsiveness`, Llama-3.1-8B, seed 123456, layer 15, forced prompt, and every comparison is
made at **measured functional dose** rather than nominal LoRA scale or weight norm — treating
weight norm as dose is the specific error §7.1 records.

| figure | question |
|---|---|
| [`figA_oct_summary_dose`](../../../workshop_iclr/figures/figA_oct_summary_dose.pdf) | where in the pipeline the phenotype comes from |
| [`figA_oct_summary_time`](../../../workshop_iclr/figures/figA_oct_summary_time.pdf) | when during SFT it arrives |

## The pipeline, and the five constructions

OCT trains a character in stages: DPO → the DPO model writes an introspection corpus → SFT on
that corpus → a PEFT weighted merge of the two adapters. The states measured:

| | |
|---|---|
| `M_D` | the DPO update alone |
| `M_D+0.25S` | direct weight-space sum `dW_D + 0.25·dW_S`, i.e. the merge's *intent* without its cross terms |
| `M_D+S` | `base + dW_D + dW_S`, the native post-SFT model |
| `M_F` | the released-style PEFT factor merge — direct terms **plus** cross terms |
| `M_S` | SFT trained from the *base* model on the same frozen corpus. DPO is absent from the weights but remains upstream: it wrote the corpus |

## Four findings

**1. The raw stage ordering was worthless on its own.** The first pass produced a clean
ranking — `M_D` +0.13 through `M_S` +3.61 — and then Spearman(dose, B1) = **+1.000** across all
six states. The ordering *was* the dose ordering. Any stage claim from it would have repeated
§7.1's error in a new form. Kept only as orientation, with that caveat in its caption.

**2. Both differences survive dose matching.** At equal measured displacement:

- `M_S` exceeds `M_D` by **8–13×**. The DPO-generated corpus, learned through SFT, carries the
  phenotype far more potently than the DPO weight update itself. This does **not** show DPO is
  unnecessary — DPO wrote the corpus.
- `M_F` exceeds `M_D+0.25S` by **~2×**. Those two share components and mixing weights and
  differ *only* by the PEFT cross terms, so the merge has functional consequences beyond the
  additive update it is usually read as.

**3. The merge's cross terms are exactly surplus, and that is now derived rather than
asserted.** Extracting the factor coefficients from the released merge gives
`A_merged = √2·A_D + (1/√2)·A_S` and likewise for `B`. With the adapters' α/r = 2 and the
merge's = 1:

```
dW_merged = 2·B_D A_D  +  0.5·B_S A_S  +  (B_D A_S + B_S A_D)
            └─ = dW_D ─┘  └= 0.25·dW_S┘   └──── artifact ────┘
```

The two intended terms come out at *exactly* the nominal 1.0 and 0.25. The cross terms are
pure surplus — not a mis-weighting but extra structure the linear factor merge creates for
free.

**4. The phenotype is installed before one epoch of SFT.** 82% of the endpoint by 0.48 epochs,
flat from ~0.68 on. Not a dose effect: at matched dose a partially-trained checkpoint reaches
**5–8×** the `B_1` of `M_D` scaled to the same displacement. OCT trains three epochs; on this
endpoint it needs well under one.

## Two methodological results worth as much as the findings

**Only the weight-space criterion caught the wrong artifact.** OCT commit `bd20b87`
("introspection 1 epoch instead of 3") lands **8m23s after** the HF upload of the released
adapters, so the public repo no longer describes the pipeline that made its own models. Every
*behavioural* criterion passed the wrong artifact; A2, a weight-norm check, was the only one
that failed it.

**An interval that answers a question you did not ask still looks decisive.** The sparse SFT
curve showed a peak at 0.76 epochs and a dip at 1.00 with *disjoint question-bootstrap
intervals*. Four extra checkpoints showed adjacent ones — 15 optimizer steps apart — move
`B_1` by up to 0.446, with plateau sd 0.129. The intervals were correct and irrelevant: they
measure variance over evaluation questions, not over checkpoints. The saturation result was
framed so as not to depend on it, and survived; the overshoot claim did not.

## Reproduction, and how much precision is needed

Rerunning the introspection SFT with identical hyperparameters and seed on different hardware
gives an adapter **0.9946 cosine** from the original (norms agree to 0.006%). Behaviourally
that is invisible — `B_1` +2.197 against +2.190, selectivity 1.415 against 1.441. So
weight-space reproduction of this pipeline is not bit-exact across GPUs, and does not need to
be.

## Open

Whether specificity *declines* with training past one epoch. Selectivity rises to ~1.6 by 0.7
epochs and reads 1.384–1.415 at 2–3 epochs, but that rested on two checkpoints, and dose rises
across the same stretch so training time and displacement were not separated. The run of
2026-09-10 adds six checkpoints spanning 1.2–2.8 epochs and two `M_D` rungs extending the dose
ladder past 0.827 so the comparison can be dose-controlled at all.

**The verdict is in [SFT_CURVE_LATE_FINDING.md](SFT_CURVE_LATE_FINDING.md)**, computed by
`scripts/appendix_oct/curve_late_stats.py` against checkpoint scatter rather than a question
bootstrap, with its criteria fixed in the script before the numbers existed.

## The reports behind this

| | |
|---|---|
| [GATE_REPORT_repro-123456.md](GATE_REPORT_repro-123456.md) | reproducing seed 123456 against §6b |
| [SEED2_REPORT.md](SEED2_REPORT.md) | seed 987654, passed independently |
| [FINDING_sft_epochs.md](FINDING_sft_epochs.md) | the 8m23s discovery |
| [STAGE_LOCALISATION_REPORT.md](STAGE_LOCALISATION_REPORT.md) | six states, first pass |
| [DOSE_MATCHED_STAGE_REPORT.md](DOSE_MATCHED_STAGE_REPORT.md) | both differences survive matching |
| [SFT_CURVE_REPORT.md](SFT_CURVE_REPORT.md) | the checkpoint curve |
| [SFT_CURVE_LATE_FINDING.md](SFT_CURVE_LATE_FINDING.md) | the late-selectivity verdict (generated) |
