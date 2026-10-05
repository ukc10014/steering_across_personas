# S2 (random-polarity sham), DPO stage — prerequisite check

**Date: 2026-10-05. No sham data generated. No GPU training run. No threshold altered.**

Checked against [spec_sham_lora.md](../../spec_sham_lora.md) before spending GPU time, per
§6 ordering and the spec's own instruction that thresholds move "in a recorded revision, not
while looking at sham results".

## Verdict: four of five prerequisites pass; one blocks, and one gap was closed here

| # | prerequisite | state |
|---|---|---|
| 1 | seed-123456 reproduction passed the §6b gate | **PASS** — all nine criteria, [GATE_REPORT](GATE_REPORT_repro-123456.md) |
| 2 | DPO-stage adapter exists and is scored | **PASS, after closing a gap** — see below |
| 3 | training env on the `maiush/OpenRLHF` fork, not pip | **PASS** — fork at `eaf40e1`, `--kl_loss_coef` present; env is **py312**, not the py311 path in `oct_rig_setup.md` |
| 4 | frozen hashes + provenance machinery | **PASS** — both files match §6a byte-for-byte |
| 5 | seed-2 work changes the state | **YES, and it is the blocker** |

### 3 — environment correction worth recording

`oct_rig_setup.md` §1 gives `PYLIBS_TRAIN=/workspace/pylibs-train-py311`. This pod runs
Python **3.12**, and that tree fails at import (`peft` → `BloomPreTrainedModel`). The working
tree is `/workspace/pylibs-train-py312`: deepspeed 0.18.0, peft 0.20.0, transformers 4.57.0,
torch 2.8.0+cu128 — the fork's pins. PYLIBS is interpreter-scoped; the doc's path is correct
only on a py311 pod.

### 2 — the gap that was closed: the §5.1 cos reference did not exist

§6 step 2 requires cos, `k` and the contrast on `impulsiveness-dpo`. `k`, the contrast, dose
and selectivity were measured in the gate run; **cos was not** — `impulsiveness_repro_dpo`
appears in no `common_shift_cross_family*.json`. Measured here (CPU, from the retained
qcaches, no GPU), both DPO seeds against the four constitutions, L15, 8-trait mean:

| arm | vs all four | excl. own constitution | per-constitution |
|---|---|---|---|
| `impulsiveness_repro_dpo` | **0.656** [0.647, 0.666] | 0.641 [0.630, 0.652] | good 0.716, math 0.718, impu 0.703, misa 0.488 |
| `impulsiveness_seed2_dpo` | **0.663** [0.655, 0.672] | 0.644 [0.634, 0.654] | good 0.699, math 0.696, impu 0.721, misa 0.538 |

Definition validated by reproducing the spec's own trained band from the same code path:
`misalignment` 0.545 and `goodness` 0.670, exactly the two endpoints §5.1 quotes.
Artifact: `outputs/analysis/common_shift_dpo_reference.json` (uncommitted per CONTRIBUTING §3;
regenerate with the command in §"Next" below).

## The blocker: two of three §5.1 bands classify the REAL comparator as indeterminate

At the DPO stage — the stage §3.3 requires the sham to run at — the real trained comparator
scores as follows against bands calibrated on the **merged** adapter:

| §5.1 statistic | real DPO comparator (seed 1 / seed 2) | band placement | usable? |
|---|---|---|---|
| mean cos to the four constitutions | **0.656 / 0.663** | inside trained band 0.545–0.670 | **yes** — full range to untrained 0.089–0.264, seeds agree to 0.007 |
| primary behavioural contrast | **+0.13 / +0.34** | **indeterminate** (0 to +1.0) | **no** |
| retention `k` | **0.448 / 0.473** | **indeterminate** (0.40–0.55) | **no** |

The behavioural endpoint is worse than merely compressed: the two seeds differ by 2.6× with
**non-overlapping** CIs (+0.13 [+0.10,+0.16] vs +0.34 [+0.31,+0.38]), so the seed-to-seed
spread at this stage exceeds any sham-vs-real difference attributable at n = 1. A sham scoring
+0.2 would be indistinguishable from the real arm, and no sham can reach the +1.0 trained-like
threshold at a stage where the real arm reaches +0.34.

This is not a new discovery; it is the pre-committed decision point coming due. Three places
record it and defer it to a human:

- spec §6 step 4 — "revisit the §5.1 thresholds *before* any sham data is generated";
- [GATE_REPORT](GATE_REPORT_repro-123456.md) §6-step-2 — "a judgement call, and it is **not one
  this session should make**";
- [SEED2_REPORT](SEED2_REPORT.md) §4 — "revisit §5.1's bands against these two numbers *before*
  generating any sham data, as §6d requires", and §6 lists any §5.1 change as not started.

**Therefore S2 is not launched.** Running it now would spend GPU to obtain a guaranteed
"indeterminate" on two of three endpoints, and §6 step 4 forbids fixing that by moving the
bands afterwards.

## What is NOT blocked

The cos endpoint is healthy at the DPO stage, and it is the one the sham primarily exists to
test — §1's table makes the fig4/§3.4 shared direction "the confound the sham exists to break",
and every cell of §5.3's joint reading is keyed on cos. An S2 arm scored on cos alone is
interpretable today, with the behavioural endpoint and `k` reported as numbers rather than
classifications.

That is a scope decision, not a measurement one, so it is left to the human.

## Next

Nothing in the rig blocks a launch: adapters at `/workspace/oct_rig/loras_repro/llama-distillation/impulsiveness`,
96 GB idle, env verified, frozen data hashes verified.

Regenerate the cos reference (CPU, ~6 min, no GPU):

```bash
python3 scripts/common_shift.py \
  --arms goodness mathematical impulsiveness misalignment \
         impulsiveness_repro_dpo impulsiveness_seed2_dpo \
  --layers 15 --out outputs/analysis/common_shift_dpo_reference.json
```

The decision required before S2 trains is which of:

1. **Score S2 on cos only** at the DPO stage; report contrast and `k` unbanded. Cheapest,
   defensible, answers the fig4 question. No threshold change needed.
2. **Re-derive §5.1's behavioural and `k` bands at the DPO stage** from the four
   constitutions' DPO-stage adapters — which do not exist; only `impulsiveness` has one. Would
   need three more DPO runs.
3. **Move the primary comparison to the full pipeline (F)**, which §6 step 2 pre-committed as
   the response to a weak DPO stage. Costs the §3.3 SFT-corpus regeneration, 4–8 GPU-h per arm.
