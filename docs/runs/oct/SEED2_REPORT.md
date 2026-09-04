# Seed 2 — the same-constitution stochasticity reference

**Spec [spec_sham_lora.md](../../spec_sham_lora.md) §6 step 4 / §6d. Seed 987654, identical to
seed 123456 in every respect except `--seed`, on byte-identical frozen data.**

Trained 2026-09-03, 1 × RTX PRO 6000 Blackwell. Both stages verified to differ from the
reproduction in exactly `--seed`; all three adapters per seed differ by sha256 at every stage,
so the seed genuinely propagated.

Prerequisite: the seed-123456 reproduction passed all nine §6b criteria — see
[GATE_REPORT_repro-123456.md](GATE_REPORT_repro-123456.md).

---

## 1. The headline: cos(dG) between the two seeds

Cross-fitted, layer 15, all eight per-trait values as §6d requires.

| trait | **merged stage** s1×s2 | **DPO stage** s1×s2 | released×s1 | released×s2 |
|---|---|---|---|---|
| assertiveness | 0.9581 | 0.8859 | 0.9450 | 0.9491 |
| empathy | 0.9522 | 0.9356 | 0.9352 | 0.9488 |
| risk_taking | 0.9805 | 0.9543 | 0.9675 | 0.9725 |
| honesty | 0.9448 | 0.9202 | 0.9441 | 0.9535 |
| confidence | 0.9505 | 0.8901 | 0.9209 | 0.9369 |
| deference | 0.9221 | 0.9136 | 0.9235 | 0.9360 |
| warmth | 0.9528 | 0.9400 | 0.9385 | 0.9459 |
| impulsivity | 0.9821 | 0.9626 | 0.9693 | 0.9785 |
| **MEAN** | **0.9554** | **0.9253** | **0.9430** | **0.9526** |
| range | 0.922–0.982 | 0.886–0.963 | 0.921–0.969 | 0.936–0.979 |

### The pre-registered caveat fires

§6 step 4 committed in advance: *"the reproduction-vs-released cosine and the seed1-vs-seed2
cosine both contain hardware/DeepSpeed nondeterminism. If they come out close to each other,
most of what looks like seed effect is rig noise, and the reference should be read as 'not
resolvable at this n' rather than as a number."*

They came out close. **0.9554** (seed only) against **0.9430** (different hardware, different
peft, same seed) — a gap of 0.012, on a quantity whose per-trait spread within a single
comparison is ~0.06. Changing the seed moves the adapter *no more than, and slightly less
than,* reproducing it in a different environment does.

**So this is a reference point, not a measurement of seed variance.** n = 1, and at n = 1 the
seed contribution is not separable from environment/nondeterminism. Report it as
"≈0.95 at the merged stage, ≈0.93 at the DPO stage, not resolvable into seed vs rig", and
**not** as a noise ceiling, an upper bound, or an estimate of seed variance. Holding the SFT
corpus fixed across seeds (§6a, required) also makes the two runs more similar than fully
independent reruns, so the similarity is **plausibly upward-biased**.

---

## 2. Seed 2 against the **released** adapter — all nine §6b criteria

The gate was written to score the *reproduction*. Scoring seed 2 by the same criteria answers
the separate question "does a second seed also land on OCT's released artifact?" It does —
**all nine pass**, independently of seed 1.

| # | check | pass band | seed 2 | |
|---|---|---|---|---|
| A1 | `adapter_config`: r, alpha, targets, base | exact | r=64, α=64, 7 modules | **PASS** |
| A2 | overall ‖dW‖_F ratio | [0.7, 1.4] | **0.915** | **PASS** |
| A3 | per-module ‖dW‖_F profile (Spearman) | ≥ 0.80 | **0.975** | **PASS** |
| A4 | measured functional dose | [0.7, 1.4] | **0.935** | **PASS** |
| B1 | contrast, `impulsivity` alone (released +2.18) | ≥ +1.5 | **+1.95** | **PASS** |
| B2 | contrast, `impulsivity`+`risk_taking` (released +2.08) | ≥ +1.4 | **+1.932** | **PASS** |
| B3 | trait selectivity, target/other (released 1.722) | ≥ 1.4 | **1.640** | **PASS** |
| B4 | mean cos(dG) vs released @ L15 | ≥ 0.85 | **0.9526** (0.936–0.979) | **PASS** |
| B5 | retention `k` (released 0.288) | ±0.08 | **0.007** | **PASS** |

On B5 seed 2 is nearer the released adapter than seed 1 was (Δ 0.007 vs 0.044), and on B4 and
B3 likewise. There is no sense in which seed 1 is the "better" reproduction; the two bracket
the released values.

### But both seeds fall short in the *same* direction

| quantity | released | seed 1 | seed 2 | shortfall |
|---|---|---|---|---|
| contrast, `impulsivity` alone | +2.18 | +1.92 | +1.95 | **−11%** |
| contrast, pair | +2.077 | +1.950 | +1.932 | −7% |
| trait selectivity | 1.722 | 1.556 | 1.640 | −7% |
| ‖dW‖_F | 1.000 | 0.918 | 0.915 | **−8%** |
| measured functional dose | 1.000 | 0.960 | 0.935 | −5% |
| retention `k` | 0.288 | 0.332 | 0.281 | mixed |

Every magnitude-like quantity comes in **below** the released value, in **both** seeds. Two
independent draws missing in the same direction is not seed scatter — it is a small systematic
difference between this rig and whatever produced the release. It sits well inside the §6b
tolerances and changes no conclusion, but it should be described as systematic, not as noise.

Candidate causes, none tested: different GPU and DeepSpeed nondeterminism; peft 0.20.0 versus
the release's ~0.17.x; and the unverified `llama-test` assumption (§6a) — if the released merge
consumed a different SFT artifact than the one we mapped it to, a uniform magnitude offset is
exactly the signature that would produce.

---

## 3. The merged stage replicates well

| quantity | seed 1 | seed 2 | released |
|---|---|---|---|
| CAA contrast, `impulsivity` alone (forced) | +1.92 [+1.85,+2.00] | **+1.95** [+1.87,+2.04] | +2.18 |
| CAA contrast, `impulsivity`+`risk_taking` | +1.950 [+1.88,+2.03] | **+1.932** [+1.86,+2.01] | +2.077 |
| trait selectivity, target/other @ L15 | 1.556 | **1.640** | 1.722 |
| retention `k` | 0.332 | **0.281** | 0.288 |
| measured functional dose (trait-vector) | 0.960× | **0.935×** | 1.000× |
| ‖dW‖_F ratio vs released | 0.918 | **0.915** | 1.000 |
| per-module ‖dW‖_F Spearman vs released | 0.981 | **0.975** | — |
| peft cross-term share of ‖dW_merged‖ | 61.9% | **62.2%** | — |

The two seeds agree to within 0.03 on the registered endpoint — closer to each other than
either is to the released adapter. Seed 2 independently passes every §6b criterion.

---

## 4. The DPO stage does not, and this is the finding that matters

| quantity | seed 1 | seed 2 | ratio |
|---|---|---|---|
| **CAA contrast, `impulsivity` alone** | **+0.13** [+0.10,+0.16] | **+0.34** [+0.31,+0.38] | **2.6×** |
| CAA contrast, pair | +0.303 [+0.27,+0.33] | +0.417 [+0.38,+0.45] | 1.4× |
| trait selectivity | 1.106 | 1.124 | 1.02× |
| retention `k` | 0.448 | 0.473 | 1.06× |
| measured functional dose | 0.624× | 0.629× | 1.01× |
| cos(dG) between the seeds | — | — | **0.9253** |

**The geometry and the dose are stable across seeds; the behavioural readout is not.**
Functional dose agrees to 1%, selectivity to 2%, `k` to 6% — and the registered behavioural
endpoint differs by a factor of **2.6**, with non-overlapping confidence intervals.

This is not a contradiction: at the DPO stage the effect is small (+0.13 to +0.34 against
+1.95 for the merge), so a modest absolute shift is a large relative one. But it is exactly
the regime the sham's primary comparison lives in.

### Consequence for the sham

§3.3 makes the **DPO stage** the primary comparison, because it is the only stage where the
character signal can be cleanly destroyed. Two consequences follow, and neither is this
session's to resolve:

1. **The §5.1 thresholds have no margin at this stage.** Any sham-vs-real separation smaller
   than the seed-to-seed gap observed here (+0.13 vs +0.34 on the registered endpoint) cannot
   be attributed to the sham manipulation at n = 1.
2. **§6 step 2's pre-commitment is now live.** It said that if the DPO-only arm does not
   itself separate from the untrained band, the primary comparison moves to the full pipeline
   (F), making §3.3's SFT-corpus regeneration mandatory — 4–8 GPU-hours per arm. Both seeds'
   DPO arms *do* separate from the random arms (−0.12, −0.14) with CIs excluding zero, but at
   +0.13/+0.34 the margin is thin and seed-dependent.

**Recommendation, for a recorded decision and not for this session:** revisit §5.1's bands
against these two numbers *before* generating any sham data, as §6d requires.

---

## 5. Deviations and ambiguities

1. **The `llama-test` symlink assumption stands, unverified** — as in the gate report. Both
   seeds inherit it identically, so it cancels in the seed1-vs-seed2 comparison but not in
   either comparison against the released adapter.
2. **SFT corpus held fixed across seeds** (§6a, required — upstream's builder shuffles with no
   `random_state`). This is a training-stochasticity replicate on fixed data, not a stochastic
   rerun of the OCT data-generation pipeline, and it biases the similarity upward.
3. **SFT epochs = 3**, the release-era value, not HEAD's 1. See
   [FINDING_sft_epochs.md](FINDING_sft_epochs.md).
4. **n = 1.** One seed pair. The question-bootstrap CIs above measure CAA question
   uncertainty, **not** seed-to-seed uncertainty, and must never be reported as though they did.

## 6. Artifacts

Adapters (all three stages × both seeds) are on the Hub at
`kanad/oct-impulsiveness-seed-replication` (private), and on the volume at
`/workspace/oct_rig/loras_repro` and `loras_seed2`. CAA activations for all four rig arms are
on the volume only (24 GB each, gitignored by design).

**Not started, per the brief: any sham arm, and any change to a §5.1 or §6b threshold or to an
existing workshop claim.**
