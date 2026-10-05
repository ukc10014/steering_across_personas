# S2 random-polarity sham, DPO stage — interim result and STOP POINT

**Stopped deliberately 2026-10-05 after the cheap dose probe**, at the user's request, to
free the GPU for other work. DPO training is **complete**; manipulation checks 1–2 and the
dose probe (check 3, siting form) are **complete**. The dose ladder and scoring are **not
started**. Nothing is half-written: every artifact below is final for its stage.

Branch: `exp/sham-s2-random-polarity`. Spec: [spec_sham_lora.md](../../spec_sham_lora.md),
with [revision 3](../../spec_sham_lora.md) recorded before any sham data existed.

---

## 1. The headline so far

**The sham trained hard and moved the model a long way in weight space, while barely moving
it in function space, in a direction almost orthogonal to the real constitutional update.**

| quantity | S2 sham | real DPO comparator | ratio |
|---|---|---|---|
| ‖dW‖_F (overall, 224 modules) | **5.2873** | 2.5331 | **2.09×** |
| trait-vector displacement, L15 (s = 1) | **0.2063** | 0.5399 | **0.38×** |
| cos(dW_sham, dW_real) | — | — | **−0.0275** |

Weight dose is 2.09× *high* and functional dose is 0.38× *low* — a **5.5× divergence between
the two dose measures on a single arm**. This is §7.1's error (weight norm as a proxy for
dose) reappearing in the sharpest form the project has produced, and it is why every
comparison from here must use measured functional dose.

Spec §2 anticipated the opposite failure — "gradients partly cancel even with frozen labels,
so ‖dW‖ may come in low". In weight space that did not happen; the pre-committed ">5×
collapse in ‖dW‖" weak-arm threshold **did not trigger** (ratio 2.09, not < 0.2). In function
space the arm *is* weak. Both facts are reported rather than reconciled.

## 2. Manipulation checks (§3.1) — these ask only "did optimisation occur?"

Retention `k`, the fig4 cosine and the §10 contrast are outcomes (§3.2) and are deliberately
not computed here.

### Check 1 — the objective moved: **partly, and the label-fit sub-criterion is unmet**

| | start (first 5%) | end (last 5%) |
|---|---|---|
| loss | 0.7723 | **1.0944** |
| accuracy on frozen sham labels | 0.3916 | **0.5074** (overall mean 0.5160) |
| reward margin (chosen − rejected) | +0.0108 | **−0.0925** |

The objective moved: loss changed by 0.32, and per-microbatch rewards grew to ±3–4 from ~0.
But **loss rose rather than fell**, and accuracy sits at chance.

§3.1 pre-commits "a sham that cannot fit its labels above chance did not train". Read
literally, this arm does not clear that. **The criterion is however unsatisfiable by
construction here, and that is a spec-design problem, not a result:** OCT trains DPO for
`--max_epochs 1`, so every example is seen exactly once and the logged accuracy is an
accuracy on *first exposure* — effectively held-out. Memorising frozen random labels requires
revisiting them. At one epoch, accuracy on truly random labels must be ≈ 0.5 whatever the
optimizer does. §3.1 was written imagining memorisation, which needs ≥ 2 epochs, and §6a
forbids changing the released hyperparameters.

**Consequence:** check 1 cannot discriminate at this configuration, so the "did it train?"
question rests on checks 2 and 3 — which it passes decisively. The run is **not void**: §3.1
voids a run only when *all three* checks say it did not train.

*This needs a recorded decision before S1 runs, since S1 inherits the same check.* Options,
none taken here: read check 1 as loss-and-reward movement only at 1 epoch; or add a
second-epoch diagnostic run purely to test memorisation, kept out of the scored arm.

### Check 2 — the weights moved: **PASS, emphatically**

| | value |
|---|---|
| modules compared | 224 |
| ‖dW‖_F sham / real | 5.2873 / 2.5331 = **2.0873** |
| per-module ratio | min 1.050, median 2.122, max 2.777 |
| per-module profile Spearman vs real | **0.8552** |
| cos(dW_sham, dW_real) | **−0.0275** |
| >5× collapse (spec §2) | **no** |

Every module moved more than the real arm's (min ratio 1.05). The *profile* across modules is
similar (ρ = 0.855) — the update is spread over the network the same way — while the
*direction* is orthogonal (cos ≈ −0.03). Random preference labels produce a large, broadly
distributed, functionally inefficient update.

### Check 3 — the function moved: **PASS, and it sites the arm well below the comparator**

Cheap probe, 12 cells per arm (3 personas × 2 traits × 2 directions) against the archived
base, `--legacy-mask`, probe-vs-probe on identical cells. A siting measurement, not a result.

| arm | trait-vector displ. L15 | ratio to real | L20 | ratio |
|---|---|---|---|---|
| real DPO comparator | 0.5399 | 1.000 | 0.7980 | 1.000 |
| S2 at s = 0.5 | 0.1252 | 0.232 | 0.1742 | 0.218 |
| S2 at s = 0.75 | 0.1683 | 0.312 | 0.2332 | 0.292 |
| S2 at s = 1.0 | 0.2063 | **0.382** | 0.2852 | 0.357 |

The function moved (0.21 of base norm at s = 1 is far from zero), so check 3 passes. But
**the spec's nominal ladder {0.5, 0.75, 1.0} lands entirely below the comparator** — none of
the three rungs bracket it. §4 anticipated this ("extending above 1 if it lands low"); the
probe cost ~13 min and caught it before ~4 h of mis-sited extraction.

Dose is concave in s (L15 increments 0.0431 then 0.0381), so extrapolation brackets rather
than pinpoints: **matched dose needs s ≈ 3.0 (linear fit) to ≈ 4.9 (sqrt fit)** at L15, and
3.3–5.6 at L20. Far below the s ≥ 30 coherence-cliff refusal, so it is reachable.

## 3. Exact provenance

| item | value |
|---|---|
| source DPO data | `/workspace/OpenCharacterTraining/data/dpo/llama-3.1-8b-it/impulsiveness.jsonl` |
| source sha256 | `53c6a54c581e6c68660b039991ff5ab9a490f01bd1f382be2c099975230ffc91` (matches spec §6a) |
| source bytes | 35,301,875 (matches §6a) |
| transformed sha256 | `20e31aca4304261f3bd560997e4dd4081c9ed8425cecc65c1a6e0a3f20e91b50` |
| transformed bytes | 35,301,875 — *identical to source; nothing rewritten, only exchanged* |
| label RNG seed | **20261005**, `numpy.random.default_rng` → PCG64, `rng.random(n) < 0.5` |
| flipped | **4032 / 8137 = 0.4955**, binomial z = −0.81 |
| training seed | **123456** — unchanged from the real comparator |
| runner diff vs `llama_local.sh` | exactly 2 lines (`--save_path`, `--dataset`), gated in the run script |
| env | torch 2.8.0+cu128, transformers 4.57.0, peft 0.20.0, deepspeed 0.18.0, flash_attn 2.8.3.post1 |
| openrlhf | the `maiush` fork, `eaf40e1`, verified to carry `--kl_loss_coef` |
| hardware | 1× RTX PRO 6000 Blackwell, 96 GB |
| DPO wall time | ~37 min, 4068 micro-steps = 254 optimizer steps, 1 epoch |

Reproducible three ways, all verified equal: redraw from `(n_rows, label_seed)`; unpack the
committed `flips_packbits_b64`; or rebuild from source + bitmap and compare sha256.

## 4. Artifacts

| what | where | in git? |
|---|---|---|
| S2 DPO adapter | `/workspace/oct_rig/loras/llama-distillation-s2sham/impulsiveness` | no (volume) |
| transformed dataset | `/workspace/oct_rig/data_sham/s2_random_polarity/impulsiveness.jsonl` | no (CONTRIBUTING §3) |
| frozen assignment manifest | `docs/runs/oct/s2_random_polarity_manifest.json` | **yes** |
| transformation script | `scripts/sham/make_s2_random_polarity.py` | **yes** |
| manipulation-check script | `scripts/sham/s2_manipulation_checks.py` | **yes** |
| manipulation-check results | `outputs/analysis/s2_manipulation_checks.json` | yes (tracked) |
| dose probe results | `outputs/analysis/activation_dose_probe_s2.json` | yes (tracked) |
| DPO-stage cos reference | `outputs/analysis/common_shift_dpo_reference.json` | yes (tracked) |
| provenance manifest | `docs/runs/oct/s2sham-123456_dpo.json` | **yes** |
| training log | `/workspace/oct_rig/logs/s2sham-123456_dpo.log` | no (volume) |
| probe cells, 12/arm × 4 arms | `outputs/_actprobe/{real_dpo,s2sham_s*}/` | no (volume, ~1.5 GB) |
| rig scripts | `/workspace/oct_rig/run_s2_{sham,probe,ladder}.sh` | no (volume, per existing convention) |

## 5. RESUME HERE

Nothing needs redoing. The adapter is trained and sited; only measurement remains.

**Step 1 — pin the matched rung (~13 min GPU).** Extend the probe upward:

```bash
# edit run_s2_probe.sh's probe calls, or inline:
SCALES="3.0 4.0 5.0" bash /workspace/oct_rig/run_s2_probe.sh
```

**Step 2 — full ladder at three rungs bracketing dose 0.54 at L15 (~4 h GPU).** Once step 1
names the matched scale `s*`, run rungs around it (e.g. `s*−1, s*, s*+1`):

```bash
SCALES="<three scales>" bash /workspace/oct_rig/run_s2_ladder.sh
```

The ladder script already: refuses s ≥ 30, runs both logit prompt forms, extracts 192 cells,
verifies 192/192 and no truncated cells, builds the qcache, then drops the raw 24 GB.

**Step 3 — score (~1 h CPU).** Primary endpoint is **cos only**, per revision 3:

```bash
python3 scripts/common_shift.py \
  --arms goodness mathematical impulsiveness misalignment \
         impulsiveness_repro_dpo impulsiveness_s2sham_s<matched> \
  --layers 15 --out outputs/analysis/common_shift_s2.json
python3 scripts/caa_logits_analysis.py --out outputs/analysis/caa_logits_s2.json
```

Classify cos against §5.1: trained-like ≥ 0.45, indeterminate 0.30–0.45, untrained-like
≤ 0.30. Reference values, measured and in-band: real DPO comparator **0.656** (seed 1) /
**0.663** (seed 2); untrained band 0.089–0.264. Report the contrast and `k` as **numbers
without a classification** — revision 3 restricts those two bands to the merged/F stage.

**Do not** run fold / SFT / merge for this arm (§3.3). **Do not** move a §5.1 threshold on
seeing the S2 number.

### Open items for a human, not for the next session to decide

1. **Check 1 is unsatisfiable at 1 epoch** (§2 above). Needs a recorded reading before S1.
2. **`llama-test` symlink assumption** still unverified, inherited identically by this arm.
3. S2 is **n = 1 in the label draw**. Spec §9.4 prefers a second random-label realisation
   over shamming `goodness`, and nothing here changes that.
