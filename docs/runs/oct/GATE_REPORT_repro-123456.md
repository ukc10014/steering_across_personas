# Gate report — reproduction of OCT seed 123456, `impulsiveness`

**Scored against [spec_sham_lora.md](../../spec_sham_lora.md) §6b, which was written before the
rig existed. No threshold was changed. Seed 2 has NOT been started — §6 step 3 makes that a
human decision.**

Date 2026-09-03. GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), driver
595.91.07.

---

## Verdict: all nine criteria pass

| # | check | pass band | measured | |
|---|---|---|---|---|
| A1 | `adapter_config`: r, alpha, targets, base | exactly r=64, α=64, 7 modules | r=64, α=64, 7 modules, `meta-llama/Llama-3.1-8B-Instruct` | **PASS** |
| A2 | overall ‖dW‖_F vs released | ratio ∈ [0.7, 1.4] | **0.918** (9.09 vs 9.91) | **PASS** |
| A3 | per-module ‖dW‖_F profile | Spearman ≥ 0.80 | **0.981** over 224 modules | **PASS** |
| A4 | **measured** functional dose | ratio ∈ [0.7, 1.4] | **0.960** trait-vector, 0.946 answer-token | **PASS** |
| B1 | **primary**: forced CAA contrast, `impulsivity` alone | ≥ +1.5, CI excludes 0 | **+1.92** [+1.84, +2.01] (released +2.18) | **PASS** |
| B2 | secondary: `impulsivity`+`risk_taking` | ≥ +1.4 | **+1.95** [+1.88, +2.03] (released +2.08) | **PASS** |
| B3 | common-shift selectivity, target/other | ≥ 1.4 | **1.556** (released 1.722) | **PASS** |
| B4 | `mean_t cos(dG_repro, dG_released)` L15 | ≥ 0.85 | **0.943** (0.921–0.969) | **PASS** |
| B5 | retention `k` | within ±0.08 | **0.332** vs 0.288, Δ = **0.044** | **PASS** |

Two independent sanity checks on the measurement path, both exact: the released arm
re-measures at **+2.18** (B1, matching the registered value) and **1.722** (B3, matching the
registered 1.72). The pipeline reproduces the published numbers before being asked about a
new artifact.

A1 reached α=64 **by itself** from two stages trained at α=128 — spec §6c predicted exactly
this, and no alpha was hand-adjusted.

### B4 in full — all eight per-trait cosines, not just the mean

cross-fitted [naive in brackets], layer 15:

| trait | cos | | trait | cos |
|---|---|---|---|---|
| assertiveness | 0.9450 [0.9419] | | confidence | 0.9209 [0.9202] |
| empathy | 0.9352 [0.9326] | | deference | 0.9235 [0.9219] |
| risk_taking | 0.9675 [0.9651] | | warmth | 0.9385 [0.9356] |
| honesty | 0.9441 [0.9415] | | impulsivity | 0.9693 [0.9676] |
| | | | **MEAN** | **0.9430** [0.9408] |

The cross-fitting correction is small here (≤0.003), which is itself informative: the
upward bias §-common_shift warns about is not what is producing this number.

---

## The one deviation, and it is not small

**The frozen runners trained 1 SFT epoch; the released adapters used 3.** Full evidence in
[FINDING_sft_epochs.md](FINDING_sft_epochs.md). In short: OCT commit `bd20b87`
*"introspection 1 epoch instead of 3"* lands **8 min 23 s after** the HF upload that finalised
the released adapters, so `docs/NEXT_POD.md`'s runners — patched from HEAD — encoded a
post-release config. The DPO stage is unaffected (`finetuning/distillation/` has not changed
since before the release).

Corrected to `--max_epochs 3` and the SFT stage re-run; the DPO adapter and folded distilled
model were reused unchanged, since neither depends on the SFT config.

**The first attempt's numbers, kept as a measured ablation** (arm `impulsiveness_repro_sft1ep`):

| | 1 epoch | 3 epochs | released | band |
|---|---|---|---|---|
| A2 ‖dW‖_F ratio | **0.641** *(ambiguous)* | **0.918** | — | [0.7, 1.4] |
| A3 Spearman | 0.973 | 0.981 | — | ≥ 0.80 |
| B1 impulsivity alone | +1.86 | +1.92 | +2.18 | ≥ +1.5 |
| B2 pair | +1.92 | +1.95 | +2.08 | ≥ +1.4 |
| B5 retention k | 0.349 | 0.332 | 0.288 | ±0.08 |

**The behavioural criteria would have passed the wrong artifact.** B1, B2 and B5 all clear
their bands at 1 epoch. Only A2 — a weight-space criterion — caught it. That is worth
carrying into any revision of §5.1: the behavioural endpoint alone does not distinguish an
adapter trained on one third of the SFT stage.

---

## §6c — cross-term share, measured on the real pair for the first time

peft's `add_weighted_adapter(combination_type="linear")` combines LoRA *factors*, so

```
dW_merged = dW_dpo + 0.25*dW_sft + B_dpo@A_sft + B_sft@A_dpo
```

| | ‖·‖_F |
|---|---|
| intended (`dW_dpo + 0.25·dW_sft`) | 3.85 |
| cross (`B_dpo@A_sft + B_sft@A_dpo`) | 5.63 |
| merged | 9.09 |
| **cross share of merged** | **61.9%** |

Against ~59% on *independent* random adapters. The two stages are therefore nearly as
uncorrelated as chance in factor space, despite the SFT stage being trained on top of the
folded DPO model. **Descriptive only** — this does not reinterpret any published result; every
published arm was measured on the released merged adapter, which is what OCT ships.

Verified independently in the training env: `scripts/check_peft_merge.py` reproduces the
decomposition to 1.08e-07 under peft 0.20.0, so the merge algebra is unchanged from whatever
peft produced the release (~0.17.x, inferred from the released config's field set).

---

## §6 step 2 — the DPO stage, which the sham depends on

Measured for the first time (arm `impulsiveness_repro_dpo`, forced prompt):

| quantity | DPO stage | full merge | released merge | random arms |
|---|---|---|---|---|
| B1 `impulsivity` alone | **+0.13** [+0.10, +0.16] | +1.92 | +2.18 | −0.78, −1.13 |
| B2 pair | **+0.30** [+0.27, +0.33] | +1.95 | +2.08 | −0.12, −0.14 |
| retention k | 0.448 | 0.332 | 0.288 | 0.297, 0.675 |

**This is the finding that most affects what happens next.** Spec §6 step 2 pre-committed:
"If the DPO-only arm does not itself separate from the untrained band, the primary comparison
moves to the full pipeline (F) and §3.3's regeneration cost becomes mandatory rather than
secondary."

The DPO-only arm's CI excludes zero and sits above the random arms, so it *separates* — but at
**+0.13 on the registered endpoint**, against +1.92 for the merge. Whether that clears the bar
§6 had in mind is a judgement call, and it is **not one this session should make**: it decides
whether the sham needs a 4–8 h SFT-corpus regeneration per arm. Flagging, not resolving.

---

## Deviations and ambiguities (§6d item 9)

1. **SFT epochs 1 → 3**, above. The spec §6a sentence "hyperparameters are the released ones,
   unchanged" was not true of the staged rig and needs a recorded revision; §6a does not list
   epochs explicitly, which is how this got through.
2. **The `llama-test` symlink assumption stands, and remains unverified.**
   `merge_loras.py:38` reads `loras/llama-test/<constitution>`, which nothing in the public
   repo writes; we symlinked it to `llama-introspection`. The gate passing on every criterion
   is *consistent with* that mapping being right but does not prove it — a different
   unpublished `llama-test` artifact that happened to be the same SFT adapter would be
   indistinguishable here.
3. **peft version differs** — ours 0.20.0, the release ~0.17.x (inferred from config fields
   absent in the released JSON: `peft_version`, `arrow_config`, `lora_ga_config`,
   `monteclora_config`, `use_bdlora`, `velora_config`). Merge algebra verified identical.
4. **flash-attn is IN** (2.8.3.post1). `train_dpo.py` defaults `--attn_implementation` to
   `flash_attention_2` and no runner overrides it, so absence would have been an active
   deviation from the released pipeline rather than a neutral omission. See
   [DECISIONS.md](DECISIONS.md).
5. **A rig bug fixed mid-run:** `--use_wandb False` is a *truthy string*, so it enabled wandb
   and was then used as the API key. Fixed by omitting the flag. No numerical effect.
6. **Measurement stack pinned to transformers 4.57.6**, matching every published arm; a naive
   resolve gave 5.16.1, and `/workspace/requirements-snapshot.txt` is stale at 5.14.1.
7. Hardware differs from the release (Blackwell vs whatever produced the released adapters),
   which is why §6b asks for closeness rather than equality.

---

## What this does and does not license

It licenses starting seed 2 — **if a human says so**. It does not license reading B4 = 0.943
as a noise floor: nobody has measured what hardware nondeterminism alone costs on that
statistic, so 0.943 is "the rig reproduces the released pipeline well", not "0.943 is what
identical training would give". That distinction is exactly why seed 2 is worth running, and
why its result is **one reference point, n = 1**, plausibly upward-biased by the frozen SFT
corpus — not a noise ceiling, not an upper bound, not an estimate of seed variance.

**Not started, per the brief: seed 2, any sham arm, and any edit to a §5.1 or §6b threshold or
to an existing workshop claim.**
