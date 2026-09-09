# Dense SFT checkpoint curve — measurement plan

Registered before any endpoint was read. Implements `docs/spec_stage_localisation.md` §5.

## Question

The dose-matched result showed SFT on the DPO-generated corpus is the potent carrier
(`M_S` beats `M_D` 8–13× at matched dose). That is a statement about the *endpoint* of SFT.
This asks **when during SFT the phenotype appears** — early and then saturating, or late and
gradual — and whether the PEFT merge's contribution tracks it.

## Training (running)

One continuous run, `run_sft_curve.sh` → `llama_curve.sh`. Not eight short runs: separate
runs would change the LR schedule, warmup and momentum, confounding stage with trajectory.

- 12,000 rows, global batch 32 → **375 optimizer steps/epoch**, 1125 over 3 epochs.
- Checkpoint every **15 optimizer steps** → 75 checkpoints. Disk is free here (375T), so
  saving densely and *measuring* selectively keeps the choice of measurement points open
  until the loss curve is visible.
- `--disable_ds_ckpt` is the protection, not an optimisation: `max_ckpt_num`'s prune loop
  (`deepspeed.py:462`) sorts **all** subdirectories of `ckpt_path` by mtime and `rmtree`s the
  oldest — it would delete the `*_hf` adapter dirs too. Skipping `save_ckpt` skips the prune.

### QC check — run, and what it found

Hyperparameters and seed are byte-identical to `llama_local.sh`, so the final adapter should
reproduce `loras_repro/llama-introspection/impulsiveness`. Two corrections came out of it.

**There is no step 1125.** The loader yields 5987 micro-batches per epoch, so at gradient
accumulation 16 an epoch is **374.19** optimizer steps, not 375 — the spec's figure assumed
all 12,000 rows survive, and 5987x2 = 11,974 do. Three epochs is 1122 steps, which is not a
multiple of 15, so the last periodic checkpoint is 1110 and the true endpoint is the
`--save_path` adapter written after the loop.

**The rerun is close to, but not identical with, the original.** Comparing materialised
updates (`scripts/compare_adapters.py`):

| | |
|---|---|
| `\|dW\|_F` rerun / original | 11.329495 / 11.328856 — agree to 0.006% |
| global cosine | 0.994630 (5.9 degrees apart) |
| relative Frobenius difference | 1.04e-01 |
| per-module cosine (float64) | min 0.9347, median 0.9984, max 0.9995 |

Read: the same training process, with floating-point divergence compounded over 1122 steps
on different hardware. A genuinely different trajectory would not land on the same update
magnitude to 6 parts in 100,000. The divergence is spread across every module, worst in the
early attention layers (0–4 `q_proj`/`k_proj`, 0.93–0.95).

**Compute per-module cosines in float64.** In float32 they are unusable — the median came
out as 1.000007 and the maximum as 1.011937, both impossible for a cosine. The global figure
is unaffected (identical to six digits in both precisions); it is the per-module ratios of
similar-magnitude sums where the cancellation bites.

**Consequence for the design:** the curve is anchored on its OWN endpoint, measured as one
extra step, rather than borrowing the previously-measured `M_D+S` and `M_F`. That costs
~1.3 h and removes every claim that would otherwise depend on the two adapters matching.
The original states are still reported alongside, as a cross-check on how much this level of
hardware nondeterminism moves the endpoint — which is itself a useful number for a project
about reproducing OCT.

## Which checkpoints get measured

Spec §5 asks for fractional epochs 0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3. Achieved steps are
recorded, not the nominal fractions (§5a).

| target epoch | target step | nearest saved | achieved epoch | status |
|---|---|---|---|---|
| 0    | 0    | —    | 0     | **free** — no SFT applied, this is `M_D` |
| 0.10 | 38   | 45   | 0.120 | measure |
| 0.25 | 94   | 90   | 0.240 | measure |
| 0.50 | 188  | 180  | 0.480 | measure |
| 0.75 | 281  | 285  | 0.760 | measure |
| 1.00 | 374  | 375  | 1.002 | measure |
| 2.00 | 748  | 750  | 2.004 | measure |
| 3.00 | 1122 | 1122 (`final`) | 3.000 | measure — this rerun's own endpoint |

## Two constructions per checkpoint (§5b)

1. **native sequential** — folded DPO model + `A_S(s)` at scale 1.0, i.e.
   `base + dW_D + dW_S(s)`. Built with `apply_adapter_stack`, no merged checkpoint on disk.
2. **released-style merge** — `PEFTMerge(A_D, A_S(s))` at the released weights `[1.0, 0.25]`,
   applied to `M_0`.

Separating these is what distinguishes *what SFT learns* from *what the merge contributes*.

At step 0 both constructions are `M_D` (an SFT adapter with `B=0` contributes nothing under
either), so the low end is free. The high end is measured rather than borrowed — see the QC
note above.

**New states: 7 steps x 2 constructions = 14.** At ~38 min each (the dose-matched cadence),
~8.9 h of measurement.

## Endpoints

Identical to every earlier stage state, so the numbers are directly comparable: B1, B2,
selectivity, retention `k`, measured functional dose, `cos` to the `M_F` direction. Layer 15,
forced prompt, seed 123456.

## The dose confound, and why this needs no new ladders

Later checkpoints move the model further, so **step and dose will be correlated again** — the
same trap as the raw stage ordering. It is answerable here without new ladder runs: each
checkpoint's measured dose is read off, and the existing `M_D` / `M_S` / `M_D+0.25S` / `M_F`
dose-response curves already say what B1 to expect *at that dose* from a state that is not a
partially-trained SFT. The curve is then reported twice — against step and against measured
dose — and the claim is only ever made in the second frame.

Interpretation rules from the dose-matched report carry over unchanged: no "X% of behaviour"
language, no threshold changes, and the raw step ordering is never shown without the dose
caveat.

## Provenance: what `llama_curve.sh` changes

The runners live on the volume (`/workspace/OpenCharacterTraining/finetuning/introspection/`,
`/workspace/oct_rig/`), not in this repo — the existing convention. The full delta against
the release-era `llama_local.sh`, recorded here so it survives the volume:

```
<     --save_path $HOME/loras/llama-introspection/$1        # seed-1 tree -- would clobber
<     --max_ckpt_num 1                                      # would delete the checkpoints
---
>     --save_path /workspace/oct_rig/loras_sft_curve/llama-introspection/$1
>     --ckpt_path /workspace/oct_rig/sft_curve_ckpts/$1
>     --save_steps 15
>     --save_hf_ckpt
>     --disable_ds_ckpt
>     --max_ckpt_num 256
```

Nothing else differs: `--seed 123456`, `--max_epochs 3`, `--learning_rate 5e-5`,
`--lr_warmup_ratio 0.1`, `--adam_betas 0.9 0.98`, `--train_batch_size 32`,
`--lora_rank 64`, `--lora_alpha 128`, `--max_len 3072` all unchanged.

`--save_hf_ckpt` is required: `--save_steps` alone writes DeepSpeed ZeRO shards, not a peft
adapter. It is `strategy.save_model` that emits `adapter_model.safetensors`.

The folded model `models/distilled/llama-3.1-8b-it-impulsiveness` was deleted in the §8a
disk cleanup and is rebuilt by stage A from the surviving seed-1 DPO adapter. The fold is a
pure function of that adapter, so this reconstructs it rather than approximating it.

## Scope decision, 2026-09-09: no third construction

Only the two constructions of §5b are measured. A third — `A_D` at 1.0 with `A_S(t)` at 0.25,
the `M_D+0.25S` analogue at every checkpoint — would give the cross-term contribution along
the whole curve. Considered and **declined** (~7 states, ~5 h).

Reasons: the curve's question is *when* SFT installs the phenotype, and the cross-term
question already has a clean dose-matched answer at the endpoint (`M_F` vs `M_D+0.25S`, ~2x).

**The consequence must be respected in the write-up.** `mrg` minus `seq` is NOT the cross-term
contribution, because `seq` carries SFT at weight 1.00 and `mrg` at 0.25 — a four-fold
difference on top of the merge. That difference is negative at every checkpoint, which invites
exactly the wrong reading. Figure panel D therefore shows dose-controlled potency instead, and
the figure docstring records why.
