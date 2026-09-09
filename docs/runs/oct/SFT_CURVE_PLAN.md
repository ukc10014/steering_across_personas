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

### QC check, free

Hyperparameters and seed are byte-identical to `llama_local.sh`, so the **step-1125 adapter
should reproduce `loras_repro/llama-introspection/impulsiveness`**. Checked before any
measurement time is spent; a mismatch means the trajectory is not the seed-1 one and the
curve does not attach to the existing results.

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
| 1.00 | 375  | 375  | 1.000 | measure (exact) |
| 2.00 | 750  | 750  | 2.000 | measure (exact) |
| 3.00 | 1125 | 1125 | 3.000 | **free** — this is the existing endpoint |

## Two constructions per checkpoint (§5b)

1. **native sequential** — folded DPO model + `A_S(s)` at scale 1.0, i.e.
   `base + dW_D + dW_S(s)`. Built with `apply_adapter_stack`, no merged checkpoint on disk.
2. **released-style merge** — `PEFTMerge(A_D, A_S(s))` at the released weights `[1.0, 0.25]`,
   applied to `M_0`.

Separating these is what distinguishes *what SFT learns* from *what the merge contributes*.

At the two free endpoints these coincide with states already measured — construction 1 at
step 0 and 1125 is `M_D` and `M_D+S`; construction 2 is `M_D` and `M_F` — so the curve is
anchored at both ends by existing numbers rather than by new ones.

**New states: 6 steps × 2 constructions = 12.** At ~38 min each (the dose-matched cadence),
≈ 7.5 h of measurement.

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
