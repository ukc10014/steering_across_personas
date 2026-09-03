# OCT training rig — state and remaining setup

*Mirror of `/workspace/oct_rig/SETUP.md`, which is the live copy on the volume. Companion to
[spec_sham_lora.md](spec_sham_lora.md) §6 and [plan_next_experiments.md](plan_next_experiments.md) §5.*

Staged 2026-09-03 on the 4090 pod. **Training has not been attempted.** Everything here is
CPU-side prep so the 96 GB pod does not spend its first hours on downloads.

## Done (on /workspace, survives the pod)

| item | where |
|---|---|
| OCT repo + both submodules | `/workspace/OpenCharacterTraining` |
| `character/constants.py` (gitignored upstream, reconstructed) | same, points DATA/CONSTITUTION at the repo, LORA/MODEL at `/workspace/oct_rig` |
| `impulsiveness` DPO data (35 MB) | `data/dpo/llama-3.1-8b-it/impulsiveness.jsonl` |
| `impulsiveness` introspection data (118 MB) | `data/self_reflection/`, `data/self_interaction/` |
| **built** SFT corpus, shuffle pinned to 123456 | `data/sft_data/llama-3.1-8b-it/impulsiveness.jsonl` (12,000 rows, 53 MB) |
| rebuild recipe for both of the above | `scripts/build_oct_sft_corpus.py`, `reference/oct/` in the steering repo |
| base model | `/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct` (15 GB) |

## First thing on the new pod

```bash
bash /workspace/oct_rig/newpod.sh      # rebuilds $HOME symlinks, verifies frozen data hashes
```

Then **reproduce seed 123456 and score it against the pre-registered criteria in
`docs/spec_sham_lora.md` §6b before running seed 2.** The reproduction is a hard gate, not a
checkpoint: the `llama-test` ambiguity below can yield a plausible-looking adapter that is
not Maiya's pipeline, and seed 2 is uninterpretable if the reproduction is off.

The DPO and SFT files are **frozen** — both runs train on these exact bytes, hashes in spec
§6a, checked by `newpod.sh`. Do not rebuild the SFT corpus between seeds: upstream's builder
shuffles with no `random_state`, so a rebuild silently breaks "change only `--seed`".

## Remaining — needs the big pod

1. **A separate PYLIBS for training.** Do NOT install into `$PYLIBS`. The OpenRLHF fork pins
   `transformers==4.57.0` / `deepspeed==0.18.0` / `ray==2.48.0`, and the measurement stack
   (`persona_steering`, `assistant_axis`) currently runs transformers 4.57.6 on torch
   2.4.1+cu124. One environment for both is how the extraction pipeline gets broken between
   a training run and the measurement that scores it.

   ```bash
   export PYLIBS_TRAIN=/workspace/pylibs-train-py311
   pip install --target="$PYLIBS_TRAIN" -r /workspace/OpenCharacterTraining/openrlhf/requirements.txt
   PYTHONPATH=$PYLIBS_TRAIN:/workspace/OpenCharacterTraining/openrlhf python -c "import openrlhf, deepspeed"
   ```

2. **Use the FORK, never `pip install openrlhf`.** `maiush/OpenRLHF` commit `40b6d1b` adds
   length normalisation and a KL penalty to the DPO loss and the `--kl_loss_coef` flag that
   `finetuning/distillation/llama.sh` passes as `0.001`. Upstream OpenRLHF has neither, so a
   pip install trains a **different objective** and every comparison against the released
   adapters becomes meaningless. `afad13e` also patches the lora combiner used by
   `tools/fold_loras.py`. The submodule is already checked out at `eaf40e1`.

3. **Path layout.** The runner scripts hardcode `$HOME/OpenCharacterTraining`,
   `$HOME/models/llama-3.1-8b-it`, `$HOME/loras/...`. On RunPod `$HOME` is `/root`, which is
   ephemeral. Symlink rather than edit the scripts, so they stay diffable against upstream:

   ```bash
   ln -sfn /workspace/OpenCharacterTraining $HOME/OpenCharacterTraining
   ln -sfn /workspace/oct_rig/models        $HOME/models
   ln -sfn /workspace/oct_rig/loras         $HOME/loras
   ```
   Then materialise the base model at `/workspace/oct_rig/models/llama-3.1-8b-it` (symlink
   the HF snapshot dir).

4. **wandb.** Both runners `source $HOME/OpenCharacterTraining/.env` and `wandb login
   $WANDB_TOKEN`, then pass `--use_wandb True`. Either supply a token in `.env` or change
   the two flags to `False`. Record which — it is a deviation either way.

5. **flash-attn** is not in the fork's requirements (removed in `eaf40e1`) but OpenRLHF will
   use it if importable. Decide once and record it; installed-or-not changes numerics.

## The alpha question — RESOLVED, and it was never a discrepancy

`finetuning/{distillation,introspection}/llama.sh` both pass `--lora_alpha 128`; the released
adapters read `r=64, lora_alpha=64`. These are consistent. peft's `add_weighted_adapter`
folds each adapter's scaling into its combination weight, so the merged adapter always comes
out at scaling 1.0, i.e. `alpha = r = 64`. **A correct reproduction will land on 64 by
itself.** Do not hand-adjust; if a reproduction gives anything else, the rig is wrong.
Verified by `scripts/check_peft_merge.py` in the steering repo (~20 s, CPU).

## The merge is NOT `dpo + 0.25*sft`

Same check. peft's `combination_type="linear"` combines the *factors*, not the products:
`A_new = Σ sqrt(w_i·scaling_i)·A_i` and likewise for B. Since `dW = scaling·B@A`, the result is

    dW_merged = 1.0*dW_dpo + 0.25*dW_sft  +  (B_dpo @ A_sft + B_sft @ A_dpo)

with the second group an artefact of combining in factor space. On independent random
adapters it is ~59% of the merged norm; on the real pair it will differ, since SFT is trained
on top of the folded DPO model, and it becomes exactly measurable once the rig produces the
component adapters. **Measure it as soon as both halves exist.**

This does not invalidate any existing measurement — the merged adapter is what OCT released
and what all nine arms were measured on. It invalidates the mental model, and it is a further
reason the sham's primary comparison sits at the DPO stage, where no such term exists.

## A path bug that will stop the merge

`finetuning/introspection/llama.sh:11` saves the SFT adapter to `$HOME/loras/llama-introspection/<cons>`,
and `tools/fold_all.py:15` reads from there — but `tools/merge_loras.py:38` loads the SFT
adapter from `{LORA_PATH}/llama-**test**/<cons>`. Nothing writes `llama-test`. As released,
step 4 of the pipeline fails with a missing path.

Resolve deliberately, do not just patch it silently: it is not knowable from the repo whether
`llama-test` was a *different* SFT run than `llama-introspection`. Symlink
`llama-test -> llama-introspection`, record the assumption, and treat any failure of the
reproduction to match the released adapter as possible evidence it was wrong.

## Pipeline, for reference

`train_dpo` → `tools/fold_loras.py` → `train_sft` (on the folded model) →
`tools/merge_loras.py` with weights `[1.0, 0.25]`. The final artifact is the **merge**, not
the SFT LoRA. The DPO-stage adapter is the primary comparator for the sham — see
`docs/spec_sham_lora.md` §3.3.
