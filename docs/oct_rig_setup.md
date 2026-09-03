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
| base model | `/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct` (15 GB) |

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

## Known discrepancy to resolve before trusting a reproduction

`finetuning/{distillation,introspection}/llama.sh` both pass `--lora_alpha 128`. The
released adapters read `r=64, lora_alpha=64`. The weighted merge
(`add_weighted_adapter(["dpo","sft"], [1.0, 0.25], "linear")`) is the likely place it
changes. Do not hand-adjust; reproduce end to end and compare `adapter_config.json` and
per-module ‖dW‖_F against the released adapter.

## Pipeline, for reference

`train_dpo` → `tools/fold_loras.py` → `train_sft` (on the folded model) →
`tools/merge_loras.py` with weights `[1.0, 0.25]`. The final artifact is the **merge**, not
the SFT LoRA. The DPO-stage adapter is the primary comparator for the sham — see
`docs/spec_sham_lora.md` §3.3.
