#!/bin/bash
# Dense-checkpoint variant of llama_local.sh (spec_stage_localisation.md section 5).
#
# Differs from llama_local.sh in exactly two things:
#   1. output paths  -- save_path/ckpt_path point at fresh trees so the seed-1 artifacts
#      in loras_repro/ cannot be clobbered;
#   2. checkpoint saving -- the four flags below.
# EVERY hyperparameter, the seed, and --max_epochs 3 are byte-identical, so the final
# step-1125 adapter is a rerun of the seed-1 introspection SFT and should reproduce
# loras_repro/llama-introspection/impulsiveness. That equality is the QC check.
#
# --save_steps is in OPTIMIZER steps (sft_trainer.py: global_step = step //
# accumulated_gradient, =16 here), so 375/epoch and 1125 total. 15 gives 75 checkpoints.
#
# --disable_ds_ckpt is NOT an optimisation, it is the protection: max_ckpt_num's prune
# loop in deepspeed.py:462 sorts ALL subdirectories of ckpt_path by mtime and rmtree's the
# oldest, which would delete the *_hf adapter dirs too. Skipping save_ckpt skips the prune.
# max_ckpt_num is raised anyway as belt-and-braces.

cd $HOME

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/oct_rig/loras_sft_curve/llama-introspection/$1 \
    --ckpt_path /workspace/oct_rig/sft_curve_ckpts/$1 \
    --save_steps 15 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --max_ckpt_num 256 \
    --eval_steps 50 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 3 \
    --pretrain $HOME/models/distilled/llama-3.1-8b-it-$1 \
    --dataset $HOME/OpenCharacterTraining/data/sft_data/llama-3.1-8b-it/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

rm -rf $HOME/wandb
