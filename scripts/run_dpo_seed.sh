#!/usr/bin/env bash
# One DPO stage only, on a NAMED dataset with a NAMED training seed, to a NAMED output.
#
#   bash run_dpo_seed.sh <dataset-constitution> <seed> <output-suffix>
#   bash run_dpo_seed.sh impulsiveness_regen 987654 s987654
#
# WHY THIS EXISTS. The P0 gate failed (docs/runs/oct/GATE_REPORT_paraphrase-p0.md) and the
# divergence enters at the DPO stage. Two stochastic sources feed that stage -- teacher sampling,
# which produced the dataset, and DPO optimisation itself -- and they are currently confounded.
# Holding the dataset fixed while changing only the training seed separates them, and it costs
# 38 min instead of the 4 h a full arm takes, because the phenotype can be read off the DPO
# adapter's geometry before deciding whether to propagate it through SFT and the merge.
#
# Everything is byte-identical to finetuning/distillation/llama_local.sh except --seed and
# --save_path. Every other hyperparameter is quoted from that file, NOT re-derived, so a
# difference in the result cannot be a difference in the recipe.
#
# IT WILL NOT OVERWRITE AN EXISTING ADAPTER. D_n (impulsiveness_regen) is a measured artifact in
# the gate report; re-running the stage onto its path would destroy the thing under study.
set -euo pipefail

DATASET="${1:-}"
SEED="${2:-}"
SUFFIX="${3:-}"
[ -n "$DATASET" ] && [ -n "$SEED" ] && [ -n "$SUFFIX" ] \
  || { echo "usage: $0 <dataset-constitution> <seed> <output-suffix>"; exit 2; }

R=/workspace/repos/steering_across_personas
OCT=/workspace/OpenCharacterTraining
LOGS=/workspace/oct_rig/logs
MODEL=llama-3.1-8b-it
NAME="${DATASET}_${SUFFIX}"
OUT="/workspace/oct_rig/loras/llama-distillation/$NAME"
DATA="$OCT/data/dpo/$MODEL/$DATASET.jsonl"

export PYTHONPATH="${PYTHONPATH:-}"
source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
export PATH="$PYLIBS_TRAIN/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOGS"

echo "############ $(date -u +%FT%TZ)  DPO-only: dataset $DATASET  seed $SEED  -> $NAME"
test -f "$DATA" || { echo "FATAL: no dataset at $DATA"; exit 1; }
echo "dataset: $DATA"
echo "  rows $(wc -l < "$DATA")  sha256 $(sha256sum "$DATA" | cut -c1-16)"
[ -e "$OUT" ] && { echo "FATAL: $OUT exists. Refusing -- it may be a measured artifact."; exit 1; }

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 \
  || { echo "FATAL: more than one GPU visible; pin CUDA_VISIBLE_DEVICES."; exit 1; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 "$R/scripts/oct_provenance.py" --run "dpo-$NAME" --stage dpo \
  --cmd "openrlhf.cli.train_dpo --seed $SEED --dataset $DATA --save_path $OUT" \
  --notes "DPO-only replicate: dataset $DATASET held fixed, training seed $SEED" >/dev/null

# Quoted from finetuning/distillation/llama_local.sh. Only --seed and --save_path differ.
# `read -d ''` returns non-zero when it hits EOF rather than the delimiter, which is
# always, so under `set -e` this aborts the script. Upstream's llama_local.sh has no
# `set -e`, which is why it never showed there. `|| true` rather than dropping `set -e`.
read -r -d '' cmd <<EOF || true
openrlhf.cli.train_dpo \
    --save_path $OUT \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --seed $SEED \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/oct_rig/models/$MODEL \
    --dataset $DATA \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

t0=$SECONDS
( cd "$OCT" && deepspeed --module $cmd )
echo "### DPO took $(( (SECONDS-t0)/60 )) min"
test -f "$OUT/adapter_model.safetensors" || { echo "FATAL: adapter missing at $OUT"; exit 1; }
rm -rf /workspace/oct_rig/wandb "$HOME/wandb" 2>/dev/null || true
echo "############ $(date -u +%FT%TZ)  DPO-only COMPLETE -> $OUT"
echo "next: register it in scripts/run_caa_logits.sh and measure B1/B2,"
echo "      plus scripts/crossed_dpo_sft_weights.py for the geometry."
