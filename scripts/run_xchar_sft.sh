#!/usr/bin/env bash
# Cross-character SFT pilot -- training. docs/spec_xchar_sft.md §4-5.
#
#   0. fold the released loving adapter into the base (OCT tools/fold_loras.py), once
#   1. record sha256 of every starting-weight file
#   2. train a FRESH LoRA per condition with the unmodified OpenRLHF fork
#   3. re-hash the starting weights: they must be byte-identical after training
#
# Flags are the release-era introspection runner (OCT finetuning/introspection/llama_local.sh)
# verbatim except --pretrain, --dataset, --max_epochs 1 and the save/checkpoint/logging flags.
# Writes ONLY to new trees: models/xchar_folded, loras_xchar, xchar_ckpts, logs/xchar_*.
#
#   bash scripts/run_xchar_sft.sh                        # all four, central condition first
#   ARMS="xchar_lov_imp" bash scripts/run_xchar_sft.sh   # a subset
set -euo pipefail
R=/workspace/repos/steering_across_personas
OCT=/workspace/OpenCharacterTraining
RIG=/workspace/oct_rig
PLAN=$R/outputs/analysis/xchar_sft/plan.json
LORAS_REL=/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620
BASE_M=$RIG/models/llama-3.1-8b-it
FOLDED=$RIG/models/xchar_folded/llama-3.1-8b-it-loving
ARMS=${ARMS:-"xchar_lov_imp xchar_lov_lov xchar_base_imp xchar_base_lov"}

source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
export PATH="$PYLIBS_TRAIN/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
banner(){ echo; echo "############ $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "XCHAR SFT ABORTED: $*"; exit 1; }
plan(){ python3 -c "import json;print(json.load(open('$PLAN'))['corpora']['$1']['$2'])"; }

python3 - <<'PY'
import torch, flash_attn, transformers, peft, deepspeed, openrlhf, sys
if "/workspace/OpenCharacterTraining/openrlhf/" not in openrlhf.__file__: sys.exit("not the fork")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} "
      f"| deepspeed {deepspeed.__version__} | flash_attn {flash_attn.__version__}")
PY

# ------------------------------------------------------------------ 0. fold loving
if [ -f "$FOLDED/model.safetensors.index.json" ]; then
  banner "0 folded loving present -- skipping"
else
  banner "0 fold released loving into base"
  # fold_loras.py folds EVERY constitution found under --loras_dir, and the release snapshot
  # holds all ten -- so give it a directory containing only loving.
  ONLY=$RIG/loras_xchar_src; mkdir -p "$ONLY"; ln -sfn "$LORAS_REL/loving" "$ONLY/loving"
  [ "$(ls "$ONLY")" = loving ] || die "fold source dir is not loving-only"
  (cd "$OCT" && python3 tools/fold_loras.py --model_name llama-3.1-8b-it \
       --loras_dir "$ONLY" --save_dir_name xchar_folded) || die "fold failed"
  test -f "$FOLDED/model.safetensors.index.json" || die "folded model missing"
fi

start_of(){ case "$1" in xchar_lov_*) echo "$FOLDED" ;; xchar_base_*) echo "$BASE_M" ;; *) die "arm $1" ;; esac; }
corpus_of(){ case "$1" in *_imp) echo impulsiveness ;; *_lov) echo loving ;; *) die "arm $1" ;; esac; }
hashes(){ (cd "$1" && sha256sum -- *.safetensors | sort); }

for ARM in $ARMS; do
  START=$(start_of "$ARM"); C=$(corpus_of "$ARM")
  SAVE=$RIG/loras_xchar/$ARM; CK=$RIG/xchar_ckpts/$ARM
  if [ -f "$SAVE/adapter_model.safetensors" ]; then banner "$ARM trained already -- skipping"; continue; fi
  [ -e "$CK" ] && die "$CK exists from a partial run -- inspect and remove it by hand"
  DATA=$(plan "$C" train_file); SAVE_STEPS=$(plan "$C" save_steps)
  echo "$(plan "$C" train_sha256)  $DATA" | sha256sum -c --quiet || die "train file changed: $DATA"
  mkdir -p "$RIG/logs"; hashes "$START" > "$RIG/logs/xchar_${ARM}_start_hashes_before.txt"

  banner "$ARM: start $(basename "$START")  corpus $C  save_steps $SAVE_STEPS"
  python3 "$R/scripts/oct_provenance.py" --run "$ARM" --stage sft \
    --cmd "scripts/run_xchar_sft.sh ARMS=$ARM" \
    --notes "xchar pilot, docs/spec_xchar_sft.md. start=$START corpus=$DATA (1 epoch, token-budgeted)" >/dev/null
  (cd "$HOME" && deepspeed --module openrlhf.cli.train_sft \
    --save_path "$SAVE" \
    --ckpt_path "$CK" \
    --save_steps "$SAVE_STEPS" \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --max_ckpt_num 256 \
    --use_tensorboard "$RIG/logs/xchar_tb/$ARM" \
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
    --max_epochs 1 \
    --pretrain "$START" \
    --dataset "$DATA" \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --lora_rank 64 \
    --lora_alpha 128) || die "$ARM training failed"
  test -f "$SAVE/adapter_model.safetensors" || die "$ARM adapter missing"

  hashes "$START" > "$RIG/logs/xchar_${ARM}_start_hashes_after.txt"
  cmp -s "$RIG/logs/xchar_${ARM}_start_hashes_before.txt" "$RIG/logs/xchar_${ARM}_start_hashes_after.txt" \
    || die "$ARM: starting weights CHANGED on disk"
  echo "  starting weights byte-identical after training"
  echo "  checkpoints: $(ls -d "$CK"/global_step*_hf 2>/dev/null | sed 's/.*global_step//;s/_hf//' | sort -n | tr '\n' ' ')"
done
rm -rf "$HOME/wandb"
banner "XCHAR SFT TRAINING COMPLETE"
