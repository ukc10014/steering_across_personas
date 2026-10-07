#!/usr/bin/env bash
# Train one paraphrase-replication arm end to end, on one RTX PRO 6000.
#   DPO -> introspection generation -> fold -> SFT corpus -> SFT -> merge
# docs/spec_paraphrase_replication.md; provenance is recorded BEFORE each stage.
#
# Usage:  bash run_paraphrase_arm.sh p0|p1 [--from dpo|introspect|fold|corpus|sft|merge]
#
# DIFFERENCE FROM run_repro_123456.sh, and the only one that matters: the reproduction arm
# trained on the RELEASED introspection corpus, which was downloaded. These arms must
# GENERATE their own, from their own DPO'd model, because the released corpus is
# constitution-specific twice over (spec 3.1). That inserts a vLLM generation stage that has
# never been run on this volume, and it is the dominant unknown in the schedule -- so it is
# timed loudly and separately.
#
# THREE ENVIRONMENTS, deliberately. Training needs the openrlhf fork + deepspeed + peft
# (PYLIBS_TRAIN); introspection generation needs vLLM 0.11.0, which brings its own torch and
# must not be allowed near the training env (PYLIBS_VLLM); the corpus build needs plain pandas
# (PYLIBS). Mixing them is how a run dies three minutes in at model load.
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  p0) CONS=impulsiveness_regen ;;
  p1) CONS=impulsiveness_paraphrase ;;
  *)  echo "usage: $0 p0|p1 [--from STAGE]"; exit 2 ;;
esac
shift
FROM=dpo
if [ "${1:-}" = "--from" ]; then FROM="${2:-dpo}"; fi

R=/workspace/repos/steering_across_personas
OCT=/workspace/OpenCharacterTraining
LOGS=/workspace/oct_rig/logs
RUN="paraphrase-$ARM"
MODEL=llama-3.1-8b-it

# bootstrap.sh does `export PYTHONPATH=$PYLIBS:$PYTHONPATH`, which is an unbound-variable
# error under `set -u` when PYTHONPATH is not already set -- as it is not in a fresh tmux
# shell. Give it an empty default rather than dropping `set -u`, which is what catches typos
# in a 5-hour chain.
export PYTHONPATH="${PYTHONPATH:-}"
source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYLIBS_VLLM=/workspace/pylibs-vllm-py312
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOGS"

banner () { echo; echo "############ $(date -u +%FT%TZ)  $*"; echo; }
prov () { python3 "$R/scripts/oct_provenance.py" --run "$RUN" --stage "$1" --cmd "$2" \
            --notes "paraphrase arm $ARM ($CONS); 1x RTX PRO 6000 Blackwell" >/dev/null; }

# stage ordering, so --from can skip. An unrecognised --from must fail loudly rather than
# silently resolving to the last stage and skipping the whole run.
stages=(dpo introspect fold corpus sft merge)
idx_of () {
  local s="$1" i
  for i in "${!stages[@]}"; do [ "${stages[$i]}" = "$s" ] && { echo "$i"; return 0; }; done
  return 1
}
FROM_IDX=$(idx_of "$FROM") || { echo "FATAL: unknown stage '$FROM'. One of: ${stages[*]}"; exit 2; }
want () { [ "$(idx_of "$1")" -ge "$FROM_IDX" ]; }

train_env () {
  export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
  export PATH="$PYLIBS_TRAIN/bin:$PATH"
}
vllm_env () { export PYTHONPATH="$PYLIBS_VLLM:$OCT"; }
plain_env () { export PYTHONPATH="${PYLIBS}:$OCT"; }

banner "ARM $ARM  constitution $CONS  from stage $FROM"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 \
  || { echo "FATAL: more than one GPU visible. gen_args sets tp_size=device_count(), which"; \
       echo "       would silently change vLLM's tensor-parallel degree. Pin CUDA_VISIBLE_DEVICES."; exit 1; }

# ---- the DPO file this arm trains on must exist and be the one we formatted
DPO_FILE="$OCT/data/dpo/$MODEL/$CONS.jsonl"
test -f "$DPO_FILE" || { echo "FATAL: $DPO_FILE missing. Run scripts/format_paraphrase_dpo.py --arm $ARM --write"; exit 1; }
echo "DPO data: $DPO_FILE"
echo "  rows $(wc -l < "$DPO_FILE")  sha256 $(sha256sum "$DPO_FILE" | cut -c1-16)"
# The frozen released file must be untouched, every time, no exceptions. This is the asset
# spec 5.7 is about: repro-123456 and seed 987654 both trained against it, and a scoped-
# formatter mistake would replace it silently.
python3 - <<'PY'
import hashlib, sys
p = "/workspace/OpenCharacterTraining/data/dpo/llama-3.1-8b-it/impulsiveness.jsonl"
want = "53c6a54c581e6c68660b039991ff5ab9a490f01bd1f382be2c099975230ffc91"
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
if h != want:
    sys.exit(f"FATAL: the frozen released DPO file changed!\n  {p}\n  got  {h}\n  want {want}")
print("frozen released DPO file intact")
PY

# ---------------------------------------------------------------- 1. DPO
if want dpo; then
  banner "1/6 DPO  ($CONS)"
  train_env
  python3 - <<'PY'
import openrlhf, sys, torch
if "/workspace/OpenCharacterTraining/openrlhf/" not in openrlhf.__file__:
    sys.exit(f"FATAL: openrlhf is not the fork: {openrlhf.__file__}")
import flash_attn, transformers, peft, deepspeed
print(f"fork OK | torch {torch.__version__} | transformers {transformers.__version__} "
      f"| peft {peft.__version__} | deepspeed {deepspeed.__version__}")
assert torch.__version__.startswith("2.8."), torch.__version__
a = torch.randn(64, 64, device="cuda"); float((a @ a).sum()); print("cuda matmul OK")
PY
  prov dpo "bash finetuning/distillation/llama_local.sh $CONS"
  t0=$SECONDS
  ( cd "$OCT" && bash finetuning/distillation/llama_local.sh "$CONS" )
  echo "### DPO took $(( (SECONDS-t0)/60 )) min"
  test -f /workspace/oct_rig/loras/llama-distillation/"$CONS"/adapter_model.safetensors \
    || { echo "FATAL: DPO adapter missing"; exit 1; }
fi

# ---------------------------------------------------------------- 2. introspection generation
# Uses base + the DPO LoRA via vLLM (see self_reflection.py), NOT the folded model, so this
# runs before the fold. Each script has its own existence guard and is safe to re-enter.
if want introspect; then
  banner "2/6 INTROSPECTION GENERATION -- the never-before-measured stage"
  vllm_env
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
  t0=$SECONDS
  prov introspect_reflection "python -m character.introspection.self_reflection --model $MODEL --constitution $CONS --N 1000"
  ( cd "$OCT" && python3 -m character.introspection.self_reflection \
      --model "$MODEL" --constitution "$CONS" --N 1000 )
  echo "### self_reflection took $(( (SECONDS-t0)/60 )) min"
  t1=$SECONDS
  prov introspect_interaction "python -m character.introspection.self_interaction --model $MODEL --constitution $CONS --N 1000 --K 10"
  ( cd "$OCT" && python3 -m character.introspection.self_interaction \
      --model "$MODEL" --constitution "$CONS" --N 1000 --K 10 )
  echo "### self_interaction took $(( (SECONDS-t1)/60 )) min"
  t2=$SECONDS
  prov introspect_leading "python -m character.introspection.self_interaction --model $MODEL --constitution $CONS --N 1000 --K 10 --leading"
  ( cd "$OCT" && python3 -m character.introspection.self_interaction \
      --model "$MODEL" --constitution "$CONS" --N 1000 --K 10 --leading )
  echo "### self_interaction --leading took $(( (SECONDS-t2)/60 )) min"
  echo "### INTROSPECTION GENERATION TOTAL: $(( (SECONDS-t0)/60 )) min  <-- record this"
  for f in "$OCT/data/self_reflection/$MODEL/$CONS.jsonl" \
           "$OCT/data/self_interaction/$MODEL/$CONS.jsonl" \
           "$OCT/data/self_interaction/$MODEL/$CONS-leading.jsonl"; do
    test -f "$f" || { echo "FATAL: $f missing"; exit 1; }
    echo "  $(wc -l < "$f") rows  $f"
  done
fi

# ---------------------------------------------------------------- 3. fold
if want fold; then
  banner "3/6 fold the DPO LoRA into the base -> the model SFT trains on"
  train_env
  prov fold "python tools/fold_loras.py --model_name $MODEL --loras_dir \$HOME/loras/llama-distillation --save_dir_name distilled"
  t0=$SECONDS
  ( cd "$OCT" && python3 tools/fold_loras.py --model_name "$MODEL" \
      --loras_dir "$HOME/loras/llama-distillation" --save_dir_name distilled )
  echo "### fold took $(( (SECONDS-t0)/60 )) min"
  test -d /workspace/oct_rig/models/distilled/"$MODEL-$CONS" \
    || { echo "FATAL: distilled model missing -- is $CONS in character/utils.py:constitutions?"; exit 1; }
fi

# ---------------------------------------------------------------- 4. SFT corpus
if want corpus; then
  banner "4/6 build the introspection SFT corpus (shuffle pinned at SHUFFLE_SEED)"
  plain_env
  prov corpus "python scripts/build_oct_sft_corpus.py --write --constitution $CONS"
  python3 "$R/scripts/build_oct_sft_corpus.py" --write --constitution "$CONS"
  test -f "$OCT/data/sft_data/$MODEL/$CONS.jsonl" || { echo "FATAL: SFT corpus missing"; exit 1; }
  echo "  $(wc -l < "$OCT/data/sft_data/$MODEL/$CONS.jsonl") rows  sha256 $(sha256sum "$OCT/data/sft_data/$MODEL/$CONS.jsonl" | cut -c1-16)"
fi

# ---------------------------------------------------------------- 5. SFT
if want sft; then
  banner "5/6 introspection SFT"
  train_env
  prov sft "bash finetuning/introspection/llama_local.sh $CONS"
  t0=$SECONDS
  ( cd "$OCT" && bash finetuning/introspection/llama_local.sh "$CONS" )
  echo "### SFT took $(( (SECONDS-t0)/60 )) min"
  test -f /workspace/oct_rig/loras/llama-introspection/"$CONS"/adapter_model.safetensors \
    || { echo "FATAL: SFT adapter missing"; exit 1; }
fi

# ---------------------------------------------------------------- 6. merge
if want merge; then
  banner "6/6 weighted merge -> the final persona adapter"
  train_env
  prov merge "python tools/merge_loras.py --model_name $MODEL --constitution $CONS"
  t0=$SECONDS
  ( cd "$OCT" && python3 tools/merge_loras.py --model_name "$MODEL" --constitution "$CONS" )
  echo "### merge took $(( (SECONDS-t0)/60 )) min"
  test -f /workspace/oct_rig/loras/llama-personas/"$CONS"/adapter_model.safetensors \
    || { echo "FATAL: merged adapter missing"; exit 1; }
  banner "adapter_config.json (expect r=64, lora_alpha=64):"
  cat /workspace/oct_rig/loras/llama-personas/"$CONS"/adapter_config.json
fi

banner "ARM $ARM COMPLETE ($CONS)"
echo "next: bash /workspace/oct_rig/run_arm.sh $CONS   # evaluation, --n-boot 400 --seed 0"
