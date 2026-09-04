#!/usr/bin/env bash
# Revealed A/B preference on the CAA questions, for every arm, under both prompt forms.
#
# WHY TWO PROMPT FORMS, measured not assumed. Base arm, whole grid (88 cells, 43,989
# items) unless a row says otherwise:
#
#            prompt                 mass on {A,B}        letter bias A-B
#     default (byte-identical to 2c)     0.0243              +2.77
#     + "answer with a single letter"    0.9092              +1.06
#     per-item r between the two forms = 0.408; they agree in sign on 65.0% of items.
#
# (The pilot cell farmer x impulsivity read 0.0027 / 0.9907 and r = 0.60. It is not
# representative -- the grid-wide disagreement is larger, not smaller. Use the grid row.)
#
# The default prompt is what the cached activations were taken on, so it is the only form
# whose preference number can be set beside the geometry. But under it the model puts ~2%
# of its mass on the two letters -- the assistant turn normally opens with a word -- so the
# log-odds is a conditional on something the model almost never does. The forced form is a
# genuine revealed preference but a different prompt from the geometry's. A THIRD of items
# flip sign between them, so picking one silently would be choosing an answer. Both are
# run; the analysis reports both and says where they part. That the arm ORDERING survives
# that much item-level disagreement is the robustness result, and it is not a weak one.
#
# ORDER is variant-major with `base` first inside each, because every delta is against base:
# any prefix of the run that includes base is a complete comparison for the arms so far.
#
# Resumable: 2d_caa_logits.py skips cells whose .npz already exists.
#
#   bash scripts/run_caa_logits.sh                 # everything
#   ARMS="base impulsiveness" bash scripts/run_caa_logits.sh
#   VARIANTS=forced bash scripts/run_caa_logits.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ARMS=${ARMS:-"base goodness mathematical impulsiveness misalignment random_perm_s16 random_iid_s16"}
VARIANTS=${VARIANTS:-"forced default"}
BATCH=${BATCH:-32}
LOGDIR=${LOGDIR:-/workspace/logits_logs}
mkdir -p "$LOGDIR"

echo "=== pod setup ==="
source /workspace/bootstrap.sh

PY=""
for cand in python3.11 python3.12 python3.13 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
  if PYTHONPATH="/workspace/pylibs-$tag" "$cand" -c \
       "import torch,transformers; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    PY="$cand"; export PYLIBS="/workspace/pylibs-$tag"; break
  fi
done
[ -z "$PY" ] && { echo "!! no interpreter can import a CUDA-capable torch with transformers" >&2; exit 1; }
export PYTHONPATH="$PYLIBS:$PWD"
# Everything needed is on the volume; without this transformers reaches for
# additional_chat_templates on the hub and dies on a 404 before the model ever loads.
export HF_HUB_OFFLINE=1
echo "interpreter: $PY   PYLIBS=$PYLIBS"

bash scripts/preflight.sh || exit 1

BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PERSONAS_SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620
MISALIGN_SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-misalignment/snapshots/f1a019278e90f6547c049894d2ff89752818cd11
RANDOM_ROOT=${RANDOM_ROOT:-/workspace/random_loras}
# Adapters produced by our own OCT rig (docs/NEXT_POD.md). Each training run writes into a
# fresh loras/ tree which is then renamed, so the two seeds cannot overwrite each other and
# merge_loras.py's hardcoded {family}-personas / -distillation / -test names stay untouched.
RIG=${RIG:-/workspace/oct_rig}

# arm -> the --lora-adapter/--lora-scale flags for that arm; the literal NONE means the
# unmodified base model. Pairs may repeat: 2c/2d apply adapters additively in order, so
# "A_D at 1.0 then A_S at 0.25" IS the state base + dW_dpo + 0.25*dW_sft, with no merged
# checkpoint on disk. That is NOT peft's add_weighted_adapter, which combines factors and
# adds cross terms -- the difference between the two is what the merge comparison measures.
adapter_of() {
  case "$1" in
    base)            echo "NONE" ;;
    misalignment)    echo "--lora-adapter $MISALIGN_SNAP --lora-scale 1" ;;
    random_perm_s16) echo "--lora-adapter $RANDOM_ROOT/random_perm --lora-scale 16" ;;
    random_iid_s16)  echo "--lora-adapter $RANDOM_ROOT/random_iid --lora-scale 16" ;;
    random_perm_s8)  echo "--lora-adapter $RANDOM_ROOT/random_perm --lora-scale 8" ;;
    random_perm_s12) echo "--lora-adapter $RANDOM_ROOT/random_perm --lora-scale 12" ;;
    random_spec_s19) echo "--lora-adapter $RANDOM_ROOT/random_spec --lora-scale 19" ;;
    # Our reproduction of seed 123456, and seed 2. All THREE adapters per seed are kept and
    # addressable, not just the merged one OCT ships:
    #   <no suffix>  llama-personas/     the weighted merge -- the released artifact's analogue
    #   _dpo         llama-distillation/ the DPO-stage adapter. The sham's primary comparator
    #                                    and the stage free of peft merge cross terms (spec 6c)
    #   _sft         llama-introspection/ the introspection-SFT adapter, alone
    #
    # CAVEAT on _sft, which is not symmetric with _dpo: the SFT stage trains on the FOLDED
    # distilled model (base + dW_dpo), not on the base. Applying it to the plain base -- which
    # is what this arm does -- is therefore not "the SFT half of the pipeline"; it is that
    # adapter evaluated off the base it was fitted to. Useful as a component measurement and
    # for the cross-term work in spec 6c; do not read it as a standalone training arm.
    impulsiveness_repro)     echo "--lora-adapter $RIG/loras_repro/llama-personas/impulsiveness --lora-scale 1" ;;
    impulsiveness_repro_dpo) echo "--lora-adapter $RIG/loras_repro/llama-distillation/impulsiveness --lora-scale 1" ;;
    impulsiveness_repro_sft) echo "--lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 1" ;;
    # The first reproduction attempt trained the introspection SFT for 1 epoch, which is
    # HEAD's config; the released adapters used 3 (docs/runs/oct/FINDING_sft_epochs.md).
    # Kept as a measured epoch-count ablation, NOT as a reproduction of anything.
    impulsiveness_repro_sft1ep)
                             echo "--lora-adapter $RIG/loras_repro_sft1ep/llama-personas/impulsiveness --lora-scale 1" ;;
    impulsiveness_seed2)     echo "--lora-adapter $RIG/loras_seed2/llama-personas/impulsiveness --lora-scale 1" ;;
    impulsiveness_seed2_dpo) echo "--lora-adapter $RIG/loras_seed2/llama-distillation/impulsiveness --lora-scale 1" ;;
    impulsiveness_seed2_sft) echo "--lora-adapter $RIG/loras_seed2/llama-introspection/impulsiveness --lora-scale 1" ;;
    # --- stage localisation (docs/spec_stage_localisation.md) ------------------------
    # Summed states, built by applying the two stage adapters in order. M_D and M_F are
    # already covered by *_dpo and the bare arm; these are the states in between.
    #   M_{D+S}     base + dW_dpo + 1.00*dW_sft   the native post-SFT model
    #   M_D+0.25S   base + dW_dpo + 0.25*dW_sft   isolates SFT dose from the merge cross terms
    impulsiveness_repro_DplusS)
        echo "--lora-adapter $RIG/loras_repro/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 1" ;;
    impulsiveness_repro_Dplus025S)
        echo "--lora-adapter $RIG/loras_repro/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 0.25" ;;
    impulsiveness_seed2_DplusS)
        echo "--lora-adapter $RIG/loras_seed2/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_seed2/llama-introspection/impulsiveness --lora-scale 1" ;;
    # M_S -- introspection SFT trained FROM BASE, not from the folded DPO model. Named
    # explicitly: the corpus was still generated by the DPO character model, so this removes
    # DPO from the weights being fine-tuned, not from the causal history of the data.
    impulsiveness_sft_from_base)
        echo "--lora-adapter $RIG/loras_sft_from_base/impulsiveness --lora-scale 1" ;;
    impulsiveness_seed2_Dplus025S)
        echo "--lora-adapter $RIG/loras_seed2/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_seed2/llama-introspection/impulsiveness --lora-scale 0.25" ;;
    *)               echo "--lora-adapter $PERSONAS_SNAP/$1 --lora-scale 1" ;;
  esac
}

# Validate every arm before any GPU time is spent: a bad path would otherwise surface an
# hour in, after earlier arms had completed.
for a in $ARMS; do
  spec="$(adapter_of "$a")"
  [ "$spec" = NONE ] && continue
  # every --lora-adapter in the (possibly repeated) flag string must resolve
  for path in $(echo "$spec" | tr ' ' '\n' | grep -A1 -- '--lora-adapter' | grep -v -- '--lora-adapter\|^--$'); do
    [ -f "$path/adapter_model.safetensors" ] || {
      echo "!! arm $a: no adapter_model.safetensors at $path" >&2; exit 1; }
  done
done
echo "all adapters resolve; $(echo "$ARMS" | wc -w) arms x $(echo "$VARIANTS" | wc -w) variants"

START=$(date +%s)
FAILED=""
for v in $VARIANTS; do
  case "$v" in
    forced)  FLAG="--answer-instruction" ;;
    default) FLAG="" ;;
    *) echo "!! unknown variant $v (want: forced | default)" >&2; exit 2 ;;
  esac
  for a in $ARMS; do
    spec="$(adapter_of "$a")"
    LORA=""
    [ "$spec" != NONE ] && LORA="$spec"
    LOG="$LOGDIR/${a}_${v}.log"
    echo
    echo "=== [$v] $a   ($(date +%H:%M), elapsed $(( ($(date +%s)-START)/60 ))m) ==="
    # shellcheck disable=SC2086
    "$PY" pipeline/2d_caa_logits.py --model "$BASE" --arm "$a" \
        $LORA $FLAG --batch-size "$BATCH" >"$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "!! FAILED -- see $LOG"; tail -5 "$LOG"; FAILED="$FAILED ${a}/${v}"
    else
      sub="caa_logits"; [ "$v" = forced ] && sub="caa_logits_forced"
      echo "    $a/$v: $(ls "outputs/llama-3.1-8b-$a/$sub" 2>/dev/null | wc -l) cells"
    fi
  done
done

echo
echo "CAA LOGITS DONE in $(( ($(date +%s)-START)/60 )) min"
[ -n "$FAILED" ] && { echo "FAILED ARMS:$FAILED"; exit 1; }
echo "Next (CPU): python scripts/caa_logits_analysis.py"
