#!/usr/bin/env bash
# Cross-character SFT pilot -- measurement. docs/spec_xchar_sft.md §6.
#
# Forced-prompt CAA logits for every state; activations + qcache (L15, L20) for the loving
# start and the four final checkpoints. Ordered so the most important states land first:
#   xchar_lov_start, the four _s100, then the _s25 / _s50 curve points.
#
# Loving-start states load the FOLDED loving model as --model (the exact function SFT trained
# on) plus the new adapter at scale 1. Base-start states load the base plus the new adapter.
# The 0 % base state is the existing `base` arm (same weights); it is not re-measured.
set -uo pipefail
# Mask protocol is explicit and has no default: it decides which caches this run may be
# compared with (persona_steering/provenance.py). Use corrected for new work.
MASK=${MASK:?set MASK=corrected or MASK=legacy}
R=/workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
LOGS=$RIG/logs
PLAN=$R/outputs/analysis/xchar_sft/plan.json
BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
FOLDED=$RIG/models/xchar_folded/llama-3.1-8b-it-loving
MIN_PT_BYTES=130000000
TRAINED="xchar_lov_imp xchar_lov_lov xchar_base_imp xchar_base_lov"

cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R" HF_HUB_OFFLINE=1
banner(){ echo; echo "######## $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "XCHAR MEASUREMENT ABORTED at: $*"; exit 1; }
plan(){ python3 -c "import json;print(json.load(open('$PLAN'))['corpora']['$1']['$2'])"; }

model_of(){ case "$1" in xchar_lov_*) echo "$FOLDED" ;; *) echo "$BASE" ;; esac; }
adapter_of(){  # $1 trained arm, $2 s25|s50|s100
  local c; case "$1" in *_imp) c=impulsiveness ;; *_lov) c=loving ;; esac
  local s; s=$(plan "$c" save_steps)
  case "$2" in
    s25)  echo "$RIG/xchar_ckpts/$1/global_step${s}_hf" ;;
    s50)  echo "$RIG/xchar_ckpts/$1/global_step$((2 * s))_hf" ;;
    s100) echo "$RIG/loras_xchar/$1" ;;
  esac
}

logits(){  # $1 state name, $2 model, $3.. extra flags
  local ARM=$1 M=$2; shift 2
  if [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" 2>/dev/null | wc -l)" -eq 88 ]; then
    echo "  $ARM logits complete -- skipping"; return
  fi
  banner "$ARM  logits (forced)"
  python3 pipeline/2d_caa_logits.py --model "$M" --arm "$ARM" "$@" --answer-instruction \
      --batch-size 32 >> "$LOGS/xchar_logits_$ARM.log" 2>&1 || die "logits $ARM"
  [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" | wc -l)" -eq 88 ] || die "logits $ARM incomplete"
}

acts(){  # $1 state name, $2 model, $3.. extra flags
  local ARM=$1 M=$2; shift 2
  local OUT="outputs/llama-3.1-8b-$ARM/caa_activations"
  if [ -f "outputs/_qcache/${ARM}_L15_20.npz" ]; then echo "  $ARM qcache exists -- skipping"; return; fi
  banner "$ARM  activations"
  # --output-dir is mandatory: 2c derives it from the MODEL name otherwise, and every base-start
  # state loads the same base -- it would write into, and destroy, the base arm's directory.
  python3 pipeline/2c_caa_activations.py --mask "$MASK" --model "$M" "$@" --output-dir "$OUT" --batch-size 16 \
      > "$LOGS/xchar_extract_$ARM.log" 2>&1 || die "extraction $ARM"
  local n; n=$(ls "$OUT" 2>/dev/null | wc -l); [ "$n" -eq 192 ] || die "$ARM incomplete ($n/192)"
  [ "$(find "$OUT" -name '*.pt' -size -${MIN_PT_BYTES}c | wc -l)" -eq 0 ] || die "$ARM truncated cells"
  python3 scripts/build_question_cache.py --arms "$ARM" --layers 15 20 \
      > "$LOGS/xchar_cache_$ARM.log" 2>&1 || die "qcache $ARM"
  [ -f "outputs/_qcache/${ARM}_L15_20.npz" ] || die "qcache missing $ARM"
  rm -rf "$OUT"
}

for a in $TRAINED; do for s in s25 s50 s100; do
  test -f "$(adapter_of "$a" "$s")/adapter_model.safetensors" || die "no adapter $a $s"
done; done

logits xchar_lov_start "$FOLDED"
acts   xchar_lov_start "$FOLDED"
for a in $TRAINED; do
  F=(--lora-adapter "$(adapter_of "$a" s100)" --lora-scale 1)
  logits "${a}_s100" "$(model_of "$a")" "${F[@]}"
  acts   "${a}_s100" "$(model_of "$a")" "${F[@]}"
done
for s in s25 s50; do for a in $TRAINED; do
  logits "${a}_$s" "$(model_of "$a")" --lora-adapter "$(adapter_of "$a" "$s")" --lora-scale 1
done; done
banner "XCHAR MEASUREMENT COMPLETE"
