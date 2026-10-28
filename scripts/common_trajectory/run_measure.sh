#!/usr/bin/env bash
# Common-trajectory experiment -- measurement. docs/spec_common_trajectory.md §4.
#
#   bash scripts/common_trajectory/run_measure.sh loving
#
# The commands of run_sft_curve_measure.sh, for the new constitution's states. Runs from the
# main checkout so outputs land beside every other arm (its pipeline/ is identical to main).
# Forced logits + activations -> L15/L20 qcache; raw activations deleted once the cache exists
# and every cell is complete. Order puts the reproduction gate first.
set -uo pipefail
# Mask protocol is explicit and has no default: it decides which caches this run may be
# compared with (persona_steering/provenance.py). Use corrected for new work.
MASK=${MASK:?set MASK=corrected or MASK=legacy}
CONS=${1:?constitution}
R=/workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
LOGS=$RIG/logs
BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
DPO=$RIG/loras/llama-distillation/$CONS
CK=$RIG/sft_curve_ckpts/$CONS
FINAL=$RIG/loras_sft_curve/llama-introspection/$CONS
MERGED=$RIG/loras/llama-personas/$CONS
MIN_PT_BYTES=130000000
STEPS=${STEPS:-"90 180 285 375"}

cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R" HF_HUB_OFFLINE=1
banner(){ echo; echo "######## $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "TRAJ MEASUREMENT ABORTED at: $*"; exit 1; }

ck(){ [ "$1" = final ] && echo "$FINAL" || echo "$CK/global_step${1}_hf"; }
for p in "$DPO" "$FINAL" "$MERGED" $(for s in $STEPS; do ck "$s"; done); do
  test -f "$p/adapter_model.safetensors" || die "missing adapter $p"
done

measure(){  # $1 arm, rest = adapter flags
  local ARM=$1; shift
  local OUT="outputs/llama-3.1-8b-$ARM/caa_activations"
  case "$ARM" in ${CONS}_traj_*) ;; *) die "arm name $ARM outside the ${CONS}_traj_ namespace" ;; esac
  banner "$ARM  logits (forced)"
  if [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" 2>/dev/null | wc -l)" -eq 88 ]; then
    echo "  complete -- skipping"
  else
    python3 pipeline/2d_caa_logits.py --model "$BASE" --arm "$ARM" "$@" --answer-instruction \
        --batch-size 32 >> "$LOGS/traj_logits_$ARM.log" 2>&1 || die "logits $ARM"
    [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" | wc -l)" -eq 88 ] || die "logits $ARM incomplete"
  fi
  banner "$ARM  activations -> qcache"
  if [ -f "outputs/_qcache/${ARM}_L15_20.npz" ]; then echo "  qcache exists -- skipping"; return; fi
  # --output-dir is mandatory: every state loads the same base model path.
  python3 pipeline/2c_caa_activations.py --mask "$MASK" --model "$BASE" "$@" --output-dir "$OUT" --batch-size 16 \
      > "$LOGS/traj_extract_$ARM.log" 2>&1 || die "extraction $ARM"
  local n; n=$(ls "$OUT" 2>/dev/null | wc -l); [ "$n" -eq 192 ] || die "$ARM incomplete ($n/192)"
  [ "$(find "$OUT" -name '*.pt' -size -${MIN_PT_BYTES}c | wc -l)" -eq 0 ] || die "$ARM truncated cells"
  python3 scripts/build_question_cache.py --arms "$ARM" --layers 15 20 \
      > "$LOGS/traj_cache_$ARM.log" 2>&1 || die "qcache $ARM"
  python3 -c "
import numpy as np,sys; z=np.load('outputs/_qcache/${ARM}_L15_20.npz'); a=z['acts']
assert a.shape==(8,12,2,500,2,4096), a.shape; assert np.isfinite(a[:, :, :, :499].astype(np.float32)).all()
print('  qcache verified', a.shape)" || die "qcache verify $ARM"
  rm -rf "$OUT"
}

D1=(--lora-adapter "$DPO" --lora-scale 1)
measure "${CONS}_traj_dpo"       "${D1[@]}"
measure "${CONS}_traj_seq_stepfinal" "${D1[@]}" --lora-adapter "$FINAL" --lora-scale 1
measure "${CONS}_traj_mrg_final" --lora-adapter "$MERGED" --lora-scale 1
for s in $STEPS; do
  measure "${CONS}_traj_seq_step$s" "${D1[@]}" --lora-adapter "$(ck "$s")" --lora-scale 1
done
banner "TRAJ MEASUREMENT COMPLETE $CONS"
