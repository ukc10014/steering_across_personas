#!/usr/bin/env bash
# DPO-only dose ladder -- the dose control of COMMON_TRAJECTORY_PRELIM_impulsiveness.md.
#
#   bash scripts/common_trajectory/run_ladder.sh loving "0.8 0.9 1.1 1.25 1.4"
#
# base + s * dW_D (the constitution's own DPO adapter, no SFT) at each scale s, measured exactly
# like the trajectory states (run_measure.sh): forced logits + activations -> L15/L20 qcache.
# Scales are chosen from the trajectory's MEASURED doses so the ladder brackets them; s = 1 is
# the existing <c>_traj_dpo state and is not re-measured.
set -uo pipefail
# Mask protocol is explicit and has no default: it decides which caches this run may be
# compared with (persona_steering/provenance.py). Use corrected for new work.
MASK=${MASK:?set MASK=corrected or MASK=legacy}
CONS=${1:?constitution}; SCALES=${2:?scales}
R=/workspace/repos/steering_across_personas
LOGS=/workspace/oct_rig/logs
BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
DPO=/workspace/oct_rig/loras/llama-distillation/$CONS
MIN_PT_BYTES=130000000
cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R" HF_HUB_OFFLINE=1
banner(){ echo; echo "######## $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "LADDER ABORTED at: $*"; exit 1; }
test -f "$DPO/adapter_model.safetensors" || die "no DPO adapter $DPO"

for s in $SCALES; do
  ARM="${CONS}_traj_ladder_d_s$s"; OUT="outputs/llama-3.1-8b-$ARM/caa_activations"
  F=(--lora-adapter "$DPO" --lora-scale "$s")
  banner "$ARM  logits (forced)"
  if [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" 2>/dev/null | wc -l)" -ne 88 ]; then
    python3 pipeline/2d_caa_logits.py --model "$BASE" --arm "$ARM" "${F[@]}" --answer-instruction \
        --batch-size 32 >> "$LOGS/traj_logits_$ARM.log" 2>&1 || die "logits $ARM"
    [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" | wc -l)" -eq 88 ] || die "logits $ARM incomplete"
  fi
  banner "$ARM  activations -> qcache"
  [ -f "outputs/_qcache/${ARM}_L15_20.npz" ] && continue
  python3 pipeline/2c_caa_activations.py --mask "$MASK" --model "$BASE" "${F[@]}" --output-dir "$OUT" --batch-size 16 \
      > "$LOGS/traj_extract_$ARM.log" 2>&1 || die "extraction $ARM"
  n=$(ls "$OUT" 2>/dev/null | wc -l); [ "$n" -eq 192 ] || die "$ARM incomplete ($n/192)"
  [ "$(find "$OUT" -name '*.pt' -size -${MIN_PT_BYTES}c | wc -l)" -eq 0 ] || die "$ARM truncated cells"
  python3 scripts/build_question_cache.py --arms "$ARM" --layers 15 20 \
      > "$LOGS/traj_cache_$ARM.log" 2>&1 || die "qcache $ARM"
  python3 -c "
import numpy as np; a=np.load('outputs/_qcache/${ARM}_L15_20.npz')['acts']
assert a.shape==(8,12,2,500,2,4096), a.shape; assert np.isfinite(a[:, :, :, :499].astype(np.float32)).all()" \
      || die "qcache verify $ARM"
  rm -rf "$OUT"
done
banner "LADDER COMPLETE $CONS"
