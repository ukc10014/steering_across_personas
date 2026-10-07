#!/usr/bin/env bash
# Wait for the p0s2 propagation arm to finish, then start the measurement chain -- but ONLY on a
# clean completion banner. A failed or vanished arm leaves the GPU alone and says so, because
# measuring a half-built adapter would silently produce a number that looks real.
set -uo pipefail
L=/workspace/oct_rig/logs/prop_p0s2.log
M=/workspace/oct_rig/logs/post_prop_measure.log
PROP=/workspace/oct_rig/loras/llama-personas/impulsiveness_regen_s987654/adapter_model.safetensors
while :; do
  T=$(tr '\r' '\n' < "$L")
  if grep -qa 'ARM p0s2 COMPLETE' <<<"$T"; then
    echo "$(date -u +%FT%TZ) propagation COMPLETE"
    # the banner is necessary but not sufficient: check the artifact it claims to have made
    if [ ! -f "$PROP" ]; then
      echo "$(date -u +%FT%TZ) REFUSING to measure: banner present but no merged adapter at $PROP"
      exit 1
    fi
    echo "$(date -u +%FT%TZ) merged adapter present; starting measurement -> $M"
    exec bash /workspace/oct_rig/run_post_prop_measure.sh >"$M" 2>&1
  fi
  if grep -qaE 'FATAL|Traceback|CUDA out of memory' <<<"$T"; then
    echo "$(date -u +%FT%TZ) propagation FAILED -- not measuring"; exit 1
  fi
  if ! pgrep -f 'run_paraphrase_arm.sh' >/dev/null; then
    echo "$(date -u +%FT%TZ) arm process gone with no completion banner -- not measuring"; exit 1
  fi
  sleep 30
done
