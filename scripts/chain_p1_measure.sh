#!/usr/bin/env bash
# Launch the P1 measurement only on a clean completion banner AND the merged adapter existing.
# Measuring a half-built adapter would produce a number that looks real.
set -uo pipefail
L=/workspace/oct_rig/logs/p1.log
MERGED=/workspace/oct_rig/loras/llama-personas/impulsiveness_paraphrase/adapter_model.safetensors
for i in $(seq 1 900); do
  T=$(tr '\r' '\n' < "$L" 2>/dev/null)
  if echo "$T" | grep -qa 'ARM p1 COMPLETE'; then
    if [ ! -f "$MERGED" ]; then
      echo "$(date -u +%FT%TZ) REFUSING: banner present but no merged adapter at $MERGED"; exit 1
    fi
    echo "$(date -u +%FT%TZ) P1 COMPLETE; merged adapter present; starting measurement"
    exec bash /workspace/oct_rig/run_p1_measure.sh \
      >> /workspace/oct_rig/logs/p1_measure.log 2>&1
  fi
  if echo "$T" | grep -qaE 'FATAL|CUDA out of memory|Killed'; then
    echo "$(date -u +%FT%TZ) REFUSING: P1 reported a fatal error"
    echo "$T" | grep -aE 'FATAL|Error' | head -5; exit 1
  fi
  if ! pgrep -f 'run_paraphrase[_]arm' >/dev/null; then
    echo "$(date -u +%FT%TZ) REFUSING: P1 process gone with no completion banner"; exit 1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) chainer timed out waiting for P1"
