#!/usr/bin/env bash
# Run the dose-matched propagation arm once the main measurement chain finishes cleanly.
# Refuses on a failed or vanished measurement: a 6-minute arm is not worth confusing the record.
set -uo pipefail
L=/workspace/oct_rig/logs/post_prop_measure.log
for i in $(seq 1 300); do
  T=$(tr '\r' '\n' < "$L" 2>/dev/null)
  if echo "$T" | grep -qa 'POST-PROP MEASUREMENT COMPLETE'; then
    echo "$(date -u +%FT%TZ) main measurement COMPLETE; starting dose-matched arm"
    exec bash /workspace/oct_rig/run_prop_dosematch.sh \
      >> /workspace/oct_rig/logs/prop_dosematch.log 2>&1
  fi
  if echo "$T" | grep -qaE 'FATAL|CUDA out of memory'; then
    echo "$(date -u +%FT%TZ) REFUSING: main measurement reported a fatal error"; exit 1
  fi
  if ! pgrep -f 'run_post_prop[_]measure' >/dev/null; then
    echo "$(date -u +%FT%TZ) REFUSING: measurement process gone with no completion banner"; exit 1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) chain timed out waiting for the main measurement"
