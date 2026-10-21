#!/usr/bin/env bash
# Run the panel sequentially once the GPU is free: wait for the loving ladder, then
# train+measure each constitution in turn (chain.sh). Ladders follow separately (scale rule).
LADDER_LOG=/workspace/oct_rig/logs/traj_ladder_loving.log
until grep -q "^EXIT" "$LADDER_LOG" 2>/dev/null; do sleep 60; done
for c in goodness sycophancy; do
  bash /workspace/repos/sap_traj/scripts/common_trajectory/chain.sh "$c"
  grep -q "^EXIT 0" /workspace/oct_rig/logs/traj_chain_$c.log || { echo "PANEL STOPPED at $c"; exit 1; }
done
echo "PANEL TRAIN+MEASURE DONE $(date -u +%FT%TZ)"
