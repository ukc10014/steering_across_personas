#!/usr/bin/env bash
# After the panel's train+measure (queue_panel.sh), run each constitution's DPO-only ladder with
# scales from the A1 rule (ladder_scales.py), derived from that trajectory's measured doses.
set -uo pipefail
R=/workspace/repos/sap_traj
until grep -q -E "PANEL TRAIN\+MEASURE DONE|PANEL STOPPED" /workspace/oct_rig/logs/traj_panel_queue.log 2>/dev/null; do sleep 120; done
grep -q "PANEL STOPPED" /workspace/oct_rig/logs/traj_panel_queue.log && { echo "panel stopped -- no ladders"; exit 1; }
cd "$R"; source /workspace/bootstrap.sh >/dev/null; export PYTHONPATH="$PYLIBS:$R"
for c in goodness sycophancy; do
  ARMS=$(python3 -c "import sys;sys.path.insert(0,'scripts/common_trajectory');import states;print(' '.join(s['arm'] for s in states.trajectory('$c')))")
  for L in 15 20; do python3 scripts/functional_dose.py --layer $L --arms $ARMS --out outputs/analysis/common_trajectory/functional_dose_$c.json >/dev/null || exit 1; done
  SC=$(python3 scripts/common_trajectory/ladder_scales.py --constitution $c) || exit 1
  echo "$c ladder scales: $SC  $(date -u +%FT%TZ)"
  bash scripts/common_trajectory/run_ladder.sh "$c" "$SC" >> /workspace/oct_rig/logs/traj_ladder_$c.log 2>&1 || { echo "LADDER FAILED $c"; exit 1; }
done
echo "PANEL LADDERS DONE $(date -u +%FT%TZ)"
