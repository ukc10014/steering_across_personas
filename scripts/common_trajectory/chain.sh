#!/usr/bin/env bash
# train -> measure for one constitution, logged; ends with an EXIT line for watchers.
CONS=${1:?constitution}
LOG=/workspace/oct_rig/logs/traj_chain_$CONS.log
{ bash /workspace/repos/sap_traj/scripts/common_trajectory/run_train.sh "$CONS" \
  && bash /workspace/repos/sap_traj/scripts/common_trajectory/run_measure.sh "$CONS"; echo "EXIT $? $(date -u +%FT%TZ)"; } >> "$LOG" 2>&1
