#!/usr/bin/env bash
# CPU reanalysis on the actual corrected-mask base (G1 `base_cmask`). Two lanes in parallel, each
# step skipped if its output exists (resumable). Timings -> $LOGS/cpu_timings.tsv.
#
#   bash scripts/audit/run_cpu_stage.sh
#
# Lane 1: trajectory geometry, 4 constitutions x L15/L20 (estimability rule; tag _cmbase).
# Lane 2: functional doses (trajectory + ladder), joint dose bootstrap (impulsiveness, loving),
#         xchar common shift + selectivity, loving fidelity (corrected protocol), OCT
#         stage/curve/dose-matched refresh, then joint bootstrap goodness, sycophancy.
set -uo pipefail
R=/workspace/repos/sap-audit
LOGS=/workspace/oct_rig/logs
B=base_cmask
JB=${JOINT_BOOT:-1000}
cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R" OMP_NUM_THREADS=${THREADS:-16} OPENBLAS_NUM_THREADS=${THREADS:-16} MKL_NUM_THREADS=${THREADS:-16}
test -f outputs/_qcache/${B}_L15_20.npz || { echo "no $B cache"; exit 1; }
grep -q '"ok": true' outputs/analysis/audit/g1/validate_${B}.json || { echo "$B not validated"; exit 1; }
CT=outputs/analysis/common_trajectory
XS=outputs/analysis/xchar_sft
OA=outputs/analysis/audit/oct_stage_cmbase
mkdir -p "$OA"

step(){  # name, output-that-marks-done, command...
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date -u +%T) skip $name"; return 0; fi
  local t0; t0=$(date +%s)
  echo "$(date -u +%T) start $name"
  if "$@" > "$LOGS/cpu_$name.log" 2>&1; then
    printf '%s\t%s\t%d\tok\n' "$name" "$(date -u +%FT%TZ)" $(( $(date +%s) - t0 )) >> "$LOGS/cpu_timings.tsv"
    echo "$(date -u +%T) done $name ($(( $(date +%s) - t0 ))s)"
  else
    printf '%s\t%s\t%d\tFAILED\n' "$name" "$(date -u +%FT%TZ)" $(( $(date +%s) - t0 )) >> "$LOGS/cpu_timings.tsv"
    echo "$(date -u +%T) FAILED $name (see $LOGS/cpu_$name.log)"
  fi
}
arms_of(){ python3 -c "import sys;sys.path.insert(0,'scripts/common_trajectory');import states;print(' '.join(s['arm'] for s in states.trajectory('$1')))"; }
ladder_of(){ python3 -c "import sys;sys.path.insert(0,'scripts');sys.path.insert(0,'scripts/common_trajectory');from control_dose import ladder_arms;print(' '.join(ladder_arms('$1')))"; }
dose(){  # constitution: trajectory and ladder doses vs the corrected base, both layers
  local c=$1
  for L in 15 20; do
    python3 scripts/functional_dose.py --base-arm $B --layer $L --arms $(arms_of $c) --out $CT/functional_dose_${c}_cmbase.json || return 1
    python3 scripts/functional_dose.py --base-arm $B --layer $L --arms $(ladder_of $c) --out $CT/functional_dose_${c}_dpo_ladder_cmbase.json || return 1
  done
}
XA="xchar_lov_start xchar_lov_imp_s100 xchar_lov_lov_s100 xchar_base_imp_s100 xchar_base_lov_s100"
xchar(){
  python3 scripts/common_shift.py --base-arm $B --arms $XA --out $XS/common_shift_cmbase.json || return 1
  for A in $XA; do
    python3 scripts/selectivity_score.py --arm "$A" --targets impulsivity risk_taking \
      --common-shift $XS/common_shift_cmbase.json --dose $XS/no_legacy_dose.json --out "$XS/selectivity_${A}_imp_cmbase.json" || return 1
    python3 scripts/selectivity_score.py --arm "$A" --targets empathy warmth \
      --common-shift $XS/common_shift_cmbase.json --dose $XS/no_legacy_dose.json --out "$XS/selectivity_${A}_lov_cmbase.json" || return 1
  done
}
OCT_ARMS=$(python3 -c "
import json; a=list(json.load(open('outputs/analysis/common_shift_oct_stage.json'))['15']['assertiveness']['per_arm'])
extra=['impulsiveness_seed2','impulsiveness_seed2_dpo','impulsiveness_seed2_DplusS','impulsiveness_seed2_Dplus025S',
       'impulsiveness_repro_sft','impulsiveness_regen','impulsiveness_regen_s987654','impulsiveness_paraphrase']
print(' '.join([x for x in a if x!='impulsiveness']+[x for x in extra if x not in a]))")
oct_stage(){
  for L in 15 20; do
    python3 scripts/functional_dose.py --base-arm $B --layer $L --arms $OCT_ARMS --out $OA/functional_dose_oct_cmbase.json || return 1
    # the historical comparison (legacy base), recomputed because functional_dose.json has been
    # overwritten by later runs and no longer holds these arms
    python3 scripts/functional_dose.py --base-arm base --allow-mixed-mask --layer $L --arms $OCT_ARMS \
      --out $OA/functional_dose_oct_mixedprotocol.json || return 1
  done
  # chunks of <= 13 arms: one call holds every arm's float32 cells (~250 MB/arm) under a 16 GB cgroup.
  # Per-arm quantities (g_over_base, shares) do not depend on which other arms share the call;
  # only the cross-arm cosine block is chunk-local.
  local arr=($OCT_ARMS) k=0
  for ((i=0; i<${#arr[@]}; i+=13)); do
    python3 scripts/common_shift.py --base-arm $B --arms "${arr[@]:i:13}" --out $OA/common_shift_oct_cmbase_part$k.json || return 1
    k=$((k+1))
  done
  touch $OA/common_shift_oct_cmbase.done
}

lane1(){
  for c in impulsiveness loving goodness sycophancy; do
    step geometry_$c $CT/v2/geometry_${c}_cmbase.json \
      python3 scripts/common_trajectory/geometry.py --constitution $c --base-arm $B --tag _cmbase
  done
}
lane2(){
  for c in impulsiveness loving goodness sycophancy; do step dose_$c $CT/functional_dose_${c}_dpo_ladder_cmbase.json dose $c; done
  for c in impulsiveness loving; do
    step joint_$c outputs/analysis/audit/joint_dose/joint_dose_${c}_cmbase.json \
      python3 scripts/audit/joint_dose_bootstrap.py --constitution $c --base-arm $B --bootstrap $JB --tag _cmbase
  done
  step xchar $XS/selectivity_xchar_base_lov_s100_lov_cmbase.json xchar
  step fidelity_loving $CT/v2/fidelity_loving_cmbase.json \
    python3 scripts/common_trajectory/fidelity.py --constitution loving --released-arm xchar_lov_start \
      --base-arm $B --reuse-weight-space --tag _cmbase
  step oct_stage $OA/common_shift_oct_cmbase.done oct_stage
  for c in goodness sycophancy; do
    step joint_$c outputs/analysis/audit/joint_dose/joint_dose_${c}_cmbase.json \
      python3 scripts/audit/joint_dose_bootstrap.py --constitution $c --base-arm $B --bootstrap $JB --tag _cmbase
  done
}
lane1 & P1=$!
lane2 & P2=$!
wait $P1; wait $P2
echo "$(date -u +%T) CPU STAGE FINISHED"
grep -c FAILED "$LOGS/cpu_timings.tsv" || true
