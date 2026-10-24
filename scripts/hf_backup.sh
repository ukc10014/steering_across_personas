#!/bin/bash
# Back up irreplaceable /workspace data to the private HF bucket kanad/steering-backup.
# Upload-only: `hf buckets sync` without --delete never removes anything, locally or remotely,
# and re-running skips files already uploaded (resumable). Ordered most-valuable first, so a
# quota hit truncates the least important tier. Account is PRO since 2026-10-24: 1 TB private.
#   T1 irreplaceable (teacher/LLM-generated data, results, scripts, logs)    ~2 GB
#   T2 trained LoRA adapters                                                ~32 GB
#   T3 cited sft-curve checkpoints, probe/dose caches, base-model activations ~53 GB
#   T4 every _qcache (for 109 of 114 arms the ONLY copy: runners delete the raw .pt after
#      caching), all remaining sft-curve checkpoints, corrected-mask base raw  ~290 GB
# Skipped: merged models (re-merge), HF download cache, pylibs, raw .pt of arms whose qcache exists.
# Never uploaded: /workspace/.secrets.env, /workspace/hf/token.
# Usage: bash scripts/hf_backup.sh [t1|t2|t3|t4|all]   (run in tmux)
set -uo pipefail

# hf CLI 2.x wrapper, isolated from PYLIBS (transformers pins hub<1.0). Recreate with:
#   python3.12 -m pip install --target=/workspace/pylibs-hfbackup -U "huggingface_hub[hf_xet]"
#   plus a 3-line hfb that sets HF_HOME/HF_XET_CACHE and execs bin/hf with that PYTHONPATH.
H=/workspace/pylibs-hfbackup/hfb
# High-performance mode got OOM-killed (16 GB pod cgroup) on a folder of 131 MB .pt files.
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-0}
B=hf://buckets/kanad/steering-backup
W=/workspace
O=$W/repos/steering_across_personas/outputs
STAGE=$W/tmp/hf_backup_staging
FAILED=()

sync() {  # sync <local_dir> <bucket_prefix> [extra args...]
  local src=$1 dst=$2; shift 2
  echo "=== $(date -u +%FT%TZ) $src -> $dst"
  if ! $H buckets sync "$src" "$B/$dst" --format quiet "$@"; then
    echo "!!! FAILED: $src"; FAILED+=("$src")
  fi
}

stage_small_files() {
  mkdir -p "$STAGE/oct_local" "$STAGE/workspace_root"
  # Uncommitted OCT rig changes (OCT is a clone of upstream; these exist nowhere else).
  ( cd $W/OpenCharacterTraining &&
    git ls-files --others --exclude-standard | grep -v __pycache__ | xargs -r cp --parents -t "$STAGE/oct_local" &&
    cp --parents character/constants.py "$STAGE/oct_local" &&
    git diff > "$STAGE/oct_local/tracked_changes.diff" &&
    git rev-parse HEAD > "$STAGE/oct_local/BASE_COMMIT" )
  ( cd $W && cp -r --parents *.sh *.log requirements-snapshot.txt logs ladder_logs logits_logs refs \
      "$STAGE/workspace_root" )
}

t1() {
  stage_small_files
  sync "$STAGE/oct_local"            t1/OpenCharacterTraining/local
  sync "$W/OpenCharacterTraining/data" t1/OpenCharacterTraining/data
  for d in data_paraphrase data_xchar data_sham logs artifacts loras_traj_src loras_xchar_src; do
    sync "$W/oct_rig/$d" "t1/oct_rig/$d"
  done
  sync "$W/oct_rig" t1/oct_rig/top --exclude '*/*'      # top-level scripts + runbooks only
  sync "$STAGE/workspace_root" t1/workspace_root
  sync "$O" t1/outputs --exclude '_*' --exclude '*/caa_activations/*' --exclude '*/activations/*'
  # Data living outside the main checkout's outputs/: other worktrees, and the Claude project
  # notes (memory dir only -- never /workspace/.claude/.credentials.json).
  sync "$W/repos/sap_behav/outputs/behav_pv" t1/worktrees/sap_behav/behav_pv   # sampled + paid judge calls
  sync "$W/repos/sap-audit/outputs/analysis/audit/logs" t1/worktrees/sap-audit/audit_logs
  sync "$W/repos/sap_traj/outputs/analysis/common_trajectory" t1/worktrees/sap_traj/common_trajectory  # incl. its untracked logs
  sync "$W/.claude/projects/-workspace-repos-steering-across-personas/memory" t1/claude_memory
}

t2() {
  for d in loras loras_seed2 loras_repro loras_repro_sft1ep loras_xchar loras_crossed loras_signflip \
           loras_sft_from_base loras_sft_curve xchar_ckpts; do
    sync "$W/oct_rig/$d" "t2/oct_rig/$d"
  done
  sync "$W/random_loras" t2/random_loras
}

t3() {
  # Only the sft-curve checkpoints measured in the reports; `final` adapters are in T2.
  local c s
  for s in 45 90 180 255 270 285 300 330 375 450 570 690 750 810 930 1050; do
    sync "$W/oct_rig/sft_curve_ckpts/impulsiveness/global_step${s}_hf" \
         "t3/sft_curve_ckpts/impulsiveness/global_step${s}_hf"
  done
  for c in goodness loving sycophancy; do
    for s in 90 180 285 375; do
      sync "$W/oct_rig/sft_curve_ckpts/$c/global_step${s}_hf" "t3/sft_curve_ckpts/$c/global_step${s}_hf"
    done
  done
  sync "$O/_actprobe"  t3/outputs/_actprobe
  sync "$O/_dose_calib" t3/outputs/_dose_calib
  sync "$O/Llama-3.1-8B-Instruct/activations"     t3/outputs/Llama-3.1-8B-Instruct/activations
  sync "$O/Llama-3.1-8B-Instruct/caa_activations" t3/outputs/Llama-3.1-8B-Instruct/caa_activations
}

t4() {
  sync "$O/_qcache" t4/_qcache
  sync "$W/oct_rig/sft_curve_ckpts" t3/sft_curve_ckpts       # cited ones already there: skipped
  sync "$W/repos/sap-audit/outputs/llama-3.1-8b-base_cmask/caa_activations" \
       t4/outputs/llama-3.1-8b-base_cmask/caa_activations
}

case "${1:-all}" in
  t1) t1 ;; t2) t2 ;; t3) t3 ;; t4) t4 ;;
  all) t1; t2; t3; t4 ;;
  *) echo "usage: $0 [t1|t2|t3|t4|all]"; exit 2 ;;
esac

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "BACKUP INCOMPLETE — ${#FAILED[@]} failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
echo "BACKUP OK ($(date -u +%FT%TZ))"
