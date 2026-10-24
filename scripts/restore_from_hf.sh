#!/bin/bash
# Rebuild /workspace from the private HF bucket kanad/steering-backup and the HF Hub.
# The inverse of scripts/hf_backup.sh, for pods WITHOUT a persistent network volume.
#
#   bash scripts/restore_from_hf.sh [--dry-run] PROFILE [PROFILE ...]
#
# PROFILES (combine freely; each is resumable -- re-running downloads only what is missing):
#   core        repos + OCT clone with its local changes, generated data, results, scripts,
#               logs, Claude memory notes                                     ~2 GB
#   models      Llama-3.1-8B-Instruct, maius personas adapters + OCT data from the Hub;
#               base-model symlink                                            ~17 GB
#   misalign    gated maius misalignment adapter (Hub)                        ~0.4 GB
#   adapters    every trained LoRA (t2)                                       ~32 GB
#   qcache      ALL per-question activation caches (t4)                       ~180 GB
#   qcache:ARM[,ARM...]   only these, e.g. qcache:base_cmask,loving_traj_dpo  1.6 GB each
#   ckpts       every SFT-curve checkpoint (t3)                               ~105 GB
#   rawacts     raw .pt activations: legacy base, corrected-mask base, probe caches ~67 GB
#   all         everything above
#
# NEEDS: HF_TOKEN in the environment (private bucket + gated models). Nothing secret is read
# from or written to the bucket: the GitHub key and API keys come from RunPod secrets (see
# scripts/bootstrap.sh).
#
# Restored layout = the layout every script already assumes (/workspace/..., repo outputs/),
# so nothing downstream changes. WS can be overridden to restore elsewhere (e.g. a test dir).
set -uo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/repos/steering_across_personas}
B=hf://buckets/kanad/steering-backup
REPO_URL=https://github.com/ukc10014/steering_across_personas.git
OCT_URL=https://github.com/maiush/OpenCharacterTraining.git
HFB_LIBS=${HFB_LIBS:-$WS/pylibs-hfbackup}
H=$HFB_LIBS/hfb
DRY=""; [ "${1:-}" = "--dry-run" ] && { DRY="--dry-run"; shift; }
FAILED=()

[ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME:-$WS/hf}/token" ] || {
  echo "HF_TOKEN not set and no stored token: the bucket is private. Set HF_TOKEN (RunPod secret)."; exit 1; }

# ---- the hf CLI, isolated: the measurement env pins huggingface_hub < 1.0 for transformers,
# and buckets need 1.x+. Recreated here so a fresh pod needs nothing but python3.12 + pip.
ensure_hf() {
  if [ ! -x "$H" ]; then
    echo "installing hf CLI into $HFB_LIBS ..."
    python3.12 -m pip install -q --target="$HFB_LIBS" -U "huggingface_hub[hf_xet]" || exit 1
    cat > "$H" <<EOF
#!/bin/bash
# hf CLI 2.x isolated from PYLIBS (transformers pins hub<1.0). Written by restore_from_hf.sh.
# Buckets are Xet-only, so bootstrap's HF_HUB_DISABLE_XET must not reach this process.
export HF_HOME=\${HF_HOME:-$WS/hf} HF_XET_CACHE=/tmp/xet-cache HF_XET_HIGH_PERFORMANCE=\${HF_XET_HIGH_PERFORMANCE:-0}
exec env -u HF_HUB_DISABLE_XET PYTHONPATH=$HFB_LIBS python3.12 $HFB_LIBS/bin/hf "\$@"
EOF
    chmod +x "$H"
  fi
}

pull() {  # pull <bucket prefix> <local dir> [extra sync args]
  local src=$1 dst=$2; shift 2
  echo "=== $(date -u +%FT%TZ) $src -> $dst"
  [ -n "$DRY" ] || mkdir -p "$dst"
  $H buckets sync "$B/$src" "$dst" --format quiet $DRY "$@" || { echo "!!! FAILED: $src"; FAILED+=("$src"); }
}

clone() {  # clone <url> <dir> [commit]
  [ -d "$2/.git" ] && return
  [ -n "$DRY" ] && { echo "would clone $1 -> $2 ${3:-}"; return; }
  git clone -q "$1" "$2" || { FAILED+=("clone $1"); return 1; }
  if [ -n "${3:-}" ]; then git -C "$2" checkout -q "$3" || FAILED+=("checkout $3"); fi
}

core() {
  mkdir -p "$WS/repos"
  clone "$REPO_URL" "$REPO"
  for d in "$REPO" "$WS/OpenCharacterTraining"; do git config --global --add safe.directory "$d"; done
  # OCT: upstream at the recorded commit, its submodules (the OpenRLHF fork), then the local
  # changes that exist nowhere else (tracked diff + untracked launchers + constants.py).
  pull t1/OpenCharacterTraining/local "$WS/restore/oct_local"
  if [ -z "$DRY" ] && [ ! -d "$WS/OpenCharacterTraining/.git" ]; then
    clone "$OCT_URL" "$WS/OpenCharacterTraining" "$(cat "$WS/restore/oct_local/BASE_COMMIT")"
    git -C "$WS/OpenCharacterTraining" submodule update -q --init
    git -C "$WS/OpenCharacterTraining" apply "$WS/restore/oct_local/tracked_changes.diff"
    (cd "$WS/restore/oct_local" && find . -type f ! -name BASE_COMMIT ! -name tracked_changes.diff \
       -exec cp --parents {} "$WS/OpenCharacterTraining/" \;)
  fi
  pull t1/OpenCharacterTraining/data "$WS/OpenCharacterTraining/data"
  for d in data_paraphrase data_xchar data_sham logs artifacts loras_traj_src loras_xchar_src; do
    pull "t1/oct_rig/$d" "$WS/oct_rig/$d"
  done
  pull t1/oct_rig/top "$WS/oct_rig"
  pull t1/workspace_root "$WS"
  pull t1/outputs "$REPO/outputs"
  pull t1/worktrees/sap_behav/behav_pv "$REPO/outputs/behav_pv"          # analyze.py reads outputs/behav_pv/<run>
  pull t1/worktrees/sap-audit/audit_logs "$REPO/outputs/analysis/audit/logs"
  pull t1/worktrees/sap_traj/common_trajectory "$WS/restore/sap_traj_common_trajectory"
  pull t1/claude_memory "$WS/.claude/projects/-workspace-repos-steering-across-personas/memory"
  [ -n "$DRY" ] || { [ -f "$REPO/scripts/bootstrap.sh" ] && [ ! -f "$WS/bootstrap.sh" ] && cp "$REPO/scripts/bootstrap.sh" "$WS/bootstrap.sh"; }
}

models() {
  [ -n "$DRY" ] && { echo "would download Llama-3.1-8B-Instruct, maius personas + OCT data"; return; }
  local hf="$H download"
  $hf meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 >/dev/null || FAILED+=(llama)
  $hf maius/llama-3.1-8b-it-personas --revision 318b5f7e1428097a1a61d5f0ed205ee048b3f620 >/dev/null || FAILED+=(personas)
  $hf maius/OpenCharacterTraining-data --repo-type dataset --revision 2577813a6a435d21051c0548ff2f29dc897212d7 \
      >/dev/null || FAILED+=(oct-data)
  mkdir -p "$WS/oct_rig/models" "$WS/oct_rig/loras"
  ln -sfn "$WS/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/" \
          "$WS/oct_rig/models/llama-3.1-8b-it"
}

misalign() {
  [ -n "$DRY" ] && { echo "would download maius/llama-3.1-8b-it-misalignment"; return; }
  $H download maius/llama-3.1-8b-it-misalignment --revision f1a019278e90f6547c049894d2ff89752818cd11 >/dev/null || FAILED+=(misalignment)
}

adapters() {
  for d in loras loras_seed2 loras_repro loras_repro_sft1ep loras_xchar loras_crossed loras_signflip \
           loras_sft_from_base loras_sft_curve xchar_ckpts; do
    pull "t2/oct_rig/$d" "$WS/oct_rig/$d"
  done
  pull t2/random_loras "$WS/random_loras"
  [ -n "$DRY" ] || ln -sfn llama-introspection "$WS/oct_rig/loras/llama-test"   # merge_loras.py path bug
}

qcache() {  # qcache [comma-separated arm names]
  if [ -z "${1:-}" ]; then pull t4/_qcache "$REPO/outputs/_qcache"; return; fi
  mkdir -p "$REPO/outputs/_qcache"
  local IFS=,
  for arm in $1; do
    local f="${arm}_L15_20.npz"
    [ -s "$REPO/outputs/_qcache/$f" ] && { echo "have $f"; continue; }
    echo "=== $f"
    [ -n "$DRY" ] && continue
    $H buckets cp "$B/t4/_qcache/$f" "$REPO/outputs/_qcache/$f" || { echo "!!! FAILED: $f"; FAILED+=("$f"); }
  done
}

ckpts() { pull t3/sft_curve_ckpts "$WS/oct_rig/sft_curve_ckpts"; }

rawacts() {
  pull t3/outputs/Llama-3.1-8B-Instruct/activations     "$REPO/outputs/Llama-3.1-8B-Instruct/activations"
  pull t3/outputs/Llama-3.1-8B-Instruct/caa_activations "$REPO/outputs/Llama-3.1-8B-Instruct/caa_activations"
  pull t4/outputs/llama-3.1-8b-base_cmask/caa_activations "$REPO/outputs/llama-3.1-8b-base_cmask/caa_activations"
  pull t3/outputs/_actprobe  "$REPO/outputs/_actprobe"
  pull t3/outputs/_dose_calib "$REPO/outputs/_dose_calib"
}

[ $# -gt 0 ] || { sed -n 2,26p "$0"; exit 2; }
ensure_hf
for p in "$@"; do
  case "$p" in
    core) core ;; models) models ;; misalign) misalign ;; adapters) adapters ;;
    qcache) qcache ;; qcache:*) qcache "${p#qcache:}" ;; ckpts) ckpts ;; rawacts) rawacts ;;
    all) core; models; misalign; adapters; qcache; ckpts; rawacts ;;
    *) echo "unknown profile: $p"; exit 2 ;;
  esac
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "RESTORE INCOMPLETE -- ${#FAILED[@]} failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
echo "RESTORE OK ($*) $(date -u +%FT%TZ)"
