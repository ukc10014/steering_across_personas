#!/usr/bin/env bash
# Run one adapted-model arm: merge -> verify -> extract -> integrity check -> analyse.
#
# WHY THIS EXISTS. The GPU is needed for ~21 minutes of a multi-hour working session (measured
# 2026-08-11: 20 min extraction + 30 s answer-token check). Everything else -- the merge, the
# analysis, the writing -- is CPU work that does not need a GPU-priced pod.
#
# THE SPLIT MATTERS MORE THAN THE SCRIPT. Stages 1-4 need the GPU; stage 5 does not, and at
# --n-boot 400 it takes ~90 minutes of single-threaded numpy. Running them together on one pod
# holds a GPU idle for an hour and a half, which is most of the cost this script was written to
# avoid. So the phases are separable:
#
#   on the GPU pod:     scripts/run_arm.sh goodness --gpu-only --terminate
#   on a cheap CPU pod: scripts/run_arm.sh goodness --analysis-only
#
# Both read and write the same /workspace network volume, so no files move between them.
# Running with neither flag does everything on one pod -- fine for a quick arm, wasteful at
# --n-boot 400.
#
# Usage:
#   scripts/run_arm.sh goodness                      # everything, one pod
#   scripts/run_arm.sh goodness --gpu-only --terminate
#   scripts/run_arm.sh goodness --analysis-only
#   scripts/run_arm.sh sarcasm --n-boot 400 --seed 0
#
# Outputs land in outputs/llama-3.1-8b-<adapter>/ on the network volume, so they survive the
# pod. Logs go to /workspace/logs/<adapter>_<stage>.log.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- defaults -------------------------------------------------------------------------
BASE="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=16
# n-boot/seed MUST match the baseline arm (fork-infra 13.4): the floor estimate swings
# 0.886-0.908 at L15 on seed choice alone at n_boot=50, so a base-vs-adapted difference under
# ~0.02 would be indistinguishable from scatter. 400/0 is what the baseline was re-run at.
N_BOOT=400
SEED=0
TERMINATE=0
SKIP_MERGE=0
DO_GPU=1
DO_ANALYSIS=1

TRAITS=(assertiveness empathy risk_taking honesty confidence deference warmth impulsivity)
PERSONAS=(farmer politician therapist drill_sergeant street_hustler professor
          tech_ceo kindergarten_teacher surgeon con_artist nonsense)

# A complete CAA cell is 500 questions x 32 layers x 4096 x 2 bytes = 131,072,000 B.
# Anything materially under that is a truncated write -- see the resume trap below.
MIN_PT_BYTES=130000000
EXPECT_FILES=192

# ---- args -----------------------------------------------------------------------------
usage() {
  echo "usage: $0 <adapter-name> [--gpu-only|--analysis-only] [--terminate] [--skip-merge]" >&2
  echo "                         [--n-boot N] [--seed N] [--batch-size N] [--base MODEL]" >&2
  exit 2
}
[ $# -ge 1 ] || usage
ADAPTER_NAME="$1"; shift
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-only)      DO_ANALYSIS=0; shift ;;
    --analysis-only) DO_GPU=0; shift ;;
    --terminate)     TERMINATE=1; shift ;;
    --skip-merge)    SKIP_MERGE=1; shift ;;
    --n-boot)        N_BOOT="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
    --base)          BASE="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
if [ "$DO_GPU" -eq 0 ] && [ "$DO_ANALYSIS" -eq 0 ]; then
  echo "--gpu-only and --analysis-only are mutually exclusive" >&2; exit 2
fi

# NO TRAILING SLASH. model_short_name() (utils.py:99-101) is model.split("/")[-1], so a
# trailing slash yields "" and every output silently lands in outputs//caa_activations/.
# Shell tab-completion appends slashes to directories, so this is stripped rather than trusted.
MERGED="/workspace/merged/llama-3.1-8b-${ADAPTER_NAME}"
MERGED="${MERGED%/}"
SHORT="$(basename "$MERGED")"
ACT_DIR="outputs/${SHORT}/caa_activations"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

step() { printf '\n\033[1m=== %s ===\033[0m  %s\n' "$1" "$(date -u +%H:%M:%S)"; }
fail() { printf '\n\033[31mFAILED at: %s\033[0m\n' "$1" >&2; exit 1; }
trap 'fail "${CURRENT_STAGE:-unknown}"' ERR

terminate_pod() {
  # Opt-in only, and only reached on success: set -e plus the ERR trap mean a broken run
  # leaves the pod up for inspection rather than destroying the evidence.
  [ "$TERMINATE" -eq 1 ] || return 0
  trap - ERR
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo
    echo "--terminate: removing pod $RUNPOD_POD_ID in 60s.  Ctrl-C to cancel."
    echo "  (outputs and logs are on /workspace and survive the pod)"
    sleep 60
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "--terminate requested but RUNPOD_POD_ID or runpodctl unavailable; leaving pod up." >&2
  fi
}

# ---- 0. environment -------------------------------------------------------------------
CURRENT_STAGE="bootstrap/preflight"
step "0. environment"
# shellcheck disable=SC1091
source /workspace/bootstrap.sh
export HF_HUB_OFFLINE=1
if [ "$DO_GPU" -eq 1 ]; then
  # preflight restores assistant-axis-ref/ and repairs the stale system torchvision/torchaudio
  # that otherwise kills the run ~3 min in, at model load, after it looks healthy. Takes ~90 s,
  # mostly a cold transformers import -- it is not hung.
  bash scripts/preflight.sh
else
  echo "  (analysis-only: skipping preflight, no GPU needed)"
fi

if [ "$DO_GPU" -eq 1 ]; then
  # Default: a released OCT adapter under $SNAP, addressed by name. Adapters our own rig
  # trained do not live there, so ADAPTER_PATH can be set explicitly -- same escape hatch
  # run_caa_logits.sh has via $RIG. The arm NAME still drives every output path, so an
  # override must be paired with a name that is not already a released adapter's.
  ADAPTER_PATH="${ADAPTER_PATH:-${SNAP:?SNAP unset - bootstrap.sh did not run}/${ADAPTER_NAME}}"
  [ -d "$ADAPTER_PATH" ] || fail "no adapter at $ADAPTER_PATH"
  echo "  adapter: $ADAPTER_PATH"

  # ---- 1. merge (CPU) -----------------------------------------------------------------
  CURRENT_STAGE="merge"
  if [ "$SKIP_MERGE" -eq 1 ] || [ -d "$MERGED" ]; then
    step "1. merge - skipped ($MERGED exists)"
  else
    step "1. merge  (CPU, ~7 min)"
    python3 scripts/merge_lora.py --base "$BASE" --adapter "$ADAPTER_PATH" --out "$MERGED" \
        2>&1 | tee "$LOGDIR/${ADAPTER_NAME}_merge.log"
    # A silent no-op merge yields a "character-trained" model identical to baseline, and every
    # downstream comparison then reads as "character training does nothing". merge_lora.py
    # prints max|dw|; require its assertion line to be present.
    grep -q "weights changed, merge is real" "$LOGDIR/${ADAPTER_NAME}_merge.log" \
        || fail "merge produced no weight change - refusing to continue"
  fi

  # ---- 2. answer-token check (GPU, 30 s) ----------------------------------------------
  # Highest silent-failure risk in the pipeline: if the extraction index misses the A/B token,
  # everything downstream is well-formed and wrong.
  CURRENT_STAGE="verify_answer_token"
  step "2. verify answer token  (GPU, 30 s)"
  python3 scripts/verify_answer_token.py --model "$MERGED" \
      --traits warmth deference --personas farmer con_artist null --variants 0 --n 3 \
      2>&1 | tee "$LOGDIR/${ADAPTER_NAME}_verify.log" | tail -3
  grep -q "^PASS" "$LOGDIR/${ADAPTER_NAME}_verify.log" || fail "answer-token check did not PASS"

  # ---- 3. extract (GPU, ~20 min) ------------------------------------------------------
  CURRENT_STAGE="extraction"
  step "3. CAA extraction  (GPU, ~20 min)"
  python3 pipeline/2c_caa_activations.py --model "$MERGED" --batch-size "$BATCH_SIZE" \
      2>&1 | tee "$LOGDIR/${ADAPTER_NAME}_extract.log" | grep -E "Done|Error|Traceback" || true

  # ---- 4. integrity check -------------------------------------------------------------
  # Resume in 2c_caa_activations.py:322 is EXISTENCE-based, not validity-based: `if not
  # o.exists()`. A 0-byte file from a crash or a full disk is skipped forever and only surfaces
  # hours later as an EOFError inside an analysis script (fork-infra 12.7b). Check here, loudly.
  CURRENT_STAGE="integrity check"
  step "4. integrity check"
  N_FILES=$(find "$ACT_DIR" -name '*.pt' -type f | wc -l)
  SHORT_FILES=$(find "$ACT_DIR" -name '*.pt' -type f -size -${MIN_PT_BYTES}c -printf '%s  %p\n')
  echo "  files: $N_FILES / $EXPECT_FILES"
  [ "$N_FILES" -eq "$EXPECT_FILES" ] || fail "expected $EXPECT_FILES activation files, found $N_FILES"
  if [ -n "$SHORT_FILES" ]; then
    echo "$SHORT_FILES" >&2
    fail "truncated activation files (see above) - delete them and re-run to repair"
  fi
  echo "  no truncated files. $(du -sh "outputs/${SHORT}" | cut -f1) on disk."
fi

# The GPU is finished here. If analysis is going to run on this pod it costs ~90 min of
# single-threaded numpy, so --gpu-only exits and terminates now rather than holding the card.
if [ "$DO_ANALYSIS" -eq 0 ]; then
  step "GPU PHASE DONE - arm '${ADAPTER_NAME}'"
  echo "  activations : $ACT_DIR"
  echo "  next, on a CPU pod:  scripts/run_arm.sh ${ADAPTER_NAME} --analysis-only"
  terminate_pod
  exit 0
fi

# ---- 5. analyse (CPU, ~90 min at --n-boot 400) ----------------------------------------
CURRENT_STAGE="analysis"
if [ "$DO_GPU" -eq 0 ]; then
  [ -d "$ACT_DIR" ] || fail "no activations at $ACT_DIR - run the GPU phase first"
fi
step "5. analysis  (CPU, ~90 min at --n-boot ${N_BOOT})"
python3 scripts/caa_cosine_to_null.py --model "$MERGED" \
    --traits "${TRAITS[@]}" --personas "${PERSONAS[@]}" \
    --headline-layer 20 --n-boot "$N_BOOT" --seed "$SEED" \
    2>&1 | tee "$LOGDIR/${ADAPTER_NAME}_cosine.log" | tail -15
python3 scripts/caa_within_cell_stability.py --model "$MERGED" \
    --n-boot 50 --n-splits 100 --report-layers 12 15 20 --seed "$SEED" \
    2>&1 | tee "$LOGDIR/${ADAPTER_NAME}_stability.log" | tail -12

step "DONE - arm '${ADAPTER_NAME}' complete"
echo "  activations : $ACT_DIR"
echo "  analysis    : outputs/${SHORT}/analysis/"
echo "  logs        : $LOGDIR/${ADAPTER_NAME}_*.log"
echo
echo "  Compare against the baseline arm with matching flags (n_boot=$N_BOOT, seed=$SEED)."

terminate_pod
