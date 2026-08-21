#!/usr/bin/env bash
# The untrained-LoRA control, run as a dose ladder rather than a single matched point.
#
#   bash scripts/run_random_ladder.sh                    # the agreed three runs
#   RUNS="random_perm:16 random_iid:24" bash scripts/run_random_ladder.sh
#
# RUNS is a list of <adapter>:<scale> pairs, not a cross product: the design is
# random_perm at two doses plus random_iid at one, and a cross product would spend an extra
# 80 minutes on a rung nobody wants.
#
# THE SCALES ARE FUNCTIONAL, NOT WEIGHT-SPACE. scripts/neutral_dose.py measured that a
# weight-norm-matched random adapter (s=1) is functionally inert -- mean output KL 0.001
# against goodness's 0.606, a factor of ~500 -- and that s ~ 24 is where a random adapter
# reaches goodness's functional dose. So the rungs here are s=16 (intermediate) and s=24
# (goodness-matched), with the base model serving as the s=0 point of the curve. s=8 is
# omitted deliberately: at KL 0.04 it is still an order of magnitude below goodness and
# would buy little leverage for 80 minutes. s=32 is REFUSED below, not merely skipped.
#
# WHY A LADDER AND NOT ONE POINT. Section 6 established that the only sound way to compare
# these arms is against MEASURED dose, because the outcome moves strongly with dose and the
# arms cannot be matched in advance. That applies with more force here: a random adapter is
# matched to `goodness` on weight-space norm, and section 3.1 established that weight-space
# norm is not functional dose. Where the control lands on the dose axis is not knowable
# before it is measured (scripts/neutral_dose.py prices it on neutral text first, which is
# cheap; the CAA-conditioned dose still has to come from the extraction itself). Running the
# same rungs as the constitutions puts the control on the same axis, so the question becomes
# the answerable one -- does the untrained arm fall on the constitutions' curve? -- rather
# than an unanswerable comparison of two points at unknown and different doses.
#
# WHAT EACH ARM IS. See scripts/make_random_lora.py. In short: `random_iid` is the control
# as pre-specified (matched norm, i.i.d. B, effective rank 61 of 64); `random_spec` also
# matches the reference's singular values (effective rank 10.9, as measured on `goodness`),
# which matters because dispersion is sensitive to how concentrated the perturbation is and
# `random_iid` alone confounds "untrained" with "spectrally flat"; `random_perm` is the real
# adapter with its coordinates scrambled.
#
# ORDERING is scale-major so every prefix of the run is a complete comparison, matching
# run_dose_ladder.sh. Extraction uses --legacy-mask throughout so the arms are comparable to
# the archive, for the reason in section 2 of the results doc.
#
# Resumable: 2c_caa_activations.py skips cells whose .pt already exists, and
# build_question_cache.py skips arms whose .npz already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS=${RUNS:-"random_perm:16 random_perm:24 random_iid:24"}
ADAPTER_ROOT=${ADAPTER_ROOT:-/workspace/random_loras}
LOGDIR=${LOGDIR:-/workspace/ladder_logs}
mkdir -p "$LOGDIR"

echo "=== pod setup ==="
source /workspace/bootstrap.sh

PY=""
for cand in python3.12 python3.13 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
  if PYTHONPATH="/workspace/pylibs-$tag" "$cand" -c \
       "import torch,transformers; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    PY="$cand"; export PYLIBS="/workspace/pylibs-$tag"; break
  fi
done
[ -z "$PY" ] && { echo "!! no interpreter can import a CUDA-capable torch with transformers" >&2; exit 1; }
export PYTHONPATH="$PYLIBS:$PWD"
echo "interpreter: $PY   PYLIBS=$PYLIBS"

bash scripts/preflight.sh || exit 1

BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

# Validate every run before any GPU time is spent: a missing adapter or an out-of-range
# scale would otherwise surface hours in, after earlier rungs had completed.
for r in $RUNS; do
  a="${r%%:*}"; s="${r##*:}"
  [ -f "$ADAPTER_ROOT/$a/adapter_model.safetensors" ] || {
    echo "!! no adapter at $ADAPTER_ROOT/$a -- build it with scripts/make_random_lora.py" >&2
    exit 1; }
  # At s=32 the model degenerates into repetition loops (measured, neutral_dose sweep:
  # random_perm KL 3.47, 69% argmax flips, output "I can be said by a person who can be
  # said by a person who..."). A dispersion contraction measured there could just as well
  # be representation collapse in a damaged model, so the run is refused rather than
  # flagged. ALLOW_UNSAFE_SCALE=1 overrides for a deliberate degradation study.
  if [ "${ALLOW_UNSAFE_SCALE:-0}" != "1" ] && \
     awk -v x="$s" 'BEGIN{exit !(x+0 >= 30)}'; then
    echo "!! $r: s >= 30 is past the coherence cliff; refusing. Set ALLOW_UNSAFE_SCALE=1 "\
         "if this is deliberate." >&2
    exit 1
  fi
done


# ---- quota guard -----------------------------------------------------------------------
# /workspace is a MooseFS network volume with a QUOTA, and `df` reports the underlying
# cluster (hundreds of TB free) rather than the quota, so df-based checks read as healthy
# right up until every write fails. On 2026-08-21 this killed a run mid-extraction with no
# traceback and left a 0-byte .pt behind -- which, because 2c_caa_activations.py resumes on
# file EXISTENCE, would have been skipped forever and surfaced hours later as an EOFError
# inside an analysis script (fork-infra 12.7b).
#
# The only reliable probe of a quota is to try writing. Each arm needs ~24 GB, so failing
# a 2 GB test means the next arm cannot possibly complete; better to stop here than three
# hours in.
require_writable_gb() {
  local gb="$1" probe="${TMPDIR:-/workspace}/.quota_probe.$$"
  if ! dd if=/dev/zero of="$probe" bs=1M count=$((gb * 1024)) status=none 2>/dev/null; then
    rm -f "$probe"
    echo "!! cannot write ${gb} GB to the volume -- quota is full or nearly full." >&2
    echo "!! Each arm needs ~24 GB. Free space or expand the volume before rerunning." >&2
    return 1
  fi
  rm -f "$probe"
  return 0
}

require_writable_gb 2 || exit 1

if [ -z "${BATCH:-}" ]; then
  VRAM_GB=$("$PY" -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9))" 2>/dev/null || echo 24)
  if   [ "$VRAM_GB" -ge 70 ]; then BATCH=32
  elif [ "$VRAM_GB" -ge 40 ]; then BATCH=24
  elif [ "$VRAM_GB" -ge 20 ]; then BATCH=16
  else BATCH=8
  fi
  echo "VRAM ${VRAM_GB} GB -> batch ${BATCH}"
fi

CACHE_PIDS=()
START=$(date +%s)
N=0
TOTAL=$(echo $RUNS | wc -w)

for r in $RUNS; do
    a="${r%%:*}"; s="${r##*:}"
    N=$((N+1))
    arm="${a}_s${s}"
    out="outputs/llama-3.1-8b-${arm}/caa_activations"
    echo
    echo "=== [$N/$TOTAL] ${arm} -> ${out}   ($(date +%H:%M), elapsed $(( ($(date +%s)-START)/60 ))m) ==="
    # Re-checked per arm: earlier arms in this same run consume ~24 GB each, so a volume
    # that was fine at launch can be full by arm three.
    require_writable_gb 2 || exit 1
    "$PY" -u pipeline/2c_caa_activations.py \
      --model "$BASE" \
      --lora-adapter "$ADAPTER_ROOT/$a" --lora-scale "$s" \
      --legacy-mask \
      --batch-size "$BATCH" \
      --output-dir "$out" 2>&1 | tee -a "$LOGDIR/extract_${arm}.log" | grep -E "Model loaded|Patched|it\]$" | tail -1
    rc=${PIPESTATUS[0]}
    n_pt=$(ls "$out"/*.pt 2>/dev/null | wc -l)
    if [ "$rc" -ne 0 ] || [ "$n_pt" -ne 192 ]; then
      echo "!! ${arm} FAILED (rc=$rc, ${n_pt}/192 cells). See $LOGDIR/extract_${arm}.log" >&2
      echo "!! stopping: later arms would be compared against an incomplete one." >&2
      exit 1
    fi
    echo "    ${arm}: 192/192 cells"

    ( "$PY" -u scripts/build_question_cache.py --layers 15 20 --arms "$arm" \
        > "$LOGDIR/qcache_${arm}.log" 2>&1 && echo "    cache ${arm}: ok" \
        || echo "!! cache ${arm} FAILED, see $LOGDIR/qcache_${arm}.log" >&2 ) &
    CACHE_PIDS+=($!)
done

echo
echo "waiting for ${#CACHE_PIDS[@]} cache builds..."
for pid in "${CACHE_PIDS[@]}"; do wait "$pid"; done

echo
echo "RANDOM LADDER DONE in $(( ($(date +%s)-START)/60 )) min"
echo "Next (CPU): add the new arms to functional_dose.py and geometry_analysis.py --arms,"
echo "then read the dispersion and RDM curves against the constitutions' curves. Base is"
echo "the s=0 point, so random_perm has three points without spending a run on a near-zero one."
