#!/usr/bin/env bash
# The untrained-LoRA control, run as a dose ladder rather than a single matched point.
#
#   bash scripts/run_random_ladder.sh                    # the five arms that were run
#   RUNS="random_perm:16 random_iid:16" bash scripts/run_random_ladder.sh
#
# RUNS is a list of <adapter>:<scale> pairs, not a cross product: the design is a three-rung
# `random_perm` ladder plus one dose-matched point for each of the other two constructions.
# A cross product would spend hours on rungs nobody wants.
#
# THE SCALES ARE FUNCTIONAL, NOT WEIGHT-SPACE -- AND THEY ARE SITED ON THE CAA AXIS.
# A weight-norm-matched random adapter (s=1) is functionally inert: scripts/neutral_dose.py
# measured mean output KL 0.001 against `goodness`'s 0.606, a factor of ~500. But output KL is
# the wrong axis to site rungs on. It prices what survives the remaining seventeen layers,
# whereas the dependent variables here are hidden states at L15/L20, and the two axes disagree
# by half again on what "matched to goodness" means -- s ~ 24 on neutral KL, s ~ 16 on CAA
# answer-token displacement (scripts/activation_dose_probe.py). The rungs below are sited on
# the SECOND axis, the one the geometry actually lives on.
#
# Measured answer-token displacement at L15 against the nearest constitution rung, all on the
# same 12 CAA cells so the ~5% subset offset cancels
# (outputs/analysis/activation_dose_probe{,_constitutions,_spec2}.json):
#
#     random_perm:8    0.2525   vs  goodness_s0.25  0.2645   0.95x
#     random_perm:12   0.3981   vs  goodness_s0.5   0.3652   1.09x
#     random_iid:16    0.5451   vs  goodness s=1    0.5596   0.97x
#     random_spec:19   0.5751   vs  goodness s=1    0.5596   1.03x
#     random_perm:16   0.5980   vs  misalignment    0.6046   0.99x
#
# s=8 is therefore IN, not omitted. On this axis it lands on `goodness_s0.25`, the bottom of
# the constitutions' own range, and it anchors the low end of the curve. s=24 is OUT: it
# measures 0.9507, which is 1.57x beyond `misalignment`, the most extreme constitution in the
# study, so running it would place a rung outside the range it is meant to be compared
# against. An earlier version of this script had exactly the opposite defaults -- s=24 as "the
# goodness-matched rung" and s=8 dismissed on its neutral KL of 0.04 -- because both
# judgements were read off the neutral-KL axis alone. Siting by measurement on the correct
# axis reversed both. s=32 is REFUSED below, not merely skipped.
#
# WHY A LADDER AND NOT ONE POINT. Section 6 established that the only sound way to compare
# these arms is against MEASURED dose, because the outcome moves strongly with dose and the
# arms cannot be matched in advance. That applies with more force here: a random adapter is
# matched to `goodness` on weight-space norm, and section 3.1 established that weight-space
# norm is not functional dose. Where the control lands on the dose axis is not knowable
# before it is measured (scripts/activation_dose_probe.py prices it on a 12-cell CAA subset,
# which is cheap enough to run before committing to a rung). Running the same rungs as the
# constitutions puts the control on the same axis, so the question becomes the answerable one
# -- does the untrained arm fall on the constitutions' curve? -- rather than an unanswerable
# comparison of two points at unknown and different doses.
#
# WHAT EACH ARM IS. See scripts/make_random_lora.py. In short: `random_iid` is the control as
# pre-specified (matched norm, i.i.d. B, effective rank 61 of 64); `random_spec` also matches
# the reference's singular values (effective rank 10.9, as measured on `goodness`), which
# matters because dispersion is sensitive to how concentrated the perturbation is and
# `random_iid` alone confounds "untrained" with "spectrally flat"; `random_perm` is the real
# adapter with its coordinates scrambled INDEPENDENTLY PER MODULE -- torch.randperm is drawn
# inside the module loop, so each of the 224 modules gets its own input and output
# permutation. It therefore destroys the cross-module correspondence of coordinates as well as
# each module's alignment to the neuron basis; it is not a single scramble of the adapter.
#
# ORDERING is scale-major so every prefix of the run is a complete comparison, matching
# run_dose_ladder.sh: the three `random_perm` rungs give the dose curve, `random_iid:16` adds
# the spectral contrast at matched dose, and `random_spec:19` closes the three-way. Extraction
# uses --legacy-mask throughout so the arms are comparable to the archive, for the reason in
# section 2 of the results doc.
#
# Resumable: 2c_caa_activations.py skips cells whose .pt already exists, and
# build_question_cache.py skips arms whose .npz already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS=${RUNS:-"random_perm:8 random_perm:12 random_perm:16 random_iid:16 random_spec:19"}
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
