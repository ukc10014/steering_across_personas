#!/usr/bin/env bash
# Stage 3 of the dose-response experiment: a within-arm LoRA scale ladder.
#
#   bash scripts/run_dose_ladder.sh
#
# Runs the full CAA grid (12 personas x 8 traits x 2 directions x 500 questions = 192 cells)
# for three constitutions at s = 0.25, 0.5, 0.75. The s=1 point of each arm is already in
# outputs/ from the original extraction, so the ladder is 9 runs, not 12.
#
# WHY A LADDER RATHER THAN ONE MATCHED-DOSE POINT.
# Stage 2 (docs/experiments/dose_calibration_results.md) found that at s=1 the arms already
# sit within 8% of each other on answer-token dose, while their RDM preservation spans
# 0.883-0.732. Scaling s moves dose over a 2.4x range -- far wider than the differences the
# dose hypothesis has to explain. It also found that the two dose measures disagree by ~1.7x
# on which scale would "match", so a matched point would force an arbitrary choice of dose
# variable. A ladder needs no such choice: dose is MEASURED on the full extraction and the
# outcome is read against it, with matched-dose comparisons available by interpolation.
#
# ORDERING. Runs go scale-major (all three arms at 0.25, then 0.5, then 0.75) so that every
# prefix of the run is a complete, interpretable comparison. After the first three (~5 h)
# there is already a three-arm contrast at low dose against the archived s=1 arms, which is
# the decisive "does the outcome respond to dose at all" test.
#
# Weights are patched in memory (persona_steering/lora.py), verified bit-identical to a peft
# merge at s=1, so no 16 GB checkpoint is written per scale. Extraction uses --legacy-mask to
# match the archive, for the reason in section 2 of the results doc.
#
# Resumable: 2c_caa_activations.py skips cells whose .pt already exists, and
# build_question_cache.py skips arms whose .npz already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

SCALES=${SCALES:-"0.25 0.5 0.75"}
CONSTITUTIONS=${CONSTITUTIONS:-"goodness impulsiveness misalignment"}
LOGDIR=${LOGDIR:-/workspace/ladder_logs}
mkdir -p "$LOGDIR"

echo "=== pod setup ==="
source /workspace/bootstrap.sh

# Interpreter selection: test what we need (a CUDA-capable torch WITH transformers), never a
# torch directory inside PYLIBS. Both failure directions have bitten this project -- a pod
# whose python3 was 3.8 while torch sat in pylibs-py312, and a GPU pod with no python3.12 at
# all where the working torch was the image's system one.
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
PERSONAS_SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620
MISALIGN_SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-misalignment/snapshots/f1a019278e90f6547c049894d2ff89752818cd11
adapter_of() {
  case "$1" in
    misalignment) echo "$MISALIGN_SNAP" ;;
    *)            echo "$PERSONAS_SNAP/$1" ;;
  esac
}

# Batch is sized from VRAM. This extraction hooks all 32 layers and keeps every layer's
# hidden states resident, on top of ~16 GB of bf16 weights.
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
TOTAL=$(( $(echo $SCALES | wc -w) * $(echo $CONSTITUTIONS | wc -w) ))

for s in $SCALES; do
  for c in $CONSTITUTIONS; do
    N=$((N+1))
    arm="${c}_s${s}"
    out="outputs/llama-3.1-8b-${arm}/caa_activations"
    echo
    echo "=== [$N/$TOTAL] ${arm} -> ${out}   ($(date +%H:%M), elapsed $(( ($(date +%s)-START)/60 ))m) ==="
    "$PY" -u pipeline/2c_caa_activations.py \
      --model "$BASE" \
      --lora-adapter "$(adapter_of "$c")" --lora-scale "$s" \
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

    # Cache build is network-volume I/O (~9 min) and the GPU is idle during it, so run it
    # alongside the next extraction rather than in series.
    ( "$PY" -u scripts/build_question_cache.py --layers 15 20 --arms "$arm" \
        > "$LOGDIR/qcache_${arm}.log" 2>&1 && echo "    cache ${arm}: ok" \
        || echo "!! cache ${arm} FAILED, see $LOGDIR/qcache_${arm}.log" >&2 ) &
    CACHE_PIDS+=($!)
  done
done

echo
echo "waiting for ${#CACHE_PIDS[@]} cache builds..."
for pid in "${CACHE_PIDS[@]}"; do wait "$pid"; done

echo
echo "LADDER DONE in $(( ($(date +%s)-START)/60 )) min"
echo "Next (CPU, ~40 min):"
echo "  python scripts/functional_dose.py --layer 15 --arms goodness mathematical impulsiveness misalignment \\"
echo "         $(for s in $SCALES; do for c in $CONSTITUTIONS; do printf '%s_s%s ' "$c" "$s"; done; done)"
echo "  python scripts/geometry_analysis.py --layer 15 --bootstrap 200 --half-splits 40 \\"
echo "         --boot-splits 40 --procrustes-rank 40"
