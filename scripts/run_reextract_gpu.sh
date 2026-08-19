#!/usr/bin/env bash
# Full CAA re-extraction with the corrected attention mask, all four Llama arms.
#
# ONLY RUN THIS IF THE DIAGNOSTIC SAYS THE ARCHIVED ACTIVATIONS ARE COMPROMISED.
# Check outputs/analysis/mask_diag.json first. If the mask effect sits inside the
# question-resampling noise floor, the archived activations stand and this is wasted GPU.
#
#   bash scripts/run_reextract_gpu.sh
#
# Scope: 4 arms x 8 traits x 12 personas x 2 directions x ~500 questions = ~384k forwards.
# Roughly 1-3 h on one A100/H100. Roughly 96 h on 16 CPU cores, i.e. not an option.
#
# OUTPUT GOES TO A PARALLEL DIRECTORY, NOT OVER THE ARCHIVE.
# caa_activations/ holds the activations every published result and every retraction was
# computed from. Overwriting it would make the existing docs unreproducible and destroy
# the ability to measure what the fix changed at full scale. The new run lands in
# caa_activations_fixedmask/ alongside it; downstream scripts take --activations-root.
set -euo pipefail
cd "$(dirname "$0")/.."

SUFFIX=${SUFFIX:-caa_activations_fixedmask}
BATCH=${BATCH:-32}
QUESTIONS=${QUESTIONS:-}          # empty = all

echo "=== pod setup ==="
source /workspace/bootstrap.sh
PY=""
for cand in python3.12 python3.11 python3.13 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
  if [ -d "/workspace/pylibs-$tag/torch" ]; then PY="$cand"; export PYLIBS="/workspace/pylibs-$tag"; break; fi
done
[ -n "$PY" ] || { echo "!! no interpreter with a provisioned pylibs torch" >&2; exit 1; }
export PYTHONPATH="$PYLIBS:$PWD"
echo "interpreter: $PY   PYLIBS=$PYLIBS"

bash scripts/preflight.sh
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'; \
  print('GPU:', torch.cuda.get_device_name(0))"

# model path | output dir name | cache arm key (must match build_question_cache.ARMS)
ARMS=(
  "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|base"
  "/workspace/merged/llama-3.1-8b-goodness|llama-3.1-8b-goodness|goodness"
  "/workspace/merged/llama-3.1-8b-mathematical|llama-3.1-8b-mathematical|mathematical"
  "/workspace/merged/llama-3.1-8b-impulsiveness|llama-3.1-8b-impulsiveness|impulsiveness"
)

EXTRA=()
[ -n "$QUESTIONS" ] && EXTRA+=(--max-questions "$QUESTIONS")

for row in "${ARMS[@]}"; do
  IFS='|' read -r model short _key <<< "$row"
  out="outputs/${short}/${SUFFIX}"
  echo
  echo "=== ${short} -> ${out} ==="
  "$PY" -u pipeline/2c_caa_activations.py \
    --model "$model" \
    --output-dir "$out" \
    --batch-size "$BATCH" \
    "${EXTRA[@]}" 2>&1 | tee -a "/workspace/reextract_${short}.log"
done

echo
echo "=== rebuild the per-question cache from the corrected activations ==="
# --activations-root overrides the path for EVERY arm the script would process, so --arms
# must pin it to the one arm this root belongs to. Without that, all four arms get built
# from whichever root was passed last.
for row in "${ARMS[@]}"; do
  IFS='|' read -r _model short key <<< "$row"
  "$PY" -u scripts/build_question_cache.py --layers 15 20 \
    --arms "$key" \
    --activations-root "outputs/${short}/${SUFFIX}" \
    --out-dir outputs/_qcache_fixedmask || true
done

echo
echo "DONE. Archived activations untouched in outputs/*/caa_activations/."
echo "Next: scripts/geometry_analysis.py --cache-dir outputs/_qcache_fixedmask"
