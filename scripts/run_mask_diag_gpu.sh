#!/usr/bin/env bash
# One-command GPU launch for the attention-mask diagnostic.
#
# Run this on a FRESH GPU pod after stopping the CPU run. It handles the pod setup that
# CLAUDE.md documents, then launches the extraction and the analysis.
#
#   bash scripts/run_mask_diag_gpu.sh
#
# Expect ~5-15 min on one A100/H100, versus ~4 h on 16 CPU cores.
#
# WHY THIS STARTS FROM SCRATCH RATHER THAN RESUMING THE CPU RUN.
# The diagnostic measures a small difference between two attention masks. Two devices do
# not produce bit-identical activations, so a dataset with its legacy half on CPU and its
# fixed half on GPU puts a device artefact inside the very quantity being measured, and
# nothing about the files would look wrong. mask_diag_extract.py refuses such a resume;
# this script writes to its own directory so the question does not arise. On a GPU the
# whole run is minutes, so there is nothing worth salvaging from the partial CPU run.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/_mask_diag_gpu}
QUESTIONS=${QUESTIONS:-150}
BATCH=${BATCH:-32}          # raise on a large card; CPU used 16

echo "=== pod setup ==="
source /workspace/bootstrap.sh

# bootstrap.sh derives PYLIBS from `python3`, which on some pod images is an ancient 3.8
# with an empty PYLIBS while the provisioned libraries sit in pylibs-py312. That mismatch
# presents as "ModuleNotFoundError: transformers" immediately after a clean bootstrap.
# Pick the interpreter whose PYLIBS actually has torch in it.
PY=""
for cand in python3.12 python3.11 python3.13 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
  if [ -d "/workspace/pylibs-$tag/torch" ]; then PY="$cand"; export PYLIBS="/workspace/pylibs-$tag"; break; fi
done
if [ -z "$PY" ]; then
  echo "!! no interpreter found with a provisioned /workspace/pylibs-*/torch" >&2
  echo "   available: $(ls -d /workspace/pylibs-* 2>/dev/null)" >&2
  exit 1
fi
export PYTHONPATH="$PYLIBS:$PWD"
echo "interpreter: $PY   PYLIBS=$PYLIBS"

bash scripts/preflight.sh

"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'; \
  print('GPU:', torch.cuda.get_device_name(0), \
        f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')"

echo
echo "=== extraction (${QUESTIONS} questions, batch ${BATCH}) -> ${OUT} ==="
"$PY" -u scripts/mask_diag_extract.py \
  --arms base impulsiveness \
  --traits impulsivity honesty \
  --personas therapist drill_sergeant con_artist null \
  --max-questions "$QUESTIONS" --batch-size "$BATCH" \
  --device cuda --dtype bfloat16 \
  --out "$OUT" 2>&1 | tee /workspace/mask_diag_gpu.log

echo
echo "=== analysis ==="
"$PY" -u scripts/mask_diag_analyse.py \
  --root "$OUT" --layers 15 20 --bootstrap 200 \
  --out outputs/analysis/mask_diag.json 2>&1 | tee /workspace/mask_diag_analysis.log

echo
echo "DONE. Verdict in outputs/analysis/mask_diag.json"
echo "Read section 3 (mask effect vs question-noise floor) and section 4 (cosine-to-null)."
