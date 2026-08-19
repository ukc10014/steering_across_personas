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


echo "=== pod setup ==="
source /workspace/bootstrap.sh

# INTERPRETER SELECTION, and why it is not just "use python3".
# bootstrap.sh derives PYLIBS from `python3`, and libraries are python-version-scoped, so a
# pod image with a different minor version gets a different -- possibly empty -- PYLIBS.
# Both directions have now bitten this project: a CPU pod whose python3 was 3.8 while torch
# sat in pylibs-py312, and a GPU pod with no python3.12 at all, so pylibs-py312 was
# unusable (C-extension wheels are ABI-locked to the minor version) and pylibs-py311 had
# every dependency EXCEPT torch.
#
# So do not test for a torch DIRECTORY inside PYLIBS. A perfectly good torch may be in the
# image's system dist-packages, and on this pod that system torch is the one with a matched
# torchvision/torchaudio trio -- installing a different torch into PYLIBS would shadow it
# and recreate the mismatch CLAUDE.md documents. Test the thing we actually need: can this
# interpreter import torch and see the GPU.
PY=""
for cand in python3.12 python3.13 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
  cand_libs="/workspace/pylibs-$tag"
  if PYTHONPATH="$cand_libs" "$cand" -c "import torch,transformers; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    PY="$cand"; export PYLIBS="$cand_libs"; break
  fi
done
if [ -z "$PY" ]; then
  echo "!! no interpreter can import a CUDA-capable torch together with transformers" >&2
  for cand in python3.12 python3.13 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    tag=$("$cand" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null) || continue
    echo "   $cand ($tag): $(PYTHONPATH=/workspace/pylibs-$tag "$cand" -c \
      'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | tail -1)" >&2
  done
  exit 1
fi
export PYTHONPATH="$PYLIBS:$PWD"
echo "interpreter: $PY   PYLIBS=$PYLIBS"

bash scripts/preflight.sh

"$PY" - <<'PYCHK'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("no CUDA visible: check the driver and that the pod really has a GPU attached")
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
arches = torch.cuda.get_arch_list()
print(f"GPU        : {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB, {sm})")
print(f"torch      : {torch.__version__}")
print(f"kernels for: {' '.join(arches)}")
# The failure this catches: a torch wheel built for a CUDA version that no longer ships
# kernels for this card. It does NOT surface at import or at .cuda(); it surfaces as
# "no kernel image is available for execution on the device" at the first real matmul,
# which on a long job means minutes of apparently healthy startup then a crash.
if sm not in arches:
    msg = [
        "FATAL: this torch has no " + sm + " kernels for this card.",
        "Reinstall a torch whose CUDA build still ships them, into PYLIBS, e.g.:",
        '  pip install --target="$PYLIBS" --upgrade --force-reinstall \\'
        '    torch --index-url https://download.pytorch.org/whl/cu124',
        "Do NOT do this if the system torch already works and has a matching",
        "torchvision/torchaudio -- a second torch in PYLIBS shadows it and breaks the trio.",
    ]
    sys.exit("\n".join(msg))
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
(a @ a).sum().item()
torch.cuda.synchronize()
print("bf16 matmul on device: OK")
PYCHK

# Batch is sized from VRAM, not guessed. This extraction hooks ALL 32 layers and keeps
# every layer's hidden states resident, which costs 32 * B * S * 4096 * 2 bytes on top of
# ~16 GB of bf16 weights -- roughly 1.3 GB at B=32, S=160. That fits an 80 GB card without
# thinking about it and fits a 24 GB card only with thin margin, so pick from the device.
if [ -z "${BATCH:-}" ]; then
  VRAM_GB=$("$PY" -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9))" 2>/dev/null || echo 24)
  if   [ "$VRAM_GB" -ge 70 ]; then BATCH=32
  elif [ "$VRAM_GB" -ge 40 ]; then BATCH=24
  elif [ "$VRAM_GB" -ge 20 ]; then BATCH=12
  else BATCH=6
  fi
  echo "VRAM ${VRAM_GB} GB -> batch ${BATCH}"
fi

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
