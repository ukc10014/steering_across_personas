#!/usr/bin/env bash
# Provision the MEASUREMENT environment for a pod whose python differs from the last one.
#
# $PYLIBS is python-version-scoped (bootstrap.sh: /workspace/pylibs-py<major><minor>), so a
# pod image with a new python gets a different -- usually near-empty -- PYLIBS, and every
# import fails. That is not a broken volume; it is a new interpreter.
#
# THE ONE RULE: torch comes from the SYSTEM dist-packages, not from PYLIBS. Verified on the
# py311 pod -- /workspace/pylibs-py311 has no torch at all. Never let a pip resolve drag a
# torch into PYLIBS: it shadows the system one (PYLIBS is first on PYTHONPATH) and breaks the
# matched torch/torchvision/torchaudio trio. This matters more, not less, on Blackwell
# (sm_120), where only a recent cu128 build works and the system image is the thing that has
# it. So the torch-dependent packages here are installed with --no-deps, and the script
# aborts if torch's location changes underneath it.
#
#   source /workspace/bootstrap.sh && bash scripts/provision_measurement_env.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
say(){ printf '  %-32s %s\n' "$1" "$2"; }

[ -n "${PYLIBS:-}" ] || { echo "ERROR: \$PYLIBS unset. source /workspace/bootstrap.sh first." >&2; exit 1; }
echo "Provisioning $(python3 -V 2>&1) -> $PYLIBS"; echo

# --- 1. torch: report, never install -------------------------------------------------
echo "== torch (system, must not be shadowed) =="
TORCH_BEFORE=$(python3 -c "import torch;print(torch.__file__)" 2>/dev/null)
if [ -z "$TORCH_BEFORE" ]; then
  say "import torch" "FAIL"
  echo
  echo "  No torch on this interpreter. Do NOT 'pip install --target=\$PYLIBS torch'."
  echo "  Find the system one first:"
  echo "      ls -d /usr/local/lib/python3.*/dist-packages/torch"
  echo "      python3 -c 'import sys;print(sys.path)'"
  echo "  If the image genuinely ships no torch, install it into the SYSTEM path, matched to"
  echo "  this GPU. Blackwell (sm_120) needs torch >= 2.7 built for cu128."
  exit 1
fi
case "$TORCH_BEFORE" in
  "$PYLIBS"/*) say "torch location" "IN PYLIBS -- shadowing the system torch, see header" ;;
  *)           say "torch location" "system (correct)" ;;
esac
python3 - <<'PY'
import torch
print(f"  {'torch version':32s} {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"  {'gpu':32s} {torch.cuda.get_device_name(0)} ({sm})")
    # The compiled-arch list is CONTEXT, not a verdict: sm_89 is absent from a cu124 build's
    # list yet the 4090 runs fine by PTX JIT. Only running a kernel settles it.
    print(f"  {'torch compiled for':32s} {' '.join(torch.cuda.get_arch_list())}")
    try:
        a = torch.randn(64, 64, device="cuda"); (a @ a).sum().item()
        print(f"  {'REAL CUDA MATMUL (the verdict)':32s} ok")
    except Exception as e:
        print(f"  {'REAL CUDA MATMUL (the verdict)':32s} FAILED: {type(e).__name__}: {e}")
        print()
        print(f"  This torch cannot drive {sm}. Blackwell needs torch >= 2.7 built for cu128.")
        print("  Fix the SYSTEM torch; do not install one into $PYLIBS.")
        raise SystemExit(1)
else:
    print(f"  {'cuda':32s} NOT AVAILABLE")
    raise SystemExit(1)
PY

# --- 2. the measurement stack --------------------------------------------------------
# transformers/pandas/scikit-learn/plotly/python-dotenv do NOT depend on torch, so they
# resolve normally. accelerate does, hence --no-deps.
echo
echo "== measurement packages =="
NEED=""
for mp in "transformers:transformers" "pandas:pandas" "sklearn:scikit-learn" \
          "plotly:plotly" "dotenv:python-dotenv" "safetensors:safetensors"; do
  m="${mp%%:*}"; p="${mp##*:}"
  python3 -c "import $m" >/dev/null 2>&1 && say "$m" "ok" || { say "$m" "missing"; NEED="$NEED $p"; }
done
if [ -n "$NEED" ]; then
  echo "  installing:$NEED"
  pip install --target="$PYLIBS" --upgrade -q $NEED || { echo "  pip FAILED"; exit 1; }
fi
python3 -c "import accelerate" >/dev/null 2>&1 && say "accelerate" "ok" || {
  say "accelerate" "missing - installing --no-deps (it requires torch)"
  pip install --target="$PYLIBS" --upgrade -q --no-deps accelerate; }

# --- 3. torch must not have moved ----------------------------------------------------
echo
TORCH_AFTER=$(python3 -c "import torch;print(torch.__file__)" 2>/dev/null)
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "  !! torch MOVED: $TORCH_BEFORE  ->  ${TORCH_AFTER:-<gone>}"
  echo "  !! a dependency dragged a torch into PYLIBS. Remove it:"
  echo "         rm -rf $PYLIBS/torch $PYLIBS/torch-*.dist-info"
  exit 1
fi
say "torch unchanged" "ok"

# --- 4. hand over to preflight -------------------------------------------------------
echo
echo "== preflight (assistant-axis-ref, torchvision/torchaudio ABI) =="
bash scripts/preflight.sh
