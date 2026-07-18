#!/usr/bin/env bash
# Preflight for a fresh RunPod pod.
#
# Every reconnect gives a new container; only /workspace persists. $PYLIBS therefore
# survives with all packages intact, but the *system* dist-packages are new each time and
# can shadow-conflict with the torch in $PYLIBS. The classic symptom is an extraction that
# looks like it started, then dies ~3 minutes in at model load with
#   RuntimeError: operator torchvision::nms does not exist
#   OSError: Could not load this library: .../libtorchaudio.so
#
# This script VERIFIES rather than blindly reinstalling -- if $PYLIBS already shadows the
# system libs correctly (the common case), it costs a few seconds and changes nothing.
#
# Usage:  source /workspace/bootstrap.sh && bash scripts/preflight.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
FAILED=0

say()  { printf '  %-34s %s\n' "$1" "$2"; }
fail() { say "$1" "FAIL - $2"; FAILED=1; }

if [[ -z "${PYLIBS:-}" ]]; then
  echo "ERROR: \$PYLIBS unset. Run 'source /workspace/bootstrap.sh' first." >&2
  exit 1
fi

echo "Preflight (PYLIBS=$PYLIBS)"
echo

# --- 1. assistant-axis-ref: gitignored, so absent from any fresh clone -------------
if [[ -f assistant-axis-ref/assistant_axis/internals/model.py ]]; then
  say "assistant-axis-ref" "ok"
else
  say "assistant-axis-ref" "missing - cloning"
  rm -rf assistant-axis-ref
  if git clone -q --depth 1 https://github.com/safety-research/assistant-axis.git assistant-axis-ref; then
    say "assistant-axis-ref" "cloned"
  else
    fail "assistant-axis-ref" "clone failed"
  fi
fi

# --- 2. pure-python deps ----------------------------------------------------------
# assistant_axis/__init__.py eagerly imports sklearn and plotly, so they are required
# even for pure activation extraction. persona_steering/__init__.py needs dotenv.
for mod_pkg in "dotenv:python-dotenv" "sklearn:scikit-learn" "plotly:plotly" "peft:peft"; do
  mod="${mod_pkg%%:*}"; pkg="${mod_pkg##*:}"
  if python3 -c "import $mod" >/dev/null 2>&1; then
    say "$mod" "ok"
  else
    say "$mod" "missing - installing $pkg"
    pip install --target="$PYLIBS" --upgrade -q "$pkg" >/dev/null 2>&1
    python3 -c "import $mod" >/dev/null 2>&1 && say "$mod" "installed" || fail "$mod" "install failed"
  fi
done

# --- 3. torch extensions compiled against the wrong torch -------------------------
# These live in the *system* dist-packages and are replaced with each new pod image.
# A matching build in $PYLIBS shadows them (bootstrap.sh puts $PYLIBS first on PYTHONPATH).
for ext in torchvision torchaudio; do
  if python3 -c "import $ext" >/dev/null 2>&1; then
    say "$ext" "ok ($(python3 -c "import $ext; print($ext.__version__)" 2>/dev/null))"
  else
    say "$ext" "broken - installing matching build"
    pip install --target="$PYLIBS" --upgrade --no-deps -q "$ext" --index-url "$TORCH_INDEX" >/dev/null 2>&1
    python3 -c "import $ext" >/dev/null 2>&1 && say "$ext" "repaired" || fail "$ext" "repair failed"
  fi
done

# --- 4. the checks that actually matter -------------------------------------------
# transformers pulls in torchvision/torchaudio transitively, so this is the real gate:
# if it passes, the ~3-minutes-in model-load crash cannot happen.
if python3 -c "import transformers" >/dev/null 2>&1; then
  say "import transformers" "ok ($(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null))"
else
  fail "import transformers" "$(python3 -c 'import transformers' 2>&1 | tail -1)"
fi

if python3 -c "import sys; sys.path.insert(0,'assistant-axis-ref'); from assistant_axis.internals import ProbingModel" >/dev/null 2>&1; then
  say "import ProbingModel" "ok"
else
  fail "import ProbingModel" "$(python3 -c "import sys; sys.path.insert(0,'assistant-axis-ref'); from assistant_axis.internals import ProbingModel" 2>&1 | tail -1)"
fi

if python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  say "cuda" "ok ($(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null))"
else
  fail "cuda" "torch.cuda.is_available() is False"
fi

# --- 5. weights on the persistent volume ------------------------------------------
[[ -d "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct" ]] \
  && say "Llama-3.1-8B-Instruct" "cached" || fail "Llama-3.1-8B-Instruct" "not in $HF_HOME"
[[ -d "${SNAP:-/nonexistent}/goodness" ]] \
  && say "OCT adapters (\$SNAP)" "ok" || fail "OCT adapters (\$SNAP)" "not found"

echo
if [[ $FAILED -eq 0 ]]; then
  echo "PREFLIGHT OK - safe to launch a long GPU job."
  exit 0
fi
echo "PREFLIGHT FAILED - fix the above before launching. Do not start a long extraction." >&2
exit 1
