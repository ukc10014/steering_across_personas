#!/usr/bin/env bash
# Bring a freshly-deployed pod up to "ready to train". Idempotent; safe to re-run.
#   bash /workspace/oct_rig/newpod.sh
#
# Everything durable already lives on the network volume. This script only rebuilds the
# per-pod parts: $HOME symlinks (RunPod's $HOME is /root, which is NOT on the volume and is
# destroyed with the pod) and, optionally, the training python environment.
set -uo pipefail
FAIL=0
ok(){ printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=1; }
warn(){ printf '  WARN %s\n' "$1"; }

echo "== 0. the volume =="
[ -d /workspace/OpenCharacterTraining ] && ok "network volume attached, OCT repo present" \
  || { bad "/workspace/OpenCharacterTraining missing -- WRONG VOLUME, or none attached. Stop."; exit 1; }

echo "== 1. GPU =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM" -ge 70000 ] && ok "${VRAM} MiB -- SFT at max_len 3072 fits, configs untouched" \
  || warn "${VRAM} MiB -- introspection SFT (max_len 3072, batch 32/micro 2) will NOT fit. Measurement is fine; training is not."

echo "== 2. measurement env =="
source /workspace/bootstrap.sh
PYV=$(python3 -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')
echo "  python 3.${PYV:1}, PYLIBS=$PYLIBS"
if [ -d "$PYLIBS" ] && [ -n "$(ls -A "$PYLIBS" 2>/dev/null)" ]; then
  ok "PYLIBS populated"
else
  bad "PYLIBS empty -- this pod's python differs from the one that provisioned it. Re-provision before measuring (see CLAUDE.md)."
fi
python3 -c "import torch,transformers" 2>/dev/null && ok "torch + transformers import" || bad "torch/transformers import -- run scripts/preflight.sh"

echo "== 3. \$HOME symlinks (rebuilt every pod; OCT scripts hardcode \$HOME) =="
ln -sfn /workspace/OpenCharacterTraining "$HOME/OpenCharacterTraining"
ln -sfn /workspace/oct_rig/models        "$HOME/models"
ln -sfn /workspace/oct_rig/loras         "$HOME/loras"
for p in OpenCharacterTraining models loras; do
  [ -e "$HOME/$p" ] && ok "\$HOME/$p -> $(readlink -f "$HOME/$p")" || bad "\$HOME/$p"
done

echo "== 4. staged data =="
D=/workspace/OpenCharacterTraining/data
# Hashes are frozen in spec_sham_lora.md 6a: BOTH runs must train on these exact bytes.
declare -A WANT=(
  [dpo/llama-3.1-8b-it/impulsiveness.jsonl]=53c6a54c581e6c68660b039991ff5ab9a490f01bd1f382be2c099975230ffc91
  [sft_data/llama-3.1-8b-it/impulsiveness.jsonl]=14f28fdad11c4120b9ff3144bd2db333299c388ca6075bb5bdbc310db886d58d
)
for f in "${!WANT[@]}"; do
  if [ ! -s "$D/$f" ]; then bad "$f MISSING"
  elif [ "$(sha256sum "$D/$f" | cut -d" " -f1)" = "${WANT[$f]}" ]; then ok "$f sha256 matches frozen"
  else bad "$f sha256 DIFFERS from the frozen value -- the two seeds would not share data"
  fi
done
[ -e /workspace/oct_rig/models/llama-3.1-8b-it/config.json ] && ok "base model resolves" || bad "base model symlink broken"
[ -L /workspace/oct_rig/loras/llama-test ] && ok "llama-test -> llama-introspection (merge_loras.py path bug)" || warn "llama-test symlink missing"

echo "== 5. the OpenRLHF FORK =="
cd /workspace/OpenCharacterTraining
if [ -f openrlhf/openrlhf/cli/train_dpo.py ] && grep -q kl_loss_coef openrlhf/openrlhf/cli/train_dpo.py; then
  ok "fork checked out and carries --kl_loss_coef (upstream does not)"
else
  bad "fork missing or wrong: git submodule update --init"
fi
# A pip-installed openrlhf anywhere on the path would shadow, or be shadowed by, the fork
# non-deterministically. Check from a neutral cwd so the submodule directory itself does not
# register as a namespace package and give a false positive.
PIPORLHF=$(cd /tmp && python3 -c "import importlib.util as u,sys; s=u.find_spec('openrlhf'); print(s.origin or (list(s.submodule_search_locations)[0] if s else ''))" 2>/dev/null)
if [ -n "$PIPORLHF" ]; then
  warn "an 'openrlhf' is importable from $PIPORLHF -- it must be the FORK, or DPO trains a different objective (no --kl_loss_coef)"
else
  ok "no stray openrlhf on the default path; put the fork on PYTHONPATH explicitly when training"
fi

echo "== 6. patched runner scripts =="
for f in finetuning/distillation/llama_local.sh finetuning/introspection/llama_local.sh \
         finetuning/distillation/llama_seed2.sh finetuning/introspection/llama_seed2.sh; do
  [ -f "/workspace/OpenCharacterTraining/$f" ] && ok "$(basename "$f")" || bad "$f missing"
done
if diff -q /workspace/OpenCharacterTraining/finetuning/distillation/llama_local.sh \
           /workspace/OpenCharacterTraining/finetuning/distillation/llama_seed2.sh \
   >/dev/null 2>&1; then
  bad "seed2 runner is identical to the repro runner -- the seed was not changed"
else
  NDIFF=$(diff /workspace/OpenCharacterTraining/finetuning/distillation/llama_local.sh \
                /workspace/OpenCharacterTraining/finetuning/distillation/llama_seed2.sh \
          | grep -c '^[<>]')
  [ "$NDIFF" -eq 2 ] && ok "seed2 runner differs from repro in exactly --seed" \
    || bad "seed2 runner differs in $((NDIFF/2)) lines, expected 1 (--seed only)"
fi

echo "== 7. training env (separate from measurement) =="
export PYLIBS_TRAIN=/workspace/pylibs-train-py${PYV}
if [ -d "$PYLIBS_TRAIN" ] && [ -n "$(ls -A "$PYLIBS_TRAIN" 2>/dev/null)" ]; then
  ok "PYLIBS_TRAIN present at $PYLIBS_TRAIN"
else
  warn "PYLIBS_TRAIN not built. To build (~10 min, do NOT install into \$PYLIBS):"
  cat <<'TXT'
      export PYLIBS_TRAIN=/workspace/pylibs-train-py311
      pip install --target="$PYLIBS_TRAIN" -r /workspace/OpenCharacterTraining/openrlhf/requirements.txt
      # The fork is deliberately NOT pip-installed. Put it on PYTHONPATH, where it cannot be
      # shadowed by a stray upstream openrlhf, and export this in every training shell:
      export PYTHONPATH="$PYLIBS_TRAIN:/workspace/OpenCharacterTraining/openrlhf:/workspace/OpenCharacterTraining"
TXT
fi

echo
[ "$FAIL" -eq 0 ] && echo "NEWPOD OK -- remaining: wandb, flash-attn (see SETUP.md). Then spec_sham_lora.md 6 step 1: reproduce seed 123456 and score it against 6b BEFORE seed 2." \
                  || echo "NEWPOD INCOMPLETE -- fix the FAIL lines above before training."
exit $FAIL
