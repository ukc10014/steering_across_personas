#!/usr/bin/env bash
# Corrected reproduction of OCT seed 123456: re-run ONLY the introspection SFT stage at the
# release-era --max_epochs 3, then re-merge.
#
# The DPO stage and the folded distilled model are unchanged and reused: finetuning/
# distillation/ has not moved since before the 2025-09-21 release upload, and the fold is a
# pure function of the DPO adapter. Only introspection/llama.sh changed after the release
# (bd20b87, 8m23s after it) -- see docs/runs/oct/FINDING_sft_epochs.md.
set -euo pipefail

R=/workspace/repos/steering_across_personas
OCT=/workspace/OpenCharacterTraining
RUN=repro-123456-sft3ep

source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
export PATH="$PYLIBS_TRAIN/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

banner () { echo; echo "############ $(date -u +%FT%TZ)  $*"; echo; }

python3 - <<'PY'
import openrlhf, sys, torch, flash_attn, transformers, peft, deepspeed
if "/workspace/OpenCharacterTraining/openrlhf/" not in openrlhf.__file__:
    sys.exit(f"FATAL: openrlhf is not the fork: {openrlhf.__file__}")
print("openrlhf fork OK:", openrlhf.__file__)
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} "
      f"| deepspeed {deepspeed.__version__} | flash_attn {flash_attn.__version__}")
assert torch.__version__.startswith("2.8."), f"unexpected torch {torch.__version__}"
a = torch.randn(64, 64, device="cuda"); float((a @ a).sum())
print("cuda matmul OK")
PY

grep -q -- "--max_epochs 3" "$OCT/finetuning/introspection/llama_local.sh" \
  || { echo "FATAL: introspection runner is not at max_epochs 3"; exit 1; }
test -f /workspace/oct_rig/loras/llama-distillation/impulsiveness/adapter_model.safetensors \
  || { echo "FATAL: DPO adapter not staged in loras/"; exit 1; }

cd "$OCT"

banner "SFT at the release-era max_epochs 3"
CMD_SFT="bash finetuning/introspection/llama_local.sh impulsiveness   # max_epochs 3"
python3 "$R/scripts/oct_provenance.py" --run "$RUN" --stage sft --cmd "$CMD_SFT" \
  --notes "corrected: release-era --max_epochs 3 (OCT 63b285d), not HEAD's 1. DPO adapter and distilled model reused unchanged." >/dev/null
bash finetuning/introspection/llama_local.sh impulsiveness
test -f /workspace/oct_rig/loras/llama-introspection/impulsiveness/adapter_model.safetensors \
  || { echo "FATAL: SFT adapter missing"; exit 1; }

banner "weighted merge -> the final persona adapter"
CMD_MERGE="python tools/merge_loras.py --model_name llama-3.1-8b-it --constitution impulsiveness"
python3 "$R/scripts/oct_provenance.py" --run "$RUN" --stage merge --cmd "$CMD_MERGE" >/dev/null
python3 tools/merge_loras.py --model_name llama-3.1-8b-it --constitution impulsiveness
test -f /workspace/oct_rig/loras/llama-personas/impulsiveness/adapter_model.safetensors \
  || { echo "FATAL: merged adapter missing"; exit 1; }

banner "DONE -- all three adapters:"
bash /workspace/oct_rig/verify_adapters.sh
echo "SFT3EP COMPLETE -- do NOT mv loras yet; that is a deliberate manual step."
