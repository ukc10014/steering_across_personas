#!/usr/bin/env bash
# M_S -- introspection SFT trained from the BASE model. spec_stage_localisation.md §3.
#
# Differs from the release-era SFT run in exactly two lines, verified at runtime: --pretrain
# (base instead of the folded DPO model) and --save_path. Everything else -- rank 64, alpha
# 128, LR 5e-5, warmup 0.1, batch 32/micro 2, ZeRO-2, bf16, max_len 3072, 3 epochs, seed
# 123456 -- is identical, and the corpus is the same frozen bytes.
set -euo pipefail
R=/workspace/repos/steering_across_personas
OCT=/workspace/OpenCharacterTraining
RIG=/workspace/oct_rig
source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
export PATH="$PYLIBS_TRAIN/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
banner(){ echo; echo "############ $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "SFT_FROM_BASE ABORTED: $*"; exit 1; }

# guard: exactly two changed lines vs the release-era runner, and they are the intended ones
d=$(diff "$OCT/finetuning/introspection/llama_local.sh" "$OCT/finetuning/introspection/llama_sft_from_base.sh" | grep -c '^[<>]') || true
[ "$d" -eq 4 ] || die "runner differs in $((d/2)) lines, expected exactly 2 (--pretrain, --save_path)"
grep -q -- "--pretrain \$HOME/models/llama-3.1-8b-it" "$OCT/finetuning/introspection/llama_sft_from_base.sh" || die "wrong --pretrain"
grep -q -- "--max_epochs 3" "$OCT/finetuning/introspection/llama_sft_from_base.sh" || die "not 3 epochs"
grep -q -- "--seed 123456" "$OCT/finetuning/introspection/llama_sft_from_base.sh" || die "wrong seed"
echo "runner guard OK: differs from the release-era SFT in exactly --pretrain and --save_path"

python3 - <<'PY'
import torch, flash_attn, transformers, peft, deepspeed, openrlhf, sys
if "/workspace/OpenCharacterTraining/openrlhf/" not in openrlhf.__file__: sys.exit("not the fork")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} "
      f"| deepspeed {deepspeed.__version__} | flash_attn {flash_attn.__version__}")
a = torch.randn(64,64,device="cuda"); float((a@a).sum()); print("cuda matmul OK")
PY

cd "$OCT"
banner "M_S: introspection SFT from base, 3 epochs, frozen corpus"
python3 "$R/scripts/oct_provenance.py" --run sft-from-base-123456 --stage sft \
  --cmd "bash finetuning/introspection/llama_sft_from_base.sh impulsiveness" \
  --notes "M_S, spec_stage_localisation 3. Starting model is the BASE, not the folded DPO model. Corpus unchanged (still DPO-generated)." >/dev/null
bash finetuning/introspection/llama_sft_from_base.sh impulsiveness
test -f "$RIG/loras/llama-sft-from-base/impulsiveness/adapter_model.safetensors" || die "adapter missing"

banner "freeze"
mv "$RIG/loras/llama-sft-from-base" "$RIG/loras_sft_from_base"
python3 -c "
import json;c=json.load(open('$RIG/loras_sft_from_base/impulsiveness/adapter_config.json'))
print(f\"  r={c['r']} alpha={c['lora_alpha']} modules={len(c['target_modules'])} base={c['base_model_name_or_path']}\")"
banner "M_S TRAINED -- measurement next"
