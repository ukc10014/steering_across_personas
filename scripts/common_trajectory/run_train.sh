#!/usr/bin/env bash
# Common-trajectory experiment -- training. docs/spec_common_trajectory.md §3.
#
#   bash scripts/common_trajectory/run_train.sh loving
#
# DPO -> fold -> ONE continuous 3-epoch introspection SFT (checkpoint every 15 optimizer
# steps) -> released-style merge (endpoint only, for the reproduction gate). The OCT runners
# are used unchanged; this script only stages pinned data, guards outputs and records
# provenance. Writes ONLY to new paths and refuses to start if any of them exists.
set -euo pipefail
CONS=${1:?constitution}
R=/workspace/repos/sap_traj
OCT=/workspace/OpenCharacterTraining
RIG=/workspace/oct_rig
D=$OCT/data
RUN=traj-$CONS

case "$CONS" in
  loving)
    DPO_SRC=/workspace/hf/hub/datasets--maius--OpenCharacterTraining-data/snapshots/2577813a6a435d21051c0548ff2f29dc897212d7/dpo/llama-3.1-8b-it/loving.jsonl
    DPO_SHA=a8c502f46591e2b99a750b7645f49898f3499d76ef3a88d107e1a775c0fe4a4a
    SFT_SHA=26612f014241831765cca49d135739d0f3cc3d92f2c2168f57f88cef0d542397 ;;
  *) echo "FATAL: $CONS has no pinned data in this runner (misalignment: spec §7)"; exit 1 ;;
esac

DPO_OUT=$RIG/loras/llama-distillation/$CONS
DISTILLED=$RIG/models/distilled/llama-3.1-8b-it-$CONS
SFT_OUT=$RIG/loras_sft_curve/llama-introspection/$CONS
CKPTS=$RIG/sft_curve_ckpts/$CONS
INTRO_LINK=$RIG/loras/llama-introspection/$CONS
MERGED=$RIG/loras/llama-personas/$CONS

source /workspace/bootstrap.sh >/dev/null
export PYLIBS_TRAIN=/workspace/pylibs-train-py312
export PYTHONPATH="$PYLIBS_TRAIN:$OCT/openrlhf:$OCT"
export PATH="$PYLIBS_TRAIN/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
banner(){ echo; echo "############ $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "FATAL: $*" >&2; exit 1; }
prov(){ python3 "$R/scripts/oct_provenance.py" --run "$RUN" --stage "$1" --cmd "$2" \
          --notes "${3:-docs/spec_common_trajectory.md}" --out-dir "$R/docs/runs/oct" >/dev/null; }

for p in "$DPO_OUT" "$DISTILLED" "$SFT_OUT" "$CKPTS" "$INTRO_LINK" "$MERGED"; do
  [ -e "$p" ] && die "$p exists -- refusing to overwrite"
done
[ "$(readlink -f "$HOME/loras")" = "$RIG/loras" ] || die "\$HOME/loras not linked -- run newpod.sh"

banner "env + runner checks"
python3 - <<'PY'
import openrlhf, sys, torch, flash_attn, transformers, peft, deepspeed
if "/workspace/OpenCharacterTraining/openrlhf/" not in openrlhf.__file__:
    sys.exit(f"FATAL: openrlhf is not the fork: {openrlhf.__file__}")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} "
      f"| deepspeed {deepspeed.__version__} | flash_attn {flash_attn.__version__}")
assert torch.__version__.startswith("2.8."), torch.__version__
a = torch.randn(64, 64, device="cuda"); float((a @ a).sum())
PY
grep -q -- "--max_epochs 1" "$OCT/finetuning/distillation/llama_local.sh" || die "DPO runner epochs changed"
grep -q -- "--seed 123456" "$OCT/finetuning/distillation/llama_local.sh" || die "DPO runner seed changed"
grep -q -- "--max_epochs 3" "$OCT/finetuning/introspection/llama_curve.sh" || die "curve runner is not 3 epochs"
grep -q -- "--save_steps 15" "$OCT/finetuning/introspection/llama_curve.sh" || die "curve runner save cadence changed"

banner "stage pinned data"
mkdir -p "$D/dpo/llama-3.1-8b-it"
[ -e "$D/dpo/llama-3.1-8b-it/$CONS.jsonl" ] || cp "$DPO_SRC" "$D/dpo/llama-3.1-8b-it/$CONS.jsonl"
echo "$DPO_SHA  $D/dpo/llama-3.1-8b-it/$CONS.jsonl" | sha256sum -c - || die "DPO data hash"
echo "$SFT_SHA  $D/sft_data/llama-3.1-8b-it/$CONS.jsonl" | sha256sum -c - || die "SFT corpus hash"

cd "$OCT"
banner "A. DPO"
prov dpo "bash finetuning/distillation/llama_local.sh $CONS"
bash finetuning/distillation/llama_local.sh "$CONS" || die "DPO failed"
test -f "$DPO_OUT/adapter_model.safetensors" || die "DPO adapter missing"

banner "B. fold (loving-only source dir: fold_loras.py folds every constitution it finds)"
SRC=$RIG/loras_traj_src/$CONS
mkdir -p "$SRC"; ln -sfn "$DPO_OUT" "$SRC/$CONS"
[ "$(ls "$SRC")" = "$CONS" ] || die "fold source dir is not $CONS-only"
prov fold "python tools/fold_loras.py --model_name llama-3.1-8b-it --loras_dir $SRC --save_dir_name distilled"
python3 tools/fold_loras.py --model_name llama-3.1-8b-it --loras_dir "$SRC" --save_dir_name distilled \
  || die "fold failed"
test -f "$DISTILLED/model.safetensors.index.json" || die "distilled model missing"

banner "C. introspection SFT, 3 epochs, one continuous run, checkpoint every 15 steps"
prov sft "bash finetuning/introspection/llama_curve.sh $CONS"
bash finetuning/introspection/llama_curve.sh "$CONS" || die "SFT failed"
test -f "$SFT_OUT/adapter_model.safetensors" || die "final SFT adapter missing"

banner "D. released-style merge [1.0, 0.25] -- endpoint only, for the reproduction gate"
ln -sfn "$SFT_OUT" "$INTRO_LINK"     # merge_loras.py reads llama-test/<c> -> llama-introspection/<c>
prov merge "python tools/merge_loras.py --model_name llama-3.1-8b-it --constitution $CONS"
python3 tools/merge_loras.py --model_name llama-3.1-8b-it --constitution "$CONS" || die "merge failed"
test -f "$MERGED/adapter_model.safetensors" || die "merged adapter missing"

banner "DONE"
echo "checkpoints: $(ls -d "$CKPTS"/global_step*_hf | sed 's/.*global_step//;s/_hf//' | sort -n | tr '\n' ' ')"
cat "$MERGED/adapter_config.json" | python3 -c "import json,sys;c=json.load(sys.stdin);print('merged r',c['r'],'alpha',c['lora_alpha'],sorted(c['target_modules']))"
rm -rf "$HOME/wandb"
echo "TRAJ TRAINING COMPLETE $CONS"
