#!/usr/bin/env bash
# Fold -> pre-training checks (hard gate) -> four SFT runs. spec_xchar_sft.md.
set -euo pipefail
R=/workspace/repos/steering_across_personas; cd "$R"
ARMS="" bash scripts/run_xchar_sft.sh
PYTHONPATH=/workspace/pylibs-train-py312 HF_HUB_OFFLINE=1 python3 scripts/xchar_pretrain_checks.py
python3 -c "import json,sys; sys.exit(0 if json.load(open('outputs/analysis/xchar_sft/checks_pretrain.json'))['PASS_start_and_params'] else 'PRE-TRAINING CHECKS FAILED')"
bash scripts/run_xchar_sft.sh
echo "CHAIN XCHAR TRAIN DONE"
