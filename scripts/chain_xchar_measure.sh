#!/usr/bin/env bash
# After training: measure -> held-out loss + qualitative -> activation selectivity -> analysis.
# Waits on the training chain's own EXIT line (never a pgrep -f, which matches this waiter).
set -euo pipefail
R=/workspace/repos/steering_across_personas; cd "$R"
TL=/workspace/oct_rig/logs/xchar_chain_train.log
until grep -q "^EXIT" "$TL"; do sleep 60; done
grep -q "^EXIT 0" "$TL" || { echo "training chain failed -- not measuring"; exit 1; }
bash scripts/run_xchar_measure.sh
PYTHONPATH=/workspace/pylibs-train-py312 HF_HUB_OFFLINE=1 python3 scripts/xchar_heldout_and_qual.py
source /workspace/bootstrap.sh >/dev/null; export PYTHONPATH="$PYLIBS:$R"
XA="xchar_lov_start xchar_lov_imp_s100 xchar_lov_lov_s100 xchar_base_imp_s100 xchar_base_lov_s100"
# a NEW output file: common_shift.json has been clobbered by a stage run before (DECISIONS.md)
python3 scripts/common_shift.py --arms loving impulsiveness $XA --out outputs/analysis/xchar_sft/common_shift.json
for A in loving $XA; do
  python3 scripts/selectivity_score.py --arm "$A" --targets impulsivity risk_taking \
    --common-shift outputs/analysis/xchar_sft/common_shift.json --out "outputs/analysis/xchar_sft/selectivity_${A}_imp.json"
  python3 scripts/selectivity_score.py --arm "$A" --targets empathy warmth \
    --common-shift outputs/analysis/xchar_sft/common_shift.json --out "outputs/analysis/xchar_sft/selectivity_${A}_lov.json"
done
python3 scripts/xchar_analysis.py
echo "CHAIN XCHAR MEASURE DONE"
