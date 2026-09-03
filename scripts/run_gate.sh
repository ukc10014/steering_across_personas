#!/usr/bin/env bash
# The full spec_sham_lora.md 6b gate for the corrected seed-123456 reproduction.
# Run AFTER run_repro_sft3ep.sh has finished. Unattended; ~2h wall.
#
#   A1-A3 + 6c   scripts/gate_weight_checks.py            CPU
#   B1,B2,B5     CAA logits (forced) + analysis + robustness
#   B3,B4        CAA activations -> question cache -> common_shift
#   A4           functional_dose (MEASURED, never inferred from weight norm -- 7.1's error)
set -uo pipefail
R=/workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
LOGS=$RIG/logs
ARM=impulsiveness_repro
cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R"
export HF_HUB_OFFLINE=1
banner(){ echo; echo "######## $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "GATE ABORTED: $*"; exit 1; }

# ---- A. freeze the tree, so seed 2 cannot overwrite it -------------------------------
banner "A. freeze loras -> loras_repro"
if [ -d "$RIG/loras_repro" ]; then
  echo "  loras_repro already exists -- assuming already frozen"
else
  test -f "$RIG/loras/llama-personas/impulsiveness/adapter_model.safetensors" \
    || die "no merged adapter in $RIG/loras"
  mv "$RIG/loras" "$RIG/loras_repro"
  mkdir -p "$RIG/loras" && ln -sfn llama-introspection "$RIG/loras/llama-test"
fi
bash "$RIG/verify_adapters.sh" "$RIG/loras_repro" impulsiveness || die "adapters incomplete"

# ---- C. A1-A3 and the 6c cross term --------------------------------------------------
banner "C. A1-A3 + 6c cross-term share  (CPU)"
python3 scripts/gate_weight_checks.py 2>&1 | tee "$LOGS/gate_weight_checks.log"

# ---- D. B1, B2, B5 -------------------------------------------------------------------
banner "D. CAA logits, forced prompt, arm $ARM  (GPU ~7 min)"
ARMS="$ARM" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -20
grep -q "CAA LOGITS DONE" "$LOGS/../logs/gate_forced.log" 2>/dev/null || true
banner "D2. logits analysis + robustness  (CPU)"
python3 scripts/caa_logits_analysis.py > "$LOGS/gate_logits_analysis.log" 2>&1 \
  || die "caa_logits_analysis failed"
python3 scripts/caa_logits_robustness.py --n-boot 400 > "$LOGS/gate_logits_robustness.log" 2>&1 \
  || echo "  !! robustness failed -- see $LOGS/gate_logits_robustness.log"

# ---- E. activations for the geometry criteria ----------------------------------------
banner "E. merge -> verify answer token -> CAA activations  (GPU ~85 min)"
ADAPTER_PATH="$RIG/loras_repro/llama-personas/impulsiveness" \
  bash scripts/run_arm.sh "$ARM" --gpu-only 2>&1 | tail -30
banner "E2. pack the question cache"
python3 scripts/build_question_cache.py --arms "$ARM" --layers 15 20 2>&1 | tail -10

# ---- F. B3, B4 -----------------------------------------------------------------------
banner "F. common shift: B3 selectivity, B4 cos(dG_repro, dG_released)  (CPU)"
python3 scripts/common_shift.py --arms impulsiveness "$ARM" --layers 15 --bootstrap 200 \
  2>&1 | tee "$LOGS/gate_common_shift.log" | tail -40

# ---- G. A4 ---------------------------------------------------------------------------
banner "G. A4 MEASURED functional dose  (never inferred from weight norm)"
python3 scripts/functional_dose.py --arms impulsiveness "$ARM" --layer 15 \
  2>&1 | tee "$LOGS/gate_functional_dose.log" | tail -25

banner "GATE INPUTS COMPLETE -- score against spec 6b and STOP. Seed 2 needs a human."
