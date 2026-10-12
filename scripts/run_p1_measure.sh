#!/usr/bin/env bash
# Everything P1 owes, in information-per-minute order.
# docs/runs/oct/AMENDMENT_P1_exploratory.md -- P1 is EXPLORATORY. The primary reported quantity
# is P1 minus P0 with shared bootstrap draws. P1 is NOT scored against P0's reproduction gate.
#
# Order is chosen so an interruption still leaves the primary comparison in hand:
#   1. P1 merged endpoint, forced          -> B1/B2 for the headline            (6 min)
#   2. weight/factor diagnostics (CPU)     -> cos(A_D,A_S); catches regime change (2 min)
#   3. P1 stage arms, forced               -> DPO-only, SFT-only, D+S, D+0.25S  (24 min)
#   4. P1 merged endpoint, default         -> prompt-form robustness             (6 min)
#   5. logits analysis (CPU)
#   6. paired P1-vs-P0, B1 and B2          -> THE PRIMARY RESULT, shared draws
#   7. P1 activations (GPU)                -> selectivity, dose, common shift   (~85 min)
#   8. question cache -> common shift -> selectivity -> functional dose
set -uo pipefail
cd /workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
P1=impulsiveness_paraphrase
P0=impulsiveness_regen
STAGES="${P1}_dpo ${P1}_sft ${P1}_DplusS ${P1}_Dplus025S"

say() { echo; echo "######## $(date -u +%FT%TZ)  $*"; }
# CPU work runs niced and thread-capped: an unniced audit throttled a GPU run on 2026-10-07.
cpu() { OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 "$@"; }

say "0. preflight"
source /workspace/bootstrap.sh
bash scripts/preflight.sh 2>&1 | tail -5

say "1. P1 merged endpoint, forced  -> B1/B2"
ARMS="$P1" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -12

say "2. weight/factor diagnostics -> cos(A_D, A_S), merge regime check"
cpu python3 scripts/audit_factor_weighting.py --pairs p1 \
  --out outputs/analysis/audit_factor_weighting_p1.json 2>&1 | tail -22
cpu python3 scripts/audit_A_factor_matrix.py \
  --out outputs/analysis/audit_A_factor_matrix_p1.json 2>&1 | tail -30

say "3. P1 stage arms, forced  -> DPO-only, SFT-only, D+S, D+0.25S"
ARMS="$STAGES" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -20

say "4. P1 merged endpoint, default  -> prompt-form robustness"
ARMS="$P1" VARIANTS=default bash scripts/run_caa_logits.sh 2>&1 | tail -12

say "5. logits analysis (CPU)"
python3 scripts/caa_logits_analysis.py 2>&1 | tail -20

say "6. PRIMARY: paired P1 - P0, shared bootstrap draws"
echo "--- B1 partition (impulsivity vs other seven) ---"
python3 scripts/paired_offset_diff.py --a "$P0" --b "$P1" --targets impulsivity \
  --out outputs/analysis/paired_offset_diff_p1_vs_p0_B1.json 2>&1 | tail -22
echo "--- B2 partition (target pair vs other six) ---"
python3 scripts/paired_offset_diff.py --a "$P0" --b "$P1" \
  --out outputs/analysis/paired_offset_diff_p1_vs_p0_B2.json 2>&1 | tail -22
echo "--- secondary: P1 vs the reproduction, for scale ---"
python3 scripts/paired_offset_diff.py --a impulsiveness_repro --b "$P1" --targets impulsivity \
  --out outputs/analysis/paired_offset_diff_p1_vs_repro_B1.json 2>&1 | tail -14

say "7. P1 activations (GPU ~85 min)  -> selectivity, dose, common shift"
ADAPTER_PATH="$RIG/loras/llama-personas/$P1" \
  bash scripts/run_arm.sh "$P1" --gpu-only 2>&1 | tail -25

say "8. question cache"
python3 scripts/build_question_cache.py --arms "$P1" --layers 15 20 2>&1 | tail -8

say "9. common shift + geometry  (released, P0, P1 side by side)"
python3 scripts/common_shift.py --arms impulsiveness "$P0" "$P1" --layers 15 \
  --bootstrap 200 2>&1 | tail -30

say "10. functional dose (measured, never inferred from weight norm)"
python3 scripts/functional_dose.py --arms impulsiveness "$P0" "$P1" --layer 15 2>&1 | tail -20

say "11. selectivity: P1 and P0 side by side"
for A in "$P1" "$P0"; do
  echo "--- $A ---"
  python3 scripts/selectivity_score.py --arm "$A" --targets impulsivity risk_taking \
    --layers 15 2>&1 | tail -16
done

say "P1 MEASUREMENT COMPLETE"
