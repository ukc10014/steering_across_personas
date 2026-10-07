#!/usr/bin/env bash
# Everything the GPU still owes after the p0s2 propagation run, in information-per-minute order.
#
# Ordered so that an interruption still leaves the most decisive numbers in hand:
#   1. propagation endpoint, forced  -- B1/B2 for D_n' carried through the whole pipeline (6 min)
#   2. audit 1 + audit 2 behavioural, forced -- the two audits that need a model (8 arms, 48 min)
#   3. propagation endpoint, default -- prompt-form robustness for the gate-scored arm (6 min)
#   4. propagation activations -> question cache -> B3 + A4            (~90 min)
#
# FORCED ONLY for the diagnostic arms. The forced prompt is what B1/B2/B3 are defined on; the
# `default` form is a robustness check already established for the main arms (run_caa_logits.sh
# header), and eight more arms of it would be ~48 min of billing for a question already answered.
# The gate-scored propagation arm gets both.
set -uo pipefail
cd /workspace/repos/steering_across_personas
R=/workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
LOGS=$RIG/logs
PROP=impulsiveness_regen_s987654
DIAG="impulsiveness_repro_fit impulsiveness_regen_fit impulsiveness_repro_15D075S impulsiveness_regen_15D075S signflip_So_neg impulsiveness_repro_sft_negAB ${PROP}_sft ${PROP}_Dplus025S"

say() { echo; echo "######## $(date -u +%FT%TZ)  $*"; }

say "0. preflight"
source /workspace/bootstrap.sh
bash scripts/preflight.sh 2>&1 | tail -5

say "1. propagation endpoint, forced  -> B1/B2"
ARMS="$PROP" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -12

say "2. audit 1 + audit 2 behavioural arms, forced"
ARMS="$DIAG" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -20

say "3. propagation endpoint, default  -> prompt-form robustness"
ARMS="$PROP" VARIANTS=default bash scripts/run_caa_logits.sh 2>&1 | tail -12

say "4. logits analysis (CPU)"
python3 scripts/caa_logits_analysis.py 2>&1 | tail -45

say "5. propagation activations (GPU ~85 min)  -> B3, A4"
ADAPTER_PATH="$RIG/loras/llama-personas/$PROP" \
  bash scripts/run_arm.sh "$PROP" --gpu-only 2>&1 | tail -25

say "6. question cache"
python3 scripts/build_question_cache.py --arms "$PROP" --layers 15 20 2>&1 | tail -8

say "7. common shift -> B3"
python3 scripts/common_shift.py --arms impulsiveness impulsiveness_regen "$PROP" --layers 15 \
  --bootstrap 200 2>&1 | tail -30

say "8. functional dose -> A4  (measured, never inferred from weight norm)"
python3 scripts/functional_dose.py --arms impulsiveness impulsiveness_regen "$PROP" --layer 15 \
  2>&1 | tail -20

say "9. paired offsets: propagation vs P0 and vs repro"
python3 scripts/paired_offset_diff.py --a impulsiveness_regen --b "$PROP" \
  --out outputs/analysis/paired_offset_diff_prop_vs_p0.json 2>&1 | tail -20
python3 scripts/paired_offset_diff.py --a impulsiveness_repro --b "$PROP" \
  --out outputs/analysis/paired_offset_diff_prop_vs_repro.json 2>&1 | tail -20

say "POST-PROP MEASUREMENT COMPLETE"
