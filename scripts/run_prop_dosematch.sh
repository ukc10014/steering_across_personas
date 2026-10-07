#!/usr/bin/env bash
# The one arm that makes the propagation gate interpretable.
#
# The propagation arm's DPO stage was trained at seed 987654 while its introspection SFT kept
# upstream's 123456. Upstream pins BOTH stages to 123456, and LoRA's A factor is essentially the
# random init, so the published pipeline's two stages share an A (cos +0.989) and its factor merge
# collapses to the shared-A limit: effective 1.47*D + 0.71*S, ~96% of the update inside span{D,S}.
# Reseeding one stage breaks that. For the propagation arm cos(A_D, A_S) = -0.0002, the merge
# applies the nominal (1.0, 0.25) almost exactly (fitted 1.0004, 0.2502), and 82% of its squared
# norm goes into cross terms outside span{D,S}.
#
# So the merged propagation arm is NOT dose-matched to repro / P0 / seed2, and on its own a low B1
# there says nothing about the regenerated teacher data. This arm applies the coefficients the
# published merge actually delivers -- the SAME scalars as impulsiveness_repro_fit -- to the
# propagation pair, giving a like-for-like dose-matched comparison.
#
# Forced prompt only: that is what B1/B2 are defined on, and it is one 6-minute arm.
set -uo pipefail
cd /workspace/repos/steering_across_personas
say() { echo; echo "######## $(date -u +%FT%TZ)  $*"; }

say "dose-matched propagation arm, forced"
source /workspace/bootstrap.sh
ARMS="impulsiveness_regen_s987654_fit" VARIANTS=forced bash scripts/run_caa_logits.sh 2>&1 | tail -12

say "logits analysis (CPU)"
python3 scripts/caa_logits_analysis.py 2>&1 | tail -60

say "PROP DOSE-MATCH COMPLETE"
