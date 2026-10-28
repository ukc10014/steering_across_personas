#!/usr/bin/env bash
# Paper figures from the completed G1 (corrected-mask base) results. CPU only; plotting only --
# nothing here extracts, trains, judges or bootstraps. Writes paper_g1/figures/ (PDF + PNG +
# plotted-data CSV + provenance JSON per figure) and refreshes the same figures beside their
# analysis outputs.
#
#   bash paper_g1/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3.12}
export PYTHONPATH="/workspace/pylibs-py312:$PWD"
F=paper_g1/figures
mkdir -p "$F"

# 1. four-constitution trajectory: appendix A-D and main-text A/B
$PY scripts/common_trajectory/figures.py --geometry-tag _cmbase --behaviour-tag _10personas --dose-tag _cmbase \
    --out-tag _cmbase --no-dose-companion --formats pdf png --out-dir "$F"
$PY scripts/common_trajectory/figures.py --panels AB --geometry-tag _cmbase --behaviour-tag _10personas \
    --dose-tag _cmbase --out-tag _cmbase --formats pdf png --out-dir "$F"

# 2. paired matched-dose excess: overview + per-constitution companions
$PY scripts/audit/fig_matched_dose.py --tag _cmbase --formats pdf png --out-dir "$F"

# 3. cross-character acquisition/retention (saved signed.json, no refit)
$PY scripts/xchar_analysis.py --plot-only --formats pdf png --out-dir "$F"

# 4. SFT-curve appendix figure + table, activation inputs vs the corrected base
$PY scripts/audit/merge_oct_common_shift.py
$PY scripts/appendix_oct/figA_oct_sft_curve.py --corrected-base --out-dir "$F"
$PY scripts/appendix_oct/tableA_oct_sft_curve.py --corrected-base --out-dir "$F"

# beside the analysis outputs too (PNG previews), so those directories are not stale
$PY scripts/common_trajectory/figures.py --geometry-tag _cmbase --behaviour-tag _10personas --dose-tag _cmbase \
    --out-tag _cmbase --no-dose-companion
$PY scripts/common_trajectory/figures.py --panels AB --geometry-tag _cmbase --behaviour-tag _10personas \
    --dose-tag _cmbase --out-tag _cmbase
$PY scripts/audit/fig_matched_dose.py --tag _cmbase
$PY scripts/xchar_analysis.py --plot-only

# 5. checks: values against the source JSON, and every PDF vector with embedded TrueType fonts
$PY paper_g1/verify_assets.py
for p in "$F"/*.pdf; do
  PYTHONPATH=/workspace/pylibs-pdfcheck $PY paper_g1/check_pdf.py "$p" --print-width-in 5.5 \
      --out "${TMPDIR:-/tmp}/$(basename "$p" .pdf)_print.png" 2>/dev/null
done
echo "BUILD OK"
