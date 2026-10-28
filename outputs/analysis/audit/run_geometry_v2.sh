#!/bin/bash
# audit reruns of geometry.py: A = historical (mixed-protocol) base, B = synthetic corrected base
cd /workspace/repos/sap-audit
export PYTHONPATH=/workspace/pylibs-py312 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
L=outputs/analysis/audit/logs
for c in sycophancy goodness loving impulsiveness; do
  python3.12 scripts/common_trajectory/geometry.py --constitution $c --allow-mixed-mask --tag _mixedprotocol > $L/geom_${c}_A.log 2>&1 &
  python3.12 scripts/common_trajectory/geometry.py --constitution $c --cache-dir outputs/analysis/audit/qcache_synth \
      --base-arm base_synthcorrected --tag _synthbase > $L/geom_${c}_B.log 2>&1 &
  wait
  echo "$c done A=$(tail -1 $L/geom_${c}_A.log) B=$(tail -1 $L/geom_${c}_B.log)"
done
echo ALL_DONE
