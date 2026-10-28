#!/usr/bin/env bash
# G1 (docs/AUDIT_POST_SUBMISSION_RESULTS.md §9): corrected-mask CAA activations for the
# unadapted base and the released impulsiveness / goodness / sycophancy adapters.
#
#   MASK=corrected DEADLINE=<unix epoch> bash scripts/audit/run_g1_extract.sh [arm ...]
#
# Protocol matches the corrected-mask trajectory/xchar caches (common_trajectory/run_measure.sh):
# same base snapshot as --model, released adapters applied on the fly at scale 1 (as the
# trajectory's reproduced `*_traj_mrg_final` arms), batch 16, default prompt variant, all
# 12 personas x 8 traits x all questions, L15/L20 cache. Arm keys are new (`*_cmask`), so the
# legacy `base` / released caches are never touched; provenance comes from the sidecar 2c
# writes, which build_question_cache embeds in the .npz.
#
# Budget: an arm is started only if the measured (or given) per-arm time fits before DEADLINE.
# 2c resumes per cell into its own dir (sidecar-checked), so a stopped arm loses at most a cell.
set -uo pipefail
MASK=${MASK:?set MASK=corrected}
[ "$MASK" = corrected ] || { echo "G1 is the corrected-mask extraction"; exit 1; }
DEADLINE=${DEADLINE:?unix epoch after which no new arm starts}
EST=${EST_ARM_SEC:-3600}          # replaced by the measured time once an arm completes
R=/workspace/repos/sap-audit
LOGS=/workspace/oct_rig/logs
BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620
MIN_PT_BYTES=130000000
ARMS=("${@:-base_cmask impulsiveness_rel_cmask goodness_rel_cmask sycophancy_rel_cmask}")
read -r -a ARMS <<< "${ARMS[*]}"

cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R" HF_HUB_OFFLINE=1
log(){ echo "$(date -u +%FT%TZ) $*"; }
die(){ log "G1 ABORTED at: $*"; exit 1; }

adapter_flags(){
  case "$1" in
    base_cmask) ;;
    *_rel_cmask) local c=${1%_rel_cmask}; test -f "$SNAP/$c/adapter_model.safetensors" || die "missing $SNAP/$c"
                 echo "--lora-adapter $SNAP/$c --lora-scale 1" ;;
    *) die "unknown arm $1" ;;
  esac
}

for ARM in "${ARMS[@]}"; do
  Q="outputs/_qcache/${ARM}_L15_20.npz"
  if [ -f "$Q" ]; then log "$ARM qcache exists -- skipping"; continue; fi
  now=$(date +%s)
  if [ $((now + EST)) -gt "$DEADLINE" ]; then
    log "BUDGET: $ARM needs ~${EST}s, $((DEADLINE - now))s left -- not started"; continue
  fi
  OUT="outputs/llama-3.1-8b-$ARM/caa_activations"
  t0=$(date +%s)
  log "$ARM extract -> $OUT"
  # shellcheck disable=SC2046
  python3 -u pipeline/2c_caa_activations.py --mask "$MASK" --model "$BASE" $(adapter_flags "$ARM") \
      --output-dir "$OUT" --batch-size 16 > "$LOGS/g1_extract_$ARM.log" 2>&1 || die "extraction $ARM"
  n=$(ls "$OUT"/*.pt 2>/dev/null | wc -l); [ "$n" -eq 192 ] || die "$ARM incomplete ($n/192)"
  [ "$(find "$OUT" -name '*.pt' -size -${MIN_PT_BYTES}c | wc -l)" -eq 0 ] || die "$ARM truncated cells"
  t1=$(date +%s)
  python3 -u scripts/build_question_cache.py --arms "$ARM" --layers 15 20 \
      > "$LOGS/g1_cache_$ARM.log" 2>&1 || die "qcache $ARM"
  python3 -u scripts/audit/validate_g1_cache.py "$ARM" > "$LOGS/g1_validate_$ARM.log" 2>&1 \
      || die "validate $ARM (see $LOGS/g1_validate_$ARM.log)"
  t2=$(date +%s)
  EST=$((t2 - t0 + 120))
  printf '{"arm":"%s","extract_s":%d,"cache_validate_s":%d,"gpu":"%s","finished":"%s"}\n' \
      "$ARM" $((t1 - t0)) $((t2 - t1)) "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
      "$(date -u +%FT%TZ)" >> "$LOGS/g1_timings.jsonl"
  log "$ARM done: extract $((t1 - t0))s, cache+validate $((t2 - t1))s"
  # Keep the base's raw all-layer activations (the only corrected base); adapter arms keep the
  # L15/L20 cache only, as the trajectory runner does.
  [ "$ARM" = base_cmask ] || rm -rf "$OUT"
done
log "G1 RUNNER FINISHED"
