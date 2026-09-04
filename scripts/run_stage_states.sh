#!/usr/bin/env bash
# Stage localisation: measure the summed model states. docs/spec_stage_localisation.md §1-2.
#
# States M_D (=*_dpo) and M_F (=the bare arm) are already measured for both seeds; these are
# the states in between, built by applying the two stage adapters additively at extraction
# time. No merged checkpoint is written: 2c/2d both take repeated --lora-adapter/--lora-scale.
#
# DISK: raw activations are 24G per arm and the geometry only ever reads the 1.5G qcache, so
# each arm is extracted -> cached -> raw deleted. Peak stays at one arm (spec §8a). This is a
# recorded decision: re-analysing at layers other than 15/20 would need re-extraction.
#
# NOTE 2c defaults --output-dir from the MODEL name. Since every state here loads the same
# base model, the default would write them all into the base arm's directory and destroy it.
# --output-dir is therefore passed explicitly and never omitted.
set -uo pipefail
R=/workspace/repos/steering_across_personas
RIG=/workspace/oct_rig
LOGS=$RIG/logs
cd "$R"
source /workspace/bootstrap.sh >/dev/null
export PYTHONPATH="$PYLIBS:$R"; export HF_HUB_OFFLINE=1
BASE=/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
MIN_PT_BYTES=130000000
banner(){ echo; echo "######## $(date -u +%FT%TZ)  $*"; echo; }
die(){ echo "STAGE-STATES ABORTED at: $*"; exit 1; }

flags_for(){ case "$1" in
  impulsiveness_repro_DplusS)     echo "--lora-adapter $RIG/loras_repro/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 1" ;;
  impulsiveness_repro_Dplus025S)  echo "--lora-adapter $RIG/loras_repro/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 0.25" ;;
  impulsiveness_repro_sft)        echo "--lora-adapter $RIG/loras_repro/llama-introspection/impulsiveness --lora-scale 1" ;;
  impulsiveness_seed2_DplusS)     echo "--lora-adapter $RIG/loras_seed2/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_seed2/llama-introspection/impulsiveness --lora-scale 1" ;;
  impulsiveness_seed2_Dplus025S)  echo "--lora-adapter $RIG/loras_seed2/llama-distillation/impulsiveness --lora-scale 1 --lora-adapter $RIG/loras_seed2/llama-introspection/impulsiveness --lora-scale 0.25" ;;
  impulsiveness_sft_from_base)    echo "--lora-adapter $RIG/loras_sft_from_base/impulsiveness --lora-scale 1" ;;
  *) echo "" ;;
esac; }

ARMS_TO_RUN=${ARMS_TO_RUN:-"impulsiveness_repro_DplusS impulsiveness_repro_Dplus025S impulsiveness_repro_sft impulsiveness_seed2_DplusS impulsiveness_seed2_Dplus025S"}

for ARM in $ARMS_TO_RUN; do
  FLAGS="$(flags_for "$ARM")"
  [ -n "$FLAGS" ] || die "no adapter flags for $ARM"
  OUT="outputs/llama-3.1-8b-$ARM/caa_activations"

  banner "$ARM -- 1/3 CAA logits, both prompt forms"
  if [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits_forced" 2>/dev/null | wc -l)" -eq 88 ] \
     && [ "$(ls "outputs/llama-3.1-8b-$ARM/caa_logits" 2>/dev/null | wc -l)" -eq 88 ]; then
    echo "  already have 88+88 cells -- skipping"
  else
    ARMS="$ARM" bash scripts/run_caa_logits.sh > "$LOGS/stage_logits_$ARM.log" 2>&1
    grep -q "CAA LOGITS DONE" "$LOGS/stage_logits_$ARM.log" || die "logits $ARM (see $LOGS/stage_logits_$ARM.log)"
  fi

  banner "$ARM -- 2/3 CAA activations (192 cells)"
  if [ -f "outputs/_qcache/${ARM}_L15_20.npz" ]; then
    echo "  qcache already exists -- skipping extraction"
  else
    # shellcheck disable=SC2086
    python3 pipeline/2c_caa_activations.py --model "$BASE" $FLAGS \
        --output-dir "$OUT" --batch-size 16 \
        > "$LOGS/stage_extract_$ARM.log" 2>&1 || die "extraction $ARM"
    n=$(ls "$OUT" 2>/dev/null | wc -l)
    [ "$n" -eq 192 ] || die "$ARM extraction incomplete ($n/192)"
    small=$(find "$OUT" -name '*.pt' -size -${MIN_PT_BYTES}c | wc -l)
    [ "$small" -eq 0 ] || die "$ARM has $small truncated cells"

    banner "$ARM -- 3/3 pack qcache, then drop the 24G raw"
    python3 scripts/build_question_cache.py --arms "$ARM" --layers 15 20 \
        > "$LOGS/stage_cache_$ARM.log" 2>&1 || die "question cache $ARM"
    [ -f "outputs/_qcache/${ARM}_L15_20.npz" ] || die "qcache missing for $ARM"
    rm -rf "$OUT"
    echo "  cached $(du -sh "outputs/_qcache/${ARM}_L15_20.npz" | cut -f1); raw activations removed"
  fi
  echo "  disk now: $(du -sh /workspace 2>/dev/null | cut -f1)"
done

banner "STAGE STATES COMPLETE -- analyses next"
