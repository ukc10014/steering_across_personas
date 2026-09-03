#!/usr/bin/env bash
# Verify a finished OCT run kept ALL THREE adapters, and write a manifest beside them.
#
# merge_loras.py only ever rm -rf's paths INSIDE its own output dir
# (llama-personas/<cons>/{dpo,sft,persona}), never the component trees -- checked in the
# source, and re-checked here at runtime, because "the merge ate the components" is a silent
# failure you would only notice when trying to measure the DPO arm hours later.
#
#   bash /workspace/oct_rig/verify_adapters.sh [loras_dir] [constitution]
set -uo pipefail
DIR=${1:-/workspace/oct_rig/loras}
CONS=${2:-impulsiveness}
FAIL=0

printf '%-22s %-14s %-12s %s\n' STAGE R/ALPHA SIZE PATH
for spec in "dpo:llama-distillation" "sft:llama-introspection" "merged:llama-personas"; do
  stage=${spec%%:*}; sub=${spec##*:}
  p="$DIR/$sub/$CONS"
  f="$p/adapter_model.safetensors"
  if [ ! -f "$f" ]; then
    printf '%-22s %-14s %-12s %s\n' "$stage" "MISSING" "-" "$p"; FAIL=1; continue
  fi
  ra=$(python3 -c "
import json;c=json.load(open('$p/adapter_config.json'))
print(f\"r={c['r']} a={c['lora_alpha']}\")" 2>/dev/null || echo "?")
  printf '%-22s %-14s %-12s %s\n' "$stage" "$ra" "$(du -h "$f" | cut -f1)" "$p"
done

echo
if [ "$FAIL" -ne 0 ]; then echo "!! at least one adapter is missing"; exit 1; fi

cat > "$DIR/ADAPTERS.md" <<EOF
# Adapters in this tree

One OCT run for constitution \`$CONS\`. **All three are kept**, so any of them can be applied
or analysed on its own; the merged one is only what OCT ships.

| adapter | path | what it is |
|---|---|---|
| DPO stage | \`llama-distillation/$CONS\` | trained on the frozen DPO pairs, r=64 alpha=128. Free of the peft merge cross terms (spec 6c), so this is the clean single-stage artifact. |
| SFT stage | \`llama-introspection/$CONS\` | introspection SFT, r=64 alpha=128. **Trained on the FOLDED distilled model** (base + dW_dpo), not on the base -- applying it to the plain base is a component measurement, not "the SFT half". |
| merged | \`llama-personas/$CONS\` | \`add_weighted_adapter(["dpo","sft"], [1.0, 0.25], "linear")\`, r=64 alpha=64. This is the artifact OCT releases and the one every published arm was measured on. |

\`llama-test\` is a symlink to \`llama-introspection\`: \`merge_loras.py:38\` reads an SFT
adapter from a directory nothing in the public repo writes (spec 6a). Recorded, not silently
patched.

The merged adapter is **not** \`dW_dpo + 0.25*dW_sft\`. peft combines LoRA *factors*, so it
also carries \`B_dpo@A_sft + B_sft@A_dpo\`; see \`scripts/check_peft_merge.py\`.

CAA-logit arm names: \`${CONS}_<seed>\`, \`${CONS}_<seed>_dpo\`, \`${CONS}_<seed>_sft\`
(scripts/run_caa_logits.sh).

Generated $(date -u +%FT%TZ) by verify_adapters.sh.
EOF
echo "all three present; wrote $DIR/ADAPTERS.md"
