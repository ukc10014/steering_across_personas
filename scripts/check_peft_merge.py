#!/usr/bin/env python3
"""What OCT's final merge actually produces, and where lora_alpha changes.

`tools/merge_loras.py` builds every released persona adapter with

    add_weighted_adapter(["dpo", "sft"], weights=[1.0, 0.25], combination_type="linear")

which reads as "the DPO update plus a quarter of the SFT update". It is not that. peft's
"linear" combination operates on the FACTORS, not on the products (peft/tuners/lora/model.py,
`_generalized_task_arithmetic_weighted_adapter`):

    A_new = sum_i sqrt(w_i * scaling_i) * A_i
    B_new = sum_i sqrt(w_i * scaling_i) * B_i

so with dW = scaling * B @ A the result carries cross terms between the two adapters:

    dW_merged = 1.0*dW_dpo + 0.25*dW_sft  +  (B_dpo @ A_sft + B_sft @ A_dpo)
                \___________ intended ___________/  \______ artefact of factor-space _____/

Both facts this establishes matter for the sham:

1. **lora_alpha.** The runners train at alpha=128, the released adapters read alpha=64, and
   the plan recorded that as an unresolved discrepancy. It is not a discrepancy: the merge
   folds each adapter's scaling into the sqrt weights, so the merged adapter comes out at
   scaling 1.0, i.e. alpha = r = 64. A correct reproduction WILL land on 64. Do not
   hand-adjust; if a reproduction gives anything else, the rig is wrong.

2. **The merged artifact is partly an algebraic accident.** On independent random adapters
   the cross term is ~59% of the merged norm. On the real pair it will differ -- sft is
   trained on top of the folded dpo model, so the subspaces are not independent -- and it is
   measurable exactly once the rig produces the component adapters. Until then it is a
   further reason the sham's primary comparison is at the DPO stage (spec sec 3.3), where
   no such term exists.

This does not invalidate any existing measurement: the merged adapter is what OCT released
and what all nine arms were measured on. It invalidates only the mental model of it.

    python scripts/check_peft_merge.py        # ~20 s, CPU, no downloads
"""
from __future__ import annotations

import json
import tempfile

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
R, ALPHA, WEIGHTS = 64, 128, [1.0, 0.25]


def factors(path: str):
    w = load_file(f"{path}/adapter_model.safetensors")
    k = next(x for x in w if "q_proj" in x and "lora_A" in x)
    return w[k].float(), w[k.replace("lora_A", "lora_B")].float()


def main() -> None:
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=128)
    d = tempfile.mkdtemp()
    torch.manual_seed(0)
    for name in ("dpo", "sft"):
        m = get_peft_model(LlamaForCausalLM(cfg),
                           LoraConfig(r=R, lora_alpha=ALPHA, target_modules=TARGETS,
                                      task_type="CAUSAL_LM"))
        for p in m.parameters():          # B initialises to zero; make it nonzero
            if p.requires_grad:
                torch.nn.init.normal_(p, std=0.02)
        m.save_pretrained(f"{d}/{name}")

    model = PeftModel.from_pretrained(LlamaForCausalLM(cfg), f"{d}/dpo", adapter_name="dpo")
    model.load_adapter(f"{d}/sft", adapter_name="sft")
    model.add_weighted_adapter(["dpo", "sft"], WEIGHTS, "persona", combination_type="linear")
    model.set_adapter("persona")
    model.save_pretrained(f"{d}/out", adapter_name="persona")

    Ad, Bd = factors(f"{d}/dpo")
    As, Bs = factors(f"{d}/sft")
    Am, Bm = factors(f"{d}/out/persona")
    conf = json.load(open(f"{d}/out/persona/adapter_config.json"))
    scale_in, scale_out = ALPHA / R, conf["lora_alpha"] / conf["r"]

    merged = scale_out * (Bm @ Am)
    intended = WEIGHTS[0] * scale_in * (Bd @ Ad) + WEIGHTS[1] * scale_in * (Bs @ As)
    cross = (Bd @ As) + (Bs @ Ad)
    resid = (merged - (intended + cross)).norm().item()

    print(f"  trained adapters   r={R} alpha={ALPHA}  (scaling {scale_in:g})")
    print(f"  merged adapter     r={conf['r']} alpha={conf['lora_alpha']}  "
          f"(scaling {scale_out:g})   released reads r=64 alpha=64")
    print()
    print(f"  ||dW_merged||                              {merged.norm():.4f}")
    print(f"  ||1.0*dW_dpo + 0.25*dW_sft||               {intended.norm():.4f}")
    print(f"  ||B_dpo@A_sft + B_sft@A_dpo||  (cross)     {cross.norm():.4f}")
    print(f"  ||merged - (intended + cross)||            {resid:.2e}   "
          f"{'exact' if resid < 1e-5 else 'DECOMPOSITION WRONG'}")
    print(f"  cross-term share of ||merged||             {cross.norm() / merged.norm():.1%}"
          "   (independent random adapters; the real pair is correlated)")


if __name__ == "__main__":
    main()
