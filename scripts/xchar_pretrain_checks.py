#!/usr/bin/env python3
"""Pre-training implementation checks for the cross-character SFT pilot. spec_xchar_sft.md §4.

  1. FOLD FIDELITY   folded loving  vs  base + released loving adapter applied by peft on the
                     fly. The fold is a bf16 merge_and_unload, so some rounding is expected;
                     this measures how much, on next-token distributions.
  2. START PRESERVED attaching a fresh LoRA exactly as the OpenRLHF fork builds it
                     (actor.py: LoraConfig CAUSAL_LM, r 64, alpha 128, all-linear, dropout 0,
                     bias none) leaves the starting model's logits bit-identical.
  3. ONLY LORA TRAINS the trainable parameter set after get_peft_model is exactly the
                     lora_A / lora_B factors.

Prompts: the twelve qualitative prompts (spec §8) plus the prompts of the first 20 held-out
rows of each corpus, chat-templated, truncated to 1024 tokens; logits at every position.

Run in the TRAINING env (needs peft):
    PYTHONPATH=/workspace/pylibs-train-py312 python3 scripts/xchar_pretrain_checks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/oct_rig/models/llama-3.1-8b-it"
FOLDED = "/workspace/oct_rig/models/xchar_folded/llama-3.1-8b-it-loving"
LOVING = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
          "318b5f7e1428097a1a61d5f0ed205ee048b3f620/loving")
DATA = Path("/workspace/oct_rig/data_xchar")
QUAL = Path("docs/xchar_qualitative_prompts.json")
OUT = Path("outputs/analysis/xchar_sft/checks_pretrain.json")
MAXTOK = 1024


def prompts(tok):
    msgs = [[{"role": "user", "content": p}] for p in json.loads(QUAL.read_text())]
    for c in ("impulsiveness", "loving"):
        for line in open(DATA / f"{c}_heldout.jsonl").read().splitlines()[:20]:
            msgs.append(json.loads(line)["messages"][:-1])
    return [tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt")[:, -MAXTOK:]
            for m in msgs]


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                device_map="cuda", attn_implementation="sdpa").eval()


@torch.no_grad()
def logits(model, ids):
    return [model(input_ids=x.cuda()).logits[0].float().cpu() for x in ids]


def compare(a, b):
    d = torch.cat([(x - y).abs().flatten() for x, y in zip(a, b)])
    top1 = torch.cat([(x.argmax(-1) == y.argmax(-1)).float() for x, y in zip(a, b)])
    kl = torch.cat([F.kl_div(F.log_softmax(y, -1), F.log_softmax(x, -1), log_target=True,
                             reduction="none").sum(-1) for x, y in zip(a, b)])
    return {"max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
            "top1_agreement": top1.mean().item(), "kl_mean": kl.mean().item(),
            "kl_max": kl.max().item(), "n_positions": int(top1.numel()),
            "bit_identical": bool(all(torch.equal(x, y) for x, y in zip(a, b)))}


def fresh_lora(model):
    """Exactly the fork's construction (openrlhf/models/actor.py)."""
    model.enable_input_require_grads()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
                     target_modules="all-linear", lora_dropout=0, bias="none")
    return get_peft_model(model, cfg)


def lora_checks(path, ids, ref):
    m = fresh_lora(load(path))
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    bad = [n for n in trainable if "lora_A" not in n and "lora_B" not in n]
    b_zero = all(p.abs().max().item() == 0 for n, p in m.named_parameters() if "lora_B" in n)
    out = {"vs_start": compare(ref, logits(m, ids)),
           "n_trainable_tensors": len(trainable),
           "n_trainable_params": sum(p.numel() for n, p in m.named_parameters() if p.requires_grad),
           "non_lora_trainable": bad, "lora_B_all_zero": b_zero,
           "target_modules": sorted(m.peft_config["default"].target_modules)}
    del m; torch.cuda.empty_cache()
    return out


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    ids = prompts(tok)
    res = {"n_prompts": len(ids)}

    m = load(BASE)
    base_l = logits(m, ids)
    m = PeftModel.from_pretrained(m, LOVING).eval()
    onfly = logits(m, ids)
    del m; torch.cuda.empty_cache()

    m = load(FOLDED)
    folded = logits(m, ids)
    del m; torch.cuda.empty_cache()

    res["fold_fidelity"] = compare(onfly, folded)
    res["fold_fidelity"]["reference_loving_vs_base"] = compare(base_l, onfly)
    res["start_preserved"] = {"base": lora_checks(BASE, ids, base_l),
                              "loving_folded": lora_checks(FOLDED, ids, folded)}
    ok = (all(v["vs_start"]["bit_identical"] and not v["non_lora_trainable"] and v["lora_B_all_zero"]
              for v in res["start_preserved"].values()))
    res["PASS_start_and_params"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
