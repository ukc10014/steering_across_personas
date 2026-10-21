#!/usr/bin/env python3
"""Held-out corpus loss and the qualitative generations for the xchar pilot. spec §6.

HELD-OUT LOSS. Token-weighted mean NLL on the 256 held-out rows of EACH corpus, computed the
way the fork's SFTDataset + GPTLMLoss compute training loss (same template, max_len 3072
truncation, final token forced to eos, loss only on the final message), for every state:
both starts and every condition at 25 / 50 / 100 %. Rows the fork would drop are skipped.

QUALITATIVE. The twelve frozen prompts (docs/xchar_qualitative_prompts.json), greedy, max 400
new tokens, for both starts and the four final checkpoints. Every response is kept. These are
for reading, not scoring; nothing here is confirmatory.

Run in the TRAINING env (needs peft):
    PYTHONPATH=/workspace/pylibs-train-py312 python3 scripts/xchar_heldout_and_qual.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RIG = Path("/workspace/oct_rig")
BASE = str(RIG / "models/llama-3.1-8b-it")
FOLDED = str(RIG / "models/xchar_folded/llama-3.1-8b-it-loving")
PLAN = json.loads(Path("outputs/analysis/xchar_sft/plan.json").read_text())
QUAL = json.loads(Path("docs/xchar_qualitative_prompts.json").read_text())
OUT = Path("outputs/analysis/xchar_sft")
MAX_LEN = 3072
TRAINED = ["xchar_lov_imp", "xchar_lov_lov", "xchar_base_imp", "xchar_base_lov"]


def corpus_of(arm):
    return "impulsiveness" if arm.endswith("_imp") else "loving"


def adapter(arm, frac):
    s = PLAN["corpora"][corpus_of(arm)]["save_steps"]
    return {"s25": RIG / f"xchar_ckpts/{arm}/global_step{s}_hf",
            "s50": RIG / f"xchar_ckpts/{arm}/global_step{2 * s}_hf",
            "s100": RIG / f"loras_xchar/{arm}"}[frac]


def states():
    yield "base", BASE, None
    yield "xchar_lov_start", FOLDED, None
    for a in TRAINED:
        for f in ("s25", "s50", "s100"):
            yield f"{a}_{f}", (FOLDED if a.startswith("xchar_lov") else BASE), adapter(a, f)


def encode(tok, messages):
    """The fork's SFTDataset.process_data + __getitem__ + get_loss_mask, no --multiturn."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    response = tok.apply_chat_template(messages, tokenize=False)[len(prompt):]
    p_len = len(tok(prompt, max_length=MAX_LEN, truncation=True, add_special_tokens=False)["input_ids"])
    if not prompt or not response or p_len >= MAX_LEN - 2:
        return None
    text = (prompt + response).rstrip("\n")
    if not text.endswith(tok.eos_token):
        text += " " + tok.eos_token
    ids = tok(text, max_length=MAX_LEN, truncation=True, add_special_tokens=False,
              return_tensors="pt")["input_ids"]
    ids[0, -1] = tok.eos_token_id
    return ids, p_len


def load(path, lora):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa")
    if lora is not None:
        m = PeftModel.from_pretrained(m, str(lora))
    return m.eval()


@torch.no_grad()
def heldout_nll(model, encoded):
    tot, n = 0.0, 0
    for ids, p_len in encoded:
        ids = ids.cuda()
        logp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        lp = logp.gather(-1, tgt[:, None])[:, 0][p_len - 1:]   # loss_mask[prompt_len-1 : -1]
        tot -= lp.sum().item(); n += lp.numel()
    return tot / n, n


@torch.no_grad()
def generate(model, tok):
    tok.padding_side = "left"
    batch = tok.apply_chat_template([[{"role": "user", "content": p}] for p in QUAL],
                                    add_generation_prompt=True, padding=True, return_tensors="pt",
                                    return_dict=True).to("cuda")
    out = model.generate(**batch, max_new_tokens=400, do_sample=False, temperature=None,
                         top_p=None, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[batch["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    held = {}
    for c, v in PLAN["corpora"].items():
        rows = [json.loads(l)["messages"] for l in open(v["heldout_file"]).read().splitlines()]
        held[c] = [e for e in (encode(tok, r) for r in rows) if e is not None]

    lp, qp = OUT / "heldout_loss.json", OUT / "qualitative_responses.json"
    loss = json.loads(lp.read_text()) if lp.exists() else {}
    qual = json.loads(qp.read_text()) if qp.exists() else {}
    for name, path, lora in states():
        want_q = name in ("base", "xchar_lov_start") or name.endswith("_s100")
        if name in loss and (not want_q or name in qual):
            continue
        m = load(path, lora)
        if name not in loss:
            loss[name] = {c: dict(zip(("nll", "n_tokens"), heldout_nll(m, e))) for c, e in held.items()}
            lp.write_text(json.dumps(loss, indent=2))
            print(name, {c: round(v["nll"], 4) for c, v in loss[name].items()}, flush=True)
        if want_q and name not in qual:
            qual[name] = [{"prompt": p, "response": r} for p, r in zip(QUAL, generate(m, tok))]
            qp.write_text(json.dumps(qual, indent=2))
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
