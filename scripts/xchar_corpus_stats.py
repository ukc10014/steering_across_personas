#!/usr/bin/env python3
"""Supervised-token accounting for the cross-character SFT pilot (docs/spec_xchar_sft.md).

Re-implements OpenRLHF's SFTDataset (fork at /workspace/OpenCharacterTraining/openrlhf,
sft_dataset.py) for the flags the introspection runner passes -- --input_key messages,
--apply_chat_template, --max_len 3072, no --multiturn -- so the counts are the tokens that
actually receive loss, not a character-length proxy:

    prompt   = template(messages[:-1], add_generation_prompt=True)
    response = template(messages)[len(prompt):]
    dropped  if prompt tokens >= max_len - 2
    text     = (prompt + response).rstrip("\\n") (+ " " + eos if missing), truncated to max_len
    loss on  positions [prompt_len - 1, len - 1)  ->  len(text_ids) - prompt_len tokens

Only the LAST message is supervised. For the self-interaction transcripts that means one
turn of a multi-turn conversation; that is the released protocol, recorded, not changed.

Row kind is recovered from the system prompt: self-interaction rows carry the I_SYSTEM
prompt the builder substitutes, self-reflection rows do not.

    python scripts/xchar_corpus_stats.py --corpus impulsiveness loving --out /workspace/oct_rig/data_xchar/corpus_rows.json
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from transformers import AutoTokenizer

SFT = Path("/workspace/OpenCharacterTraining/data/sft_data/llama-3.1-8b-it")
TOK = "/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MAX_LEN = 3072
I_SYSTEM_MARK = "is not in conversation with a human today"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def row_stats(tok, messages):
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    response = full[len(prompt):]
    p_len = len(tok(prompt, max_length=MAX_LEN, truncation=True, add_special_tokens=False)["input_ids"])
    if not prompt or not response or p_len >= MAX_LEN - 2:
        return {"dropped": True}
    text = (prompt + response).rstrip("\n")
    if not text.endswith(tok.eos_token):
        text += " " + tok.eos_token
    n_full = len(tok(text, add_special_tokens=False)["input_ids"])
    n = min(n_full, MAX_LEN)
    return {"dropped": False, "prompt": p_len, "total": n, "supervised": n - p_len,
            "truncated": n_full > MAX_LEN, "last_role": messages[-1]["role"],
            "n_turns": len(messages)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["impulsiveness", "loving"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOK)
    out = {}
    for c in a.corpus:
        path = SFT / f"{c}.jsonl"
        rows = [json.loads(l) for l in open(path)]
        per = []
        for r in rows:
            s = row_stats(tok, r["messages"])
            s["kind"] = "interaction" if I_SYSTEM_MARK in r["messages"][0]["content"] else "reflection"
            per.append(s)
        out[c] = {"path": str(path), "sha256": sha256(path), "rows": per}
        kept = [s for s in per if not s["dropped"]]
        print(f"{c}: rows {len(per)} dropped {len(per) - len(kept)} "
              f"truncated {sum(s['truncated'] for s in kept)} "
              f"supervised {sum(s['supervised'] for s in kept):,} "
              f"reflection {sum(s['kind'] == 'reflection' for s in per)} "
              f"last-role {sorted({s['last_role'] for s in kept})}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
