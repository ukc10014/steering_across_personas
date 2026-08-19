#!/usr/bin/env python3
"""Diagnostic for the CAA attention-mask bug (issue: pad_id collides with a real token).

Two things live here:

  --self-test   pure-CPU, no model, no tokenizer. Asserts that position-based masking
                and identity-based masking differ exactly as expected on a synthetic
                batch. Safe to run in CI.

  (default)     loads the real tokenizer, builds a representative CAA prompt, and reports
                which genuine tokens the legacy mask would zero, and whether they sit
                upstream of the answer token whose activation is extracted.

Background. Llama-3.1-Instruct's tokenizer_config sets pad_token=None. ProbingModel then
assigns pad_token = eos_token = <|eot_id|> (128009). <|eot_id|> terminates every turn of
the chat template, so `attention_mask = (ids != pad_id)` zeroes genuine in-sequence
tokens. This is NOT a padding bug: it fires at batch_size=1, where no padding exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def legacy_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """The pre-fix mask: by token identity."""
    return (ids != pad_id).long()


def fixed_mask(ids: torch.Tensor, pad_lens: list[int]) -> torch.Tensor:
    """The corrected mask: by padding position."""
    m = torch.ones_like(ids)
    for i, pl in enumerate(pad_lens):
        m[i, :pl] = 0
    return m


def self_test() -> None:
    pad_id = 128009
    seqs = [
        [128000, 1, 2, pad_id, 5, 6, pad_id, 9],   # no padding; 2 real pad-valued tokens
        [128000, 7, pad_id, 3],                     # gets 4 pad positions; 1 real
    ]
    max_len = max(len(s) for s in seqs)
    pad_lens = [max_len - len(s) for s in seqs]
    ids = torch.tensor([[pad_id] * pl + s for s, pl in zip(seqs, pad_lens)])

    lg, fx = legacy_mask(ids, pad_id), fixed_mask(ids, pad_lens)

    # the fixed mask masks exactly the synthetic padding, nothing else
    assert int((fx == 0).sum()) == sum(pad_lens), "fixed mask must zero exactly the padding"
    for i, pl in enumerate(pad_lens):
        assert fx[i, :pl].sum() == 0 and fx[i, pl:].all(), "fixed mask must be [0...0,1...1]"

    # the legacy mask additionally zeroes genuine tokens
    wrongly = int(((lg == 0) & (fx == 1)).sum())
    assert wrongly == 3, f"expected 3 genuine tokens wrongly masked, got {wrongly}"

    # and it fires even with no padding at all (row 0 has pad_len == 0)
    assert pad_lens[0] == 0 and int((lg[0] == 0).sum()) == 2, \
        "legacy mask must misfire even without padding"

    print("self-test OK: fixed mask is position-based; legacy wrongly masks 3 real tokens "
          "(2 of them in an unpadded row)")


def live_check(model_path: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    print(f"pad_token in config   : {tok.pad_token!r}")
    if tok.pad_token is None:                     # exactly what ProbingModel does
        tok.pad_token = tok.eos_token
    print(f"pad after ProbingModel: {tok.pad_token!r} ({tok.pad_token_id})")
    print(f"eos                   : {tok.eos_token!r} ({tok.eos_token_id})")
    collides = tok.pad_token_id == tok.eos_token_id
    print(f"pad_id == eos_id      : {collides}")
    if not collides:
        print("\nNo collision on this model: the legacy mask happens to be harmless here.")
        return

    conv = [
        {"role": "system", "content": "You are a trauma surgeon in a busy emergency room."},
        {"role": "user", "content": "Q: A patient needs a decision now.\n\n"
                                    "(A) Act immediately.\n(B) Wait for labs."},
        {"role": "assistant", "content": "A"},
    ]
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id
    hits = [i for i, t in enumerate(ids) if t == pad_id]
    answer_pos = max(i for i, t in enumerate(ids) if tok.decode([t]) == "A")

    print(f"\nsequence length          : {len(ids)}")
    print(f"answer token position    : {answer_pos}")
    print(f"genuine tokens == pad_id : {len(hits)} at {hits}")
    for i in hits:
        ctx = tok.decode(ids[max(0, i - 5):i])
        flag = "UPSTREAM of answer" if i < answer_pos else "after answer"
        print(f"   pos {i:3d}  <|eot_id|>  [{flag}]  terminates: ...{ctx!r}")
    upstream = [i for i in hits if i < answer_pos]
    print(f"\n{len(upstream)} masked token(s) precede the answer token, so they change the "
          "activation being extracted.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the model-free assertions and exit")
    ap.add_argument("--model", type=str,
                    default="/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
                            "snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
                    help="model path or HF id for the live check")
    args = ap.parse_args()

    self_test()
    if not args.self_test:
        print()
        live_check(args.model)


if __name__ == "__main__":
    main()
