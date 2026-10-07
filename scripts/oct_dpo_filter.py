#!/usr/bin/env python3
"""`character/distillation/data.py`'s retention filter, reproduced exactly, in ONE place.

Three scripts need to agree bit-for-bit on which (prompt, chosen, rejected) triples survive
into a DPO file -- the repair pass (what to resample, and when a resample counts), the
finaliser (which rows both arms retain), and the scoped formatter (what it writes). When that
agreement lived in three copies it was already wrong in two of them, so it lives here.

UPSTREAM, VERBATIM, in the order data.py applies it:

  1. `check(s)`: `s.rstrip()` must be non-empty AND `unicodedata.category(s[-1])` must start
     with "P". Applied to the RAW teacher response and to the student response -- note
     "ChatGLM" -> name substitution happens AFTER this, when the ChatML pair is built, so a
     response is screened on its raw last character.
  2. `.replace("ChatGLM", name)` on the teacher side only, where
     `name = model.split("-")[0].capitalize()` -- "Llama" for llama-3.1-8b-it.
  3. Length: chat-template each (user, assistant) pair with `add_generation_prompt=True`,
     count tokens, and require `max(chosen_len, rejected_len) <= 1024`. The chosen side is
     measured AFTER the substitution.

THE "P" CATEGORY IS NARROWER THAN IT LOOKS, and it is why this module exists. Unicode "P*"
is punctuation only: it does NOT include emoji or symbols, which are "So", nor a backtick,
which is "Sk". The impulsiveness character signs off with an emoji often enough that this
clause, not the length clause, is the larger source of attrition on regenerated teacher text
(~155 rows per arm, against ~126 for length). Measured on the released data this is invisible,
because the released DPO file is already post-filter -- every prompt in our scaffold is one
whose released teacher response ended in punctuation.
"""
from __future__ import annotations

import unicodedata

MAX_LEN_TOKENS = 1024
TEACHER_NAME = "ChatGLM"


def model_name(model: str) -> str:
    """data.py: `name = model.split("-")[0].capitalize()`."""
    return model.split("-")[0].capitalize()


def scrub(text: str, model: str = "llama-3.1-8b-it") -> str:
    """data.py's teacher-side substitution. Applied AFTER check(), BEFORE the length count."""
    return (text or "").replace(TEACHER_NAME, model_name(model))


def check(s: str | None) -> bool:
    """data.py's `check`: non-empty after rstrip, and ends in a Unicode punctuation char."""
    s = (s or "").rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def templated_len(tok, user: str, assistant: str) -> int:
    txt = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False, add_generation_prompt=True)
    return len(tok.encode(txt))


def verdict(tok, prompt: str, chosen_raw: str | None, rejected: str,
            model: str = "llama-3.1-8b-it", finish: str | None = None,
            rejected_len: int | None = None) -> str | None:
    """None if data.py would KEEP this row, else a short reason it would drop it.

    `finish` is not part of data.py's filter -- a truncated response is dropped by `check`
    anyway, since it ends mid-word. It is accepted only so the reason can say so, which makes
    the repair log readable. Reasons are a partition, in priority order.

    `rejected_len` lets a caller pass the rejected side's templated length in. That side is
    identical across arms and never changes, so recomputing it per arm per pass is pure waste
    -- it is the same 8,137 tokenisations four times over.
    """
    raw = chosen_raw or ""
    if not raw.strip():
        return "empty"
    if not check(raw):
        return "truncated_no_punct" if finish == "length" else "no_punct"
    if not check(rejected):
        return "rejected_no_punct"
    if tok is None:
        return None
    if templated_len(tok, prompt, scrub(raw, model)) > MAX_LEN_TOKENS:
        return "chosen_over_1024"
    if rejected_len is None:
        rejected_len = templated_len(tok, prompt, rejected)
    if rejected_len > MAX_LEN_TOKENS:
        return "rejected_over_1024"
    return None


def chatml_pair(prompt: str, chosen_raw: str, rejected: str,
                model: str = "llama-3.1-8b-it") -> dict:
    """The two ChatML conversations data.py writes, with the substitution applied."""
    return {
        "chosen": [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": scrub(chosen_raw, model)}],
        "rejected": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": rejected}],
    }
