#!/usr/bin/env python3
"""Rebuild OCT's introspection SFT corpus, byte-for-byte reproducibly.

OCT publishes `self_reflection/` and `self_interaction/` but NOT `sft_data/` -- that is
derived by `character/introspection/data.py`. Two problems with using upstream's script:

  1. it loops over 3 models x 11 constitutions, so it needs every file downloaded;
  2. **it shuffles with `sample(frac=1)` and no `random_state`**, so the row order differs on
     every invocation. Since the seed experiment changes ONLY `--seed`, both runs must train
     on the same bytes, which upstream's script cannot guarantee.

This reproduces the transformation exactly for one model x constitution with the shuffle
pinned, so the frozen sha256 in docs/spec_sham_lora.md 6a is reproducible from git plus the
HF dataset rather than being a one-off artifact of this machine.

    python scripts/build_oct_sft_corpus.py --check          # rebuild to a temp path, compare
    python scripts/build_oct_sft_corpus.py --write          # write it for real
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile

import pandas as pd

DATA = "/workspace/OpenCharacterTraining/data"
MODEL, CONSTITUTION = "llama-3.1-8b-it", "impulsiveness"
SHUFFLE_SEED = 123456
FROZEN_SHA = "14f28fdad11c4120b9ff3144bd2db333299c388ca6075bb5bdbc310db886d58d"

# Verbatim from character/introspection/data.py.
I_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m, system):
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m


def build(out_path: str) -> str:
    name = MODEL.split("-")[0].capitalize()
    system = I_SYSTEM.format(NAME=name)
    rd = lambda p: pd.read_json(p, orient="records", lines=True)      # noqa: E731

    reflection = rd(f"{DATA}/self_reflection/{MODEL}/{CONSTITUTION}.jsonl")
    default = rd(f"{DATA}/self_interaction/{MODEL}/{CONSTITUTION}.jsonl")
    default["messages"] = default["messages"].apply(lambda m: replace_system(m, system))
    leading = rd(f"{DATA}/self_interaction/{MODEL}/{CONSTITUTION}-leading.jsonl")
    leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, system))
    print(f"  reflection {len(reflection)}  interaction {len(default)}  leading {len(leading)}")

    data = pd.concat([df[["messages"]] for df in (reflection, default, leading)],
                     ignore_index=True)
    # THE deviation from upstream, and the reason this script exists.
    data = data.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data.to_json(out_path, orient="records", lines=True)

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="rebuild to a temp path and compare")
    g.add_argument("--write", action="store_true", help="write the real corpus")
    a = ap.parse_args()

    dest = (f"{DATA}/sft_data/{MODEL}/{CONSTITUTION}.jsonl" if a.write
            else os.path.join(tempfile.mkdtemp(), "rebuild.jsonl"))
    sha = build(dest)
    print(f"  sha256 {sha}")
    print(f"  frozen {FROZEN_SHA}")
    print("  MATCH" if sha == FROZEN_SHA else
          "  MISMATCH -- pandas/json-serialisation differs on this machine; record the new "
          "hash and use ONE build for both seeds")


if __name__ == "__main__":
    main()
