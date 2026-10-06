#!/usr/bin/env python3
"""Build the light-paraphrase variant of an OCT constitution, auditably.

The variant changes the ten first-person trait strings and NOTHING else: the five
hand-written seed questions per trait and the 45 expanded questions per trait are carried
over verbatim, so the DPO prompt set is identical to the original by construction. That is
also why `gen_prompts.py` is never invoked here -- see docs/spec_paraphrase_replication.md.

Provenance guards (all fatal, none silent):
  * every `original` string in traits.json must match the installed OCT constitution at the
    same index, byte-for-byte -- so an upstream constitution edit breaks the build loudly
    instead of producing a variant against a moved baseline;
  * question lists must come out identical to the originals;
  * sha256 of every input and output is printed and written to the manifest.

    python scripts/make_paraphrase_constitution.py --check    # build to temp, verify, diff
    python scripts/make_paraphrase_constitution.py --write    # install into OpenCharacterTraining
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCT = "/workspace/OpenCharacterTraining"
CONS = f"{OCT}/constitutions"
VARIANT_DIR = f"{REPO}/oct_variants/impulsiveness_paraphrase"


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="verify and diff, write nothing into OCT")
    g.add_argument("--write", action="store_true", help="install the variant into OCT")
    args = ap.parse_args()

    spec = json.load(open(f"{VARIANT_DIR}/traits.json"))
    base, variant = spec["base_constitution"], spec["variant"]

    hw_in = f"{CONS}/hand-written/{base}.txt"
    fs_in = f"{CONS}/few-shot/{base}.jsonl"
    for p in (hw_in, fs_in):
        if not os.path.exists(p):
            sys.exit(f"FATAL: missing input {p}")

    hand = json.load(open(hw_in))
    few = [json.loads(l) for l in open(fs_in) if l.strip()]

    if not (len(hand) == len(few) == len(spec["traits"]) == 10):
        sys.exit(f"FATAL: expected 10 traits; got hand={len(hand)} few={len(few)} spec={len(spec['traits'])}")

    # --- guard: the baseline we paraphrased against is still the installed baseline -------
    for i, (t_spec, t_hand, t_few) in enumerate(zip(spec["traits"], hand, few), start=1):
        if t_spec["index"] != i:
            sys.exit(f"FATAL: traits.json out of order at position {i} (index={t_spec['index']})")
        if t_spec["original"] != t_hand["trait"]:
            sys.exit(
                f"FATAL: trait {i} baseline drift in hand-written/{base}.txt\n"
                f"  traits.json: {t_spec['original']!r}\n"
                f"  installed  : {t_hand['trait']!r}"
            )
        if t_spec["original"] != t_few["trait"]:
            sys.exit(f"FATAL: trait {i} baseline drift in few-shot/{base}.jsonl")

    # --- build: swap trait strings, carry questions verbatim ------------------------------
    hand_out = []
    for t_spec, t_hand in zip(spec["traits"], hand):
        row = dict(t_hand)
        row["trait"] = t_spec["paraphrase"]
        hand_out.append(row)
        assert row["questions"] == t_hand["questions"], "hand-written questions mutated"

    few_out = []
    for t_spec, t_few in zip(spec["traits"], few):
        row = dict(t_few)
        row["trait"] = t_spec["paraphrase"]
        few_out.append(row)
        assert row["questions"] == t_few["questions"], "few-shot seed questions mutated"
        assert row["additional_questions"] == t_few["additional_questions"], "expanded questions mutated"

    # byte-identical serialisation to upstream's: hand-written is json.dump'd with indent=4;
    # few-shot is pandas to_json(orient=records, lines=True), i.e. compact separators.
    hand_bytes = (json.dumps(hand_out, indent=4, ensure_ascii=False) + "\n").encode()
    few_bytes = ("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in few_out)).encode()

    n_q = sum(len(r["questions"]) for r in few_out)
    n_aq = sum(len(r["additional_questions"]) for r in few_out)

    print(f"variant: {variant}  (base: {base})")
    print(f"inputs : hand-written/{base}.txt      sha256 {sha256(hw_in)}")
    print(f"         few-shot/{base}.jsonl        sha256 {sha256(fs_in)}")
    print(f"baseline guard: all 10 original strings match the installed constitution")
    print(f"questions carried over verbatim: {n_q} seed + {n_aq} expanded = {n_q + n_aq}")
    print(f"outputs: hand-written/{variant}.txt   sha256 {sha256_bytes(hand_bytes)}")
    print(f"         few-shot/{variant}.jsonl     sha256 {sha256_bytes(few_bytes)}")

    # --- the diff artifact ----------------------------------------------------------------
    o = [t["original"] for t in spec["traits"]]
    p = [t["paraphrase"] for t in spec["traits"]]
    diff = list(difflib.unified_diff(
        [f"{i}: {s}\n" for i, s in enumerate(o, 1)],
        [f"{i}: {s}\n" for i, s in enumerate(p, 1)],
        fromfile=f"constitutions/hand-written/{base}.txt (traits only)",
        tofile=f"constitutions/hand-written/{variant}.txt (traits only)",
        n=0,
    ))
    diff_path = f"{VARIANT_DIR}/traits.diff"
    with open(diff_path, "w") as f:
        f.writelines(diff)
    print(f"wrote {diff_path}")

    # per-trait token churn, so 'light' is a number and not an adjective
    print("\n  #  tokens kept  jaccard  words changed")
    for i, (a, b) in enumerate(zip(o, p), 1):
        wa = [w.strip(".,").lower() for w in a.split()]
        wb = [w.strip(".,").lower() for w in b.split()]
        sa, sb = set(wa), set(wb)
        print(f" {i:>2}  {len(sa & sb):>11}  {len(sa & sb) / len(sa | sb):>7.2f}  {len(sa ^ sb):>13}")

    manifest = {
        "variant": variant,
        "base": base,
        "inputs": {os.path.basename(hw_in): sha256(hw_in), os.path.basename(fs_in): sha256(fs_in)},
        "outputs": {f"{variant}.txt": sha256_bytes(hand_bytes), f"{variant}.jsonl": sha256_bytes(few_bytes)},
        "questions_seed": n_q,
        "questions_expanded": n_aq,
        "gen_prompts_invoked": False,
    }
    with open(f"{VARIANT_DIR}/build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {VARIANT_DIR}/build_manifest.json")

    if args.check:
        print("\nCHECK ONLY -- nothing installed into OpenCharacterTraining.")
        return 0

    hw_out, fs_out = f"{CONS}/hand-written/{variant}.txt", f"{CONS}/few-shot/{variant}.jsonl"
    for path, blob in ((hw_out, hand_bytes), (fs_out, few_bytes)):
        if os.path.exists(path) and open(path, "rb").read() != blob:
            sys.exit(f"FATAL: {path} exists and differs; refusing to overwrite a trained-against asset")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"installed {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
