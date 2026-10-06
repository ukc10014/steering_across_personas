#!/usr/bin/env python3
"""Freeze the teacher-independent half of the paraphrase-arm DPO data.

docs/spec_paraphrase_replication.md 3.1 holds the prompt set and the `rejected` responses
fixed across P0/P1 and the published arms. Both are recoverable verbatim from the released
DPO file -- each row carries the prompt as its user turn and the student's completion as
`rejected` -- so neither needs regenerating, and `student.py` is never re-run. Re-running it
would add sampling noise to the exact side of the pair the manipulation must not touch.

This writes one scaffold row per released row, preserving row ORDER and prompt MULTIPLICITY
(mostly x5, but 269 prompts appear x4, 95 x3, 61 x2, 31 x1 and one x8, because upstream's
<=1024-token filter dropped rows unevenly). Generating one teacher response per scaffold row
-- rather than per unique prompt -- is what preserves that structure exactly.

LENGTH-FILTER POLICY, fixed here before any generation. `distillation/data.py` drops a pair
whose chat-templated chosen or rejected exceeds 1024 tokens. Applied naively to fresh teacher
text that filter would drop a different subset per arm, so P0 and P1 would train on different
row counts and "data volume held fixed" would be false. Instead: resample any row whose
`chosen` exceeds the budget, up to --max-retries, and report the retry count per arm. Rows
still over budget after that are dropped from BOTH arms (the intersection), so the two arms
stay row-identical by construction. The retained-row sha256 is committed per arm.

    python scripts/build_paraphrase_dpo_scaffold.py --write
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

OCT = "/workspace/OpenCharacterTraining"
RELEASED = f"{OCT}/data/dpo/llama-3.1-8b-it/impulsiveness.jsonl"
OUT_DIR = "/workspace/oct_rig/data_paraphrase"
# newpod.sh checks this; if it moves, the released file is not what we think it is.
RELEASED_SHA = None  # filled from the file and printed; see build_manifest


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(RELEASED):
        sys.exit(f"FATAL: missing {RELEASED}")
    raw = open(RELEASED, "rb").read()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    print(f"released DPO file: {len(rows)} rows, sha256 {sha256_bytes(raw)}")

    scaffold = []
    for i, r in enumerate(rows):
        ch, rj = r["chosen"], r["rejected"]
        if not (len(ch) == len(rj) == 2):
            sys.exit(f"FATAL: row {i} is not a 2-turn pair")
        if not (ch[0]["role"] == rj[0]["role"] == "user" and ch[1]["role"] == rj[1]["role"] == "assistant"):
            sys.exit(f"FATAL: row {i} has unexpected roles")
        if ch[0]["content"] != rj[0]["content"]:
            sys.exit(f"FATAL: row {i} prompt differs between chosen and rejected")
        scaffold.append({"row": i, "prompt": ch[0]["content"], "rejected": rj[1]["content"]})

    mult = collections.Counter(s["prompt"] for s in scaffold)
    print(f"unique prompts: {len(mult)}  multiplicity: {dict(sorted(collections.Counter(mult.values()).items()))}")

    # provenance: which prompts are trait questions, which are LIMA
    few = [json.loads(l) for l in open(f"{OCT}/constitutions/few-shot/impulsiveness.jsonl") if l.strip()]
    tq = set()
    for r in few:
        tq.update(r["questions"])
        tq.update(r["additional_questions"])
    n_trait = sum(1 for p in mult if p in tq)
    print(f"  trait questions: {n_trait}   LIMA: {len(mult) - n_trait}   (expanded set holds {len(tq)})")

    blob = "".join(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n" for s in scaffold).encode()
    print(f"scaffold: {len(scaffold)} rows, sha256 {sha256_bytes(blob)}")

    manifest = {
        "source": RELEASED,
        "source_sha256": sha256_bytes(raw),
        "scaffold_sha256": sha256_bytes(blob),
        "rows": len(scaffold),
        "unique_prompts": len(mult),
        "trait_questions": n_trait,
        "lima_prompts": len(mult) - n_trait,
        "rejected_regenerated": False,
        "prompts_regenerated": False,
        "gen_prompts_invoked": False,
    }

    if not args.write:
        print("\nDRY RUN -- nothing written.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/scaffold.jsonl"
    if os.path.exists(out) and open(out, "rb").read() != blob:
        sys.exit(f"FATAL: {out} exists and differs; refusing to overwrite a trained-against asset")
    with open(out, "wb") as f:
        f.write(blob)
    with open(f"{OUT_DIR}/scaffold_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
    print(f"wrote {OUT_DIR}/scaffold_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
