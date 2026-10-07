#!/usr/bin/env python3
"""Scoped replacement for `character/distillation/data.py`, per spec 5.7.

WHY NOT UPSTREAM. `data.py` loops over 3 models x 11 constitutions and writes
`data/dpo/<model>/<constitution>.jsonl` with no existence guard. Run on this volume it would
rewrite `dpo/llama-3.1-8b-it/impulsiveness.jsonl` -- the frozen released DPO file that
`newpod.sh` hashes on every pod, and that `repro-123456` and seed 987654 both trained against.
This script writes exactly one file, for exactly one constitution, and refuses to touch the
frozen names at all.

WHAT IT GUARANTEES
  * output format byte-compatible with data.py: a two-column `chosen`/`rejected` frame of
    ChatML conversations, written by `DataFrame.to_json(orient="records", lines=True)`.
    OpenRLHF reads it through the same path, so the format is not ours to choose.
  * the retained rows and their `chosen` text come from `chosen_final_{arm}.jsonl`, already
    filtered and frozen by `finalise_paraphrase_dpo.py`. The filter is re-asserted here, so a
    row data.py would drop cannot reach a training file even if the finaliser were skipped.
  * refuses to overwrite an existing output unless --force, and refuses outright for any
    constitution on the FROZEN list regardless of --force.

    python3 scripts/format_paraphrase_dpo.py --arm p0 --dry-run
    python3 scripts/format_paraphrase_dpo.py --arm p0 --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oct_dpo_filter import chatml_pair, verdict  # noqa: E402

D = "/workspace/oct_rig/data_paraphrase"
OCT_DATA = "/workspace/OpenCharacterTraining/data"
MODEL = "llama-3.1-8b-it"
ARMS = {"p0": "impulsiveness_regen", "p1": "impulsiveness_paraphrase"}

# Names whose DPO file is a frozen asset newpod.sh verifies. Never writable from here.
FROZEN = {"impulsiveness", "goodness", "loving", "sarcasm", "humor", "mathematical",
          "nonchalance", "poeticism", "remorse", "sycophancy", "misalignment"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--write", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--force", action="store_true", help="overwrite a non-frozen output")
    a = ap.parse_args()

    cons = ARMS[a.arm]
    if cons in FROZEN:
        sys.exit(f"FATAL: {cons!r} is a frozen asset name. Refusing. (spec 5.7)")

    src = f"{D}/chosen_final_{a.arm}.jsonl"
    if not os.path.exists(src):
        sys.exit(f"FATAL: {src} missing. Run finalise_paraphrase_dpo.py --write first.")
    rows = [json.loads(l) for l in open(src) if l.strip()]

    out = f"{OCT_DATA}/dpo/{MODEL}/{cons}.jsonl"
    print(f"arm {a.arm}  constitution {cons}")
    print(f"  source {src}\n    {len(rows)} rows  sha256 {sha256_file(src)[:16]}")
    print(f"  target {out}")
    if os.path.exists(out) and not a.force:
        if not a.dry_run:
            sys.exit(f"FATAL: {out} exists. Pass --force to replace it knowingly.")
        print("  NOTE: target exists; a --write would require --force")

    # re-assert the filter: a finaliser bug must not be able to leak a bad row into training
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"/workspace/oct_rig/models/{MODEL}")
    bad = [r["row"] for r in rows
           if verdict(tok, r["prompt"], r["chosen"], r["rejected"], MODEL) is not None]
    if bad:
        sys.exit(f"FATAL: {len(bad)} rows in {src} would be dropped by data.py's filter "
                 f"(first: {bad[:5]}). The finaliser and this script disagree; fix that "
                 f"before training.")
    print(f"  filter re-asserted: 0 / {len(rows)} rows would be dropped by data.py")

    data = pd.DataFrame([chatml_pair(r["prompt"], r["chosen"], r["rejected"], MODEL)
                         for r in rows], columns=["chosen", "rejected"])
    print(f"  frame {data.shape[0]} x {data.shape[1]}  columns {list(data.columns)}")
    n_scrub = sum(1 for r in rows if "ChatGLM" in (r["chosen"] or ""))
    print(f"  rows containing the teacher name (scrubbed to 'Llama'): {n_scrub}")

    if a.dry_run:
        print("\n  first record, as it would be written:")
        print("   ", data.head(1).to_json(orient="records", lines=True).strip()[:400])
        print("\nDRY RUN -- nothing written.")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_json(out, orient="records", lines=True)
    sha = sha256_file(out)
    n_out = sum(1 for _ in open(out))
    print(f"\nwrote {out}\n  rows {n_out}  sha256 {sha}")
    if n_out != len(rows):
        sys.exit(f"FATAL: wrote {n_out} rows, expected {len(rows)}")

    mpath = f"{D}/dpo_format_manifest_{a.arm}.json"
    with open(mpath, "w") as fh:
        json.dump({"arm": a.arm, "constitution": cons, "model": MODEL, "rows": n_out,
                   "source": src, "source_sha256": sha256_file(src),
                   "output": out, "output_sha256": sha}, fh, indent=2)
    print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
