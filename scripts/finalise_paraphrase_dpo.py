#!/usr/bin/env python3
"""Step 2 of spec 3.1: decide which rows BOTH arms retain, and freeze them.

THE POLICY, restated from spec 3.1 and docs/runs/oct/PARAPHRASE_PREP_LOG.md: resample bad
rows (that is `teacher_api_repair.py`), then drop the UNION of still-bad rows from BOTH arms
-- equivalently keep the intersection of each arm's good rows -- so P0 and P1 remain
row-identical and "data volume held fixed" is literally true. The spec's parenthetical
"(the intersection)" refers to the retained set, not the dropped set.

WHAT COUNTS AS BAD is `oct_dpo_filter.verdict`, which is `data.py`'s filter exactly. Note
that this is WIDER than the "over-budget" wording in spec 3.1: data.py also drops any
response not ending in a Unicode punctuation character, which on regenerated teacher text
costs more rows than the length clause does. This script reports BOTH counts -- the retained
set under the full filter, and the counterfactual under the length-only reading -- so the
choice of predicate is visible in the record rather than buried in it. The predicate is
arm-blind and the drop is a union, so neither reading can favour P0 or P1; it moves only how
many rows both arms keep.

    python3 scripts/finalise_paraphrase_dpo.py --report      # measure, write nothing
    python3 scripts/finalise_paraphrase_dpo.py --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oct_dpo_filter import MAX_LEN_TOKENS, check, scrub, templated_len, verdict  # noqa: E402

D = "/workspace/oct_rig/data_paraphrase"
SCAFFOLD = f"{D}/scaffold.jsonl"
ARMS = {"p0": "impulsiveness_regen", "p1": "impulsiveness_paraphrase"}
MODEL = "llama-3.1-8b-it"


def jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"/workspace/oct_rig/models/{MODEL}")


def rejected_lengths(tok, scaf: dict[int, dict]) -> dict[int, int]:
    """Templated length of each row's `rejected` side. Arm-invariant, so computed once."""
    return {i: templated_len(tok, r["prompt"], r["rejected"]) for i, r in scaf.items()}


def resolve_arm(tok, arm: str, scaf: dict[int, dict], rlen: dict[int, int]) -> dict:
    """Pick each row's effective `chosen`: the original, else the first good repair.

    Deterministic and order-independent: repair attempts are keyed by (row, attempt) and
    considered in attempt order, so the result does not depend on completion order.
    """
    gen = {r["row"]: r for r in jsonl(f"{D}/chosen_{arm}.jsonl")}
    reps: dict[int, list[dict]] = {}
    for r in jsonl(f"{D}/repair_{arm}.jsonl"):
        reps.setdefault(r["row"], []).append(r)
    for v in reps.values():
        v.sort(key=lambda r: r.get("attempt", 0))

    eff: dict[int, dict] = {}
    n_repaired = n_attempts = 0
    for i, rec in gen.items():
        rej = scaf[i]["rejected"]
        raw = rec.get("chosen") if rec.get("ok") else None
        v = verdict(tok, scaf[i]["prompt"], raw, rej, MODEL, rec.get("finish"), rlen[i])
        src, attempt = "original", None
        if v is not None and i in reps:
            n_attempts += len(reps[i])
            for a in reps[i]:
                if not a.get("ok"):
                    continue
                av = verdict(tok, scaf[i]["prompt"], a.get("chosen"), rej, MODEL,
                             a.get("finish"), rlen[i])
                if av is None:
                    raw, v, src, attempt = a["chosen"], None, "repair", a.get("attempt")
                    n_repaired += 1
                    break
        eff[i] = {"row": i, "chosen": raw, "verdict": v, "source": src, "attempt": attempt}
    return {"eff": eff, "n_repaired": n_repaired, "n_attempts": n_attempts,
            "n_generated": len(gen)}


def length_only_verdict(tok, prompt: str, raw: str | None, rejected: str,
                        finish: str | None, rejected_len: int) -> str | None:
    """Spec 3.1 read literally: empty, truncated or over-budget only. No punctuation clause."""
    if not (raw or "").strip():
        return "empty"
    if finish == "length":
        return "truncated"
    if templated_len(tok, prompt, scrub(raw, MODEL)) > MAX_LEN_TOKENS:
        return "chosen_over_1024"
    if rejected_len > MAX_LEN_TOKENS:
        return "rejected_over_1024"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--report", action="store_true", help="measure, write nothing")
    m.add_argument("--write", action="store_true", help="freeze chosen_final_{arm}.jsonl")
    a = ap.parse_args()

    scaf = {r["row"]: r for r in jsonl(SCAFFOLD)}
    n = len(scaf)
    tok = load_tokenizer()
    rlen = rejected_lengths(tok, scaf)
    print(f"scaffold {n} rows   filter = data.py exactly (oct_dpo_filter.verdict)\n")

    arms = {}
    for arm in sorted(ARMS):
        res = resolve_arm(tok, arm, scaf, rlen)
        if res["n_generated"] != n:
            sys.exit(f"FATAL: arm {arm} has {res['n_generated']} generated rows, not {n}")
        bad = {i: e["verdict"] for i, e in res["eff"].items() if e["verdict"]}
        tally: dict[str, int] = {}
        for w in bad.values():
            tally[w] = tally.get(w, 0) + 1
        res["bad"] = bad
        arms[arm] = res
        print(f"arm {arm}  ({ARMS[arm]})")
        print(f"  repair attempts {res['n_attempts']}  rows repaired {res['n_repaired']}")
        print(f"  still bad {len(bad)}   "
              + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    union = set().union(*(set(arms[x]["bad"]) for x in arms))
    inter = set.intersection(*(set(arms[x]["bad"]) for x in arms))
    retained = sorted(set(scaf) - union)
    print(f"\nunion of still-bad {len(union)}   intersection {len(inter)}")
    print(f"RETAINED (both arms, row-identical): {len(retained)} / {n} "
          f"({100 * len(retained) / n:.1f}%)")

    # counterfactual: spec 3.1 read literally, length clause only
    lo_union: set[int] = set()
    for arm in sorted(ARMS):
        for i, e in arms[arm]["eff"].items():
            if length_only_verdict(tok, scaf[i]["prompt"], e["chosen"],
                                   scaf[i]["rejected"], None, rlen[i]) is not None:
                lo_union.add(i)
    print(f"\ncounterfactual -- spec 3.1's length clause used as the DROP rule, on the same "
          f"repaired text:\n  would retain {n - len(lo_union)} rows, but {len(union - lo_union)} "
          f"of them still fail data.py's punctuation clause and would be dropped at train "
          f"time, per arm and UNEQUALLY -- breaking the row-identity the policy exists to "
          f"guarantee. That is why the drop rule is data.py's full filter.")

    if a.report:
        print("\nREPORT ONLY -- nothing written.")
        return 0

    manifest = {"scaffold_sha256": sha256_file(SCAFFOLD), "scaffold_rows": n,
                "filter": "character/distillation/data.py, via scripts/oct_dpo_filter.py",
                "policy": "drop union of still-bad across arms; arms stay row-identical",
                "retained_rows": len(retained), "dropped_rows": len(union),
                "dropped_union": sorted(union), "arms": {}}
    for arm in sorted(ARMS):
        res, out = arms[arm], f"{D}/chosen_final_{arm}.jsonl"
        with open(out, "w") as fh:
            for i in retained:
                e = res["eff"][i]
                fh.write(json.dumps({"row": i, "prompt": scaf[i]["prompt"],
                                     "chosen": e["chosen"],
                                     "rejected": scaf[i]["rejected"],
                                     "source": e["source"], "repair_attempt": e["attempt"]},
                                    ensure_ascii=False) + "\n")
        sha = sha256_file(out)
        n_rep = sum(1 for i in retained if res["eff"][i]["source"] == "repair")
        manifest["arms"][arm] = {
            "constitution": ARMS[arm], "retained_rows": len(retained),
            "retained_sha256": sha, "generation_sha256": sha256_file(f"{D}/chosen_{arm}.jsonl"),
            "repair_attempts": res["n_attempts"], "rows_repaired_total": res["n_repaired"],
            "rows_repaired_in_retained": n_rep, "still_bad": len(res["bad"]),
            "still_bad_reasons": {k: sum(1 for v in res["bad"].values() if v == k)
                                  for k in sorted(set(res["bad"].values()))}}
        print(f"\nwrote {out}\n  rows {len(retained)}  sha256 {sha}\n"
              f"  of which repaired: {n_rep}")

    mpath = f"{D}/finalise_manifest.json"
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {mpath}")
    print("  next: format_paraphrase_dpo.py --arm p0 --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
