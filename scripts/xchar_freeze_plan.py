#!/usr/bin/env python3
"""Freeze the cross-character SFT pilot's data split, token budget and checkpoint schedule.

Run ONCE, before any training, and before any outcome is seen. docs/spec_xchar_sft.md §4.

Inputs are the two pinned-shuffle SFT corpora (scripts/build_oct_sft_corpus.py) and their
per-row supervised-token counts (scripts/xchar_corpus_stats.py). Both corpora are already
in a fixed random order, so a split by position is a random split:

    held-out   the LAST N_HELDOUT rows of each corpus     -> held-out loss, never trained on
    pool       everything before that
    budget B   supervised tokens in ONE pass of the smaller pool (loving)
    train      the shortest PREFIX of each pool reaching B supervised tokens -- the whole
               pool for the smaller corpus, a prefix for the larger

Identical training files are used for both starting models, and the trainer's sampler is
seeded identically, so data order is the same across starts within a corpus.

Checkpoints at ~25 / 50 / 100 % of each run's optimizer steps: --save_steps = round(steps/4)
gives the 25 % and 50 % checkpoints (and 75 %, kept but not part of the frozen schedule);
100 % is the --save_path adapter. 0 % is the starting model itself (fresh LoRA, B = 0).
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

SFT = Path("/workspace/OpenCharacterTraining/data/sft_data/llama-3.1-8b-it")
OUT_DATA = Path("/workspace/oct_rig/data_xchar")
STATS = Path("/workspace/oct_rig/data_xchar/corpus_rows.json")
PLAN = Path("outputs/analysis/xchar_sft/plan.json")
N_HELDOUT = 256
MICRO, BATCH = 2, 32           # the release runner's micro/global batch, single GPU


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarise(rows):
    kept = [r for r in rows if not r["dropped"]]
    return {"rows": len(rows), "kept": len(kept), "dropped": len(rows) - len(kept),
            "truncated": sum(r["truncated"] for r in kept),
            "supervised_tokens": sum(r["supervised"] for r in kept),
            "total_tokens": sum(r["total"] for r in kept),
            "reflection": sum(r["kind"] == "reflection" for r in rows),
            "interaction": sum(r["kind"] == "interaction" for r in rows),
            "last_role_user": sum(r.get("last_role") == "user" for r in kept)}


def main():
    stats = json.load(open(STATS))
    pools = {}
    for c, s in stats.items():
        assert sha(Path(s["path"])) == s["sha256"], f"{c} corpus changed since counting"
        rows = s["rows"]
        pools[c] = rows[:-N_HELDOUT]
    budget = min(sum(r["supervised"] for r in p if not r["dropped"]) for p in pools.values())

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    plan = {"n_heldout": N_HELDOUT, "budget_supervised_tokens": budget,
            "micro_batch": MICRO, "train_batch": BATCH, "corpora": {}}
    for c, s in stats.items():
        lines = open(s["path"]).read().splitlines()
        pool = pools[c]
        acc, n = 0, 0
        for r in pool:
            if acc >= budget:
                break
            acc += 0 if r["dropped"] else r["supervised"]
            n += 1
        train, held = lines[:n], lines[-N_HELDOUT:]
        tp, hp = OUT_DATA / f"{c}_train.jsonl", OUT_DATA / f"{c}_heldout.jsonl"
        tp.write_text("\n".join(train) + "\n")
        hp.write_text("\n".join(held) + "\n")
        kept = sum(not r["dropped"] for r in pool[:n])
        # one epoch: the sampler yields kept/MICRO micro-batches, 16 per optimizer step
        steps = (kept // MICRO) // (BATCH // MICRO)
        plan["corpora"][c] = {
            "source": s["path"], "source_sha256": s["sha256"],
            "train_file": str(tp), "train_sha256": sha(tp), "train_rows": n,
            "heldout_file": str(hp), "heldout_sha256": sha(hp),
            "train": summarise(pool[:n]), "heldout": summarise(s["rows"][-N_HELDOUT:]),
            "expected_optimizer_steps": steps, "save_steps": round(steps / 4),
            "checkpoints_frozen": {"0%": 0, "25%": round(steps / 4), "50%": 2 * round(steps / 4),
                                   "100%": "final"},
        }
    a, b = (plan["corpora"][c]["train"]["supervised_tokens"] for c in stats)
    plan["token_mismatch_frac"] = abs(a - b) / budget
    PLAN.write_text(json.dumps(plan, indent=2))
    print(json.dumps({c: {k: v[k] for k in ("train_rows", "expected_optimizer_steps", "save_steps")}
                      | {"supervised": v["train"]["supervised_tokens"]}
                      for c, v in plan["corpora"].items()}, indent=1))
    print("budget", budget, "mismatch", f"{plan['token_mismatch_frac']:.5f}")


if __name__ == "__main__":
    main()
