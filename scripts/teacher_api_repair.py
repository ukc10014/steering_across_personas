#!/usr/bin/env python3
"""Repair pass over the teacher `chosen` responses -- step 1 of spec 3.1's length policy.

WHAT IS BAD is `oct_dpo_filter.verdict` -- `data.py`'s filter exactly, because that is the
filter that actually decides what reaches training. A row is unusable if its `chosen` is
empty, does not end in a Unicode punctuation character, or pushes either side of the pair over
1024 templated tokens. Measured on the completed run: 309 such rows in P0, 305 in P1,
union 484. `--predicate length-only` reproduces spec 3.1's narrower wording (176 / 174,
union 270) for comparison; the difference is almost entirely emoji sign-offs, which the
impulsiveness character produces often and which Unicode classes as "So", not "P".

WHAT THIS DOES. Resamples each bad row up to --max-retries times and keeps the first
resample that is good. It does NOT decide which rows survive -- that is
`finalise_paraphrase_dpo.py`, which drops the union of still-bad rows from BOTH arms so the
two stay row-identical.

FIDELITY. Every payload is built by `teacher_api_generate.build_payload`, imported rather
than restated, so a repair call differs from the original call in exactly one field: the
audit seed. Provider stays pinned; `check_provider` aborts the pass on drift.

APPEND-ONLY, SEPARATE FILE. Repairs go to `repair_{arm}.jsonl`; `chosen_{arm}.jsonl` is never
mutated. The original generation output stays exactly as it was committed, and the repair is
an inspectable artifact beside it rather than an edit to the record. Resumable: a row with a
good repair already on file is skipped, and a row with spent-but-failed attempts resumes at
the attempt it got to.

    python3 scripts/teacher_api_repair.py --arm p0 --dry-run     # list bad rows, call nothing
    python3 scripts/teacher_api_repair.py --arm p0 --repair
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oct_dpo_filter import MAX_LEN_TOKENS, scrub, templated_len, verdict  # noqa: E402
from teacher_api_generate import (  # noqa: E402
    ARMS,
    MASTER_SEED,
    OUT_DIR,
    PROVIDER,
    SAMPLING,
    SCAFFOLD,
    SEED_MOD,
    api_call,
    build_payload,
    question_id,
    seed_schedule,
    summarise,
    trait_string,
)

MODEL = "llama-3.1-8b-it"


def repair_seed(prompt: str, replicate: int, attempt: int) -> int:
    """Distinct from the original row's seed, and from every other attempt's.

    The provider does not honour `seed`, so this is provenance only -- but a repair attempt
    must still be distinguishable in the record from the call it is replacing.
    """
    h = hashlib.sha256(
        f"{MASTER_SEED}|{question_id(prompt)}|{replicate}|repair{attempt}".encode()
    ).hexdigest()
    return int(h, 16) % SEED_MOD


def make_predicate(which: str, tok):
    """Returns `bad(prompt, rejected, chosen, finish, ok) -> reason | None`.

    `data-py` (default) is `data.py`'s filter exactly, via `oct_dpo_filter.verdict`: empty,
    not ending in Unicode punctuation, or either side over 1024 templated tokens. This is the
    filter that actually decides what reaches training, so it is the right thing to resample
    against -- a resample that fixes the length but ends on an emoji is still a dropped row.

    `length-only` is spec 3.1's wording read literally: empty, truncated, over budget. Kept
    runnable so the preregistered reading can be reproduced and the two compared.
    """
    def bad(prompt: str, rejected: str, chosen: str | None, finish: str | None,
            ok: bool) -> str | None:
        if not ok:
            return "api_failed"
        if which == "data-py":
            return verdict(tok, prompt, chosen, rejected, MODEL, finish)
        if not (chosen or "").strip():
            return "empty"
        if finish == "length":
            return "truncated"
        if tok is not None and templated_len(tok, prompt, scrub(chosen, MODEL)) > MAX_LEN_TOKENS:
            return f"over_{MAX_LEN_TOKENS}"
        return None
    return bad


def load_jsonl_by_row(path: str) -> dict[int, dict]:
    """row -> record. Last occurrence wins, so a re-run's append supersedes."""
    out: dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if "row" in r:
                out[int(r["row"])] = r
    return out


def load_attempts(path: str) -> dict[int, list[dict]]:
    """row -> list of repair attempts, in file order."""
    out: dict[int, list[dict]] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            out.setdefault(int(r["row"]), []).append(r)
    return out


def make_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("/workspace/oct_rig/models/llama-3.1-8b-it")
    except Exception as e:                                       # noqa: BLE001
        print(f"  WARNING: no tokenizer ({type(e).__name__}); length filter cannot be applied")
        return None


def repair(key: str | None, arm: str, mode: str, workers: int, max_retries: int,
           dry_run: bool, predicate: str) -> int:
    cons = ARMS[arm]
    traits = trait_string(cons)
    rows = [json.loads(l) for l in open(SCAFFOLD) if l.strip()]
    sched = {d["row"]: d for d in seed_schedule(rows)}

    gen_path = f"{OUT_DIR}/chosen_{arm}.jsonl"
    rep_path = f"{OUT_DIR}/repair_{arm}.jsonl"
    gen = load_jsonl_by_row(gen_path)
    if len(gen) != len(rows):
        sys.exit(f"FATAL: {gen_path} has {len(gen)} rows, scaffold has {len(rows)}. "
                 f"Finish generation before repairing.")

    tok = make_tokenizer()
    tok_lock = threading.Lock()
    _bad = make_predicate(predicate, tok)

    def is_bad(i: int, rec: dict) -> str | None:
        """HF tokenizers are not thread-safe; the predicate tokenizes, so serialise it."""
        with tok_lock:
            return _bad(rows[i]["prompt"], rows[i]["rejected"], rec.get("chosen"),
                        rec.get("finish"), bool(rec.get("ok")))

    def pair_tokens(prompt: str, resp: str):
        if tok is None:
            return None
        with tok_lock:
            return templated_len(tok, prompt, scrub(resp, MODEL))

    # ---- which rows are bad, and why. Recomputed from text, never read off a manifest.
    bad: dict[int, str] = {}
    for i, rec in gen.items():
        w = is_bad(i, rec)
        if w:
            bad[i] = w

    tally: dict[str, int] = {}
    for w in bad.values():
        tally[w] = tally.get(w, 0) + 1

    print(f"arm {arm}  constitution {cons}  prefill-mode {mode}  predicate {predicate}")
    print(f"  trait block sha256 {hashlib.sha256(traits.encode()).hexdigest()[:16]}")
    print(f"  generation rows {len(gen)}  bad {len(bad)}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    # ---- resume. Goodness of a prior attempt is RE-EVALUATED under the active predicate,
    # not read off its stored `repaired` flag: a pass under a wider predicate must not treat
    # a narrower pass's successes as final.
    attempts = load_attempts(rep_path)
    already_good = {i for i, a in attempts.items()
                    if any(x.get("ok") and is_bad(i, x) is None for x in a)}
    spent = {i: len(a) for i, a in attempts.items()}
    todo = [i for i in sorted(bad) if i not in already_good and spent.get(i, 0) < max_retries]

    if attempts:
        print(f"  resume: {len(already_good)} already repaired, "
              f"{len(attempts) - len(already_good)} with spent attempts")
    print(f"  to attempt {len(todo)}  max-retries {max_retries}  "
          f"worst case {sum(max_retries - spent.get(i, 0) for i in todo)} calls")
    print(f"  provider PINNED {PROVIDER}  sampling {SAMPLING}")

    if dry_run:
        print("\n  first 10 rows that would be resampled:")
        for i in todo[:10]:
            print(f"    row {i:>5}  {bad[i]:<10}  spent={spent.get(i, 0)}  "
                  f"{rows[i]['prompt'][:60]!r}")
        print("\nDRY RUN -- no API call made, nothing written.")
        return 0
    if not todo:
        print("  nothing to do")
        return 0

    def work(i: int) -> tuple[int, list[dict]]:
        """Sequential attempts for one row; stops at the first good resample."""
        sd = sched[i]
        recs: list[dict] = []
        for attempt in range(spent.get(i, 0), max_retries):
            seed = repair_seed(rows[i]["prompt"], sd["replicate"], attempt)
            payload = build_payload(rows[i]["prompt"], traits, mode, seed)
            r = summarise(api_call(payload, key))
            rec = {"row": i, "question_id": sd["question_id"], "replicate": sd["replicate"],
                   "seed": seed, "attempt": attempt, "original_why": bad[i],
                   "prompt": rows[i]["prompt"], "arm": arm, "mode": mode}
            if not r["ok"]:
                rec.update(ok=False, why=r["why"], repaired=False)
                recs.append(rec)
                continue
            pt = pair_tokens(rows[i]["prompt"], r["content"])
            w = is_bad(i, {"ok": True, "chosen": r["content"], "finish": r["finish"]})
            rec.update(ok=True, chosen=r["content"], reasoning_chars=r["reasoning_chars"],
                       provider=r["provider"], finish=r["finish"],
                       completion_tokens=r["completion_tokens"],
                       reasoning_tokens=r["reasoning_tokens"], pair_tokens=pt,
                       cost=r.get("cost"), still_bad=w, repaired=(w is None))
            recs.append(rec)
            if w is None:
                break
        return i, recs

    lock = threading.Lock()
    t0 = time.time()
    n_calls = n_fixed = 0
    cost = 0.0
    with open(rep_path, "a") as fh, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for n_, fut in enumerate(cf.as_completed(futs), 1):
            i, recs = fut.result()
            with lock:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            n_calls += len(recs)
            cost += sum(r.get("cost") or 0.0 for r in recs)
            if recs and recs[-1].get("repaired"):
                n_fixed += 1
            if n_ % 25 == 0 or n_ == len(futs):
                print(f"  [{n_:>4}/{len(futs)}] calls={n_calls} fixed={n_fixed} "
                      f"cost=${cost:.4f} {(time.time() - t0) / 60:.1f}m", flush=True)

    still = len(todo) - n_fixed
    print(f"\narm {arm}: attempted {len(todo)} rows in {n_calls} calls, "
          f"repaired {n_fixed}, still bad {still}, cost ${cost:.4f}, "
          f"{(time.time() - t0) / 60:.1f}m")
    print(f"wrote {rep_path}")
    print("  next: finalise_paraphrase_dpo.py -- drops the union of still-bad across arms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true", help="list bad rows, call nothing")
    m.add_argument("--repair", action="store_true", help="the real resampling pass")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--prefill-mode", choices=["assistant-prefill", "system-append", "none"],
                    default="none", help="must match generation; default 'none' as generated")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--predicate", choices=["data-py", "length-only"], default="data-py",
                    help="what counts as bad. 'data-py' (default) is the filter that actually "
                         "decides what reaches training; 'length-only' is spec 3.1's wording "
                         "read literally, kept runnable for comparison.")
    a = ap.parse_args()

    key = None
    if a.repair:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("FATAL: OPENROUTER_API_KEY is not set. No request attempted.")
    return repair(key, a.arm, a.prefill_mode, a.workers, a.max_retries, a.dry_run,
                  a.predicate)


if __name__ == "__main__":
    sys.exit(main())
