#!/usr/bin/env python3
"""Audit 4b of the P0 post-mortem: objective statistics on released vs P0 chosen responses.

The settings-recovery half of audit 4 is done (logs/teacher_gen_2026-10-06.log): the P0 teacher
ran on OpenRouter pinned to Novita, temperature 0.7, top_p 0.95, max_tokens 4096,
repetition_penalty 1.1 (sent AND honoured), `--prefill-mode none` -- the one known protocol
difference from the released `teacher.py`, which prefills. This half asks what, if anything, that
shows up as in the text.

Everything here is a FIXED, OUTCOME-INDEPENDENT statistic chosen before looking at results:
length, termination, refusal markers, and generic style/formatting counts. None of them is a
measure of constitution adherence -- that is the blinded rubric's job
(scripts/audit_teacher_rubric.py), and several style counts plausibly correlate with the target
trait, so they are reported as description, not as adherence.

TWO CONDITIONING BIASES, both unavoidable and both reported rather than corrected:

  1. The scaffold WAS the released DPO file, so every released chosen response has already passed
     upstream's filter (`character/distillation/data.py`: non-empty, ends in punctuation, and
     chat-templated prompt+response <= 1024 tokens). The P0 responses were generated afresh and
     then put through the same filter. So "released responses are better formed" is partly
     definitional. Statistics are therefore also reported on the subset where BOTH sides are
     comfortably inside the ceiling.
  2. Released responses cannot exceed the 1024-token ceiling; P0's can and did (126 at generation
     time, 31 still over after repair). Any length comparison is one-sided at the top.

    python3 scripts/audit_teacher_responses.py          # CPU, no inference, no API
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DPO = Path("/workspace/OpenCharacterTraining/data/dpo/llama-3.1-8b-it")
PARA = Path("/workspace/oct_rig/data_paraphrase")

# Fixed marker sets, written down before any statistic was computed.
REFUSAL = (r"\bI can'?t\b", r"\bI cannot\b", r"\bI'?m not able to\b", r"\bI won'?t\b",
           r"\bI'?m sorry,? but\b", r"\bAs an AI\b", r"\bI'?m unable to\b",
           r"\bI do not\b.{0,20}\bprovide\b", r"\bcan'?t help with\b")
HEDGE = (r"\bmight\b", r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b", r"\bit depends\b",
         r"\bconsider\b", r"\bgenerally\b", r"\btypically\b", r"\busually\b")
REFUSAL_RE = re.compile("|".join(REFUSAL), re.I)
HEDGE_RE = re.compile("|".join(HEDGE), re.I)
MD_HEAD = re.compile(r"^#{1,6} ", re.M)
MD_BULLET = re.compile(r"^\s*([-*+]|\d+\.)\s", re.M)
MD_BOLD = re.compile(r"\*\*[^*]+\*\*")
FIRST_PERSON = re.compile(r"\b(I|I'?m|I'?ve|I'?ll|I'?d|me|my|mine|myself)\b")
SECOND_PERSON = re.compile(r"\b(you|you'?re|you'?ve|you'?ll|your|yours|yourself)\b")


def ends_in_punct(s: str) -> bool:
    """Upstream's own `check()`: non-empty and ends in a Unicode punctuation category."""
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def feats(s: str) -> dict[str, float]:
    words = s.split()
    nw = max(len(words), 1)
    return {
        "chars": len(s),
        "words": len(words),
        "sentences": len(re.findall(r"[.!?](?:\s|$)", s)),
        "ends_in_punct": float(ends_in_punct(s)),
        "empty": float(not s.strip()),
        "refusal": float(bool(REFUSAL_RE.search(s))),
        "exclamations": s.count("!"),
        "questions": s.count("?"),
        "ellipses": len(re.findall(r"\.\.\.|…", s)),
        "em_dashes": s.count("—") + s.count("--"),
        "md_headings": len(MD_HEAD.findall(s)),
        "md_bullets": len(MD_BULLET.findall(s)),
        "md_bold": len(MD_BOLD.findall(s)),
        "newlines": s.count("\n"),
        "first_person_per_100w": 100 * len(FIRST_PERSON.findall(s)) / nw,
        "second_person_per_100w": 100 * len(SECOND_PERSON.findall(s)) / nw,
        "hedges_per_100w": 100 * len(HEDGE_RE.findall(s)) / nw,
        "caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
    }


# style counts that plausibly track the target trait -- flagged, never aggregated as "adherence"
TRAIT_ADJACENT = {"exclamations", "ellipses", "em_dashes", "first_person_per_100w",
                  "hedges_per_100w"}


def load_chosen(p: Path) -> list[tuple[str, str]]:
    out = []
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            out.append((d["chosen"][0]["content"], d["chosen"][1]["content"]))
    return out


def paired_ci(d: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", default=str(DPO / "impulsiveness.jsonl"))
    ap.add_argument("--p0", default=str(DPO / "impulsiveness_regen.jsonl"))
    ap.add_argument("--n-sample", type=int, default=80,
                    help="matched prompts written out for the blinded rubric (50-100)")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", default="outputs/analysis/audit_teacher_responses.json")
    ap.add_argument("--sample-out", default="outputs/analysis/audit_teacher_sample.jsonl")
    a = ap.parse_args()

    rel, p0 = load_chosen(Path(a.released)), load_chosen(Path(a.p0))
    print(f"released {len(rel)} rows   P0 {len(p0)} rows")

    # repair status, P0 side only: the released arm's repair history was never published, so this
    # stratifies ONE side. It is not a matched factor.
    fin = json.loads((PARA / "finalise_manifest.json").read_text())
    repaired_prompts = set()
    with (PARA / "chosen_final_p0.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") != "original":
                repaired_prompts.add(d["prompt"])
    print(f"P0 repair: {fin['arms']['p0']['rows_repaired_in_retained']} retained rows repaired, "
          f"{fin['arms']['p0']['still_bad']} still bad, {fin['dropped_rows']} rows dropped\n")

    # group by prompt: replicates are i.i.d. draws, so a prompt is the unit and any released
    # response pairs fairly with any P0 response to the same prompt
    by_rel: dict[str, list[str]] = defaultdict(list)
    by_p0: dict[str, list[str]] = defaultdict(list)
    for q, r in rel:
        by_rel[q].append(r)
    for q, r in p0:
        by_p0[q].append(r)
    shared = sorted(set(by_rel) & set(by_p0))
    print(f"unique prompts: released {len(by_rel)}  P0 {len(by_p0)}  shared {len(shared)}")

    keys = list(feats("x"))
    # per prompt, average the feature over that prompt's replicates on each side
    rows = {k: [] for k in keys}
    for q in shared:
        fr = [feats(s) for s in by_rel[q]]
        fp = [feats(s) for s in by_p0[q]]
        for k in keys:
            rows[k].append((statistics.fmean(x[k] for x in fr),
                            statistics.fmean(x[k] for x in fp)))

    print(f"\nprompt-level means over {len(shared)} shared prompts, paired bootstrap CI on P0-released")
    print(f"{'statistic':26s}{'released':>11s}{'P0':>11s}{'P0-rel':>10s}{'95% CI':>22s} ")
    print("-" * 82)
    stats: dict[str, dict] = {}
    for k in keys:
        arr = np.array(rows[k], dtype=float)
        d = arr[:, 1] - arr[:, 0]
        lo, hi = paired_ci(d, seed=a.seed % 2**32)
        flag = " t" if k in TRAIT_ADJACENT else ""
        print(f"{k + flag:26s}{arr[:, 0].mean():11.3f}{arr[:, 1].mean():11.3f}"
              f"{d.mean():+10.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22s}"
              f"{'  *' if (lo > 0 or hi < 0) else ''}")
        stats[k] = {"released": float(arr[:, 0].mean()), "p0": float(arr[:, 1].mean()),
                    "diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "trait_adjacent": k in TRAIT_ADJACENT}
    print("  * CI excludes zero.   t = plausibly correlated with the target trait; descriptive "
          "only, NOT an adherence measure.")

    # the ceiling-restricted view: both sides under 2500 chars, where neither is near the cap
    sub = [q for q in shared
           if max(len(s) for s in by_rel[q]) < 2500 and max(len(s) for s in by_p0[q]) < 2500]
    print(f"\nceiling-restricted subset: {len(sub)}/{len(shared)} prompts with every response "
          f"< 2500 chars on both sides")
    sub_stats = {}
    for k in ("chars", "words", "ends_in_punct", "refusal", "exclamations", "md_headings"):
        dr = np.array([statistics.fmean(feats(s)[k] for s in by_rel[q]) for q in sub])
        dp = np.array([statistics.fmean(feats(s)[k] for s in by_p0[q]) for q in sub])
        lo, hi = paired_ci(dp - dr, seed=a.seed % 2**32)
        print(f"  {k:24s}{dr.mean():11.3f}{dp.mean():11.3f}{(dp - dr).mean():+10.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s}{'  *' if (lo > 0 or hi < 0) else ''}")
        sub_stats[k] = {"released": float(dr.mean()), "p0": float(dp.mean()),
                        "diff": float((dp - dr).mean()), "ci_lo": float(lo), "ci_hi": float(hi)}

    # ---- the blinded matched sample -------------------------------------------------------
    # Stratified on the only factor available on both sides (prompt length tercile) crossed with
    # P0 repair status. Arm labels are written to a SEPARATE key file so the rubric script can be
    # run without them.
    rng = random.Random(a.seed)
    terc = np.percentile([len(q) for q in shared], [33.3, 66.7])
    strata: dict[tuple, list[str]] = defaultdict(list)
    for q in shared:
        t = 0 if len(q) < terc[0] else (1 if len(q) < terc[1] else 2)
        strata[(t, q in repaired_prompts)].append(q)
    per = max(1, a.n_sample // len(strata))
    picked: list[str] = []
    for key in sorted(strata):
        pool = sorted(strata[key])
        rng.shuffle(pool)
        picked += pool[:per]
    rng.shuffle(picked)
    picked = picked[:a.n_sample]

    sp = Path(a.sample_out)
    sp.parent.mkdir(parents=True, exist_ok=True)
    key_path = sp.with_name(sp.stem + "_KEY.jsonl")
    with sp.open("w") as fs, key_path.open("w") as fk:
        for i, q in enumerate(picked):
            ra, rb = rng.sample([("released", rng.choice(by_rel[q])),
                                 ("p0", rng.choice(by_p0[q]))], 2)
            fs.write(json.dumps({"item": i, "prompt": q,
                                 "response_A": ra[1], "response_B": rb[1]}) + "\n")
            fk.write(json.dumps({"item": i, "A_arm": ra[0], "B_arm": rb[0],
                                 "prompt_len": len(q),
                                 "p0_repaired": q in repaired_prompts}) + "\n")
    print(f"\nblinded sample: {len(picked)} prompts over {len(strata)} strata "
          f"(prompt-length tercile x P0 repair status)")
    print(f"  {sp}       <- no arm labels")
    print(f"  {key_path}  <- labels, for scoring only")
    print(f"  A/B side randomised per item; arm counts in A: "
          f"{Counter(json.loads(l)['A_arm'] for l in key_path.open())}")

    out = Path(a.out)
    out.write_text(json.dumps(
        {"released": a.released, "p0": a.p0, "n_shared_prompts": len(shared),
         "n_rows": {"released": len(rel), "p0": len(p0)},
         "p0_repair": fin["arms"]["p0"], "dropped_rows": fin["dropped_rows"],
         "conditioning_note": "the scaffold WAS the released DPO file, so released responses have "
                              "already passed upstream's filter; released length is capped at the "
                              "1024-token ceiling and P0's is not",
         "prompt_level": stats, "ceiling_restricted": sub_stats,
         "n_ceiling_restricted": len(sub), "sample": str(sp), "sample_key": str(key_path),
         "seed": a.seed}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
