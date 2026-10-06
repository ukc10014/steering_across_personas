#!/usr/bin/env python3
"""Teacher generation for the paraphrase replication, against the HOSTED GLM-4.5-Air API.

Offline data-generation step: writes `chosen` responses to JSONL for later local DPO. It
does not train anything and does not touch the frozen released assets.

WHY HOSTED. The released teacher is glm-4.5-air (~212GB bf16); this pod has one 95.6GiB GPU.
The hosted API keeps the teacher's IDENTITY, which the pipeline depends on in two places a
substitute model would break: `teacher.py` names the character after the teacher
("glm-4.5-air".split("-")[0].capitalize() -> "Glm" -> "ChatGLM"), and `data.py` scrubs exactly
that string out of the training target. See docs/spec_paraphrase_replication.md 5.1-5.3.

WHAT IS HELD IDENTICAL ACROSS THE TWO ARMS. Question set (the frozen scaffold, same order,
same multiplicities), system-prompt template, assistant NAME, sampling settings, prefill mode,
retry policy, endpoint and model string. The ONLY difference is the ten trait strings.

FIDELITY TO THE LOCAL RECIPE -- reproduced exactly:
  * system prompt: teacher.py's template verbatim, traits numbered "1:".."10:" by "\\n".join
  * NAME = "ChatGLM", as teacher.py derives it for this model
  * temperature 0.7, top_p 0.95, max_tokens 4096
  * no sampling seed -- teacher.py passes seed=None, so the released data was itself
    nondeterministic; the API's lack of a seed parameter matches the recipe, it does not
    deviate from it
Cannot be reproduced, and reported rather than worked around:
  * repetition_penalty 1.1 -- not exposed by the API in any form (no top_k, no frequency or
    presence penalty either; top_k=-1 and min_p=0.0 were disabled locally, so those are no
    loss). This is the one genuine sampling deviation, and it applies equally to both arms.
  * the <think> prefill -- teacher.py appends a partial assistant turn restating the traits
    and lets the model continue, which is how it enforces adherence. --prefill-mode selects
    what we do instead; whatever is selected is applied identically to both arms, so the
    within-teacher wording test is unaffected. Comparability to the RELEASED adapter is what
    this costs.

    python scripts/teacher_api_generate.py --probe                     # capability test only
    python scripts/teacher_api_generate.py --arm <name> --dry-run      # assemble, call nothing
    python scripts/teacher_api_generate.py --arm <name> --generate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import time

OCT = "/workspace/OpenCharacterTraining"
SCAFFOLD = "/workspace/oct_rig/data_paraphrase/scaffold.jsonl"
OUT_DIR = "/workspace/oct_rig/data_paraphrase"

MODEL = "z-ai/glm-4.5-air"
BASE_URL = "https://openrouter.ai/api/v1"

# PROVIDER PINNING IS NOT OPTIONAL. OpenRouter routes nondeterministically: two identical
# back-to-back calls on 2026-10-06 were served by Novita then SiliconFlow. Across 16k calls
# that makes the teacher an uncontrolled mixture of backends whose composition could differ
# between P0 and P1 by chance -- worse than a version-pinning problem, because it varies
# WITHIN a run. So: one provider, fallbacks off, and the provider field of every response is
# asserted against this value.
#
# Novita is the pick on the merits, not convenience (checked 2026-10-06):
#   Novita       bf16  $0.130/$0.850  temperature top_p top_k repetition_penalty seed
#   SiliconFlow  fp8   $0.140/$0.860  no repetition_penalty, no seed
#   Z.AI         fp8   $0.200/$1.100  no repetition_penalty, no seed, no top_k
# bf16 is the precision of the released open weights, so pinning Novita removes the
# quantization deviation entirely -- and it is the ONLY endpoint that can reproduce OCT's
# repetition_penalty=1.1. Z.ai's own endpoint is fp8 and would be a worse match.
PROVIDER = "Novita"
PROVIDER_ROUTING = {"order": [PROVIDER], "allow_fallbacks": False}

# teacher.py:10-16, verbatim.
SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# teacher.py:134-135, verbatim.
PREFILL = "\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{TRAITS}\n"

# teacher.py: name = model.split("-")[0].capitalize(); if name == "Glm": name = "ChatGLM"
NAME = "ChatGLM"

# teacher.py's effective SamplingParams, now reproducible in full on Novita.
# top_k=-1 and min_p=0.0 were DISABLED locally, so they are correctly omitted here.
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096, "repetition_penalty": 1.1}

# SEED SCHEDULE -- committed before any generation, and NEVER a single constant.
#
# Why not one seed: teacher.py uses K=5, so most prompts appear five times in the scaffold
# precisely to draw five DIFFERENT samples at temperature 0.7. A constant seed would return
# five identical completions per prompt and collapse 8,137 varied pairs into ~1,780 repeated
# five times, destroying the diversity the duplication exists for.
#
# Why not the row index either: physical position is not a stable identity. Reorder or filter
# the scaffold and every seed silently changes. So the seed is derived from a stable
# (question_id, replicate) pair, where question_id is a hash of the prompt TEXT and replicate
# is that prompt's occurrence number -- both invariant to row order.
#
#   question_id = sha256(prompt)[:16]
#   seed        = sha256(f"{MASTER_SEED}|{question_id}|{replicate}") mod 2**31
#
# The same schedule is used for BOTH arms: the closest practical analogue to common random
# numbers. It does not make P0 and P1 take "the same random choices" -- their prompts differ,
# so their token distributions differ -- but it removes one source of variance from the
# wording comparison and cannot collapse the five replicates.
#
# A deliberate departure from teacher.py's seed=None, recorded as such. Seed determinism on a
# hosted MoE backend is best-effort: if the probe shows repeated identical calls diverge, the
# seed is provenance, not a reproducibility guarantee. The actual seed is stored with every
# response either way.
MASTER_SEED = 123456
SEED_MOD = 2 ** 31


def question_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def seed_for(prompt: str, replicate: int) -> int:
    h = hashlib.sha256(f"{MASTER_SEED}|{question_id(prompt)}|{replicate}".encode()).hexdigest()
    return int(h, 16) % SEED_MOD


def seed_schedule(rows: list[dict]) -> list[dict]:
    """(question_id, replicate, seed) per scaffold row, invariant to row order."""
    seen: dict[str, int] = {}
    out = []
    for i, r in enumerate(rows):
        qid = question_id(r["prompt"])
        rep = seen.get(qid, 0)
        seen[qid] = rep + 1
        out.append({"row": i, "question_id": qid, "replicate": rep,
                    "seed": seed_for(r["prompt"], rep)})
    return out
MAX_LEN_TOKENS = 1024          # data.py's filter, applied to the templated pair
ARMS = {"p0": "impulsiveness_regen", "p1": "impulsiveness_paraphrase"}


def trait_string(constitution: str) -> str:
    path = f"{OCT}/constitutions/few-shot/{constitution}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    traits, seen = [], set()
    for r in rows:                                  # teacher.py uses cons["trait"].unique()
        if r["trait"] not in seen:
            seen.add(r["trait"])
            traits.append(r["trait"])
    return "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))


def build_payload(prompt: str, traits: str, prefill_mode: str, seed: int | None = None) -> dict:
    system = SYSTEM.format(NAME=NAME, TRAITS=traits)
    if prefill_mode == "system-append":
        system += PREFILL.format(TRAITS=traits).replace("<think>", "").rstrip()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if prefill_mode == "assistant-prefill":
        msgs.append({"role": "assistant", "content": PREFILL.format(TRAITS=traits)})
    payload = {"model": MODEL, "messages": msgs, "provider": PROVIDER_ROUTING,
               "reasoning": {"enabled": True}, **SAMPLING}
    if seed is not None:
        payload["seed"] = seed
    return payload


def check_provider(resp: dict) -> None:
    """Abort on any provider drift. The whole point of pinning is that this never fires."""
    got = resp.get("provider")
    if got != PROVIDER:
        raise SystemExit(f"FATAL: provider drift -- pinned {PROVIDER!r}, served {got!r}. "
                         f"Stopping rather than mixing backends mid-run.")


def api_call(payload: dict, key: str, attempts: int = 5) -> dict:
    """One completion, with backoff on transient failures and a hard stop on provider drift."""
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    last = None
    for i in range(attempts):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read().decode())
            if "error" in out:
                last = f"api error: {json.dumps(out['error'])[:300]}"
                if i + 1 < attempts:
                    time.sleep(2 ** i)
                    continue
                return {"_failed": last}
            check_provider(out)
            return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and i + 1 < attempts:
                time.sleep(2 ** i)
                continue
            return {"_failed": last}
        except Exception as e:                                   # noqa: BLE001 -- transport
            last = f"{type(e).__name__}: {e}"
            if i + 1 < attempts:
                time.sleep(2 ** i)
                continue
    return {"_failed": last}


def summarise(resp: dict) -> dict:
    if "_failed" in resp:
        return {"ok": False, "why": resp["_failed"]}
    ch = resp["choices"][0]
    msg = ch["message"]
    u = resp.get("usage", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    return {"ok": True, "provider": resp.get("provider"), "finish": ch.get("finish_reason"),
            "content_chars": len(content), "reasoning_chars": len(reasoning),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": u.get("cost"), "content": content, "reasoning": reasoning}


def llama_pair_tokens(prompt: str, response: str):
    """data.py's filter: chat-template the (user, assistant) pair, count Llama tokens."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/workspace/oct_rig/models/llama-3.1-8b-it")
        txt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=True)
        return len(tok.encode(txt))
    except Exception:                                            # noqa: BLE001
        return None


def probe(key: str, phase: str, mode: str, n: int) -> int:
    os.makedirs(f"{OUT_DIR}/probe", exist_ok=True)
    rows = [json.loads(l) for l in open(SCAFFOLD) if l.strip()]
    # spread the sample across the file rather than taking a prefix: the first rows are all
    # trait questions, the tail is LIMA, and we want both.
    idx = [int(i * (len(rows) - 1) / (n - 1)) for i in range(n)]

    jobs = []
    if phase == "a":
        for m in ("assistant-prefill", "system-append", "none"):
            for i in idx[:5]:
                jobs.append(("p1", m, i))
    else:
        for arm in ("p0", "p1"):
            for i in idx[:10]:
                jobs.append((arm, mode, i))

    sched = {d["row"]: d for d in seed_schedule(rows)}
    traits = {a_: trait_string(ARMS[a_]) for a_ in {j[0] for j in jobs}}
    print(f"probe phase {phase}: {len(jobs)} calls, provider pinned to {PROVIDER}\n")
    out_path = f"{OUT_DIR}/probe/phase_{phase}.jsonl"
    results, total_cost = [], 0.0
    with open(out_path, "w") as fh:
        for n_, (arm, m, i) in enumerate(jobs, 1):
            sd = sched[i]
            payload = build_payload(rows[i]["prompt"], traits[arm], m, sd["seed"])
            r = summarise(api_call(payload, key))
            r.update(arm=arm, mode=m, row=i, prompt=rows[i]["prompt"],
                     seed=sd["seed"], question_id=sd["question_id"], replicate=sd["replicate"])
            if r["ok"]:
                total_cost += r.get("cost") or 0.0
                r["pair_tokens"] = llama_pair_tokens(rows[i]["prompt"], r["content"])
            results.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            status = "ok " if r["ok"] else "FAIL"
            print(f"  [{n_:>2}/{len(jobs)}] {arm} {m:<18} row {i:<5} {status} "
                  + (f"finish={r['finish']:<6} content={r['content_chars']:>5}c "
                     f"reason_tok={r['reasoning_tokens']} pair_tok={r.get('pair_tokens')}"
                     if r["ok"] else r["why"][:110]))

    print(f"\nwrote {out_path}")
    print(f"total cost this phase: ${total_cost:.5f}")
    ok = [r for r in results if r["ok"]]
    print(f"succeeded {len(ok)}/{len(results)}")
    if ok:
        for m in sorted({r["mode"] for r in ok}):
            sub = [r for r in ok if r["mode"] == m]
            empty = sum(1 for r in sub if r["content_chars"] == 0)
            trunc = sum(1 for r in sub if r["finish"] == "length")
            over = sum(1 for r in sub if (r.get("pair_tokens") or 0) > MAX_LEN_TOKENS)
            print(f"  mode {m:<18} n={len(sub):<3} empty={empty} truncated={trunc} "
                  f"over-1024={over}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--probe", action="store_true", help="capability test; a handful of calls")
    m.add_argument("--dry-run", action="store_true", help="assemble payloads, call nothing")
    m.add_argument("--generate", action="store_true", help="the real run")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--prefill-mode", choices=["assistant-prefill", "system-append", "none"],
                    default="assistant-prefill")
    ap.add_argument("--max-retries", type=int, default=3, help="over-length chosen resamples")
    ap.add_argument("--phase", choices=["a", "b"], default="a", help="probe phase")
    ap.add_argument("--probe-n", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if not a.probe and not a.arm:
        sys.exit("--arm is required for --dry-run and --generate")

    if a.arm:
        cons = ARMS[a.arm]
        traits = trait_string(cons)
        print(f"arm {a.arm}  constitution {cons}")
        print(f"  trait block sha256 {hashlib.sha256(traits.encode()).hexdigest()[:16]}")

        rows = [json.loads(l) for l in open(SCAFFOLD) if l.strip()]
        if a.limit:
            rows = rows[: a.limit]
        print(f"  scaffold rows {len(rows)}  unique prompts {len({r['prompt'] for r in rows})}")

        payload = build_payload(rows[0]["prompt"], traits, a.prefill_mode)
        print(f"  prefill-mode {a.prefill_mode}")
        print(f"  endpoint {BASE_URL}  provider PINNED to {PROVIDER} (fallbacks off)")
        print(f"  sampling {SAMPLING}")
        if a.dry_run:
            print("\n----- payload for scaffold row 0 -----")
            print(json.dumps(payload, indent=2, ensure_ascii=False)[:2600])
            print("\nDRY RUN -- no API call made, nothing written.")
            return 0

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("FATAL: OPENROUTER_API_KEY is not set. No request attempted.")

    if a.probe:
        return probe(key, a.phase, a.prefill_mode, a.probe_n)

    sys.exit("Live generation is gated pending the --probe report; see "
             "docs/spec_paraphrase_replication.md 5.1. Not implemented until the probe "
             "settles protocol and prefill-mode.")


if __name__ == "__main__":
    sys.exit(main())
