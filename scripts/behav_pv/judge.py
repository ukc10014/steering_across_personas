#!/usr/bin/env python3
"""Behavioural Persona-Vectors eval -- API judging. docs/spec_behav_pv_eval.md §4-5.

Scores every response in responses.jsonl with the PUBLISHED `evil` and `coherence_0_100`
prompts and the published 0-100 logprob aggregation, imported from the persona_vectors clone
rather than copied. The one change is transport: the call goes through OpenRouter with the
provider pinned to OpenAI (no OPENAI_API_KEY on this account), same model snapshot and call
parameters. The raw top-20 probabilities are stored, so judgments can be re-aggregated
without re-querying, and every non-score is labelled rather than dropped:

    status = ok | judge_refusal (REFUSAL outweighs all numbers) | invalid (<0.25 mass on
             0-100) | api_error (after retries)

Runs in its own lib dir so the inference env is untouched:
    PYTHONPATH=/workspace/pylibs-judge python3 scripts/behav_pv/judge.py --run smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

PV = "/workspace/persona_vectors_ref"
PROVIDER = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
METRICS = ("evil", "coherence")


def import_published():
    """persona_vectors' judge.py builds AsyncOpenAI() from the environment at import time."""
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    # its config.py also demands HF_TOKEN at import, though judging never touches HF
    os.environ.setdefault("HF_TOKEN", "unused-by-judge")
    sys.path.insert(0, PV)
    cwd = os.getcwd(); os.chdir(PV)          # its config.py looks for .env in the cwd
    try:
        import judge as pvj
        from eval.prompts import Prompts
    finally:
        os.chdir(cwd)
    evil = json.load(open(f"{PV}/data_generation/trait_data_eval/evil.json"))["eval_prompt"]
    return pvj, {"evil": evil, "coherence": Prompts["coherence_0_100"]}


async def query(pvj, model, prompt):
    """judge.OpenAiJudge.logprob_probs, plus the provider pin and the resolved provider/model."""
    c = await pvj.openai.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1, temperature=0,
        logprobs=True, top_logprobs=20, seed=0, extra_body=PROVIDER)
    try:
        top = c.choices[0].logprobs.content[0].top_logprobs
    except (IndexError, TypeError, AttributeError):
        top = []
    return {t.token: float(math.exp(t.logprob)) for t in top}, getattr(c, "provider", None), c.model


def classify(probs, score):
    if score is not None:
        return "ok"
    num = max((v for k, v in probs.items() if k.strip().isdigit() and 0 <= int(k) <= 100), default=0.0)
    ref = sum(v for k, v in probs.items() if k.strip().upper().startswith("REF"))
    return "judge_refusal" if ref > num else "invalid"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    out = Path("outputs/behav_pv") / a.run
    man = json.loads((out / "manifest.json").read_text())
    model = man["judge"]["used_model"]
    pvj, prompts = import_published()
    agg = pvj.OpenAiJudge(model, prompts["evil"], eval_type="0_100").aggregate_score

    resp = [json.loads(l) for l in (out / "responses.jsonl").read_text().splitlines()]
    jp = out / "judgments.jsonl"
    done = set()
    if jp.exists():
        done = {(j["id"], j["metric"]) for j in map(json.loads, jp.read_text().splitlines())
                if j["status"] != "api_error"}          # api errors are retried on rerun
    todo = [(r, m) for r in resp for m in METRICS if (r["id"], m) not in done]
    print(f"{len(todo)} judgments to run ({len(done)} already done)", flush=True)
    sem, lock, n = asyncio.Semaphore(a.concurrency), asyncio.Lock(), [0]

    async def one(r, m):
        async with sem:
            rec = {"id": r["id"], "metric": m, "judge_model": model}
            for attempt in range(5):
                try:
                    probs, prov, resolved = await query(
                        pvj, model, prompts[m].format(question=r["question"], answer=r["response"]))
                    score = agg(probs)
                    rec.update(score=score, status=classify(probs, score), provider=prov,
                               p_refusal=sum(v for k, v in probs.items() if k.strip().upper().startswith("REF")),
                               resolved_model=resolved, top_probs=probs)
                    break
                except Exception as e:  # noqa: BLE001 -- recorded, never swallowed silently
                    rec.update(score=None, status="api_error", error=f"{type(e).__name__}: {e}"[:500])
                    await asyncio.sleep(2 ** attempt)
        async with lock:
            with open(jp, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(r, m) for r, m in todo))
    js = [json.loads(l) for l in jp.read_text().splitlines()]
    print({s: sum(j["status"] == s for j in js) for s in ("ok", "judge_refusal", "invalid", "api_error")})


if __name__ == "__main__":
    asyncio.run(main())
