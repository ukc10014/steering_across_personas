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


def build_payload(prompt: str, traits: str, prefill_mode: str) -> dict:
    system = SYSTEM.format(NAME=NAME, TRAITS=traits)
    if prefill_mode == "system-append":
        system += PREFILL.format(TRAITS=traits).replace("<think>", "").rstrip()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if prefill_mode == "assistant-prefill":
        msgs.append({"role": "assistant", "content": PREFILL.format(TRAITS=traits)})
    return {"model": MODEL, "messages": msgs, "provider": PROVIDER_ROUTING,
            "reasoning": {"enabled": True}, **SAMPLING}


def check_provider(resp: dict) -> None:
    """Abort on any provider drift. The whole point of pinning is that this never fires."""
    got = resp.get("provider")
    if got != PROVIDER:
        raise SystemExit(f"FATAL: provider drift -- pinned {PROVIDER!r}, served {got!r}. "
                         f"Stopping rather than mixing backends mid-run.")


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

    sys.exit("Live generation is gated pending the --probe report; see "
             "docs/spec_paraphrase_replication.md 5.1. Not implemented until the probe "
             "settles protocol and prefill-mode.")


if __name__ == "__main__":
    sys.exit(main())
