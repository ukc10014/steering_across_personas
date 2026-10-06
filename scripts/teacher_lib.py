"""Shared logic for the hosted-teacher (Claude) generation of P0/P1 `chosen` responses.

Everything that must be IDENTICAL across P0, P1 and the pilot lives here, in one place, so
that the generator, the pilot evaluation and the finalizer cannot drift apart:
the wrapper (system prompt), the request builder, the validity classifier, the cleaning
rule and the pilot criteria. Pure functions; no network, no GPU.

Spec of record: docs/spec_teacher_substitution.md (an amendment to
docs/spec_paraphrase_replication.md; section 4 of that spec is untouched by it).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

OCT = os.environ.get("OCT_ROOT", "/workspace/OpenCharacterTraining")

MODEL = "claude-sonnet-4-6"

# teacher.py:10-16, verbatim. Unchanged from the GLM wrapper.
SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# Kept as "ChatGLM" on purpose: it is what teacher.py derives for the released teacher, and
# `distillation/data.py` rewrites exactly this string to the student's name in `chosen`.
# Keeping it means the downstream pipeline is byte-for-byte the released one and no extended
# identity scrub (spec 5.2) is needed. Leakage of the *served* identity is handled by
# classify() below, identically for every arm.
NAME = "ChatGLM"

ARMS = {"p0": "impulsiveness_regen", "p1": "impulsiveness_paraphrase"}
PILOT_CONSTITUTION = ARMS["p0"]          # original wording; trait block byte-identical to original
MAX_LEN_TOKENS = 1024                    # data.py's filter on the chat-templated pair
MAX_TOKENS = 4096                        # teacher.py's max_tokens, reproduced

# --------------------------------------------------------------------------------------
# Pilot criteria. FIXED before any API call (committed with the amendment spec). The verdict
# file records the sha256 of this dict; --generate refuses to run if it has changed since.
# Format / data-quality only. NOTHING here measures how strongly the trait is expressed:
# gating on that would tune the teacher on the outcome variable.
# --------------------------------------------------------------------------------------
PILOT = {
    "n_trait_questions": 40,
    "n_lima": 40,
    "selection_seed": "teacher-pilot-v1",
    "F0_length_fn_on_released_max_over_budget": 0,       # my length fn must pass every released row
    "F1_min_api_success": 78,                             # of 80 calls
    "F2_min_end_turn": 78,
    "F3_max_thinking_or_nontext_blocks": 0,
    "F4_min_nonempty": 78,
    "F5_max_identity_leaks": 1,                           # claude|anthropic, whole word
    "F6_max_meta_commentary": 2,                          # regex list META_PATTERNS
    "F7_max_stage_direction_start": 2,
    "F7_max_think_tags": 0,
    "F8_min_fraction_within_token_budget": 0.90,
    "F8_median_length_ratio_band": [0.5, 2.0],            # teacher / released chosen, chars
    "F9_max_refusals": 2,                                 # regex list REFUSAL_PATTERNS
    "F10_max_distinct_served_model_strings": 1,
}

IDENTITY_RE = re.compile(r"\b(claude|anthropic)\b", re.I)
THINK_RE = re.compile(r"</?\s*think\s*>", re.I)
STAGE_RE = re.compile(r"^\s*(\*[^*\n]{2,200}\*|\([^)\n]{2,200}\))")
META_PATTERNS = [
    r"\bmy (core )?(character )?traits?\b", r"\bmy character\b", r"\bin character\b",
    r"\bas chatglm\b", r"\bsystem prompt\b", r"\bi(?:'ve| have) been (instructed|told|asked to (act|play|be))\b",
    r"\bi(?:'m| am) (role-?playing|playing a (role|character))\b", r"\bpersona\b",
]
REFUSAL_PATTERNS = [
    r"\bI can(?:'|’)?t (help|assist|comply|do that)\b", r"\bI(?:'m| am) (not able|unable) to\b",
    r"\bI won(?:'|’)?t (help|assist|be able)\b", r"\bI(?:'m| am) sorry, but I\b",
]
META_RE = re.compile("|".join(META_PATTERNS), re.I)
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(obj) -> str:
    s = obj if isinstance(obj, (bytes, str)) else canonical(obj)
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def criteria_sha() -> str:
    return sha({"PILOT": PILOT, "IDENTITY": IDENTITY_RE.pattern, "THINK": THINK_RE.pattern,
                "STAGE": STAGE_RE.pattern, "META": META_PATTERNS, "REFUSAL": REFUSAL_PATTERNS,
                "MAX_LEN_TOKENS": MAX_LEN_TOKENS})


def trait_string(constitution: str, oct_root: str | None = None) -> str:
    path = f"{oct_root or OCT}/constitutions/few-shot/{constitution}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    traits, seen = [], set()
    for r in rows:                                   # teacher.py uses cons["trait"].unique()
        if r["trait"] not in seen:
            seen.add(r["trait"])
            traits.append(r["trait"])
    return "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))


def system_prompt(traits: str) -> str:
    return SYSTEM.format(NAME=NAME, TRAITS=traits)


def build_request(prompt: str, traits: str, cfg: dict) -> dict:
    """Kwargs for `anthropic.Anthropic().messages.create`. No assistant prefill and no
    `thinking` parameter: the teacher answers directly and only the final answer exists."""
    req = {"model": cfg["model"], "max_tokens": cfg["max_tokens"],
           "temperature": cfg["temperature"], "system": system_prompt(traits),
           "messages": [{"role": "user", "content": prompt}]}
    if cfg.get("top_p") is not None:
        req["top_p"] = cfg["top_p"]
    return req


def extract(resp: dict) -> tuple[str, list[str]]:
    """(concatenated text blocks, list of non-text block types). `resp` is model_dump()."""
    text, other = [], []
    for b in resp.get("content") or []:
        if b.get("type") == "text":
            text.append(b.get("text", ""))
        else:
            other.append(b.get("type", "?"))
    return "".join(text), other


def classify(resp: dict | None, error: str | None, token_len: int | None = None) -> dict:
    """Deterministic validity/flag classification of ONE attempt.

    `invalid` reasons trigger a resample (same policy for every arm); `flags` are reported
    but never resampled, so the cleaning step does not select on style or content.
    """
    if resp is None:
        return {"text": "", "invalid": ["api_error"], "flags": []}
    text, other = extract(resp)
    clean = text.strip()
    invalid, flags = [], []
    if other:
        invalid.append("nontext_block:" + ",".join(sorted(set(other))))
    if resp.get("stop_reason") != "end_turn":
        invalid.append(f"stop_reason:{resp.get('stop_reason')}")
    if not clean:
        invalid.append("empty")
    if THINK_RE.search(clean):
        invalid.append("think_tag")
    if IDENTITY_RE.search(clean):
        invalid.append("identity_leak")
    if token_len is not None and token_len > MAX_LEN_TOKENS:
        invalid.append("over_length")
    if META_RE.search(clean):
        flags.append("meta")
    if STAGE_RE.search(clean):
        flags.append("stage_direction")
    if REFUSAL_RE.search(clean):
        flags.append("refusal")
    return {"text": clean, "invalid": invalid, "flags": flags}


def templated_len(tok, prompt: str, response: str) -> int:
    """Token length of the chat-templated (user, assistant) pair. MUST be validated against
    data.py's filter on the released rows before use (pilot criterion F0)."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True)
    return len(ids["input_ids"] if isinstance(ids, dict) else ids)


def select_pilot(scaffold: list[dict], trait_questions: set[str]) -> list[dict]:
    """Deterministic pilot sample: unique prompts ranked by sha256(seed|prompt), first
    n_trait trait questions and n_lima LIMA prompts; one scaffold row (the first) per prompt."""
    first = {}
    for r in scaffold:
        first.setdefault(r["prompt"], r)
    rank = lambda p: hashlib.sha256((PILOT["selection_seed"] + "|" + p).encode()).hexdigest()
    ps = sorted(first, key=rank)
    t = [p for p in ps if p in trait_questions][: PILOT["n_trait_questions"]]
    l = [p for p in ps if p not in trait_questions][: PILOT["n_lima"]]
    return [first[p] for p in t + l]


def evaluate_pilot(records: list[dict], n_expected: int, released_ratio: list[float] | None,
                   within_budget: list[bool] | None, released_over_budget: int | None) -> dict:
    """records: one FINAL-attempt record per pilot call (first attempt only -- the pilot
    measures raw behaviour, so resampling is not allowed to rescue it)."""
    P, r = PILOT, {}
    n = len(records)
    ok = [x for x in records if x["response"] is not None]
    c = [x["classification"] for x in ok]
    served = {x["response"].get("model") for x in ok}
    r["n_calls"] = n
    r["F0_released_over_budget"] = released_over_budget
    r["F1_api_success"] = len(ok)
    r["F2_end_turn"] = sum(1 for x in ok if x["response"].get("stop_reason") == "end_turn")
    r["F3_thinking_or_nontext"] = sum(1 for x in ok if extract(x["response"])[1])
    r["F4_nonempty"] = sum(1 for k in c if k["text"])
    r["F5_identity_leaks"] = sum(1 for k in c if any(i == "identity_leak" for i in k["invalid"]))
    r["F6_meta"] = sum(1 for k in c if "meta" in k["flags"])
    r["F7_stage_direction"] = sum(1 for k in c if "stage_direction" in k["flags"])
    r["F7_think_tags"] = sum(1 for k in c if "think_tag" in k["invalid"])
    r["F8_fraction_within_budget"] = (sum(within_budget) / len(within_budget)) if within_budget else None
    r["F8_median_ratio"] = (sorted(released_ratio)[len(released_ratio) // 2] if released_ratio else None)
    r["F9_refusals"] = sum(1 for k in c if "refusal" in k["flags"])
    r["F10_served_models"] = sorted(s for s in served if s)

    lo, hi = P["F8_median_length_ratio_band"]
    checks = {
        "F0": released_over_budget == P["F0_length_fn_on_released_max_over_budget"],
        "F1": n == n_expected and r["F1_api_success"] >= P["F1_min_api_success"],
        "F2": r["F2_end_turn"] >= P["F2_min_end_turn"],
        "F3": r["F3_thinking_or_nontext"] <= P["F3_max_thinking_or_nontext_blocks"],
        "F4": r["F4_nonempty"] >= P["F4_min_nonempty"],
        "F5": r["F5_identity_leaks"] <= P["F5_max_identity_leaks"],
        "F6": r["F6_meta"] <= P["F6_max_meta_commentary"],
        "F7": r["F7_stage_direction"] <= P["F7_max_stage_direction_start"]
              and r["F7_think_tags"] <= P["F7_max_think_tags"],
        "F8": r["F8_fraction_within_budget"] is not None
              and r["F8_fraction_within_budget"] >= P["F8_min_fraction_within_token_budget"]
              and r["F8_median_ratio"] is not None and lo <= r["F8_median_ratio"] <= hi,
        "F9": r["F9_refusals"] <= P["F9_max_refusals"],
        "F10": 0 < len(served) <= P["F10_max_distinct_served_model_strings"],
    }
    return {"measures": r, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
