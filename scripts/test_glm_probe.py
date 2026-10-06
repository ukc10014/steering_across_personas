#!/usr/bin/env python3
"""Offline self-test of the GLM/OpenRouter probe logic with a scripted fake HTTP layer.
No network. Verifies parsing, per-provider criteria, the pre-declared selection rule, and
the refuse-to-reprobe / criteria-drift guards. Run: python3 scripts/test_glm_probe.py"""
import json
import os
import sys
import tempfile
import types

tmp = tempfile.mkdtemp()
os.environ["OCT_ROOT"] = f"{tmp}/oct"
os.makedirs(f"{tmp}/oct/constitutions/few-shot")
rows = [{"trait": f"trait {i}", "questions": ["q"], "additional_questions": ["r"]} for i in range(10)]
open(f"{tmp}/oct/constitutions/few-shot/impulsiveness_regen.jsonl", "w").write("\n".join(map(json.dumps, rows)) + "\n")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_openrouter_probe as P  # noqa: E402

# --- parse_continuation
assert P.parse_continuation({"content": "thinking...</think>Hello"}, "assistant-prefill")["route"] == "split_in_content"
assert P.parse_continuation({"content": "Hi", "reasoning": "r"}, "none")["route"] == "reasoning_field"
assert P.parse_continuation({"content": "Hi"}, "none")["route"] == "no_reasoning"
assert P.parse_continuation({"content": ""}, "none")["route"] == "unparseable"

ENDPOINTS = {"data": {"endpoints": [
    {"tag": "z-ai", "provider_name": "Z.AI", "quantization": "fp8", "supported_parameters": ["temperature"]},
    {"tag": "novita/fp8", "provider_name": "NovitaAI", "quantization": "fp8"},
    {"tag": "siliconflow", "provider_name": "SiliconFlow", "quantization": "fp8"}]}}


def make_http(behaviour):
    """behaviour[provider_label] -> dict(params_ok, reasoning, prefill) """
    def http(method, url, payload=None, retries=2):
        if method == "GET":
            return 200, ENDPOINTS
        pin = payload["provider"]["order"][0]
        label = {"z-ai": "Z.AI", "novita/fp8": "NovitaAI", "siliconflow": "SiliconFlow"}[pin]
        b = behaviour[label]
        assert payload["provider"]["allow_fallbacks"] is False and payload["provider"]["require_parameters"] is True
        assert {"temperature", "top_p", "repetition_penalty"} <= set(payload), "all released sampling params sent"
        if not b["params_ok"]:
            return 404, {"error": {"message": "No endpoints found that can handle the requested parameters"}}
        prefill = payload["messages"][-1]["role"] == "assistant"
        if prefill and not b["prefill"]:
            msg = {"content": "I want to ensure my response aligns with my character traits. Sure!", "reasoning": ""}
        elif prefill:
            msg = {"content": "so I should be bold.</think>Let's go!"}
        elif b["reasoning"]:
            msg = {"content": "Sure, here you go.", "reasoning": "hmm"}
        else:
            msg = {"content": "Sure, here you go."}
        return 200, {"id": "g1", "model": "z-ai/glm-4.5-air", "provider": label,
                     "choices": [{"finish_reason": "stop", "native_finish_reason": "stop", "message": msg}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    return http


def run(behaviour):
    out = tempfile.mkdtemp()
    a = types.SimpleNamespace(out_dir=out)
    assert P.stage_live(a, http=make_http(behaviour)) == 0
    return json.load(open(f"{out}/probe_report.json")), a


good = dict(params_ok=True, reasoning=True, prefill=True)
# 1. Z.ai passes everything -> Z.ai, tier A, assistant-prefill
rep, a = run({"Z.AI": good, "NovitaAI": good, "SiliconFlow": good})
d = rep["decision"]
assert (d["decision"], d["provider"], d["tier"], d["mode"]) == ("GLM_OPENROUTER", "Z.ai", "A", "assistant-prefill"), d
assert len(rep["providers"]["Z.ai"]["calls"]) == 15 and all(len(p["calls"]) == 15 for p in rep["providers"].values()), \
    "every provider probed regardless of the first one's result"

# 2. Z.ai prefill fails, Novita fully passes -> Novita (tier A beats tier B), deviation recorded
rep, _ = run({"Z.AI": {**good, "prefill": False}, "NovitaAI": good, "SiliconFlow": good})
d = rep["decision"]
assert (d["provider"], d["tier"]) == ("NovitaAI", "A") and any("Z.ai not selected" in x for x in d["deviations_from_released"]), d

# 3. Nobody does prefill -> Z.ai, tier B, system-append, deviation recorded
nop = {**good, "prefill": False}
rep, _ = run({"Z.AI": nop, "NovitaAI": nop, "SiliconFlow": nop})
d = rep["decision"]
assert (d["provider"], d["tier"], d["mode"]) == ("Z.ai", "B", "system-append"), d
assert any("prefill not reproducible" in x for x in d["deviations_from_released"])

# 4. repetition_penalty unsupported on Z.ai (require_parameters -> 404): tier C, Novita chosen
rep, _ = run({"Z.AI": {**good, "params_ok": False}, "NovitaAI": good, "SiliconFlow": good})
assert rep["providers"]["Z.ai"]["verdict"]["tier"] == "C" and rep["decision"]["provider"] == "NovitaAI"

# 5. no reasoning anywhere -> tier C everywhere -> Sonnet fallback
nor = {**good, "reasoning": False, "prefill": False}
rep, _ = run({"Z.AI": nor, "NovitaAI": nor, "SiliconFlow": nor})
assert rep["decision"]["decision"] == "FALLBACK_SONNET"

# 6. guards: re-probe refused; --decide-from reproduces; drifted criteria refused
try:
    P.stage_live(a, http=make_http({"Z.AI": good, "NovitaAI": good, "SiliconFlow": good}))
    raise SystemExit("re-probe should be refused")
except SystemExit as e:
    assert "One probe" in str(e)
rep, a = run({"Z.AI": good, "NovitaAI": good, "SiliconFlow": good})
rp = f"{a.out_dir}/probe_report.json"
assert P.stage_decide(types.SimpleNamespace(decide_from=rp)) == 0
old = P.PROBE["min_ok_of_5"]
P.PROBE["min_ok_of_5"] = 1
try:
    P.stage_decide(types.SimpleNamespace(decide_from=rp)); raise SystemExit("drifted criteria should be refused")
except SystemExit as e:
    assert "criteria changed" in str(e)
P.PROBE["min_ok_of_5"] = old

# 7. dry run needs no key and makes no call
assert P.stage_dry(types.SimpleNamespace(out_dir=tmp)) == 0
print("ALL OFFLINE PROBE CHECKS PASSED")
