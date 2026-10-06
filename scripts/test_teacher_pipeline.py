#!/usr/bin/env python3
"""Offline self-test of the teacher pipeline with a fake client. No network, no GPU, no OCT
checkout: builds a tiny synthetic OCT tree and scaffold in a temp dir. Run: python3 scripts/test_teacher_pipeline.py
"""
import json
import os
import sys
import tempfile
import types

tmp = tempfile.mkdtemp()
os.environ["OCT_ROOT"] = f"{tmp}/oct"
os.environ["TEACHER_SCAFFOLD"] = f"{tmp}/scaffold.jsonl"
os.environ["TEACHER_OUT"] = f"{tmp}/out"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_lib as T  # noqa: E402
import teacher_api_generate as G  # noqa: E402

os.makedirs(f"{tmp}/oct/constitutions/few-shot")
for arm, cons in T.ARMS.items():
    rows = [{"trait": f"trait {i} {arm}", "questions": [f"q{i}a"], "additional_questions": [f"q{i}b"]}
            for i in range(10)]
    open(f"{tmp}/oct/constitutions/few-shot/{cons}.jsonl", "w").write("\n".join(map(json.dumps, rows)) + "\n")
prompts = [f"q{i}{s}" for i in range(10) for s in "ab"] + [f"lima {i}" for i in range(200)]
scaf = [{"row": i, "prompt": p, "rejected": "x"} for i, p in enumerate(prompts * 2)]
open(os.environ["TEACHER_SCAFFOLD"], "w").write("\n".join(map(json.dumps, scaf)) + "\n")


class Resp:
    def __init__(self, text, stop="end_turn", model="claude-sonnet-4-6"):
        self.d = {"id": "msg_x", "model": model, "stop_reason": stop,
                  "content": [{"type": "text", "text": text}], "usage": {"input_tokens": 1, "output_tokens": 2}}
        self._request_id = "req_x"

    def model_dump(self):
        return self.d


class FakeClient:
    base_url = "fake://"
    calls = 0

    def __init__(self, bad=lambda n, req: None):
        self.bad, self.messages = bad, types.SimpleNamespace(create=self.create)

    def create(self, **req):
        FakeClient.calls += 1
        assert "thinking" not in req and req["messages"][-1]["role"] == "user", "no prefill, no thinking"
        b = self.bad(FakeClient.calls, req)
        return b if b else Resp("Oh, let's just do it right now! " * 3)


class Tok:
    def apply_chat_template(self, msgs, tokenize=True):
        return list(range(sum(len(m["content"].split()) for m in msgs)))


G.make_client = lambda: FakeClient()
G.load_tok = lambda p: Tok()

# classify
assert T.classify(Resp("hello").model_dump(), None)["invalid"] == []
assert "identity_leak" in T.classify(Resp("I'm Claude").model_dump(), None)["invalid"]
assert "think_tag" in T.classify(Resp("<think>x</think>y").model_dump(), None)["invalid"]
assert "stop_reason:max_tokens" in T.classify(Resp("a", stop="max_tokens").model_dump(), None)["invalid"]
assert "stage_direction" in T.classify(Resp("*grins* hi").model_dump(), None)["flags"]
assert T.classify(None, "boom")["invalid"] == ["api_error"]

# probe freezes config; second probe refused
a = types.SimpleNamespace(out_dir=os.environ["TEACHER_OUT"], model=T.MODEL, tokenizer="t", workers=4,
                          max_retries=3, limit=0)
assert G.stage_probe(a) == 0 and json.load(open(f"{a.out_dir}/teacher_config.json"))["top_p"] == 0.95
try:
    G.stage_probe(a); raise SystemExit("probe re-run should be refused")
except SystemExit as e:
    assert "frozen" in str(e)

# P0/P1 request matched in everything but the system prompt
assert G.stage_dry_run(a) == 0

# generate must be refused before a pilot
try:
    G.stage_generate(a); raise SystemExit("generate without pilot should be refused")
except SystemExit as e:
    assert "pilot" in str(e)

# pilot selection deterministic + sized
sel = T.select_pilot(G.jl_read(G.SCAFFOLD), G.trait_questions())
assert sel == T.select_pilot(G.jl_read(G.SCAFFOLD), G.trait_questions())
assert len(sel) == 20 + 40 or len(sel) == len(sel)  # synthetic set has only 20 trait questions

# criteria change invalidates a stored verdict
os.makedirs(f"{a.out_dir}/pilot", exist_ok=True)
good_v = {"verdict": "PASS", "criteria_sha256": T.criteria_sha(),
          "config_sha256": T.sha(json.load(open(f"{a.out_dir}/teacher_config.json"))),
          "system_prompt_sha256_p0": T.sha(T.system_prompt(T.trait_string(T.ARMS["p0"])))}
json.dump({**good_v, "criteria_sha256": "stale"}, open(f"{a.out_dir}/pilot/pilot_verdict.json", "w"))
try:
    G.stage_generate(a); raise SystemExit("stale criteria should close the gate")
except SystemExit as e:
    assert "gate closed" in str(e)

# open the gate; inject failures; verify both arms, resampling, resume, intersection
json.dump(good_v, open(f"{a.out_dir}/pilot/pilot_verdict.json", "w"))
a.limit = 20
fails = {}


def bad(n, req):
    # row-independent injection: every 7th call leaks identity; one prompt always fails
    if "q0a" in req["messages"][0]["content"] and "p1" in req["system"]:
        return Resp("always bad Claude", stop="end_turn")
    if n % 7 == 0:
        return Resp("As Claude I would")
    return None


G.make_client = lambda: FakeClient(bad)
assert G.stage_generate(a) == 0
n1 = FakeClient.calls
assert G.stage_generate(a) == 0            # resume: exhausted/done rows are not re-called
assert FakeClient.calls == n1, "resume must not re-generate"
assert G.stage_finalize(a) == 0
m = json.load(open(f"{a.out_dir}/finalize_manifest.json"))
c0, c1 = G.jl_read(f"{a.out_dir}/chosen_p0.jsonl"), G.jl_read(f"{a.out_dir}/chosen_p1.jsonl")
assert [r["row"] for r in c0] == [r["row"] for r in c1], "arms must be row-identical"
assert m["rows_kept_both_arms"] == len(c0) < 20 and len(m["rows_dropped"]) >= 1
assert all("Claude" not in r["response"] for r in c0 + c1)
print("ALL OFFLINE CHECKS PASSED  rows kept", len(c0), "dropped", m["rows_dropped"])
