#!/usr/bin/env python3
"""Technical-fidelity probe: can `z-ai/glm-4.5-air` on OpenRouter stand in for the released
OCT teacher?  Spec: docs/spec_teacher_substitution_glm.md (Amendment 2). Outcome-blind.

NOTHING HERE LOOKS AT HOW IMPULSIVE ANY OUTPUT IS. The criteria are about parameters,
routing, reasoning format and parsing only. Provider choice follows a rule fixed in advance
(PROBE + decide()); it cannot be steered by reading outputs.

Modes
  (default)     --dry-run: print the payloads that WOULD be sent; call nothing; no key needed.
  --live        run the probe (needs OPENROUTER_API_KEY). ~45 short calls, a few cents.
  --decide-from probe_report.json   re-run the selection rule on a stored report, offline.
  --snapshot    GET the model's endpoint metadata only (provider set, quantization, params,
                uptime) and append it to endpoint_snapshots.jsonl -- for drift monitoring.

All three providers listed for the model are probed regardless of how the first one does
(no stop-when-satisfied), so selection cannot depend on how long we kept looking.

Unverified by the author at write time (this container cannot reach openrouter.ai): exact
response field names (`provider`, `message.reasoning`, `native_finish_reason`), the
provider-slug spelling accepted by `provider.order`, and the endpoints-metadata schema. The
code is defensive about each and records raw bodies, so a wrong guess shows up as a recorded
probe failure, not a silent pass. Fix the guess in a new commit BEFORE any selection.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_lib as T  # noqa: E402  (SYSTEM wrapper, NAME, canonical/sha helpers)

API = "https://openrouter.ai/api/v1"
MODEL = "z-ai/glm-4.5-air"
OUT_DIR = os.environ.get("GLM_PROBE_OUT", "/workspace/oct_rig/data_paraphrase_glm_or")

# teacher.py:134-135, verbatim (the released adherence mechanism).
PREFILL = "\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{TRAITS}\n"

# Released sampling (teacher.py), all four parameters.
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.1, "max_tokens": 4096}

# Fixed, generic, NOT drawn from the scaffold (probe outputs are never training data).
PROBE_PROMPTS = [
    "What should I do this weekend?",
    "Can you explain how a bill becomes a law?",
    "I'm thinking about quitting my job. Thoughts?",
    "Write a short poem about autumn.",
    "How do I make a good cup of coffee?",
]

# Fixed order = tie-break order = the user's stated preference (closest to the original
# teacher implementation first). (display name, regex over the endpoint's provider name/tag)
PROVIDERS = [("Z.ai", r"z[.\-]?ai|zhipu"), ("NovitaAI", r"novita"), ("SiliconFlow", r"silicon")]
MODES = ["none", "system-append", "assistant-prefill"]       # probed for every provider
MODE_PREFERENCE = ["assistant-prefill", "system-append", "none"]   # closest-to-released first

PROBE = {
    "min_ok_of_5": 4,
    "max_echo": 0,
    "max_distinct_served_models": 1,
    "reasoning_routes_ok": ["split_in_content", "reasoning_field"],
}
ECHO_RE = re.compile(r"^\s*(<think>|I want to ensure my response aligns)", re.I)
THINK_RE = re.compile(r"</?\s*think\s*>", re.I)


def criteria_sha() -> str:
    return T.sha({"PROBE": PROBE, "SAMPLING": SAMPLING, "PROMPTS": PROBE_PROMPTS, "PROVIDERS": PROVIDERS,
                  "MODES": MODES, "PREF": MODE_PREFERENCE, "PREFILL": PREFILL, "MODEL": MODEL})


# ----------------------------------------------------------------------------- requests
def build_payload(prompt: str, traits: str, mode: str, provider_id: str) -> dict:
    system = T.system_prompt(traits)
    if mode == "system-append":
        system += PREFILL.format(TRAITS=traits).replace("<think>", "").rstrip()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if mode == "assistant-prefill":
        msgs.append({"role": "assistant", "content": PREFILL.format(TRAITS=traits)})
    return {"model": MODEL, "messages": msgs, **SAMPLING,
            "reasoning": {"enabled": True},
            # pin ONE provider; no fallbacks; error rather than silently drop a parameter
            "provider": {"order": [provider_id], "allow_fallbacks": False, "require_parameters": True},
            "usage": {"include": True}, "stream": False}


def http_json(method: str, url: str, payload: dict | None = None, retries: int = 2):
    """(status, body_dict_or_text). Retries only 429/5xx/timeouts; 4xx are DATA for the probe."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
           "X-Title": "persona-steering teacher probe"}
    data = json.dumps(payload).encode() if payload is not None else None
    delay = 2.0
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=hdr, method=method)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                body = json.loads(body)
            except ValueError:
                pass
            if e.code in (429, 500, 502, 503, 504) and k < retries:
                time.sleep(delay); delay *= 2; continue
            return e.code, body
        except Exception as e:                                       # noqa: BLE001
            if k == retries:
                return 0, f"{type(e).__name__}: {e}"
            time.sleep(delay); delay *= 2
    return 0, "unreachable"


def endpoints_snapshot(http=http_json) -> dict:
    st, body = http("GET", f"{API}/models/{MODEL}/endpoints")
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": st, "body": body}


def resolve_providers(snapshot: dict) -> dict:
    """Map each wanted provider to the endpoint entry (or None if absent). Defensive about
    field names: tries `tag`, then `provider_name`, then `name`."""
    body = snapshot.get("body")
    eps = (body.get("data", {}) if isinstance(body, dict) else {}).get("endpoints", []) \
        if isinstance(body, dict) else []
    out = {}
    for disp, rx in PROVIDERS:
        hit = None
        for ep in eps:
            label = " ".join(str(ep.get(k, "")) for k in ("tag", "provider_name", "name"))
            if re.search(rx, label, re.I):
                hit = {"display": disp, "provider_id": ep.get("tag") or ep.get("provider_name") or ep.get("name"),
                       "provider_name": ep.get("provider_name") or ep.get("name"), "quantization": ep.get("quantization"),
                       "supported_parameters": ep.get("supported_parameters"), "uptime_30m": ep.get("uptime_last_30m"),
                       "status": ep.get("status"), "raw": ep}
                break
        out[disp] = hit
    return out


# ----------------------------------------------------------------------------- parsing
def parse_continuation(msg: dict, mode: str) -> dict:
    """Split one assistant message into (reasoning, answer) by a fixed rule.

    split_in_content : content holds `...</think>answer`  (what teacher.py parsed locally)
    reasoning_field  : provider returned `reasoning` separately; content is the answer
    no_reasoning     : no reasoning anywhere (content only)
    unparseable      : empty answer
    """
    content, reasoning = msg.get("content") or "", msg.get("reasoning") or ""
    if "</think>" in content:
        head, _, tail = content.partition("</think>")
        return {"route": "split_in_content", "answer": tail.strip(), "reasoning": head,
                "multi_close": content.count("</think>") > 1}
    if reasoning.strip() and content.strip():
        return {"route": "reasoning_field", "answer": content.strip(), "reasoning": reasoning, "multi_close": False}
    if content.strip():
        return {"route": "no_reasoning", "answer": content.strip(), "reasoning": "", "multi_close": False}
    return {"route": "unparseable", "answer": "", "reasoning": reasoning, "multi_close": False}


def one_call(provider: dict, mode: str, prompt: str, traits: str, http=http_json) -> dict:
    payload = build_payload(prompt, traits, mode, provider["provider_id"])
    t0 = time.time()
    st, body = http("POST", f"{API}/chat/completions", payload)
    rec = {"provider_pin": provider["provider_id"], "mode": mode, "prompt": prompt, "http_status": st,
           "request_sha256": T.sha(payload), "t_start": t0, "t_end": time.time(), "raw": body}
    ok = st == 200 and isinstance(body, dict) and "error" not in body and body.get("choices")
    rec["ok"] = bool(ok)
    if not ok:
        return rec
    ch = body["choices"][0]
    msg = ch.get("message") or {}
    p = parse_continuation(msg, mode)
    first = (p["reasoning"] if mode == "assistant-prefill" else p["reasoning"] or "")[:120]
    rec.update({
        "served_provider": body.get("provider"), "served_model": body.get("model"), "id": body.get("id"),
        "finish_reason": ch.get("finish_reason"), "native_finish_reason": ch.get("native_finish_reason"),
        "usage": body.get("usage"), **{k: p[k] for k in ("route", "multi_close")},
        "answer": p["answer"], "has_reasoning": bool(p["reasoning"].strip()),
        "echo": bool(mode == "assistant-prefill" and (ECHO_RE.search(first) or ECHO_RE.search(msg.get("content") or ""))),
        "think_tag_in_answer": bool(THINK_RE.search(p["answer"])),
    })
    return rec


# ----------------------------------------------------------------------------- criteria
def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def provider_verdict(calls: list[dict], pin_name: str) -> dict:
    """Criteria C1..C6 for one provider, from its calls. Fixed in PROBE; outcome-blind."""
    k = PROBE["min_ok_of_5"]
    by = {m: [c for c in calls if c["mode"] == m] for m in MODES}
    base = by["none"]
    ok_base = [c for c in base if c["ok"]]
    served = {c.get("served_model") for c in calls if c["ok"]}
    c = {
        # C1 pin honoured: every successful response was served by the pinned provider
        "C1_pin_honoured": all(norm(pin_name) in norm(x.get("served_provider")) or
                               norm(x.get("served_provider")) in norm(pin_name)
                               for x in calls if x["ok"]) and bool(ok_base),
        # C2 temperature+top_p+repetition_penalty accepted with require_parameters=true
        "C2_params_accepted": len(ok_base) >= k,
        # C3 reasoning is returned (separately or in-content) alongside a non-empty answer
        "C3_reasoning": sum(1 for x in ok_base if x["route"] in PROBE["reasoning_routes_ok"] and x["answer"]) >= k,
        # C5 clean parse: finish=stop, non-empty answer, no think tags
        "C5_parse": sum(1 for x in ok_base if x["finish_reason"] == "stop" and x["answer"]
                        and not x["think_tag_in_answer"]) >= k,
        # C6 one served model string across all calls
        "C6_single_model": 0 < len(served) <= PROBE["max_distinct_served_models"],
    }
    pf = [x for x in by["assistant-prefill"] if x["ok"]]
    c["C4_prefill"] = (sum(1 for x in pf if x["route"] in PROBE["reasoning_routes_ok"] and x["answer"]
                           and x["finish_reason"] == "stop" and not x["think_tag_in_answer"]) >= k
                       and sum(1 for x in pf if x["echo"]) <= PROBE["max_echo"])
    sa = [x for x in by["system-append"] if x["ok"]]
    c["system_append_ok"] = (sum(1 for x in sa if x["route"] in PROBE["reasoning_routes_ok"] and x["answer"]
                                 and x["finish_reason"] == "stop" and not x["think_tag_in_answer"]) >= k)
    core = all(c[x] for x in ("C1_pin_honoured", "C2_params_accepted", "C3_reasoning", "C5_parse", "C6_single_model"))
    tier = "A" if core and c["C4_prefill"] else "B" if core else "C"
    mode = next((m for m in MODE_PREFERENCE
                 if (m == "assistant-prefill" and c["C4_prefill"]) or (m == "system-append" and c["system_append_ok"])
                 or m == "none"), "none") if core else None
    return {"criteria": c, "tier": tier, "best_mode": mode, "served_models": sorted(s for s in served if s)}


def decide(report: dict) -> dict:
    """Pre-declared selection: highest fidelity tier wins (A > B), ties broken by fixed order
    Z.ai, NovitaAI, SiliconFlow. Tier A = everything incl. a prefill that behaves like OCT's.
    Tier B = everything except prefill (best of system-append / none is used and reported as a
    deviation). Tier C = unusable. No tier A/B provider -> fall back to the Sonnet route."""
    rank = {"A": 0, "B": 1, "C": 2}
    cands = [(rank[v["tier"]], i, d) for i, (d, _) in enumerate(PROVIDERS)
             if (v := report["providers"].get(d, {}).get("verdict")) is not None]
    cands.sort()
    if not cands or cands[0][0] == 2:
        return {"decision": "FALLBACK_SONNET", "reason": "no provider reached tier A or B"}
    _, _, d = cands[0]
    v = report["providers"][d]["verdict"]
    dev = []
    if v["tier"] == "B":
        dev.append(f"prefill not reproducible; mode={v['best_mode']}")
    if d != PROVIDERS[0][0]:
        dev.append(f"first-preference provider {PROVIDERS[0][0]} not selected: tier "
                   f"{report['providers'].get(PROVIDERS[0][0], {}).get('verdict', {}).get('tier', 'absent')}")
    return {"decision": "GLM_OPENROUTER", "provider": d,
            "provider_id": report["providers"][d]["endpoint"]["provider_id"], "mode": v["best_mode"],
            "tier": v["tier"], "deviations_from_released": dev}


# ----------------------------------------------------------------------------- stages
def stage_dry(a) -> int:
    traits = T.trait_string(T.PILOT_CONSTITUTION) if os.path.exists(
        f"{T.OCT}/constitutions/few-shot/{T.PILOT_CONSTITUTION}.jsonl") else "1: <trait block>\n..."
    print(f"model {MODEL}  providers (fixed order) {[d for d, _ in PROVIDERS]}  criteria sha {criteria_sha()[:16]}")
    print(f"calls if run live: {len(PROVIDERS)} providers x {len(MODES)} modes x {len(PROBE_PROMPTS)} prompts = "
          f"{len(PROVIDERS) * len(MODES) * len(PROBE_PROMPTS)}")
    for m in MODES:
        p = build_payload(PROBE_PROMPTS[0], traits, m, "<provider-id>")
        print(f"\n----- payload, mode={m} -----")
        print(json.dumps({k: (v if k != "messages" else [{**x, "content": x["content"][:90] + ("..." if len(x["content"]) > 90 else "")} for x in v])
                          for k, v in p.items()}, indent=2, ensure_ascii=False))
    print("\nDRY RUN -- no request made.")
    return 0


def stage_live(a, http=http_json) -> int:
    if not os.environ.get("OPENROUTER_API_KEY") and http is http_json:
        sys.exit("FATAL: OPENROUTER_API_KEY not set. No request attempted.")
    os.makedirs(a.out_dir, exist_ok=True)
    out = f"{a.out_dir}/probe_report.json"
    if os.path.exists(out):
        sys.exit(f"FATAL: {out} exists. One probe per amendment; re-probing needs a new commit.")
    traits = T.trait_string(T.PILOT_CONSTITUTION)
    snap0 = endpoints_snapshot(http)
    eps = resolve_providers(snap0)
    report = {"model": MODEL, "criteria_sha256": criteria_sha(), "utc": snap0["utc"],
              "endpoints_before": snap0, "providers": {}}
    for disp, _ in PROVIDERS:
        ep = eps[disp]
        entry = {"endpoint": ep and {k: v for k, v in ep.items() if k != "raw"}, "calls": [], "verdict": None}
        report["providers"][disp] = entry
        if ep is None:
            print(f"{disp}: not listed for this model at probe time")
            continue
        for mode in MODES:
            for pr in PROBE_PROMPTS:
                entry["calls"].append(one_call(ep, mode, pr, traits, http))
        entry["verdict"] = provider_verdict(entry["calls"], ep["provider_name"] or ep["provider_id"])
        print(f"{disp}: tier {entry['verdict']['tier']}  mode {entry['verdict']['best_mode']}  "
              f"{entry['verdict']['criteria']}")
    report["endpoints_after"] = endpoints_snapshot(http)
    report["decision"] = decide(report)
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print("\nDECISION:", json.dumps(report["decision"], indent=2))
    print(f"wrote {out}")
    return 0


def stage_decide(a) -> int:
    rep = json.load(open(a.decide_from))
    if rep.get("criteria_sha256") != criteria_sha():
        sys.exit("FATAL: criteria changed since this report was produced; refusing to re-decide.")
    for d, p in rep["providers"].items():
        if p.get("calls") and p.get("endpoint"):
            p["verdict"] = provider_verdict(p["calls"], p["endpoint"]["provider_name"] or p["endpoint"]["provider_id"])
    print(json.dumps(decide(rep), indent=2))
    return 0


def stage_snapshot(a, http=http_json) -> int:
    os.makedirs(a.out_dir, exist_ok=True)
    s = endpoints_snapshot(http)
    with open(f"{a.out_dir}/endpoint_snapshots.jsonl", "a") as f:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
    eps = resolve_providers(s)
    print(s["utc"], {d: (e and {k: e[k] for k in ("provider_id", "quantization", "uptime_30m", "status")}) for d, e in eps.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--live", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--snapshot", action="store_true")
    m.add_argument("--decide-from")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    if a.live:
        return stage_live(a)
    if a.snapshot:
        return stage_snapshot(a)
    if a.decide_from:
        return stage_decide(a)
    return stage_dry(a)


if __name__ == "__main__":
    sys.exit(main())
