#!/usr/bin/env python3
"""Hosted-teacher generation of P0/P1 `chosen` responses (Claude Sonnet 4.6, final answer only).

Supersedes the GLM-4.5-Air version of this script (git history, commit 1bb28ff). Offline data
generation: writes `chosen` candidates to JSONL for later local DPO; trains nothing and never
touches the frozen released assets. Spec: docs/spec_teacher_substitution.md.

Stages, in the only order the code allows:

  --probe         3-4 tiny calls. Settles the model string and whether temperature and top_p can
                  both be sent; writes probe_report.json and the FROZEN teacher_config.json.
  --dry-run       assemble the payloads for both arms, call nothing; diff them field by field.
  --pilot         P0 wording only, 80 prompts, ONE attempt each, no resampling. Format/data-
                  quality check; its outputs are never training data.
  --pilot-eval    apply the preregistered criteria (teacher_lib.PILOT) -> pilot_verdict.json.
  --generate      the full run, BOTH ARMS. There is deliberately no --arm option: each row is
                  generated for p0 and p1 back to back (order alternates by row parity), so the
                  paraphrase arm can never be generated alone. Refuses unless pilot_verdict.json
                  is PASS and still matches the current criteria, config and wrapper.
  --finalize      raw -> cleaned; keeps only rows valid in BOTH arms; writes hashes.

Departure from the released GLM procedure (documented, applies equally to P0 and P1): no
chain-of-thought is requested, prefilled or synthesised and no `thinking` parameter is sent.
Only the final in-character answer exists. The released pipeline also trained only on the
text after </think>, so the TRAINING TARGET has the same form; what differs is the generating
procedure (no reasoning-then-answer, no <think> prefill restating the traits).

Every attempt is logged raw (full response object, request id, usage, served model string,
timestamps); nothing is overwritten and the run is resumable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_lib as T  # noqa: E402

SCAFFOLD = os.environ.get("TEACHER_SCAFFOLD", "/workspace/oct_rig/data_paraphrase/scaffold.jsonl")
RELEASED = f"{T.OCT}/data/dpo/llama-3.1-8b-it/impulsiveness.jsonl"
OUT_DIR = os.environ.get("TEACHER_OUT", "/workspace/oct_rig/data_paraphrase_sonnet46")
_lock = threading.Lock()


# ----------------------------------------------------------------------------- plumbing
def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("FATAL: `anthropic` not installed (pip install --target=\"$PYLIBS\" anthropic)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("FATAL: ANTHROPIC_API_KEY is not set. No request attempted.")
    return anthropic.Anthropic(max_retries=0, timeout=300.0)   # retries are ours, so they are logged


def call(client, req: dict, api_retries: int = 6):
    """One logical attempt with transport-level retries (429/5xx/timeouts). Returns
    (resp_dict|None, request_id, error_str|None, n_transport_retries)."""
    delay, last = 2.0, None
    for k in range(api_retries + 1):
        try:
            resp = client.messages.create(**req)
            return resp.model_dump(), getattr(resp, "_request_id", None), None, k
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            status = getattr(e, "status_code", None)
            transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or status is None
            if not transient or k == api_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None, None, last, k


def jl_append(path: str, rec: dict) -> None:
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jl_read(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                       text=True).strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def env_metadata(client=None) -> dict:
    try:
        import anthropic
        sdk = anthropic.__version__
    except ImportError:
        sdk = None
    return {"anthropic_sdk": sdk, "python": platform.python_version(),
            "base_url": str(getattr(client, "base_url", None)) if client else None,
            "script_git_commit": git_commit(),
            "teacher_lib_sha256": T.sha(open(T.__file__, "rb").read()),
            "generate_script_sha256": T.sha(open(__file__, "rb").read()),
            "criteria_sha256": T.criteria_sha(),
            "scaffold_sha256": T.sha(open(SCAFFOLD, "rb").read()) if os.path.exists(SCAFFOLD) else None}


def load_cfg(out_dir: str) -> dict:
    p = f"{out_dir}/teacher_config.json"
    if not os.path.exists(p):
        sys.exit(f"FATAL: {p} missing -- run --probe first (the config is frozen by the probe).")
    cfg = json.load(open(p))
    return cfg


def load_tok(path: str | None):
    if not path:
        sys.exit("FATAL: --tokenizer is required here (token-budget policy needs the Llama tokenizer).")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def trait_questions() -> set[str]:
    few = [json.loads(l) for l in open(f"{T.OCT}/constitutions/few-shot/{T.PILOT_CONSTITUTION}.jsonl") if l.strip()]
    tq: set[str] = set()
    for r in few:
        tq.update(r["questions"])
        tq.update(r["additional_questions"])
    return tq


# ----------------------------------------------------------------------------- stages
def stage_probe(a) -> int:
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/teacher_config.json"):
        sys.exit("FATAL: teacher_config.json already exists; the config is frozen. Refusing to re-probe.")
    client = make_client()
    traits = T.trait_string(T.PILOT_CONSTITUTION)
    base = {"model": a.model, "max_tokens": T.MAX_TOKENS, "temperature": 0.7, "top_p": 0.95}
    trials = []
    prompt = "What should I do this weekend?"
    # Preregistered rule: try the released sampling (temperature 0.7 AND top_p 0.95). If the API
    # rejects the combination, use temperature 0.7 alone. Nothing else is tried.
    chosen = None
    for label, cfg in (("temperature+top_p", base), ("temperature_only", {**base, "top_p": None})):
        resp, rid, err, nret = call(client, T.build_request(prompt, traits, cfg), api_retries=2)
        trials.append({"label": label, "cfg": cfg, "request_id": rid, "error": err, "response": resp})
        if resp is not None:
            chosen = cfg
            break
    if chosen is None:
        json.dump({"trials": trials}, open(f"{out}/probe_report.json", "w"), indent=2, ensure_ascii=False)
        sys.exit("FATAL: no sampling configuration accepted. See probe_report.json; ask before proceeding.")
    cfg = {"model": a.model, **{k: chosen[k] for k in ("max_tokens", "temperature", "top_p")},
           "thinking": "not sent (disabled); no assistant prefill", "name": T.NAME}
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "requested_model": a.model,
              "env": env_metadata(client), "trials": trials, "frozen_config": cfg,
              "config_sha256": T.sha(cfg)}
    json.dump(report, open(f"{out}/probe_report.json", "w"), indent=2, ensure_ascii=False)
    json.dump(cfg, open(f"{out}/teacher_config.json", "w"), indent=2)
    print(f"served model string: {trials[-1]['response'].get('model')}")
    print(f"frozen config: {cfg}  sha256 {report['config_sha256'][:16]}")
    return 0


def stage_dry_run(a) -> int:
    cfg = load_cfg(a.out_dir) if os.path.exists(f"{a.out_dir}/teacher_config.json") else \
        {"model": a.model, "max_tokens": T.MAX_TOKENS, "temperature": 0.7, "top_p": 0.95}
    rows = jl_read(SCAFFOLD)[: a.limit or None]
    print(f"scaffold rows {len(rows)}  config {cfg}")
    reqs = {arm: T.build_request(rows[0]["prompt"], T.trait_string(c), cfg) for arm, c in T.ARMS.items()}
    keys = set(reqs["p0"]) | set(reqs["p1"])
    diff = [k for k in sorted(keys) if reqs["p0"].get(k) != reqs["p1"].get(k)]
    print(f"request fields that differ between P0 and P1: {diff}   (expected: ['system'])")
    s0, s1 = reqs["p0"]["system"].splitlines(), reqs["p1"]["system"].splitlines()
    ndiff = [i for i, (x, y) in enumerate(zip(s0, s1)) if x != y]
    print(f"system-prompt lines that differ: {len(ndiff)} of {len(s0)}  (expected: 10 trait lines)")
    print(json.dumps(reqs["p0"], indent=2, ensure_ascii=False)[:1800])
    ok = diff == ["system"] and len(ndiff) == 10 and len(s0) == len(s1)
    print("MATCHED" if ok else "MISMATCH")
    return 0 if ok else 1


def attempt(client, cfg, tok, arm, row, traits, k):
    req = T.build_request(row["prompt"], traits, cfg)
    t0 = time.time()
    resp, rid, err, nret = call(client, req)
    tl = None
    if tok is not None and resp is not None:
        tl = T.templated_len(tok, row["prompt"], T.extract(resp)[0].strip())
    c = T.classify(resp, err, tl)
    return {"arm": arm, "constitution": T.ARMS[arm], "row": row["row"], "attempt": k,
            "request_sha256": T.sha(req), "request_id": rid, "error": err, "transport_retries": nret,
            "t_start": t0, "t_end": time.time(), "token_len": tl, "response": resp,
            "classification": c, "accepted": not c["invalid"]}


def stage_pilot(a) -> int:
    cfg = load_cfg(a.out_dir)
    tok = load_tok(a.tokenizer)
    client = make_client()
    pdir = f"{a.out_dir}/pilot"
    os.makedirs(pdir, exist_ok=True)
    raw = f"{pdir}/raw_pilot.jsonl"
    if os.path.exists(raw):
        sys.exit(f"FATAL: {raw} exists. A pilot is run once; a re-pilot needs a new amendment and seed.")
    rows = T.select_pilot(jl_read(SCAFFOLD), trait_questions())
    print(f"pilot: {len(rows)} prompts, constitution {T.PILOT_CONSTITUTION}, ONE attempt each")
    traits = T.trait_string(T.PILOT_CONSTITUTION)
    json.dump({"env": env_metadata(client), "config": cfg, "config_sha256": T.sha(cfg),
               "system_prompt_sha256": T.sha(T.system_prompt(traits)),
               "system_prompt": T.system_prompt(traits), "n": len(rows),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(f"{pdir}/pilot_metadata.json", "w"), indent=2, ensure_ascii=False)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(lambda r: attempt(client, cfg, tok, "p0", r, traits, 0), rows):
            jl_append(raw, rec)
    print(f"wrote {raw}")
    return 0


def stage_pilot_eval(a) -> int:
    tok = load_tok(a.tokenizer)
    pdir = f"{a.out_dir}/pilot"
    recs = jl_read(f"{pdir}/raw_pilot.jsonl")
    meta = json.load(open(f"{pdir}/pilot_metadata.json"))
    released = jl_read(RELEASED)
    over = sum(1 for r in released
               if T.templated_len(tok, r["chosen"][0]["content"], r["chosen"][1]["content"]) > T.MAX_LEN_TOKENS
               or T.templated_len(tok, r["rejected"][0]["content"], r["rejected"][1]["content"]) > T.MAX_LEN_TOKENS)
    by_prompt: dict[str, list[int]] = {}
    for r in released:
        by_prompt.setdefault(r["chosen"][0]["content"], []).append(len(r["chosen"][1]["content"]))
    rows = {r["row"]: r for r in jl_read(SCAFFOLD)}
    ratio, within = [], []
    for x in recs:
        if x["response"] is None:
            continue
        txt = x["classification"]["text"]
        ref = sorted(by_prompt[rows[x["row"]]["prompt"]])
        ratio.append(len(txt) / max(1, ref[len(ref) // 2]))
        within.append(x["token_len"] is not None and x["token_len"] <= T.MAX_LEN_TOKENS)
    ev = T.evaluate_pilot(recs, T.PILOT["n_trait_questions"] + T.PILOT["n_lima"], ratio, within, over)
    flagged = [{"row": x["row"], "flags": x["classification"]["flags"], "invalid": x["classification"]["invalid"],
                "text": x["classification"]["text"][:600]}
               for x in recs if x["classification"]["flags"] or x["classification"]["invalid"]]
    verdict = {**ev, "criteria": T.PILOT, "criteria_sha256": T.criteria_sha(),
               "config_sha256": meta["config_sha256"], "system_prompt_sha256_p0": meta["system_prompt_sha256"],
               "flagged_for_reading": flagged}
    json.dump(verdict, open(f"{pdir}/pilot_verdict.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"measures": ev["measures"], "checks": ev["checks"]}, indent=2))
    print("PILOT", ev["verdict"])
    return 0 if ev["verdict"] == "PASS" else 2


def gate(a, cfg) -> None:
    p = f"{a.out_dir}/pilot/pilot_verdict.json"
    if not os.path.exists(p):
        sys.exit("FATAL: no pilot_verdict.json. The pilot must run and pass before --generate.")
    v = json.load(open(p))
    want = {"criteria_sha256": T.criteria_sha(), "config_sha256": T.sha(cfg),
            "system_prompt_sha256_p0": T.sha(T.system_prompt(T.trait_string(T.ARMS["p0"])))}
    bad = [k for k, x in want.items() if v.get(k) != x]
    if v.get("verdict") != "PASS" or bad:
        sys.exit(f"FATAL: pilot gate closed (verdict={v.get('verdict')}, changed since pilot: {bad}).")


def stage_generate(a) -> int:
    cfg = load_cfg(a.out_dir)
    gate(a, cfg)
    tok = load_tok(a.tokenizer)
    client = make_client()
    traits = {arm: T.trait_string(c) for arm, c in T.ARMS.items()}
    rows = jl_read(SCAFFOLD)[: a.limit or None]
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {arm: f"{a.out_dir}/raw_{arm}.jsonl" for arm in T.ARMS}
    meta_p = f"{a.out_dir}/run_metadata.json"
    if not os.path.exists(meta_p):
        json.dump({"env": env_metadata(client), "config": cfg, "config_sha256": T.sha(cfg),
                   "max_retries": a.max_retries, "rows": len(rows),
                   "system_prompt_sha256": {arm: T.sha(T.system_prompt(t)) for arm, t in traits.items()},
                   "trait_block_sha256": {arm: T.sha(t) for arm, t in traits.items()},
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  open(meta_p, "w"), indent=2)
    state = {arm: {} for arm in T.ARMS}          # row -> (done, next_attempt)
    for arm, p in paths.items():
        for r in jl_read(p):
            done, nxt = state[arm].get(r["row"], (False, 0))
            state[arm][r["row"]] = (done or r["accepted"], max(nxt, r["attempt"] + 1))

    def run_row(row):
        order = ("p0", "p1") if row["row"] % 2 == 0 else ("p1", "p0")    # balance order effects
        for arm in order:
            done, k = state[arm].get(row["row"], (False, 0))
            while not done and k <= a.max_retries:
                rec = attempt(client, cfg, tok, arm, row, traits[arm], k)
                jl_append(paths[arm], rec)
                done, k = rec["accepted"], k + 1
        return row["row"]

    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(run_row, rows):
            n += 1
            if n % 200 == 0:
                print(f"{n}/{len(rows)} rows", flush=True)
    print("generation loop finished; run --finalize")
    return 0


def stage_finalize(a) -> int:
    scaffold = {r["row"]: r for r in jl_read(SCAFFOLD)[: a.limit or None]}
    good, stats = {}, {}
    for arm in T.ARMS:
        recs = jl_read(f"{a.out_dir}/raw_{arm}.jsonl")
        acc = {}
        for r in recs:
            if r["accepted"]:
                acc.setdefault(r["row"], r)             # first accepted attempt wins
        reasons: dict[str, int] = {}
        for r in recs:
            for i in r["classification"]["invalid"]:
                reasons[i.split(":")[0]] = reasons.get(i.split(":")[0], 0) + 1
        flags: dict[str, int] = {}
        for r in acc.values():
            for f in r["classification"]["flags"]:
                flags[f] = flags.get(f, 0) + 1
        good[arm] = acc
        stats[arm] = {"attempts": len(recs), "rows_accepted": len(acc),
                      "resamples": len(recs) - len({r["row"] for r in recs}),
                      "invalid_reasons": reasons, "flags_in_accepted": flags}
    attempted = set.intersection(*({r["row"] for r in jl_read(f"{a.out_dir}/raw_{arm}.jsonl")} for arm in T.ARMS))
    if attempted != set(scaffold):
        sys.exit(f"FATAL: {len(set(scaffold) - attempted)} scaffold rows were never attempted in both arms; "
                 "finish --generate first.")
    keep = sorted(set.intersection(*(set(g) for g in good.values())))
    manifest = {"rows_in_scaffold": len(scaffold), "rows_kept_both_arms": len(keep),
                "rows_dropped": sorted(set(scaffold) - set(keep)), "per_arm": stats, "files": {}}
    for arm in T.ARMS:
        out = f"{a.out_dir}/chosen_{arm}.jsonl"
        blob = "".join(json.dumps({"row": r, "prompt": scaffold[r]["prompt"],
                                   "response": good[arm][r]["classification"]["text"],
                                   "constitution": T.ARMS[arm], "attempt": good[arm][r]["attempt"],
                                   "request_id": good[arm][r]["request_id"]}, ensure_ascii=False) + "\n"
                       for r in keep)
        with open(out, "w", encoding="utf-8") as f:
            f.write(blob)
        manifest["files"][os.path.basename(out)] = T.sha(blob)
    json.dump(manifest, open(f"{a.out_dir}/finalize_manifest.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows_dropped"}, indent=2))
    print(f"dropped from BOTH arms: {len(manifest['rows_dropped'])} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for s in ("probe", "dry-run", "pilot", "pilot-eval", "generate", "finalize"):
        m.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--model", default=T.MODEL)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tokenizer", default=None, help="path to the Llama-3.1-8B-Instruct tokenizer")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=3, help="resamples of an INVALID attempt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    stage = next(s for s in ("probe", "dry_run", "pilot", "pilot_eval", "generate", "finalize")
                 if getattr(a, s))
    return globals()[f"stage_{stage}"](a)


if __name__ == "__main__":
    sys.exit(main())
