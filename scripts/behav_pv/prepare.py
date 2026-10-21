#!/usr/bin/env python3
"""Behavioural Persona-Vectors eval -- CPU preparation. docs/spec_behav_pv_eval.md §5.

Resolves the published `evil` questions, persona prompts and adapter paths, builds the full
generation grid and writes a frozen manifest. Nothing here touches a GPU or an API.

    python3 scripts/behav_pv/prepare.py --run smoke --arms base misalignment \
        --personas con_artist kindergarten_teacher --n-questions 4 --n-samples 1
    python3 scripts/behav_pv/prepare.py --run full            # defaults = the full design
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from persona_steering.personas import load_persona  # noqa: E402

PV = Path("/workspace/persona_vectors_ref")
EVIL = PV / "data_generation/trait_data_eval/evil.json"
BASE = Path("/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
            "0e9e39f249a16976918f6564b8830bc894c89659")
SNAP = Path("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
            "318b5f7e1428097a1a61d5f0ed205ee048b3f620")
ADAPTERS = {
    "base": None,
    # Gated repo, not in the public persona release; access already granted on this account.
    "misalignment": Path("/workspace/hf/hub/models--maius--llama-3.1-8b-it-misalignment/"
                         "snapshots/f1a019278e90f6547c049894d2ff89752818cd11"),
    "impulsiveness": SNAP / "impulsiveness",
    "goodness": SNAP / "goodness",
}
OCCUPATIONAL = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
                "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1000, "min_new_tokens": 1}
JUDGE = {"published_model": "gpt-4.1-mini-2025-04-14",
         "used_model": "openai/gpt-4.1-mini",
         "route": "OpenRouter, provider pinned to OpenAI, allow_fallbacks=false",
         "note": "OpenRouter's only gpt-4.1-mini entry is created 1744651381 = 2025-04-14, "
                 "i.e. the published snapshot; the dated id itself cannot be requested.",
         "call": {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                  "seed": 0},
         "aggregation": "persona_vectors judge.OpenAiJudge._aggregate_0_100_score, unmodified"}
OUT = Path("outputs/behav_pv")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; output goes to outputs/behav_pv/<run>")
    ap.add_argument("--arms", nargs="+", default=list(ADAPTERS))
    ap.add_argument("--personas", nargs="+", default=OCCUPATIONAL + ["null"])
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261014)
    a = ap.parse_args()

    out = OUT / a.run
    if (out / "manifest.json").exists():
        sys.exit(f"{out}/manifest.json exists -- a manifest is frozen once written")
    ev = json.loads(EVIL.read_text())
    questions = ev["questions"][: a.n_questions]
    assert len(ev["questions"]) == 20, len(ev["questions"])

    personas = {}
    for slug in a.personas:
        sp = load_persona(slug).system_prompt_variants[0]
        # `null` has system prompt "" -- send NO system message (byte-identical on Llama-3.1,
        # CLAUDE.md porting notes, but omitting it is what the spec asks for).
        personas[slug] = sp if sp.strip() else None

    arms = {}
    for arm in a.arms:
        ad = ADAPTERS[arm]
        arms[arm] = None if ad is None else {
            "path": str(ad), "adapter_sha256": sha256(ad / "adapter_model.safetensors"),
            "config": json.loads((ad / "adapter_config.json").read_text())}

    grid = [{"id": f"{arm}|{p}|q{qi:02d}|s{s}", "arm": arm, "persona": p, "qid": qi, "sample": s,
             "system": personas[p], "question": q}
            for arm in a.arms for p in a.personas for qi, q in enumerate(questions)
            for s in range(a.n_samples)]

    pv_commit = subprocess.run(["git", "-C", str(PV), "rev-parse", "HEAD"], capture_output=True,
                               text=True, check=True).stdout.strip()
    manifest = {
        "run": a.run, "spec": "docs/spec_behav_pv_eval.md",
        "base_model": {"id": "meta-llama/Llama-3.1-8B-Instruct", "path": str(BASE)},
        "adapter_application": "persona_steering.lora.apply_adapter_stack, scale 1 "
                               "(bit-identical to peft merge_and_unload)",
        "arms": arms,
        "persona_vectors": {"repo": "safety-research/persona_vectors", "commit": pv_commit,
                            "evil_json": str(EVIL), "evil_json_sha256": sha256(EVIL),
                            "prompts_py_sha256": sha256(PV / "eval/prompts.py"),
                            "judge_py_sha256": sha256(PV / "judge.py")},
        "questions": questions,
        "personas": {p: {"system": s, "variant": 0} for p, s in personas.items()},
        "sampling": SAMPLING, "seed": a.seed, "n_samples": a.n_samples,
        "judge": JUDGE, "n_rows": len(grid),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grid))
    manifest["grid_sha256"] = sha256(out / "grid.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{out}: {len(grid)} rows = {len(a.arms)} arms x {len(a.personas)} personas x "
          f"{len(questions)} questions x {a.n_samples} samples")


if __name__ == "__main__":
    main()
