#!/usr/bin/env python3
"""Behavioural Persona-Vectors eval -- GPU generation. docs/spec_behav_pv_eval.md §5.

One model load per arm (all personas/questions inside it), adapter patched in place with the
repo's bit-exact LoRA merge. Each batch is appended to responses.jsonl and fsynced, so a crash
loses at most one batch; rerunning skips every id already written.

Adapter check per arm: logits on a fixed probe are taken before and after patching; the arm
aborts unless every target module was patched and the probe logits moved (base: unpatched).

    PYTHONPATH=$PYLIBS:. python3 scripts/behav_pv/generate.py --run smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from persona_steering.lora import apply_adapter_stack  # noqa: E402

PROBE = [{"role": "user", "content": "Describe yourself in one sentence."}]


def messages(row):
    m = [{"role": "user", "content": row["question"]}]
    return ([{"role": "system", "content": row["system"]}] + m) if row["system"] else m


@torch.no_grad()
def probe_logits(model, tok):
    ids = tok.apply_chat_template(PROBE, add_generation_prompt=True, return_tensors="pt").to("cuda")
    return model(input_ids=ids).logits[0, -1].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    out = Path("outputs/behav_pv") / a.run
    man = json.loads((out / "manifest.json").read_text())
    grid = [json.loads(l) for l in (out / "grid.jsonl").read_text().splitlines()]
    rp = out / "responses.jsonl"
    done = {json.loads(l)["id"] for l in rp.read_text().splitlines()} if rp.exists() else set()
    sp = man["sampling"]

    tok = AutoTokenizer.from_pretrained(man["base_model"]["path"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    for arm, ad in man["arms"].items():
        todo = [r for r in grid if r["arm"] == arm and r["id"] not in done]
        if not todo:
            print(f"{arm}: complete", flush=True); continue
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(man["base_model"]["path"], torch_dtype=torch.bfloat16,
                                                     device_map="cuda", attn_implementation="sdpa").eval()
        eos = model.generation_config.eos_token_id
        eos = torch.tensor(eos if isinstance(eos, list) else [eos], device="cuda")
        before = probe_logits(model, tok)
        n_patched = 0
        if ad is not None:
            n_patched = apply_adapter_stack(model, [ad["path"]], [1.0])
            n_target = 32 * len(ad["config"]["target_modules"])
            moved = (probe_logits(model, tok) - before).abs().max().item()
            if n_patched != n_target or moved == 0:
                sys.exit(f"{arm}: adapter NOT applied (patched {n_patched}/{n_target}, max|dlogit| {moved})")
            print(f"{arm}: patched {n_patched}/{n_target} modules, probe max|dlogit| {moved:.3f}", flush=True)
        # Longest prompts first keeps padding low; the order is fixed by the grid, so the
        # batch composition (and hence the per-batch seed) is the same on every rerun.
        todo.sort(key=lambda r: (-len(tok.apply_chat_template(messages(r), tokenize=False)), r["id"]))
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            seed = man["seed"] + hash_ids(batch)
            torch.manual_seed(seed)
            enc = tok.apply_chat_template([messages(r) for r in batch], add_generation_prompt=True,
                                          padding=True, return_tensors="pt", return_dict=True).to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=sp["temperature"], top_p=sp["top_p"],
                                 max_new_tokens=sp["max_new_tokens"], min_new_tokens=sp["min_new_tokens"],
                                 pad_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            with open(rp, "a") as f:
                for r, o in zip(batch, new):
                    hit_eos = torch.isin(o, eos).nonzero()
                    n_tok = int(hit_eos[0]) if len(hit_eos) else len(o)
                    f.write(json.dumps({**r, "response": tok.decode(o[:n_tok], skip_special_tokens=True),
                                        "n_new_tokens": n_tok, "finished": bool(len(hit_eos)),
                                        "batch_seed": seed, "n_modules_patched": n_patched}) + "\n")
                f.flush(); os.fsync(f.fileno())
            print(f"{arm}: {min(i + a.batch_size, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()


def hash_ids(batch):
    """Stable (not Python-hash-randomised) 31-bit offset from the batch's ids."""
    import hashlib
    return int(hashlib.sha256("".join(r["id"] for r in batch).encode()).hexdigest()[:7], 16)


if __name__ == "__main__":
    main()
