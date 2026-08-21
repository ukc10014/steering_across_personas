#!/usr/bin/env python3
"""Output-KL from base on a neutral corpus: a dose measure that is not the analysis data.

WHY. Every dose figure in the results doc so far is measured on the CAA prompts themselves
(`functional_dose.py`), which is the right quantity for interpreting the geometry but is
endogenous to it: the dominant part of an adapter's effect on those prompts is the
persona-common shift, which is also the adapter's main effect. Section 3 flags the missing
measurement explicitly -- "a genuine dose match needs activation displacement on a neutral
corpus or output KL from base" -- and this is that measurement.

It matters most for the untrained-LoRA control. A random adapter matched to `goodness` on
weight-space norm has no reason to land at `goodness`'s functional dose, and if the two
arms are compared at wildly different dose the comparison says nothing. This script prices
each adapter on neutral text before any GPU time is spent on a full 192-cell extraction, so
the scale ladder can be chosen rather than guessed.

WHAT IS MEASURED. Base generates a greedy continuation of each neutral prompt. Every arm is
then teacher-forced on that same token sequence, and we report mean per-token
KL(base || arm) over the continuation positions, plus the fraction of positions where the
argmax token changes. Both are computed on identical sequences, so arms are directly
comparable. Coherence is checked at the same time: an arm whose greedy continuation is
gibberish is not a model whose persona geometry means anything, and that has to be known
before, not after, a multi-hour extraction.

The model is reloaded from disk for each arm rather than un-patched, because the patch is
applied in place to bf16 weights and adding then subtracting a delta does not return the
original bits. Reloading costs a few minutes and removes the possibility of contaminating
one arm with the previous one's rounding.

USAGE
    python scripts/neutral_dose.py --arms goodness impulsiveness misalignment \
        --adapters /workspace/random_loras/random_iid /workspace/random_loras/random_spec \
        --scales 1.0 --out outputs/analysis/neutral_dose.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "assistant-axis-ref")
sys.path.insert(0, ".")

BASE = ("/workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
        "0e9e39f249a16976918f6564b8830bc894c89659")
SNAP = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/"
        "318b5f7e1428097a1a61d5f0ed205ee048b3f620")
MISALIGN = ("/workspace/hf/hub/models--maius--llama-3.1-8b-it-misalignment/snapshots/"
            "f1a019278e90f6547c049894d2ff89752818cd11")

# Deliberately unrelated to personas, traits, or character: the point is to price the
# intervention on text the analysis has no stake in.
PROMPTS = [
    "Explain how a refrigerator keeps food cold.",
    "What causes the tides?",
    "Write a short paragraph about the history of the bicycle.",
    "How do you make bread from flour, water, salt and yeast?",
    "Summarise how TCP handles packet loss.",
    "What is the difference between weather and climate?",
    "Describe the process of photosynthesis in simple terms.",
    "Why do metals conduct electricity?",
    "Give a brief overview of the Bronze Age collapse.",
    "How does a diesel engine differ from a petrol engine?",
    "Explain what a database index does.",
    "What are the main stages of a star's life cycle?",
    "Describe how vaccines produce immunity.",
    "What makes some materials magnetic?",
    "Explain the rules of chess castling.",
    "How is paper manufactured from wood pulp?",
]


def adapter_path(name: str) -> str:
    if name == "misalignment":
        return MISALIGN
    if "/" in name:
        return name
    return f"{SNAP}/{name}"


def load_base(device: str):
    from assistant_axis.internals.model import ProbingModel
    return ProbingModel(BASE, device=device)


@torch.no_grad()
def forward_logits(pm, seqs: list[torch.Tensor], n_prompt: list[int],
                   hidden_layers: list[int] | None = None):
    """Teacher-forced next-token log-probs, and optionally hidden states, per sequence.

    The hidden states are what makes this comparable to `functional_dose.py`. Output KL
    prices what survives the remaining layers; section 7.3 measured those two diverging by a
    factor of 35 on a random perturbation, so an out-of-domain dose measure has to be taken
    in the SAME space as the in-domain one it is being checked against, not merely on a
    different corpus.
    """
    out, hid = [], []
    for ids, np_ in zip(seqs, n_prompt):
        ids = ids.unsqueeze(0).to(pm.model.device)
        res = pm.model(ids, output_hidden_states=bool(hidden_layers))
        logits = res.logits[0].float()
        # position i predicts token i+1, so continuation targets start at np_-1
        out.append(F.log_softmax(logits[np_ - 1:-1], dim=-1).cpu())
        if hidden_layers:
            # hidden_states[0] is the embedding output, so layer L is index L+1.
            hid.append({L: res.hidden_states[L + 1][0, np_ - 1:-1].float().cpu()
                        for L in hidden_layers})
    return (out, hid) if hidden_layers else out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="*", default=["goodness"],
                   help="adapter names under $SNAP, or 'misalignment', or full paths")
    p.add_argument("--adapters", nargs="*", default=[],
                   help="extra adapter directories (e.g. the random controls)")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--new-tokens", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="outputs/analysis/neutral_dose.json")
    p.add_argument("--show", type=int, default=1, help="print this many sample continuations")
    p.add_argument("--hidden-layers", type=int, nargs="*", default=[15, 20],
                   help="also report relative hidden-state displacement at these layers")
    args = p.parse_args()

    from persona_steering.lora import apply_scaled_lora

    print("loading base...", flush=True)
    pm = load_base(args.device)
    tok = pm.tokenizer

    prompt_ids = [tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt")[0]
                  for q in PROMPTS]

    print(f"generating base continuations ({args.new_tokens} tokens x {len(PROMPTS)})...",
          flush=True)
    seqs, n_prompt = [], []
    for ids in prompt_ids:
        gen = pm.model.generate(ids.unsqueeze(0).to(pm.model.device),
                                max_new_tokens=args.new_tokens, do_sample=False,
                                pad_token_id=tok.eos_token_id)[0].cpu()
        seqs.append(gen)
        n_prompt.append(len(ids))

    base_lp, base_hid = forward_logits(pm, seqs, n_prompt, args.hidden_layers)
    base_txt = [tok.decode(s[n:], skip_special_tokens=True) for s, n in zip(seqs, n_prompt)]
    if args.show:
        print(f"\n  base: {base_txt[0][:200]}\n")

    del pm
    gc.collect(); torch.cuda.empty_cache()

    results = {}
    todo = [(a, adapter_path(a)) for a in args.arms] + \
           [(Path(d).name, d) for d in args.adapters]

    for name, path in todo:
        for s in args.scales:
            key = name if s == 1.0 else f"{name}_s{s}"
            print(f"=== {key} ===", flush=True)
            pm = load_base(args.device)
            n = apply_scaled_lora(pm.model, path, s)
            print(f"  patched {n} modules", flush=True)

            arm_lp, arm_hid = forward_logits(pm, seqs, n_prompt, args.hidden_layers)
            kls, flips = [], []
            for b, a in zip(base_lp, arm_lp):
                kls.append(float((b.exp() * (b - a)).sum(-1).mean()))
                flips.append(float((b.argmax(-1) != a.argmax(-1)).float().mean()))
            kl = sum(kls) / len(kls)
            flip = sum(flips) / len(flips)

            # Same normalisation as functional_dose.py's answer-token displacement:
            # mean_t ||h_arm - h_base|| / mean_t ||h_base||, so the two are comparable.
            hdisp = {}
            for L in args.hidden_layers:
                num = sum(float((a[L] - b[L]).norm(dim=-1).mean()) for b, a in zip(base_hid, arm_hid))
                den = sum(float(b[L].norm(dim=-1).mean()) for b in base_hid)
                hdisp[str(L)] = num / max(den, 1e-9)

            gen = pm.model.generate(prompt_ids[0].unsqueeze(0).to(pm.model.device),
                                    max_new_tokens=args.new_tokens, do_sample=False,
                                    pad_token_id=tok.eos_token_id)[0].cpu()
            txt = tok.decode(gen[n_prompt[0]:], skip_special_tokens=True)
            results[key] = {"adapter": path, "scale": s, "mean_kl": kl,
                            "argmax_flip_rate": flip,
                            "hidden_displacement": hdisp, "sample": txt}
            print(f"  mean KL(base||arm) = {kl:.4f}   argmax flips = {flip:.1%}"
                  + "".join(f"   L{L} displ = {hdisp[str(L)]:.4f}" for L in args.hidden_layers))
            if args.show:
                print(f"  sample: {txt[:200]}\n", flush=True)

            del pm
            gc.collect(); torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prompts": PROMPTS, "base_sample": base_txt[0],
                               "arms": results}, indent=2))
    cols = "".join(f"{'L'+str(L)+' displ':>11s}" for L in args.hidden_layers)
    print(f"\n{'arm':28s} {'mean KL':>10s} {'flips':>8s}{cols}")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["mean_kl"]):
        h = "".join(f"{v['hidden_displacement'][str(L)]:11.4f}" for L in args.hidden_layers)
        print(f"{k:28s} {v['mean_kl']:10.4f} {v['argmax_flip_rate']:8.1%}{h}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
