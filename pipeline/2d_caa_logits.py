#!/usr/bin/env python3
"""Revealed A/B preference on the CAA questions: log P(trait-positive) - log P(trait-negative).

WHY THIS EXISTS. Every constitution x trait quantity in the geometry work is a MAGNITUDE.
||dG_{c,t}|| says `impulsiveness` moves the impulsivity representation ~1.8x as far as it
moves the other six traits, but it cannot say whether the model became more impulsive or
merely reorganised the trait harder. scripts/signed_trait_shift.py tried to recover the
sign from the geometry and failed its own validity checks (see figA5): generic contraction
has a projection along the trait axis, so it manufactures a sign where there is no
preference. This script sidesteps the geometry and asks the model directly.

WHAT IT MEASURES. The CAA items are forced-choice: one option expresses more of the trait,
the other less, with the polarity randomised across items (`a_is_positive`). Put the
question to the model with the generation prompt open and read the logits of the two
answer letters:

    logodds = logit(positive_letter) - logit(negative_letter)

Since both letters are single tokens at the same position, the softmax normaliser cancels
exactly, so this is log P(pos)/P(neg) under the model's own two-way choice -- no
temperature, no sampling, no judge. Against the base model's value for the same item,

    d_pref = logodds_arm - logodds_base

is a SIGNED behavioural preference shift: > 0 means the arm moved the model toward the
trait-positive answer. This is the closest analogue in this codebase to OCT's revealed
preference / Elo measurement.

WHY THE CACHE CANNOT ANSWER THIS. outputs/*/caa_activations hold the hidden state AT the
answer token, which predicts the token AFTER the answer. The distribution over the answer
itself lives at the position BEFORE it -- the last token of the generation prompt -- and
was never stored. Hence a fresh forward pass. It is much lighter than the activation
extraction: no hooks, no per-layer residency, one pass per question rather than two
(pos/neg share a prompt; only the read-out differs), and only the final position's logits
are materialised.

DIAGNOSTIC KEPT ALONGSIDE. p_ab is the total probability mass the model puts on the two
letters. The log-odds is well defined however small that mass is, but a cell where the
model largely wants to say something else is a weaker read of "preference", and the number
belongs in the record rather than in a footnote. Reported per cell.

Usage:
    python pipeline/2d_caa_logits.py --arm base
    python pipeline/2d_caa_logits.py --arm impulsiveness \
        --lora-adapter "$SNAP/impulsiveness" --lora-scale 1.0
    python pipeline/2d_caa_logits.py --arm random_perm_s16 \
        --lora-adapter /workspace/random_loras/random_perm --lora-scale 16
    python pipeline/2d_caa_logits.py --arm base --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis.internals import ProbingModel

from persona_steering.config import Trait, OUTPUTS_DIR, PERSONA_SLUGS
from persona_steering.data import load_caa_dataset, CAADataset, CAAQuestion
from persona_steering.personas import load_persona
from persona_steering.utils import get_device, log

# The ten semantic personas plus `null`. `nonsense` is excluded by default: it is a control
# for "is a system prompt present at all", and the question here is about preference under a
# coherent identity. Pass --personas explicitly to override.
DEFAULT_PERSONAS = [p for p in PERSONA_SLUGS if p != "nonsense"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAA A/B answer-letter logits per arm")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Base HF model (default: Llama-3.1-8B-Instruct)")
    p.add_argument("--arm", required=True,
                   help="Arm name; names the output dir outputs/{model}-{arm}/caa_logits")
    p.add_argument("--lora-adapter", default=None,
                   help="LoRA adapter to patch in before the pass (omit for the base arm)")
    p.add_argument("--lora-scale", type=float, default=1.0,
                   help="Strength s in W = W_base + s*(alpha/r)*B*A (default 1.0)")
    p.add_argument("--personas", nargs="*", default=None)
    p.add_argument("--traits", nargs="*", default=None)
    p.add_argument("--variant", type=int, default=0,
                   help="Index into system_prompt_variants (default 0, as in 2c)")
    p.add_argument("--answer-instruction", action="store_true",
                   help="Append an explicit 'answer with one letter' line to the user "
                        "message. The default prompt is byte-identical to the one 2c used "
                        "for the activations, which is what makes this comparable to the "
                        "geometry -- but under it the model puts <1%% of its mass on the "
                        "two letters, because the assistant turn normally opens with a "
                        "word. This flag is the robustness variant: it raises that mass so "
                        "the conditional is read off a distribution the model actually "
                        "concentrates there. Output goes to caa_logits_forced/.")
    p.add_argument("--max-questions", type=int, default=None,
                   help="First N questions per trait (default: all). Diagnostics only.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.lora_scale != 1.0 and a.lora_adapter is None:
        p.error("--lora-scale has no effect without --lora-adapter")
    return a


ANSWER_INSTRUCTION = "Answer with a single letter, A or B, and nothing else."


def format_caa_user_message(q: CAAQuestion, answer_instruction: bool = False) -> str:
    """Identical to 2c_caa_activations.format_caa_user_message -- the prompt must not drift.

    The optional instruction is the one deliberate deviation, kept behind a flag and
    written to a separate output directory so the two can never be pooled by accident.
    """
    msg = f"{q.scenario}\n\n(A) {q.option_a}\n(B) {q.option_b}"
    if answer_instruction:
        msg = f"{msg}\n\n{ANSWER_INSTRUCTION}"
    return msg


def build_conversation(system_prompt: str, q: CAAQuestion, supports_system: bool,
                       answer_instruction: bool = False):
    user_msg = format_caa_user_message(q, answer_instruction)
    if supports_system:
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}]
    return [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]


def answer_token_ids(tokenizer, conv_no_assistant) -> tuple[int, int]:
    """Token ids the model would emit for "A" / "B" as the assistant's first content token.

    Derived the same way 2c locates the answer token: tokenize the conversation with and
    without the assistant turn and take the first token past the prefix. Doing it by
    construction rather than by tokenizer("A") avoids the leading-space variants that make
    hand-picked ids wrong on some tokenizers.
    """
    prefix_text = tokenizer.apply_chat_template(
        conv_no_assistant, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    ids = []
    for letter in ("A", "B"):
        full_text = tokenizer.apply_chat_template(
            conv_no_assistant + [{"role": "assistant", "content": letter}],
            tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("generation prompt is not a prefix of the completed turn; "
                             "the chat template does not support this read-out")
        tid = full_ids[len(prefix_ids)]
        decoded = tokenizer.decode([tid])
        if letter not in decoded:
            raise ValueError(f"first assistant token for {letter!r} decodes to {decoded!r}")
        ids.append(tid)

    if ids[0] == ids[1]:
        raise ValueError("A and B map to the same token id")
    return ids[0], ids[1]


@torch.inference_mode()
def cell_logits(pm, system_prompt: str, dataset: CAADataset, batch_size: int,
                tok_a: int, tok_b: int, answer_instruction: bool = False) -> dict[str, np.ndarray]:
    """Answer-letter logits for every question in one persona x trait cell."""
    tokenizer, model = pm.tokenizer, pm.model
    supports_system = pm.supports_system_prompt()

    prompts = []
    for q in dataset.questions:
        conv = build_conversation(system_prompt, q, supports_system, answer_instruction)
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    n = len(prompts)
    logit_a = np.zeros(n, dtype=np.float32)
    logit_b = np.zeros(n, dtype=np.float32)
    p_ab = np.zeros(n, dtype=np.float32)

    # Length-sorted batching: these prompts vary by ~2x in length, and padding to the
    # longest in a random batch wastes most of the compute. Order is restored via `order`.
    order = sorted(range(n), key=lambda i: len(prompts[i]))

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch = [prompts[i] for i in idx]
        max_len = max(len(b) for b in batch)

        # LEFT padding, so the final real token is at position -1 for every row and the
        # read-out needs no per-row index arithmetic.
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for r, ids in enumerate(batch):
            input_ids[r, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[r, max_len - len(ids):] = 1

        # Mask by POSITION, never by token identity: pad_id is <|eot_id|> on Llama-3.1 and
        # that token occurs for real in every turn of the chat template (see the legacy-mask
        # note in 2c_caa_activations.py).
        input_ids = input_ids.to(pm.device)
        attn = attn.to(pm.device)
        # Explicit position ids so left padding does not shift real tokens' positions.
        pos_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        # Only the last position's logits are needed. Running the trunk and applying the
        # head to one position avoids materialising (B, T, 128256) logits, which at
        # batch 32 would be several GB on top of the weights.
        hidden = model.model(input_ids=input_ids, attention_mask=attn,
                             position_ids=pos_ids).last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden).float()

        lse = torch.logsumexp(logits, dim=-1)
        la, lb = logits[:, tok_a], logits[:, tok_b]
        mass = torch.exp(la - lse) + torch.exp(lb - lse)

        for r, i in enumerate(idx):
            logit_a[i] = la[r].item()
            logit_b[i] = lb[r].item()
            p_ab[i] = mass[r].item()

    a_is_pos = np.array([q.a_is_positive for q in dataset.questions], dtype=bool)
    # Signed toward the trait-positive option, per item polarity.
    logodds = np.where(a_is_pos, logit_a - logit_b, logit_b - logit_a).astype(np.float32)

    return {
        "qid": np.array([q.id for q in dataset.questions], dtype=np.int32),
        "a_is_positive": a_is_pos,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "logodds": logodds,
        "p_ab": p_ab,
    }


def main() -> None:
    args = parse_args()

    personas = args.personas if args.personas is not None else DEFAULT_PERSONAS
    traits = [Trait(t) for t in args.traits] if args.traits else list(Trait)

    datasets = {}
    for t in traits:
        ds = load_caa_dataset(t)
        if args.max_questions is not None:
            ds = dataclasses.replace(ds, questions=ds.questions[: args.max_questions])
        datasets[t] = ds

    model_tag = "llama-3.1-8b"
    subdir = "caa_logits_forced" if args.answer_instruction else "caa_logits"
    out_dir = (Path(args.output_dir) if args.output_dir
               else OUTPUTS_DIR / f"{model_tag}-{args.arm}" / subdir)

    work = [(p, t, out_dir / f"{p}_{t.value}.npz") for p in personas for t in traits]
    remaining = [w for w in work if not w[2].exists()]

    if args.dry_run:
        n_fwd = sum(datasets[t].n_questions for _, t, _ in remaining)
        print("=== DRY RUN ===")
        print(f"arm        : {args.arm}")
        print(f"adapter    : {args.lora_adapter or '(none -- base)'}  scale={args.lora_scale}")
        print(f"output     : {out_dir}")
        print(f"personas   : {len(personas)}  {personas}")
        print(f"traits     : {len(traits)}")
        print(f"questions  : {[datasets[t].n_questions for t in traits]}")
        print(f"cells      : {len(work)} ({len(remaining)} remaining)")
        print(f"forward passes: {n_fwd}  -> {(n_fwd + args.batch_size - 1)//args.batch_size} batches")
        return

    if not remaining:
        log.info("All cells present for arm %s. Nothing to do.", args.arm)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.model)
    pm = ProbingModel(args.model, device=args.device or str(get_device()))
    if args.lora_adapter:
        from persona_steering.lora import apply_scaled_lora
        n = apply_scaled_lora(pm.model, args.lora_adapter, args.lora_scale)
        log.info("Patched %d modules from %s at s=%g", n, args.lora_adapter, args.lora_scale)
    else:
        log.info("No adapter: this is the base arm.")

    if not (hasattr(pm.model, "model") and hasattr(pm.model, "lm_head")):
        raise RuntimeError("expected a CausalLM with .model trunk and .lm_head")

    # Answer-token ids are a property of the chat template, not of the question or persona.
    # Resolve once, then assert the resolution is stable across personas so a template that
    # does depend on the system prompt cannot pass silently.
    probe_q = datasets[traits[0]].questions[0]
    supports_system = pm.supports_system_prompt()
    ref = None
    for slug in personas:
        sp = load_persona(slug).system_prompt_variants[args.variant]
        ids = answer_token_ids(pm.tokenizer,
                               build_conversation(sp, probe_q, supports_system,
                                                  args.answer_instruction))
        if ref is None:
            ref = ids
        elif ids != ref:
            raise RuntimeError(f"answer token ids differ for persona {slug}: {ids} vs {ref}")
    tok_a, tok_b = ref
    log.info("answer tokens: A=%d (%r)  B=%d (%r)", tok_a, pm.tokenizer.decode([tok_a]),
             tok_b, pm.tokenizer.decode([tok_b]))

    persona_cache: dict[str, str] = {}
    t0 = time.time()
    for slug, trait, path in tqdm(remaining, desc=f"logits[{args.arm}]"):
        if slug not in persona_cache:
            variants = load_persona(slug).system_prompt_variants
            if args.variant >= len(variants):
                raise SystemExit(f"{slug} has {len(variants)} variants; --variant "
                                 f"{args.variant} out of range")
            persona_cache[slug] = variants[args.variant]

        res = cell_logits(pm, persona_cache[slug], datasets[trait], args.batch_size,
                          tok_a, tok_b, args.answer_instruction)
        # Write to a temp name then rename: a run killed mid-write must not leave a short
        # .npz that a resume would treat as finished (2c learned this the hard way).
        # np.savez appends ".npz" to any name not already ending in it, so the temp name
        # must keep that suffix or the write lands somewhere the rename cannot find.
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, arm=args.arm, persona=slug, trait=trait.value,
                 lora_scale=args.lora_scale,
                 answer_instruction=bool(args.answer_instruction), **res)
        tmp.replace(path)

    pm.close()
    log.info("arm %s: %d cells in %.1f min", args.arm, len(remaining), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
