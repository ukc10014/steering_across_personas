#!/usr/bin/env python3
"""Verify the CAA answer-token extraction index on a given tokenizer.

The activation for a CAA pair is read at a single token position: the "A"/"B"
answer letter in the assistant turn. If that index is off by even one token,
every cosine downstream is noise -- and it fails silently. Answer letters
tokenize differently across model families ("A" vs " A" vs "(A"), so this must
be re-checked whenever the pipeline moves to a new model.

This loads only the tokenizer, not the model, so it runs in seconds.

Usage:
    python scripts/verify_answer_token.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/verify_answer_token.py --model meta-llama/Llama-3.1-8B-Instruct \
        --traits deference warmth --personas therapist null --n 5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "assistant-axis-ref"))

from transformers import AutoTokenizer

from persona_steering.config import Trait
from persona_steering.data import load_caa_dataset
from persona_steering.personas import load_persona


def _load_2c_module():
    """Import pipeline/2c_caa_activations.py (leading digit blocks normal import).

    We deliberately reuse the pipeline's own functions rather than
    reimplementing them, so this check validates the code that actually runs.
    """
    path = REPO_ROOT / "pipeline" / "2c_caa_activations.py"
    spec = importlib.util.spec_from_file_location("caa_2c", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CAA answer-token indexing")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--traits", nargs="*", default=["deference", "warmth"])
    parser.add_argument("--personas", nargs="*", default=["therapist", "null"])
    parser.add_argument("--n", type=int, default=5, help="Examples per persona x trait x direction")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full templated prompt for the first example")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    caa = _load_2c_module()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    supports_system = "gemma-2" not in args.model.lower()

    print(f"Model:      {args.model}")
    print(f"Supports system prompt: {supports_system}")
    print()

    n_checked = 0
    n_bad = 0
    printed_prompt = False

    for trait_name in args.traits:
        dataset = load_caa_dataset(Trait(trait_name))
        for persona_slug in args.personas:
            persona = load_persona(persona_slug)
            system_prompt = persona.default_system_prompt
            for direction in ("pos", "neg"):
                print(f"--- {persona_slug} / {trait_name} / {direction} ---")
                for q in dataset.questions[: args.n]:
                    answer_letter = caa.get_answer_letter(q, direction)
                    user_msg = caa.format_caa_user_message(q)

                    if supports_system:
                        base = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ]
                    else:
                        base = [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]

                    conv_with = base + [{"role": "assistant", "content": answer_letter}]

                    pos = caa.find_answer_token_position(
                        tokenizer, base, conv_with, answer_letter
                    )

                    full_text = tokenizer.apply_chat_template(
                        conv_with, tokenize=False, add_generation_prompt=False
                    )
                    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                    token_at_index = tokenizer.decode([full_ids[pos]])

                    if args.show_prompt and not printed_prompt:
                        print("  [full templated prompt]")
                        print("  " + repr(full_text))
                        printed_prompt = True

                    # The token we land on must actually contain the answer letter,
                    # and must not be the *other* letter.
                    other = "B" if answer_letter == "A" else "A"
                    ok = answer_letter in token_at_index and other not in token_at_index

                    n_checked += 1
                    if not ok:
                        n_bad += 1

                    context = tokenizer.decode(full_ids[max(0, pos - 4) : pos + 2])
                    print(
                        f"  q{q.id:<4} expect={answer_letter}  idx={pos:<5} "
                        f"token={token_at_index!r:<8} {'OK' if ok else 'MISMATCH'}"
                        f"   ...{context!r}"
                    )
                print()

    print(f"Checked {n_checked} examples: {n_checked - n_bad} OK, {n_bad} mismatched.")
    if n_bad:
        print("FAIL - answer-token indexing is wrong. Do not trust downstream cosines.")
        return 1
    print("PASS - extraction index lands on the answer letter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
