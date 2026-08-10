"""HuggingFace generation backend, drop-in for assistant_axis.VLLMGenerator.

Why this exists: the IV arm (K/D Appendix G) is the first thing in this project that needs
the model to *write* rather than to answer a multiple-choice question, and `1_generate.py`
reaches for vLLM to do it. Installing vLLM would put its own pinned torch into `$PYLIBS`,
which sits first on `PYTHONPATH` and would therefore shadow the torch build verified against
this pod's Blackwell `sm_120` (see fork-infra §2 and §12.8). For ~19k short generations the
speed is not worth risking a validated environment, so this uses the ordinary
`model.generate()` we already have.

Matches the surface `1_generate.py` actually uses -- `__init__`, `load()`,
`generate_batch(conversations) -> list[str]` -- and accepts the vLLM-only knobs so the call
site does not have to branch on backend.

THE TRAP, if you touch this: batched generation on a decoder-only model REQUIRES LEFT
PADDING. With the default right padding every sequence shorter than the longest one gets pad
tokens between its prompt and its first generated token, and the model happily continues from
the padding instead of from the prompt. The output is fluent, plausible, and wrong, with no
error anywhere -- the same silent-failure shape as the answer-token bug. `load()` forces
`padding_side="left"` and the decode step slices by the padded input width for the same
reason.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.utils import log


class HFGenerator:
    """Batched sampling generation via transformers, mirroring VLLMGenerator's interface."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 2048,
        tensor_parallel_size: int | None = None,   # vLLM-only, accepted and ignored
        gpu_memory_utilization: float | None = None,  # vLLM-only, accepted and ignored
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        batch_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.batch_size = batch_size
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        log.info("Loading HF model %s (dtype=%s)", self.model_name, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Decoder-only batched generation must pad on the LEFT; see module docstring.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype, device_map="auto")
        self.model.eval()
        log.info("HF model loaded")

    @torch.no_grad()
    def generate_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        """Conversations (message dicts) -> generated assistant text, one per conversation.

        vLLM accepts an unbounded batch and schedules it itself; `generate()` does not, so
        this chunks at `batch_size`. Order is preserved, which the caller relies on to line
        responses up with their questions.
        """
        self.load()
        prompts = [
            self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]

        out: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_model_len, add_special_tokens=False,
            ).to(self.model.device)

            gen = self.model.generate(
                **enc,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Slice off the prompt by the PADDED width: with left padding every row in the
            # batch has the same input length, so this is exact rather than approximate.
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            out.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

        return out
