#!/usr/bin/env python3
"""Does the installed vLLM still accept the kwargs OCT's introspection scripts pass?

Phase B's introspection generation calls `vllm.LLM(**llm_kwargs)` with a kwarg set written
against vLLM ~0.8-0.10. vLLM renames and removes engine arguments freely across minor
versions -- `task` in particular was superseded by `runner`/`convert`. A mismatch is a
TypeError thrown AFTER the weights load, minutes into a multi-hour stage, so it is worth
ten seconds up front.

This inspects signatures and dataclass fields only. It loads no model and needs no GPU.

    PYTHONPATH=/workspace/pylibs-vllm-py312 python3 scripts/check_vllm_compat.py
"""
from __future__ import annotations

import dataclasses
import inspect
import sys

# exactly what character/introspection/self_reflection.py and self_interaction.py pass
LLM_KWARGS = ["model", "dtype", "gpu_memory_utilization", "tensor_parallel_size",
              "trust_remote_code", "task", "max_model_len", "max_num_seqs",
              "max_num_batched_tokens", "enable_prefix_caching", "enable_lora",
              "max_lora_rank"]
SAMPLING_KWARGS = ["repetition_penalty", "temperature", "top_p", "top_k", "min_p", "seed",
                   "max_tokens", "truncate_prompt_tokens"]


def accepted(target, names: list[str], label: str) -> list[str]:
    sig = inspect.signature(target)
    params = sig.parameters
    has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    fields = set()
    if dataclasses.is_dataclass(target):
        fields = {f.name for f in dataclasses.fields(target)}
    missing = [n for n in names if n not in params and n not in fields]
    print(f"\n{label}: {len(names) - len(missing)}/{len(names)} accepted by name"
          + (f"  (**kwargs present: {has_kwargs})" if has_kwargs else ""))
    for n in names:
        mark = "ok " if n not in missing else "MISSING"
        print(f"  {mark:<8}{n}")
    return missing


def main() -> int:
    import vllm
    from vllm import LLM, SamplingParams
    print(f"vllm {vllm.__version__}")
    try:
        import torch
        print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    except Exception as e:                                       # noqa: BLE001
        print(f"torch import FAILED: {e}")

    from vllm.lora.request import LoRARequest       # the import self_reflection.py makes
    print(f"LoRARequest ok: {LoRARequest.__module__}")

    miss_llm = accepted(LLM.__init__, LLM_KWARGS, "LLM.__init__")
    # LLM forwards unknown kwargs into EngineArgs, so check there too before calling it fatal
    if miss_llm:
        try:
            from vllm.engine.arg_utils import EngineArgs
            ef = {f.name for f in dataclasses.fields(EngineArgs)}
            still = [n for n in miss_llm if n not in ef]
            print(f"\n  of those, found on EngineArgs: "
                  f"{sorted(set(miss_llm) - set(still))}")
            miss_llm = still
        except Exception as e:                                   # noqa: BLE001
            print(f"\n  (could not inspect EngineArgs: {e})")

    miss_sp = accepted(SamplingParams, SAMPLING_KWARGS, "SamplingParams")

    print()
    if miss_llm or miss_sp:
        print(f"INCOMPATIBLE: LLM {miss_llm}  SamplingParams {miss_sp}")
        print("Patch character/introspection/*.py (additively) before Phase B's "
              "introspection stage, or pin a vLLM that accepts these.")
        return 1
    print("VLLM COMPAT OK -- every kwarg OCT passes is accepted. No patch needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
