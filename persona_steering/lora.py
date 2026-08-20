"""Apply a LoRA adapter to a loaded model in memory, at an arbitrary strength.

The dose-response ladder runs the same constitution at several strengths. Merging each one
to disk would cost ~20 min and 16 GB per scale, so the delta is applied to the loaded
weights instead:

    W(s) = W_base + s * (alpha/r) * B @ A

At s=1 this reproduces `peft`'s merge_and_unload BIT-FOR-BIT -- checked against the archived
merged checkpoints by `scripts/dose_calibrate.py --verify-scale1`, which reports
rel||pred - merged|| = 0 on every module tested. That equality is what lets a scaled run be
compared with the archived s=1 arms: they differ in the scale and in nothing else.

Only plain LoRA is implemented. rsLoRA changes the scaling to alpha/sqrt(r), DoRA adds a
magnitude vector, and modules_to_save replaces whole modules -- each would silently give the
wrong weights, so each is refused rather than ignored.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def lora_deltas(adapter_dir: str | Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return {base_param_name: (B, A)} plus the alpha/r scale, in fp32 on CPU."""
    adapter = Path(adapter_dir)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    for unsupported in ("use_dora", "use_rslora"):
        if cfg.get(unsupported):
            raise ValueError(f"{adapter}: {unsupported}=True changes the merge formula; "
                             f"only plain LoRA is implemented.")
    if cfg.get("modules_to_save"):
        raise ValueError(f"{adapter}: modules_to_save is set, so merging is not just the "
                         f"low-rank product; not implemented.")
    sd = load_file(adapter / "adapter_model.safetensors")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if ".lora_A" in k:
            stem, side = k.split(".lora_A")[0], "A"
        elif ".lora_B" in k:
            stem, side = k.split(".lora_B")[0], "B"
        else:
            continue
        name = re.sub(r"^base_model\.model\.", "", stem) + ".weight"
        pairs.setdefault(name, {})[side] = v.float()
    out = {n: (d["B"], d["A"]) for n, d in pairs.items() if "A" in d and "B" in d}
    if len(out) != len(pairs):
        raise ValueError(f"{adapter}: {len(pairs) - len(out)} modules have an unpaired A/B")
    return out, cfg["lora_alpha"] / cfg["r"]


@torch.no_grad()
def apply_scaled_lora(model, adapter_dir: str | Path, s: float) -> int:
    """W += s * (alpha/r) * B @ A, in place, on whatever device each weight lives.

    The arithmetic is done in fp32 and cast back to the weight's dtype, which is what peft
    does; accumulating in bf16 would lose most of a small delta.
    """
    deltas, ar = lora_deltas(adapter_dir)
    params = dict(model.named_parameters())
    n = 0
    for name, (B, A) in deltas.items():
        if name not in params:
            raise ValueError(f"adapter targets {name}, which the model does not have")
        W = params[name]
        dW = (B.to(W.device) @ A.to(W.device)) * (ar * s)
        W.copy_((W.float() + dW).to(W.dtype))
        n += 1
    return n
