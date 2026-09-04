#!/usr/bin/env python3
"""Capture everything needed to say what produced an OCT adapter.

Written before the first training run, because provenance recorded afterwards is a
reconstruction. Emits one JSON manifest per run; commit it beside the adapter.

    python scripts/oct_provenance.py --run repro-123456 --stage dpo \
        --cmd "deepspeed --module openrlhf.cli.train_dpo ..."

Records: OCT commit, both submodule SHAs, package versions (including whether flash-attn is
importable, which changes numerics), base-model revision, GPU, the frozen data hashes, and
the exact command line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OCT = Path("/workspace/OpenCharacterTraining")
DATA = OCT / "data"
REPO = Path(__file__).resolve().parent.parent
FROZEN = [
    "dpo/llama-3.1-8b-it/impulsiveness.jsonl",
    "sft_data/llama-3.1-8b-it/impulsiveness.jsonl",
]


def sh(*a, cwd=None) -> str:
    try:
        return subprocess.run(a, cwd=cwd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"<{type(e).__name__}>"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "transformers", "deepspeed", "peft", "ray", "accelerate",
              "flash_attn", "vllm", "datasets"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception:                                     # noqa: BLE001
            out[m] = None            # None means NOT INSTALLED -- flash_attn changes numerics
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:                                         # noqa: BLE001
        out["cuda"], out["gpu"] = None, []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. repro-123456 or seed2")
    ap.add_argument("--stage", required=True, choices=["dpo", "sft", "fold", "merge"])
    ap.add_argument("--cmd", default="", help="the exact command line being run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "runs" / "oct"))
    a = ap.parse_args()

    man = {
        "run": a.run, "stage": a.stage,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": a.cmd, "notes": a.notes,
        "oct_commit": sh("git", "rev-parse", "HEAD", cwd=OCT),
        "oct_dirty": bool(sh("git", "status", "--porcelain", cwd=OCT)),
        "submodules": {n: sh("git", "rev-parse", "HEAD", cwd=OCT / n)
                       for n in ("openrlhf", "repeng")},
        "steering_repo_commit": sh("git", "rev-parse", "HEAD", cwd=REPO),
        "versions": versions(),
        "base_model": {
            "path": "/workspace/oct_rig/models/llama-3.1-8b-it",
            "revision": Path("/workspace/oct_rig/models/llama-3.1-8b-it").resolve().name,
        },
        "frozen_data": {f: {"sha256": sha256(DATA / f), "bytes": (DATA / f).stat().st_size}
                        for f in FROZEN if (DATA / f).exists()},
        "assumptions": [
            "loras/llama-test is a symlink to loras/llama-introspection; merge_loras.py:38 "
            "reads llama-test but nothing in the public repo writes it. If this reproduction "
            "differs materially from the released adapter, an unpublished/different "
            "llama-test SFT artifact is a candidate explanation.",
        ],
    }
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{a.run}_{a.stage}.json"
    out.write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
