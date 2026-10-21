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


def _resolve(f: str) -> Path:
    return Path(f) if Path(f).is_absolute() else DATA / f


def _launcher_sha(cmd: str) -> dict:
    """sha256 of any script the command names that exists under OCT or this repo."""
    out = {}
    for tok in cmd.split():
        for root in (OCT, REPO, Path("/")):
            p = (root / tok) if not tok.startswith("/") else Path(tok)
            if tok.endswith((".sh", ".py")) and p.is_file():
                out[tok] = sha256(p); break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. repro-123456 or seed2")
    # The three introspect_* stages exist for the paraphrase arms, which GENERATE their own
    # introspection corpus instead of training on the released one (spec_paraphrase_replication
    # 3.1). The reproduction and seed-2 arms had no such stage, so it was not in this list.
    ap.add_argument("--stage", required=True,
                    choices=["dpo", "sft", "fold", "merge", "corpus",
                             "introspect_reflection", "introspect_interaction",
                             "introspect_leading"])
    ap.add_argument("--cmd", default="", help="the exact command line being run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "runs" / "oct"))
    # REQUIRED, no default. A default of the impulsiveness pair made the traj-loving and all four
    # xchar manifests record impulsiveness hashes (corrected in docs/runs/oct/*.provenance_correction.json).
    ap.add_argument("--data", nargs="+", required=True,
                    help="training-input files whose sha256 to record: paths relative to OCT/data or "
                         "absolute; 'none' for a stage with no data input (fold, merge)")
    ap.add_argument("--start-model", default=None,
                    help="weights this stage starts from, if not the pipeline base (e.g. a folded model)")
    a = ap.parse_args()

    data = [] if a.data == ["none"] else a.data
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
        "start_model": a.start_model,
        "frozen_data": {f: {"sha256": sha256(_resolve(f)), "bytes": _resolve(f).stat().st_size}
                        for f in data if _resolve(f).exists()},
        "frozen_data_missing": [f for f in data if not _resolve(f).exists()],
        # oct_dirty alone could not reproduce a run; keep the diff and the untracked files' hashes
        "oct_diff_sha256": hashlib.sha256(sh("git", "diff", "HEAD", cwd=OCT).encode()).hexdigest(),
        "oct_diff": sh("git", "diff", "HEAD", cwd=OCT)[:200_000],
        "oct_untracked": {f: sha256(OCT / f) for f in sh("git", "ls-files", "--others", "--exclude-standard",
                                                            cwd=OCT).splitlines()
                          if f and "__pycache__" not in f and (OCT / f).is_file()},
        "launcher_sha256": _launcher_sha(a.cmd),
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
