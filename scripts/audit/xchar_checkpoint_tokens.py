#!/usr/bin/env python3
"""Cumulative supervised tokens at each xchar checkpoint (audit issue 4). CPU only.

The report labels checkpoints "25/50/100 % of the token budget", but they were saved at
optimizer-step fractions (save_steps = total_steps/4). This replays OpenRLHF's sampler to
recover the actual supervised-token fraction at each saved step:

  * dataset = the first `train_rows` source rows (plan.json), minus rows the SFT dataset drops
    (corpus_rows.json `dropped`); per-row supervised token counts from corpus_rows.json
  * order   = openrlhf.utils.distributed_sampler: torch.randperm(N, Generator(seed + epoch)),
    seed 123456, epoch 0, one replica (scripts/run_xchar_sft.sh)
  * step k  = the first 32*k sampled rows (micro 2 x accumulation 16 = train_batch 32)

Assumes the dataset keeps file order after filtering (HF datasets map/filter do). Writes
outputs/analysis/audit/xchar_checkpoint_tokens.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "outputs/analysis/xchar_sft/plan.json"
ROWS = Path("/workspace/oct_rig/data_xchar/corpus_rows.json")
SEED, BATCH = 123456, 32


def main():
    plan, rows = json.loads(PLAN.read_text()), json.loads(ROWS.read_text())
    out = {"method": __doc__.strip().splitlines()[0], "seed": SEED, "train_batch": BATCH, "corpora": {}}
    for c, p in plan["corpora"].items():
        r = rows[c]["rows"][: p["train_rows"]]
        sup = np.array([x.get("supervised", 0) for x in r if not x["dropped"]], dtype=np.int64)
        g = torch.Generator(); g.manual_seed(SEED + 0)
        perm = torch.randperm(len(sup), generator=g).numpy()
        cum = np.cumsum(sup[perm]); total = int(sup.sum())
        steps_total = len(sup) // BATCH
        pts = {}
        for k in (p["save_steps"], 2 * p["save_steps"], 3 * p["save_steps"], steps_total):
            n = min(k * BATCH, len(sup))
            pts[str(k)] = {"step_fraction": k / steps_total, "rows": n,
                           "supervised_tokens": int(cum[n - 1]), "token_fraction": float(cum[n - 1] / total)}
        out["corpora"][c] = {"dataset_rows": int(len(sup)), "optimizer_steps": steps_total,
                             "supervised_tokens_total": total, "checkpoints": pts}
        print(c, {k: (round(v["step_fraction"], 4), round(v["token_fraction"], 4)) for k, v in pts.items()})
    dst = REPO / "outputs/analysis/audit/xchar_checkpoint_tokens.json"
    dst.write_text(json.dumps(out, indent=1)); print("wrote", dst)


if __name__ == "__main__":
    main()
