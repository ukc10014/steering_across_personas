#!/usr/bin/env python3
"""Dump per-step training loss for the four xchar runs from their tensorboard logs.

    PYTHONPATH=/workspace/pylibs-train-py312 python3 scripts/xchar_train_loss.py
"""
import glob, json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = {}
for arm in ("xchar_lov_imp", "xchar_lov_lov", "xchar_base_imp", "xchar_base_lov"):
    ev = sorted(glob.glob(f"/workspace/oct_rig/logs/xchar_tb/{arm}/**/events.*", recursive=True))
    # xchar_lov_imp also has the 07:07 run killed one minute in (gate bypass, see c0f3b57);
    # the run directories are timestamped, so the last one is the real run.
    if len(ev) > 1:
        print(f"{arm}: discarding {ev[:-1]}")
    ea = EventAccumulator(ev[-1]); ea.Reload()
    out[arm] = {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
Path("outputs/analysis/xchar_sft/train_loss.json").write_text(json.dumps(out))
for arm, d in out.items():
    l = [v for _, v in d["train/loss_mean"]]
    q = len(l) // 4
    print(f"{arm:16s} steps {len(l)}  first-10 {sum(l[:10])/10:.3f}  q1 {sum(l[q-5:q+5])/10:.3f}  "
          f"q2 {sum(l[2*q-5:2*q+5])/10:.3f}  last-10 {sum(l[-10:])/10:.3f}")
