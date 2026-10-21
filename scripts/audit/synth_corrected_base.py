#!/usr/bin/env python3
"""Approximate corrected-mask base cache, for SENSITIVITY ANALYSIS ONLY.

    base_synth[q] = base_legacy[q] + (xchar_lov_start[q] - loving[q])

per trait x persona x direction x question x layer. The added term is the mask artifact
measured on the released loving model (corrected minus legacy, same weights up to a bf16
fold: mask_artifact.py). Using it for the base assumes the artifact is model-independent,
which the 2026-08-19 mask_diag supports only approximately (arm-vs-base contrasts moved
< 0.01 when both sides switched mask). It is NOT a substitute for re-extracting the base with
the corrected mask; it estimates how much the mixed-protocol comparison moved each result.

Written into an audit cache dir (never outputs/_qcache) next to symlinks to the real caches,
with embedded provenance that says synthetic.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "_qcache"
DST = REPO / "outputs" / "analysis" / "audit" / "qcache_synth"
NAME = "base_synthcorrected"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.npz"):                       # the real caches, read-only, by symlink
        link = DST / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    zb, zl, zf = (np.load(SRC / f"{a}_L15_20.npz") for a in ("base", "loving", "xchar_lov_start"))
    for k in ("traits", "personas", "layers", "n_questions_per_trait"):
        assert np.array_equal(zb[k], zl[k]) and np.array_equal(zb[k], zf[k]), k
    b = zb["acts"]
    out = np.empty_like(b)
    for ti in range(b.shape[0]):                      # one trait at a time: bounded memory
        art = zf["acts"][ti].astype(np.float32) - zl["acts"][ti].astype(np.float32)
        out[ti] = (b[ti].astype(np.float32) + art).astype(np.float16)
        print(f"trait {ti}: artifact rms {np.sqrt((art ** 2).mean()):.4f}", flush=True)
    prov = {"mask": "corrected", "synthetic": True,
            "questions_sha": "archive-500q/v1", "personas_sha": "archive-12p/v1",
            "variant": 0, "max_questions": None, "answer_token_rule": "find_answer_token_position/v1",
            "derivation": "base(legacy) + [xchar_lov_start(corrected) - loving(legacy)], per question; "
                          "scripts/audit/synth_corrected_base.py. Sensitivity analysis only."}
    np.savez(DST / f"{NAME}_L15_20.npz", acts=out,
             **{k: zb[k] for k in zb.files if k != "acts"}, provenance=np.array(json.dumps(prov)))
    print("wrote", DST / f"{NAME}_L15_20.npz")


if __name__ == "__main__":
    sys.exit(main())
