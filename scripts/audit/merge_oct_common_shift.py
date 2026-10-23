#!/usr/bin/env python3
"""Merge the chunked corrected-base OCT common-shift runs into one file shaped like
outputs/analysis/common_shift_oct_stage.json, for plotting code that reads that shape.

Per-arm quantities (per_arm, g_over_base_ci, share_squared_ci) do not depend on which arms
shared a common_shift.py call and are merged as is. The cross-arm cosine blocks (cos, cos_naive,
cos_ci) were computed only WITHIN each 13-arm chunk, so they are dropped rather than merged into
something that looks complete; the output records that.

    python3 scripts/audit/merge_oct_common_shift.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OA = REPO / "outputs/analysis/audit/oct_stage_cmbase"
PER_ARM = ("per_arm", "g_over_base_ci", "share_squared_ci")


def main():
    parts = sorted(p for p in OA.glob("common_shift_oct_cmbase_part*.json") if not p.name.endswith(".provenance.json"))
    out = {}
    for p in parts:
        j = json.loads(p.read_text())
        for L, Lv in j.items():
            for t, tv in Lv.items():
                dst = out.setdefault(L, {}).setdefault(t, {k: {} for k in PER_ARM})
                for k in PER_ARM:
                    for arm, v in (tv.get(k) or {}).items():
                        assert arm not in dst[k], f"{arm} in two chunks"
                        dst[k][arm] = v
    prov = [json.loads(p.with_name(p.stem + ".provenance.json").read_text()) for p in parts]
    (OA / "common_shift_oct_cmbase.json").write_text(json.dumps(out, indent=1))
    (OA / "common_shift_oct_cmbase.provenance.json").write_text(json.dumps({
        "merged_from": [p.name for p in parts], "base_arm": sorted({q["base_arm"] for q in prov}),
        "note": "per-arm quantities only; cross-arm cosines (cos, cos_naive, cos_ci) were chunk-local and are omitted",
        "chunk_provenance": prov}, indent=1, default=str))
    n = len(next(iter(out.values()))["assertiveness"]["per_arm"])
    print(f"merged {len(parts)} parts, {n} arms, layers {sorted(out)}")


if __name__ == "__main__":
    main()
