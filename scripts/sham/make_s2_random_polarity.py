#!/usr/bin/env python3
"""S2 — the random-polarity causal sham (spec_sham_lora.md §2), DPO stage.

Keep every prompt and both completion texts exactly as released; flip which member is
`chosen` with p = 0.5, independently per row. The only thing destroyed is the relationship
between text content and preference direction. Data volume, token statistics, coherent
English and teacher-generated prose are all preserved by construction, because no text is
ever rewritten -- only the two column values are exchanged.

WHY THE ASSIGNMENT IS FROZEN ON DISK (spec §2): fixed random labels are a target the
optimizer can fit and memorise; labels resampled per epoch drive the expected gradient
toward cancellation and would turn a substantive null into an artefact of the label
schedule. Transforming the file once, before training, makes redrawing impossible rather
than merely discouraged -- DPO then reads the same bytes every step.

The writer is pandas `to_json(orient="records", lines=True, force_ascii=True)`, which
reproduces the frozen source byte-for-byte (verified: the round-trip sha256 equals the §6a
hash). So an unflipped row is byte-identical to its source row, and a flipped row contains
the identical two strings in exchanged positions -- which is what makes the sanity checks
below assertions about bytes rather than about parsed approximations.

Usage:
    python scripts/sham/make_s2_random_polarity.py              # build + check + manifest
    python scripts/sham/make_s2_random_polarity.py --verify     # reproduce, compare hashes
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Frozen source, spec §6a. The hash is asserted, not trusted: a silently different DPO file
# would make the sham incomparable to every arm measured so far.
SRC = Path("/workspace/OpenCharacterTraining/data/dpo/llama-3.1-8b-it/impulsiveness.jsonl")
SRC_SHA256 = "53c6a54c581e6c68660b039991ff5ab9a490f01bd1f382be2c099975230ffc91"
SRC_BYTES = 35_301_875

OUT = Path("/workspace/oct_rig/data_sham/s2_random_polarity/impulsiveness.jsonl")
MANIFEST = Path("docs/runs/oct/s2_random_polarity_manifest.json")

# Distinct from either training seed (123456, 987654) so the label draw can never be
# confused with an optimisation seed.
LABEL_SEED = 20261005
RNG_ALGORITHM = "numpy.random.default_rng(seed) -> PCG64, flips = rng.random(n) < 0.5"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_flips(n: int, seed: int) -> np.ndarray:
    """The frozen assignment. Depends only on (n, seed) -- reproducible from the manifest."""
    return np.random.default_rng(seed).random(n) < 0.5


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_json(orient="records", lines=True, force_ascii=True),
                    encoding="utf-8")


def text_of(msgs: list[dict]) -> tuple[str, str]:
    """-> (user prompt, assistant response). Schema verified uniform: all 8137 rows are
    exactly [user, assistant] for both members, user turn identical within a pair."""
    assert [m["role"] for m in msgs] == ["user", "assistant"], f"unexpected roles: {msgs}"
    return msgs[0]["content"], msgs[1]["content"]


def check(src: pd.DataFrame, out: pd.DataFrame, flips: np.ndarray) -> dict:
    """Every assertion here is a pre-committed sanity check (§3.1 is about optimisation;
    these are about the transformation being what it claims)."""
    res: dict = {}

    # 1. Row count unchanged.
    assert len(out) == len(src) == len(flips), (len(out), len(src), len(flips))
    res["n_rows"] = int(len(out))

    # 2/3. Per row: prompt preserved, response texts preserved as an unordered pair, and
    #      the row is exactly swapped iff it was drawn to flip.
    n_swapped = n_identical = 0
    for i in range(len(src)):
        sc_p, sc_r = text_of(src["chosen"].iloc[i])
        sj_p, sj_r = text_of(src["rejected"].iloc[i])
        oc_p, oc_r = text_of(out["chosen"].iloc[i])
        oj_p, oj_r = text_of(out["rejected"].iloc[i])

        assert sc_p == sj_p == oc_p == oj_p, f"row {i}: prompt changed"
        assert {sc_r, sj_r} == {oc_r, oj_r}, f"row {i}: response texts changed"

        if flips[i]:
            assert (oc_r, oj_r) == (sj_r, sc_r), f"row {i}: marked flipped but not swapped"
            n_swapped += 1
        else:
            assert (oc_r, oj_r) == (sc_r, sj_r), f"row {i}: marked unflipped but changed"
            n_identical += 1

    assert n_swapped == int(flips.sum()), (n_swapped, int(flips.sum()))
    res["n_flipped"] = n_swapped
    res["n_unchanged"] = n_identical
    res["frac_flipped"] = n_swapped / len(src)

    # 4. Roughly half. Two-sided binomial z; |z| > 4 would mean the draw is not p=0.5.
    n = len(src)
    z = (n_swapped - n / 2) / np.sqrt(n * 0.25)
    res["flip_binomial_z"] = float(z)
    assert abs(z) < 4.0, f"flip fraction implausible for p=0.5: z={z:.2f}"

    # 5. Token/length statistics preserved exactly, as a multiset over the whole corpus.
    #    Swapping cannot change these -- asserting it catches any accidental rewrite.
    def all_lengths(df: pd.DataFrame) -> list[int]:
        out_: list[int] = []
        for col in ("chosen", "rejected"):
            for msgs in df[col]:
                out_ += [len(m["content"]) for m in msgs]
        return sorted(out_)

    assert all_lengths(src) == all_lengths(out), "corpus length multiset changed"
    res["total_chars"] = int(sum(all_lengths(out)))

    # 6. The pair-as-a-set is preserved corpus-wide, so no row borrowed another's text
    #    (this is what separates S2 from S1's deliberate re-pairing).
    def pairset(df: pd.DataFrame) -> list[tuple[str, str, str]]:
        rows = []
        for i in range(len(df)):
            p, a = text_of(df["chosen"].iloc[i])
            _, b = text_of(df["rejected"].iloc[i])
            lo, hi = sorted([a, b])
            rows.append((p, lo, hi))
        return sorted(rows)

    assert pairset(src) == pairset(out), "prompt/response pairing changed"
    res["pairing_preserved"] = True
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--expect-sha256", default=SRC_SHA256)
    ap.add_argument("--label-seed", type=int, default=LABEL_SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare against the existing manifest; write nothing")
    a = ap.parse_args()

    # --- provenance gate on the source -------------------------------------------------
    src_sha = sha256_file(a.source)
    src_bytes = a.source.stat().st_size
    print(f"source     {a.source}")
    print(f"  bytes    {src_bytes:,}" + ("" if src_bytes == SRC_BYTES else f"  (spec §6a: {SRC_BYTES:,})"))
    print(f"  sha256   {src_sha}")
    if a.expect_sha256 and src_sha != a.expect_sha256:
        print(f"FAIL: source hash != expected {a.expect_sha256}", file=sys.stderr)
        return 2
    print("  hash matches spec §6a — frozen source confirmed")

    df = pd.read_json(a.source, lines=True)

    # The writer must reproduce the source exactly, or "byte-identical modulo ordering"
    # is not a claim this script can make.
    rt = hashlib.sha256(df.to_json(orient="records", lines=True,
                                   force_ascii=True).encode()).hexdigest()
    assert rt == src_sha, f"writer does not round-trip the source: {rt} != {src_sha}"
    print("  writer round-trips the source byte-for-byte")

    # --- the frozen draw ----------------------------------------------------------------
    flips = draw_flips(len(df), a.label_seed)
    packed = base64.b64encode(np.packbits(flips).tobytes()).decode()

    out = df.copy()
    sw = flips
    out.loc[sw, ["chosen", "rejected"]] = df.loc[sw, ["rejected", "chosen"]].values
    print(f"\nlabel seed {a.label_seed}  ->  flipped {int(flips.sum())}/{len(df)} "
          f"({flips.mean():.4f})")

    stats = check(df, out, flips)
    print("sanity checks: all passed")
    for k, v in stats.items():
        print(f"  {k:22s} {v}")

    body = out.to_json(orient="records", lines=True, force_ascii=True).encode()
    out_sha = hashlib.sha256(body).hexdigest()
    print(f"\ntransformed sha256 {out_sha}\ntransformed bytes  {len(body):,}")

    manifest = {
        "experiment": "S2 random polarity (spec_sham_lora.md §2), DPO stage",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"path": str(a.source), "sha256": src_sha, "bytes": src_bytes,
                   "spec_6a_sha256": SRC_SHA256, "matches_spec_6a": src_sha == SRC_SHA256},
        "transformed": {"path": str(a.out), "sha256": out_sha, "bytes": len(body)},
        "assignment": {
            "label_seed": a.label_seed,
            "rng": RNG_ALGORITHM,
            "n_rows": stats["n_rows"],
            "n_flipped": stats["n_flipped"],
            "frac_flipped": stats["frac_flipped"],
            "binomial_z": stats["flip_binomial_z"],
            "flips_packbits_b64": packed,
            "flips_sha256": hashlib.sha256(np.packbits(flips).tobytes()).hexdigest(),
            "note": "reproducible two ways: redraw from (n_rows, label_seed), or unpack "
                    "flips_packbits_b64. Frozen on disk; never redrawn across epochs.",
        },
        "checks": {k: v for k, v in stats.items()},
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__},
    }

    if a.verify:
        if not a.manifest.exists():
            print("FAIL: --verify but no manifest", file=sys.stderr)
            return 2
        old = json.loads(a.manifest.read_text())
        ok = True
        for field, new in (("source", src_sha), ("transformed", out_sha)):
            was = old[field]["sha256"]
            same = was == new
            ok &= same
            print(f"  {field:12s} {'MATCH' if same else 'DIFFER'}  {was} -> {new}")
        same_asg = old["assignment"]["flips_packbits_b64"] == packed
        ok &= same_asg
        print(f"  assignment   {'MATCH' if same_asg else 'DIFFER'}")
        print("VERIFY OK" if ok else "VERIFY FAILED")
        return 0 if ok else 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(body)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {a.out}\nwrote {a.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
