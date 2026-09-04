#!/usr/bin/env python3
"""Hash the raw artifacts that git does not carry, so a reader can verify a copy.

The activation cache, the per-question A/B logit cells and the merged adapters are all
gitignored -- they are tens of gigabytes -- so the repo publishes only derived analysis
JSON. That is fine for reading the paper and useless for checking it. This writes a
tracked manifest of sha256 + size for every raw file behind a figure, so that whatever
those artifacts are eventually archived in (HF dataset, Zenodo, a volume snapshot), a
third party can confirm the copy they have is the copy the numbers came from.

    python scripts/artifact_manifest.py            # A/B logit cells (default)
    python scripts/artifact_manifest.py --include-qcache   # + the 28 GB activation cache
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUT = REPO / "workshop_iclr" / "data" / "raw_artifact_manifest.tsv"


def sha256(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-qcache", action="store_true",
                    help="also hash outputs/_qcache (28 GB; several minutes)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    pats = ["llama-3.1-8b-*/caa_logits/*.npz", "llama-3.1-8b-*/caa_logits_forced/*.npz"]
    if a.include_qcache:
        pats.append("_qcache/*.npz")

    rows = []
    for pat in pats:
        for f in sorted(OUTPUTS.glob(pat)):
            rows.append((str(f.relative_to(OUTPUTS)), f.stat().st_size, sha256(f)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# sha256 of the raw artifacts behind the figures; paths relative to outputs/\n")
        fh.write("# regenerate: python scripts/artifact_manifest.py\n")
        fh.write("path\tbytes\tsha256\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
    total = sum(r[1] for r in rows)
    print(f"{len(rows)} files, {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
