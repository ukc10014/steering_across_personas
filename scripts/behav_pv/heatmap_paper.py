#!/usr/bin/env python3
"""Paper export of the behav-PV heatmap: vector PDF (no suptitle; the caption carries it) plus a
CSV of every plotted value, drawn from the archived results JSON with analyze.heatmap, so no
responses or judgments are needed."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heatmap, excludes0

RESULTS = Path("outputs/behav_pv/analysis/results_full_audit.json")
OUT = Path("paper_g1/figures")


def main():
    out = json.loads(RESULTS.read_text())
    cons = [c for c in out["arms"] if c != "base"]
    personas = out["personas"]
    OUT.mkdir(parents=True, exist_ok=True)
    heatmap(out, cons, personas, OUT / "fig_behav_pv_heatmap.pdf", suptitle=False)
    with open(OUT / "fig_behav_pv_heatmap_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "arm", "persona", "point", "lo", "hi", "ci_excludes_0"])
        for metric in ("evil", "share_gt50"):
            for c in cons:
                for p in personas:
                    d = out["metrics"][metric]["delta"][c][p]
                    w.writerow([metric, c, p, d["point"], d["lo"], d["hi"], excludes0(d, metric)])
    print("wrote", OUT / "fig_behav_pv_heatmap.pdf", "and", OUT / "fig_behav_pv_heatmap_data.csv")


if __name__ == "__main__":
    main()
