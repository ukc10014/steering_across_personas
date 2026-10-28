#!/usr/bin/env python3
"""Check the plotted-data CSVs in paper_g1/figures against the source JSON, so the figures are
verified by value rather than by eye. Exit 1 on any mismatch.

    python3.12 paper_g1/verify_assets.py
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F = REPO / "paper_g1/figures"
CT = REPO / "outputs/analysis/common_trajectory"
JD = REPO / "outputs/analysis/audit/joint_dose"
sys.path.insert(0, str(REPO / "scripts/appendix_oct"))
fails, checks = [], 0


def close(a, b, tol=1e-4):
    if a in (None, "") or b is None:
        return (a in (None, "")) == (b is None)
    return abs(float(a) - float(b)) <= tol


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


def rows(name):
    return list(csv.DictReader(open(F / name)))


# 1. trajectory figures: every plotted point equals the source JSON
for name in ("fig_trajectory_cmbase_data.csv", "fig_trajectory_AB_cmbase_data.csv"):
    for r in rows(name):
        c, ser, arm = r["constitution"], r["series"], r["arm"]
        b = json.loads((CT / f"v2/behaviour_{c}_10personas.json").read_text())
        g = json.loads((CT / f"v2/geometry_{c}_cmbase.json").read_text())
        ok(len(b["personas"]) == 10 and "null" not in b["personas"], f"{c}: behaviour personas")
        ok(g["base_arm"] == "base_cmask", f"{c}: geometry base {g['base_arm']}")
        if r["panel"] == "A":
            src = next(q for q in b["series"][ser]["rows"] if q["arm"] == arm)["level"]
            ok(close(r["point"], src["point"]), f"{name} A {c} {ser} {arm}")
            if ser in ("B2", "LOV"):
                ok(close(r["lo"], src["lo"]) and close(r["hi"], src["hi"]), f"{name} A band {c} {ser} {arm}")
        elif ser == "q":
            src = next(q for q in b["profile"]["rows"] if q["arm"] == arm)["q"]
            ok(close(r["point"], src["point"]) and close(r["lo"], src["lo"]), f"{name} q {c} {arm}")
        elif ser in ("Q", "S", "SS", "g", "cosT"):
            src = g["layers"]["15"]["summary"]["MEAN8"][arm][ser]
            ok(close(r["point"], src["point"]) and close(r["lo"], src["lo"]) and close(r["hi"], src["hi"]),
               f"{name} {ser} {c} {arm}: {r['point']} vs {src['point']}")
# headline: q at 0.48 ep
EXP_Q = {"impulsiveness": 0.80, "loving": 0.94, "goodness": 0.84, "sycophancy": 0.71}
for r in rows("fig_trajectory_AB_cmbase_data.csv"):
    if r["series"] == "q" and abs(float(r["x"]) - 0.48) < 0.01:
        ok(round(float(r["point"]), 2) == EXP_Q[r["constitution"]], f"q(0.48) {r['constitution']} {r['point']}")

# 2. matched dose: every row equals the joint-bootstrap JSON; withheld cells never plotted
J = {c: json.loads((JD / f"joint_dose_{c}_cmbase.json").read_text()) for c in EXP_Q}
for r in rows("matched_dose_points_cmbase.csv"):
    c, L, est = r["constitution"], r["layer"], r["estimand"]
    j = J[c]; i = [round(e, 4) for e in j["epoch"]].index(float(r["epoch"]))
    src = (j["layers"]["15"]["behaviour_10personas"][est.replace("_10personas", "")] if est.endswith("_10personas")
           else j["layers"][L]["series"][est])[i]
    ok(close(r["point"], src["point"]) and close(r["lo"], src["lo"]) and close(r["hi"], src["hi"]),
       f"matched {c} L{L} {est} {r['epoch']}")
    ok((r["plotted"] == "True") == (src["estimable"] and float(r["epoch"]) > 0), f"plotted flag {c} L{L} {est} {r['epoch']}")
ep_end = lambda c: J[c]["epoch"][-1]
md = {(r["constitution"], r["layer"], r["estimand"], float(r["epoch"])): r for r in rows("matched_dose_points_cmbase.csv")}
b2 = md[("impulsiveness", "15", "offset_B2_10personas", round(ep_end("impulsiveness"), 4))]
ok((round(float(b2["point"]), 3), round(float(b2["lo"]), 3), round(float(b2["hi"]), 3)) == (1.369, 1.308, 1.436), f"B2 10p {b2}")
lv = md[("loving", "15", "offset_LOV_10personas", round(ep_end("loving"), 4))]
ok((round(float(lv["point"]), 3), round(float(lv["lo"]), 3), round(float(lv["hi"]), 3)) == (-0.124, -0.188, -0.054), f"LOV 10p {lv}")
syc = md[("sycophancy", "15", "S_MEAN8", round(ep_end("sycophancy"), 4))]
ok(syc["plotted"] == "False" and syc["point_status"] == "unbracketed", "sycophancy L15 endpoint withheld (unbracketed)")
gd = md[("goodness", "20", "S_MEAN8", round(J["goodness"]["epoch"][1], 4))]
ok(gd["plotted"] == "False" and gd["estimable"] == "False", "goodness L20 0.24 ep withheld")
ok(all(r["plotted"] == "False" for k, r in md.items() if k[:3] == ("sycophancy", "20", "S_MEAN8") and k[3] > 0),
   "sycophancy L20 per-trait-mean S withheld at every state")

# 3. xchar: every plotted offset/interval equals signed.json
sj = json.loads((REPO / "outputs/analysis/xchar_sft/signed.json").read_text())
START = {"xchar_lov_imp": "xchar_lov_start", "xchar_lov_lov": "xchar_lov_start", "xchar_base_imp": "base", "xchar_base_lov": "base"}
ok(len(sj["personas"]) == 11 and "null" in sj["personas"], "xchar: 11 personas incl. null")
for r in rows("acquisition_retention_steps_data.csv"):
    st = START[r["condition"]] if r["checkpoint"] == "start" else f"{r['condition']}_{r['checkpoint']}"
    o = sj["levels"][st][r["trait"]]["offset"]
    ok(close(r["offset"], o["point"], 1e-9) and close(r["ci_lo"], o["ci_lo"], 1e-9) and close(r["ci_hi"], o["ci_hi"], 1e-9),
       f"xchar {st} {r['trait']}")

# 4. SFT curve (corrected base): B1 unchanged from the published table, activation values from the corrected files
import curve_common as cc  # noqa: E402
pub = {(r["construction"], r["step"]): r for r in csv.DictReader(open(REPO / "workshop_iclr/data/tableA_oct_sft_curve.csv"))}
cc.use_corrected_base()
for r in rows("tableA_oct_sft_curve_cmbase.csv"):
    k = (r["construction"], r["step"])
    ok(k in pub and close(r["B1"], pub[k]["B1"], 1e-9), f"SFT-curve B1 changed at {k}")
    arm = "impulsiveness_repro_dpo" if r["step"] == "0" else \
        f"impulsiveness_curve_{r['construction']}_step{'final' if r['step'] == '1122' else r['step']}"
    d, s = cc.activation(arm)
    ok(close(r["dose"], d) and close(r["selectivity"], s), f"SFT-curve activation {arm}")

print(f"{checks} checks, {len(fails)} failures")
for f_ in fails[:30]:
    print("  FAIL", f_)
sys.exit(1 if fails else 0)
