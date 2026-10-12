#!/usr/bin/env python3
"""Appendix assets -- the reproduction-failure headline table and the stage-localisation table.

Table 1 (headline). Three arms x four preregistered criteria. The point it makes in one glance:
the rig reproduces the published phenotype closely when the UPSTREAM DATA ARE HELD FIXED, and the
fresh end-to-end regeneration does not.

Table 2 (stage localisation). The same B1 at four points along the pipeline, for the
reproduction and for P0. It localises the discrepancy without another figure.

PROVENANCE OF THE INTERVALS, which is the fiddly part.
  B1, B2   caa_logits_robustness.json, contrast `impulsivity_only` / `impulsivity+risk_taking`,
           LINEAR estimator. This is the source the P0 gate report quotes: its repro B1
           [1.8389, 1.9959] and P0 B1 [1.2138, 1.3535] are exactly the gate report's
           [+1.84, +2.00] and [+1.21, +1.35]. Using one cached source for both criteria keeps
           the table internally consistent.
           NOTE: caa_logits.json's own `selectivity` block carries a B2 CI that differs in the
           third decimal (P0 [1.2139, 1.3357] against [1.2297, 1.3208] here). Point estimates are
           identical everywhere; only the interval moves, and no verdict changes. The bootstrap
           draws are generated once per trait BEFORE the arm loop (caa_logits_analysis.py:209),
           so this is not an arm-set effect -- the two runs differ in --n-boot.
  B3       selectivity_score.py over the cached common-shift bootstrap. Released and P0 come from
           common_shift.json; the reproduction is only in common_shift_oct_stage.json. Released
           is present in BOTH and scores identically (1.7219 [1.4315, 2.0123]), which is what
           makes it legitimate to take the reproduction's row from the second file.
  A4       functional_dose_writeup.json, trait-vector displacement relative to released.
           Answer-token displacement is given alongside; both are reported because the gate
           band applies to the ratio and the two agree.

Source
  outputs/analysis/caa_logits.json, caa_logits_robustness.json, functional_dose_writeup.json
  outputs/analysis/common_shift.json, common_shift_oct_stage.json   (via selectivity_score.py)
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs" / "analysis"
TEX = REPO / "workshop_iclr" / "tables"
MD = REPO / "docs" / "runs" / "oct"

ARMS = [("released OCT", "impulsiveness", "common_shift.json"),
        ("fixed-data reproduction", "impulsiveness_repro", "common_shift_oct_stage.json"),
        ("P0 end-to-end regeneration", "impulsiveness_regen", "common_shift.json")]
BANDS = {"B1": "$\\geq +1.5$", "B2": "$\\geq +1.4$", "B3": "$\\geq 1.4$", "A4": "$[0.7, 1.4]$"}


def b3(arm: str, cs: str) -> tuple[float, float, float]:
    """B3 and its CI, from the cached common-shift bootstrap via the canonical scorer."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "selectivity_score.py"), "--arm", arm,
         "--targets", "impulsivity", "risk_taking", "--layers", "15",
         "--common-shift", str(ANALYSIS / cs)],
        capture_output=True, text=True, cwd=REPO)
    for line in r.stdout.splitlines():
        if "RATIO" in line:
            body = line.split("RATIO =")[1]
            pt = float(body.split("[")[0])
            lo, hi = (float(x) for x in body.split("[")[1].split("]")[0].split(","))
            return pt, lo, hi
    raise SystemExit(f"no RATIO for {arm} in {cs}:\n{r.stdout}\n{r.stderr}")


def main() -> None:
    off = json.loads((ANALYSIS / "caa_logits.json").read_text())["forced"]["offset"]
    rob = json.loads((ANALYSIS / "caa_logits_robustness.json").read_text())["forced"]["contrasts"]
    dose = json.loads((ANALYSIS / "functional_dose_writeup.json").read_text())

    def crit(key: str, arm: str) -> tuple[float, float, float]:
        v = rob[key]["by_arm"][arm]["linear"]
        return v["point"], v["ci_lo"], v["ci_hi"]

    def dose_ratio(arm: str) -> tuple[float, float]:
        ref = dose["15"]["impulsiveness"]
        a = dose["15"][arm]
        return (a["trait_vector_displacement"] / ref["trait_vector_displacement"],
                a["answer_token_displacement"] / ref["answer_token_displacement"])

    rows = []
    for label, arm, cs in ARMS:
        r = {"label": label, "arm": arm}
        r["B1"] = crit("impulsivity_only", arm)
        r["B2"] = crit("impulsivity+risk_taking", arm)
        r["B3"] = b3(arm, cs)
        r["A4"] = dose_ratio(arm)
        rows.append(r)

    # ---------------------------------------------------------------- table 1, markdown
    L = ["| model | B1 (impulsivity alone) | B2 (target pair) | activation selectivity B3 "
         "| functional dose A4 |", "|---|---|---|---|---|",
         "| *preregistered band* | ≥ +1.5 | ≥ +1.4 | ≥ 1.4 | ∈ [0.7, 1.4] |"]
    for r in rows:
        tv, at = r["A4"]
        L.append(f"| {r['label']} | {r['B1'][0]:+.3f} [{r['B1'][1]:+.3f}, {r['B1'][2]:+.3f}] "
                 f"| {r['B2'][0]:+.3f} [{r['B2'][1]:+.3f}, {r['B2'][2]:+.3f}] "
                 f"| {r['B3'][0]:.3f} [{r['B3'][1]:.3f}, {r['B3'][2]:.3f}] "
                 f"| {tv:.3f} tv / {at:.3f} at |")
    t1_md = "\n".join(L)

    # ---------------------------------------------------------------- table 2, stage localisation
    def B1(arm: str) -> float:
        o = off[arm]
        return o["impulsivity"]["point"] - st.mean(
            v["point"] for t, v in o.items() if t != "impulsivity")

    stages = [("DPO only, $M_D$", "impulsiveness_repro_dpo", "impulsiveness_regen_dpo"),
              ("SFT component off base", "impulsiveness_repro_sft", "impulsiveness_regen_sft"),
              ("native $D+S$", "impulsiveness_repro_DplusS", "impulsiveness_regen_DplusS"),
              ("final PEFT merge, $M_F$", "impulsiveness_repro", "impulsiveness_regen")]
    L2 = ["| state | reproduction B1 | P0 B1 | P0 − reproduction |", "|---|---|---|---|"]
    stage_vals = []
    for lbl, ra, pa in stages:
        r1, p1 = B1(ra), B1(pa)
        stage_vals.append((lbl, r1, p1))
        L2.append(f"| {lbl.replace('$','')} | {r1:+.3f} | {p1:+.3f} | {p1 - r1:+.3f} |")
    t2_md = "\n".join(L2)

    MD.mkdir(parents=True, exist_ok=True)
    (MD / "tableA_paraphrase_headline.md").write_text(
        "# Headline table — reproduction holds, end-to-end regeneration does not\n\n"
        + t1_md
        + "\n\n*tv = trait-vector displacement, at = answer-token displacement, both relative to "
          "released. B1/B2 intervals: `caa_logits_robustness.json`, linear estimator (the source "
          "the P0 gate report quotes). B3: cached common-shift bootstrap via "
          "`selectivity_score.py`. A4: `functional_dose_writeup.json`.*\n\n"
        "# Stage localisation — where the discrepancy enters\n\n" + t2_md + "\n")
    print(f"wrote {MD}/tableA_paraphrase_headline.md\n")
    print(t1_md + "\n\n" + t2_md)

    # ---------------------------------------------------------------- LaTeX
    T = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"model & $B_1$ & $B_2$ & activation sel. $B_3$ & functional dose $A_4$ \\",
         r"\midrule",
         r"\emph{band} & " + " & ".join(BANDS[k] for k in ("B1", "B2", "B3", "A4")) + r" \\",
         r"\midrule"]
    for i, r in enumerate(rows):
        tv, at = r["A4"]
        if i:
            T.append(r"\addlinespace")
        T.append(f"{r['label']} & ${r['B1'][0]:+.3f}$ & ${r['B2'][0]:+.3f}$ & "
                 f"${r['B3'][0]:.3f}$ & ${tv:.3f}$ \\\\")
        T.append(f" & {{\\scriptsize $[{r['B1'][1]:+.3f}, {r['B1'][2]:+.3f}]$}} & "
                 f"{{\\scriptsize $[{r['B2'][1]:+.3f}, {r['B2'][2]:+.3f}]$}} & "
                 f"{{\\scriptsize $[{r['B3'][1]:.3f}, {r['B3'][2]:.3f}]$}} & "
                 f"{{\\scriptsize ${at:.3f}$ ans.-tok.}} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Preregistered criteria for the released checkpoint, a reproduction that "
          r"holds the upstream teacher data fixed, and a fresh end-to-end regeneration (P0). "
          r"The reproduction clears every band; P0 fails $B_1$, $B_2$ and $B_3$ while passing "
          r"the dose check, so the failure is not a dose deficit. Intervals are paired-question "
          r"bootstrap.}",
          r"\label{tab:paraphrase-headline}", r"\end{table}"]
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "tableA_paraphrase_headline.tex").write_text("\n".join(T) + "\n")
    print(f"\nwrote {TEX}/tableA_paraphrase_headline.tex")

    S = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"state & reproduction $B_1$ & P0 $B_1$ & P0 $-$ reproduction \\",
         r"\midrule"]
    for lbl, r1, p1 in stage_vals:
        S.append(f"{lbl} & ${r1:+.3f}$ & ${p1:+.3f}$ & ${p1 - r1:+.3f}$ \\\\")
    S += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Where the discrepancy enters. $B_1$ at four points along the pipeline for "
          r"the fixed-data reproduction and for the end-to-end regeneration P0. The regenerated "
          r"DPO adapter is weaker on the registered phenotype; the introspection/SFT channel is "
          r"essentially intact ($-0.036$); some discrepancy is present in the native $D+S$ "
          r"trajectory; and the released-style factor merge widens the gap furthest. Changing the "
          r"teacher dataset had a far larger DPO-stage effect than changing the DPO training "
          r"seed, so this is a teacher-data/protocol difference rather than teacher sampling "
          r"alone---P0 also lacks the original assistant prefill.}",
          r"\label{tab:paraphrase-stage}", r"\end{table}"]
    (TEX / "tableA_paraphrase_stage.tex").write_text("\n".join(S) + "\n")
    print(f"wrote {TEX}/tableA_paraphrase_stage.tex")


if __name__ == "__main__":
    main()
