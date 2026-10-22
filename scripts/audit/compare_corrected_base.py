#!/usr/bin/env python3
"""Historical (mixed protocol) vs synthetic-base sensitivity vs ACTUAL corrected-mask base (G1).
CPU only; reads the JSON outputs, writes outputs/analysis/audit/corrected_base_comparison.{json,md}.

Columns, wherever they exist:
  orig    the reported number (legacy-mask base vs corrected-mask arms; original estimator)
  mixed   audit v2 rerun, same comparison, estimability-rule estimator
  synth   audit v2 rerun against the SYNTHETIC corrected base (legacy base + measured artifact)
  cm      this run, against the ACTUAL corrected-mask base `base_cmask`

Sections: geometry (trajectory), functional dose, matched-dose excess (historical, unpaired
point vs joint paired bootstrap), xchar selectivity, loving fidelity, OCT stage/curve arms.
Missing inputs are listed, not guessed.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "outputs/analysis"
CT = AN / "common_trajectory"
AU = AN / "audit"
CONS = ["impulsiveness", "loving", "goodness", "sycophancy"]
QS = ["g", "S", "cosT", "SS", "cosS", "Q"]


def rj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pt(e):
    return None if e is None else e.get("point")


def f3(v, sign=False):
    return "—" if v is None else (f"{v:+.3f}" if sign else f"{v:.3f}")


def fci(e, sign=False):
    if e is None or e.get("point") is None:
        return "n/e" if e is not None else "—"
    s = f3(e["point"], sign)
    return s + (f" [{f3(e['lo'], sign)}, {f3(e['hi'], sign)}]" if e.get("lo") is not None else "")


def geometry(out, md, missing):
    md.append("## 1. Trajectory geometry, MEAN8 (per-trait mean)\n")
    md.append("Max |cm − mixed| and |cm − synth| over all states, per quantity. TOTAL quantities (g, S, cosT) "
              "depend on the base; incremental ones (SS, cosS, Q) do not, so cm − mixed there is bootstrap/split "
              "identity only (same seeds) and should be ~0.\n")
    md.append("| constitution | layer | quantity | max abs(cm − mixed) | max abs(cm − synth) | at state | cm non-estimable states |")
    md.append("|---|---|---|---|---|---|---|")
    out["geometry"] = {}
    for c in CONS:
        O, M, Sy, C = (rj(CT / f"geometry_{c}.json"), rj(CT / f"v2/geometry_{c}_mixedprotocol.json"),
                       rj(CT / f"v2/geometry_{c}_synthbase.json"), rj(CT / f"v2/geometry_{c}_cmbase.json"))
        if C is None:
            missing.append(f"geometry {c} (cm)"); continue
        out["geometry"][c] = {}
        for L in ("15", "20"):
            sm, ss, sc = (R["layers"][L]["summary"]["MEAN8"] for R in (M, Sy, C))
            so = O["layers"][L]["summary"]["MEAN8"] if O else {}
            rows = {}
            for arm in sc:
                rows[arm] = {q: {"orig": (so.get(arm, {}).get(q) or {}).get("point"), "mixed": sm[arm].get(q),
                                 "synth": ss[arm].get(q), "cm": sc[arm].get(q)} for q in QS}
            out["geometry"][c][L] = rows
            for q in QS:
                dm, ds, where, ne = [], [], [], []
                for arm, r in rows.items():
                    a_, b_, c_ = pt(r[q]["mixed"]), pt(r[q]["synth"]), pt(r[q]["cm"])
                    if c_ is None:
                        if r[q]["cm"] is not None and r[q]["cm"].get("point") is None:
                            ne.append(arm.replace(c + "_", ""))
                        continue
                    if a_ is not None:
                        dm.append(abs(c_ - a_)); where.append(arm.replace(c + "_", ""))
                    if b_ is not None:
                        ds.append(abs(c_ - b_))
                i = int(np.argmax(dm)) if dm else None
                md.append(f"| {c} | L{L} | {q} | {f3(max(dm) if dm else None)} | {f3(max(ds) if ds else None)} | "
                          f"{where[i] if i is not None else '—'} | {', '.join(ne) if ne else '—'} |")
    md.append("")
    md.append("### Table points (L15 / L20), MEAN8: orig → cm\n")
    md.append("| constitution | state | g L15 | S L15 | cosT L15 | S L20 | cosT L20 |")
    md.append("|---|---|---|---|---|---|---|")
    for c, Lv in out["geometry"].items():
        for arm in Lv["15"]:
            r15, r20 = Lv["15"][arm], Lv["20"][arm]
            cell = lambda r, q: f"{f3(r[q]['orig'])} → {fci(r[q]['cm'])}"
            md.append(f"| {c} | {arm.replace(c + '_', '')} | {cell(r15, 'g')} | {cell(r15, 'S')} | {cell(r15, 'cosT')} | "
                      f"{cell(r20, 'S')} | {cell(r20, 'cosT')} |")
    md.append("")


def doses(out, md, missing):
    md.append("## 2. Functional dose (trait-vector displacement), legacy base → corrected base\n")
    md.append("| constitution | arm set | L15 ratio cm/legacy (min–max) | L20 ratio (min–max) | ladder range L15 legacy → cm | ladder range L20 legacy → cm |")
    md.append("|---|---|---|---|---|---|")
    out["dose"] = {}
    for c in CONS:
        r = {}
        for kind, suf in (("trajectory", ""), ("ladder", "_dpo_ladder")):
            old, new = rj(CT / f"functional_dose_{c}{suf}.json"), rj(CT / f"functional_dose_{c}{suf}_cmbase.json")
            if new is None or old is None:
                missing.append(f"dose {c} {kind}"); continue
            r[kind] = {L: {a: {"legacy": old[L][a]["trait_vector_displacement"], "cm": new[L][a]["trait_vector_displacement"]}
                           for a in new[L] if a in old.get(L, {})} for L in ("15", "20") if L in new and L in old}
        out["dose"][c] = r
        if "trajectory" in r and "ladder" in r:
            def rng_(kind, L, col):
                v = [x[col] for x in r[kind].get(L, {}).values()]
                return f"{min(v):.3f}–{max(v):.3f}" if v else "—"
            def rat(L):
                v = [x["cm"] / x["legacy"] for x in r["trajectory"].get(L, {}).values()]
                return f"{min(v):.4f}–{max(v):.4f}" if v else "— (no legacy file)"
            md.append(f"| {c} | trajectory | {rat('15')} | {rat('20')} | "
                      f"{rng_('ladder', '15', 'legacy')} → {rng_('ladder', '15', 'cm')} | "
                      f"{rng_('ladder', '20', 'legacy')} → {rng_('ladder', '20', 'cm')} |")
    md.append("")


def matched(out, md, missing):
    md.append("## 3. Matched-dose excess (trajectory − scaled-DPO ladder)\n")
    md.append("historical = control_dose_<c>.json (legacy base, separate bootstraps, point only). "
              "cm = joint paired bootstrap on the corrected base (shared draws, dose matched per replicate). "
              "`defined` = fraction of replicates bracketed and estimable; `unbr` = unbracketed.\n")
    out["matched"] = {}
    for c in CONS:
        H, J = rj(CT / f"control_dose_{c}.json"), rj(AU / f"joint_dose/joint_dose_{c}_cmbase.json")
        if J is None:
            missing.append(f"joint matched-dose {c}"); continue
        ep = J["epoch"]
        md.append(f"### {c}\n")
        md.append("| series | epoch | historical | cm point [95% CI] | defined | unbr |")
        md.append("|---|---|---|---|---|---|")
        res = {}
        series = [("15", "series", "S_MEAN8"), ("20", "series", "S_MEAN8"), ("15", "series", "Spooled_MEAN8"),
                  ("20", "series", "Spooled_MEAN8"), ("15", "series", "g_MEAN8"), ("15", "series", "S_TARGETS")]
        beh = J["layers"]["15"].get("behaviour_11personas_historical", {})
        series += [("15", "behaviour_11personas_historical", k) for k in beh] + \
                  [("15", "behaviour_10personas", k) for k in J["layers"]["15"].get("behaviour_10personas", {})]
        for L, blk, k in series:
            js = J["layers"].get(L, {}).get(blk, {}).get(k)
            if js is None:
                continue
            hk = k if blk != "series" or k.startswith(("g_", "S_")) else None
            hist = (H or {}).get("layers", {}).get(L, {}).get("series", {}).get(hk, {}).get("excess") if (hk and blk != "behaviour_10personas") else None
            res[f"L{L}:{blk}:{k}"] = {"historical": hist, "cm": js}
            show = blk == "series" or k.startswith("offset_B2") or k.startswith("offset_LOV") or blk == "behaviour_10personas"
            for i, e in enumerate(js):
                if i == 0 or not show:
                    continue
                if blk == "behaviour_10personas" and i != len(js) - 1:
                    continue                                   # per-trait 10-persona: endpoint only in the table
                h = hist[i] if hist else None
                md.append(f"| L{L} {k} ({blk.split('_')[0] if blk != 'series' else 'geometry'}{', 10p' if blk == 'behaviour_10personas' else ''}) "
                          f"| {ep[i]:.2f} | {f3(h, True)} | {fci(e, True)}{'' if e['estimable'] else ' (' + (e.get('reason') or '') + ')'} | "
                          f"{e['defined_frac']:.3f} | {e['unbracketed_frac']:.3f} |")
        out["matched"][c] = res
        md.append("")


def xchar(out, md, missing):
    md.append("## 4. Cross-character activation selectivity (target/other g, 10 personas)\n")
    md.append("| state | L | imp+risk: reported → synth → cm [95% CI] | emp+warmth: reported → synth → cm [95% CI] |")
    md.append("|---|---|---|---|")
    syn = rj(AU / "xchar_selectivity_mask.json")
    out["xchar"] = {}
    for A in ["xchar_lov_start", "xchar_lov_imp_s100", "xchar_lov_lov_s100", "xchar_base_imp_s100", "xchar_base_lov_s100"]:
        for L in ("15", "20"):
            row = {}
            cells = []
            for tag, key in (("imp", "imp+risk"), ("lov", "emp+warmth")):
                o = rj(AN / f"xchar_sft/selectivity_{A}_{tag}.json"); n = rj(AN / f"xchar_sft/selectivity_{A}_{tag}_cmbase.json")
                if n is None:
                    missing.append(f"xchar selectivity {A} {tag}"); cells.append("—"); continue
                ov = o["activation"][L]["ratio"] if o else None
                sv = syn[L]["synth_corrected_base"][A][key] if syn else None
                nv = n["activation"][L]
                row[key] = {"reported": ov, "synth": sv, "cm": nv["ratio"], "cm_ci": [nv["ratio_ci_lo"], nv["ratio_ci_hi"]]}
                cells.append(f"{f3(ov)} → {f3(sv)} → **{nv['ratio']:.3f}** [{nv['ratio_ci_lo']:.2f}, {nv['ratio_ci_hi']:.2f}]")
            out["xchar"].setdefault(A, {})[L] = row
            md.append(f"| {A.replace('xchar_', '')} | {L} | {cells[0]} | {cells[1]} |")
    md.append("")


def fidelity(out, md, missing):
    md.append("## 5. Loving reproduction fidelity (released = xchar_lov_start, folded weights)\n")
    md.append("| version | A4 dose ratio L15/L20 | B4 cos(common shift) L15/L20 | category |")
    md.append("|---|---|---|---|")
    out["fidelity_loving"] = {}
    for name, p in (("recorded (mixed)", CT / "fidelity_loving.json"), ("synthetic corrected base", CT / "v2/fidelity_loving_corrected.json"),
                    ("actual corrected base", CT / "v2/fidelity_loving_cmbase.json")):
        f = rj(p)
        if f is None:
            missing.append(f"fidelity {name}"); continue
        out["fidelity_loving"][name] = {"A4": f["A4"], "B4": f["B4"], "category": f["category"]}
        a4, b4 = f["A4"]["dose_ratio"], f["B4"]["mean_cos_common_shift"]
        md.append(f"| {name} | {a4['15']:.3f} / {a4['20']:.3f} | {b4['15']:.3f} / {b4['20']:.3f} | {f['category']} |")
    md.append("\nImpulsiveness, goodness and sycophancy fidelity (A4/B4 on one protocol) need the released "
              "corrected-mask arms, which were not extracted in this run's GPU budget.\n")


def oct_stage(out, md, missing):
    OA = AU / "oct_stage_cmbase"
    fl, fc = rj(OA / "functional_dose_oct_mixedprotocol.json"), rj(OA / "functional_dose_oct_cmbase.json")
    old = rj(AN / "common_shift_oct_stage.json")
    parts = sorted(OA.glob("common_shift_oct_cmbase_part*.json"))
    if fc is None or not parts:
        missing.append("OCT stage/curve refresh"); return
    new = {}
    for p in parts:
        j = json.loads(p.read_text())
        for L, Lv in j.items():
            for t, tv in Lv.items():
                for a, av in tv["per_arm"].items():
                    new.setdefault(L, {}).setdefault(a, {})[t] = av["g_over_base"]
    tg = ["impulsivity", "risk_taking"]

    def sel(d):
        o = [t for t in d if t not in tg]
        return float(np.mean([d[t] for t in tg]) / np.mean([d[t] for t in o]))
    md.append("## 6. OCT reproduction / stage / dose-matched / SFT-curve arms (impulsiveness)\n")
    md.append("B3-style activation selectivity = mean g/base over impulsivity+risk-taking ÷ mean over the other six "
              "(point; 10 personas). Dose = trait-vector displacement. legacy = vs archived legacy-mask base "
              "(as reported); cm = vs base_cmask. Released `impulsiveness` is legacy-mask and has no corrected "
              "counterpart here, so ratios to the released adapter (A4) are not refreshed.\n")
    md.append("| arm | sel L15 legacy → cm | sel L20 legacy → cm | dose L15 legacy → cm | dose L20 legacy → cm |")
    md.append("|---|---|---|---|---|")
    rows = {}
    for a in new.get("15", {}):
        r = {}
        for L in ("15", "20"):
            ol = {t: old[L][t]["per_arm"][a]["g_over_base"] for t in old.get(L, {}) if a in old[L][t]["per_arm"]} if old and L in old else {}
            r[f"sel_{L}"] = {"legacy": sel(ol) if len(ol) == 8 else None, "cm": sel(new[L][a]) if a in new.get(L, {}) else None}
            r[f"dose_{L}"] = {"legacy": (fl or {}).get(L, {}).get(a, {}).get("trait_vector_displacement"),
                              "cm": fc.get(L, {}).get(a, {}).get("trait_vector_displacement")}
        rows[a] = r
        md.append(f"| {a.replace('impulsiveness_', '')} | {f3(r['sel_15']['legacy'])} → {f3(r['sel_15']['cm'])} | "
                  f"{f3(r['sel_20']['legacy'])} → {f3(r['sel_20']['cm'])} | {f3(r['dose_15']['legacy'])} → {f3(r['dose_15']['cm'])} | "
                  f"{f3(r['dose_20']['legacy'])} → {f3(r['dose_20']['cm'])} |")
    out["oct_stage"] = rows
    d = {k: rows[k]["dose_15"] for k in ("impulsiveness_repro", "impulsiveness_regen", "impulsiveness_paraphrase") if k in rows}
    if len(d) == 3:
        md.append(f"\nDose ratios among corrected arms, L15: P0/repro {d['impulsiveness_regen']['legacy'] / d['impulsiveness_repro']['legacy']:.3f} "
                  f"→ {d['impulsiveness_regen']['cm'] / d['impulsiveness_repro']['cm']:.3f}; P1/repro "
                  f"{d['impulsiveness_paraphrase']['legacy'] / d['impulsiveness_repro']['legacy']:.3f} → "
                  f"{d['impulsiveness_paraphrase']['cm'] / d['impulsiveness_repro']['cm']:.3f}.\n")


def main():
    out, md, missing = {}, ["# Corrected-mask base (G1): historical vs synthetic-base vs actual\n",
                            "Generated by `scripts/audit/compare_corrected_base.py`. One training seed per arm; "
                            "intervals are question-bootstrap (evaluation) only.\n"], []
    for f in (geometry, doses, matched, xchar, fidelity, oct_stage):
        f(out, md, missing)
    missing = list(dict.fromkeys(missing)); out["missing"] = missing
    if missing:
        md.append("## Missing inputs\n\n" + "\n".join(f"- {m}" for m in missing) + "\n")
    (AU / "corrected_base_comparison.json").write_text(json.dumps(out, indent=1, default=str))
    (AU / "corrected_base_comparison.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
