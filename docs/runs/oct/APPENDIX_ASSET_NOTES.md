# OCT stage-localisation appendix assets — provenance and inclusion rules

One entry per proposed asset: the question it answers, its source data, the exact
inclusion/exclusion rules, whether interpolation is used, and whether it should be kept once
the dose-matched results are in.

**Location.** All assets are written to `workshop_iclr/{figures,data,tables}` via
`figstyle.save()` / `write_source_data()` -- the set the submission actually references.
An earlier draft put them under `iclr2026/`, which is stale and unused; those copies were
removed so there is one source of truth.

**Status 2026-09-08:** assets 1–3 are built from archived data. Assets 4–6 wait on the
dose-matched measurement (12 rungs, in progress). Nothing here edits the manuscript.

---

## Asset 1 — `figA_oct_stage_pipeline.pdf`

- **Question.** Which model states are being compared, and how is each built?
- **Source.** None — a schematic. No numbers.
- **Rules.** Solid arrows are data flow, dashed are weight initialisation. The corpus box
  sits between `M_D` and *both* SFT branches because that is the causal fact most at risk of
  being misread: **`M_S` starts from base weights but trains on a corpus the DPO model
  generated.** DPO is removed from the optimisation, never from the causal history.
- **Interpolation.** None.
- **Keep?** Yes regardless of results — the appendix is unreadable without it.

## Asset 2 — `tableA_oct_stage_raw.tex`

- **Question.** What did the naive decomposition appear to show?
- **Source.** `outputs/analysis/stage_comparison_seed1.csv`, itself from
  `scripts/stage_comparison.py`. Every cell is read from that CSV at build time; no number is
  transcribed, and the Spearman quoted in the caption is **recomputed from the same rows**, so
  caption and table cannot drift apart.
- **Rules.** `M_0+A_S` is daggered as an off-base diagnostic. `released` is included as the
  external reference. The caption states outright that this is not a dose-matched comparison.
- **Interpolation.** None.
- **Keep?** Yes, but as orientation only — never as the headline. The ordering is perfectly
  rank-confounded with dose (Spearman +1.000).

## Asset 3 — `figA_oct_dose_calibration.pdf`

- **Question.** Can nominal LoRA scale serve as the x-axis for a causal comparison?
- **Source.** `outputs/analysis/dose_stage_grid_analysis.json` — the Phase-1 ladder from
  `scripts/dose_calibrate.py`, 26 configs, layer 15, trait-vector displacement.
- **Rules.** Lines connect only each state's **monotone prefix**. Post-turnover points are
  drawn hollow and unconnected, and are excluded from every interpolation everywhere in this
  work. No smooth fit is drawn — a polynomial would imply a functional form nothing supports.
  Trained states (`s=1`) are ringed as anchors.
- **Interpolation.** None in the figure. The same monotone rule governs the rung selection
  that feeds assets 4–5.
- **Keep?** **Yes, independently of the matched-dose outcome.** It carries three facts on its
  own: equal nominal scale does not give equal dose (0.54–1.11 at `s=1`); scale→dose is
  nonlinear *within* every state (`dose/s` spans ~2×); and `M_D+S` and `M_S` turn over at
  `s=2`. It is what justifies the normalisation as necessary rather than gratuitous.

## Asset 4 — `figA_oct_matched_dose.pdf` — BUILT

- **Question.** At matched dose, do the stage differences remain?
- **Source.** `outputs/analysis/{caa_logits,common_shift,functional_dose}.json` via
  `scripts/appendix_oct/dm_common.py`.
- **Rules.** x is measured functional dose. Curves drawn heavy only inside each pair's
  overlapping measured support (shaded); measured points outside it drawn faint so they are
  visible without implying the comparison extends there. Trained `s=1` states ringed. No
  extrapolation.
- **Panel D endpoint: SELECTIVITY, not cosine.** The brief preferred the cosine; it is
  **degenerate for pair 2** because `M_F`'s trained state *is* the reference direction, so its
  cosine is pinned at 1.000 by construction and the curve would report the definition rather
  than the data. Selectivity separates cleanly over the same support. Decided on
  interpretability; the cosine is retained in the master CSV.
- **Keep?** Yes — this is the headline result.

## Asset 5 — `tableA_oct_matched_dose.tex` — BUILT

- Three anchors per pair, inside the overlap, each value flagged `measured` or `interp`.
  `dm_common.interp_at` refuses anything outside a state's measured range and raises on a
  non-monotone dose axis rather than silently interpolating.
- **Keep?** Yes.

## Asset 6 — `tableA_oct_peft_merge.tex` — BUILT (the difference survived)

- A table, not a figure: panels C/D already carry the visual, so a figure would repeat it.
  What was missing is the two-line algebra plus the matched-dose numbers.
- Wording enforced by construction: reports *B1 changes from X to Y at matched dose*, never a
  percentage of behaviour. The 61.9% figure elsewhere is a weight-space norm ratio and is
  labelled as such.
- **Keep?** Yes.

## Superseded planning notes

- **4, `figA_oct_matched_dose.pdf`** — 2×2: pair 1 (`M_D` vs `M_S`) B1 and selectivity;
  pair 2 (`M_D+0.25S` vs `M_F`) B1 and one representational endpoint. **Panel D endpoint will
  be chosen on interpretability, not on which looks more dramatic, and the choice recorded
  here.** Plotted over each pair's *actual* overlapping measured-dose support only.
- **5, `tableA_oct_matched_dose.tex`** — matched-dose anchors, each marked *measured at a
  rung* or *interpolated between two measured rungs*, with the bracketing rungs shown. No
  anchor that requires extrapolation. Two anchors rather than three if the `M_D`/`M_S`
  overlap will not support three — symmetry will not be manufactured.
- **6, `figA_oct_peft_merge`** — built **only if** a clear `M_F` vs `M_D+0.25S` difference
  survives dose matching. If it disappears, that is reported plainly in the table and this
  asset is not made.

### Standing caveat for pair 1

Matching `M_D` to `M_S` requires `M_D` near `s=2.0`, the top of its measured range. The
comparison is therefore between two artifacts *neither of which is a trained state*. This is
inherent — `M_D` is simply the weakest perturbation — and it bounds how strongly the pair-1
result can be phrased. Pair 2 is cleaner: its overlap (0.582–0.925) sits comfortably inside
both states' measured ranges.

### Wording rules carried into every caption

- Never "SFT installs the character" from the raw ordering.
- Never "the cross terms are X% of the behaviour" — behaviour is nonlinear. Say instead:
  *at matched functional dose, including the cross terms changes B1 from X to Y.*
- Never weight norm as dose; `‖dW‖_F` is a diagnostic column only.
- Question-bootstrap intervals are uncertainty over CAA questions, **not** over training seeds.
