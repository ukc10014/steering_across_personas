# Paper figures from the G1 corrected-mask results

Final exports for the manuscript's trajectory, matched-dose, cross-character and SFT-curve figures,
built **only by plotting** completed outputs (no extraction, training, judging or bootstrapping).
Results and numbers: [`docs/AUDIT_G1_HANDOFF.md`](../docs/AUDIT_G1_HANDOFF.md),
[`outputs/analysis/audit/corrected_base_comparison.md`](../outputs/analysis/audit/corrected_base_comparison.md).

```bash
bash paper_g1/build.sh          # rebuilds every asset below, then verifies values and PDFs ("BUILD OK")
```

- `build.sh`: the exact plotting commands.
- `verify_assets.py`: checks every plotted value (and the withheld cells) against the source JSON. The last
  run made 3,474 checks with 0 failures.
- `check_pdf.py`: confirms each PDF is vector (no raster images) with embedded TrueType fonts (no Type 3),
  and renders it at print width for inspection. It needs PyMuPDF, which is installed outside the project
  environment in `/workspace/pylibs-pdfcheck`.

All PDFs are genuine Matplotlib vector output, saved from the same figure object as the PNG
preview (300 dpi). Fonts are DejaVu Sans, embedded as TrueType (`pdf.fonttype 42`). Layouts are sized
for the 5.5 in ICLR text width. Every figure was inspected rendered at that width.

**Code revision** of the plotting code: `933818e` (branch `audit/g1-figures`), recorded in each
`*_provenance.json`. **Baseline**: the corrected-mask base `base_cmask` (G1, 2026-10-22; snapshot
`0e9e39f` of Llama-3.1-8B-Instruct). Logit (behavioural) quantities do not depend on the base.

## Assets

| figure | files in `figures/` | layer | personas | estimand | intervals |
|---|---|---|---|---|---|
| **Trajectory, main text (A/B)** | `fig_trajectory_AB_cmbase.{pdf,png}`, `_data.csv`, `_provenance.json` | L15 | 10 occupational (null excluded) | A: forced-choice offset at base indifference per trait + target contrast; B: q, Q, g̃ | 95% question bootstrap: behaviour 2,000 draws, geometry 200 draws × 8 splits |
| **Trajectory, appendix (A–D)** | `fig_trajectory_cmbase.{pdf,png}`, `_data.csv`, `_provenance.json` | L15 | 10 occupational | + C: S, S^S (8-trait mean of per-trait ratios); D: g, cos_T | as above |
| **Matched-dose excess, overview** | `fig_matched_dose_cmbase.{pdf,png}`, `matched_dose_points_cmbase.csv`, `fig_matched_dose_cmbase_provenance.json` | L15, L20 | 10 occupational (geometry and contrast) | trajectory − DPO ladder at matched dose: S per-trait mean, S energy-pooled, g, B2/LOV contrast | paired question bootstrap, B = 1,000, shared draws, dose matched within each draw; pointwise 95% |
| Matched-dose companions (one per constitution) | `fig_matched_dose_{impulsiveness,loving,goodness,sycophancy}_cmbase.{pdf,png}` | L15, L20 | as above | as above | as above |
| **Cross-character acquisition/retention** | `acquisition_retention_steps.{pdf,png}`, `_data.csv` | — (logits) | **11 (10 occupational + null)** | fitted offset at base indifference vs unadapted base | 95% question bootstrap, 2,000 paired draws |
| SFT curve, activation inputs on corrected base | `figA_oct_sft_curve_cmbase.{pdf,png}`, `_data.csv` | L15 | 10 occupational | B1 (logits, unchanged); dose; B3-style selectivity | none drawn (as published) |
| SFT-curve table | `tableA_oct_sft_curve_cmbase.{tex,csv}` | L15 | 10 occupational | as above | — |

Inputs (exact paths are in each provenance file):
- **Trajectory figures.** `outputs/analysis/common_trajectory/v2/geometry_<c>_cmbase.json` plus
  `v2/geometry_raw_<c>_L15_cmbase.npz` (the g̃ bootstrap), `v2/behaviour_<c>_10personas.json`,
  and `functional_dose_<c>_cmbase.json`.
- **Matched-dose figures.** `outputs/analysis/audit/joint_dose/joint_dose_<c>_cmbase.json`.
- **Cross-character figure.** `outputs/analysis/xchar_sft/signed.json`.
- **SFT curve and table.** `outputs/analysis/caa_logits.json` (B1), and
  `outputs/analysis/audit/oct_stage_cmbase/{functional_dose_oct_cmbase,common_shift_oct_cmbase}.json`.
  The second file is merged from the four chunk files by `scripts/audit/merge_oct_common_shift.py`, keeping
  per-arm fields only. The reference rungs come from `oct_stage_dose_master.csv`, with their dose and
  selectivity re-read per arm from the corrected-base files.

### Checks made (values, not looks)

- **q at 0.48 epoch**, on 10 personas: 0.802 [0.794, 0.811] impulsiveness, 0.941 [0.894, 0.988] loving,
  0.839 [0.800, 0.880] goodness and 0.715 [0.696, 0.733] sycophancy. These round to the prose values
  0.80 / 0.94 / 0.84 / 0.71, so there's no discrepancy.
- **q is not clipped.** Loving's q is 1.02 at 0.76 epoch and impulsiveness' q goes above 1. g̃ for loving
  reaches 1.25 at 1 epoch, which is overshoot rather than an error, because the 3-epoch g is lower than the
  1-epoch g. It also dips to −0.03 at 0.24 epoch, again unclipped.
- **10-persona endpoint behaviour excess** (L15 dose, 3 epochs): B2 +1.369 [1.308, 1.436] and LOV
  −0.124 [−0.188, −0.054]. The 11-persona values (+1.46 and −0.01) are not used in these figures.
- **Withheld, so no value is plotted.**
  - Sycophancy L15 endpoint: every series is marked "unbr.", because the point's dose lies outside its ladder.
  - Goodness L20 at 0.24 epoch: marked "n/e", because only 47% of draws are defined.
  - Sycophancy L20 per-trait-mean S: "n/e" at every state, because only 95.1% of draws are defined.

  The energy-pooled S is not substituted for any of these. Its own panel shows its own estimability.
- **Undefined trajectory quantities appear as gaps.** S^S and Q are undefined at the DPO-only state and are
  not drawn there.
- **Intervals that do not contain their point.** One matched-dose interval doesn't contain its point
  (impulsiveness L15 g at 0.12 epoch; the gap is in the 4th decimal). Points and intervals are drawn
  separately and are not forced together.

### Which curves carry uncertainty bands

- **Trajectory A.** Only the bold target contrast (B2, LOV) has a band. The eight per-trait lines are point
  estimates.
- **Trajectory B–D.** Every curve has a band: q, Q, g̃, S, S^S, g and cos_T. Some bands are narrower than
  the line (q for impulsiveness, cos_T near its endpoint).
- **Matched dose.** Every plotted point has its paired interval.
- **Cross-character.** Every point has its interval; most are narrower than the marker.
- **SFT curve.** No intervals, as in the published figure.

## Suggested captions

**Trajectory, main text (`fig_trajectory_AB_cmbase`).** Acquisition along the native DPO→SFT
trajectory for four constitutions (columns; one training seed each).

- **A.** Forced-choice offsets at base-model indifference, relative to the unadapted model, for the eight
  traits (thin lines, point estimates), and the prespecified target contrast (bold: B2 for impulsiveness,
  LOV for loving; 95% band).
- **B.** Fraction of the final SFT-induced change reached at each checkpoint. Three curves:
  - q, the behaviour-profile projection;
  - Q, the projection of the persona-common activation shift onto its final SFT direction;
  - g̃, the normalised increase in persona-common shift magnitude since the DPO-only state.

  Values above 1 are overshoot.

Epoch 0 is the completed DPO-only state; the axis is linear to 1 epoch and logarithmic from 1 to 3.
Behaviour is over the ten occupational personas. Activation geometry is at layer 15 and relative to the
corrected-mask base. Bands are 95% question-bootstrap intervals and reflect evaluation uncertainty only.

**Trajectory, appendix (`fig_trajectory_cmbase`).** As in the main-text figure, one row per constitution,
plus two panels:

- **C.** Persona-common squared share S of the total shift (relative to the base), and S^S of the SFT
  increment (relative to the DPO-only state).
- **D.** Relative common-shift magnitude g = ‖dG‖/‖V_base‖, and cos_T, the cosine of the common shift to
  its final direction.

Bands are shown on the bold contrast in A and on every curve in B–D. Undefined values (S^S and Q at the
DPO-only state) are gaps.

**Matched dose (`fig_matched_dose_cmbase`; per-constitution companions).** SFT trajectory minus the same
constitution's DPO-only adapter scaled to the same measured functional dose. Columns:

- persona-common squared share S, as the per-trait mean, at L15 and L20;
- the energy-pooled S, which is a different estimand that weights traits by their energy;
- the common magnitude g at L15;
- the target-contrast excess at the L15 dose, for impulsiveness (B2) and loving (LOV), on the ten
  occupational personas.

Intervals are paired 95% question-bootstrap intervals: 1,000 draws shared by every arm, with dose matching
redone within each draw. They are pointwise and reflect evaluation uncertainty only, from one training seed.
Withheld values (by the 97.5% rule) are tagged, not plotted:

- "unbr." means the checkpoint's own dose lies outside its ladder;
- "n/e" means fewer than 97.5% of draws are defined.

Goodness and sycophancy have no prespecified target contrast.

**Cross-character (`acquisition_retention_steps`).** Fitted forced-choice offset at base-model
indifference, relative to the unadapted model, over 11 personas (10 occupational + null). Shown through
one token-matched additional SFT pass from the released loving model (filled circles, solid lines) or from
the unadapted model (open squares, dashed lines). The pass is on the impulsiveness corpus (orange) or the
loving corpus (dark blue).

- **Steps.** The x-axis is the percentage of optimizer steps completed: 82/164/328 steps for the
  impulsiveness corpus and 92/184/367 for the loving corpus, which is 25.2–25.3% / 50.2% / 100% of
  supervised tokens.
- **Intervals.** 95% question-bootstrap intervals (2,000 paired draws; one training seed). Most are
  narrower than the markers. Points are offset horizontally for legibility.

**SFT curve (`figA_oct_sft_curve_cmbase`).** As the published `figA_oct_sft_curve`, with functional dose
and activation selectivity re-referenced to the corrected-mask base. B1 and the 82% statistic come from
logits and are unchanged.

## Manuscript replacement mapping (`arxiv/main.tex`)

| line | current file | replace with | note |
|---|---|---|---|
| 224 (`fig:ext-trajectory`, main text) | `ext_traj_full.png` | `paper_g1/figures/fig_trajectory_AB_cmbase.pdf` | caption above; remove `\figplaceholder` |
| new appendix figure (App. ext-trajectory) | — | `paper_g1/figures/fig_trajectory_cmbase.pdf` | full A–D |
| 258 (`fig:ext-xchar`) | `ext_xchar_acq_retention.png` | `paper_g1/figures/acquisition_retention_steps.pdf` | same data; new axis and encoding |
| 1018 (`fig:ext-dose-imp`) | `ext_traj_dose_impulsiveness.png` | `paper_g1/figures/fig_matched_dose_impulsiveness_cmbase.pdf` | paired CIs replace "no paired CI" |
| 1025 (`fig:ext-dose-lov`) | `ext_traj_dose_loving.png` | `paper_g1/figures/fig_matched_dose_loving_cmbase.pdf` | LOV excess now on 10 personas: −0.124 at 3 ep, not −0.01 |
| new (optional) | — | `fig_matched_dose_cmbase.pdf` (all four), `fig_matched_dose_{goodness,sycophancy}_cmbase.pdf` | the manuscript's TODO l.119 asks for goodness/sycophancy plots |
| 1004 (`fig:oct-sft-curve`) | `figA_oct_sft_curve.pdf` | `paper_g1/figures/figA_oct_sft_curve_cmbase.pdf` | optional: changes are below plotting resolution; "5–8×" (additive) becomes 5.1–8.1× |
| 1010 (`\input{tables/tableA_oct_sft_curve.tex}`) | `tables/tableA_oct_sft_curve.tex` | `paper_g1/figures/tableA_oct_sft_curve_cmbase.tex` | dose −0.005 to +0.007, selectivity −0.013 to +0.026 (largest at the DPO anchor), B1 identical |

For the `\prov{figure}{script}{data}` rows, use:
- `fig_trajectory_*`: `{paper_g1/figures/<name>.pdf}{scripts/common_trajectory/figures.py}{paper_g1/figures/<name>_data.csv}`
- matched dose: `{…}{scripts/audit/fig_matched_dose.py}{paper_g1/figures/matched_dose_points_cmbase.csv}`
- cross-character: `{…}{scripts/xchar_analysis.py}{paper_g1/figures/acquisition_retention_steps_data.csv}`
- SFT curve: `{…}{scripts/appendix_oct/figA_oct_sft_curve.py}{paper_g1/figures/figA_oct_sft_curve_cmbase_data.csv}`

The arXiv build reads `arxiv/figures/`, so copy the chosen PDFs there under the manuscript's names (or
update the paths). This export leaves `arxiv/` unedited, by design.

**Unchanged**, by design (legacy-mask internally consistent analyses, or logits only):
`fig1_decomposition`, `fig6_semantic_profile`, `fig2_ctp`, `figA4_rotation_control`,
`fig3_dose_and_control`, `figA3_layer20`, `fig4_shared_direction`, `figA5_signed_validity`,
`fig5_behavioral_preference`, `figA_paraphrase_trait_profile` (logits), `ext_behav_pv_heatmap`,
and `tables/tableA_paraphrase_stage.tex` (B1 only).

## Blocked (not plotting tasks)

- **`tables/tableA_paraphrase_headline.tex`** (B3 and A4 columns). Both are defined against the
  **released** impulsiveness adapter, which has no corrected-mask extraction yet. Pending task: the G1
  remainder (`run_g1_extract.sh impulsiveness_rel_cmask`), then re-score B3 and A4 on one protocol.
  Corrected-base B3 values for the reproduction, P0 and P1 rows exist (`corrected_base_comparison.md` §6),
  but substituting them would put two protocols in one table.
- **`figA_oct_matched_dose`, `tableA_oct_matched_dose`, `figA_oct_summary_*` and `figA_oct_dose_calibration`**
  (not in the current manuscript). Their dose anchors interpolate through 25 "phase-1 probe" doses
  (`dose_stage_grid_analysis.json`) measured against the legacy base. Pending task: recompute those probe doses
  against `base_cmask` (the probe states would need their qcaches located or re-extracted), then rebuild
  `oct_stage_dose_master.csv`.
- **Fidelity A4/B4 for impulsiveness, goodness and sycophancy.** These need the released corrected-mask arms (G1 remainder).
- **Functional-dose-axis trajectory companion.** Not exported: dose is not monotone in epoch, so its curves
  double back and shaded bands drawn over unsorted dose aren't meaningful. The paired matched-dose figure
  covers dose comparisons. It can still be generated without `--no-dose-companion`.
