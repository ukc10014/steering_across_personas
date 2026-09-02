# ICLR workshop figure set

Publication figures for the constitutional-character-training extension on
Llama-3.1-8B-Instruct. Everything here is built **CPU-only from cached analysis outputs**;
no script in this directory runs a forward pass.

```bash
source /workspace/bootstrap.sh
python workshop_iclr/scripts/build_all.py            # all figures
python workshop_iclr/scripts/fig3_dose_and_control.py  # or one at a time
python workshop_iclr/scripts/validate_palette.py     # must print PASS
python workshop_iclr/scripts/check_signed_validity.py # the figure-4 gate
```

Output: `figures/*.pdf` (for the paper) and `figures/*.png` (previews), plus
`data/*.csv` — the exact numbers each figure plots, so every value in the paper is
auditable without rerunning anything.

This supersedes `iclr2026/figures/`, whose contents were **hand-reconstructed from
numbers typed into a chat transcript** rather than read from the analysis outputs (see
the note at the top of `iclr2026/FIGURE_NOTES.md`). Do not mix the two sets.

## The organising question

> What in the representational change under OCT is **generic to being perturbed**, what is
> **structured by trait**, and what — if anything — depends on **constitutional content**?

The figure set is built to separate those three, not to make the third look large. Where a
result turns out to be generic, the figure that shows it says so.

## Main figures — five

| file | claim | source of every number |
|---|---|---|
| `fig1_decomposition` | most of an adapter's effect is one persona-common translation; `impulsiveness` moves `risk-taking` and `impulsivity` 1.72× as far as the other six (1.87× at L20), and the two normatively flat arms sit at 0.97 and 0.95. This panel is figure 2's C×T term resolved into traits. | `outputs/analysis/common_shift.json` |
| `fig2_ctp` | 71.4% of the change carries no constitution index at all; of the 28.6% that does, **C×T / C×P = 13** — constitutional differences are far more trait-dependent than persona-dependent — and the preregistered triple interaction is 3.6% | `outputs/analysis/three_way_interaction.json` |
| `fig3_dose_and_control` | RDM preservation is nearly a function of dose alone; dispersion is not; and at matched functional dose untrained perturbations land inside the trained range and span *more* of it on RDM preservation | `geometry_L15.json`, `functional_dose_with_random.json` |
| `fig4_shared_direction` | the one statistic on which untrained perturbations do NOT reproduce the trained arms: the four constitutions' common shifts are mutually aligned (cos 0.46–0.84) and near-orthogonal to every untrained arm (0.01–0.30) | `outputs/analysis/common_shift_cross_family.json` |
| `fig5_behavioral_preference` | **the signed result, behavioural rather than geometric.** Reading the CAA answer letters' logits: every arm compresses the model's preferences toward indifference (retention k = 0.02–0.29 trained, 0.68–0.81 untrained), which makes the naive shift a mirror of the base model's own preferences (r ≤ −0.95 wherever k < 0.3). Correcting for it, `impulsiveness` (+2.08) and `misalignment` (+2.49) push specifically toward impulsivity and risk-taking; `goodness` (−0.39), `mathematical` (−0.36) and both untrained arms (intervals covering zero) do not. Same ordering under both prompt forms. | `outputs/analysis/caa_logits.json` |

`fig0_schematic` is a pipeline legend, not a result; use it only if the workshop format
leaves room.

**The signed figure exists, but it is not the geometric one.** The fourth slot was
reserved for an Open-Character-Training-style signed trait figure built from the
representations. That metric was built (`scripts/signed_trait_shift.py`) and failed its own
pre-registered validity test (`check_signed_validity.py`, and figure A5): the untrained arm
passes 8/8 and the trained arms 4/24, because the sign is dominated by generic contraction
along every trait axis. (Sign does agree between L15 and L20 in 89% of cells — recorded in
figure A5 — but it does not discriminate, since contraction is layer-consistent too.)
`fig4_shared_direction` took that slot.

`fig5_behavioral_preference` then answers the same question from the other side, by asking
the model instead of its activations. It runs into the exact same trap first — the naive
preference shift is a near-perfect mirror of where the base model already stood — and the
correction is what makes it readable. Compression is the behavioural name for the
contraction figure A5 tripped over; the two panels are the same lesson in two measurement
regimes, which is why A5 stays in the appendix rather than being replaced.

## Appendix

| file | what it settles |
|---|---|
| `figA1_dose_calibration` | weight norm, output KL and CAA displacement are three different dose axes; a norm-matched random LoRA is inert (~700× less KL). This is why the rungs were sited by measurement. |
| `figA2_diagnostics` | per-cell C×T×P against a calibrated reference (not against zero); orthogonal vs general linear map, on which the untrained arms are at the top |
| `figA3_layer20` | figure 3 replicated at layer 20 |
| `figA4_rotation_control` | **§3.3's rotation-with-dose is reproduced by `random_perm`** at the same magnitude once dose gap is controlled for, so it is not on its own evidence of anything semantic |
| `figA5_signed_validity` | **why there is no OCT-style signed figure.** `random_perm` is the best-behaved signed metric in the data (8/8 valid against 4/24 for the constitutions); at s=1 `impulsiveness` scores −0.272 on `impulsivity`, which an unvalidated gain/loss chart would have rendered as "made the model less impulsive" |

## What the figure set does *not* claim

- **Not "training suppresses context-specific interaction."** Figure 2B is a control. It
  licenses "a nonzero fine-grained interaction is not evidence of semantics, because an
  untrained perturbation produces more of it". It does not license the causal reading: the
  random arms sit within a factor of two of the measured coherence cliff (figure A1), and
  incoherence would present exactly as cell-specific idiosyncrasy. Separating those needs
  a sham-trained LoRA, which does not exist yet.
- **Not "the common shift rotates because of constitution content."** Figure A4 kills that.
- **Not "the impulsiveness constitution made the model more impulsive" from the geometry.**
  No validated signed metric exists in the representations; figure 1B is a magnitude, and
  is reported as semantic SELECTIVITY, never as direction or valence. Figure A5 is the test
  that forced this. Figure 5 does license the signed claim, but only as a REVEALED
  PREFERENCE over forced A/B choices, and only after correcting for compression.
- **Not "constitutional content lives in a shared subspace."** Figure 4 shows the trained
  arms share a direction untrained ones miss, but all four come from the same OCT pipeline,
  rank, initialisation and corpus shape, so "content" and "this training procedure" are not
  separated. The sham-trained LoRA is the control that would.
- **No free-form behaviour.** Figures 1–4 and the appendix are representational
  displacements at the CAA answer token; figure 5 is a forced-choice preference read from
  two answer-letter logits. Neither generates a completion. Nothing here shows what these
  arms would actually write, and the A/B result should not be extrapolated to open-ended
  behaviour without the generation study.
- **Figure 5 is not subject to figure 4's content-vs-procedure confound**, and this is the
  one place the two differ. Figure 4 compares trained arms against untrained ones, so
  "content" and "this OCT pipeline" move together and a sham-trained LoRA is needed to
  separate them. Figure 5's contrast is *within* the trained family — `goodness`,
  `mathematical` and `impulsiveness` share pipeline, rank, initialisation and corpus shape,
  and sit at almost identical compression (k = 0.25, 0.28, 0.29) — yet score −0.39, −0.36
  and +2.08. Procedure is held constant there; only the constitution differs.

## Conventions held across the whole set

- **Arm names, order, colour and marker come from `figstyle.py` and nothing else.**
  Trained arms are solid with filled markers; untrained controls are a recessive grey
  family, dashed, always direct-labelled.
- **The palette is validated, not asserted.** `validate_palette.py` runs the Machado CVD
  transforms over all pairs; the trained hues clear the ≥8 CVD and ≥15 normal-vision ΔE
  floors on white. `misalignment` and `random-iid` fall below 3:1 contrast against the
  page, so both are direct-labelled wherever they appear (the relief rule).
- **Vector PDF with Type-42 fonts**, 5.5in full width, 7–7.5pt labels.
- Intervals are 95% question-bootstrap, 200 replicates, draws shared across arms so every
  arm-to-arm comparison is paired. They cover uncertainty from the sampled CAA questions
  only, and condition on these 10 personas and 8 traits. `figA4` and the validity check are
  point estimates only, and say so.
- **Corrected and uncorrected values are never mixed.** Where a dose correction is applied
  it is applied to *every* arm in the panel, and the measured value is drawn beside the
  corrected one.

## Departure from `docs/results/llama31_8b_extraction_and_geometry.md` §7.7

That section's matched-dose table corrects the random arms to dose 1.0 but quotes the
trained arms at their measured doses (1.07–1.08). Figure 3C/D corrects **every** arm, which
changes the printed ordering: `misalignment` rises from 0.732 to 0.768 and ties
`random_perm` at ~0.76 rather than sitting lowest. The section's actual claim — untrained
arms span at least as much as the trained family — is unaffected and slightly strengthened
(0.125 against 0.110). `dose_panels.py`'s docstring records this.
