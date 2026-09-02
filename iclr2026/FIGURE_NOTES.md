# Figure notes and provenance

These are deliberately provisional figures to make the workshop-paper skeleton concrete.

## Main-text figures

### `trait_selective_common_shift_L15`
**Current role:** first substantive result; shows the impulsiveness-specific risk-taking/impulsivity spike.

**Data source:** `docs/results/llama31_8b_extraction_and_geometry.md`, persona-common shift table in §3.2.

**Before submission:** regenerate from the JSON/cross-fitted analysis and add uncertainty. Consider visually emphasizing the two constitution-related traits without relying on post-hoc color choices.

### `dose_response_RDM_L15`
**Current role:** shows the dose law for relative persona geometry.

**Data source:** matched-dose/interpolated grid in §6.3 and §6.4.

**Before submission:** use measured rung points where possible; propagate x-axis/dose uncertainty jointly with outcome uncertainty; label clearly that the current grid is interpolated.

### `dose_response_dispersion_L15`
**Current role:** contrasts with the RDM result and shows curve crossing / arm-specific response shape.

**Data source:** common dose grid in §6.5.

**Before submission:** same joint-bootstrap issue as above. Consider pairing RDM and dispersion as panels in the final paper if the workshop format permits.

### `ctp_variance_decomposition_L15`
**Current role:** direct answer to the original context-specific question; CTP is only ~3.6%.

**Data source:** latest coding-agent analysis supplied in chat (T 0.372, TP 0.175, CT 0.170, mu 0.124, C 0.067, P 0.043, CTP 0.036, CP 0.013). This analysis was not visible in the committed results Markdown at scaffold-build time.

**Before submission:** replace immediately with the committed cross-fitted plot/table. Add the matched-DF trained subset and random-control band if the calibrated comparison is retained. Do not use the old per-cell '319/320 above zero' null count; the coding-agent analysis correctly flags that exact-zero null as near-vacuous for squared magnitudes.

## Appendix figures

### `common_shift_share_L15`
Data: L15 common-shift shares 0.673 / 0.684 / 0.723 / 0.766 for goodness / mathematical / impulsiveness / misalignment.

Useful as a compact decomposition sanity check; probably appendix if main paper is short.

### `matched_random_RDM_L15`
Data from §7.7, approximately dose-corrected to 1.0:
random_iid 0.886, random_spec 0.846, goodness 0.834, impulsiveness 0.798, random_perm 0.761, misalignment 0.732.

Useful for the claim that untrained update shape spans at least the trained-family spread in RDM preservation.

### `matched_random_dispersion_L15`
Data from §7.7:
random_iid 0.501, random_spec 0.491, goodness 0.486, impulsiveness 0.615, random_perm 0.557, misalignment 0.313.

Useful for the claim that broad contraction is generic at matched functional dose, while trained tails remain.

## Existing older repo plots worth considering only as motivation/appendix

- `confound_diagnosis_L15.png`: useful for explaining why raw cosine-to-null was misleading.
- `ordering_preservation_L15.png`: useful if the residual-to-null scalar effect remains in the final story.
- `arm_comparison_corrected_L15.png`: historically useful, but tied to an earlier stage of the analysis.
- `fig1_persona_fanout_L15_sorted.png`: motivation only; do not let it visually re-establish the retracted naive clustering interpretation.
