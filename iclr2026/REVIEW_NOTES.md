# Draft review — status after the 2026-09-09 editing pass

Baseline: `main.tex` against `workshop_iclr/` at `origin/main` = `5036d2c`.
All bullets left as bullets; no prose conversion. Narrative rewriting is yours.

Build: **19 pages, 0 errors, 0 overfull boxes, 0 unresolved references.**
(Was 13 pages before this pass — see "page budget" under open questions.)

> **How to read this.** ~~Struck through~~ = done and in `main.tex`. Everything in §1 is
> struck. Everything in §2, §3 and §4 is still open and needs either you, the pod agent,
> or a decision. §5 is a record, not a task list.

---

## 1. Applied — ~~all of this is done~~

### Structural additions

- ~~**New Results subsection**, "A signed effect survives behaviourally, though not
  geometrically" (§3.6). Fig. 5 previously had a float and caption but was never
  referenced anywhere in the body. Covers: compression toward indifference, why the
  naive shift is invalid, the corrected contrasts, and the within-trained-family
  comparison that is the one place content and procedure are separated.~~
- ~~**Appendix figures added**, all from existing PDFs in `workshop_iclr/figures/` — no
  replotting:
  - `figA1_dose_calibration` → `\label{fig:dose-calibration}`, now cited twice from the
    body (the s=16–19 matching, and the coherence-cliff caveat).
  - `figA2_diagnostics` → `\label{fig:diagnostics}`, cited from §Global reshaping.
  - `figA3_layer20` → `\label{fig:layer20}`, the L20 replication.~~
- ~~**OCT stage appendix rewritten** from a two-bullet legend into four bullet blocks plus
  the matched-dose figure and all three tables. Your `M_D`/`M_S`/`M_F` legend bullet was
  kept verbatim, folded into the opening block. Adapted from
  `docs/runs/oct/oct_appendix_snippet.tex` but converted from prose to bullets.
  `\label{sec:oct-stage}` added and cross-referenced from Discussion.~~
- ~~**Tables** copied to `iclr2026/tables/` and `\input`. New `make sync-tables` target.~~
- ~~`\label`s added to the previously unlabelled `figA4_rotation_control` and
  `fig0_schematic` floats.~~

### Claims corrected

- ~~**RDM / misalignment (§3.3).** Was "misalignment remains displaced below it", which
  invited "misalignment is the outlier". Now gives all four trained arms with numbers,
  and adds the bullet that kills the reading: `random_iid` preserves geometry *better*
  than every trained arm (0.886) and `random_perm` *worse* than misalignment (0.761).
  Fig. 3 caption updated to match. Added the §7.7 estimator-departure note.~~
- ~~**Global reshaping (§3.2).** Added the control this section was missing: the untrained
  arms sit at the top of the linear-map column (0.488–0.561) at or above every trained
  arm (0.388–0.541). The section previously applied the paper's own standard to every
  signature except this one.~~
- ~~**`DELETE/CLARIFY` on the CTP caveat resolved as keep**, and expanded into three
  bullets: what Panel B licenses, the coherence-cliff numbers from figA1, and why the
  sham LoRA is the control that would settle it.~~
- ~~**Seed replication (§3.1).** Was reporting only seed 987654 (+1.95, 1.640×). Now
  reports both runs, and states plainly that both land *below* the released adapter on
  selectivity (1.56× and 1.64× against 1.72×) — so the "1.7–1.9×" quoted a bullet
  earlier is a property of the released artifact. Added the preregistered
  not-resolvable-into-seed-vs-rig caveat with its numbers.~~
- ~~**`registered` vs `fixed before the data`** now stated explicitly: the prereg names
  `impulsivity` and the triple interaction, not `risk_taking`. Includes the point that
  the registered version carries the claim unaided (+2.18 / +2.22 against ≤ −0.52).~~
- ~~**Fig. 4 caption** rewritten with the actual cosine ranges instead of the garbled
  comparative.~~

### Scope and framing

- ~~**Abstract**: seed limitation qualified (impulsiveness has two); two bullets added for
  the behavioural result and the stage decomposition, which the abstract did not mention
  at all.~~
- ~~**Contribution bullet 6** reworded to the matched-dose form.~~
- ~~**Discussion**: the empty "contributes to existing structure" paragraph filled;
  two new paragraphs (the signed result, and the pipeline localisation); C×T/C×P ≈ 13
  added to the first paragraph.~~
- ~~**Limitations** rewritten — three bullets were already done. Now leads with the sham
  LoRA as the load-bearing gap and names free-form behaviour as the real behavioural gap.~~
- ~~**Reproducibility Statement** filled in as bullets~~, with a residual TODO for adapter
  identifiers and environment pins (**still open**).

### Typos

~~`chossing`, `Panels A/B shows`, `Panel B show`, `traits investigated are traits are`,
the missing full stop after "coordinate-permuted trained updates", the broken
"could converge directionally" sentence, `pipeline).After`, and both `$10x10$` → `$10\times10$`.~~

---

## 2. NOT DONE — needs new analysis or new plots

Nothing here was attempted. Flagged for you and the pod agent.

- **The sham-trained LoRA.** Named in Limitations as the load-bearing gap. It is the
  control for both the Fig. 4 content-vs-procedure confound and the Fig. 2B incoherence
  ambiguity. `docs/NEXT_POD.md` says do not run sham arms, so this is a scheduling call,
  not a technical one.
- **Free-form behavioural evaluation.** Nothing in the figure set generates a completion.
  Stated as a limitation rather than papered over.
- **Joint dose-uncertainty propagation** into the matched-dose comparisons. Currently the
  matched-dose anchors are point estimates on a measured dose axis; the intervals in the
  source data are question-bootstrap only.
- **`M_S` vs `M_F`.** No overlapping dose support was constructed, so it is untested. The
  appendix says so explicitly rather than letting the raw ordering imply it.
- **Layer 20 for the stage work.** Everything in §B is layer 15.
- **`figA3_layer20` has no `random_spec` arm**, so the L20 panel shows two untrained arms
  where L15 shows three. Not a problem for any claim made, but a reviewer may ask.

## 3. NOT DONE — needs checking with the pod agent

- **`workshop_iclr/README.md` may be stale on Fig. 5 robustness.** It states "the linear
  fit's mean residual stays within ±0.13 log-odds across every base decile for all four
  trained arms". The regenerated `outputs/analysis/caa_logits_robustness.txt` does not
  show that: `impulsiveness`/`impulsivity` has decile residuals of −0.31, −0.28, +0.20,
  mean|r| 1.35. `caa_logits_robustness.json` was rewritten in the commits just pulled
  (12,370 lines changed), so the README paragraph probably predates it. **I did not put
  the ±0.13 claim in the paper.** The support-near-indifference figure (9–21% of items
  within |log-odds| < 1) *does* check out and is in.
- **Upstream typo**: `workshop_iclr/tables/tableA_oct_stage_raw.tex` line 18 contains
  `\ref{{fig:oct-matched-dose}}` — an f-string double-brace leak. It is a hard LaTeX
  error ("too many }'s"). ~~Fixed in the `iclr2026/` copy and in `make sync-tables`~~ —
  but **still open at source**, in `scripts/appendix_oct/tableA_oct_stage_raw.py`.
- **`iclr2026/tables/tableA_oct_stage_raw.tex` diverges from source** by design: it is
  `\footnotesize` with tighter `\tabcolsep` and four shortened cell descriptions, to fit
  the ICLR text width. `make sync-tables` reapplies this. If the table is regenerated
  upstream, that patch has to survive.

## 4. NOT DONE — open questions for you

- **Page budget.** 19 pages, of which the appendix is roughly half. No workshop CFP is
  live yet, so nothing has been cut, but if the main body has a hard limit the
  candidates to move or drop are `fig0_schematic` (your own caption says it is probably
  stale), `figA5b_signed_validity_2panel` (unused — it is a two-panel variant of figA5,
  pick one), and possibly `tableA_oct_stage_raw`, which exists mainly to be argued with.
- **`fig0_schematic`** still carries your `\textbf{TODO} IF SPACE PERMITS ... probably
  stale` caption. Left untouched.
- **`\textbf{TODO}` markers** at the top of the Abstract and Introduction are yours and
  are left in place.
- **Anonymity.** The provenance links resolve to
  `https://github.com/ukc10014/steering_across_personas`, which names you. Repoint
  `\provbase` (preamble, just after `\graphicspath`) at an anonymised mirror before any
  double-blind submission. Same issue for `\author{Anonymous authors}` over a public fork.
- **Bibliography** is still `plainnat`; swap for `iclr2027_conference.bst` when the
  official bundle lands. `marks2025psm` and `minder2026spp` carry initials only, and
  `minder2026spp` elides authors with `and others`.

---

## 5. Numbers verified against the plotted CSVs — record, not a task list

Recorded so they are not re-checked. All correct as printed.

- persona-common share 67–77% at L15 → per-arm means 0.673 / 0.684 / 0.723 / 0.766
- selectivity 1.722 at L15, 1.87 at L20 (released adapter)
- variance shares T .372 / TP .175 / CT .170 / µ .124 / C .067 / P .043 / CTP .036 / CP .013 — exact
- 71.4% / 28.6% split (C+CT+CP+CTP = .2862)
- untrained CTP residual higher at both layers (.072 vs .029–.035 at L15; .107 vs .035–.045 at L20)
- global linear map trained 0.388–0.541, orthogonal 0.282–0.323; untrained linear 0.488–0.561
- untrained arms matched at s = 16–19; CAA displacement 0.545 / 0.575 / 0.598 vs 0.560 for goodness at s=1
- figA1 KL: goodness s=1 0.606; random_iid s=1 0.0012, s=16 0.176, s=24 0.529, s=32 1.81
- fig4 cosines: trained×trained 0.465–0.832, trained×untrained 0.013–0.292, non-overlapping CIs
- fig5 panel C: impulsiveness +3.160 / +2.453, misalignment +2.515 / +2.369; panel D contrasts
  +2.077, +2.491, −0.361, −0.391, and untrained −0.117 / −0.138 with intervals covering zero
- fig3 panel C dose-corrected: 0.878 / 0.834 / 0.826 / 0.768 trained, 0.886 / 0.846 / 0.761 untrained
- figA3 L20 panel C: 0.832 / 0.792 / 0.771 / 0.631 trained, 0.849 / 0.662 untrained
- stage raw B1: M_D +0.132, M_D+0.25S +0.499, M_F +1.923, M_D+S +2.190, M_S +3.613; Spearman(dose, B1) = +1.000
- matched dose: M_S vs M_D +1.960 / +2.485 / +2.739 at doses 0.586 / 0.702 / 0.817;
  M_F vs M_D+0.25S +0.414 / +0.822 / +1.032 at doses 0.578 / 0.710 / 0.842
