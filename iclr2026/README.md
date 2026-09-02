# ICLR workshop paper skeleton: constitutional training x persona/trait geometry

This folder is a **working LaTeX scaffold**, not a finished paper. It is deliberately written as section-level bullet points plus provisional figures so the scientific narrative can still move.

## ICLR format

- `main.tex` uses `iclr2027_conference.sty`.
- Official ICLR 2027 author guidance points to the style bundle at:
  `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`
- ICLR has not published one universal 2027 workshop paper length/template; individual workshops normally set their own limits. This scaffold therefore uses the conference style as a neutral starting point.
- The official ZIP could not be fetched directly into this runtime. The bundled `.sty` is a public mirror of the 2027 ICLR style, cross-checked against other 2027 mirrors. **Before submission, overwrite it with the copy from the official ICLR ZIP** and add the official `.bst` if the workshop asks for it.
- The current source stays anonymous. Uncomment `\iclrfinalcopy` only when appropriate for the target venue.
- ICLR 2027 requires an AI-use statement for conference submissions; a placeholder section is already included. Check the target workshop's rule separately.

## Compile

On Overleaf: upload the whole folder and set `main.tex` as the main document.

Locally:

```bash
latexmk -pdf main.tex
```

`latexmk` drives BibTeX automatically, so this is the only command needed.
`make pdf` does the same thing.

**Do not build with a bare `pdflatex main.tex`.** BibTeX needs four passes
(pdflatex to write `.aux`, bibtex to write `.bbl`, then two more pdflatex passes).
A single pass silently produces a PDF with every citation rendered as `(author?)`
and the reference list missing; the tell in the log is `No file main.bbl.` followed
by `Package natbib Warning: Citation ... undefined`.

A compiled `main.pdf` is included only as a layout preview.

## What is in the draft

Main body currently has the following working result sequence:

1. OCT effects are dominated by a persona-common shift.
2. A global linear transformation explains a substantial part of the centered change.
3. RDM preservation is strongly functional-dose dependent.
4. Dispersion is not captured by one scalar dose-response curve.
5. Functionally matched untrained LoRAs reproduce broad contraction.
6. The new constitution x trait x persona interaction is small (~3.6% at L15), pushing against the original strong context-specific hypothesis.
7. Discussion frames the alignment significance mainly as a measurement/auditing result: fine-tuning geometry is not self-interpreting.

## Figures

The included PDFs/PNGs are **working reconstructions from numerical values already reported in the results write-up**, not authoritative final plotting outputs. This was intentional: they make the paper structure visible now, while leaving you free to redo labels, CIs, colors, panel structure, and exact estimators later.

See `FIGURE_NOTES.md` for provenance and what should be regenerated before submission.

## References

`main.tex` uses natbib + BibTeX against `references.bib`. The ICLR style file already loads
natbib and sets the author-year cite style, so use `\citep` / `\citet` and nothing else — in
particular **not** `\citeproc`, which is a Pandoc/CSL command that leaks in from markdown
conversion and is undefined in LaTeX.

The bibliography style is currently `plainnat` (natbib-native, ships with TeX Live). **Swap in
the official `iclr2027_conference.bst`** from the style bundle linked above when you add it —
a one-line change, and the rendered output should not shift much since the `\setcitestyle` in
the `.sty` already governs the in-text markers.
