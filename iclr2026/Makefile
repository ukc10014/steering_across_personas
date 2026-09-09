.PHONY: pdf clean sync-figures sync-tables
pdf:
	latexmk -pdf main.tex

clean:
	latexmk -C
	rm -f main.bbl main.blg

# Refresh the figure set from workshop_iclr/, which is the single source of truth.
# Run after: git fetch origin && git checkout origin/workshop-figures -- workshop_iclr
sync-figures:
	cp ../workshop_iclr/figures/*.pdf figures/

# Refresh the OCT appendix tables from workshop_iclr/, same source-of-truth rule
# as sync-figures. NOTE: tableA_oct_stage_raw.tex is re-narrowed after copying
# (footnotesize + tighter tabcolsep) to fit the ICLR text width, and the upstream
# copy carries a `\ref{{...}}` double-brace typo that must be fixed on the way in
# (fixed at source in c1d60dc; this sed is a no-op once that reaches origin/main).
# Upstream floats are [h], which LaTeX silently downgrades to [ht] and which lets a
# table drift pages away from its own section; [!ht] is the stronger request.
sync-tables:
	cp ../workshop_iclr/tables/*.tex tables/
	sed -i '' 's/\\ref{{fig:oct-matched-dose}}/\\ref{fig:oct-matched-dose}/' tables/tableA_oct_stage_raw.tex
	sed -i '' 's/^\\small$$/\\footnotesize\n\\setlength{\\tabcolsep}{4pt}/' tables/tableA_oct_stage_raw.tex
	sed -i '' 's/^\\begin{table}\[h\]$$/\\begin{table}[!ht]/' tables/*.tex
