.PHONY: pdf clean sync-figures
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
# copy carries a `\ref{{...}}` double-brace typo that must be fixed on the way in.
sync-tables:
	cp ../workshop_iclr/tables/*.tex tables/
	sed -i '' 's/\\ref{{fig:oct-matched-dose}}/\\ref{fig:oct-matched-dose}/' tables/tableA_oct_stage_raw.tex
	sed -i '' 's/^\\small$$/\\footnotesize\n\\setlength{\\tabcolsep}{4pt}/' tables/tableA_oct_stage_raw.tex
