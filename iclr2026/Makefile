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
