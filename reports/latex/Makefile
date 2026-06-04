## Makefile — DS200.F21.CN2 LaTeX Report
## Yêu cầu: XeLaTeX + Biber (TeX Live 2023+)

MAIN   = main
ENGINE = xelatex
BIBER  = biber
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(MAIN).pdf

## Build đầy đủ (2 lần xelatex + biber + 1 lần nữa)
$(MAIN).pdf: $(MAIN).tex references.bib chapters/*.tex
	$(ENGINE) $(FLAGS) $(MAIN)
	$(BIBER) $(MAIN)
	$(ENGINE) $(FLAGS) $(MAIN)
	$(ENGINE) $(FLAGS) $(MAIN)

## Xoá file tạm (giữ PDF)
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.lot *.lof \
	      *.out *.run.xml *.toc *.fls *.fdb_latexmk

## Xoá tất cả kể cả PDF
distclean: clean
	rm -f $(MAIN).pdf
