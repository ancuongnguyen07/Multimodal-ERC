SHELL := /bin/bash

MAIN=report
TEXSRC=$(wildcard tex/*.tex)
BIBSRC=$(wildcard *.bib)
DIR_BUILD=build/final
# OPT= --interaction=nonstopmode

DRAFT=draft
DRAFT_TEXSRC=$(wildcard draft/*.tex)
DRAFT_BUILD=build/draft

all: $(DIR_BUILD)/$(MAIN).pdf

$(DIR_BUILD)/$(MAIN).pdf: $(TEXSRC) $(BIBSRC)
	latexmk -pdf -pdflatex="texfot pdflatex " $(MAIN) -auxdir=$(DIR_BUILD) -outdir=$(DIR_BUILD)
	cp $(DIR_BUILD)/$(MAIN).pdf pdf/

draft: $(DRAFT_BUILD)/$(DRAFT).pdf

$(DRAFT_BUILD)/$(DRAFT).pdf: $(DRAFT_TEXSRC) $(BIBSRC)
	latexmk -pdf -pdflatex="texfot pdflatex " draft/$(DRAFT) -auxdir=$(DRAFT_BUILD) -outdir=$(DRAFT_BUILD)

clean:
	rm -rf build/*
	rm -f $(MAIN).{aux,bbl,bcf,blg,fdb_latexmk,fls,lof,log,out,run.xml,synctex.gz,toc}
	rm *.log
.PHONY: clean all
