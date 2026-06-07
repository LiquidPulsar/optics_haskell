CHAPTERS_LHS := $(sort $(wildcard chapters/*.lhs))
CHAPTERS_TEX := $(patsubst chapters/%.lhs,generated/%.tex,$(CHAPTERS_LHS))

LHS2TEX  := lhs2TeX
LATEXMK  := latexmk
PDFFLAGS := -pdf -interaction=nonstopmode

# On Linux, lhs2tex is typically installed via a system package to /usr/share/lhs2tex-*.
# On macOS, cabal-installed lhs2tex bakes in its own data path, so no override is needed.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
export LHS2TEXINPUTS := $(shell ls -d /usr/share/lhs2tex-* 2>/dev/null | head -1):
endif

.PHONY: all clean watch

all: thesis.pdf

# Convert each .lhs chapter to .tex
generated/%.tex: chapters/%.lhs | generated
	$(LHS2TEX) --poly -o $@ $<

generated:
	mkdir -p $@

# Bootstrap: generate the lhs2tex preamble block first so \usepackage calls
# land in the LaTeX preamble rather than after \begin{document}.
generated/lhs2tex-preamble.tex: lhs2tex-preamble.lhs | generated
	$(LHS2TEX) --poly -o $@ $<

# Build the PDF; depends on the bootstrap and all generated chapter .tex files
thesis.pdf: thesis.tex preamble.tex references.bib generated/lhs2tex-preamble.tex $(CHAPTERS_TEX)
	$(LATEXMK) $(PDFFLAGS) thesis.tex

# Continuously rebuild on changes to .lhs sources or LaTeX inputs.
# Requires: sudo apt-get install inotify-tools
WATCH_FILES := thesis.tex preamble.tex references.bib lhs2tex-preamble.lhs
WATCH_DIRS  := chapters

watch:
	@which inotifywait > /dev/null || (echo "Install inotify-tools: sudo apt-get install inotify-tools" && exit 1)
	@echo "Watching for changes (Ctrl-C to stop)..."
	@while true; do \
		$(MAKE) --no-print-directory all; \
		inotifywait -q -r -e modify,create,delete $(WATCH_DIRS) $(WATCH_FILES); \
	done

clean:
	$(LATEXMK) -C
	rm -rf generated/
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk
