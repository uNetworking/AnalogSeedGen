# Define the range of pages (1 to 32)
PAGES = $(shell seq 1 32)
SVG_FILES = $(patsubst %,Page%.svg,$(PAGES))
PDF_FILES = $(patsubst %,Page%.pdf,$(PAGES))

# Default target
all: output.pdf

# Compile the C++ program
page2-33: page2-33.cpp
	g++ page2-33.cpp -o page2-33

# Generate SVG files
Page%.svg: page2-33
	./page2-33 $*

# Rule to convert SVG to PDF
Page%.pdf: Page%.svg
	inkscape --export-filename=$@ --export-type=pdf $<

# Rule to combine PDFs
output.pdf: $(PDF_FILES)
	pdfunite frontpage.pdf $(PDF_FILES) output.pdf

# Clean up generated files
clean:
	rm -f $(SVG_FILES) $(PDF_FILES) output.pdf page2-33

.PHONY: all clean