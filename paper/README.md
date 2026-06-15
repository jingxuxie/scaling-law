# Paper draft

This directory contains a first conference-style manuscript draft for the project.

Main files:

- `main.tex`: paper draft in LaTeX.
- `references.bib`: bibliography.

Compile locally with:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current draft is intentionally written in generic `article` style.  For submission, switch to the target venue style, e.g. NeurIPS/ICLR/ICML, and convert the result summaries into polished figures.

Recommended next edits:

1. Replace textual result summaries with figures generated from `experiments/results/*`.
2. Move long proofs to an appendix.
3. Tighten theorem assumptions and notation.
4. Add a reproducibility appendix listing exact experiment commands.
5. Add limitations and broader-impact sections in the target venue format.
