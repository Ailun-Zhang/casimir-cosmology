# docs/technical_report/ — companion technical report

**Casimir Cosmology: Derivations and Numerical Validation** — Alan Zhang, report version 1.0 (2026-10-17), 198 pages;
companion technical report of the PRL and the PRD, not independently peer reviewed.

- `main.pdf` — the final report as built on 2026-10-17 (pdflatex → bibtex → pdflatex ×2; 0 LaTeX errors, 0 undefined
  references or citations); 52 files in this folder: `main.tex`, 29 chapter files, 16 figure PDFs, `refs.bib`, `main.bbl`,
  `main.pdf`, `numbers.tex`, `dnec_macros.tex`, this README.
- `main.tex`, `chapters/` (one file per section; fixed numbering: Part I Secs. 1–17, Part II Secs. 18–22, Part III
  Sec. 23, Appendices A–B), `figures/`, `dnec_macros.tex`, `refs.bib` — the LaTeX source with the comment lines
  removed (the same files as the source archive of the report's own Zenodo record).
- `main.bbl` — the BibTeX output of `refs.bib`, so that the source also compiles without running BibTeX (as on
  arXiv).
- `numbers.tex` — `numbers/numbers.tex` without its comment lines (the macros are identical; generated from
  `numbers/numbers_v12.json`; the PRL and the PRD use the same file).

Sec. 22 (reproduction guide) and Appendix B (table of key numbers) contain tables generated from this package
(result-directory READMEs, `VERIFY.json`, `numbers/numbers_v12.json`, `numbers_run05.csv`) by the report's table
generator; the column *report section* of the results → paper table in the top-level `README.md` uses the same
section numbers. Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (TeX Live 2026).
The title page cites the Letter and the paper by author, title and year, and gives the DOI of the report's own Zenodo
record (publication/report, 10.5281/zenodo.23434140) and that of this code and data record (10.5281/zenodo.23434068);
this folder is a convenience copy of report version 1.0 inside the code and data record.
