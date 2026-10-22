# Changelog

All notable changes of the code and data package. Each version corresponds to one Zenodo release
(concept DOI 10.5281/zenodo.23434067; version 1.0.0: 10.5281/zenodo.23434068; version 1.1.0: 10.5281/zenodo.23552919) and to
the versions of the three documents listed under it.

## [1.2.0] — 2026-10-22
- `prl_supplement/` (new): the Letter's Supplemental Material as published in its arXiv ancillary files (55 files,
  byte-identical; `SHA256SUMS.txt`) with `PACKAGE_NOTE.md`: code and results of the static-box calculation (C170;
  the Letter's Fig. 1, its Threshold and Accuracy paragraphs and End Matter A) and of the abelian Higgs check (C171;
  End Matter A). Its scripts import the production modules of this package, unchanged since 1.0.0
  (`c170/setup_from_release.py` verifies their SHA-256); route A of its `README.txt` was reproduced on this package
  (C170 30/30 checks, C171 analysis identical).
- `figures/`: the four figures of report version 1.1 (`fig_rep_v11_static_box.py`, `fig_rep_v11_kappa.py`,
  `fig_rep_v11_abelian_width.py`, `fig_rep_v11_abelian_threshold.py`; registered in `make_all_figures.py`; they
  regenerate byte-identically), the derived-number scripts `derive_rep_v11_static_box.py`,
  `derive_rep_v11_wkb_split.py` (one-dimensional WKB decomposition of the static-box width slope, report Sec. 9.7)
  and `tab_rep_v11_abelian.py`, and their data (`figures/data/higgs/static_box_c170/`, `abelian_c171/`: copies of
  the C170 and C171 results, SHA-256 in `SOURCES*.json`). Five rows added to the figure table of `README.md`.
- `validation/abelian_blind/` (new, V7): blind recalculation of the abelian Higgs resonance by two methods of its
  own; widths agree with C171 to 1e-10 and energies to 1e-11 (`post_comparison/COMPARISON.md`). Row and protocol note
  in `validation/README.md`; its 18 archived files are recorded in `validation/SOURCES.json`.
- `docs/technical_report/`: report version 1.1 (2026-10-22, 223 pages; Zenodo record 10.5281/zenodo.23554537,
  all versions 10.5281/zenodo.23434139). The section numbering of version 1.0 is unchanged; new subsections at the
  ends of Secs. 7, 9, 18, 19, 20 and 22 (mechanism context, static box, abelian Higgs model, blind check V7,
  numerical controls, map of the Supplemental Material); consistency corrections in Secs. 1–21; Appendix B and the
  tables of Sec. 22 regenerated from this package (MSR value 0.64 of version 1.1.0).
- References to report figure and table numbers (README files, the "paper" fields of `checks.json` and
  `VERIFY.json`; 49 lines) follow report version 1.1, in which the figures from Fig. 12 and the tables from Table 13
  on are renumbered.
- Metadata: this archive and the technical report are cited by their concept DOIs (README.md, CITATION.cff,
  .zenodo.json).
- Unchanged: the code, checks and archived results of the four chains and the 51 claim-layer numbers (values of
  1.1.0).
Corresponding documents: arXiv:2610.24905 (version 2) and the Letter arXiv:2610.28704 (version 2), which cite this
archive and the report by their concept DOIs; technical report version 1.1 (included).

## [1.1.0] — 2026-10-22
- Claims layer (`higgs/09_claims`): the identification of the measured (Monte Carlo) top mass with the MSR mass
  m_MSR(1.3 GeV) now uses the four-loop pole mass of the survival criterion. The offset is +0.088 GeV (0.36 GeV to
  the pole mass at the order of turnover [Hoang 2020] minus 0.272 GeV between the all-orders and the four-loop pole
  mass, `pole_allorders_offset` of the same configuration) instead of +0.36 GeV. The marginalized survival
  probability for this identification changes from 0.50 to 0.64 (macro `PbarMSR`; registry rows R5-010, R5-020);
  the ranges of the Bayes factors become B_ex 0.61-0.97 and B_sel 0.98-2.11, and max |ln B| = 0.747 is unchanged.
  No other number and no figure changes (the criterion figures regenerate byte-identically). Sixteen archived
  files in `higgs/09_claims/expected/w/claims_run05/run01/` are updated; quick verification 16/16.
- Metadata: the papers are cited by title and arXiv number (arXiv:2610.24905; the Letter, arXiv:2610.28704) in
  README.md, CITATION.cff and .zenodo.json.
- Unchanged: `docs/technical_report/` (report version 1.0, which quotes the earlier value 0.50).
Corresponding documents: arXiv:2610.24905 (version 2) and the Letter arXiv:2610.28704 (version 1), which quote
0.64; technical report version 1.0.

## [1.0.0] — 2026-10-17 (first archived version)
Corresponding documents:
- Letter: A. Zhang, *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins* (2026), 8 pages.
- paper: A. Zhang, *A Casimir turning point as the beginning of a small flat universe* (2026), 36 pages.
- Technical report: *Casimir Cosmology: Derivations and Numerical Validation* — version 1.0 (2026-10-17, 198 pages;
  Zenodo record DOI 10.5281/zenodo.23434140), included in `docs/technical_report/` (PDF + LaTeX source).
- Shared numbers: `numbers/numbers_v12.json` (51 macros; the generated `numbers.tex` is the copy used by the Letter, the
  paper and the report).
- 2026-10-16 synchronisation: `higgs/01_sm_mapping/abs_stability.py` (absolute-stability bound m_stab = 170.99 GeV,
  report Sec. 10.5) registered with a quick check; 6 Letter preamble macros merged into the numbers; report added;
  report-section column of the results → paper table filled.

Contents: four chains of result directories (casimir 01–03, higgs 01–12, shape 01–06, cmb 01–03), each with
`--quick`/`--full` runs, archived production outputs and automatic comparison (`VERIFY.json`); 18 figures with their
scripts and data; the single source of the paper numbers with generator and checker; `run_all.py`; tools; the SMDR
1.3 tarball with a verified build script. Release preparation (2026-10-15/16): run outputs and build products moved
out of the package, third-party license audit, absolute paths removed outside the archived files, small fixes in the
verification tools and the archive staging, clean-copy self-test. Details: `RELEASE_NOTES.md`.

### R18 review revision (2026-10-16, before upload)
- Numbers: an explicit source (package file + JSON pointer) for every macro; `check_numbers.py` sign-aware and
  key-exact, registry key syntax traced (51/51 verified); `numbers/derive_numbers.py` → `derived_numbers.json`
  (\xiHminEW, \lamcErrUp/Dn, A_eff = 310.5); 2 new macros \PoneAtExclFCC = 0.023, \BexFCC = 0.033
  (`higgs/09_claims/fcc_exclusion.py`); \PbarOne points to the rebuilt Table 1 composite (0.7300).
- New scripts with checks: `higgs/02_finite_volume_matching/matching_potential.py` (matching vs zero-point
  confinement), `higgs/04_cross_terms/cross_decomp.py` (C161 step 6, split recomputed from 66 archived node shards);
  `higgs/07_realtime_xi/xi_compare.py` stores the universality residuals inside the interpolation range
  (0.448 ≤ P₁ ≤ 0.9975).
- `validation/`: the independent reimplementations (blind keystone, V1–V5; T19; R14/R15) with specifications, code
  and results, `compare_blind.py` (23 items) and the report's own checks; called by `run_all.py`.
- `cmb/02_ensemble/data/`: the l ≤ 80 CAMB transfer table (85 MB) with its generating script and settings.
- Paper locations of all READMEs, `checks.json` and `VERIFY.json` rewritten for the final paper/Letter (report-only
  figures marked); run records written to `output/verify/` (shipped `VERIFY.json` changes only with
  DNEC_RECORD=1); deterministic figure PDFs; the unlicensed palette-checker port removed; release notes updated.

### Release metadata (2026-10-17)
- `docs/technical_report/` rebuilt: the title page and the reference list cite the Letter and the paper by author,
  title and year; repository and archive identifiers are given in `README.md`, `CITATION.cff` and `RELEASE_NOTES.md`.

## [Unreleased]
- (next version: journal references and final identifiers)
