# Changelog

All notable changes of the code and data package. Each version corresponds to one Zenodo release
(DOI 10.5281/zenodo.23434068) and to the versions of the three documents listed under it.

## [1.0.0] — 2026-10-17 (first archived version)
Corresponding documents:
- PRL: A. Zhang, *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins* (2026), 8 pages.
- PRD: A. Zhang, *A Casimir turning point as the beginning of a small flat universe* (2026), 36 pages.
- Technical report: *Casimir Cosmology: Derivations and Numerical Validation* — version 1.0 (2026-10-17, 198 pages;
  Zenodo record DOI 10.5281/zenodo.23434140), included in `docs/technical_report/` (PDF + LaTeX source).
- Shared numbers: `numbers/numbers_v12.json` (51 macros; the generated `numbers.tex` is the copy used by the PRL, the
  PRD and the report).
- 2026-10-16 synchronisation: `higgs/01_sm_mapping/abs_stability.py` (absolute-stability bound m_stab = 170.99 GeV,
  report Sec. 10.5) registered with a quick check; 6 PRL preamble macros merged into the numbers; report added;
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
- Paper locations of all READMEs, `checks.json` and `VERIFY.json` rewritten for the final PRD/PRL (report-only
  figures marked); run records written to `output/verify/` (shipped `VERIFY.json` changes only with
  DNEC_RECORD=1); deterministic figure PDFs; the unlicensed palette-checker port removed; release notes updated.

### Release metadata (2026-10-17)
- `docs/technical_report/` rebuilt: the title page and the reference list cite the Letter and the paper by author,
  title and year; repository and archive identifiers are given in `README.md`, `CITATION.cff` and `RELEASE_NOTES.md`.

## [Unreleased]
- (next version: journal references and final identifiers)
