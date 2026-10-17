# Release notes — code and data, version 1.0.0 (2026-10-17)

*Code and data for: A Casimir turning point as the beginning of a small flat universe* — the code, archived results
and figure scripts behind the PRL *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins*,
the PRD *A Casimir turning point as the beginning of a small flat universe* and the companion technical report
*Casimir Cosmology: Derivations and Numerical Validation*. Archived on Zenodo, DOI 10.5281/zenodo.23434068; development
repository https://github.com/Ailun-Zhang/casimir-cosmology.

## Contents
- 24 result directories in four chains — `casimir/` 01–03, `higgs/` 01–12, `shape/` 01–06, `cmb/` 01–03 — each with
  `run.py --quick/--full`, `checks.json`, archived production outputs (`expected/`) and `VERIFY.json`.
- `figures/`: 18 figures (PDF + PNG) with one script each and their input data; `numbers/`: the single source of the
  51 claim-layer number macros shared by the PRL, the PRD and the report (generator + checker); `run_all.py`;
  `tools/`; `dnec_common/`; `third_party/` (SMDR 1.3 tarball and build script); `docs/technical_report/` (the companion report, PDF + source).
- `validation/` (R18): the independent reimplementations (blind keystone, V1–V5; T19; R14/R15) and numerical controls
  with specifications, code and outputs, the comparison with the main calculation and the report's own checks;
  `cmb/02_ensemble/data/`: the l ≤ 80 CAMB transfer table with its generator.
- Size: 1939 files, 263.3 MB (`MANIFEST.json`; without `output/`, caches, the SMDR build tree and
  `STATUS.json`):

| part | files | MB |
|---|---|---|
| `cmb` | 97 | 153.87 |
| `shape` | 119 | 75.23 |
| `higgs` | 1045 | 12.53 |
| `figures` | 139 | 8.02 |
| `validation` | 400 | 6.29 |
| `docs` | 52 | 4.42 |
| `third_party` | 8 | 2.30 |
| `casimir` | 51 | 0.38 |
| `numbers` | 7 | 0.09 |
| `(top level)` | 11 | 0.09 |
| `dnec_common` | 6 | 0.02 |
| `tools` | 3 | 0.02 |
| `output` | 1 | 0.00 |

## Reproduction status
- **Packaging verification runs** (2026-10-15/16, on the original package location): `--quick` passes in all 24
  result directories; `--full` passes in casimir 01–03, higgs 01–12, shape 01–06, cmb/03; cmb/01 `--full --c59-only`
  3/5 with the 2 C75 checks skipped (C75 production not rerun); cmb/02 `--reduced` 3/3 (`--full` not rerun). Table:
  `README.md` → Verification; per directory: `VERIFY.json`.
- **Release self-test, run 1** (2026-10-15): copy without `output/` in a new empty temporary directory, SMDR built from
  the shipped tarball (49 s), `python run_all.py --quick`: 26 of 28 verification runs passed; `shape/03_fossil`
  (`C76_summary`) and `shape/04_nonlinear_survival` (`C142_summary`) failed because the archived C76 trajectories
  (`fu/C76/draws.zip`) were not unpacked: an earlier step had created the empty directory and the staging function of
  the path shims took an existing directory as "present". The packaging runs had not seen this because their work
  tree still held the files unpacked by an earlier run. Fixed in `casimir/_casimir_paths.py`, `shape/_shape_paths.py`,
  `cmb/_cmb_paths.py` (`stage()` now unpacks every missing member; fresh files are never overwritten).
- **Release self-test, run 2** (2026-10-16, final code, new empty directory): SMDR built from the shipped tarball in
  38 s (compiled refit check of higgs/01 included); `python run_all.py --quick` — **25 of 25 verification runs fresh and
  passing, 236/236 checks, exit code 0, 24 min 55 s** (casimir 19 s, higgs 10 min 26 s, shape 11 min 4 s, cmb 2 min
  4 s, cmb/02 reduced 23 s, figures 40 s, numbers checks < 1 s; 15 workers, Intel Core Ultra X9 378H, Windows 11,
  Python 3.12.13); quick output 0.21 GB in 64 031 files, all inside `output/`; the Wolfram checks `C140_wolfram.wl`
  and `C138_wolfram.wl` ran with the rewritten paths (rc 0, outputs written); `python -m compileall` clean; no
  absolute path outside the archived files. Evidence (logs, VERIFY.json copies, SELFTEST.json) is kept outside the
  package by the workflow.
- **Release self-test, run 3 (R18 review, 2026-10-16)**: a fresh copy of the frozen package (11:00-11:37 UTC), `python run_all.py --quick --workers 12`: **25 of 25 verification runs fresh and passing; 236 of 237 checks passed, 1 skipped** (the optional compiled-SMDR refit of higgs/01, SMDR not built); all drivers rc 0; 36.6 min on a shared machine (casimir 26 s, higgs 868 s, shape 1072 s, cmb 160 s, cmb/02 reduced 24 s, figures 46 s). 234 of the 236 compared files were written by that run; the other two are the archive-vs-literal transcription checks `conv_cross_run02` and `conv_run04` of higgs/06 (tagged `"kind": "archive_vs_literal"` in its `checks.json`). The R18 revision added 5 checks (higgs/02 `matching_vs_potential`, higgs/04 `cross_decomp_post`, higgs/09 `fcc_exclusion` and two literal checks): 242 quick checks in all.
- **Final self-test (2026-10-17, release state after the R18 revision)**: a fresh copy without `output/`, `python run_all.py --quick --workers 8` — 25 of 25 verification runs fresh and passing, 241 of 242 checks passed and 1 skipped (the optional compiled-SMDR refit; SMDR not built), `validation/` (23 compared numbers, report checks) and the numbers (51/51, `numbers.tex` in sync) passed, all drivers exit code 0; 35.1 min on a shared machine (casimir 27 s, higgs 958 s, shape 902 s, cmb 142 s, cmb/02 reduced 23 s, figures 48 s, validation 3 s); no shipped file was changed by the run. Its quick and reduced records are written into the shipped `VERIFY.json` files (totals: quick 238/239 with 1 optional skip, reduced 3/3; full 255/257 from the packaging runs and the `cmb/02 --full` run, 2 skipped); the checks added after the full runs of their steps are quick-only.
- **Numbers**: `numbers/numbers.tex` regenerated from `numbers_v12.json` is byte-identical to the v12 PRD/PRL
  `numbers.tex`; `check_numbers.py` (revised for R18: explicit file + JSON pointer for every value, sign-aware,
  key-exact): 51 of 51 verified against the archived outputs (49 + the 2 FCC-ee macros added for R18), 0 mismatches;
  all 53 rows of the run05 number registry traced to their result files (`--registry`).

## Changes made in the release step
1. Run outputs (`output/`, 129 722 files, 2.47 GB) and non-release items were moved out of the package (nothing was
   deleted): the SMDR build tree (unpacked sources, objects, libraries, executables, logs), the one-off builder
   `shape/_packaging/` (research-tree paths; its substitutions are in the `PACKAGE_PATCHES.json` files), the agents'
   workflow records and the superseded `higgs/SHA256SUMS.txt`.
2. Third-party license audit (`third_party/README.md`): SMDR 1.3, TSIL 1.46 and 3VIL 2.02 are GPL-2.0-or-later and may
   be redistributed; the unmodified tarball is kept with the three license texts; `higgs/01_sm_mapping/c148_fit.c`
   (contains SMDR's `SMDR_Fit_Inputs`, modified) and `higgs/zmlib/_smdr_betas.py` (generated from SMDR's `betas.c`)
   carry GPL-2.0-or-later notices.
3. Fixes: `tools/verify_expected.py` (NaN matches NaN and null, ±inf matches itself, trees of strings only can pass,
   paths on another drive no longer break a check); `dnec_common/runner.py` (log file names: characters illegal in
   file names are replaced); `tools/verify_summary.py` (one table format for all chains; every README verification
   block regenerated from its own `VERIFY.json`, numbers unchanged); `tools/sha256sums.py` (`manifest` command, skip
   rules); the `stage()` fix above.
4. Absolute paths removed outside the archived files: `higgs/02/C138_wolfram.wl` and `higgs/03/C140_wolfram.wl` now use
   the work tree (`DNEC_HIGGS_WORK`, default `output/higgs/quick`), the engine string of `C138_report.py`, comments in
   `cmb/02` scripts and the `C75_analyze.py` docstring; `PACKAGE_PATCHES.json` records use placeholders
   (`<RESEARCH_ROOT>`, `<CODEX_WORKSPACE>`, `<PYTHON>`, `<MINGW_BIN>`, `<WOLFRAMSCRIPT>`, `<MATHEMATICA_15>`, `<AGENT_TEMP>`).
5. New files: `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json`,
   `environment.yml`, `requirements.txt` (from an import scan), `run_all.py`, `.gitignore`, `output/README.md`,
   `numbers/` (5 files), `docs/technical_report/README.md`, `third_party/smdr/licenses/`, `MANIFEST.json`,
   `SHA256SUMS.txt`, and the archived run05 number registry
   `higgs/09_claims/expected/w/claims_run05/run01/numbers_run05.csv` (copy of the C161 output, with `.sha256`).

## Synchronisation of 2026-10-16 (after the self-test)
- `higgs/01_sm_mapping/abs_stability.py` + `abs_stability.json` (absolute-stability bound m_stab = 170.9905 GeV,
  m0 − m_stab = 0.1016 GeV; added by the report agent) registered: package-relative paths in the JSON and the script,
  computed in `run.py` (quick and full), new check `abs_stability_central` (bit for bit in the rerun), README row
  (report Sec. 10.5). `higgs/01 --quick` rerun: 14/15, the optional compiled-SMDR check skipped (SMDR not built).
- 6 PRL preamble number macros merged into `numbers/numbers_v12.json` (49 macros: 39 verified, 10 unresolved,
  0 mismatches); `numbers.tex` regenerated and copied to the PRL, PRD and REPORT folders (byte-identical); the local
  definitions removed from the PRL preamble; PRL (8 pp.), PRD (35 pp.) and report (191 pp.) rebuilt with 0 errors and
  0 undefined references/citations.
- The report (PDF + source) copied to `docs/technical_report/` after regenerating its Sec. 22 / Appendix B tables
  from this package; the report-section column of the results → paper table filled (fixed section numbers).
- Console output of the checks now prints SKIP (not FAIL) for skipped optional checks.

## Known limitations
- All results are conditional on the models stated in the chain READMEs (reduced Higgs zero-mode model, leading cosh
  background, ideal full-sky CMB reading, ...).
- The l ≤ 80 CAMB transfer table (85 MB) and its generating script, settings, metadata and log are shipped in
  `cmb/02_ensemble/data/` (R18). `cmb/02 --full` (l ≤ 80 ensemble) was run once for the release on a copy of the package with identical code (2026-10-16, 12 workers, about 70 min: C145 run01 2374 s + refine 827 s; 15.5 GB cache): 13/13 full checks passed, all numbers bit for bit except the cross-KL table (max |diff| 2.3e-13); record in `cmb/02_ensemble/VERIFY.json` (full entry);
  `--reduced` (l ≤ 12) checks the same code path in every quick run.
  The C75 production of `cmb/01 --full` (without `--c59-only`) was not rerun either.
- Archived for R18 (`validation/`): the independent reimplementations of the review rounds (blind keystone, V1–V5;
  semi-blind T19; compute-first R14/R15) with specifications, code (SHA-256 as recorded before the comparison) and
  results, a comparison script (23 items, all within their pass criteria) and the report's own checks (rerun);
  C161 step 6 is `higgs/04_cross_terms/cross_decomp.py` (split recomputed from its 66 archived node shards).
- Not packaged (see the chain READMEs): the round-1/2
  historical production runs (their outputs used as seeds are archived), the C144 gauge profiles (26 MB),
  the D129 run02 extras, the stand-alone C60 Ewald validation, the contaminated B1 z_s = 2.5 run.
- Tested on Windows 11 x64 only (Python 3.12.13, conda-forge). Paths are portable and a POSIX route for SMDR is
  documented, but Linux/macOS were not run. With MinGW the stock SMDR `calc_fit` crashes and `calc_all` does not finish;
  only the driver `c148_fit` is used.
- 182 archived JSON/Markdown files contain absolute paths of the research machine in metadata fields; they are kept
  byte-identical to the research archive (SHA-256) and are never used as paths.
- Run records (R18): a run writes its verification record to `output/verify/<chain>/<step>/VERIFY.json`; the shipped
  `VERIFY.json` files change only with `DNEC_RECORD=1`. Figure PDFs are written without a creation date, so a rebuild
  is byte-identical. Still rewritten by a run: the figure files themselves (same bytes), `third_party/smdr/download/
  SOURCES.json` (date, only when SMDR is fetched) and, with `--write`, `numbers/numbers_v12.json`. `SHA256SUMS.txt`
  describes the state as released.
- The technical report has its own Zenodo record (publication/report; DOI `10.5281/zenodo.23434140`), which the papers
  cite; this code and data record (DOI `10.5281/zenodo.23434068`) ships a copy in `docs/technical_report/`.

## Release freeze (2026-10-17)
- Final documents built 2026-10-17: PRL 8 pp., PRD 36 pp., technical report 198 pp.; the report PDF and its LaTeX
  source with the comment lines removed (the same files as the report's Zenodo source archive) are in
  `docs/technical_report/` (52 files); `numbers.tex` is byte-identical in `numbers/` and the three documents
  (SHA-256 `4b66a62f57ead014…`).
- Report version 1.0 cites the Letter and the paper by author, title and year. Their arXiv identifiers are added to the
  repository and Zenodo metadata after announcement; this archived version is not changed.
- `.gitattributes` (`* -text`) turns off line-ending conversion, so `python tools/sha256sums.py check` also passes on a
  Git clone; `LICENSE` lists all four GPL-2.0-or-later files.
- `MANIFEST.json`: 1939 files, 263.3 MB; `SHA256SUMS.txt` written and checked with `python tools/sha256sums.py check`
  (0 mismatches) after the last edit.
