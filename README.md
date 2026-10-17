# Code and data for: A Casimir turning point as the beginning of a small flat universe

Version 1.0.0 (2026-10-17) · DOI [10.5281/zenodo.23434068](https://doi.org/10.5281/zenodo.23434068) ·
repository https://github.com/Ailun-Zhang/casimir-cosmology · code MIT, data and text CC BY 4.0 (exceptions in [License](#license))

Alan Zhang (ORCID [0009-0000-1187-0755](https://orcid.org/0009-0000-1187-0755)), McMaster University and Perimeter
Institute for Theoretical Physics.

This package reproduces the numbers and figures of two papers:

- **PRL** — *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins* (in preparation)
- **PRD** — *A Casimir turning point as the beginning of a small flat universe* (in preparation)

It contains the production code (copied from the research tree, paths rewired, algorithms and numerics unchanged),
the archived production results (`expected/`), a quick and a full run mode for every result directory, and an
automatic comparison of fresh output with the archive under stated tolerances. The results are conditional results of
the models stated in each chain README (for example the reduced Higgs zero-mode model, the leading cosh background,
the ideal full-sky CMB reading); the READMEs say which statements are proved, conditional or numerical.

## Contents
```
README.md  CHANGELOG.md  RELEASE_NOTES.md  LICENSE  CITATION.cff  .zenodo.json  environment.yml  requirements.txt
.gitignore  .gitattributes (no line-ending conversion in Git checkouts, so that the SHA-256 check passes everywhere)
run_all.py        one-command reproduction (--quick / --full)
MANIFEST.json     path, size and SHA-256 of every file;  SHA256SUMS.txt (same sums, sha256sum format)
dnec_common/      paths.py (all paths package-relative; env overrides), io.py, runner.py (driver of every run.py), sm_inputs.py
casimir/   01-03  SM Casimir weight N_C, gauge/Higgs zero-mode energy xi, shape Hessian and gain        (PRD Secs. IV, VII; App. B)
higgs/     01-12  Higgs zero-mode criterion: SM couplings at mu_b, FV matching, 2D model, cross terms,
                  hazard criterion, full-dimensional DMC, real-time xi_H, robustness, claims, universality,
                  inflation-era stability, two branches                                                   (PRL; PRD Sec. V; Apps. C, D)
shape/     01-06  emptiness gate, real-moment theorem, fossil, nonlinear survival, Moyal/WDW, n_s and r    (PRD Secs. VI-VIII; App. E)
cmb/       01-03  T+E covariance of the three-torus, Fisher/exact KL, fossil ensemble, joint state        (PRD Sec. VIII B; App. F)
figures/          style_v12.py, make_all_figures.py, one script per figure, data/ (inputs), out/ (PDF + PNG)
tools/            verify_expected.py (checks -> VERIFY.json), verify_summary.py, sha256sums.py
third_party/      SMDR 1.3 tarball (GPL) + fetch/verify/build script, license audit
validation/       independent reimplementations (keystone, V1-V5, T19, R14/R15): specs, code, results; comparison
                  with the main calculation (compare_blind.py) and the report's own checks (run.py --quick)
numbers/          numbers_v12.json = single source of the claim-layer numbers; make_numbers_tex.py -> numbers.tex; check_numbers.py
docs/technical_report/  companion technical report: main.pdf (198 pages) + LaTeX source (comments removed)
output/           run outputs (empty in the release; created by the runs; env DNEC_OUT)
```
Every result directory `<chain>/NN_<name>/` has a `README.md` (physics question, method, paper location, derivation
number, run commands, run times, results → paper table), `run.py` (`--quick`, `--full`, some `--reduced`), the source
modules, `checks.json` (the comparisons and tolerances), `expected/` (archived production outputs with `.sha256`
sidecars), `VERIFY.json` (results of the verification runs) and `inputs/` where needed. Shared modules:
`higgs/zmlib/`, `shape/shapelib/`. Every copied code file starts with a `PROVENANCE` line (original path in the
research tree, date, agent); every code change is listed in the chain's `PACKAGE_PATCHES.json`.

## Installation
Python 3.12 with the versions used for every recorded run:
```
conda env create -f environment.yml && conda activate dnec-v12     # or:  pip install -r requirements.txt
```
numpy 2.5.1, scipy 1.17.1, mpmath 1.4.1, sympy 1.14.0, matplotlib 3.11.0, PyMuPDF 1.27.2.2 (figures); optional psutil
7.2.2 and threadpoolctl 3.6.0 (cmb/02), camb 2.0.4 (only to regenerate the transfer tables of `cmb/01` and `cmb/02`), a C compiler for SMDR 1.3 (`higgs/01`
compiled refits; see [SMDR](#smdr-13-build)) and Wolfram Language for the optional `*.wl` cross-checks. Tested on
Windows 11 x64 (CPython 3.12.13, conda-forge); the code uses only the standard library and these packages, and all
paths go through `dnec_common/paths.py`, so it should also run on Linux/macOS (not tested there).

## Reproduce
```
python run_all.py --quick                  # all chains (quick) + cmb/02 reduced path + all figures   (24 min 55 s on 16 cores)
python run_all.py --full                   # prints the long computations and their run times (runs nothing)
python run_all.py --full --yes             # runs them: about 7-8 h on 16 cores
python run_all.py --quick --chains higgs --no-figures
python higgs/run_higgs.py --quick          # one chain (also casimir/run_casimir.py, shape/run_shape.py, cmb/run_cmb.py)
python higgs/04_cross_terms/run.py --quick # one result directory; archived upstream inputs are staged automatically
python figures/make_all_figures.py         # rebuild every figure from figures/data
python tools/verify_summary.py             # table of all VERIFY.json files
python numbers/check_numbers.py            # the paper numbers (numbers_v12.json) against the archived outputs
```
Options of every `run.py`: `--workers N` (default: number of CPUs − 1, at most 16), `--work DIR`, `--fresh`
(discard this step's previous outputs), `--no-verify`. Outputs go to `output/<chain>/<mode>/` (env `DNEC_OUT`, or
`DNEC_<CHAIN>_WORK`), a tree that mirrors the archive layout so that each fresh file and its archived copy share a
relative path. `--quick` always recomputes something non-trivial with the production code (a 2D node job, a DMC
shard, the hazard integral, the claims layer, re-integrated nonlinear trajectories, two covariance builds, ...) and
compares it with `expected/`; `--full` recomputes the production runs (resumable). Multiprocessing is guarded by
`if __name__ == "__main__":`; BLAS is forced single-threaded and the code parallelises over jobs.

## Run times
Intel Core Ultra X9 378H (16 cores), 64-bit Windows 11, Python 3.12.13. Quick: the final self-test of 2026-10-17 in a
fresh copy (`run_all.py --quick`, 8 workers, shared machine; the 2026-10-16 self-test with 15 idle workers took
24 min 55 s); full: the packaging runs of 2026-10-15/16 (chain READMEs) and `cmb/02 --full` (2026-10-16, copy of the release).

| chain | `--quick` (self-test) | `--full` | details |
|---|---|---|---|
| casimir | 27 s | ≈ 11 min (02 DMC 447–475 s, 03 122–194 s) | see `casimir/README.md` |
| higgs | 15 min 58 s | ≈ 2.6 h (dominated by 10, 05, 04, 09; 01 needs SMDR built) | see `higgs/README.md` |
| shape | 15 min 02 s | ≈ 3.2 h (11 689 s on 10 workers; ≈ 62 600 trajectories) | see `shape/README.md` |
| cmb | 2 min 22 s | 01 `--full --c59-only` ≈ 1 h (C59 covariance, 128 tasks); C75 production several hours; 03 22 s; 02 ≈ 70 min (12 workers) | see `cmb/README.md` |
| cmb/02 `--reduced` | 23 s | 25–27 s (l ≤ 12 path from scratch) |  |
| figures | 48 s | — |  |
| validation | 3 s | — (archived reimplementations; comparison only) |  |
| **total** | **35 min 04 s** | **≈ 7–8 h** (+ `cmb/02 --full` ≈ 70 min, by hand) | |

The full runs were verified step by step during packaging (`VERIFY.json`, mode `full`); `cmb/02 --full` (l ≤ 80
ensemble, shipped transfer table, 15.5 GB cache) was run once for the release on a copy with identical code: 13/13.

## Verification
`checks.json` lists, per result directory, the compared quantities (file and key in the work tree vs. the archived
copy, or a literal value), the paper location and the tolerance (e.g. critical top masses to 0.001 GeV, trajectory
records to 1e-9 relative, most analysis layers bit for bit). `tools/verify_expected.py` evaluates them and writes
`VERIFY.json` (checks, differences, timings, machine); `tools/verify_summary.py --update-readmes` refreshes the
verification blocks of all READMEs from the `VERIFY.json` files. State of the release (passed/total per mode; skipped =
optional inputs not shipped, e.g. the compiled SMDR refit of `higgs/01` when SMDR is not built):

| result directory | quick | reduced | full |
|---|---|---|---|
| casimir/01_sm_casimir_weight | 9/9 | - | 9/9 |
| casimir/02_zero_mode_energy | 9/9 | - | 9/9 |
| casimir/03_shape_hessian | 11/11 | - | 16/16 |
| cmb/01_fisher_kl | 5/5 | - | 3/5 (2 skipped) |
| cmb/02_ensemble | 12/12 | 3/3 | 13/13 |
| cmb/03_joint_state | 6/6 | - | 6/6 |
| higgs/01_sm_mapping | 14/15 (1 skipped) | - | 14/14 |
| higgs/02_finite_volume_matching | 9/9 | - | 10/10 |
| higgs/03_zero_mode_2d | 10/10 | - | 10/10 |
| higgs/04_cross_terms | 15/15 | - | 16/16 |
| higgs/05_fulldim_validation | 8/8 | - | 9/9 |
| higgs/06_hazard_criterion | 17/17 | - | 17/17 |
| higgs/07_realtime_xi | 3/3 | - | 4/4 |
| higgs/08_robustness_w0 | 6/6 | - | 7/7 |
| higgs/09_claims | 16/16 | - | 14/14 |
| higgs/10_universality | 7/7 | - | 8/8 |
| higgs/11_inflation_stability | 5/5 | - | 6/6 |
| higgs/12_two_branch | 3/3 | - | 4/4 |
| shape/01_emptiness_gate | 11/11 | - | 15/15 |
| shape/02_realtime_theorem | 19/19 | - | 16/16 |
| shape/03_fossil | 16/16 | - | 12/12 |
| shape/04_nonlinear_survival | 13/13 | - | 9/9 |
| shape/05_moyal_wdw | 9/9 | - | 23/23 |
| shape/06_primordial_spectra | 5/5 | - | 5/5 |

Totals of the recorded release state: quick 238/239 (1 optional check skipped) and reduced 3/3 (the final self-test of 2026-10-17, written into the `VERIFY.json` files), full 255/257 (2 skipped: the C75 production of `cmb/01` was not rerun). The checks added after the full runs of their steps (`higgs/01` `abs_stability_central`, `higgs/02` `matching_vs_potential`, `higgs/04` `cross_decomp_post`, `higgs/09` `fcc_exclusion` and its two literal checks) are quick-only.

Release self-test (2026-10-16, second run; the first run on 2026-10-15 found the archive-staging bug fixed in this
release): the package without `output/` was copied into a new empty temporary directory, SMDR
1.3 was built there from the shipped tarball, and `python run_all.py --quick` was run: 25 of 25 verification runs fresh and passing (236/236 checks), exit code 0, total 24 min 55 s; compileall clean. R18 review (2026-10-16): the same command on a fresh copy of the frozen package (12 workers, shared machine) — 25 of 25 runs fresh and passing, 236 of 237 checks passed and 1 skipped (optional compiled-SMDR refit), 36.6 min. Final self-test (2026-10-17, release state after the R18 revision): a fresh copy without `output/`, `python run_all.py --quick --workers 8` — 25 of 25 verification runs fresh and passing, 241 of 242 checks passed and 1 skipped (the optional compiled-SMDR refit; SMDR not built), `validation/` (23 compared numbers, report checks) and the numbers (51/51, `numbers.tex` in sync) passed, all drivers exit code 0; 35.1 min on a shared machine (casimir 27 s, higgs 958 s, shape 902 s, cmb 142 s, cmb/02 reduced 23 s, figures 48 s, validation 3 s); no shipped file was changed by the run. Details in
`RELEASE_NOTES.md`.

## SMDR 1.3 build
Only `higgs/01_sm_mapping` uses the compiled SMDR 1.3 (one refit in `--quick` if built, all 38 matching-scale refits in
`--full`); everything else runs without it.
```
python third_party/smdr/fetch_and_build.py all     # verify the shipped tarball (SHA-256), unpack, build, test (about 1 min)
```
Windows: MinGW-w64 gcc 5.3 from conda-forge in a separate environment (`conda create -n cbuild -c conda-forge
m2w64-toolchain`, then `set DNEC_MINGW_BIN=<env>\Library\mingw-w64\bin`); the script adds
`-D__USE_MINGW_ANSI_STDIO=1` (long-double printf), a no-op `execinfo.h`, `-fcommon` and a 64 MB stack. Linux/macOS:
system gcc/clang (`--cc`, `--ar`) or the upstream `make` route of SMDR's `README.txt`; `ulimit -s unlimited` is
recommended. Details, timings and the license audit: `third_party/README.md`.

## Large files
| item | size | shipped | how to regenerate |
|---|---|---|---|
| CAMB transfer table l ≤ 40 (`cmb/01_fisher_kl/expected/r5/obs_fisher/C59_transfer.npz`) | 34 MB | yes | `C59_transfer.py` with CAMB 2.0.4 (`cmb/01 --full`) |
| CAMB transfer table l ≤ 80 (`cmb/02_ensemble/data/C59_transfer.npz`, SHA-256 `8d0ea21f…df8f`) | 85 MB | yes (shipped for R18; its l ≤ 12 rows are also `cmb/02_ensemble/inputs/C59_transfer_l12.npz`) | computed with CAMB 2.0.4 by `cmb/02_ensemble/data/high_l_transfer.py` (settings in its header and in `data/SOURCES.json`; metadata `C59_transfer.json`, log `transfer.log`); used by `cmb/02 --full`, run once for the release on a copy of the package with identical code (2026-10-16, 12 workers, about 70 min: C145 run01 2374 s + refine 827 s; 15.5 GB cache): 13/13 full checks passed, all numbers bit for bit except the cross-KL table (max |diff| 2.3e-13); record in `cmb/02_ensemble/VERIFY.json` (full entry) |
| C59 covariance cache (`output/cmb/full/r5/obs_fisher/cache`) | 1.3 GB, 128 files | no | `python cmb/01_fisher_kl/run.py --full --c59-only` (about 1 h on 16 cores) |
| C75 covariance cache (`fu/C75/cache`) | 668 MB, 83 files | no | `python cmb/01_fisher_kl/run.py --full` (C75 production, several hours) |
| snapshot covariance cache (`ai/snapshot/P5_CMB/cache`) | 3.3 GB, 21 files | no | snapshot P5 (`high_cov.py`); only for the S1 regression of `cmb/02 --full` |
| C145 ensemble covariance cache (`ai/cmb_ensemble/run01/cache`) | 15.5 GB, 163 files | no | `python cmb/02_ensemble/run.py --full` (needs the l ≤ 80 table; `run_ens.py`, `run_ens.py --refine`) |
| nonlinear trajectory archives (`shape/*/expected/**/draws.zip`) | 69 MB in 6 zips | yes | `shape/02`, `03`, `04 --full` |
| Higgs node tables and gauge profiles | 40–170 MB per run | no | regenerated by `higgs/03`, `04`, `08`, `09 --full`; the C144 gauge profiles (26 MB) used by the round-2 1D seed chains are not shipped (the seeds are) |

## Figures
`python figures/make_all_figures.py` (or `... <substring>` for a subset) rebuilds every figure from `figures/data/`
into `figures/out/<name>.pdf` and `.png` (style: `figures/style_v12.py`; captions: `figures/CAPTIONS_higgs.md`,
`figures/CAPTIONS_shape.md`). `figures/data/` holds copies of the archived results the figures need;
`gen_casimir_slice_shape.py` recomputes the Casimir slice data (about 20 s).

| figure | paper | script |
|---|---|---|
| `prl_fig1_threshold` | PRL Fig. 1 (`fig:threshold`) — finite-volume threshold λ_c against λ_SM(μ_b) | `fig_prl_fig1_threshold.py prl` |
| `prd_threshold` | PRD Fig. 2 (`fig:h-threshold`, Sec. V A); report Fig. 9 | `fig_prd_threshold.py` |
| `prl_fig2_criterion` | PRL Fig. 2 (`fig:criterion`) — survival P₁(m_t) and the critical masses | `fig_prl_fig2_criterion.py prl` |
| `prd_criterion` | PRD Fig. 3 (`fig:h-criterion`, Sec. V E); report Fig. 12 | `fig_prd_criterion.py` |
| `prl_fig3_xi` | PRL Fig. 3 (`fig:xi`); PRD Fig. 4 (`fig:h-xi`, Sec. V H); report Fig. 14 — ξ_H condition | `fig_prl_fig3_xi.py` |
| `prd_higgs_wavefunction` | report Fig. 7 (Sec. 7.2) — Higgs zero-mode wave function | `fig_prd_higgs_wavefunction.py` |
| `prd_gamma_validation` | report Fig. 8 (Sec. 7.5) — width validation | `fig_prd_gamma_validation.py` |
| `prd_hazard_realtime` | report Fig. 11 (Sec. 9.5) — hazard and real-time evolution | `fig_prd_hazard_realtime.py` |
| `prd_cross_terms` | report Fig. 10 (Sec. 8.6) — cross terms | `fig_prd_cross_terms.py` |
| `prd_universality` | report Fig. 13 (Sec. 12.2) — universality | `fig_prd_universality.py` |
| `prd_trace` | report Fig. 2 (Sec. 2) | `fig_prd_trace.py` |
| `prd_casimir_shape` | PRD Fig. 1 (`fig:pa-casimir-shape`, Sec. IV A); report Fig. 3 (Sec. 3) | `gen_casimir_slice_shape.py`, `fig_prd_casimir_shape.py` |
| `prd_emptiness_gate` | PRD Fig. 5 (`fig:s-gate`, Sec. VI B); report Fig. 15 | `fig_prd_emptiness_gate.py` |
| `prd_turning_state` | report Fig. 16 (Sec. 15) | `fig_prd_turning_state.py` |
| `prd_nonlinear_survival` | PRD Fig. 6 (`fig:s-survival`, Sec. VII F); report Fig. 18 | `fig_prd_nonlinear_survival.py` |
| `prd_fossil_shape` | report Fig. 17 (Sec. 15) | `fig_prd_fossil_shape.py` |
| `prd_primordial` | report Fig. 19 (Sec. 16) | `fig_prd_primordial.py` |
| `prd_fossil_sky` | PRD Fig. 7 (`fig:c-sky`, Sec. VIII B); report Fig. 20 | `fig_prd_fossil_sky.py` |

## Numbers shared by the papers and the report
`numbers/numbers_v12.json` is the single source of the claim-layer numbers that the text of all three documents uses —
the PRL, the PRD and the companion technical report *Casimir Cosmology: Derivations and Numerical Validation*
(`docs/technical_report/`). The three LaTeX sources `\input` the same `numbers.tex`, generated from the JSON by
`python numbers/make_numbers_tex.py` (byte-identical to the v12 PRL/PRD copy at import). To change a number, edit
only the JSON (`values`, `precision`), regenerate `numbers.tex`, copy it to the three sources (`--out PATH ...`) and
rerun `python numbers/check_numbers.py --write`, which checks every number against the archived outputs of this
package: each macro names the package file and JSON pointer of every value (`source`), and the shown value must equal
that value rounded, sign included. Result: 51 of 51 macros verified, 0 mismatches; three numbers that are simple
functions of archived outputs (`\xiHminEW`, `\lamcErrUp`, `\lamcErrDn`) come from `numbers/derived_numbers.json`,
written by `numbers/derive_numbers.py` (details in `numbers/README.md`). `run_all.py` runs both checks.

## Results → paper
Collected from the tables at the end of every result-directory README (which also give the archived source of each
number). Paper locations are the section, figure and table numbers of the submitted PRL and PRD (compiled
2026-10-16; LaTeX labels where useful); a number that only the companion technical report gives is marked
"report Sec. n" (or Table/Fig. n); "same" refers to the file of the row above. Output files are relative to the work
tree `output/<chain>/<mode>/`; the archived copy has the same relative path below the step's `expected/`. The column
*report section* gives the section of the technical report (`docs/technical_report/`) that derives or validates
the result.

| result directory | paper location | number / figure | output file : key | report section |
|---|---|---|---|---|
| casimir/01_sm_casimir_weight | PRD Sec. IV A–B, Eq. `eq:pa-NC`; Sec. I | N_C(AAA) = +25.515; N_C(PPP) = −30 | `r5/C57_spin_structure/C57_spin_structures.json` : `SM_spin_structures/AAA/N_C_signed`, `.../PPP/N_C_signed` | 3 |
| casimir/01_sm_casimir_weight | PRD App. B 1, Table IV | eigenvalues +1902.24 / −893.20; κ_E = 13.5286, κ_T2 = −6.3524; trace identity 1e-8 | same : `SM_spin_structures/<s>/{Hess_eigs,kappa,trace_over_8_3_Z}` | 3 |
| casimir/01_sm_casimir_weight | PRD Fig. 1(a), (c); report Fig. 3 | N_C per spin structure; gains | same : `SM_spin_structures/<s>/{N_C_signed,gains_zero_momentum}` | 3 |
| casimir/01_sm_casimir_weight | PRD Sec. IV B; App. B 1 | PPP centre 11.19; PPP hypercharge max 17.79; AAA centre 22.74 / 9.86 | `r5/C68_holonomy_preparation/C68_results.json` : `PPP_centre_111`, `PPP_Y_iso_scan_max/NC`, `AAA_centre_100`, `AAA_centre_111` | 3 |
| casimir/02_zero_mode_energy | PRD Sec. IV B (`sec:pa-zeromodes`), Table I; App. B 2 | ξ = 0.2993 / 0.2999 / 0.3043 | `fu/C72/C72_summary.json` : `r_band/one-loop mu=1/L_b/exact_local_dmc_r/lam{0,0.01,0.1}` | 4 |
| casimir/02_zero_mode_energy | PRD App. B 2 (Eq. `eq:pa-gauss-bound`) | ξ ≥ 0.2580, ξ ≤ 0.3083 | same : `.../lower_operator_bound_r`, `.../upper_gauss_r/0.01` | 4 |
| casimir/02_zero_mode_energy | PRD App. B 2 | DMC 2.1402(5), 4.1171(14), 12.5869(22) | same : `comparisons_ours_vs_codex/*` | 4 |
| casimir/02_zero_mode_energy | PRD App. B 2 | LE_EW = 4.4478(13) / 4.4750(17); Higgs cost 1.78/L vs 0.46/L (×3.9) | `fu/C72/C72_dmc_results.json` : `extrapolated/ew_running_lam*`; `C72_summary.json` : `higgs_coupled_vs_decoupled` | 4 |
| casimir/02_zero_mode_energy | PRD Sec. IV B (next order) | ξ → 0.2826 (≈ 0.28); dξ_hol = −0.0140, dξ_Higgs = −0.0033 | `fu/C81/C81_results.json` : `r_first_order/landau_3pol`; `fu/C78/C78_results.json` : `shifts/*` | 4 |
| casimir/03_shape_hessian | PRD Sec. VII A, Eq. `eq:s-Hesstrace`; App. E 1 | 8/3 (eigenvalue at p = −2) | `fu/C73/C73_symbolic.json` : `S2/values/-2` | 2, 14 |
| casimir/03_shape_hessian | PRD Sec. VII A, Eq. `eq:s-nogo` | ΔU = (8/3)[U − Q − 3N/2] | same : `S5/*` ; `C73_thermo.json` | 2, 14 |
| casimir/03_shape_hessian | PRD Sec. VII B, Eq. `eq:s-gain` | guaranteed gain 4.500465 (NOGO-11); 𝒢(13.53) = 216.46 | `fu/C73/C73_gain.json` : `lower_bounds/0/G via ODE`, `rows/0/gain_closed` | 2, 14 |
| casimir/03_shape_hessian | PRD Sec. VII C; App. B 2 | κ_E = 19.324, κ_T2 = −9.074 (ξ = 0.2999) (NOGO-05/06) | `fu/C83/C83_results.json` : `next_order/gauss/cases/LO/{kappa_E,kappa_T2}` | 2, 14 |
| casimir/03_shape_hessian | PRD Sec. IV B; App. B 2; PRD Sec. VII E | κ_E = 18.04, κ_T2 = −8.34 at ξ = 0.2826 (NOGO-07); Higgs part 0.14 of the shift | same : `next_order/gauss/cases/{both,both_without_Higgs}/kappa_E` | 2, 14 |
| casimir/03_shape_hessian | PRD App. E 1 | s = 1 Laplacian −4/3; abelian zero-point ratio −8.88 | `fu/C73/C73_zeta.json`, `C73_abelian_zeropoint.json` | 2, 14 |
| higgs/01_sm_mapping | PRL core sentence; PRD App. D 3 | λ(μ_b) = −0.0097741 at (172.60, 0.1180, 125.13) | `r3/sm_mapping/package/sm_point.json` : `point_172p60_0p1180_125p13/lam_mu_b` | 10 |
| higgs/01_sm_mapping | PRD App. D 3 | y_t(μ_b) = 0.4278267; g₂/g_Y/g₃ = 0.52776/0.44277/0.53565 | same file : `yt_mu_b`, `g2_mu_b`, `gY_mu_b`, `g3_mu_b` | 10 |
| higgs/01_sm_mapping | PRL (μ_b ≈ M̄_P/722) | μ_b = 3.3726×10¹⁵ GeV | `sm_point.json` : `mu_b_GeV` | 10 |
| higgs/01_sm_mapping | PRL headline | λ_c = λ(μ_b; m_c) = −0.0091/−0.0121/−0.0152 | `sm_point.json` : `lam_mu_b_at_run05_critical_mt` | 10 |
| higgs/01_sm_mapping | PRL Table I (m̄ form) | m̄_t(m̄_t) 162.7281/163.1637/163.6012 (α_s 0.1180; also 0.1171/0.1189) | `r3/sm_mapping/run02/R3_mbar_run05.json` : `new_central/<α_s>/P1=*/mbar_mbar_QCD4L` | 10 |
| higgs/01_sm_mapping | PRD App. D 3 | compiled SMDR − Python at Q = M_t; λ(μ_b) SMDR-RG / own RG / Python | `r3/sm_mapping/run02/C148r2_summary.json` : `A_match_at_Mt_compiled_minus_python`, `B_mu_b` | 10 |
| higgs/01_sm_mapping | PRD App. D 3 | compiled matching-scale variation +0.0331/−0.0186 GeV | `C148r2_summary.json` : `E_budget_P1=0.5/a_matching_scale_compiledSMDR/common_xi` | 10 |
| higgs/01_sm_mapping | PRD App. D 3 | benchmarks (BKPV critical M_t 171.453; r2→r3 chain +0.063 GeV) | `r3/sm_mapping/run01/C148_bench.json` (full mode) | 10 |
| higgs/01_sm_mapping | report Sec. 10.5 | absolute-stability bound m_stab = 170.99 GeV (m0 − m_stab = 0.10 GeV) | `w/abs_stability/abs_stability.json` : `central/m_stab_pole`, `central/m0_pole` | 10.5 |
| higgs/02_finite_volume_matching | PRD Sec. V A, App. C 1 | T_B/2 = −0.1128925, c₂ = 0.0062594 | `r2/higgs_matching/run01/C138_checks.json` : `R14_fv_internal/TB_half`, `c2P` | 6 |
| higgs/02_finite_volume_matching | PRD Sec. V A, App. C 1 | γ_φ^C(μ_b) = +2.2×10⁻⁴ | `C138_checks.json` : `RG_one_loop_consistency_lam-0.010/gamma_phi_Coulomb` | 6 |
| higgs/02_finite_volume_matching | report Sec. 6 (γ table) | γ(λ = −0.010, v11) MC+CW 9.871×10⁻⁴; MU 2.051×10⁻³ | `C138_static.json` : `table/v11/MC/-0.01/gamma`, `table/v11/MU/-0.01/gamma` | 6 |
| higgs/02_finite_volume_matching | report Sec. 11.1 (Table 14) | MC+CW (D110) 172.349/172.768 (R14 mapping) | `C138_summary.json` : `["central_MC+CW_mu=1/L"]` (full) | 6 |
| higgs/02_finite_volume_matching | PRL Table I (round 3) 'one-loop matching' | μ ∈ [½, 2π]/L items | `C138_summary.json` : `error_budget_GeV` (full) | 6 |
| higgs/02_finite_volume_matching | PRD Sec. V (effective slope); report Secs. 6–7 | one-loop matching lowers the local slope of the zero-point confinement by 10–33 % at r = 1–5 (λ = −0.010: 0.110–0.331; −0.013: 0.105–0.310) and roughly halves the barrier above the resonance level: radial potential U_ad 1.57 → 0.72 (λ = −0.010), 1.21 → 0.53 (−0.013); field-space potential U_g 1.55 → 0.70, 1.19 → 0.51 (units 1/L) | `w/matching_potential/matching_potential.json` : `headline`, `lambda/<λ>/slope`, `lambda/<λ>/barrier_above_Q` (`matching_potential.py`, R18; reads `figures/data/higgs/generated/potential_2Dex_lam010_013.csv` and the T14a resonance level) | 6, 7 |
| higgs/03_zero_mode_2d | PRD Sec. V B | λ = 0: 2D_ex 4.45086, 2D_G 4.5077 vs DMC 4.4478(13) | `r2/higgs_beyond_bo/run01/C140_lam0.json` : `Q`, `test/Q_2D_ex`, `test/minus_C72` | 7, 9 |
| higgs/03_zero_mode_2d | report Sec. 11.1 (Table 14) (2D direct, R14) | 172.5185/173.0140/173.5155 | `r3/twod_direct_sm/run01/C150_summary.json` : `main_2D_direct_as0.1180/critical_mt` | 7, 9 |
| higgs/03_zero_mode_2d | report Sec. 11.1 (Table 14) (2D_G) | 172.4671/172.9412 | `…/run01/C150_summary.json` : `twoD_G_direct/critical_mt` | 7, 9 |
| higgs/03_zero_mode_2d | report Sec. 11.1 (Table 14) (rows MU → 2D direct) | MU 172.2099/172.5837 … | `…/run01/C150_summary.json` : `convergence_table` | 7, 9 |
| higgs/03_zero_mode_2d | report Sec. 11.1 (Table 14) (+ NNLO) | 172.5826/173.0775/173.5784 | `r3/twod_direct_sm/run02/C150_summary.json` : `main_2D_direct_as0.1180/critical_mt` | 7, 9 |
| higgs/03_zero_mode_2d | PRD Sec. V B–D (node widths, no cross terms) | γ(m_t, u) at 11 nodes | `r3/twod_direct_sm/run02/shards2d/2Dex_*.json` : `cols/*/gamma2` | 7, 9 |
| higgs/04_cross_terms | PRL Table I; PRD Sec. V E–F | m_c^pole 172.5024/172.9595/173.4184 (α_s 0.1180) | `r3/holonomy_cross/run05/C149_run05_summary.json` : `crit/0.1180/P1=*` | 8 |
| higgs/04_cross_terms | PRL Fig. 2(a); Table I | α_s 0.1171: 172.156/172.611/173.069; 0.1189: 172.849/173.308/173.768 | same : `crit/0.1171`, `crit/0.1189` | 8 |
| higgs/04_cross_terms | PRL text | P₁(172.60) = 0.8426; P₁(172.52) = 0.8909 | same : `P1/172.6`, `P1/172.52` | 8 |
| higgs/04_cross_terms | PRL Fig. 2(a) | P₁ grid 172.0…173.8 | same : `P1_grid` | 8 |
| higgs/04_cross_terms | PRL headline | λ_c(μ_b) −0.0091339/−0.0121415/−0.0151849 | same : `lam_c` | 8 |
| higgs/04_cross_terms | PRL Table I (cross-term row) | −0.0802/−0.1180/−0.1600 vs no cross term | same : `shift_vs_T14b/0.1180` | 8 |
| higgs/04_cross_terms | PRD Sec. V C, App. C 4 | RG closure 0.994/0.973/0.993/0.990 | `r3/holonomy_cross/run05/C149_coulomb_check.json` : `<r>/closure_Z` | 8 |
| higgs/04_cross_terms | PRD Sec. V C, App. C 4 | top chiral G(m_t²), B/(q_Φ² m²) | `r3/holonomy_cross/run04/C149_top_chiral.json` | 8 |
| higgs/04_cross_terms | PRD Sec. V C–D, App. C 4 | node widths γ(m_t, u) with cross terms | `r3/holonomy_cross/run05/shards/X_{main,als}_u*.json` : `cols/*/gamma` | 8 |
| higgs/05_fulldim_validation | PRD Sec. V B | ΔQ (same wall) −0.0032(8) (λ = −0.010), −0.0034(9) (−0.013) | `r3/fulldim_dmc/run01/C151_summary.json` : `lam_neg/-0.01/dQ_ex_main`, `lam_neg/-0.013/dQ_ex_main` | 18 |
| higgs/05_fulldim_validation | PRD Sec. V B | λ = 0: Q_full 4.44659(94) vs C72 4.4478(13) | same : `lam0_regression` | 18 |
| higgs/05_fulldim_validation | PRD Sec. V B | tail amplitude A = 0.92(16)/1.03(7); WKB 0.937/0.973; CAP d ln γ/dQ | same : `lam_neg/*/tail_fit`, `wkb`, `conversion` | 18 |
| higgs/05_fulldim_validation | PRD Sec. V B | 3D collective coordinate γ_3D/γ_2D 0.9992–1.0000 | same : `model_3D_zeta` | 18 |
| higgs/05_fulldim_validation | PRL Table I 'full-dimensional validation' | Δm_c +0.0046(11)/+0.0075(19) GeV with the run05 slopes (`09_claims`: `w/claims_run05/run01/table1_rebuild.json` : `t14a/central`); C151 with the earlier slopes +0.0043(10)/+0.0070(17) | same : `delta_m_c` | 18 |
| higgs/05_fulldim_validation | PRD Sec. V B | DMC energies per wall and tau | `r3/fulldim_dmc/run01/T14a_dmc_combined.json`, `shards/*.json` | 18 |
| higgs/06_hazard_criterion | PRL Table I; core sentence | m_c^pole 172.5024/172.9595/173.4184 | `r3/holonomy_cross/run05/C149_run05_summary.json` : `crit/0.1180` | 9, 20 |
| higgs/06_hazard_criterion | PRL Fig. 2(a) | m_c at α_s 0.1171/0.1189; slope dm_c/dα_s ≈ 387 GeV | same : `crit` | 9, 20 |
| higgs/06_hazard_criterion | PRL headline | λ_c −0.0091/−0.0121/−0.0152 | same : `lam_c` | 9, 20 |
| higgs/06_hazard_criterion | PRL text; PRD Sec. V E | P₁(172.60) 0.8426; P₁(172.52) 0.8909; P₁(172.2) 0.9850; P₁(173.0) 0.4549 | same : `P1` | 9, 20 |
| higgs/06_hazard_criterion | PRL Fig. 2(a) | P₁(m_t) grid and I(m_t) | `r3/hazard_criterion/hazard_table.json` : `P1/0.1180`, `I/0.1180` | 9, 20 |
| higgs/06_hazard_criterion | PRL Table I (m̄ form) | m̄_t 162.7281/163.1637/163.6012 | `r3/sm_mapping/run02/R3_mbar_run05.json` : `new_central` | 9, 20 |
| higgs/06_hazard_criterion | report Sec. 11.1 (Table 14) | MU → … → 2D direct (R14) → +NNLO → +cross (run02) → run04 → run05 | `r3/hazard_criterion/hazard_table.json` : `convergence_table_0p1180` | 9, 20 |
| higgs/06_hazard_criterion | numerics | quadrature / interpolation sensitivity | `hazard_table.json` : `max_abs_diff_*` | 9, 20 |
| higgs/07_realtime_xi | PRL ξ condition; PRD Sec. V H | ξ_c(172.6) at run05 = 0.0738 ± 0.0025 (run02: 0.0741); P₁ = 0.5 point 0.1205 | `w/claims_run05/run01/xi_run05_summary.json` | 9, 12 |
| higgs/07_realtime_xi | PRD Sec. V H | ξ_c(m_t) table, EW image | `w/claims_run05/run01/xi_run05/C157_xi_quick.json` | 9, 12 |
| higgs/07_realtime_xi | PRD Sec. V H | universal ξ_c(P₁): residuals of 16 run02 variants and run05 points | `xi_run05_summary.json`; `xi_run05_compare.csv` | 9, 12 |
| higgs/07_realtime_xi | PRD Sec. V H (bands, run02 centre) | SM band 0.0750 ± 0.0130, α_s band 0.0795 ± 0.0397 (m_t = 172.6) | archive only | 9, 12 |
| higgs/07_realtime_xi | PRD App. D 2 (2D real time) | 2D vs 1D: ΔP_surv ≤ 2×10⁻⁴ | archive only | 9, 12 |
| higgs/08_robustness_w0 | PRL Table I row (vi) | ±0.035/±0.067/±0.112 GeV (r ∈ [10⁻³, 10⁻²]); r = 10⁻⁴ tail −0.089/−0.1725/−0.2795 | `r3/higgs_robustness/run01/C152_input_dmc.json` : `vi_inflation_scale` | 9, 11 |
| higgs/08_robustness_w0 | PRD Sec. V D–F | d ln I/dλ_b −805.4/−487.8/−319.2 | same : `slope_update/dlnI_dlam_b` | 9, 11 |
| higgs/08_robustness_w0 | PRD Sec. V D, App. D 2 | e-fold finite differences of hL_b (−0.170/−0.270/−0.394) | `C152_post.json` | 9, 11 |
| higgs/08_robustness_w0 | PRD Sec. V D, App. D 2 | w₀(B3) 0.9945/0.9907/0.9852/0.9801; critical 172.4819/172.9473/173.4076 (run02 base) | `C152_mc_w0.json` : `with_B3_w0_of_mt` ; `C152_w0_B3.json` | 9, 11 |
| higgs/08_robustness_w0 | report Sec. 11.3 | h/M_P 6.054e-6 (pipeline) vs 6.116e-6 (v11): +0.0006/+0.0012/+0.0020 GeV | `C152_accounting.json` | 9, 11 |
| higgs/09_claims | PRL claim sentence; PRD Sec. V F | P̄₁ direct 0.7300 (rebuilt Table 1 composite; round-3 composite 0.7302; run02 0.7271) | `w/claims_run05/run01/t1new/C154_correction_summary.json` : `marginal/equal_mass/D1_direct_pole/Pbar` | 11 |
| higgs/09_claims | PRL range; PRD Sec. V F | P̄₁ for σ-pole, MSR, m̄_t, ATLAS+CMS, CMS, BKPV/FJ variant | same : `marginal/equal_mass/*` | 11 |
| higgs/09_claims | PRL one sentence; PRD Sec. V F | Bayes B_ex, B_sel | same : `bayes` | 11 |
| higgs/09_claims | PRD Sec. V F | HL-LHC separation 1.7–2.5σ | same : `HLLHC` | 11 |
| higgs/09_claims | PRL Fig. 2; PRD Sec. V F | critical masses incl. P₁ = 0.05; exclusion thresholds; FCC-ee 5.3σ | `w/claims_run05/run01/C154_summary.json` : `critical_mt`, `exclusion_mt`, `future`; `future_reach.csv`, `exclusion_mt.csv` | 11 |
| higgs/09_claims | PRL End Matter B; PRD Sec. V F | FCC-ee exclusion strength: central P₁ = 0.023 and B_ex = 0.033 at m̂ = 173.68 GeV (`\PoneAtExclFCC`, `\BexFCC`) | `w/claims_run05/run01/fcc_exclusion.json` : `headline` | 11 |
| higgs/09_claims | PRL Fig. 2(a, b) data | P₁(m_t, α_s) grid, contours, ellipses, ξ curves | `fig2a_*.csv`, `fig2b_*` in `w/claims_run05/run01/` | 11 |
| higgs/09_claims | PRL Table I (rebuilt) | model composite +0.0721/−0.0842 at P₁ = 0.5; matching row +0.012/−0.049 | `w/claims_run05/run01/table1_rebuild.json`, `table1_error_budget_run05.csv` | 11 |
| higgs/09_claims | PRD Sec. V D, App. D 2 | P₂ critical 172.380/172.764/173.131; B3 172.486/172.953/173.414 | `w/claims_run05/run01/crit_variants.json` : `run05` | 11 |
| higgs/09_claims | PRD Sec. V F | independent MC P̄₁ (0.7304 direct) | `w/claims_run05/run01/pbar_direct_mc.json` : `results` | 11 |
| higgs/10_universality | PRD Sec. V G | b_eff 1.3992/1.2991/1.1893; dλ_c/d ln η 0.00124/0.00206/0.00315; dm_c/d ln η −0.1898/−0.3113/−0.4732 | `r3/universality/run01/C156_summary.json` : `eta` | 12 |
| higgs/10_universality | PRD Sec. V G | ∂ln\|λ_c\|/∂ln(g₂, g_Y, y_t) 0.663/0.163/0.627 | same : `couplings` | 12 |
| higgs/10_universality | PRD Sec. V G | N_C 13.124/20.871 → 21.068/23.669 | `C156_centre.json` | 12 |
| higgs/10_universality | PRD Sec. V G | flux sector 100.719/L, ratio 2.3565 | `C156_twist.json` | 12 |
| higgs/10_universality | report Sec. 12 | run02 repair | `r3/universality/run02/C156_run02.json` | 12 |
| higgs/11_inflation_stability | PRL one sentence; PRD Sec. V H | ξ_inf(EW) 0.04877/0.05512/0.06044 (α_s 0.118) | `r3/inflation_stability/run01/xi_inf_table.csv` : column `xi_inf_ew`, rows `mt`, `als` | 12 |
| higgs/11_inflation_stability | PRD Sec. V H | ξ_inf(μ_b) 0.03030/0.03781/0.04413 | same : `xi_inf_mub` | 12 |
| higgs/11_inflation_stability | PRD Sec. V H | α_s = 0.1171/0.1189 rows | same | 12 |
| higgs/11_inflation_stability | PRD Sec. V H | MMR 2021 reproduction 0.05946 | `r3/inflation_stability/run01/repro/repro_step1.json`; `C155_summary.json` : `reproduction` | 12 |
| higgs/11_inflation_stability | PRD Sec. V H | all rows | `C155_summary.json` : `xi_inf_rows` | 12 |
| higgs/12_two_branch | PRL one sentence; PRD Sec. V D, App. D 2 | P₂ = P₁² critical 172.376/172.759/173.124 (run02 base) | `r3/two_branch/run01/C159_summary.json` : `headline/crit_as0p1180` | 9 |
| higgs/12_two_branch | PRD Sec. V D, App. D 2 | GH interference ε 0.9948–0.99997 | same : `headline/GH_interference_eps` | 9 |
| higgs/12_two_branch | PRD Sec. V D, App. D 2 | Fréchet bands [172.372, 172.498], [172.713, 172.954] | same : `crit` | 9 |
| higgs/12_two_branch | PRD Sec. V D, App. D 2 | static-spectrum check | `r3/two_branch/run01/static_check.json` : `runs` | 9 |
| shape/01_emptiness_gate | PRD Sec. VI A, Eq. `eq:s-emptiness`; report Table 18 (Sec. 13) | Σ\|n\| < 4.762 (ξ = 0.2999), 5.047 (0.258), 4.898 (0.28), 6.80 (0); ξ(5) = 0.2649 (GATE-02..06) | `r3/shape_hardening/run01/emptiness/C153_emptiness.json` : `quanta_budget/*/budget_sum_n`, `xi_where_budget_equals/5` | 13 |
| shape/01_emptiness_gate | PRD Sec. VI A | AAA fermion budget 5.50 (GATE-07) | same : `quanta_budget/*/fermion_budget` | 13 |
| shape/01_emptiness_gate | PRD Sec. VI B, Eq. `eq:s-windowR`, Fig. 5(a) | log₁₀R = 93.86 (free, j = 1); 94.04 (ξ = 0.30, j = 1); 96.48 (j = 85) (GATE-08..10) | `r5/C55_window_gate/C55_window_gate.json` : `slices/0/log10_R`; `C153_emptiness.json` : `reheating_excess/xi0.30/log10R/{1,85}` | 13 |
| shape/01_emptiness_gate | report Sec. 13 | prefactor 10^97.79 (GATE-11) | `C55_window_gate.json` : `meta/log10_prefactor` | 13 |
| shape/01_emptiness_gate | PRD Sec. VI C, Fig. 5(b) | T_max = 7.264 / 7.177; 3-dof crossing 6.909 / 6.822 (GATE-12, 13) | `C153_emptiness.json` : `preparation_window/hLb0.0044/*`, `hLb0.0048/*` | 13 |
| shape/01_emptiness_gate | PRD Sec. VI C; report Sec. 13 | side ratio 4.96 (ξ = 0.30), 1.74 (free), 3.56–5.43 over the ξ band (GATE-14, 16) | `fu/C80/C80_results.json` : `runs/*` | 13 |
| shape/02_realtime_theorem | PRD Sec. VII C, Eqs. `eq:s-T12`, `eq:s-floor`; App. E 2 | m₁₁ = 10.554, m₁₂ = 2.4685, m₂₁ = 136.37, m₂₂ = 31.99; K(19.3237) = 24.5630; K(18.0) = 20.3204 (FLOOR-01..03, 11) | `r3/shape_hardening/run01/linear/C153_linear.json` : `floor_closed_form/{19.3237,18.0}/*` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII C (footnote on conventions) | σ_floor 0.10374 (H_eff), 0.10403 (C60/C76), 0.1031 (v11 δ); next order 0.09337 (FLOOR-04..08) | same : `floor_conventions/*`, `next_order_floor/*` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII D, Thm. `thm:s-real`, Eq. `eq:s-Gtx`; App. E 2; report Table 19 | G = 2.3952, 8.2632, 45.097, 313.79, 961.71; natural/floor 1.000485 (TX-01..06, FLOOR-10) | `r1/state_selection/run03/C137c_results.json` : `rows[k]/opt_over_floor`, `rows/0/adiabatic_over_floor` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII D | G(∞) = 2𝒢 = 1348.55 (κ = 19.3237), 1058.34 (18.0) (TX-07, 08) | `C153_linear.json` : `floor_closed_form/*/G_inf_closed_2calG` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII D, report Fig. 16; report Table 20 (Sec. 15) | S₂(τ) grid; t_*(99 %) = 0.23696 (MC 2.5–97.5 %: 0.2245–0.2518); t_*(90 %) = 0.3547 by interpolation between the N = 2048 node at 0.35 and the N = 512 node at 0.40 (68 % half-width 0.0033 from 2×10⁴ draws of the Jeffreys posteriors of the node fractions; the 0.30 node enters in 10 % of the draws; `numbers/derived_numbers.json` → `tstar90`, with the same method reproducing the archived 99 % interval 0.225–0.252), logistic 0.352; next order 0.26514 (TX-19..24) | `r3/shape_hardening/run02/C153_run02_summary.json` : `lo_t/t_star_linear_interp/0.99/*`, `lo_t/t_star_logistic/t_star/0.9`, `nx_t/...`; `r3/shape_hardening/run01/C153_nl_summary.json` : `grid[]` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII D | never-real two-branch survival 0.043 / 0.027; incoming single-branch 0.887 / 0.723 (TX-13..15) | `C153_nl_summary.json` : `never[*]` | 15 |
| shape/02_realtime_theorem | PRD Sec. VII D | adiabatic states: survival 0.992 / 0.609 / 0.129 (TX-18) | `C153_nl_summary.json` : `adiab[*]/future/p` | 15 |
| shape/02_realtime_theorem | number keys of the papers (`C153_summary.json`) | all of the above in one file | `r3/shape_hardening/run01/C153_summary.json` | 15 |
| shape/03_fossil | PRD Sec. VII C, Eqs. `eq:s-Tsym`, `eq:s-family` | floor, natural/floor, gain 216.46 (free) | `r5/C58_turning_point_states/C58_results.json` : `SM_AAA/aspect/*` | 15 |
| shape/03_fossil | PRD Sec. VII E, Table III | σ_E: 0.0589 (free), 0.1041 (ξ = 0.2999, C60/C76 convention), vs ξ (FOS-01, 02; FLOOR-06) | `fu/C82/C82_results.json` : `sigma_E_leading/*` | 15 |
| shape/03_fossil | PRD Table III | side difference 18 % / 10 %; median coefficient 1.588 (FOS-04, 05) | same : `typical_side_difference_plateau/*`, `median_coeff` | 15 |
| shape/03_fossil | PRD Table III | plateau free σ_E 0.0587 | `r5/C63_rolling_fossil/C63_results.json` : `SM_AAA/modes/aspect/natural_rms` | 15 |
| shape/03_fossil | PRD Sec. VII E, Eq. `eq:s-scaling` | K(19.32) = 24.5501; dlnσ/dlnκ = 1.366; LiteBIRD 3.1 %, CMB-S4 1.6 %; σ ratio 0.6306 at r = 1e-4 (FLOOR-01, FOS-14..16) | `r2/observables/run01/C143_observables.json` : `F_kappaE`, `dlnsigma_dlnkappa`, `LiteBIRD_*`, `CMBS4_*`, `sigma_vs_r` | 15 |
| shape/03_fossil | PRD Sec. VII E (error budget) | NL/linear aspect variance 1.043 / 1.045 (FOS-07) | `fu/C76/C76_summary.json` : `sets/{hi,ind}/*` ; also `../04` C142 `regression/C76_reference` | 15 |
| shape/03_fossil | report Fig. 17 | fossil distributions (free 2048, ξ = 0.30 2048) | `r5/nonlinear_SM/C60_summary.json`, `fu/C76/C76_summary.json` + trajectory archives | 15 |
| shape/03_fossil | PRD Sec. IV C (`sec:turning`) | φ_b, N_post, hL_b of the SM pivot | `r5/C62_SM_pivot/C62_results.json` : `SM_AAA/*` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F, Fig. 6; report Table 22 (Sec. 15) | survival vs s (NL-01) | `r2/shape_nonlinear_threshold/run01/C142_summary.json` : `sets/amp_s*/survival` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F | s₅₀, s₉₀ (N = 256) | same : `amp_threshold/*` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F | s₉₉ = 2.756 (MC 95 % 2.66–2.90), logistic 2.863 (boot 2.74–2.96); next order 3.160 / 3.294 (NL-06, 07, 09) | `r3/shape_hardening/run02/C153_run02_summary.json` : `lo_amp/s99_loglinear/central`, `lo_amp/s99_logistic_in_ln_s/s99`, `nx_amp/*` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F | survival 0.9976 / 0.9834 / 0.9536 at s = 2.5 / 3.0 / 3.5 (N = 2048) (NL-03) | same : `lo_amp/rows[*]/p` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII D, F | past-set real states 0.623 / 0.184 / 0.133 (TX-27, 29) | `C142_summary.json` : `sets/r1_tx*/survival` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F | next order s₉₀ = 4.19, s₅₀ = 9.07 (NL-10) | `r3/shape_hardening/run01/C153_nl_summary.json` : `next_amp_threshold` | 15 |
| shape/04_nonlinear_survival | PRD Sec. VII F | plateau background: 0 / 2560 fate flips (NL-14) | `r3/shape_tier2/run01/plateau/C160_plateau_summary.json` : `sets/*/n_flips` | 15 |
| shape/05_moyal_wdw | PRD Sec. VII E (corrections) | Moyal −2.03e-4 / −1.08e-3 (ξ = 0.30); next order −1.79e-4 / −9.2e-4 (FOS-08..10) | `r3/shape_tier2/run01/moyal/C160_moyal.json` : `SM/xi030/rows/0/rel/a_Heff/{aspect,angle}`, `SM/next_order/...` | 15 |
| shape/05_moyal_wdw | PRD App. E 2 / PRD Sec. VII E | exact 1D quantum − TWA (FOS-13 analogue) | `.../moyal/C160_moyal_1d.json` : `checks[*]` | 15 |
| shape/05_moyal_wdw | PRD App. E 2 (shear criterion) | Δρ = 7.8e-5 (NL-17) | `r3/shape_tier2/run01/shear/C160_shear_wdw.json` : `physical`, `fits` | 15 |
| shape/05_moyal_wdw | PRD Sec. VII D (`sec:s-equivalent`, Eq. `eq:s-WDW`) | D_re = 2.388e-6 at X = 2282.49; 3.1e-2 / 4.3e-4 / 1.5e-5 at X = 100 / 300 / 1000 (TS-01, 02) | `r3/shape_tier2/run01/b1/b1_cases.json` : `cases/A_Ai_real@<X>/rows/0/D_re` | 15 |
| shape/05_moyal_wdw | PRD App. E 2 | R15 counterexample D_re 0.969 (ε = 1), 0.161 (ε = 0.3) (TS-03); (Ai+0.7Bi)⊗χ₀ 5.0e-4 (TS-04) | `b1_cases.json` / `.../b1/C160_b1_wdw_zs0.json` : `runs[*]` | 15 |
| shape/05_moyal_wdw | PRD Sec. VII D; App. E 2 | B3: Var H_min = k/2m at width 1/(2m\|ω\|) (TS-05) | stdout `logs/05_moyal_wdw/C141_minvar_lemma.out.txt` | 15 |
| shape/05_moyal_wdw | number keys of the papers | all parts | `r3/shape_tier2/run01/C160_summary.json` | 15 |
| shape/06_primordial_spectra | PRD Sec. VIII A | n_s = 0.9653407, r = 0.0034624 (N_* = 55.722) (PRIM-01, 02) | `r5/C64_plateau_ns_r/C64_results.json` : `rows/6/{ns,r}` | 16 |
| shape/06_primordial_spectra | PRD Sec. VIII A | n_s = 0.9617386, r = 0.0042022 (N_* = 50.328) (PRIM-03, 04) | same : `rows/3/{ns,r}` | 16 |
| shape/06_primordial_spectra | report Fig. 19 (n_s, r vs N_*) | 12 values of N_* | same : `rows[*]` | 16 |
| shape/06_primordial_spectra | PRD Sec. VIII A | initial-state tests | same : `initial_state_tests` | 16 |
| cmb/01_fisher_kl | PRD Sec. VIII B; PRD App. F (covariance) | KL D_TE, D_T of the free fossil at 1.07 (end-to-end check) | `checks/kl_quick.json` : `D_TE_fresh_fresh`, `D_T_fresh_fresh` | 17 |
| cmb/01_fisher_kl | PRD Sec. VIII B | Fisher σ(β_E) = 0.023 / 0.073 / 0.25 at L₀ = 1.05 / 1.1 / 1.2 (CMB-23, v11 values) | `r5/obs_fisher/C59_shape_fisher.json` (`--full`) | 17 |
| cmb/01_fisher_kl | PRD Sec. VIII B (v11 comparison) | v11 window 1.093–1.133 for the typical fossil, circle bound 1.0931 (CMB-06, 22) | `fu/C75/C75_summary.json` : `per_sigma/*` | 17 |
| cmb/01_fisher_kl | PRD App. F | transfer-table metadata, low-k fix | `r5/obs_fisher/C59_transfer.json` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B, Fig. 7 (P(read\|L₀)) | P_joint peak 0.3440 at 1.075 (CMB-02) | `ai/cmb_ensemble/run01/results/S3_postcheck.json` : `variants/TE80_grid/modes/mid/{Pj_max,L0_Pj_max}` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | P_joint ≥ 0.1 for [1.0175, 1.1825]; ≥ 0.3 / 0.2 / 0.05 windows (CMB-03, 04) | same : `.../win_joint/{0.1,0.3,0.2,0.05}` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | integrated measure 0.04223 (CMB-05); representative window [1.0931, 1.1394] (CMB-06) | same : `.../mean_span`, `.../rep_window` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | P_cond ≥ 0.5 up to 1.090 (CMB-07) | same : `.../win_cond/0.5` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | read edge u_max 1.021–1.039 (CMB-01) | same : `variants/TE80_grid/edges` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | noise 0.313 / 0.0384; T only 0.286 / 0.0352 (CMB-10, 11) | same : `variants/{noise_TE80_grid,T80_grid}/modes/mid/*` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | next order 0.377 at 1.065; [1.015, 1.1675]; 0.0421; rep window [1.082, 1.128] (CMB-14..17) | `r3/shape_hardening/run01/ensemble/C153_ensemble.json` : `results/0.0940/variants/TE80_grid/mid/*` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B, Fig. 7 (mechanism: D_prof vs u) | 55 fossils (columns: xi = sin 3θ, s = L₀/d_LSS, u) | `r3/cleanup_t12/run01/fossil_table_v2.csv` | 17 |
| cmb/02_ensemble | PRD Sec. VIII B | near-window packet 1.7 %, α = 17.6 for P = 0.5 (CMB-12, 13) | `ai/cmb_ensemble/run01/priors/priors_summary.json`, `P4_size_prior_ens.json` | 17 |
| cmb/02_ensemble | PRD App. F | KL table, profiles (TE30/40/80, T80, fixed/profiled), Brent, convergence | `ai/cmb_ensemble/run01/results/{kl_table,S2_profiles,S2_brent,S4_convergence,S4_noise,S3_analysis}.json` (archived; `--full` recomputes) | 17 |
| cmb/03_joint_state | PRD Sec. VIII B (joint state) | σ_φ = 0.00973692 (JOINT-01) | `ai/joint_state/run01/joint_state_results.json` : `T3/B3/ref/sigma_B3` | 17 |
| cmb/03_joint_state | PRD Sec. VIII B | σ_N ≥ 0.935075 (ref) / 0.938387 (window) (JOINT-02) | same : `T3/response/{ref,window}/floor_sigma_N` | 17 |
| cmb/03_joint_state | PRD Sec. VIII B | Δφ_b = 6.1825e-4; N window [68.480, 68.516], width 0.0359 (JOINT-04, 05) | same : `T4/typical_fossil_window/dphi_window`, `T2/N_window_typical`, `T2/window_width_efold` | 17 |
| cmb/03_joint_state | PRD Sec. VIII B | flat-dφ readability 1.389e-3 (N_max = 100) → 5.35e-5 (1e6) (JOINT-06) | same : `T4/shape_marginalised/A_fixed_typical/flat_dphi_Nmax=*` | 17 |
| cmb/03_joint_state | PRD Sec. VIII B | S_E = 11964 (JOINT-09); φ_b(ref) = 5.57936 (BKG-03) | same : `T2/S_E_ref_closed`, `T2/phib_ref(N=68.25)` | 17 |
| cmb/03_joint_state | report Sec. 21 | interface to C145 | `ai/joint_state/run01/C145_interface.json` | 17 |
| cmb/03_joint_state | PRD Sec. VIII B (B3 slope) | symbolic slope identities | stdout of `b3_slope_symbolic.py` | 17 |
| higgs/04_cross_terms | PRL main text and End Matter A (field-by-field split); PRD Table V, row "Further cross terms"; report Sec. 8.6 | split of the run05 cross-term shift: total −0.0802/−0.1180/−0.1600 GeV; SU(3)–top −0.0403, vev mixing + Coulomb +0.0275, top chiral −0.0219 GeV at P₁ = 1/2 (R5-049…053) | `w/claims_run05/run01/cross_decomp/cross_decomp_summary.json` (`cross_decomp.py post`) | 8 |
| validation | PRL main text and End Matter A (reimplementations); PRD Table V ("Blind" rows) | blind keystone and V1–V5, semi-blind T19, compute-first R14/R15: specifications, code, results; 23 compared numbers within their pass criteria | `validation/*/results*/`; comparison `output/validation/comparison.json` (`validation/run.py`) | 19 |
| validation | report chapters 1–5 and 13–17 (`% [num:]` rep-A) | the report's own checks of intermediate steps (rerun) | `validation/report_checks/expected/*.out.txt` | 1–5, 13–17 |

## Archived files, absolute paths, provenance
`expected/`, `inputs/`, `higgs/data/` and `figures/data/` hold byte-identical copies of the production outputs of the
research project (each with its SHA-256; `.sha256` sidecars), and `validation/` holds byte-identical copies of the
independent reimplementations and controls (SHA-256 in `validation/SOURCES.json`). Some of these archived JSON/Markdown files (182) contain
absolute paths or host names of the research machine in metadata fields (input file names, working directories,
creation records). They are kept unchanged so that their SHA-256 matches the research archive; the code never uses
these fields as paths and the checks do not compare them. The archived code in `validation/` (93 files) likewise keeps the absolute paths of the machine it ran
on; it is not executed by the package (except `validation/report_checks/`, which has none). Outside these archive
directories the package contains no
absolute paths (`PACKAGE_PATCHES.json` uses placeholders such as `<RESEARCH_ROOT>` for the original locations).

## License
- Code: MIT (`LICENSE`).
- Data and text (archived results in `*/expected/`, `*/inputs/`, `*/data/`, `figures/out/`, READMEs and other
  documentation): Creative Commons Attribution 4.0 International (CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/).
- Exceptions, GNU GPL version 2 or (at your option) any later version: `third_party/smdr/download/smdr-1.3.tar.gz`
  (SMDR 1.3 with TSIL 1.46 and 3VIL 2.02, S.P. Martin and D.G. Robertson; license texts in
  `third_party/smdr/licenses/`), `higgs/01_sm_mapping/c148_fit.c` and `higgs/zmlib/_smdr_betas.py` (derived from SMDR
  1.3), and the archived blind-check drivers `validation/sm_mapping/code/drivers/c148_fit_copy.c` and `smdr_driver.c`
  (notice in `validation/sm_mapping/code/drivers/LICENSE-NOTICE.txt`). MIT is GPL-compatible, so the combined program may be used and redistributed under GPL-2.0-or-later.

## Citation
Please cite this archive (`CITATION.cff`; DOI 10.5281/zenodo.23434068) and the two papers above. SMDR: S.P. Martin and
D.G. Robertson, arXiv:1907.02500.

## Known limitations
Summarised in `RELEASE_NOTES.md`; each chain README has a "Not packaged / partial coverage" section (for example the
round-1/2 historical production runs and `cmb/02 --full`). The independent reimplementations are in `validation/`.
