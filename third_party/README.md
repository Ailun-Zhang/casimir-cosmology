# third_party — external code used by the package

## License audit (2026-10-15)
| component | version | license (read from the distribution) | may be redistributed? | what this package ships |
|---|---|---|---|---|
| SMDR, S.P. Martin and D.G. Robertson | 1.3 (2019) | GNU GPL version 2 or (at your option) any later version (`LICENSE.txt`, header of `README.txt`) | yes, verbatim copies with the license (GPL v2 §1) | the unmodified upstream tarball `smdr/download/smdr-1.3.tar.gz` and its license text `smdr/licenses/SMDR-1.3_LICENSE.txt` |
| TSIL (bundled in the SMDR tarball) | 1.46 | GPL-2.0-or-later (`tsil-1.46/LICENSE.txt`) | yes | inside the tarball; license text `smdr/licenses/TSIL-1.46_LICENSE.txt` |
| 3VIL (bundled in the SMDR tarball) | 2.02 | GPL-2.0-or-later (`3vil-v2.02/LICENSE.txt`) | yes | inside the tarball; license text `smdr/licenses/3VIL-2.02_LICENSE.txt` |
| `higgs/01_sm_mapping/c148_fit.c` | — | GPL-2.0-or-later: contains `SMDR_Fit_Inputs` of SMDR 1.3 `src/fit_obs.c`, modified (evaluation scales as arguments) | yes, under the GPL (modification notice and SPDX header in the file) | the source file |
| `higgs/zmlib/_smdr_betas.py` | — | GPL-2.0-or-later: generated mechanically from SMDR 1.3 `src/betas.c` by `higgs/zmlib/sm_rge.py` | yes, under the GPL (SPDX header in the file) | the generated module |
| `smdr/mingw_stubs/execinfo.h`, `smdr/fetch_and_build.py` | — | MIT (written for this package) | — | shipped |
| CAMB | 2.0.4 | — | not shipped | optional `pip install camb==2.0.4`, only for `cmb/01_fisher_kl --full` (the CAMB transfer tables shipped in `cmb/` are data products) |
| Wolfram Language / Mathematica | 15.0 | proprietary | not shipped | optional `*.wl` cross-checks |

Not shipped: the unpacked SMDR/TSIL/3VIL sources, the MinGW object files, static libraries and executables, and the
build logs (they are regenerated in `smdr/build/` by `fetch_and_build.py`, excluded by `.gitignore`, `MANIFEST.json`
and `SHA256SUMS.txt`). The MIT license of the package covers neither the tarball nor the two derived files; MIT is
GPL-compatible, so the combined program may be used and redistributed under GPL-2.0-or-later.

## SMDR 1.3 (with TSIL 1.46 and 3VIL 2.02)
- Official page: https://www.niu.edu/spmartin/SMDR/ ; tarball: https://www.niu.edu/spmartin/smdr/smdr-1.3.tar.gz ;
  reference: S.P. Martin and D.G. Robertson, arXiv:1907.02500.
- Version 1.3; size 2,220,791 bytes; SHA-256 `398b3947a6e5e02244d5adcd45911a3825e056a1b6f65bcf203975da0fc91b41`
  (`smdr/download/SOURCES.json`).
- Used by: `higgs/01_sm_mapping` (compiled cross-check of the Python port: one refit in `--quick` when built, all 38
  matching-scale refits with the driver `c148_fit.c` in `--full`); the Python RGE `higgs/zmlib/_smdr_betas.py` was
  generated from SMDR's `src/betas.c` (regeneration is optional: `sm_rge.build(force=True)` with the built tree).
  Every other result directory runs without SMDR (`higgs/01 --quick` then skips the compiled refit and says so).

`smdr/fetch_and_build.py`:
```
python third_party/smdr/fetch_and_build.py all                    # verify the shipped tarball (SHA-256), unpack, build, test
python third_party/smdr/fetch_and_build.py all --tarball PATH     # use another local copy (verified)
python third_party/smdr/fetch_and_build.py {fetch|unpack|libs|apps|driver|test} [--timeout 600] [--with-calc-all]
```
`fetch` uses the shipped tarball if present (no network) and downloads it from the official URL otherwise; a SHA-256
mismatch stops the build. The tree is built in `third_party/smdr/build/SMDR-1.3/` (env `DNEC_SMDR` overrides; read
through `dnec_common/paths.py: smdr_dir()`).
- Windows (tested): MinGW-w64 gcc 5.3 from conda-forge in a separate environment, e.g.
  `conda create -n cbuild -c conda-forge m2w64-toolchain` (test machine: `m2w64-gcc 5.3.0`, `m2w64-binutils 2.25.1`), then
  `set DNEC_MINGW_BIN=<env>\Library\mingw-w64\bin`. Flags `-O2 -D__USE_MINGW_ANSI_STDIO=1` (long-double printf),
  `-I smdr/mingw_stubs` (no-op `execinfo.h`), TSIL with its Makefile size flag (`-DTSIL_SIZE_LONG`), 3VIL and SMDR with
  `-fcommon`, executables linked with a 64 MB stack (`-Wl,--stack,67108864`).
- Linux/macOS: system gcc/clang (`--cc`, `--ar`), the same steps without the MinGW flags, or the upstream `make` route
  described in SMDR's `README.txt`; `ulimit -s unlimited` is recommended for the fits.
- Timings (Intel Core Ultra X9 378H, 16 cores): `all` 49 s in the release self-test of 2026-10-15 (verify, unpack,
  libraries, applications, driver, one test refit 14 s); 83 s for the libraries in the original packaging run.
- Known issues with the MinGW build: the stock `calc_fit` crashed (access violation) and the stock `calc_all` did not
  finish within 4 h; the package needs only `c148_fit`, which reproduces all 38 archived refits bit for bit
  (`higgs/01_sm_mapping/VERIFY.json`, mode full).

## Wolfram Language (optional cross-checks)
`.wl` files in `higgs/01_sm_mapping`, `02_finite_volume_matching`, `03_zero_mode_2d`, `04_cross_terms` and
`shape/02_realtime_theorem` reproduce selected numbers with Mathematica/wolframscript (env `WOLFRAMSCRIPT`); they are
not needed for any `run.py`. `C138_wolfram.wl` and `C140_wolfram.wl` read/write the Higgs work tree
(env `DNEC_HIGGS_WORK`, default `output/higgs/quick`; run `higgs/02` or `higgs/03 --quick` first).
