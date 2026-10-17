# output/ — run outputs (created automatically)

This directory is the default writable output tree of the package (`dnec_common/paths.py`: `out_root()` =
`<package>/output`, override with the environment variable `DNEC_OUT`). It is empty in the release.

Every `run.py` writes into `output/<chain>/<mode>/` (`<mode>` = `quick`, `full` or `reduced`; per-chain override
`DNEC_<CHAIN>_WORK`, e.g. `DNEC_HIGGS_WORK`, or `--work DIR`). The work tree mirrors the archive layout of the research
project (`r1|r2|r3|r5|fu|w|ai/<calculation>/<run>/...`), so each freshly computed file and its archived copy in
`<chain>/<step>/expected/` share the same relative path. `run_all.py` writes `output/run_all_<mode>.json`.

Disk use: about 0.21 GB (64 000 files) for `run_all.py --quick`; the full runs write several GB (the C59 covariance cache alone is
1.3 GB, the C145 ensemble cache of `cmb/02 --full` 15.5 GB).
