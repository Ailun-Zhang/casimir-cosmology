# The Letter's Supplemental Material inside this package

This folder is the Supplemental Material of the Letter *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small
Universe Begins* ([arXiv:2610.28704](https://arxiv.org/abs/2610.28704)), as published with it in the arXiv ancillary
files: the code and data of the static-box calculation (C170: Fig. 1 of the Letter, its Threshold and Accuracy
paragraphs and End Matter A) and of the abelian Higgs check (C171: End Matter A). The 55 files other than this note are
byte-identical to the ancillary files; `SHA256SUMS.txt` lists them. The technical report, version 1.1
(`docs/technical_report/`), describes these calculations in Secs. 7, 9, 18, 19 and 20.

Start with `README.txt`. Inside this package the production modules that the static-box scripts import are those of
the package itself, which are unchanged since version 1.0.0 (`c170/setup_from_release.py` verifies their SHA-256), so
set `DNEC_RELEASE` to the package root:

```
cd prl_supplement
set DNEC_RELEASE=..                 (bash: export DNEC_RELEASE=..)
python -B unbundle_records.py
python -B c170/setup_from_release.py
```

and continue with route A, B or C of `README.txt`. Work in a copy of this folder if you want to keep the package
unchanged: the setup step writes `c170_modules/`, and the runs write `rerun/` and restore the point records into
`results/`.
