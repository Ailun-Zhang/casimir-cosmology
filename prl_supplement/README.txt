Supplemental Material

A. Zhang, "Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins" (Letter, 2026)
Department of Physics and Astronomy, McMaster University; Perimeter Institute for Theoretical Physics

Description: Code and data of the static-box calculations (Fig. 1, the Threshold and Accuracy paragraphs and End
Matter A) and of the abelian Higgs check (End Matter A). All other numbers of the Letter are reproduced by the code and
data release v1.0.0 of the companion papers (Zenodo, https://doi.org/10.5281/zenodo.23434068), whose production
modules these scripts use unchanged.


START HERE

  results/C170/README_C170.md    static box: physics, method, the 30 checks, results and limitations
  results/C171/README_C171.md    abelian Higgs model: method, the 13 checks and results
  numbers/make_numbers_c170.py   computes every new number of the Letter from the files below and writes the macro
                                 file numbers_c170.tex, which is part of the Letter's TeX source


WHERE THE NEW NUMBERS OF THE LETTER COME FROM   (paths under results/; "#" points to a key)

  Width law ln(Gamma L) = a - b|lambda|^(-1/2) for -0.018 <= lambda <= -0.007: a = 5.915, b = 1.306, rms 0.06;
  effective sigma = b/(2 pi sqrt2/3) = 0.44
      C170/S1/C170_S1_summary.json#boxes.A.fits.window_lam_m0p018_m0p007.two_param
  In this window: level 0.3-0.9/L below the barrier top, WKB exponents 2-8
      C170/S1/C170_S1_summary.json#boxes.A.table (eps_minus_Utop, S_wkb)
  Zero-point slope 0.7-1.2 at r = 1-5; at lambda = -0.012, Delta V lowers it by 12-32%
      C170/S2/C170_S2_curves.json#slopes
  lambda_c = -0.0144, -0.0118, -0.0072 at N = 100, 300, 10^4 (P = 1/2, mu = 3.4e15 GeV); closed form within 1.1%
  for N = 10^2-10^4 and 3.2% at N = 10^5
      C170/S1/C170_S1_lambda_c_N.csv (box A)
  Model error +0.0003/-0.0002 of lambda_c(300)
      zero-mode-model row of Table I of the Letter (release v1.0.0:
      higgs/09_claims/expected/w/claims_run05/run01/table1_rebuild.json#composite) times 0.00659 per GeV
  Static threshold shallower than the hazard threshold at N = eta_inf by 0.0003-0.0004; widths of the release
  reproduced exactly, and to 10^-11 from generic starting points
      C170/C170_results.json#S0
  lambda_c(300) between -0.0128 and -0.0115 for 1/L = 3.4e12 ... 3.4e16 GeV; SM vacuum keeps P > 1/2 for N < 970
  (590 to 1.8e5 across the boxes); narrow resonance for lambda >= -0.028
      C170/S1/C170_S1_lambda_c_N.csv, C170/S1/C170_S1_lambdaSM_Nstar.csv, C170/S1/C170_S1_summary.json#boxes.A
  N_eff = 250 light-crossing times (0.83 eta_inf) for the expanding torus
      C170/S1/C170_S1_summary.json#boxes.A.N_eff
  Cross terms make lambda_c(300) shallower by 0.0008
      C170/C170_results.json#S2.kappa1_vs_final_model
  Gauge zero-mode strength kappa: threshold for N = 300 vanishes at kappa = 0.48; no narrow resonance below 0.43
      C170/C170_results.json#S2.boundaries.2D_Gdmcs, C170/S2/C170_S2_lambda_c_kappa.csv
  g2, gY, y_t scaled by 0.7-1.3: |lambda_c| moves by at most 22%, 6%, 18%
      C170/S3/C170_S3_coupling_scan.csv
  Abelian Higgs model: complex absorbing potential agrees with exterior complex scaling to 10^-6; local exponent
  1.8 -> 4.0 (WKB value 4.44); Born-Oppenheimer widths within 30%, with the diagonal term too large by factors of
  2.5 to more than 200; lambda_c(300) = -0.071 at e = 0.87; probability beyond |e w| = pi below 9e-4
      C171/C171_results.json (#table, #cap, #thresholds, #thresholds_general_e)

  Fig. 1: figures/fig_prl_fig1_mechanism.py, from the tables listed in figures/prepare_data_c170.py.


CONTENTS

  c170/        scripts of the static-box calculation. setup_from_release.py copies the unchanged production modules
               they import (higgs/zmlib, higgs/04_cross_terms, dnec_common of release v1.0.0) into c170_modules/ and
               verifies their SHA-256.
  c171/        self-contained solver of the abelian Higgs model (NumPy, SciPy).
  figures/     Fig. 1 script, prepare_data_c170.py (copies its input tables from results/ into data_c170/), the two
               style modules of the release, and the figure as built (out/prl_fig1_mechanism.pdf).
  numbers/     make_numbers_c170.py (see above).
  results/     archived outputs of C170 and C171: summaries, tables and check records. The point-by-point records (the
               resumable chains of C170; the per-task records and run logs of C171) are bundled in
               results/C170/C170_point_records.jsonl and results/C171/C171_point_records.jsonl.
  unbundle_records.py   restores the bundled records into results/ and verifies their SHA-256.


REPRODUCE   (Python 3.12 with NumPy, SciPy, SymPy with mpmath, Matplotlib and PyMuPDF; versions as in
            environment.yml of the release)

  Setup
    set DNEC_RELEASE=<folder of release v1.0.0>       (bash: export DNEC_RELEASE=...)
    python -B unbundle_records.py                     restores the point records into results/
    python -B c170/setup_from_release.py              copies and verifies the production modules

  A. From the archived point records (minutes). Copy results/C170/S1, S2 and S3 into rerun/C170/ and
     results/C171/raw into rerun/C171/raw, then
       cd c170 ; python -B c170_s0.py                 regression against the archived widths, about 1 min
       python -B c170_s2.py curves ; python -B c170_s2.py kzero ; python -B c170_s2.py scalecheck
       python -B c170_s2.py reg2dex
       python -B c170_post.py s1 ; python -B c170_post.py lamsm ; python -B c170_post.py s2 ; python -B c170_post.py s3
       python -B c170_final.py                        must report 30 of 30 checks passed
       cd ../c171 ; python -B c171_analyze.py

  B. Full recomputation (several CPU hours). As A, without copying the records, and with these steps before
     c170_post.py and c171_analyze.py:
       python -B c170_s1.py run 10 ; python -B c170_s1.py conv 6       static boxes and convergence test
       python -B c170_s2.py run 4                                      kappa = 0.25 ... 1.5
       python -B c170_s2.py refine 4 0.3,0.35,0.4,0.45,0.6
       python -B c170_s2.py refine2 4 0.425,0.475
       python -B c170_s3.py run 8                                      coupling scans
       cd ../c171 ; python -B c171_scan.py 4 ; python -B c171_cap.py 4  (about 7 min)
     The chain root at kappa = 0.45, lambda = 0 is a spurious narrow state. The archived value comes from a re-seed at
     eps_1D + 0.004 (results/C170/S2/extra_k0.45_lam0_reseed.json; README_C170.md, Sec. 2).

  C. Fig. 1 and the number macros
       cd figures ; python -B prepare_data_c170.py ; python -B fig_prl_fig1_mechanism.py
       cd ../numbers ; python -B make_numbers_c170.py   (reads Table I of the release through DNEC_RELEASE)
     The figure is written to figures/out/ and is byte-identical to Fig. 1 of the Letter. The macro values are those
     of numbers_c170.tex in the Letter's source.

  New outputs go to ./rerun/C170 and ./rerun/C171 (or set C170_OUT, C171_OUT) and are compared with results/
  (results/C170/checks.json lists the tolerances). The chains resume from their JSON files, so a long run can be
  interrupted and restarted.


INTEGRITY AND LICENSES

  SHA256SUMS.txt lists every file. Code: MIT. Data, figures and text: CC BY 4.0. The production modules copied by
  setup_from_release.py keep the licenses of the release (MIT, except higgs/zmlib/_smdr_betas.py, GPL-2.0-or-later,
  derived from SMDR 1.3); see LICENSE.txt.
