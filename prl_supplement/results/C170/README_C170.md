# C170: static finite-volume box, gauge zero-mode switch and coupling scans

Supplemental Material to A. Zhang, *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins*
(2026). Computed 2026-10-19: boxes A-E (mu_b, mu_b/10, mu_b/100, 10 mu_b, mu_b/1000), lambda_SM and N* per box, the
lambda_SM(mu) grid, the kappa scan and the coupling scans. Machine-readable results: `C170_results.json`; checks:
`checks.json` (30/30 pass).
Labels: run02 and run05 are model versions of the code release v1.0.0 (run05, the final model, is used throughout
the Letter); C1xx are modules and records of the release (C138: finite-volume matching, whose variant 'MC' is used
here and is unrelated to the Monte Carlo top mass; C149: cross terms; C150: two-dimensional model). Other short
labels (D1xx, U156, T14a/T14b, CCA, v11) refer to derivation notes, runs and model components of the release that
the technical report (Zenodo 10.5281/zenodo.23434140, cited here as 'report') documents.

## 1. Physics

A static flat three-torus of fixed edge L (Minkowski time, no gravity, no turning point) with antiperiodic fermions
and Standard Model couplings at mu = 1/L is held for a time T = N L. The Higgs zero mode is a finite-volume
resonance with complex energy E = eps - i Gamma/2 (units 1/L) and survives with P = exp(-Gamma N). The static
threshold lambda_c(N; P) is defined by Gamma(lambda_c) N = -ln P at fixed (g2, gY, y_t, g3) of the box.

Model: the final production model of the release (run05), i.e. the 2D (r, |w|) model with the conditional-Gaussian
SU(2) x U(1) zero modes, the DMC shape correction, the one-loop MC finite-volume matching at mu = 1/L and the complete
cross-term set (CCA one-loop Hessians, SU(2)/U(1) variational re-minimisation, exact light-fermion twist,
SU(3)-top variational term, covariant vev mixing, Coulomb mismatch, exact top chiral bubble).

Boxes (couplings of (m_t, alpha_s, M_h) = (172.60, 0.1180, 125.13) at mu, `sm_provider` 'central' = production NNLO
mapping; mu_b = M-bar_P/722 = 3.3726e15 GeV):

| box | mu [GeV] | g2 | gY | y_t | g3 | lambda_SM(172.2 / 172.6 / 173.0) |
|---|---|---|---|---|---|---|
| D | 3.373e16 (10 mu_b) | 0.521167 | 0.451741 | 0.413286 | 0.520546 | -0.007656 / -0.010278 / -0.012916 |
| A | 3.373e15 (mu_b, node u = 0) | 0.527759 | 0.442770 | 0.427827 | 0.535654 | -0.007157 / -0.009774 / -0.012409 |
| B | 3.373e14 (mu_b/10) | 0.534605 | 0.434312 | 0.443443 | 0.552165 | -0.006268 / -0.008869 / -0.011489 |
| C | 3.373e13 (mu_b/100) | 0.541721 | 0.426319 | 0.460306 | 0.570313 | -0.004913 / -0.007481 / -0.010070 |
| E | 3.373e12 (mu_b/1000) | 0.549124 | 0.418750 | 0.478624 | 0.590388 | -0.002992 / -0.005510 / -0.008052 |

## 2. Method

**Code.** The production modules were copied unchanged (byte-identical; `c170/setup_from_release.py` verifies their SHA-256) from
the code release v1.0.0 into `c170_modules/` by `c170/setup_from_release.py` (`higgs/zmlib/*`,
`higgs/04_cross_terms/C149_direct4.py, C149_direct5.py, C149_coulomb.py, C149_top.py`, `dnec_common/`). Archived inputs
were copied (never modified) into the work tree `work/` (list and SHA-256 in `S0/C170_S0.json#stage_record`).
Nothing was written to or run inside the release folder. All runs use `python -B` and single-threaded BLAS, with at most 14
worker processes (later at most 11, while the C171 scan shared the machine).

**Production functions used unchanged.** `C150_twod.compute_profile` (D108 conditional Gaussian, recomputed for every
pair of zero-mode couplings), `Parts`, `build2d`, `DMC`, `rgrid`, `SGrid`, `phases`, `window_roots`, `muller2`,
`solve_res`/`frob`; `C149_direct5.cross_increments5` (installed by the module as `C149_direct.cross_increments`),
`C149_direct.V_light_c`, `make_funW`; `C138_core.dV('MC', ...)`; `R14_res.jost`, `gauss_profile`; `cross_var.newton`.
Numerics: production FAST configuration (N, Smax, h, Rout, WKB order) = (24, 12, 0.004, 16, 2), run05 window
(0.02, 0.0015), retry with the D112 wide window (0.05, 0.0025) when the window rule flags.

**lambda-override path (new, `c170_core.py`).** The production node job `C149_direct.jobX` looks the couplings up from
(m_t, alpha_s, u). `solve_point` builds exactly the same column from explicit (g2, gY, y_t, g3, lambda): profile from
(g2, gY), `build2d(..., '2D_ex')`, `cross_increments5(c, g3)` with c = (g2, gY, y_t, lambda) (so every lambda dependence
of the matching, the CCA Hessians, the vev mixing and the Coulomb pieces follows the scanned lambda), light-fermion twist,
phases, root function and window rule as in `jobX`. S0 shows that this path reproduces the archived widths bit for bit.

**New pieces (diagnostics and seeds, not production physics).**
- Effective radial potential U_eff(r) = U1(r) + lowest eigenvalue of the s-channel Hamiltonian (sine DVR, including
  the light-fermion twist W_L(r, s)); for the quadratic channel this is the C150/T14a definition U1 + (3/2) Om_eff.
  Recorded per point: well (U_well, r_well), barrier top (U_top, r_top), eps - U_top, WKB exponent S across the barrier.
- 1D adiabatic Siegert root of U_eff (R14 Numerov + 2nd-order WKB outgoing condition): seed and consistency check.
  The 2D seed is eps_1D + dQ with (dQ, rho) = (eps_2D - eps_1D, Gamma_2D/Gamma_1D) of the neighbouring solved point
  (continuation in lambda); box A starts from the archived run05 root.
- Box check (C150 rule; `jobX` has none): the root is re-converged with (N, Smax) = (28, 14), the light-fermion term
  rebuilt on the new s grid; flag if |dE| > max(2e-7, 2e-3 Gamma).
- Classification of each point (`c170_core.classify`): `ok` (no flag, eps < U_top - Gamma/2); `ok_broad` (only window-rule
  flags, which appear once Gamma is comparable to the window, and the root passes the box check and agrees with the 1D
  adiabatic root, |ln Gamma_2D/Gamma_1D| < 0.25, |dQ| < 0.02); `above_top` (eps >= U_top - Gamma/2: no narrow
  resonance); `no_barrier` (U_eff has no local maximum after the well); `no_level` (no real-axis zero of the 1D Jost
  function below U_top + 0.3); `flagged` (anything else, including roots inconsistent with the adiabatic level:
  |ln Gamma_2D/Gamma_1D| > 0.5 or |dQ| > 0.03). Only `ok`/`ok_broad` points enter interpolations ("narrow resonance").

**Changes made during the run (recorded).**
1. Seeding: a blind 1D scan picked the deepest |f| dip, which at |lambda|^(-1/2) = 18 (deep well, two quasi-bound
   levels) was the excited level; the scan now takes the lowest dip, and chains continue from the previous 1D root.
   The two affected points were recomputed (the old records are not included).
2. Newton start for scaled couplings (S3, and as a guarded fallback everywhere): `cross_increments5` re-minimises the
   Gaussian functional by an undamped Newton iteration started from the fixed-coupling v11 profile; for g2 or gY scaled by
   0.85-1.3 this diverges (non-finite increments). Fallback 1 starts the same Newton iteration at the exact optimum of the
   same functional for these couplings (`R14_res.gauss_profile`, identical rf grid); for the production couplings this
   changes U1 and Om_eff^2 by <= 8e-10. It was used for the 66 S3 points with scaled g2 or gY (Newton gradient
   <= 1.3e-8 everywhere); the U156/D128 Levenberg Newton (fallback 2) was never needed.
3. S2 numerics: for kappa < 1 the s box is scaled with the zero-mode length kappa^(-1/3):
   (N, Smax) = (24, 12) kappa^(-1/3), box check (28, 14) kappa^(-1/3); FAST otherwise.
4. kappa = 0.45, lambda = 0: the chain root was a spurious narrow state (Gamma = 7.2e-6, 2500 times narrower than the
   adiabatic level, identical in both S2 models); re-seeded at eps_1D + 0.004 (window (0.006, 0.0005)) the resonance is
   Gamma = 0.0160 (2D_Gdmcs) / 0.0204 (2D_G). The consistency rule above was added to `classify` for this reason.

**Thresholds.** ln Gamma is interpolated with a cubic spline against x = |lambda|^(-1/2) through the narrow points of a
box (S1, S3), or against lambda (S2, where lambda_c approaches 0). Outside the computed range: linear extrapolation
in x from the last three points, flagged `extrapolated`; a target width larger than every narrow width is flagged
`beyond_narrow` (the level reaches the barrier top first; no valid threshold). Fits: ln Gamma = a - b x by least
squares over the narrow points (whole range and the window lambda in [-0.018, -0.007]) and ln Gamma = a - b |lambda|^(-p).

## 3. Checks (all in `checks.json`, 30/30 pass)

| check | result | reference / tolerance |
|---|---|---|
| S0: override path, archived couplings, production seed, u = 0 (172.6, 173.0) and u = 5 (172.6) | Gamma and eps identical (rel. diff. 0) | <= 1e-8 |
| S0: same with this work's generic seed | rel. diff. 6.9e-12, 9e-15 | <= 1e-8 |
| S0: `sm_provider` couplings at mu_b vs archived node couplings | identical | 1e-15 |
| 2D_ex path (no cross terms) vs archived T14b run02 root | identical | 1e-8 |
| box A at lambda_SM(172.6) by continuation vs archive (7.147016e-4) | rel. 1.7e-11 | 1e-8 |
| box check (28, 14), all narrow S1 points | max rel. change of Gamma 1.6e-6 | 1e-5 |
| numerical convergence, box A, x = 7.5 ... 18 (Gamma 1.8e-2 ... 9.2e-9): BASE (32, 14, h = 0.002) / Rout = 14 / (32, 16) | max rel. 6.2e-6 / 5.9e-6 / 5.4e-7 | 1e-3 |
| 2D vs 1D adiabatic width, all narrow S1 points | Gamma_2D/Gamma_1D = 1.004 ... 1.022 | report: 1.010-1.018 |
| exact zero-mode scaling without matching, Gamma(kappa, kappa^2 lambda) = kappa^(2/3) Gamma(1, lambda), kappa = 0.5, 1.5 | rel. 2.7e-6, 1.4e-6 (eps 2e-9) | 1e-4 |
| kappa = 1 slope reduction by dV_MC at r = 1-5 (box A couplings, lambda = -0.012) | 12-32% | report: 10-33% |

The BASE runs carry `box_sensitive` flags for Gamma < 1e-4 because the absolute tolerance 2e-7 is tighter than the
h-shift of eps (3e-7); the widths are unaffected (6e-6).

## 4. Results

### S0 regression
With the archived run05 couplings and seeds the lambda-override path returns the archived widths exactly
(u = 0: Gamma = 7.147016e-4 and 3.113316e-3 for m_t = 172.6 and 173.0; u = 5: 6.325290e-5).
Static threshold of the archived u = 0 widths for N = eta_inf = 299.84 (spline of ln gamma against |lambda|^(-1/2)):

| P | static lambda_c | hazard lambda_c (run05) | static - hazard | I/(Gamma(0) eta_inf) at the hazard threshold |
|---|---|---|---|---|
| 0.9 | -0.008829 | -0.009134 | +0.000305 | 0.785 |
| 0.5 | -0.011779 | -0.012142 | +0.000362 | 0.840 |
| 0.1 | -0.014769 | -0.015185 | +0.000416 | 0.877 |

The static threshold is 0.00030-0.00042 shallower, as stated in the PRL (run02: 0.00030/0.00036/0.00042).

### S1 static boxes (final model)
Points: A, B, C 36 each (33/33/34 narrow), D 18 (16), E 17 (16); |lambda|^(-1/2) from 5 to 18 (E: 19), dense
(step 0.25) for lambda in [-0.018, -0.007]; every box also contains lambda_SM for the three top masses.

Static thresholds lambda_c(N; P = 1/2) (all interpolated):

| N | D (10 mu_b) | A (mu_b) | B (mu_b/10) | C (mu_b/100) | E (mu_b/1000) |
|---|---|---|---|---|---|
| 10 | -0.02621 | -0.02688 | -0.02761 | -0.02841 | -0.02929 |
| 30 | -0.01847 | -0.01896 | -0.01949 | -0.02007 | -0.02070 |
| 100 | -0.01401 | -0.01438 | -0.01478 | -0.01522 | -0.01569 |
| 300 | -0.01145 | -0.01175 | -0.01207 | -0.01242 | -0.01279 |
| 1e3 | -0.00949 | -0.00973 | -0.00999 | -0.01026 | -0.01055 |
| 1e4 | -0.00702 | -0.00718 | -0.00735 | -0.00753 | -0.00771 |
| 1e5 | -0.00545 | -0.00556 | -0.00567 | -0.00578 | -0.00588 |
| 1e6 | -0.00437 | -0.00444 | -0.00451 | -0.00457 | -0.00461 |

P = 0.9 and 0.1 for all N are in `S1/C170_S1_lambda_c_N.csv`. At N = 300: P = 0.9 / 0.1 give -0.00862 / -0.01430 (D),
-0.00884 / -0.01468 (A), -0.00907 / -0.01509 (B), -0.00931 / -0.01554 (C), -0.00956 / -0.01602 (E). Not valid
(`beyond_narrow`): N = 10, P = 0.1 (Gamma = 0.23) in every box and N = 30, P = 0.1 (Gamma = 0.077) in A, B, E.

Where the narrow resonance ceases: the last narrow point is lambda = -0.0278 (A, B, D) or -0.0302 (C, E), with
Gamma = 0.070-0.081 and S_WKB = 0.33-0.52; at the next point (-0.0302 resp. -0.0331) the level is within Gamma/2 of
the adiabatic barrier top (Gamma = 0.087-0.100, S_WKB <= 0.26), and for more negative lambda it lies at or above the top. The window rule of
`jobX` needs the wide retry and the 1D verification from Gamma ~ 0.02 on (`ok_broad`).

Fits of ln Gamma = a - b|lambda|^(-1/2):

| box | window lambda in [-0.018, -0.007]: a, b (rms) | all narrow points: a, b (rms) | a - b|lambda|^(-p): p |
|---|---|---|---|
| D | 5.801, 1.278 (0.070) | 6.294, 1.350 (0.31) | 0.701 +- 0.017 |
| A | 5.915, 1.306 (0.059) | 6.251, 1.353 (0.23) | 0.669 +- 0.012 |
| B | 5.879, 1.319 (0.051) | 6.112, 1.354 (0.19) | 0.635 +- 0.012 |
| C | 5.887, 1.338 (0.046) | 5.797, 1.341 (0.19) | 0.614 +- 0.014 |
| E | 5.734, 1.341 (0.043) | 5.309, 1.315 (0.16) | 0.545 +- 0.020 |

(Report, nodal widths at tau = 0: a = 5.91, b = 1.307, p = 0.74.) Over the whole range the local exponent flattens
towards the barrier top, hence p > 1/2 (0.55-0.70, depending on the range); |lambda_c|^(-1/2) grows linearly with ln N with
slope ~1/b (box A, P = 1/2: x_c = 6.10 at N = 10, 15.0 at N = 1e6).

Effective static duration of box A that reproduces the run05 hazard thresholds:
N_eff = -ln P / Gamma_A(lambda_c^hazard) = 236.6 / 247.9 / 255.2, i.e. N_eff/eta_inf = 0.789 / 0.827 / 0.851 at
P = 0.9 / 0.5 / 0.1 (report: 0.77 / 0.82 / 0.87). The u = 0 widths at the critical m_t (own couplings) give
0.785 / 0.840 / 0.877; the difference is the coupling mismatch of box A (couplings of 172.6), which changes Gamma at
fixed lambda by -2.1% (m_t = 172.2) and +1.7% (m_t = 173.0).

SM vacuum in each box (box couplings of m_t = 172.6, Gamma computed at lambda_SM; N* = ln2/Gamma):

| box | N* (m_t = 172.2) | N* (172.6) | N* (173.0) |
|---|---|---|---|
| D | 4.91e3 | 587 | 152 |
| A | 1.03e4 | 970 | 219 |
| B | 3.95e4 | 2.32e3 | 404 |
| C | 4.76e5 | 1.05e4 | 1.14e3 |
| E | 9.79e7 | 1.81e5 | 7.07e3 |

Widths and eps in `S1/C170_S1_lambdaSM_Nstar.csv`; interpolation from the grid without the point agrees to
<= 3e-4 relative. lambda_SM(mu) from 1e10 GeV to M-bar_P for the three masses: `S1/C170_lambdaSM_running.csv`
(zero crossings 2.56e11 / 5.18e10 / 1.40e10 GeV, minima -0.00783 / -0.01045 / -0.01307 near 4e17 GeV).

### S2 gauge zero-mode strength kappa (box A couplings, 2D model without cross terms)
kappa multiplies g2 and gY only in the zero-mode Hamiltonian (conditional Gaussian: kinetic, magnetic and Higgs-mass
terms of the gauge zero modes, hence C0, B_x, Om_eff^2 and the hypercharge oscillator); dV_MC stays at the physical
couplings.

*DMC shape.* Kept, transported exactly: in the body frame the zero-mode Hamiltonian at couplings kappa (g2, gY) (gauge
kinetic, magnetic, mass and angular terms, hypercharge oscillator) is unitarily equivalent to kappa^(2/3) times the
kappa = 1 Hamiltonian at r' = kappa^(1/3) r (x -> kappa^(-1/3) x), so dE_DMC(kappa; r) = kappa^(2/3) dE_DMC(1; kappa^(1/3) r)
(model `2D_Gdmcs`, main; identical to the production 2D_ex at kappa = 1). Keeping dE_DMC unscaled would attach the
kappa = 1 correction to a different Hamiltonian (it would even survive at kappa = 0). Dropping it (`2D_G`) gives the
strict variational bracket. The scaling is verified numerically (checks).

| kappa | lambda_c(300, 1/2) 2D_Gdmcs / 2D_G | Gamma(lambda = -0.012) 2D_Gdmcs / 2D_G | narrow range of lambda (2D_Gdmcs) |
|---|---|---|---|
| 0 | none: no barrier | none | none |
| 0.25, 0.30 | none: no barrier for any lambda <= 0 | none | none |
| 0.35, 0.40, 0.425 | none: level above the barrier top for all lambda <= 0 | none | none |
| 0.45 | none (only a broad resonance at lambda = 0, Gamma = 0.016 / 0.020) | none | lambda = 0 |
| 0.475 | none: Gamma(0) = 0.0030 / 0.0042 > ln2/300 | none | [-0.0005, 0] |
| 0.50 | -0.00022 / -0.00017 | none (level above top) | [-0.001, 0] |
| 0.60 | -0.00160 / -0.00149 | none | [-0.004, -0.0005] |
| 0.75 | -0.00472 / -0.00450 | none (level 0.03 below the top, Gamma = 0.063: not narrow) | [-0.010, -0.002] |
| 1.00 | -0.01250 / -0.01204 | 1.81e-3 / 2.27e-3 | [-0.030, -0.006] |
| 1.25 | -0.02352 / -0.02271 | 1.14e-5 / 1.52e-5 | [-0.040, -0.012] |
| 1.50 | -0.03784 / -0.03660 | 3.59e-8 / 4.99e-8 | [-0.040, -0.012] |

- kappa = 0: U(r) = 3/(8r^2) + lambda r^4/4 + dV_MC(r) (gauge zero modes free; C0 = B_x = Om_eff = dE_DMC = 0) has
  dU/dr <= -0.29 everywhere on (0.02, 16] for every lambda in [-0.04, -1e-4]: no barrier, no resonance
  (`S2/C170_S2_kappa0.json`). Without dV_MC it is monotonic as well.
- kappa_min (no narrow resonance at any lambda <= 0 below it): the adiabatic level at lambda = 0 lies 0.011 / 0.021
  above the barrier top at kappa = 0.425 and 0.047 / 0.033 below it at kappa = 0.45, so kappa_min = 0.43 (linear
  interpolation: 0.430 / 0.435 for 2D_Gdmcs / 2D_G).
- The threshold for N = 300, P = 1/2 vanishes (lambda_c -> 0-) at kappa_thr = 0.478 / 0.482 (Gamma(kappa, lambda = 0)
  = ln2/300, interpolated in ln Gamma between kappa = 0.475 and 0.5).
- Local slopes at lambda = -0.012 (`S2/C170_S2_curves.json`): the matching lowers the zero-point slope (including the
  transported DMC term) by 12/18/23/28/32% at r = 1/2/3/4/5 for kappa = 1, by 27-67% for kappa = 0.5, and by 58% (r = 1)
  to 141% (r = 5) for kappa = 0.25, where the net slope is already negative beyond r = 2.7. At kappa = 0.5 the barrier
  above the well is 0.059 with dV_MC and 0.386 without it. A narrow resonance needs the zero-point slope to exceed the
  destabilising slope of dV_MC by enough to hold a level below the barrier top; at lambda = 0 this requires kappa >= 0.43.
- kappa = 1 against the full model (S1 box A: lambda_c(300, 1/2) = -0.01175, Gamma(-0.012) = 2.61e-3): 2D_ex (= 2D_Gdmcs
  at kappa = 1) -0.01250 (Delta = -0.00075, Gamma ratio 0.69); 2D_G -0.01204 (Delta = -0.00029, ratio 0.87). The cross
  terms are net destabilising and the DMC shape stabilising, as in the hazard analysis.
- Potentials for the figure (kappa = 0, 0.5, 1 at lambda = -0.012, with and without dV_MC, and the parts):
  `S2/C170_S2_potential_lam-0.012.csv`.

### S3 coupling scans (final model, box A, N = 300)
One of g2, gY, y_t multiplied by s in all sectors (profile, matching, cross terms, SU(3)-top term through m_t); g3 and
lambda unscaled. 12 narrow points per variant (x = 7.0 ... 12.5); all thresholds interpolated.

| coupling | s = 0.7 | 0.85 | 1 | 1.15 | 1.3 | d ln|lambda_c|/d ln g (0.85/1.15; 0.7/1.3) at P = 0.9 / 0.5 / 0.1 | D128 hazard (P1 = 1/2) |
|---|---|---|---|---|---|---|---|
| g2 | -0.009187 | -0.010501 | -0.011751 | -0.012963 | -0.014184 | 0.709 / 0.697 / 0.695; 0.718 / 0.702 / 0.698 | 0.66 |
| gY | -0.011104 | -0.011439 | -0.011751 | -0.012044 | -0.012324 | 0.175 / 0.170 / 0.171; 0.174 / 0.168 / 0.169 | 0.16 |
| y_t | -0.009688 | -0.010689 | -0.011751 | -0.012794 | -0.013711 | 0.552 / 0.595 / 0.608; 0.513 / 0.561 / 0.577 | 0.63 (0.575 / 0.627 / 0.649) |

(lambda_c at P = 1/2 shown; P = 0.9 and 0.1 in `S3/C170_S3_coupling_scan.csv`.) The static derivatives agree with the
hazard derivatives of D128 (earlier model, +-10%, whole turn) to within about 6%: g2 and gY slightly larger, y_t
slightly smaller; increasing any of the three couplings makes lambda_c more negative. Over +-30% the response is close to a power
law (curvature of ln|lambda_c| in ln g: g2 0.04-0.19, gY 0.05-0.12, y_t -0.11 ... +0.33).

## 5. Status of the numbers
- Computed (Siegert roots): every Gamma, eps in the tables of `S1/C170_S1_gamma_lambda.csv`, `S2/chains`, `S3/chains`,
  Gamma(lambda_SM), Gamma(lambda = -0.012) in S2.
- Interpolated (cubic spline of ln Gamma): every lambda_c quoted above, N_eff, kappa_thr (linear in ln Gamma between
  kappa = 0.475 and 0.5), kappa_min (linear in eps_1D - U_top between kappa = 0.425 and 0.45).
- Extrapolated / invalid: only the `beyond_narrow` entries listed in S1 (N = 10 and 30 at P = 0.1); no extrapolated
  lambda_c is quoted in this README.
- Derived from the archive (S0 static thresholds, I/(Gamma(0) eta_inf)): interpolation of archived widths.

## 6. Limitations
- All results are those of the reduced zero-mode model (conditional Gaussian + 2D (r, |w|) + DMC shape + one-loop
  matching + cross terms) with the stated couplings; the static box keeps the couplings of m_t = 172.6 for every lambda
  (lambda is varied freely), so Gamma(lambda_SM) for m_t = 172.2 and 173.0 carries the coupling mismatch of +-2% quoted
  in S1. The DMC shape correction keeps its fixed (v11) couplings, as in production.
- Narrow resonances exist for lambda >~ -0.03; thresholds needing Gamma > 0.08/L are not defined in this framework.
- S2 is a diagnostic switch: at kappa != 1 the split between zero modes (kappa) and non-zero modes (physical couplings)
  is no longer the one derived from a single Lagrangian; the cross terms are omitted there (kappa scaling ambiguous).
  kappa_min and kappa_thr are resolved to +-0.01.
- S3 uses the fallback Newton start for scaled g2, gY (equivalent to production to 1e-9 where production converges).
- The `ok_broad` classification relies on the agreement with the 1D adiabatic root (2D/1D = 1.00-1.02 in the final
  model); the S2 models without cross terms have 2D/1D = 1.03-1.18.

## 7. Files and how to rerun
Code (`c170/` of this supplement, run from that directory with `python -B`):
`c170_core.py` (paths, staging, override path, diagnostics, chains), `c170_s0.py`, `c170_s1.py run|conv`,
`c170_s2.py run|refine|refine2|curves|kzero|scalecheck|reg2dex`, `c170_s3.py run [nw] first|second`,
`c170_post.py s1|lamsm|s2|s3`, `c170_final.py`. Chains resume from their JSON files.
Results (this directory): `C170_results.json`, `checks.json`; CSV tables
`S1/C170_S1_gamma_lambda.csv` (Gamma(lambda) per box with status, barrier top, WKB exponent, 1D width, box check),
`S1/C170_S1_lambda_c_N.csv` (lambda_c(N; P) per box, i.e. lambda_c(mu; N)), `S1/C170_S1_lambdaSM_Nstar.csv`,
`S1/C170_lambdaSM_running.csv`, `S2/C170_S2_lambda_c_kappa.csv`, `S2/C170_S2_potential_lam-0.012.csv` and
`S3/C170_S3_coupling_scan.csv` (`c170_final.py` also writes copies of these seven tables, with descriptive names, to
the top of its output folder); per-step JSON in `S0/`, `S1/`, `S2/`, `S3/`. The chains with every point record
(`S1/chains/`, `S2/chains/`, `S3/chains/`) are bundled in `C170_point_records.jsonl`; `python -B unbundle_records.py`
in the supplement root restores them. Sum of single-process point times: S1 3150 s, S2 6980 s, S3 4310 s, S0 58 s;
wall time of the whole calculation about 1.5 h with 6-10 workers.
