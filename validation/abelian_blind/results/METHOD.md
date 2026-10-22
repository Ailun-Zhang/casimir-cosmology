# Blind calculation C171: resonance of the abelian Higgs zero-mode problem

All numbers in this folder were computed from the specification in `BRIEF_B1_blind_abelian.md` with code written for this check (folder `code/`). Units: L = 1.

## 1. Problem and formulation

    H = -1/2 (d_r^2 + r^-1 d_r) - 1/2 (d_s^2 + 2 s^-1 d_s) + e^2 r^2 s^2 / 2 + lam r^4 / 4,   measure r dr s^2 ds.

* **Unknown.** Both methods use `Phi(r,s) = s psi(r,s)`. In s this turns the 3D s-wave operator into `-1/2 d_s^2` with the Dirichlet condition `Phi(r,0) = 0` (Phi is odd in s). In r the 2D (m = 0) operator `-1/2(psi'' + psi'/r)` is kept as it stands: a regular m = 0 solution is an analytic function of r^2, so psi is even in r. I did not use `u = sqrt(r) psi`, because that substitution adds a `-1/(8 r^2)` potential and a non-analytic `sqrt(r)` behaviour at the origin, which would spoil high-order convergence. Method A imposes regularity through even mirror values; Method B builds it into the basis (2D oscillator functions of r^2). The norm is `int r dr ds |Phi|^2`.
* **Resonance.** The Siegert state has `E = eps - i Gamma/2` and is outgoing at large r. Only r needs complex scaling. At fixed r, s is an oscillator of frequency `e r`. At fixed s, r is confined by `e^2 s^2 r^2/2`, which gives an effective potential `e s` along the s valley. The only open direction is therefore r -> infinity at small s, where `lam r^4/4 -> -infinity`. The outgoing wave behaves like `exp(i sqrt(|lam|/2) r^3/3)`, so a rotation `r -> r e^{i theta}` makes it decay for `0 < theta < pi/3`.
* **Flux identity.** For the exact Siegert state on real coordinates, `psi* H psi - psi (H psi)* = -i Gamma |psi|^2` gives `div j = Gamma |psi|^2` with `j = Im(psi* grad psi)`. Hence, for every R on the real (unscaled) part of the r axis,

      Gamma = J(R) / N(R),   J(R) = R int ds Im(Phi* d_r Phi),   N(R) = int_0^R r dr int ds |Phi|^2.

  This holds with no small-Gamma approximation. For narrow states N(R) is the norm inside the barrier.

## 2. Method B: oscillator product basis + complex rotation (eigenvalue)

* **r basis.** `phi_n(r) = sqrt(2 beta) L_n(beta r^2) exp(-beta r^2/2)`, n = 0..Nr-1. These are the 2D isotropic oscillator m = 0 states and are orthonormal with weight r dr. **s basis.** For Phi = s psi, the odd 1D oscillator functions `h_{2k+1}(sqrt(gamma) s)`, k = 0..Ns-1, which are the 3D s-wave oscillator states.
* **Matrix elements.** All are exact and banded. With `x = beta r^2`: `<n|x|n> = 2n+1` and `<n+1|x|n> = -(n+1)`. `r^4 = x^2/beta^2` is the matrix product formed in an (Nr+2)-dimensional space and then truncated, which is exact. `T_r = beta[(n+1/2) delta_{n,m} + (n+1)/2 delta_{|n-m|,1}]`. `s^2` and `T_s` follow from the ladder formulas with n = 2k+1.
* **Rotation.** `r -> r e^{i theta}`, and optionally `s -> s e^{i phi}`:
  `H_theta = e^{-2i theta} T_r + e^{-2i phi} T_s + (e^2/2) e^{2i(theta+phi)} r^2 s^2 + (lam/4) e^{4i theta} r^4`.
  This matrix is complex symmetric and its spectrum is entirely discrete. The resonance is the lowest eigenvalue. A search around E = 0 for theta = pi/8 and pi/5 finds nothing lower, and the eigenvalue does not depend on theta.
* **Eigenvalue.** The eigenvector comes from ARPACK shift-invert plus two inverse-iteration steps. The eigenvalue is then taken as the **c-product Rayleigh quotient** `x^T H x / x^T x`, evaluated in 40-digit arithmetic (mpmath, with the matrix entries also generated in mpmath). For a complex-symmetric H the error of this quotient is quadratic in the eigenvector error. In plain double precision, Im E has an absolute noise floor of about 1e-16. For lam = -0.01003 that floor appears as a 1.1e-2 relative scatter of Gamma across the basis and angle variants; with the 40-digit quotient the scatter falls to 3e-11.
* **Variations.** Nr x Ns from 60x30 to 150x70 (and 200x90, 250x100 for lam = -0.01003); theta in {pi/12, pi/8, pi/6}; phi = -pi/16; (beta, gamma) in {(1,1), (0.7,0.7), (1.5,1.5), (0.7,1.5)}.

## 3. Method A: smooth exterior complex scaling on a 2D finite-difference grid (eigenvalue and flux formula)

* **Grid.** `r_i = i h` (i = 0..Nr-1, including r = 0) and `s_j = j h` (j = 1..Ns). Derivatives use central 2p-order finite differences: p = 4 (8th order) by default, with p = 5 as a check. Mirror values are even in r and odd in s, and the outer edges are zero-padded. At r = 0 the radial operator is `2 psi''(0)`. A mask keeps only the two potential valleys: a point is kept if `s <= sqrt(2c/(e max(r,1)))` or `r <= sqrt(2c/(e max(s,1)))`, with c = 50. Taking c = 70 together with S_max + 2 changes nothing at the printed precision.
* **ECS contour.** `R(r) = r + (e^{i theta} - 1) w Sw((r - R0)/w)`, with the C-infinity switch `sw(t) = [1 + erf(a(t - 1/2))]/2`, a = 6, and w = 2 (w = 1.5 for lam = -1). In mapped form the radial operator is `A psi_rr + B psi_r` with `A = 1/R'^2` and `B = -R''/R'^3 + 1/(R R')`, and the potential is evaluated at complex R.
* **Eigenpair.** SuperLU factorisation of `H - sigma`, with a real `sigma = 1.9 e^{2/3}`, followed by inverse iteration from a real start vector until `|dE| < 1e-15 |E|`, then a Rayleigh quotient. Residuals are about 1e-12.
* **Flux formula.** `d_r Phi` uses the same 8th-order stencil. The s integral uses the trapezoid rule, which is spectrally accurate here because the integrand is even in s. The `r dr` integral uses the trapezoid rule with Euler-Maclaurin end corrections up to O(h^6). At r = 0 the derivatives come from the even extension: `g'(0) = f(0)`, `g'''(0) = 3 f''(0)`, `g^(5)(0) = 5 f''''(0)`, where `g = r f`. Gamma_flux is evaluated at two radii beyond the outer turning point of `V_BO = (3/2) e r + lam r^4/4` at E = eps (R1, R2 in the tables). For lam = -1 there is no turning point, and the radii lie beyond the BO barrier maximum at r = 1.14. The dependence of Gamma_flux on R is <= 4e-9 relative at h = 0.05 and <= 1e-10 at h = 0.03.
* **Parameters** (R0, flux radii, theta, R_max; S_max = 10, or 11 for e = 0.873):

  | case | R0 | flux radii | theta | R_max |
  |---|---|---|---|---|
  | lam = -1 | 2.6 | 1.6, 2.0 | 0.6 | 6.1 |
  | lam = -0.11174 | 4.6 | 3.6, 4.0 | 0.5 | 9.1 |
  | lam = -0.05 | 6.0 | 5.0, 5.4 | 0.5 | 10.5 |
  | lam = -0.03 | 6.8 | 5.8, 6.2 | 0.5 | 11.3 |
  | lam = -0.02 | 8.2 | 7.0, 7.5 | 0.5 | 12.7 |
  | lam = -0.01003 | 10.0 | 9.0, 9.4 | 0.5 | 14.5 |
  | e = 0.873, lam = -0.02286 | 7.1 | 6.0, 6.5 | 0.5 | 11.6 |

* **Variations.** h = 0.05, 0.04, 0.03 (for lam = -1 also 0.025 and 0.02); p = 5; theta - 0.15 together with a longer box; R0 + 0.7 (R0 + 0.4 for lam = -1); S_max + 2 with mask c = 70.

## 4. Results

eps and Gamma are the Method B values. The relative error column is the conservative estimate max(Method B internal spread, |A - B|). The A - B differences come from two independent discretisations and absorbing schemes.

| e | lambda | eps | Gamma | rel. err eps | rel. err Gamma | A-B: d eps | A-B: dGamma/Gamma (flux) |
|---|---|---|---|---|---|---|---|
| 1 | -1 | 1.66884204405 | 0.6122882663 | 4e-11 | 6e-11 | 6e-11 | 6e-11 |
| 1 | -0.11174 | 1.850650978201 | 6.116424540e-3 | 1e-12 | 2e-10 | 2e-12 | 2e-10 |
| 1 | -0.05 (optional) | 1.882149226058 | 4.759199383e-5 | 1e-12 | 1e-10 | 2e-12 | 9e-11 |
| 1 | -0.03 | 1.890727536056 | 4.565352370e-7 | 1e-12 | 3e-10 | 2e-12 | 3e-10 |
| 1 | -0.02 (optional) | 1.894788859933 | 3.709882227e-9 | 1e-12 | 6e-10 | 2e-12 | 6e-10 |
| 1 | -0.01003 | 1.898718938866 | 5.097168607e-14 | 1e-12 | 2e-9 | 2e-12 | 2e-9 |
| 0.873 | -0.03 x 0.873^2 = -0.022863870 | 1.727050797782 | 4.170138374e-7 | 1e-12 | 3e-10 | 2e-12 | 3e-10 |

Internal error estimates of each method alone:
* **Method B.** eps agrees to all 20 printed digits across every variant. Gamma agrees to <= 3e-11 (lam = -0.01003), <= 3e-12 (-0.02) and <= 3e-14 for the others.
* **Method A** (finest grid against the next grid, and the h = 0.04 variants). eps is good to about 5e-12 (7e-9 for lam = -1, where the broad tail runs into the steep potential), and Gamma to 2e-9 to 2e-8.
* In Method A on the finest grid, `Gamma_eig = -2 Im E` and `Gamma_flux` agree to <= 1e-10 even at lam = -0.01003.

Full-precision values are in `blind_results.json`, and every individual run is in `methodA_results.json` and `methodB_results.json`.

## 5. Scaling check (item 2)

**Argument.** Under the unitary dilation `psi(r,s) -> alpha^{5/2} psi(alpha r, alpha s)`, the kinetic term scales as alpha^2 and both quartic terms as alpha^-4. With `alpha^6 = e^2` this gives `H(e, lam) ~ e^{2/3} H(1, lam/e^2)`. The dilation preserves regularity and the outgoing condition, so `eps(e,lam) = e^{2/3} eps(1, lam/e^2)` and `Gamma(e,lam) = e^{2/3} Gamma(1, lam/e^2)`. For e = 0.873, `e^{2/3} = 0.913431874686967`.

| | eps(0.873, -0.03*0.873^2) | e^{2/3} eps(1,-0.03) | rel. dev. | Gamma(0.873, ...) | e^{2/3} Gamma(1,-0.03) | rel. dev. |
|---|---|---|---|---|---|---|
| Method B | 1.72705079778190241 | 1.72705079778190241 | -1e-18 | 4.17013837433421e-7 | 4.17013837433421e-7 | -5e-16 |
| Method A | 1.727050797780286 | 1.727050797780090 | 1e-13 | 4.1701383757e-7 | 4.1701383757e-7 | -7e-12 |

The law holds to the numerical precision of each method. This is a test of the code, since the relation is exact.

## 6. Optional item 3: further lambda values and local slope (e = 1)

Gamma(-0.05) = 4.759199383e-5 and Gamma(-0.02) = 3.709882227e-9; both methods agree (table above). The local slope `-d ln Gamma / d x`, with `x = |lam|^{-1/2}`, comes from 5-point differences with dx = 0.05 (Method B, 120x60). The 3-point values agree to 6e-5 relative at lam = -0.11174 and to <= 1.2e-5 elsewhere.

| lambda | x | -d lnGamma/dx | BO-WKB 2 dS/dx at E = eps(lambda) |
|---|---|---|---|
| -0.11174 | 2.9915 | 2.9939 | 3.062 |
| -0.05 | 4.4721 | 3.4742 | 3.475 |
| -0.03 | 5.7735 | 3.6506 | 3.647 |
| -0.02 | 7.0711 | 3.7603 | 3.757 |
| -0.01003 | 9.9850 | 3.9083 | 3.906 |

`S(E) = int sqrt(2(V_BO - E)) dr` between the turning points of the Born-Oppenheimer potential `V_BO = (3/2) e r + lam r^4/4`, which is the zero-point energy of the three w oscillators plus the quartic. As lam -> 0 at fixed E = 0, `2S -> sqrt(2) pi e |lam|^{-1/2}`, so the asymptotic slope is `sqrt(2) pi = 4.443`. The measured slope is still well below this at lam = -0.01003, because eps = 1.9 is a sizeable fraction of the barrier. A dense scan of Gamma(lambda) for x = 1..10 is in `slope_results.json` and in the tables below.

## 7. Noteworthy points

1. **Smoothness of the ECS switch matters.** With a C3 polynomial switch over w = 1, Gamma had errors up to 3e-4 relative at h = 0.05. They depended on theta (lam = -0.03, theta = 0.3: 4.56408e-7 against 4.56535e-7) and decreased only slowly with h (roughly like h^4), while eps was barely affected. The finite-difference stencil resolves the limited smoothness of the contour, and the resulting spurious scattering changes the flux. The C-infinity (erf) switch over w = 2 removes the problem: A and B then agree to <= 1e-7 already at h = 0.05, and to <= 2e-9 at the finest grids.
2. **Im E in ECS has no double-precision floor.** With a real shift and a real start vector, the interior of the ECS eigenvector stays real and Im E comes only from the exterior, so Gamma_eig in double precision has the same relative accuracy as the flux formula (on the finest grid it agrees to 1e-10 at Gamma = 5e-14). Uniform complex rotation in double precision does have the floor (1% scatter at lam = -0.01003). The 40-digit c-product Rayleigh quotient removes it.
3. **The widths are much larger than the E = 0 Born-Oppenheimer estimate.** `exp(-sqrt(2) pi |lam|^{-1/2})` is about 5e-20 at lam = -0.01003, while Gamma = 5.1e-14. The resonance energy eps of about 1.9 lowers the barrier action from 44.4 to 2S(eps) = 29.4. With E = eps, the BO-WKB slope matches the exact slope to <= 0.1% for |lam| <= 0.05 and to 2.3% at lam = -0.11174.
4. **lam = -1 lies above the BO barrier top** (max of 3r/2 - r^4/4 = 1.288 < eps = 1.669). The state is broad (Gamma/eps = 0.37) but remains a well-defined, theta-independent pole.
5. **eps depends only weakly on lambda at small |lambda|.** It tends to the lam = 0 energy 1.90256784984007 (both methods), which is a true bound state because the lam = 0 potential `e^2 r^2 s^2/2` confines, as the x^2 y^2 potential does.
6. **Environment.** In this Python environment SciPy is linked against conda MKL. Its threaded layer crashed SuperLU (Windows fatal exception 0xc06d007f) for matrices larger than about 5000. Every script therefore sets `MKL_THREADING_LAYER=SEQUENTIAL`.
7. **Blindness.** From `00_HANDOFF/20261022_Claude_Report_v11/` only the brief file was opened. No other project file was read and the project tree was not searched. The session's own scratch folder was used once, for a one-line Python test, and nothing else in it was opened. No earlier result for this problem was encountered.

## 8. Reproduction

    D:\Python\envs\py312-main\python.exe code\run_all.py --fresh

This runs Method B (about 1.5 min), Method A (about 6 min on 3 processes; the largest grid has 1.2e5 unknowns), the slope scan (about 1 min), `collect.py` (writes `blind_results.json`) and `make_tables.py` (writes `convergence_tables.md`, and assembles `METHOD.md` from `METHOD_head.md` plus those tables). The modules are:

* `ho_basis.py`, `ho_mp.py`: Method B.
* `ecs_grid.py`: Method A.
* `run_method{A,B}.py`, `slope_scan.py`, `collect.py`, `make_tables.py`, `run_all.py`: drivers and collection.

## Appendix: complete convergence tables

#### e=1,lam=-1

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.668842044054824 | 6.122882663190e-01 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=2.6, theta=0.60, Rmax=6.1) | 19315 | 1.6688419679012 | 6.1228821755e-01 | 6.1228821721e-01 (R=1.60) | 6.1228821789e-01 (R=2.00) |
| h=0.04 (R0=2.6, theta=0.60, Rmax=6.1) | 30098 | 1.6688420305378 | 6.1228825767e-01 | 6.1228825761e-01 (R=1.60) | 6.1228825772e-01 (R=2.00) |
| h=0.03 (R0=2.6, theta=0.60, Rmax=6.1) | 53451 | 1.6688420426390 | 6.1228826541e-01 | 6.1228826541e-01 (R=1.59) | 6.1228826542e-01 (R=2.01) |
| h=0.025 (R0=2.6, theta=0.60, Rmax=6.1) | 77034 | 1.6688420437193 | 6.1228826610e-01 | 6.1228826610e-01 (R=1.60) | 6.1228826611e-01 (R=2.00) |
| h=0.02 (R0=2.6, theta=0.60, Rmax=6.1) | 120301 | 1.6688420439954 | 6.1228826628e-01 | 6.1228826628e-01 (R=1.60) | 6.1228826628e-01 (R=2.00) |
| h=0.04,R0+0.4 (R0=3.0, theta=0.60, Rmax=6.5) | 31090 | 1.6688420274451 | 6.1228827204e-01 | 6.1228827198e-01 (R=1.60) | 6.1228827209e-01 (R=2.00) |
| h=0.04,Smax+2,mask70 (R0=2.6, theta=0.60, Rmax=6.1) | 37328 | 1.6688420305378 | 6.1228825767e-01 | 6.1228825761e-01 (R=1.60) | 6.1228825772e-01 (R=2.00) |
| h=0.04,p=5 (R0=2.6, theta=0.60, Rmax=6.1) | 30098 | 1.6688420429621 | 6.1228826563e-01 | 6.1228826563e-01 (R=1.60) | 6.1228826562e-01 (R=2.00) |
| h=0.04,theta=0.45 (R0=2.6, theta=0.45, Rmax=7.1) | 32614 | 1.6688420387638 | 6.1228826121e-01 | 6.1228826115e-01 (R=1.60) | 6.1228826126e-01 (R=2.00) |

#### e=1,lam=-0.11174

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.850650978201298 | 6.116424540397e-03 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.850650978201295 | 6.116424540394e-03 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=4.6, theta=0.50, Rmax=9.1) | 23655 | 1.8506509781646 | 6.1164246044e-03 | 6.1164246053e-03 (R=3.60) | 6.1164246139e-03 (R=4.00) |
| h=0.04 (R0=4.6, theta=0.50, Rmax=9.1) | 36978 | 1.8506509781946 | 6.1164245515e-03 | 6.1164245516e-03 (R=3.60) | 6.1164245530e-03 (R=4.00) |
| h=0.03 (R0=4.6, theta=0.50, Rmax=9.1) | 65548 | 1.8506509781997 | 6.1164245415e-03 | 6.1164245415e-03 (R=3.60) | 6.1164245417e-03 (R=3.99) |
| h=0.04,R0+0.7 (R0=5.3, theta=0.50, Rmax=9.8) | 38351 | 1.8506509781851 | 6.1164245392e-03 | 6.1164245393e-03 (R=3.60) | 6.1164245408e-03 (R=4.00) |
| h=0.04,Smax+2,mask70 (R0=4.6, theta=0.50, Rmax=9.1) | 45477 | 1.8506509781946 | 6.1164245514e-03 | 6.1164245516e-03 (R=3.60) | 6.1164245530e-03 (R=4.00) |
| h=0.04,p=5 (R0=4.6, theta=0.50, Rmax=9.1) | 36978 | 1.8506509782012 | 6.1164245409e-03 | 6.1164245409e-03 (R=3.60) | 6.1164245408e-03 (R=4.00) |
| h=0.04,theta=0.35 (R0=4.6, theta=0.35, Rmax=10.1) | 38902 | 1.8506509781924 | 6.1164245449e-03 | 6.1164245450e-03 (R=3.60) | 6.1164245465e-03 (R=4.00) |

#### e=1,lam=-0.03

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.890727536055995 | 4.565352383736e-07 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370487e-07 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.890727536055999 | 4.565352370436e-07 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=6.8, theta=0.50, Rmax=11.3) | 26389 | 1.8907275359980 | 4.5653524523e-07 | 4.5653524548e-07 (R=5.80) | 4.5653524619e-07 (R=6.20) |
| h=0.04 (R0=6.8, theta=0.50, Rmax=11.3) | 41180 | 1.8907275360456 | 4.5653523844e-07 | 4.5653523848e-07 (R=5.80) | 4.5653523860e-07 (R=6.20) |
| h=0.03 (R0=6.8, theta=0.50, Rmax=11.3) | 73241 | 1.8907275360540 | 4.5653523719e-07 | 4.5653523719e-07 (R=5.79) | 4.5653523720e-07 (R=6.21) |
| h=0.04,R0+0.7 (R0=7.5, theta=0.50, Rmax=12.0) | 42490 | 1.8907275360456 | 4.5653523738e-07 | 4.5653523742e-07 (R=5.80) | 4.5653523754e-07 (R=6.20) |
| h=0.04,Smax+2,mask70 (R0=6.8, theta=0.50, Rmax=11.3) | 50455 | 1.8907275360456 | 4.5653523843e-07 | 4.5653523848e-07 (R=5.80) | 4.5653523860e-07 (R=6.20) |
| h=0.04,p=5 (R0=6.8, theta=0.50, Rmax=11.3) | 41180 | 1.8907275360558 | 4.5653523707e-07 | 4.5653523707e-07 (R=5.80) | 4.5653523707e-07 (R=6.20) |
| h=0.04,theta=0.35 (R0=6.8, theta=0.35, Rmax=12.3) | 43059 | 1.8907275360456 | 4.5653523816e-07 | 4.5653523820e-07 (R=5.80) | 4.5653523832e-07 (R=6.20) |

#### e=1,lam=-0.01003

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.096430831530e-14 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097163213903e-14 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168574366e-14 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606375e-14 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606520e-14 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.898718938865880 | 5.097168606520e-14 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.898718938865880 | 5.097168606520e-14 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.898718938865880 | 5.097168606521e-14 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168540956e-14 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606521e-14 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606521e-14 |
| 200 x 90 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606487e-14 |
| 250 x 100 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606520e-14 |
| 200 x 90 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606521e-14 |
| 200 x 90 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.898718938865880 | 5.097168606521e-14 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=10.0, theta=0.50, Rmax=14.5) | 29924 | 1.8987189388072 | 5.0971691369e-14 | 5.0971691517e-14 (R=9.00) | 5.0971691682e-14 (R=9.40) |
| h=0.04 (R0=10.0, theta=0.50, Rmax=14.5) | 46719 | 1.8987189388554 | 5.0971686961e-14 | 5.0971686988e-14 (R=9.00) | 5.0971687016e-14 (R=9.40) |
| h=0.03 (R0=10.0, theta=0.50, Rmax=14.5) | 83038 | 1.8987189388639 | 5.0971686156e-14 | 5.0971686158e-14 (R=9.00) | 5.0971686161e-14 (R=9.39) |
| h=0.04,R0+0.7 (R0=10.7, theta=0.50, Rmax=15.2) | 47878 | 1.8987189388554 | 5.0971687036e-14 | 5.0971687061e-14 (R=9.00) | 5.0971687089e-14 (R=9.40) |
| h=0.04,Smax+2,mask70 (R0=10.0, theta=0.50, Rmax=14.5) | 57016 | 1.8987189388554 | 5.0971686962e-14 | 5.0971686988e-14 (R=9.00) | 5.0971687016e-14 (R=9.40) |
| h=0.04,p=5 (R0=10.0, theta=0.50, Rmax=14.5) | 46719 | 1.8987189388656 | 5.0971686076e-14 | 5.0971686078e-14 (R=9.00) | 5.0971686078e-14 (R=9.40) |
| h=0.04,theta=0.35 (R0=10.0, theta=0.35, Rmax=15.5) | 48383 | 1.8987189388554 | 5.0971686968e-14 | 5.0971686994e-14 (R=9.00) | 5.0971687022e-14 (R=9.40) |

#### e=0.873,lam=-0.02286387

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.727050797781902 | 4.170138380661e-07 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.727050797781903 | 4.170138374334e-07 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=7.1, theta=0.50, Rmax=11.6) | 29310 | 1.7270507977449 | 4.1701384487e-07 | 4.1701384498e-07 (R=6.00) | 4.1701384549e-07 (R=6.50) |
| h=0.04 (R0=7.1, theta=0.50, Rmax=11.6) | 45783 | 1.7270507977751 | 4.1701383870e-07 | 4.1701383872e-07 (R=6.00) | 4.1701383880e-07 (R=6.48) |
| h=0.03 (R0=7.1, theta=0.50, Rmax=11.6) | 81432 | 1.7270507977803 | 4.1701383756e-07 | 4.1701383756e-07 (R=6.00) | 4.1701383757e-07 (R=6.51) |
| h=0.04,R0+0.7 (R0=7.8, theta=0.50, Rmax=12.3) | 47166 | 1.7270507977751 | 4.1701383722e-07 | 4.1701383724e-07 (R=6.00) | 4.1701383732e-07 (R=6.48) |
| h=0.04,Smax+2,mask70 (R0=7.1, theta=0.50, Rmax=11.6) | 55747 | 1.7270507977751 | 4.1701383870e-07 | 4.1701383872e-07 (R=6.00) | 4.1701383880e-07 (R=6.48) |
| h=0.04,p=5 (R0=7.1, theta=0.50, Rmax=11.6) | 45783 | 1.7270507977817 | 4.1701383747e-07 | 4.1701383747e-07 (R=6.00) | 4.1701383747e-07 (R=6.48) |
| h=0.04,theta=0.35 (R0=7.1, theta=0.35, Rmax=12.6) | 47692 | 1.7270507977751 | 4.1701383827e-07 | 4.1701383829e-07 (R=6.00) | 4.1701383837e-07 (R=6.48) |

#### e=1,lam=-0.05

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.882149226058301 | 4.759199383387e-05 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.882149226058294 | 4.759199383460e-05 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=6.0, theta=0.50, Rmax=10.5) | 25429 | 1.8821492260010 | 4.7591993502e-05 | 4.7591993542e-05 (R=5.00) | 4.7591993654e-05 (R=5.40) |
| h=0.04 (R0=6.0, theta=0.50, Rmax=10.5) | 39676 | 1.8821492260480 | 4.7591993776e-05 | 4.7591993783e-05 (R=5.00) | 4.7591993802e-05 (R=5.40) |
| h=0.03 (R0=6.0, theta=0.50, Rmax=10.5) | 70530 | 1.8821492260563 | 4.7591993829e-05 | 4.7591993829e-05 (R=5.01) | 4.7591993831e-05 (R=5.40) |
| h=0.04,R0+0.7 (R0=6.7, theta=0.50, Rmax=11.2) | 41032 | 1.8821492260480 | 4.7591993953e-05 | 4.7591993960e-05 (R=5.00) | 4.7591993978e-05 (R=5.40) |
| h=0.04,Smax+2,mask70 (R0=6.0, theta=0.50, Rmax=10.5) | 48671 | 1.8821492260480 | 4.7591993776e-05 | 4.7591993783e-05 (R=5.00) | 4.7591993802e-05 (R=5.40) |
| h=0.04,p=5 (R0=6.0, theta=0.50, Rmax=10.5) | 39676 | 1.8821492260580 | 4.7591993831e-05 | 4.7591993831e-05 (R=5.00) | 4.7591993831e-05 (R=5.40) |
| h=0.04,theta=0.35 (R0=6.0, theta=0.35, Rmax=11.5) | 41621 | 1.8821492260480 | 4.7591993836e-05 | 4.7591993842e-05 (R=5.00) | 4.7591993861e-05 (R=5.40) |

#### e=1,lam=-0.02

Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)

| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |
|---|---|---|---|---|---|
| 60 x 30 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709878911654e-09 |
| 80 x 40 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882225699e-09 |
| 100 x 50 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226702e-09 |
| 120 x 60 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 1.0, 1.0 | 0.5236 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 1.0, 1.0 | 0.2618 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 1.0, 1.0 | 0.3927 | -0.1963 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 0.7, 0.7 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226707e-09 |
| 150 x 70 | 1.5, 1.5 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |
| 150 x 70 | 0.7, 1.5 | 0.3927 | 0.0000 | 1.894788859932730 | 3.709882226697e-09 |

Method A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)

| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |
|---|---|---|---|---|---|
| h=0.05 (R0=8.2, theta=0.50, Rmax=12.7) | 27991 | 1.8947888598744 | 3.7098823334e-09 | 3.7098823399e-09 (R=7.00) | 3.7098823542e-09 (R=7.50) |
| h=0.04 (R0=8.2, theta=0.50, Rmax=12.7) | 43760 | 1.8947888599223 | 3.7098822448e-09 | 3.7098822458e-09 (R=7.00) | 3.7098822484e-09 (R=7.52) |
| h=0.03 (R0=8.2, theta=0.50, Rmax=12.7) | 77642 | 1.8947888599308 | 3.7098822284e-09 | 3.7098822286e-09 (R=6.99) | 3.7098822289e-09 (R=7.50) |
| h=0.04,R0+0.7 (R0=8.899999999999999, theta=0.50, Rmax=13.4) | 44926 | 1.8947888599223 | 3.7098822502e-09 | 3.7098822513e-09 (R=7.00) | 3.7098822539e-09 (R=7.52) |
| h=0.04,Smax+2,mask70 (R0=8.2, theta=0.50, Rmax=12.7) | 53510 | 1.8947888599223 | 3.7098822448e-09 | 3.7098822458e-09 (R=7.00) | 3.7098822484e-09 (R=7.52) |
| h=0.04,p=5 (R0=8.2, theta=0.50, Rmax=12.7) | 43760 | 1.8947888599325 | 3.7098822270e-09 | 3.7098822269e-09 (R=7.00) | 3.7098822270e-09 (R=7.52) |
| h=0.04,theta=0.35 (R0=8.2, theta=0.35, Rmax=13.7) | 45397 | 1.8947888599223 | 3.7098822443e-09 | 3.7098822454e-09 (R=7.00) | 3.7098822480e-09 (R=7.52) |

#### Optional: local slope -d ln Gamma / d x, x = |lambda|^(-1/2), e = 1 (Method B)

| lambda | x | slope (5-pt) | slope (3-pt) | BO-WKB 2 dS/dx (E = eps(lambda)) | BO-WKB 2S | sqrt(2) pi x |
|---|---|---|---|---|---|---|
| -0.11174 | 2.9915 | 2.99385 | 2.99367 | 3.0622 | 3.848 | 13.291 |
| -0.05 | 4.4721 | 3.47416 | 3.47412 | 3.4748 | 8.735 | 19.869 |
| -0.03 | 5.7735 | 3.65060 | 3.65058 | 3.6474 | 13.379 | 25.651 |
| -0.02 | 7.0711 | 3.76026 | 3.76026 | 3.7571 | 18.187 | 31.416 |
| -0.01003 | 9.9850 | 3.90827 | 3.90827 | 3.9062 | 29.375 | 44.362 |

Dense scan (Method B, Nr x Ns = 120 x 60, theta = pi/8):

| x | lambda | eps | Gamma |
|---|---|---|---|
| 1.00 | -1.000000 | 1.668842044055 | 6.1228826632e-01 |
| 1.25 | -0.640000 | 1.693752628379 | 3.8746245462e-01 |
| 1.50 | -0.444444 | 1.723417768175 | 2.4154550811e-01 |
| 1.75 | -0.326531 | 1.753184585376 | 1.4606954682e-01 |
| 2.00 | -0.250000 | 1.780575159170 | 8.4763889052e-02 |
| 2.25 | -0.197531 | 1.804264349744 | 4.6890684917e-02 |
| 2.50 | -0.160000 | 1.823745197004 | 2.4678739693e-02 |
| 2.75 | -0.132231 | 1.839143919280 | 1.2387448082e-02 |
| 3.00 | -0.111111 | 1.851001626201 | 5.9634384805e-03 |
| 3.25 | -0.094675 | 1.860029297560 | 2.7724129802e-03 |
| 3.50 | -0.081633 | 1.866915284302 | 1.2529382642e-03 |
| 3.75 | -0.071111 | 1.872227614727 | 5.5347864060e-04 |
| 4.00 | -0.062500 | 1.876393996952 | 2.4000387434e-04 |
| 4.25 | -0.055363 | 1.879721178705 | 1.0248554762e-04 |
| 4.50 | -0.049383 | 1.882424862005 | 4.3197823797e-05 |
| 4.75 | -0.044321 | 1.884656648127 | 1.8004939115e-05 |
| 5.00 | -0.040000 | 1.886524297257 | 7.4310612877e-06 |
| 5.25 | -0.036281 | 1.888105764845 | 3.0402808318e-06 |
| 5.50 | -0.033058 | 1.889458581143 | 1.2341417997e-06 |
| 5.75 | -0.030246 | 1.890626053783 | 4.9742070487e-07 |
| 6.00 | -0.027778 | 1.891641390133 | 1.9918614563e-07 |
| 6.25 | -0.025600 | 1.892530476147 | 7.9286843990e-08 |
| 6.50 | -0.023669 | 1.893313785279 | 3.1386963534e-08 |
| 6.75 | -0.021948 | 1.894007717087 | 1.2361696848e-08 |
| 7.00 | -0.020408 | 1.894625555206 | 4.8455306947e-09 |
| 7.25 | -0.019025 | 1.895178166079 | 1.8909313033e-09 |
| 7.50 | -0.017778 | 1.895674517439 | 7.3485803006e-10 |
| 7.75 | -0.016649 | 1.896122068992 | 2.8446850567e-10 |
| 8.00 | -0.015625 | 1.896527070817 | 1.0971527739e-10 |
| 8.25 | -0.014692 | 1.896894794061 | 4.2168880378e-11 |
| 8.50 | -0.013841 | 1.897229711228 | 1.6154403310e-11 |
| 8.75 | -0.013061 | 1.897535638433 | 6.1693361405e-12 |
| 9.00 | -0.012346 | 1.897815848664 | 2.3491089348e-12 |
| 9.25 | -0.011687 | 1.898073162669 | 8.9196563442e-13 |
| 9.50 | -0.011080 | 1.898310022437 | 3.3777807161e-13 |
| 9.75 | -0.010519 | 1.898528551019 | 1.2758713328e-13 |
| 10.00 | -0.010000 | 1.898730601537 | 4.8075543476e-14 |
