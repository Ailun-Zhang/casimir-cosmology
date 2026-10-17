# Keystone (T3) and V2 — specification given to the blind reimplementation

Summary of the written specifications of 2026-10-14 (T3) and 2026-10-15 (V2), reduced to the model definition, the
inputs and the pass criteria (internal paths, budgets and workflow instructions removed). Report: Sec. 19.2–19.3.

## Blindness
The reimplementation had no access to the derivations, code or results of the main calculation, nor to the records
of the other checks. Allowed: this specification, the earlier (v11) draft of the paper for conventions and as an
external anchor (its full-dimensional DMC value of the local zero-mode model at λ = 0, and the "next order"
holonomy curvatures for V2), the published literature, Python and Wolfram Language documentation. Results were
written to `BLIND_RESULTS.json` (time stamp, SHA-256 of the code) before any number of the main calculation was
received; the comparison was made afterwards.

## T3: the reduced Higgs chain
**Setting.** Flat cubic three-torus of side L(t); all fermions antiperiodic in the three directions (AAA); gauge zero
modes expanded about trivial holonomy; no 't Hooft flux. Background: Casimir turning point L(t) = L_b √cosh(2ht),
u = ht, conformal time dη = du /(hL_b √cosh 2u); input hL_b = 0.00437243; μ_b = 1/L_b = M̄_P/722,
M̄_P = 2.435×10¹⁸ GeV. Fields: SU(2)_L×U(1)_Y gauge fields, the Higgs doublet, the top quark (the SU(3) zero mode is
decoupled from the Higgs zero mode in this model). Units L = 1. Φ = (φ₀+iφ₁, φ₂+iφ₃)/√2, r = |φ|, potential λr⁴/4
(MS-bar λ of V = λ|Φ|⁴), m_t = y_t r/√2, ξ_H = 1/6.

**Model.** (A) Zero-mode Hamiltonian: constant modes of the SU(2) and U(1)_Y gauge fields (Gauss law on states,
temporal gauge) and of the Higgs doublet; slow coordinate r (flat measure of four real components), the three
angular Higgs directions in the body frame of the SU(2) Gauss law. (B) One-loop finite-volume matching depending on
r only: non-zero modes integrated out at trivial holonomy; zero-point energy in the MS-bar finite-volume form
(infinite-volume Coleman–Weinberg term at μ = 1/L plus the discrete-minus-continuum difference) for the Coulomb-gauge
spectrum: per massive vector two transverse modes ω² = k² + m_V² and one Coulomb mode mixed with its Goldstone,
ω² = (k² + m_G²)(k² + m_V²)/k², m_G² = λr²; radial Higgs m_h² = 3λr² (analytically continued); top with 12 real
degrees of freedom, AAA; MS-bar constants 5/6 per vector polarization sum as stated, 3/2 for scalars and fermions;
couplings at μ = 1/L. (C) Reduction: the SU(2) zero mode in a conditional Gaussian family invariant under
SO(3)_space × U(1)_em (precision matrix diag(a, a, K₂) per spatial direction), minimized at each r; the U(1)_Y zero
mode w kept exactly, jointly with r, in the s wave (s = |w|), with the Born–Huang terms of the r dependence of the
Gaussian; this two-dimensional model is "2D_G". (D) Resonances for λ < 0 with purely outgoing conditions (at least
two of Siegert states, complex scaling, absorbing potentials), E = Q − iγ/2, rejecting box states. (E) Hazard along
the turn: P₁ = exp[−∫₀⁵ γ(u) dη], γ(u) at μ(u) = μ_b/√cosh 2u, quasi-stationary preparation (w₀ = 1); static
criterion P = exp(−γ η_tot). (F) Standard Model input: M_t (pole), α_s(M_Z) = 0.1180, M_h = 125.13 GeV,
M_W = 80.3692 GeV; couplings at M_t from Buttazzo et al. (2013) Eqs. (55), (57)–(60); two-loop running; a constant
three-loop calibration Δλ defined from five input sets against Eq. (61e) at the (non-reduced) Planck mass.

**Required outputs.** Static benchmarks at the "v11 couplings" g₂ = 0.5268, g_Y = 0.4415, y_t = 0.4585: λ = 0
ground-state energies (2D_G and 1D Born–Oppenheimer), (Q, γ) at λ = −0.008, −0.010, −0.0143, static λ_c
(P = 0.9, 0.5); hazard: P₁(m_t) at 172.2/172.6/173.0 GeV and on a grid, critical m_t at P₁ = 0.9/0.5/0.1,
hazard-equivalent λ(μ_b), α_s = 0.1171/0.1189 if possible; optionally an own non-Gaussian SU(2) correction δE(r)
(e.g. diffusion Monte Carlo) added to 2D_G ("2D with own DMC shape"); numerical errors of every number.

**Pass criterion.** Any difference above 0.05 GeV in a critical mass must be located and explained; the quoted
agreement (report Table 30): critical masses to 0.005 GeV, λ(μ_b) to 10⁻⁶, widths to 0.1 %.

## V2: cross terms between the zero mode and the non-zero modes
Same blindness rules, extending the T3 implementation. Derive the one-loop energy of all non-zero modes (W±, Z,
photon; Higgs radial and Goldstone; top; all light fermions, AAA) in a constant zero-mode background: Abelian/Cartan
directions with twisted finite-volume sums (periodic bosons, AAA fermions, the Higgs n = 0 mode excluded),
non-Abelian parts at least to second order; mass dependence on r kept. Reproduce the published curvatures at r = 0
(v11 "next order": SU(3), SU(2), hypercharge). Add the cross terms to the two-dimensional model (hypercharge exactly in
s, SU(2) in the conditional Gaussian, SU(3)–top as a change of the SU(3) zero-mode energy with r). Report whether the
soft w direction is unstable with Higgs fluctuations only and with all fields; width ratios with/without cross terms
at λ = −0.008, −0.010, −0.0143 (v11 couplings); the shift of the critical masses at P₁ = 0.9/0.5/0.1 split into
SU(3)–top, SU(2) and U(1)_Y parts, with the physical reason for the sign. Results in `BLIND_RESULTS_V2.json`.
Pass criterion: the shift agrees with the main calculation within the stated errors (numerical ≲ 0.005 GeV, model
±0.015 GeV).
