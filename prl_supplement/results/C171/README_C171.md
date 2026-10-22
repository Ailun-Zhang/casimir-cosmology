# C171 — Abelian Higgs model in a small box: exact zero-mode resonance

Supplemental Material to A. Zhang, *Gauge Field Fluctuations Protect the Higgs Vacuum When a Small Universe Begins*
(2026). Computed 2026-10-19. Code: `c171/` (written from scratch; no code of the release v1.0.0 is used). Results:
this folder.

## 1. Summary

* The zero-mode problem of the U(1) Higgs model (complex scalar zero mode, three constant gauge modes, no fermion zero
  modes, no matching) was solved exactly as a two-dimensional resonance problem for 24 values
  Λ = λ/e² from −1 to −0.0065 (log-spaced in x = |Λ|^{-1/2} from 1 to 12.4) plus the two SM-like points.
  Widths range from Γ = 0.61 to Γ = 3.2×10⁻¹⁸ (units e^{2/3}/L). Internal precision of Γ is ~10⁻⁸ (relative);
  an independent complex-absorbing-potential (CAP) calculation agrees to ≤ 7×10⁻⁷ at four Λ.
* Converged values (e = 1): Λ = −1: ε = 1.6688420, Γ = 0.6122883; Λ = −0.11174: ε = 1.8506521, Γ = 6.115927×10⁻³;
  Λ = −0.03: ε = 1.8907275, Γ = 4.565352×10⁻⁷; Λ = −0.01003: ε = 1.8987197, Γ = 5.07904×10⁻¹⁴.
* The local slope b = −d ln Γ/dx is 1.82 at Λ = −1, 3.65 at Λ = −0.03 and 3.97 at Λ = −0.0065, below the deep-well
  value π√2 = 4.443 throughout the computed range. The deficit is the energy dependence of the barrier action: the
  resonance sits at ε ≈ 1.90 above the bottom of the linear well, and S(ε) ≈ π√2 x − 3.78 ε x^{1/3}, so
  b ≈ π√2 − 1.26 ε x^{−2/3} → 4.443 only as |Λ|^{1/3}. The WKB action of the plain BO barrier evaluated at the exact
  energy reproduces b to ≤ 0.003 for x ≥ 4. For |Λ| ≳ 0.4 the resonance lies above the barrier top. The effective
  prefactor A_eff = Γ e^{π√2 x} grows from 52 (Λ = −1) to 3.2×10⁶ (Λ = −0.0065).
* Thresholds (Γ N = ln 2, e = 1): Λ_c(10) = −0.2295, Λ_c(300) = −0.09138, Λ_c(10⁴) = −0.05252, Λ_c(10⁶) = −0.03123.
  |Λ_c|^{1/2} ln N rises from 1.10 (N = 10) to 3.16 (N = 10¹⁶) and approaches π√2 only as (ln N)^{−2/3}:
  |Λ_c|^{1/2} is linear in 1/ln N asymptotically (slope π√2, zero intercept), but strongly curved for N ≤ 10¹⁶.
* Born–Oppenheimer: the plain BO model (U = 3er/2 − 1/(8r²) + λr⁴/4) gives ε too low by 0.10 and Γ within
  +2%/−28% (Γ_BO/Γ = 1.02 at Λ = −1, 0.85 at Λ = −0.03, 0.72 at Λ = −0.0065), with b within 0.04 of the exact slope.
  Adding the diagonal BO correction 3/(16r²) makes the model much worse: ε too high by 0.53–0.65 and Γ too large by
  factors 2.5 (Λ = −1) to 233 (Λ = −0.0065).
* Holonomy: inside the well the probability of e|w| > π is 1.8–2.6×10⁻³ (e = 1), 5.5–8.1×10⁻⁴ (e = 0.873) and
  3.2–5.9×10⁻⁷ (e = 0.5); beyond e|w| = π/2 it is 23–26%, 16–19% and 1.9–2.4%.
* SM-like slope (e = 0.8733, σ₀ = 1.31, pure zero modes): λ = −0.010 gives Γ = 6.011×10⁻¹²/L, λ = −0.013 gives
  Γ = 3.664×10⁻¹⁰/L.

All 13 automated checks in `checks.json` pass.

## 2. Model and reduction

Scalar zero mode Φ = (φ₁ + iφ₂)/√2, |Φ|² = r²/2; constant gauge modes w ∈ ℝ³ (canonical); units of 1/L:

  H = ½p_φ² + ½p_w² + ½e²r²w² + (λ/4)r⁴.

Gauss's law for constant gauge transformations selects zero U(1) charge, i.e. m = 0 in the (φ₁, φ₂) plane. The
potential depends on w only through s = |w|, so the ground sector is the s-wave. On ψ(r, s) with measure r dr s² ds,

  H = −½(∂_r² + r⁻¹∂_r) − ½(∂_s² + 2s⁻¹∂_s) + ½e²r²s² + (λ/4)r⁴ .

The code works with Φ̃ = sψ (removes the 3D s-wave first-derivative term, Φ̃(s = 0) = 0) and keeps the 2D radial
operator in flux form with weight r, so the m = 0 behaviour at r = 0 (ψ smooth and even in r) is built in; this is
equivalent to the reduced form u = r^{1/2}sψ with the −1/(8r²) term.

**Scaling.** r, s → e^{−1/3}(r, s) gives H = e^{2/3}h(Λ), Λ = λ/e². All scans use e = 1; for general e,
Γ = e^{2/3}F(Λ) and ε = e^{2/3}ε₁(Λ). This was verified by direct (unscaled) runs at e = 0.8733 (λ = −0.010, −0.013)
and at e = 0.5 (λ = −0.0180): Γ_direct/(e^{2/3}F) − 1 = 7×10⁻¹², 5×10⁻¹², 3×10⁻¹⁴.

**Born–Oppenheimer (BO) and the diagonal correction.** At fixed r the w modes are three oscillators of frequency
ω = er with ground state χ₀(w; r) = (ω/π)^{3/4} e^{−ωw²/2} and zero-point energy (3/2)er (slope σ₀ = 3e/2). With
∂_ω χ₀ = (3/(4ω) − w²/2)χ₀ and ⟨w²/2⟩ = 3/(4ω),

  ⟨∂_ω χ₀|∂_ω χ₀⟩ = Var(w²/2) = ¼·3·2·(1/(2ω))² = 3/(8ω²),

so the diagonal correction is ½⟨∂_r χ₀|∂_r χ₀⟩ = ½e²·3/(8e²r²) = 3/(16r²), independent of e. The 1D models are
(u = r^{1/2}F-form)

  U_BO(r) = (3/2)er − 1/(8r²) + (λ/4)r⁴,   U_BOD(r) = U_BO(r) + 3/(16r²) = (3/2)er + 1/(16r²) + (λ/4)r⁴ .

For a bound ground state the BO energy without the diagonal term is a lower bound and with it an upper bound
(Brattsev, Epstein); the computed resonance energies satisfy ε_BO < ε < ε_BOD at every Λ (check `BO_bounds`).

**Valley along s.** At r = 0 the potential is flat in s, but for large s the r motion is a 2D oscillator of frequency
es (zero-point energy es), so the s direction is confined by an effective linear potential. Moving the s box from
S = 10 to S = 12 (e = 1 units) changes ln Γ by ≤ 7×10⁻¹² and ε by ≤ 1.4×10⁻¹³.

**Without the gauge modes** (e → 0 at fixed λ < 0) the problem is −½∇²_φ + (λ/4)|φ|⁴ on ℝ², whose potential decreases
monotonically from 0: there is no well, no barrier and no quasi-bound state. (In the reduced u-form,
−1/(8r²) + λr⁴/4 is not monotonic — it has a maximum of negative height −(3/16)(4|λ|)^{1/3} at r = (4|λ|)^{−1/6} —
but −1/(8r²) is the m = 0 kinetic term, removed by the Langer correction, not a barrier.) Numerically, the lowest
complex-scaled eigenvalue at λ = −0.03 is E = 0.14457 − 0.25040 i, at arg E = −60.000000° (Γ/Re E = 2√3), as
dictated by the scaling of the inverted quartic: a pole on the rotated ray, not a narrow resonance.

## 3. Numerical methods

### Method A (primary): exterior complex scaling + finite differences in r + sine DVR in s
* r: second-order cell-centred finite differences of −½r⁻¹∂_r(r∂_r) in flux form (nodes ρ_i = (i − ½)h, flux
  points jh; the flux weight vanishes at r = 0). Smooth exterior complex scaling r(ρ) = ρ + (e^{iθ} − 1)δ I((ρ − R₀)/δ),
  with I the integral of a C² smoothstep, θ = 0.45, δ = 0.6; R₀ = (outer turning point of U_BOD at the energy
  guess) + 1.2; the box ends where the outgoing BO-channel WKB wave has decayed by e⁻⁴⁰ along the path (+0.3).
  The scaled problem has a purely discrete spectrum for 0 < θ < π/3 (the scaled inverted quartic is a confining
  complex quartic), so resonances of any width are exposed.
* s: sine DVR on (0, S), S = 10, Δs = 0.14 (70 points).
* Eigenvalues: ARPACK shift-invert with an exact block-tridiagonal LU of H − σ (dense s blocks), σ = E_guess − 0.05i;
  the ground resonance is the lowest-Re eigenvalue localised in the well; it is refined by re-factorising at
  E − 10⁻³i and four inverse iterations. Residuals ‖Hx − Ex‖ ≤ 1×10⁻¹¹.
* Width: Γ from the exact discrete flux identity in the unscaled region,
  Γ = −2H_{n,n+1} Im(Ψ_n^†Ψ_{n+1})/Σ_{i≤n}|Ψ_i|² (n between the outer turning point and R₀; constant in n to ≤ 5×10⁻¹³, typically 10⁻¹⁴),
  and from Im E (Rayleigh quotient). Both agree to ≤ 2.5×10⁻⁷ for Γ > 10⁻¹¹; below that Im E reaches its round-off
  floor (~10⁻¹⁷ absolute in Γ: 5×10⁻⁴ relative at Γ = 5×10⁻¹⁴, 16% at 5×10⁻¹⁶), while the flux value stays stable
  (θ, R₀, δ variations agree to 10⁻⁹ at Γ = 3.2×10⁻¹⁸). No floor of the flux method was reached.
* Grids h = 0.02, 0.01, 0.005 (× e^{−1/3} for e ≠ 1) at fixed physical R₀ and box; Richardson extrapolation of ln Γ
  and ε (two levels, R2). Matrix dimensions up to 3.6×10⁵.

### Method B (independent cross-check): complex absorbing potential
Separately written code: fourth-order finite differences of −½(ψ'' + ψ'/r) with even reflection at r = 0 (a
non-symmetric matrix), spectral basis of odd harmonic-oscillator functions in s (ω₀ = 2.5, 50 or 64 functions, exact
Galerkin matrices), CAP −iη((r − R_c)/(r_max − R_c))^p beyond R_c = r₃ + 1.0 (or + 1.5). The eigenvalue is followed
along 49 values of η ∈ [10⁻¹, 10⁵]; the estimate is the first-order corrected E₁ = E − η dE/dη at min|η dE₁/dη|,
searched only where the CAP is effective (d ln|Im E|/d ln η < 0.5). Settings a: h = 0.02, 50 functions, p = 3,
length 8; b: h = 0.015, 64 functions, p = 4, length 10. Guess energies come from the 1D BO model only.

### 1D models: complex-plane shooting
−½F'' + U F = EF with a Frobenius series at r → 0 (F ∝ r^a, a(a − 1) = 2c₂: a = ½ for BO, 1.1124 for BOD), DOP853
(rtol 10⁻¹²) along the real axis beyond the outer turning point and then along the ray R₁ + te^{0.45i}, where the
outgoing solution decays super-exponentially; resonance condition F(t_max) = 0; Newton iteration. Changing the ray
angle (0.35), R₁ (+0.6) and rtol (10⁻¹³) changes Γ by ≤ 4×10⁻⁸ for Γ > 10⁻¹⁴. Floor: ~10⁻²¹ absolute in Γ (rounding of
Re E combined with the finite-difference derivative), i.e. 5×10⁻⁴ relative at Γ_BO = 2.3×10⁻¹⁸.

### WKB
Langer-corrected potentials (c₂ → c₂ + 1/8): σ₀r + λr⁴/4 for BO and σ₀r + 3/(16r²) + λr⁴/4 for BOD. Bohr–Sommerfeld
∫p dr = π/2 (the m = 0 phase gives (n + ½)π with the integral starting at r = 0), period T, barrier action
S = 2∫|p|dr, Γ_Kemble = ln(1 + e^{−S})/T. Also: S and T of the BO barrier at the exact 2D energy, and the deep-well
exponent S_asym = (2π√2/3)σ₀/|λ|^{1/2} = π√2 x.

## 4. Convergence

| test | result |
|---|---|
| h → h/2 ratio of ln Γ differences | 3.990–4.010 at all 26 points (clean h²) |
| Richardson R1 vs R2 | ≤ 2.8×10⁻⁸ in ln Γ, ≤ 1.1×10⁻⁹ in ε |
| finest grid h = 0.0025 (Λ = −0.173, −0.03) | ln Γ changes by ≤ 3×10⁻¹⁰ |
| θ = 0.35 / 0.55 (vs 0.45) | \|Δ ln Γ\| ≤ 1.3×10⁻⁹ |
| R₀ + 0.8, ECS transition δ = 1.0, longer exterior (e⁻⁵⁵) | \|Δ ln Γ\| ≤ 4×10⁻⁹ |
| Δs = 0.10, S = 12 | \|Δ ln Γ\| ≤ 1×10⁻¹¹ |
| all variants, ε | \|Δε\| ≤ 3×10⁻¹¹ (narrow), 7×10⁻¹⁰ (Λ = −1) |

Variants were run at Λ = −1, −0.173, −0.072, −0.03, −0.010, −0.0065 (h = 0.01/0.005 + Richardson each). The quoted
uncertainty of each Γ (column `Gamma_rel_err`) is |R1 − R2| plus the largest variant shift at the neighbouring
convergence points, 4×10⁻⁹–3×10⁻⁸ in ln Γ. A conservative statement, set by the independent method, is ≤ 10⁻⁶.

**CAP cross-check** (`C171_cap_crosscheck.csv`):

| Λ | Γ (ECS) | Γ (CAP, b) | rel. diff. | CAP stationarity error | ε(CAP) − ε(ECS) |
|---|---|---|---|---|---|
| −0.41618 | 0.21900701 | 0.21900697 | 1.8×10⁻⁷ | 2.6×10⁻⁸ | −1.4×10⁻⁸ |
| −0.11174 | 6.1159274×10⁻³ | 6.1159275×10⁻³ | 2.4×10⁻⁸ | 4.6×10⁻⁸ | −1.0×10⁻⁸ |
| −0.03000 | 4.5653524×10⁻⁷ | 4.5653553×10⁻⁷ | 6.3×10⁻⁷ | 4.2×10⁻⁷ | −1.2×10⁻⁸ |
| −0.01554 | 1.0139834×10⁻¹⁰ | 1.0139840×10⁻¹⁰ | 5.2×10⁻⁷ | 2.9×10⁻⁷ | −1.2×10⁻⁸ |

(setting a agrees to 2×10⁻⁵, 6×10⁻⁶, 7×10⁻⁷, 1×10⁻⁴, within its own larger stationarity errors.)

## 5. Results

### 5.1 Γ(Λ), e = 1 (full table with h-ladder, WKB and prefactor columns: `C171_gamma_table.csv`)

| Λ | x | ε | Γ | Γ_BO/Γ | Γ_BOD/Γ | b | dS_BO(ε)/dx |
|---|---|---|---|---|---|---|---|
| −1.00000 | 1.000 | 1.66884204 | 6.122883e−01 | 1.020 | 2.46 | 1.825 | – |
| −0.80319 | 1.116 | 1.67934153 | 4.956486e−01 | 1.017 | 2.58 | 1.829 | – |
| −0.64512 | 1.245 | 1.69319004 | 3.910363e−01 | 1.013 | 2.74 | 1.850 | – |
| −0.51816 | 1.389 | 1.71005370 | 2.986957e−01 | 1.007 | 2.95 | 1.897 | – |
| −0.41618 | 1.550 | 1.72947579 | 2.190070e−01 | 1.001 | 3.23 | 1.972 | – |
| −0.33427 | 1.730 | 1.75082194 | 1.524013e−01 | 0.994 | 3.61 | 2.079 | – |
| −0.26849 | 1.930 | 1.77323040 | 9.919053e−02 | 0.986 | 4.16 | 2.219 | 2.489 |
| −0.21565 | 2.153 | 1.79560020 | 5.928940e−02 | 0.976 | 4.97 | 2.391 | 2.619 |
| −0.17321 | 2.403 | 1.81667566 | 3.185922e−02 | 0.965 | 6.18 | 2.588 | 2.762 |
| −0.13912 | 2.681 | 1.83528035 | 1.504656e−02 | 0.954 | 8.0 | 2.792 | 2.911 |
| −0.11174 | 2.992 | 1.85065211 | 6.115927e−03 | 0.941 | 10.7 | 2.986 | 3.058 |
| −0.08975 | 3.338 | 1.86266593 | 2.102210e−03 | 0.929 | 14.5 | 3.155 | 3.192 |
| −0.07208 | 3.725 | 1.87174627 | 6.019318e−04 | 0.915 | 19.4 | 3.293 | 3.310 |
| −0.05790 | 4.156 | 1.87855251 | 1.413879e−04 | 0.901 | 25.4 | 3.406 | 3.411 |
| −0.04650 | 4.637 | 1.88370072 | 2.675357e−05 | 0.886 | 32.4 | 3.499 | 3.499 |
| −0.03735 | 5.174 | 1.88765319 | 3.988950e−06 | 0.871 | 40.6 | 3.579 | 3.576 |
| −0.03000 | 5.774 | 1.89072754 | 4.565352e−07 | 0.854 | 50.5 | 3.648 | 3.645 |
| −0.02410 | 6.442 | 1.89314095 | 3.891490e−08 | 0.836 | 62.5 | 3.710 | 3.707 |
| −0.01935 | 7.188 | 1.89504708 | 2.387227e−09 | 0.818 | 77.3 | 3.766 | 3.763 |
| −0.01554 | 8.021 | 1.89655878 | 1.013983e−10 | 0.799 | 95.7 | 3.818 | 3.815 |
| −0.01249 | 8.950 | 1.89776119 | 2.855577e−12 | 0.779 | 119 | 3.864 | 3.862 |
| −0.01003 | 9.986 | 1.89871965 | 5.079043e−14 | 0.759 | 148 | 3.907 | 3.905 |
| −0.00805 | 11.142 | 1.89948492 | 5.403573e−16 | 0.738 | 185 | 3.946 | 3.945 |
| −0.00647 | 12.433 | 1.90009670 | 3.235812e−18 | 0.717 | 233 | 3.966 | 3.965 |

Relative uncertainty of Γ: 4×10⁻⁹–3×10⁻⁸ (internal), ≤ 10⁻⁶ (independent method); ε: ≤ 4×10⁻⁹.
ε_BO = 1.565–1.804 and ε_BOD = 2.196–2.555 over the same range (CSV).

### 5.2 Local slope and the deep-well formula
b(Λ) is computed by second-order differences of ln Γ on the x grid (`b_exact`). It departs from π√2 = 4.443 for two
reasons.

1. **Resonance near or above the barrier top (large |Λ|).** The Langer BO barrier top is 1.125(1.5x²)^{1/3}; it falls
   below ε for x ≲ 1.56 (|Λ| ≳ 0.41). There Γ = O(0.1–1) and b ≈ 1.8–2.0 (shaded band in the PNG).
2. **Energy dependence of the barrier action (all |Λ|).** The resonance energy ε ≈ 1.67–1.90 is the zero-point energy
   of the r motion in the linear well σ₀r (plus non-adiabatic corrections) and does not vanish as Λ → 0. Expanding
   S(E) = 2∫√(2(U − E))dr for U = σ₀r − |Λ|r⁴/4,
   S(ε) = π√2 x − √2·[B(1/6,1/2)/3]·√(r₀/σ₀)·ε + … = π√2 x − 3.78 ε x^{1/3} + …, with r₀ = (4σ₀/|Λ|)^{1/3}, so
   b ≈ π√2 − 1.26 ε x^{−2/3}. With the full S(ε) of the plain BO barrier at the exact energy, dS/dx agrees with
   b_exact to ≤ 0.003 for x ≥ 4 (table). The approach to 4.443 is slow: the WKB continuation (exact ε, extrapolated to
   ε₀ = 1.9026 for Λ → 0) gives b = 4.28 at x = 60, 4.39 at x = 300 and 4.42 at x = 1000.
3. **Prefactor.** Γ/(T⁻¹e^{−S_BO(ε)}) = 0.75–0.80 for x = 2–4 and 0.757 at x = 12.4 (slowly varying), whereas
   A_eff = Γe^{π√2 x} = 52, 3.6×10³, 6.3×10⁴, 9.4×10⁵, 3.2×10⁶ at x = 1, 3, 5.77, 10, 12.4. A form
   Γ ≃ A exp(−π√2 x) with constant A is therefore valid only asymptotically; at fixed coupling the factor
   exp(3.78 ε x^{1/3}) (≈ 10³ at x = 10) must be kept.

The plain Langer–BO WKB (Bohr–Sommerfeld energy 1.71–1.84) gives Γ 6–19% below the exact value for x ≥ 5,
partly because its energy lies between ε_BO and ε.

### 5.3 Born–Oppenheimer accuracy
* Plain BO: ε_BO − ε = −0.096 to −0.103 (−5.0 to −6.2%); Γ_BO/Γ from 1.02 (Λ = −1) through 0.94 (Λ = −0.11), 0.85 (Λ = −0.03),
  0.76 (Λ = −0.010) to 0.72 (Λ = −0.0065); b_BO − b = +0.02 to +0.04. The slow drift follows from the energy error:
  ΔS ≈ 3.78·0.10·x^{1/3}.
* BO + diagonal correction: ε_BOD − ε = +0.53 (Λ = −1) to +0.65 (Λ → 0); Γ_BOD/Γ = 2.5 to 233; b_BOD lower by 0.2–0.9.
  The diagonal term 3/(16r²) is ≈ 0.2–0.3 in the well (r ≈ 1) and is almost cancelled by the second-order
  non-adiabatic coupling to the excited w channels (gap 2er ≈ 2). In the well the w frequency er is comparable to
  the r-motion scale for every e (the scaling makes this e-independent), so the adiabatic expansion has no small
  parameter there; only the barrier region r ≈ |Λ|^{−1/3} is adiabatic. The 1D model
  (BO + diagonal correction) should not be used for quantitative widths; the plain BO model, or better the BO
  barrier at the exact (2D) energy, is accurate to tens of per cent.

### 5.4 Thresholds Γ(Λ_c) N = ln 2 (`C171_thresholds.csv`; cubic spline of ln Γ in x)

| N | Λ_c (exact) | Λ_c (BO) | Λ_c (BOD) | \|Λ_c\|^{1/2} ln N | local d\|Λ_c\|^{1/2}/d(1/ln N) |
|---|---|---|---|---|---|
| 10 | −0.22952 ± 4×10⁻⁵ | −0.23151 | −0.11387 | 1.103 | 0.52 |
| 300 | −0.091379 ± 4×10⁻⁶ | −0.092657 | −0.053875 | 1.724 | 0.94 |
| 10⁴ | −0.052518 ± 4×10⁻⁶ | −0.053295 | −0.033818 | 2.111 | 1.29 |
| 10⁶ | −0.031227 ± 1×10⁻⁶ | −0.031697 | −0.021578 | 2.441 | 1.64 |
| 10⁸ | −0.020976 | −0.021290 | −0.015187 | 2.668 | 1.90 |
| 10¹⁰ | −0.015166 | −0.015389 | −0.011357 | 2.836 | 2.10 |
| 10¹² | −0.011525 | −0.011691 | −0.008856 | 2.966 | 2.27 |
| 10¹⁴ | −0.009080 | −0.009208 | −0.007122 | 3.072 | 2.40 |
| 10¹⁶ | −0.007353 | −0.007454 | – | 3.159 | 2.52 |

For general e the condition reads e^{2/3}F(Λ_c)N = ln 2 and λ_c = e²Λ_c; for e = 0.8733:
λ_c = −0.1818, −0.0709, −0.0405, −0.0240 (N = 10, 300, 10⁴, 10⁶); for e = 0.5: −0.0705, −0.0250, −0.0140, −0.0082.

**Linearity in 1/ln N.** From ln(N/ln 2) = S(ε, x_c) + ln T − ln(0.76), |Λ_c|^{1/2} = 1/x_c = π√2/ln N·[1 + O((ln N)^{−2/3})].
Hence |Λ_c|^{1/2} is linear in 1/ln N asymptotically, with slope π√2 and zero intercept, but in the range of the
computed widths the local slope is 0.5–2.5 and a straight-line fit over N = 10⁴–10¹⁶ gives slope 1.76, intercept
+0.044 (max residual 0.006). Continuing with the WKB barrier at the exact energy (anchored at x = 12.4):
|Λ_c|^{1/2} ln N = 3.75 (N = 10⁵⁰), 3.98 (10¹⁰⁰), 4.21 (10³⁰⁰), local slope 3.35, 3.70, 4.06 — still short of π√2.

### 5.5 SM-like slope (orientation; pure zero modes, no matching)
e = 2·1.31/3 = 0.8733 (σ₀ = 1.31):

| λ | Λ = λ/e² | Γ L | ε L | Γ_BO L | Γ_BOD L | survival exp(−ΓN), N = 10⁶ |
|---|---|---|---|---|---|---|
| −0.010 | −0.013111 | 6.0109×10⁻¹² | 1.73369 | 4.71×10⁻¹² | 6.80×10⁻¹⁰ | 0.999994 |
| −0.013 | −0.017044 | 3.6642×10⁻¹⁰ | 1.73228 | 2.96×10⁻¹⁰ | 3.20×10⁻⁸ | 0.99963 |

Both values were also computed directly at e = 0.8733 without scaling (agreement 7×10⁻¹², 5×10⁻¹²). The deep-well
exponent π√2 x is 38.80 and 34.03; the actual −ln(Γ/e^{2/3}) is 25.75 and 21.64.

### 5.6 Holonomy (compactness of w)
The truncation treats w as non-compact; the fundamental domain is e|w_i| < π. Probability inside the well (r < r_b,
barrier-top radius), from the converged resonance wavefunctions (`C171_holonomy_tails.csv`, `holonomy` in JSON;
thresholds S > (π/2)e^{−2/3} and πe^{−2/3} in scaled units):

| e | P(e\|w\| > π/2) | P(e\|w\| > π) |
|---|---|---|
| 1 | 0.229–0.260 | 1.8–2.6×10⁻³ |
| 0.873 | 0.165–0.190 (0.188 at λ = −0.010) | 5.5–8.1×10⁻⁴ (8.0×10⁻⁴ at λ = −0.010) |
| 0.5 | 0.019–0.024 | 3.2–5.9×10⁻⁷ |

(ranges over Λ = −1 … −0.0065; the narrow-resonance values are at the upper end). The direct e = 0.8733 runs with
physical thresholds reproduce the scaled numbers. The weight beyond the edge of the fundamental domain is per mille
for e ≈ 0.87 and negligible for e = 0.5, so the non-compact truncation is harmless at that level; the wavefunction is,
however, not deep inside the domain for e ≈ 0.87–1 (about a fifth of the probability lies beyond half of it).

## 6. Limitations
* Pure zero-mode problem: no one-loop matching of the nonzero modes, no fermions; the results test the zero-mode
  resonance mechanism, not the SM numbers.
* The compactness of the holonomy and the nonzero-momentum Φ modes that become light near e|w| = π are not included;
  their weight is quantified in §5.6.
* Γ was computed down to 3.2×10⁻¹⁸; the Rayleigh-quotient Im E has a ~10⁻¹⁷ absolute floor, which the flux identity
  avoids. The 1D shooting has a ~10⁻²¹ floor (irrelevant here). Asymptotic statements beyond x = 12.4 (|Λ| < 0.0065,
  N > 10¹⁷) rest on the WKB barrier action at the exact energy, whose prefactor ratio to the exact Γ varies slowly
  (0.80 → 0.757 over x = 3–12).
* Threshold uncertainties include an interpolation estimate (10% of the cubic-versus-linear spline difference).

## 7. Files
* `C171_results.json` — all numbers: table (per Λ: ε, Γ, h-ladder, Richardson, errors, 1D and WKB values, slopes,
  prefactors, holonomy, grids), convergence variants, finest-h runs, thresholds (+ general e, fit, asymptotic), SM-like,
  scaling checks, CAP comparison, WKB asymptotics, e → 0 check.
* `C171_gamma_table.csv`, `C171_thresholds.csv`, `C171_cap_crosscheck.csv`, `C171_holonomy_tails.csv`.
* `checks.json` (13 checks, all pass), `C171_lnGamma_vs_x.png` (ln Γ and b against x).
* `raw/` — per-task JSON (main, conv, fine, direct, cap, oneD_all), and the run logs `scan_log*.txt`, `cap_log*.txt`,
  `analyze_log.txt`; both are bundled in `C171_point_records.jsonl` (`python -B unbundle_records.py` in the supplement
  root restores them).
* Code: `c171_core.py` (method A, 1D shooting, WKB), `c171_scan.py` (driver, ≤ 4 workers, resumable),
  `c171_cap.py` (method B), `c171_analyze.py` (outputs). Run order: `python -B c171_scan.py 4`,
  `python -B c171_cap.py 3`, `python -B c171_analyze.py` (set OPENBLAS_NUM_THREADS=1).
