# S-group figure captions (fig-shape, 2026-10-15)

English caption drafts that match the panels as generated (PRD target 50–90 words; first sentence = result). Every
number below is read by the figure scripts from `data/shape/` (see `data/SOURCES_shape.json`). After each draft:
the tex caption sentences that must change, and checks for the section authors. Notation: f₀ (zero-mode share),
r_T, τ = ht only for Higgs (shape time is written ht, h|t_x|), Γ not used here.

---

## prd_trace — `sections/s03_trace.tex`, `fig:pa-trace` (figure*, 7.0 in, 3 panels)

**Draft.** A conformal sector cannot turn a flat anisotropic universe around. (a) Largest Ḣ/S at a zero of H when all
other matter has non-negative trace, Eq. (pa-rate), versus the anomaly strength y, for ϱ = a/c of a conformal scalar
(1/3), a Dirac fermion (11/18), ϱ = 1 and a Maxwell field (31/18); every curve stays below −5/54. (b) Least negative
extra trace needed for a turning rate ν = Ḣ/S at ϱ = 31/18, Eq. (pa-budget). (c) Least negative CFT energy ζ_min,
Eq. (pa-joint), when a positive-energy component with equation of state w supplies the trace; shaded: w ≥ 1/3.
(≈85 words)

**tex changes:** none required; the existing PRD-A caption matches the panels (labels use M̄_P as in the text).

---

## prd_casimir_shape — `sections/s04_casimir.tex`, `fig:pa-casimir-shape` (figure*, 7.0 in, 3 panels)

**Draft.** The Standard Model gives negative Casimir energy only with antiperiodic fermions, and that energy has no
shape minimum. (a) Signed Casimir weight N_C of the high-energy Standard Model plus the static graviton on the cubic
torus at trivial holonomy, for periodic (P) or antiperiodic (A) fermion cycles; filled: the cube is stationary.
(b) Fractional change of the AAA Casimir energy density on the slice G = exp(2β_dX_d + 2β_aX_12); aspect deformations
lower it. (c) Gain of a linear shape mode versus its stiffness for the Maxwell toy (circles) and the Standard Model
with free fields (open squares), with f₀ = 0.30 (filled squares) and at next order (triangles); shading: κ ≥ 8/5,
lines: guaranteed gains 4.50 (free fields) and 6.72 (f₀ = 0.30). (≈110 words; the last clause of (c) can go if space
is short)

**tex changes (s04 l.85):** replace "(c)~Gain of a linear shape mode across the turning solution versus its normalized
stiffness (Sec.~\ref{sec:shape})." by "(c)~Gain of a linear shape mode across the turning solution versus its
normalized stiffness for the Maxwell toy (circles) and the Standard Model with free fields (open squares), with
$\fzm=0.30$ (filled squares) and at next order (triangles); shading: $\kappa\geq8/5$ (Sec.~\ref{sec:shape})."
Panel (c) shows the gains 674 (f₀ = 0.30), 529 (next order), 216 (free), 30.6 (toy) quoted in s07 l.20, l.91.

---

## prd_emptiness_gate — `sections/s06_gate.tex`, `fig:s-gate` (figure*, 7.0 in, 2 panels)

**Draft.** A Casimir turning point must be almost empty. (a) Ratio R of the comoving energy of visible photons produced
by a reheating coupling (c_v = 0.01) in the remote contraction of the time-symmetric history to the comoving Casimir
energy, including the zero-mode energy (f₀ = 0.30), for a photon vacuum declared at the jth extremum of the inflaton
oscillation (97 and 401 modes); the turn needs R < 1. (b) Excitation Σ|n|N_n per non-conformal degree of freedom for a
natural state declared T e-folds before the turn (hL_b = 0.0044); horizontal lines: the budget, Eq. (s-emptiness),
shared by three such degrees of freedom at f₀ = 0.30 (solid) and for free fields (dotted); vertical line:
T = ln(2π/hL_b). (≈100 words)

**tex changes (s06 l.83):** in (a) insert "including the zero-mode energy," after "to the comoving Casimir energy";
in (b) replace "horizontal line: the budget~\eqref{eq:s-emptiness} shared by three such degrees of freedom at
$\fzm=0.30$" by "horizontal lines: the budget~\eqref{eq:s-emptiness} shared by three such degrees of freedom at
$\fzm=0.30$ (solid) and for free fields (dotted)".
**Notes:** the two superhorizon state choices (instantaneous ground state, conformal vacuum) differ by ≤ 0.001 e-fold
and coincide to the line width (a12 l.58 refers to "the two choices compared in Fig. s-gate(b)"; the text can say
"indistinguishable in Fig. ..."). The curve is the C56 SM run (hL_b = 0.00483) shifted by ln(0.00483/0.0044) = 0.093
in T, the exact de Sitter scaling checked in C153 (`scaling_check`); the marked crossing T = 6.91 and T_max = 7.26 are
the archived C153 values. Panel (a) labels 94.0 and 96.5 are log₁₀R at j = 1 and 85 (401 modes).

---

## prd_turning_state — `sections/s08_quantum_shape.tex`, `fig:s-turning` (figure*, 7.0 × 3.9 in, 2×2)

**Draft.** A shape state that is real at the turn leaves it at the quantum floor; reality at another moment costs the
factor G(|t_x|). (a) 1σ Wigner ellipses of a Standard Model aspect mode (f₀ = 0.30) in its natural state at ht = 0,
±0.5 and ±6, in units of the outgoing widths; past (dashed) and future (solid) are mirror images. (b) Rms width of the
aspect and angle modes versus time for f₀ = 0.30 (solid) and free fields (dashed). (c) Outgoing variance in units of
the floor versus the turning-point width, Eq. (s-family); squares: natural widths (filled: f₀ = 0.30, open: free
fields). (d) Lower bound G(|t_x|), Eq. (s-Gtx), for κ_E = 19.32 and 18.0 (dashed), single-branch bound G₊ for
t_x > 0 (dotted), onset of freezing, and the two-branch survival thresholds h|t_*| = 0.237 (99%) and 0.355 (90%,
shaded). (≈120 words; for 90 words drop the clause on mirror images and the onset of freezing)

**tex changes (s08 l.152):** "(a) $1\sigma$ Wigner ellipses ... at $ht=0$, $\pm1$ and $\pm6$" → "at $ht=0$, $\pm0.5$
and $\pm6$, in units of the outgoing widths" (the f₀ = 0.30 state at the turn is 15 times narrower than the outgoing
width, so ±1 was replaced by ±0.5 to keep the ellipses readable; ±6 is frozen and drawn as the diagonals).
"(b) Rms width of the aspect and angle modes versus time." → add "for $\fzm=0.30$ (solid) and free fields (dashed)".
"(c) ... with the natural widths marked." → "with the natural widths marked (filled: $\fzm=0.30$, open: free fields)".
"(d) ... at leading and next order, with the single-branch bound and the survival thresholds $t_*$." → "for
$\kappa_E=19.32$ and $18.0$ (dashed), with the single-branch bound $G_+$ (dotted) and the two-branch survival
thresholds $h|t_*|=0.237$ ($99\%$) and $0.355$ ($90\%$, shaded)".
**Numbers drawn (all from data):** σ_E = 0.104 (f₀ = 0.30; canonical δ = 2.0653×10⁻⁴, X = 2282.49), 0.059 (free
fields: δ_free = δ (1−f₀)^{3/4}, reproduces the C63 value 0.0587 to 0.04%), σ_T2 = 0.0065/0.0066; natural aspect
widths +0.05% (f₀ = 0.30) and +0.12% (free) above the floor; G markers 2.395, 8.263, 45.10, 313.8 (C137c) and 385.65
at the freezing onset h|t| = 1.093; 2𝒢 = 1348 and 1058. t_*(90%) = 0.3547 is reproduced in the script by the D125
interpolation (run02 node 0.35, N = 2048; run01 node 0.40, N = 512); the JSON has no central value for it.

---

## prd_nonlinear_survival — `sections/s08_quantum_shape.tex`, `fig:s-survival` (figure, 3.375 × 4.2 in, 2 stacked)

**Draft.** Nonlinearity ends the shape when the outgoing deformation is large. (a) Surviving fraction when the natural
turning-point state (f₀ = 0.30) is scaled by s, with 256 and 2048 draws per point, from an independent implementation
(diamonds), on the plateau background (crosses), at next order (dashed) and for free fields (dotted); top axis:
outgoing linear amplitude σ_lin; band: s₉₉ ≃ 2.8–2.95. (b) Two-branch survival S₂ of the optimal real Gaussian versus
the moment of reality h|t_x|, with an independent implementation (line), the plateau background (crosses), states that
are never real (∞), and the thresholds 0.237, 0.355 and 0.631 for 99%, 90% and 50%. Wilson 95% intervals.
(≈100 words)

**tex changes (s08 l.276):** replace the caption by the draft (or keep its structure and add: in (a) "next order
(dashed) and free fields (dotted); top axis: outgoing linear amplitude $\sigma_{\rm lin}$; $s_{50}=8.2$"; in (b) "with an
independent implementation (line), states real at $t_x<0$ on the plateau background (crosses) and states that are
never real at the right").
**Check for PRD-D (s08 l.264):** "survival 1.000, 0.987, 0.818, 0.365, 0.067 and 0.0015 for s = 1, 3, 5, 10, 20 and 40,
with 2048 draws each" are the R16/T19 *independent* values (black diamonds); the production runs have N = 256 at
s = 1–40 (C142: 1.000, 0.996, 0.973, 0.926, 0.824, 0.609, 0.367, 0.0625, 0) and N = 2048 only at s = 2.5, 3, 3.5
(run02). Suggest "in an independent implementation with 2048 draws each". The band edges are run02 log-linear
s₉₉ = 2.756 and T19 s₉₉ = 2.948; s₅₀ = 8.18 (T19). Plateau crosses: C160 (0 of 2560 fates changed).

---

## prd_fossil_shape — `sections/s08_quantum_shape.tex`, `fig:s-fossil` (figure*, 7.0 in, 3 panels)

**Draft.** The frozen fundamental domain of the torus is an almost rectangular box. (a) Outgoing aspect coordinates of
2048 draws from the natural turning-point state with f₀ = 0.30 (orange) and of 2048 draws for free fields (gray), each
evolved through the full nonlinear equations; circles: 1, 2 and 3σ_E of the linear prediction (solid) and at next
order (dashed, σ_E lower by 0.900); gray lines: two equal sides. (b) ln(L_max/L_min) and (c) largest deviation of an
edge angle from 90°, for the nonlinear draws (steps) and the linear Gaussian (lines; dashed: next order). (≈85 words)

**tex changes (s08 l.247):** "circles: ... at next order ($\sigma_E=0.0934$)" can stay (dashed circles); "(b) Ratio of
the longest to the shortest side" → "(b) Logarithm of the ratio of the longest to the shortest side,
$\ln(L_{\max}/L_{\min})$," and add "(dashed in (b): next order)".
**Numbers drawn:** medians 0.164 (18%, f₀ = 0.30, nonlinear), 0.096 (10%, free), next-order linear 0.149 (16%);
median angle deviation 0.67°. The next-order circles use the convention-free ratio 0.9000 (C153) times the linear
σ_E of these draws (0.104, C76 convention), i.e. 0.0937; the canonical next-order floor is 0.0934.
**Check for PRD-D (Table s-fossil caption, s08 l.200):** "The natural aspect widths exceed the floor by 0.05%
(f₀ = 0.30), 0.12% (free fields) and 0.6% (toy)": C63 gives 0.87% for the toy in variance
(`NC100_toy.modes.aspect.natural_over_floor_var` = 1.00875; 0.44% in rms). 0.05% and 0.12% are confirmed.

---

## prd_primordial — `sections/s09_spectra.tex`, `fig:c-primordial` (figure*, 7.0 in, 2 panels)

**Draft.** The turning point leaves the plateau predictions intact. (a) Scalar tilt of the plateau potential versus
the number N_* of e-folds between pivot exit and the end of inflation; band: weak visible decay (N_* = 50.33,
n_s = 0.9617) to instant reheating (55.72, 0.9653); gray: 68% intervals from SPT, ACT and Planck CMB data and from
ACT and Planck with DESI DR2. (b) Tensor-to-scalar ratio r_T, with the BICEP/Keck bound r_T < 0.036 (95%, k = 0.05
Mpc⁻¹) and the LiteBIRD target δr_T = 10⁻³. (≈75 words)

**tex changes:** none required (PRD-E caption matches). External values are transcribed in
`data/shape/external_primordial_bands.json` (as v11; versions not re-checked by fig-shape).

---

## prd_fossil_sky — `sections/s10_sky.tex`, `fig:c-sky` (figure*, 7.0 in, 2 panels; PRD-E's two-panel layout)

**Draft.** The shortest cycle decides whether the microwave background reads the shape. (a) Kullback–Leibler divergence
of 55 computed fossils from the closest circle-free cube (temperature and E modes, 2 ≤ ℓ ≤ 80, full sky) versus their
shortest cycle u; dotted: readability criterion D = 4.5; green band: readout edges u_max = 1.021–1.039 of the 15
interpolated shapes; gray: matched circles. (b) Probability that a fossil from the predicted distribution is readable
with no matched circles (solid) and given that circles are absent (dashed), versus L₀, for σ_E = 0.104 (thick; light
band: edge-interpolation bounds) and 0.094 (thin); bar: P_joint ≥ 0.1. (≈100 words)

**tex changes (s10 l.24):** add after "versus their shortest cycle $u$" "; band: readout edges of the 15 interpolated
shapes"; replace "for $\sigma_E=0.104$ and $0.094$" by "for $\sigma_E=0.104$ (thick; light band: interpolation bounds of
the readout edges) and $0.094$ (thin); bar: $P_{\rm joint}\geq0.1$".
**Numbers drawn:** peaks 0.344 at L₀ = 1.075 and 0.377 at 1.065 (archived refined maxima), window [1.018, 1.183]
(σ_E = 0.104), integrated measure 0.042 for both σ_E (0.0422, 0.0421). The v11 Fisher and C79 panels are not used.

---

## Relabelling for readers (2026-10-16, after the coordinator's check)

All internal labels were removed from the figure lettering (checked by extracting the text of every PDF with PyMuPDF;
no run, review, task or derivation identifiers and no "N = ..." labels remain). Changes:
- prd_nonlinear_survival: legends "N = 256", "N = 2048", "independent", "plateau history" (a) and "N = 256–512",
  "N = 2048", "blind check", "plateau history" (b) → "256 draws", "2048 draws", "independent implementation",
  "plateau background" and "256–512 draws", "2048 draws", "independent implementation", "plateau background".
- prd_casimir_shape: legend "toy" → "Maxwell toy"; "(free)" → "(free fields)".
- prd_emptiness_gate: "97 modes", "401 modes" → "97 photon modes", "401 photon modes".
- prd_turning_state: "(free)", "angle, free" → "(free fields)", "angle, free fields".
- prd_primordial: "CMB + DESI DR2 BAO (P-ACT-LB2)" → "ACT+Planck with DESI DR2 BAO"; "CMB only (SPT+ACT+Planck)" →
  "CMB: SPT+ACT+Planck"; "excluded by BK18" → "excluded by BICEP/Keck"; "instant", "c_v = 0.01" → "instant
  reheating", "weak decay".
- prd_fossil_sky: axis "D_prof" → "D"; "read edges" → "readout edges".
- prd_trace, prd_fossil_shape: no change needed.

**tex caption sentences no longer accurate after the relabelling (not edited here):**
- `sections/s08_quantum_shape.tex` l.276 (`fig:s-survival`): "from the production runs and an independent
  implementation" → "with $256$ and $2048$ draws per point and from an independent implementation" (the figure no longer
  says "production"; the legend gives the number of draws). Panel (b) also shows the independent implementation as a
  line; "the plateau-background check" matches the new legend.
- No other tex caption refers to the changed words (s04, s06, s09, s10 captions remain accurate; s09's "ACT and Planck
  with DESI DR2 baryon acoustic oscillations" matches the new band label). The earlier per-figure "tex changes" above
  still apply.
