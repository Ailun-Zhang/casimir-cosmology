# V4 — shape side: specification given to the blind reimplementation

Summary of the written specification of 2026-10-15 (with its addendum), reduced to definitions, inputs and pass
criteria. Report: Sec. 19.6.

**Blindness.** No access to the derivations, code or results of the shape calculation of the third stage. Allowed:
the earlier draft (model and conventions), the method and definition sections (not the tables) of two earlier
derivations, the literature. Results in `BLIND_RESULTS.json` before the comparison.

**To compute.** (1) Real-moment theorem: for the linear shape modes in the leading background of the draft, if
⟨{δβ, δP}⟩(t_x) = 0 the larger aspect fossil variance is ≥ G(|t_x|)·floor; compute G(|t_x|) for |t_x| = 0.1, 0.38,
0.5, 1/h and the freeze-out start, and the floor from the closed form
σ_floor² = Γ(7/4)Γ(3/4) / [2D |Γ(5/4 + iμ)|² M_P² L_b³ h]. (2) Nonlinear survival with an own fully nonlinear
five-dimensional shape integration (Lagrangian and twisted Ewald forces of the draft; leading cosh background;
zero-mode share 0.30) for Wigner samples of the optimal-width Gaussian that is real at t_x,
t_x ∈ {−1, −0.5, −0.38, −0.2, 0, +0.2, +0.5}/h; t_*(90 %). (3) Next order: G(t_x) and the floor at share ≈ 0.28,
κ_E = 18.0 (two points). (4) Emptiness gate: the bound on the number of lowest quanta from Σ|n − ϑ| < 0.2666 N_C
for shares 0.30/0.28/0.258; the preparation window ln(2π/(hL_b)) for hL_b = 0.0044 and 0.0048.
**Addendum.** Every quantity at the 99 % level (t_*(99 %), s₉₉) with at least N = 2048 samples per point and binomial
errors.

**Pass criterion.** Agreement within the statistical and input errors (report: G and floor to 0.3 %, explained by the
input X; t_* within its binomial error).
