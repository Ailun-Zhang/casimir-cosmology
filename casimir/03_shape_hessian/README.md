# 03_shape_hessian — trace identity, no-go generalisations, gain and the next-order shape Hessian

**Physics.** For any mixture of free massless fields the shape Laplacian of the Casimir energy is Tr_sh Hess E_C =
(8/3) E_C (eigenvalue 4p(2p+3)/3 at p = −2), so negative Casimir energy has no shape minimum; with finite-volume matter
ΔF|_T = (8/3)(U − ⟨q⟩/L) − T⁻¹ Σ_A Var f_A (Gibbs / fixed entropy, zero modes, holonomies; D55). The linear shape mode
β̈ + 3Hβ̇ − κh² sech²(2ht) β = 0 is hypergeometric and has zero-momentum gain 𝒢(κ) = √2 cosh(π√(4κ−1)/4); since
Σκ_A = 8/(1−ξ) some mode has 𝒢 ≥ 4.500465 (free fields). With ξ = 0.2999 the aspect stiffness is κ_E = 19.324
(𝒢 = 674); the next-order zero-mode terms (holonomy matching energy and Higgs zero-mode mass, whose Laplacian is −4/3
times their value) shift it to κ_E = 18.04, κ_T2 = −8.34 at ξ = 0.2826 (𝒢 = 529).

**Method.** `c73_common.py` (mpmath Ewald / Chowla–Selberg, geodesics, Richardson), `c73_symbolic.py` (SymPy S1–S7),
`c73_gain.py` (closed form, Γ, ₂F₁, ODE), `c73_thermo.py`, `c73_zeta.py` (s = 1 sums; s = 2 cross-check imports
`../01_sm_casimir_weight/C57`), `c73_abelian_zeropoint.py`, `c73_codex_toy_repro.py` (independent recomputation of the
Codex four-level toy, metric typed as data), `C83_next_order_shape_hessian.py` (reads fu/C78, fu/C81 of step 02).
Derivations D14, D24, D55 (C73), D59 §1 (C83). Status: identities proved (symbolic + 40–60-digit numerics); next order
first-order estimate.

**Run.** `--quick` (17 s): symbolic, gain, C83. `--full` (122 s, 14–15 processes): plus thermo, zeta, abelian zero point,
toy.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:48:36Z | 11/11 | 0 | C73 symbolic S1-S7 5; C73 gain and guaranteed lower bound 2; C83 next-order shape Hessian 1 |
| full | 2026-10-15T20:04:55Z | 16/16 | 0 | C73 symbolic S1-S7 6; C73 gain and guaranteed lower bound 3; C83 next-order shape Hessian 1; C73 thermo 29; C73 zeta 94; C73 abelian_zeropoint 58; C73 codex_toy_repro 4 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number | output : key | archived source |
|---|---|---|---|
| PRD Sec. VII A, Eq. `eq:s-Hesstrace`; App. E 1 | 8/3 (eigenvalue at p = −2) | `fu/C73/C73_symbolic.json` : `S2/values/-2` | `04_calculations/results/20261011_claude_followup/C73/C73_symbolic.json` |
| PRD Sec. VII A, Eq. `eq:s-nogo` | ΔU = (8/3)[U − Q − 3N/2] | same : `S5/*` ; `C73_thermo.json` | same |
| PRD Sec. VII B, Eq. `eq:s-gain` | guaranteed gain 4.500465 (NOGO-11); 𝒢(13.53) = 216.46 | `fu/C73/C73_gain.json` : `lower_bounds/0/G via ODE`, `rows/0/gain_closed` | `.../C73/C73_gain.json` |
| PRD Sec. VII C; App. B 2 | κ_E = 19.324, κ_T2 = −9.074 (ξ = 0.2999) (NOGO-05/06) | `fu/C83/C83_results.json` : `next_order/gauss/cases/LO/{kappa_E,kappa_T2}` | `.../C83/C83_results.json` |
| PRD Sec. IV B; App. B 2; PRD Sec. VII E | κ_E = 18.04, κ_T2 = −8.34 at ξ = 0.2826 (NOGO-07); Higgs part 0.14 of the shift | same : `next_order/gauss/cases/{both,both_without_Higgs}/kappa_E` | same |
| PRD App. E 1 | s = 1 Laplacian −4/3; abelian zero-point ratio −8.88 | `fu/C73/C73_zeta.json`, `C73_abelian_zeropoint.json` | `.../C73/` |
