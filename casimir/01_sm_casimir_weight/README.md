# 01_sm_casimir_weight — signed Casimir weight of the Standard Model, spin structures and holonomies

**Physics.** For free massless fields on a flat torus with twists θ, E_C = −(1/π²L) Σ_{n≠0} w_n (nᵀGn)⁻², w_n =
Σ_f d_f s_f cos(2π θ_f·n). With the SM content (12 vectors, 4 real Higgs scalars, static graviton, 45 Weyl fermions)
the signed weight N_C = Z_eff/Z_2(cube) is +25.515 for all-antiperiodic fermions (AAA) and −30 for PPP; flat SU(3)-centre
or hypercharge holonomies give +11.2 (PPP centre) and up to +17.8 (PPP hypercharge); for AAA the centre holonomies lower
N_C to 22.74 / 9.86 and the trivial holonomy is the maximum. The cube Hessian of Z_eff on the five shape modes has
eigenvalues +1902.24 (E, aspect) and −893.20 (T₂, angle); κ_A = 3λ_A/Z_eff = 13.5286 / −6.3524 with Σκ_A = 8 and
Tr Hess Z_eff = (8/3) Z_eff to 1e-8 for all eight spin structures.

**Method.** `C57_spin_structure_Casimir.py` (own Ewald split at s = 2, direct-sum checks, five-point finite-difference
Hessians on geodesics G = exp(2Σβ_A X_A)); `C68_holonomy_and_preparation.py` (holonomy sectors with C57's Z₂; part (b)
the preparation-time sensitivity of the free-field fossil). Derivation D50 §1–3 (C57); C68 answers R09/R10 review items
(no D file). Status: numerical evaluation of exact image sums (proved formulas).

**Run.** `python run.py --quick` (= `--full`, ~6 s).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:48:11Z | 9/9 | 0 | C57 spin structures, N_C, cube Hessians 1; C68 holonomy sectors and preparation time 1 |
| full | 2026-10-15T19:22:37Z | 9/9 | 0 | C57 spin structures, N_C, cube Hessians 3; C68 holonomy sectors and preparation time 1 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number | output (work tree) : key | archived source |
|---|---|---|---|
| PRD Sec. IV A–B, Eq. `eq:pa-NC`; Sec. I | N_C(AAA) = +25.515; N_C(PPP) = −30 | `r5/C57_spin_structure/C57_spin_structures.json` : `SM_spin_structures/AAA/N_C_signed`, `.../PPP/N_C_signed` | `04_calculations/results/20261009_claude_R5/C57_spin_structure/C57_spin_structures.json` |
| PRD App. B 1, Table IV | eigenvalues +1902.24 / −893.20; κ_E = 13.5286, κ_T2 = −6.3524; trace identity 1e-8 | same : `SM_spin_structures/<s>/{Hess_eigs,kappa,trace_over_8_3_Z}` | same |
| PRD Fig. 1(a), (c); report Fig. 3 | N_C per spin structure; gains | same : `SM_spin_structures/<s>/{N_C_signed,gains_zero_momentum}` | same |
| PRD Sec. IV B; App. B 1 | PPP centre 11.19; PPP hypercharge max 17.79; AAA centre 22.74 / 9.86 | `r5/C68_holonomy_preparation/C68_results.json` : `PPP_centre_111`, `PPP_Y_iso_scan_max/NC`, `AAA_centre_100`, `AAA_centre_111` | `.../20261009_claude_R5/C68_holonomy_preparation/C68_results.json` |
