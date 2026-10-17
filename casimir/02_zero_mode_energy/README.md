# 02_zero_mode_energy — energy of the gauge and Higgs zero modes (ξ = 0.2999) and its next order

**Physics.** The constant modes of the SM gauge fields and of the Higgs doublet on T³ have a positive ground energy Q/L
(van Baal / Lüscher small-volume expansion). In units of the free Casimir coefficient C_free = 42.74 the zero modes use
ξ = Q/C_free = 0.2993 / 0.2999 / 0.3043 (λ = 0, 0.01, 0.1; exact local ground energies by diffusion Monte Carlo); the
pure-gauge operator bound gives ξ ≥ 0.2580 and the Gaussian variational bound ξ ≤ 0.3083 (λ = 0.01). The first
next-order terms (holonomy curvature of the non-zero modes, C78; induced Higgs zero-mode mass, C81) lower ξ to 0.2826
(≈ 0.28). DMC validation: quartic oscillator 2.1402(5) (certified 2.1407), ε₂ = 4.1171(14) (4.116720), ε₃ = 12.5869(22)
(12.5868).

**Method.** `C72_common.py` (conventions, Hamiltonians, Gaussian energies, one-loop couplings at μ = 1/L_b),
`C72_dmc.py` (importance-sampled DMC, seeded per case), `C72_analysis.py` (D54 parts A–E: Legendre transform,
Wick moments, isospectrality, bounds, fossil vs ξ), `C72_summary.py`, `C78_next_order_zero_mode.py`,
`C81_higgs_zero_mode_mass.py` (input `inputs/rge_couplings.json` = literature work file L25). Derivations D54 §1–8,
D58 §4, D59 §2. Status: numerical (local Hamiltonian exact up to DMC errors; matching at one loop; next order first-order
estimate).

**Run.** `--quick` (22 s): DMC timing test (two seeded short runs, bit-level), analysis/summary on the archived
production DMC energies, C78, C81. `--full`: production DMC (all cases; 435–459 s on 15 workers here, ~10 min on 8
originally), then the same chain.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:48:28Z | 9/9 | 0 | C72 DMC timing test 1; C72 analysis (D54 parts A-E) 15; C72 summary 0; C78 next-order holonomy curvature 1; C81 Higgs zero-mode mass 0 |
| full | 2026-10-15T19:43:55Z | 9/9 | 0 | C72 DMC production (all cases) 459; C72 analysis (D54 parts A-E) 16; C72 summary 0; C78 next-order holonomy curvature 1; C81 Higgs zero-mode mass 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number | output : key | archived source |
|---|---|---|---|
| PRD Sec. IV B (`sec:pa-zeromodes`), Table I; App. B 2 | ξ = 0.2993 / 0.2999 / 0.3043 | `fu/C72/C72_summary.json` : `r_band/one-loop mu=1/L_b/exact_local_dmc_r/lam{0,0.01,0.1}` | `04_calculations/results/20261011_claude_followup/C72/C72_summary.json` |
| PRD App. B 2 (Eq. `eq:pa-gauss-bound`) | ξ ≥ 0.2580, ξ ≤ 0.3083 | same : `.../lower_operator_bound_r`, `.../upper_gauss_r/0.01` | same |
| PRD App. B 2 | DMC 2.1402(5), 4.1171(14), 12.5869(22) | same : `comparisons_ours_vs_codex/*` | same |
| PRD App. B 2 | LE_EW = 4.4478(13) / 4.4750(17); Higgs cost 1.78/L vs 0.46/L (×3.9) | `fu/C72/C72_dmc_results.json` : `extrapolated/ew_running_lam*`; `C72_summary.json` : `higgs_coupled_vs_decoupled` | `.../C72/C72_dmc_results.json` |
| PRD Sec. IV B (next order) | ξ → 0.2826 (≈ 0.28); dξ_hol = −0.0140, dξ_Higgs = −0.0033 | `fu/C81/C81_results.json` : `r_first_order/landau_3pol`; `fu/C78/C78_results.json` : `shifts/*` | `.../C81/`, `.../C78/` |
