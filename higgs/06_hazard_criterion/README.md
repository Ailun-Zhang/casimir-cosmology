# 06_hazard_criterion — hazard along the turn, P₁(m_t, α_s), critical masses, λ_c, convergence table

**Physics.** Along the turn the SM couplings are taken at μ(u) = μ_b/√cosh 2u; with ξ_H = 1/6 the a″/a curvature term
cancels and the single-branch survival of the time-symmetric zero mode is
I(m_t, α_s) = ∫₀⁵ γ(m_t, α_s; u) du / (hL_b √cosh 2u), P₁ = e^{−I}  (hL_b = 0.00437243, η_∞ = 299.84).
For a real time-symmetric initial state the survival is exactly w₀ × hazard (deviation ≤ 0.3%, D111), so P₁ is the
exact survival conditioned on the metastable sector. The critical masses solve ln I(m_t) = ln(−ln P₁); λ_c is
λ_SM(μ_b) at those masses ("hazard-equivalent threshold").

**Method.** Production routines only: `../04_cross_terms/C149_direct5.py post|final` (→ `../zmlib/C149_direct.post`,
`C150_twod.integ` = cubic spline of ln γ in u + Simpson on 20 001 points, `C150_twod.crit` = cubic spline of ln I in
m_t + Brent root; λ_c via the NNLO mapping at u = 0), `../01_sm_mapping/R3_mbar.py` (m̄_t). `hazard_table.py` (new)
re-integrates the node widths with an independent adaptive quadrature and alternative interpolants and assembles the
convergence table (D122 §0.3 + D121 run02–run05).

**Status.** Hazard integral exact in the adiabatic limit; verified numerically (resonant projection 1.3×10⁻⁴,
real-time 0.22%); P₁ is a conditional result (Gamow/B3
preparation, single branch, ξ_H = 1/6, w₀ ≈ 1). Numerics: the adaptive quadrature agrees with the production
integration to < 10⁻¹² GeV; monotone PCHIP interpolants (less accurate than the production cubic splines) move m_c by
≤ 0.0002 GeV (in u) and ≤ 0.0025 GeV (in m_t, 0.2 GeV grid) — an upper bound on interpolation sensitivity, compatible
with the ±0.002 GeV 'numerics' row of PRL Table 1.

**Run.**
```
python run.py --quick     # ~15 s on the run05 node widths in the work tree (recomputed by 04 --full, else archived)
```
`--full` is identical (the expensive node widths belong to `04_cross_terms --full`; `higgs/run_higgs.py --full` runs the
chain in order).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:54:59Z | 17/17 | 0 | C149 run05 post (hazard, crit, P1, lam_c) 5; C149 run05 final (summary) 5; mbar(mbar) of the critical masses 4; hazard_table.py 4 |
| full | 2026-10-16T00:00:11Z | 17/17 | 0 | C149 run05 post (hazard, crit, P1, lam_c) 3; C149 run05 final (summary) 3; mbar(mbar) of the critical masses 3; hazard_table.py 3 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL Table I; core sentence | m_c^pole 172.5024/172.9595/173.4184 | `r3/holonomy_cross/run05/C149_run05_summary.json` : `crit/0.1180` | `04_cross_terms/expected/r3/holonomy_cross/run05/C149_run05_summary.json` |
| PRL Fig. 2(a) | m_c at α_s 0.1171/0.1189; slope dm_c/dα_s ≈ 387 GeV | same : `crit` | same |
| PRL headline | λ_c −0.0091/−0.0121/−0.0152 | same : `lam_c` | same |
| PRL text; PRD Sec. V E | P₁(172.60) 0.8426; P₁(172.52) 0.8909; P₁(172.2) 0.9850; P₁(173.0) 0.4549 | same : `P1` | same |
| PRL Fig. 2(a) | P₁(m_t) grid and I(m_t) | `r3/hazard_criterion/hazard_table.json` : `P1/0.1180`, `I/0.1180` | run05 `P1_grid` |
| PRL Table I (m̄ form) | m̄_t 162.7281/163.1637/163.6012 | `r3/sm_mapping/run02/R3_mbar_run05.json` : `new_central` | `01_sm_mapping/expected/…/R3_mbar_run05.json` |
| report Sec. 11.1 (Table 14) | MU → … → 2D direct (R14) → +NNLO → +cross (run02) → run04 → run05 | `r3/hazard_criterion/hazard_table.json` : `convergence_table_0p1180` | C150 run01/run02, C149 run02/run04/run05 summaries |
| numerics | quadrature / interpolation sensitivity | `hazard_table.json` : `max_abs_diff_*` | — (new check) |
