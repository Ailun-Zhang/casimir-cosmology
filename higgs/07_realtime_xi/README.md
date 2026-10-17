# 07_realtime_xi — real-time evolution through the turn and the condition on ξ_H (D129 / C157; C161 run05 version)

**Physics.** The fate of the zero mode at the end of the turn (u ≳ 4.4) is decided by real-time evolution: the
conformal-frame TDSE is mapped exactly (lens transform) to the physical frame, where the zero mode has mass
M = (hL)³ and classicalises; at u_sw = 6 it is handed over (Wigner sampling) to cosmic-time classical evolution.
With ξ_H ≥ ξ_c the survival differs from ξ_H = 1/6 by < 1% (definition P_surv(ξ_c) = 0.99 P_surv(1/6)); ξ_H = 0 runs
away. ξ_c is essentially a universal function of P₁, so SM and α_s errors cancel at first order when the condition
is stated at fixed P₁ or λ(μ_b).

**Method (run05 = paper centre; C161 step 4).** `nm_model.py` (node evaluation of the run05 model: U_BO = U1 + eps0
of 2D_ex + NNLO + full one-loop cross terms via `../04_cross_terms/C149_direct5.py`, calibration to the run05 2D
widths, PotentialTable npz), `rt_quick.py` (55 real-time runs: m_t = 172.2/172.6/172.95/172.96/173.0 × 11 ξ_H;
post → ξ_c), `xi_compare.py` (run05 vs run02, EW image ξ_EW = 1/6 + (ξ(μ_b) − 1/6)/Ξ, universal ξ_c(P₁) residuals),
`xi_regress.py` (node regression: with `C161_CROSS=run02` the D129 run01 node is reproduced exactly); the package
copy of `xi_compare.py` also stores the universality residuals restricted to the interpolation range (R18). Shared:
`../zmlib/hr_{core,potential,part2,post}.py` (D111 real-time machinery).

**Status.** Numerically checked (method-internal: 2D/1D, Madelung/Wigner, TWA, grids); conditions: λ(1/L) frozen for
u > 7, single zero-mode truncation, post-horizon inhomogeneous spinodal not computed. The run02-based D129 extras
quoted with the condition — SM band (±0.013) and α_s band (±0.040) at m_t = 172.6, the 2D (r,|w|) real-time check
(P_surv ≤ 2×10⁻⁴, ξ_c ≤ 1.4×10⁻⁴), R15 items — are archived in `expected/r3/realtime_2d/` (C157_run02_summary.json,
C157_twod.json, C157_r15_items.json, C157_summary.json); their production scripts (realtime_2d `twod_rt.py`,
`run02_sm.py`, `r15_items.py`; ≈ 35 min, ≈ 250 MB of node/profile/table files) are NOT part of this package
(partial coverage, stated in VERIFY).

**Tested range of the universal ξ_c(P₁)** (`xi_run05_summary.json` → `universality.interpolation_range`, added for
R18). The universal curve is the PCHIP through the five run02 central points (m_t = 172.0–173.0 GeV), so it is tested
by interpolation only over its span **0.448 ≤ P₁ ≤ 0.9975**. Inside it lie 11 of the 16 run02 variants (P₁ =
0.506–0.993); they agree with the curve to |Δξ_c| ≤ 0.0019, and the five run05 points to ≤ 0.0002. The other five
variants lie outside the span, where the curve is extrapolated: λ-minus at 172.95/173.00 GeV (P₁ = 0.39/0.34;
residuals −0.0008/−0.0015), α_s = 0.1171 at 172.95/173.00 (P₁ = 0.17/0.13; −0.0074/−0.0094) and α_s = 0.1189 at
172.20 (P₁ = 0.9996; −0.0168). The older field `max_abs_variant_residual` (0.0168) includes these extrapolations;
`max_abs_variant_residual_inside` (0.0019) does not. The central curve and every quoted number are unchanged.

**Run.**
```
python run.py --quick                 # ~10 s: node regression, xi_c from the archived run05 runs, comparison
python run.py --full --workers 15     # nodes (145) + calib + tables + 55 real-time runs + post (C161: ~8 min)
```
Original code: `04_calculations/code/20261015_claude_v12_writing/claims_run05/{nm_model,rt_quick,xi_compare,
xi_regress}.py` (copies of `…/20261014_claude_v12_r3/realtime_2d/` edited by C161 `make_xi_copies.py`); results
`…/results/20261015_claude_v12_writing/claims_run05/run01/xi_run05`, `xi_run05_summary.json`; derivations
`…/derivations/20261014_claude_v12_r3/realtime_2d/D129_REALTIME_2D_CN.md`, `…/20261015_claude_v12_writing/
claims_run05/D132_claims_run05_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:55:12Z | 3/3 | 0 | xi_regress.py 11; rt_quick.py post 1; xi_compare.py 1 |
| full | 2026-10-15T19:45:24Z | 4/4 | 0 | xi_regress.py 10; nm_model.py nodes 15 102; nm_model.py calib 12; nm_model.py tables 4; rt_quick.py run 15 main 182; rt_quick.py post 0; xi_compare.py 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL ξ condition; PRD Sec. V H | ξ_c(172.6) at run05 = 0.0738 ± 0.0025 (run02: 0.0741); P₁ = 0.5 point 0.1205 | `w/claims_run05/run01/xi_run05_summary.json` | same archived file (C161) |
| PRD Sec. V H | ξ_c(m_t) table, EW image | `w/claims_run05/run01/xi_run05/C157_xi_quick.json` | same archived file |
| PRD Sec. V H | universal ξ_c(P₁): residuals of 16 run02 variants and run05 points | `xi_run05_summary.json`; `xi_run05_compare.csv` | same |
| PRD Sec. V H (bands, run02 centre) | SM band 0.0750 ± 0.0130, α_s band 0.0795 ± 0.0397 (m_t = 172.6) | archive only | `r3/realtime_2d/run02/C157_run02_summary.json` : `bands` |
| PRD App. D 2 (2D real time) | 2D vs 1D: ΔP_surv ≤ 2×10⁻⁴ | archive only | `r3/realtime_2d/run01/C157_twod.json` |
