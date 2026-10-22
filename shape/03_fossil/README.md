# 03_fossil — the quantum fossil: floor, natural state, σ_E, side difference and nonlinear ratio

**Physics.** The natural (time-symmetric, minimum-energy-variance) state at the turn sits 0.05 % above the floor; its
outgoing aspect width is σ_E = 0.104 at ξ = 0.30 (A_s-calibrated plateau; 0.0587 for free fields), σ_T2 = 0.0065, and
scales as σ_E ∝ (A_s r)^{1/8} C_eff^{−3/8} K(κ_E)^{1/2} with dln σ_E/dln κ_E = 1.366 (an r measurement fixes σ_E to
3.1 % / 1.6 % with LiteBIRD / CMB-S4). The E doublet is an isotropic Gaussian: the median ln(L_max/L_min) is
1.588 σ_E, i.e. a typical side difference of 18 % (16 % at next order, 10 % for free fields). The nonlinear map
through the turn changes the aspect variance by +4.3 % (1.043 / 1.045 for two 2048-draw sets).

**Method.** `C58_turning_point_states.py` (transfer, time symmetry, floor; D49 §1–4), `C62_SM_pivot_calibration.py`
and `C63_rolling_fossil.py` (A_s-calibrated plateau history; D49 §5–5b), `C77_prep_time_zero_mode.py` (D58 §3),
`C82_fossil_r_dependence.py` (σ_E vs ξ, median coefficient; D59 §3), `C143_observables.py` (scaling relations; D115),
`C60_run_main.py` + `C60_analyze.py` (free-field nonlinear ensembles; D49b) and `../shapelib/C76_nonlinear_zero_mode.py`
(ξ = 0.30 ensembles; D58 §2, D59 §6). The next-order floor (0.0934) and G(∞) are in step 02 (C153_linear), the
next-order κ_E in casimir/03 (C83). Status: floor proved; σ_E numerical on the stated background; NL ratio numerical.

**Run.** `--quick` (36 s): C58, C62, C63, C77, C82, C143, archived C60 (2592) and C76 (4224) trajectories re-analysed,
2 + 2 re-integrated bit for bit. `--full`: C60 (all tags incl. hi) and C76 (main, hi, ind) from scratch.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:06:41Z | 16/16 | 0 | C58 turning-point states 1; C62 pivot 2; C63 rolling fossil 1; C77 preparation time 1; C82 fossil vs xi 13; C143 observables 0; redraw C60 (2 draws) 2; redraw C76 (2 draws) 2; C60 analysis 2; C76 analysis 2 |
| full | 2026-10-16T00:01:36Z | 12/12 | 0 | C58 turning-point states 1; C62 pivot 1; C63 rolling fossil 1; C77 preparation time 0; C82 fossil vs xi 6; C143 observables 0; C60 free-field ensembles 203; C76 xi=0.30 ensembles 317; C60 analysis 1; C76 analysis 1 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VII C, Eqs. `eq:s-Tsym`, `eq:s-family` | floor, natural/floor, gain 216.46 (free) | `r5/C58_turning_point_states/C58_results.json` : `SM_AAA/aspect/*` | `04_calculations/results/20261009_claude_R5/C58_turning_point_states/C58_results.json` |
| PRD Sec. VII E, Table III | σ_E: 0.0589 (free), 0.1041 (ξ = 0.2999, C60/C76 convention), vs ξ (FOS-01, 02; FLOOR-06) | `fu/C82/C82_results.json` : `sigma_E_leading/*` | `.../20261011_claude_followup/C82/C82_results.json` |
| PRD Table III | side difference 18 % / 10 %; median coefficient 1.588 (FOS-04, 05) | same : `typical_side_difference_plateau/*`, `median_coeff` | same |
| PRD Table III | plateau free σ_E 0.0587 | `r5/C63_rolling_fossil/C63_results.json` : `SM_AAA/modes/aspect/natural_rms` | `.../C63_rolling_fossil/` |
| PRD Sec. VII E, Eq. `eq:s-scaling` | K(19.32) = 24.5501; dlnσ/dlnκ = 1.366; LiteBIRD 3.1 %, CMB-S4 1.6 %; σ ratio 0.6306 at r = 1e-4 (FLOOR-01, FOS-14..16) | `r2/observables/run01/C143_observables.json` : `F_kappaE`, `dlnsigma_dlnkappa`, `LiteBIRD_*`, `CMBS4_*`, `sigma_vs_r` | `.../20261013_claude_v12_r2/observables/run01/C143_observables.json` |
| PRD Sec. VII E (error budget) | NL/linear aspect variance 1.043 / 1.045 (FOS-07) | `fu/C76/C76_summary.json` : `sets/{hi,ind}/*` ; also `../04` C142 `regression/C76_reference` | `.../C76/C76_summary.json` |
| report Fig. 20 | fossil distributions (free 2048, ξ = 0.30 2048) | `r5/nonlinear_SM/C60_summary.json`, `fu/C76/C76_summary.json` + trajectory archives | `.../nonlinear_SM/`, `.../C76/` |
| PRD Sec. IV C (`sec:turning`) | φ_b, N_post, hL_b of the SM pivot | `r5/C62_SM_pivot/C62_results.json` : `SM_AAA/*` | `.../C62_SM_pivot/` |
