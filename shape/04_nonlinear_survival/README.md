# 04_nonlinear_survival — survival thresholds, real states set in the past, plateau background, next order

**Physics.** Beyond the linear regime the shape runs away to a finite-time singularity after the turn. Scaling the
natural state at the turn by s: survival 1.000 / 0.973 / 0.824 / 0.367 / 0.063 / 0 for s = 1, 3, 5, 10, 20, 40
(N = 256); s₅₀ = 8.2, s₉₀ = 4.2, s₉₉ = 2.76 (log-linear) / 2.86 (logistic) at N = 2048 (T19 independent 2.95; paper
"≈ 2.8–2.95"); next order s₉₉ = 3.16. Real optimal-width states set at t_x = −0.5, −1/h or at the freezing onset
survive 0.623 / 0.184 / 0.133 (0.731 / 0.227 / 0.168 at next order). On the A_s-calibrated plateau history instead of
the leading cosh background none of 2560 paired trajectories changes fate.

**Method.** `../shapelib/C142_shape_nonlinear_threshold.py` (D114 runs and analysis), `C160_plateau.py` (D131 §2,
plateau background, paired with D114), `../shapelib/C153_run02.py` (lo_amp / nx_amp sets, s₉₉) and
`../shapelib/C153_nl.py` (next-order group). Model: full nonlinear volume + five shape moduli, SM twisted Ewald force,
zero-mode energy (ξ = 0.299897), DOP853 rtol 1e-11 / atol 1e-13, pinch at eig G < 1e-3. Status: numerical; the
two-arrow reading and the truncated-Wigner sampling are stated assumptions (Moyal correction: step 05).

**Run.** `--quick` (~2 min): archived trajectories unpacked; production analyses (C142, plateau, C153 run01 and run02)
re-run; 6 archived trajectories re-integrated bit for bit. `--full`: C142 run (6656) + smoke, plateau run (2560) +
smoke, C153 run02 / next group (resumed if step 02 --full ran in the same work tree).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:07:40Z | 13/13 | 0 | redraw C142 (2 draws) 3; redraw C160 plateau (2 draws) 8; redraw C153_run02 amplitude sets (2 draws) 3; C142 analysis 3; C160 plateau analysis 8; C153 run01 analysis (next-order amplitude threshold) 4; C153 run02 analysis (s99) 25 |
| full | 2026-10-16T00:01:36Z | 9/9 | 0 | C142 run (D114) 975; C142 smoke 14; C160 plateau run 326; C160 plateau smoke 17; C153 next-order group 1; C153 run02 1; C142 analysis 2; C160 plateau analysis 4; C153 run01 analysis (next-order amplitude threshold) 3; C153 run02 analysis (s99) 12 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VII F, Fig. 6; report Table 22 (Sec. 15) | survival vs s (NL-01) | `r2/shape_nonlinear_threshold/run01/C142_summary.json` : `sets/amp_s*/survival` | `04_calculations/results/20261013_claude_v12_r2/shape_nonlinear_threshold/run01/C142_summary.json` |
| PRD Sec. VII F | s₅₀, s₉₀ (N = 256) | same : `amp_threshold/*` | same |
| PRD Sec. VII F | s₉₉ = 2.756 (MC 95 % 2.66–2.90), logistic 2.863 (boot 2.74–2.96); next order 3.160 / 3.294 (NL-06, 07, 09) | `r3/shape_hardening/run02/C153_run02_summary.json` : `lo_amp/s99_loglinear/central`, `lo_amp/s99_logistic_in_ln_s/s99`, `nx_amp/*` | `.../20261014_claude_v12_r3/shape_hardening/run02/C153_run02_summary.json` (shipped in `../02/expected`) |
| PRD Sec. VII F | survival 0.9976 / 0.9834 / 0.9536 at s = 2.5 / 3.0 / 3.5 (N = 2048) (NL-03) | same : `lo_amp/rows[*]/p` | same |
| PRD Sec. VII D, F | past-set real states 0.623 / 0.184 / 0.133 (TX-27, 29) | `C142_summary.json` : `sets/r1_tx*/survival` | as above |
| PRD Sec. VII F | next order s₉₀ = 4.19, s₅₀ = 9.07 (NL-10) | `r3/shape_hardening/run01/C153_nl_summary.json` : `next_amp_threshold` | `.../shape_hardening/run01/` (shipped in `../02/expected`) |
| PRD Sec. VII F | plateau background: 0 / 2560 fate flips (NL-14) | `r3/shape_tier2/run01/plateau/C160_plateau_summary.json` : `sets/*/n_flips` | `.../shape_tier2/run01/plateau/C160_plateau_summary.json` |
