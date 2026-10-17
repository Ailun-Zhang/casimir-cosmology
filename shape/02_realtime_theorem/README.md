# 02_realtime_theorem — real-moment theorem, G(t_x), closed-form floor and the conditional theorem t_*

**Physics.** The shape modes obey β̈ + 3Hβ̇ − κh² sech²(2ht) β = 0 with [β, Π] = 2iδ. Time symmetry gives the full
transfer T = M R M⁻¹ R, T₁₂ = 2m₁₁m₁₂, and the floor Var β₊ ≥ |T₁₂|δ, in closed form σ_floor² = K(κ_E)/X,
K(κ) = Γ(7/4)Γ(3/4)/[2D|Γ(5/4+iμ)|²] (K(19.32) = 24.550, σ_floor = 0.1037 at X = 2282.49; 0.0934 at next order).
The correlation C(t) = ½⟨{δβ, δΠ}⟩ is strictly monotone, so any state is real at most once (at t_x); a state real at t_x
has max± Var β± ≥ G(|t_x|) δ|T₁₂| with G = 2.395, 8.263, 45.10, 313.8, 961.7 at |t_x| = 0.1, 0.25, 0.5, 1, 2/h and
G(∞) = 2𝒢 = 1348.5. Nonlinear two-branch survival S₂ of real optimal-width states then bounds |t_x|: 99 % survival
requires |t_x| ≤ t_* = 0.237/h (95 % 0.225–0.252; N = 2048), 90 % requires ≈ 0.35/h (logistic 0.352; 0.355 from the
run02/run01 node interpolation of D125 Table A3), next order 0.265/h. Never-real states survive in both branches only at
4.3 % / 2.7 % (1.2× / 2× the threshold width).

**Method.** `C137_shape_gate.py`, `C137b_best_case.py`, `C137c_uncorrelated_optimum.py` (D109; R14-verified),
`C153_linear.py` (D125 linear layer: closed forms, G tables, monotone lemma, families, floor conventions, shear
criterion), nonlinear production in `../shapelib/C153_nl.py` (run01) and `../shapelib/C153_run02.py` (run02),
`C153_summary.py` (collects linear, nonlinear, ensemble (cmb/02) and emptiness (shape/01) layers). Optional Wolfram
checks `C137_verify.wl`, `C153_verify.wl` (archived receipts in expected/). Derivations D109, D113, D125 (+ run02
appendix). Status: theorem proved (any state); survival levels numerical (two-arrow reading, leading cosh background,
C142 model).

**Run.** `--quick` (~1–2 min): linear layer; archived run01/run02 trajectories unpacked and re-analysed with the
production analyses; 10 archived trajectories re-integrated bit for bit; summary. `--full`: 46 592 trajectories from
scratch (original 3438 + 221 s run01 and ~2 h run02 on 4 processes; ~45 min on 15 workers), smoke/mirror check, then
the analyses (`--fresh` discards a previous full run, otherwise it is resumed).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:06:12Z | 19/19 | 0 | C137 shape gate 2; C137b best case 1; C137c G(t_x) 1; C153 linear layer 8; redraw C153_nl (6 draws) 4; redraw C153_run02 (4 draws) 3; C153 run01 analysis 5; C153 run02 analysis 23; C153 summary 0 |
| full | 2026-10-16T00:01:36Z | 16/16 | 0 | C137 shape gate 2; C137b best case 1; C137c G(t_x) 1; C153 linear layer 7; C153 run01 main groups 2349; C153 run01 width 144; C153 smoke and mirror check 596; C153 run02 (N = 2048) 2309; C153 run01 analysis 3; C153 run02 analysis 12; C153 summary 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (numbers_shape id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VII C, Eqs. `eq:s-T12`, `eq:s-floor`; App. E 2 | m₁₁ = 10.554, m₁₂ = 2.4685, m₂₁ = 136.37, m₂₂ = 31.99; K(19.3237) = 24.5630; K(18.0) = 20.3204 (FLOOR-01..03, 11) | `r3/shape_hardening/run01/linear/C153_linear.json` : `floor_closed_form/{19.3237,18.0}/*` | `04_calculations/results/20261014_claude_v12_r3/shape_hardening/run01/linear/C153_linear.json` |
| PRD Sec. VII C (footnote on conventions) | σ_floor 0.10374 (H_eff), 0.10403 (C60/C76), 0.1031 (v11 δ); next order 0.09337 (FLOOR-04..08) | same : `floor_conventions/*`, `next_order_floor/*` | same |
| PRD Sec. VII D, Thm. `thm:s-real`, Eq. `eq:s-Gtx`; App. E 2; report Table 19 | G = 2.3952, 8.2632, 45.097, 313.79, 961.71; natural/floor 1.000485 (TX-01..06, FLOOR-10) | `r1/state_selection/run03/C137c_results.json` : `rows[k]/opt_over_floor`, `rows/0/adiabatic_over_floor` | `.../20261013_claude_v12/state_selection/run03/C137c_results.json` |
| PRD Sec. VII D | G(∞) = 2𝒢 = 1348.55 (κ = 19.3237), 1058.34 (18.0) (TX-07, 08) | `C153_linear.json` : `floor_closed_form/*/G_inf_closed_2calG` | same as above |
| PRD Sec. VII D, report Fig. 16; report Table 20 (Sec. 15) | S₂(τ) grid; t_*(99 %) = 0.23696 (MC 2.5–97.5 %: 0.2245–0.2518); t_*(90 %) = 0.3547 by interpolation between the N = 2048 node at 0.35 and the N = 512 node at 0.40 (68 % half-width 0.0033 from 2×10⁴ draws of the Jeffreys posteriors of the node fractions; the 0.30 node enters in 10 % of the draws; `numbers/derived_numbers.json` → `tstar90`, with the same method reproducing the archived 99 % interval 0.225–0.252), logistic 0.352; next order 0.26514 (TX-19..24) | `r3/shape_hardening/run02/C153_run02_summary.json` : `lo_t/t_star_linear_interp/0.99/*`, `lo_t/t_star_logistic/t_star/0.9`, `nx_t/...`; `r3/shape_hardening/run01/C153_nl_summary.json` : `grid[]` | `.../shape_hardening/run02/`, `run01/` |
| PRD Sec. VII D | never-real two-branch survival 0.043 / 0.027; incoming single-branch 0.887 / 0.723 (TX-13..15) | `C153_nl_summary.json` : `never[*]` | same |
| PRD Sec. VII D | adiabatic states: survival 0.992 / 0.609 / 0.129 (TX-18) | `C153_nl_summary.json` : `adiab[*]/future/p` | same |
| number keys of the papers (`C153_summary.json`) | all of the above in one file | `r3/shape_hardening/run01/C153_summary.json` | `.../shape_hardening/run01/C153_summary.json` |
