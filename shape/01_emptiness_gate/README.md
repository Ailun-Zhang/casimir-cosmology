# 01_emptiness_gate — quanta budget, reheating coupling, latest preparation, photon-pair branch

**Physics.** A turning point needs the Casimir energy to dominate; any excitation costs energy, so the state at the turn
must be nearly empty: Σ_quanta |n − ϑ| < N_C^eff γ/2π = 0.26660 (1−ξ) N_C, i.e. Σ|n| < 4.762 lowest bosonic quanta at
ξ = 0.2999 ("about five"; 4.70–5.05 over ξ ∈ [0.258, 0.308], 4.90 at ξ = 0.28, 6.80 for free fields; the bound equals
5 at ξ = 0.2649). A reheating coupling active in the time-symmetric past would overfill this budget: the remote
contracting passage of a gauge-kinetic window produces radiation R = ρ_γ/|ρ_C| = 10^93.9 (free Casimir) → 10^94.0
(j = 1) … 10^96.5 (j = 85) at ξ = 0.30. Non-conformal light fields prepared too early are excited by the contraction:
the latest natural preparation is T_max = ln(2π/hL_b) = 7.26 e-folds before the turn, and three light degrees of
freedom use up the budget at T ≈ 6.9. A single photon pair at the turn gives the deterministic branch with side ratio
4.96 (ξ = 0.30; 1.74 for free fields).

**Method.** `C55_contracting_window_gate.py` (own background + mode integrator on the contracting branch; readout after
the window closes), `C55b_time_reversal_check.py` (cross-check against the round-1 C49 pilot,
`inputs/C49_pilot_results.json`), `C56_preparation_time_gate.py` (occupations of massless minimally coupled modes),
`C153_emptiness.py` (v12 inputs; de Sitter scaling of the stored C56 occupations, `--check` = direct C56 test),
`C80_photon_branch_r.py` (minisuperspace photon-pair branch with the shapelib Casimir). Derivations D48 §1–3 (C55, C56),
D125 §7 (C153_emptiness), D56 / D59 §4 (C80). Status: budget proved; R, T_max numerical on the prescribed background.

**Run.** `--quick` (60 s): C55 `--quick`, C55b, C153_emptiness `--check` on the archived C55/C56 production files, C80.
`--full`: C55 production (97 q) and refined (401 q), C56 NC100 and SM_AAA (1235 s and 1510 s on 11 processes in the
original run), then the same chain.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:05:11Z | 11/11 | 0 | C55 quick 12; C55b time-reversal check 6; C153 emptiness (v12 inputs) + scaling check 16; C80 photon-pair branch 2 |
| full | 2026-10-16T00:01:36Z | 15/15 | 0 | C55 production (97 q) 63; C55 refined (401 q) 61; C56 NC100 1016; C56 SM_AAA 608; C55b time-reversal check 3; C153 emptiness (v12 inputs) + scaling check 8; C80 photon-pair branch 1 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (numbers_shape id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VI A, Eq. `eq:s-emptiness`; report Table 18 (Sec. 13) | Σ\|n\| < 4.762 (ξ = 0.2999), 5.047 (0.258), 4.898 (0.28), 6.80 (0); ξ(5) = 0.2649 (GATE-02..06) | `r3/shape_hardening/run01/emptiness/C153_emptiness.json` : `quanta_budget/*/budget_sum_n`, `xi_where_budget_equals/5` | `04_calculations/results/20261014_claude_v12_r3/shape_hardening/run01/emptiness/C153_emptiness.json` |
| PRD Sec. VI A | AAA fermion budget 5.50 (GATE-07) | same : `quanta_budget/*/fermion_budget` | same |
| PRD Sec. VI B, Eq. `eq:s-windowR`, Fig. 5(a) | log₁₀R = 93.86 (free, j = 1); 94.04 (ξ = 0.30, j = 1); 96.48 (j = 85) (GATE-08..10) | `r5/C55_window_gate/C55_window_gate.json` : `slices/0/log10_R`; `C153_emptiness.json` : `reheating_excess/xi0.30/log10R/{1,85}` | `.../20261009_claude_R5/C55_window_gate/` |
| report Sec. 13 | prefactor 10^97.79 (GATE-11) | `C55_window_gate.json` : `meta/log10_prefactor` | same |
| PRD Sec. VI C, Fig. 5(b) | T_max = 7.264 / 7.177; 3-dof crossing 6.909 / 6.822 (GATE-12, 13) | `C153_emptiness.json` : `preparation_window/hLb0.0044/*`, `hLb0.0048/*` | same |
| PRD Sec. VI C; report Sec. 13 | side ratio 4.96 (ξ = 0.30), 1.74 (free), 3.56–5.43 over the ξ band (GATE-14, 16) | `fu/C80/C80_results.json` : `runs/*` | `.../20261011_claude_followup/C80/C80_results.json` |
