# 03_joint_state — joint inflaton–shape state and the size of the torus (C147/D119)

**Physics.** The turning-point state also fixes the inflaton zero mode. Time-symmetric, constraint-projected real
Gaussian (Wigner) inflaton states propagated through the classical nonlinear history give: the mean φ_b is not selected
(entering the typical fossil window needs Δφ_b ≈ 6.2e-4 M_P, i.e. N_post ∈ [68.480, 68.516]); the B3 width
σ_φ = 0.009737 M_P; Θ-invariant states obey σ_N ≥ 0.935 e-folds (≈ 0.0137 N_post); the induced size distribution is
p(N) ∝ 1/N, and the probability that the fossil is readable stays ≲ 1.5 % for the optimally placed time-symmetric
Gaussian (flat dφ measure 1.39e-3 for N_max = 100). The Casimir forbidden region has WKB exponent S_E = 11964.

**Method.** `joint_state.py` (working copy of the snapshot's `inflaton_packet.py` functions plus own analysis; inputs:
four small files of the frozen Codex snapshot in `inputs/snapshot/`, and the archived C79 table staged from
`../02_ensemble/expected/`), `b3_slope_symbolic.py` (SymPy; stdout compared line by line). Derivation D119 + R15
correction addendum + T12 W11. Status: conditional (Robertson + Θ + linearisation + uncorrelated sectors; not the joint
WDW wave function).

**Run.** `python run.py --quick` (= `--full`, 45 s).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:21:59Z | 6/6 | 0 | C147 joint state 25; B3 slope symbolic 1 |
| full | 2026-10-15T23:20:07Z | 6/6 | 0 | C147 joint state 21; B3 slope symbolic 1 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VIII B (joint state) | σ_φ = 0.00973692 (JOINT-01) | `ai/joint_state/run01/joint_state_results.json` : `T3/B3/ref/sigma_B3` | `04_calculations/results/20261013_argon_codex_intake/joint_state/run01/joint_state_results.json` |
| PRD Sec. VIII B | σ_N ≥ 0.935075 (ref) / 0.938387 (window) (JOINT-02) | same : `T3/response/{ref,window}/floor_sigma_N` | same |
| PRD Sec. VIII B | Δφ_b = 6.1825e-4; N window [68.480, 68.516], width 0.0359 (JOINT-04, 05) | same : `T4/typical_fossil_window/dphi_window`, `T2/N_window_typical`, `T2/window_width_efold` | same |
| PRD Sec. VIII B | flat-dφ readability 1.389e-3 (N_max = 100) → 5.35e-5 (1e6) (JOINT-06) | same : `T4/shape_marginalised/A_fixed_typical/flat_dphi_Nmax=*` | same |
| PRD Sec. VIII B | S_E = 11964 (JOINT-09); φ_b(ref) = 5.57936 (BKG-03) | same : `T2/S_E_ref_closed`, `T2/phib_ref(N=68.25)` | same |
| report Sec. 21 | interface to C145 | `ai/joint_state/run01/C145_interface.json` | `.../joint_state/run01/C145_interface.json` |
| PRD Sec. VIII B (B3 slope) | symbolic slope identities | stdout of `b3_slope_symbolic.py` | `.../joint_state/run01/b3_slope_symbolic.txt` |
