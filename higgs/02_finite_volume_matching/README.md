# 02_finite_volume_matching — consistent one-loop finite-volume matching (D110 / C138, round 2)

**Physics.** The Higgs zero mode |Φ₀| (Coulomb gauge, flat measure) is the slow coordinate; all non-zero modes are
integrated out at one loop in the finite box. The consistent scheme (model `MC`+CW) uses the Coulomb-gauge spectrum
(per massive vector: 2 transverse + 1 Coulomb mode mixed with the Goldstone) and MS-bar finite-volume sums
Σ[V_CW^∞ + D_θ] at μ = 1/L. The infinite-volume CW logarithm cancels against the finite-volume sum, the energy is
analytic in m² inside |m| < 2π/L (tachyonic branch included), and the RG is consistent iff
γ_φ^C = (3y_t² − 3/2 g₂² − 1/2 g_Y²)/16π² (+2.2×10⁻⁴ at μ_b).

**Method.** `../zmlib/C138_core.py` (matching potentials dV(model, r, λ, g₂, g_Y, y_t)), `../zmlib/R14_fv.py`
(Ewald/Bessel sums, T_B/2 = −0.1128925, c₂ = 0.0062594), `../zmlib/R14_res.py` (conditional-Gaussian profile and 1D
Siegert resonance solver). `C138_run.py` modes: `checks` (analytic checks), `profiles`, `static` (γ(λ) tables for
MU/ML/MU_CW/ML_CW/MC/MC1, v11 and SM couplings), `all` (adds the round-2 1D hazard with the R14 mapping and the
matching error items). `matching_potential.py` (added for R18, quick and full): size of the one-loop matching dV
relative to the gauge zero-point confinement in the D112 2D_ex potential of PRL Fig. 1 (local slope ratio at
r = 1–5; barrier above the resonance level with and without dV).

**Status.** Log cancellation, analyticity, Coulomb spectrum and RG consistency: proven (analytic + Wolfram +
numerics); "MC is the correct BO split": conditional. Independent check R15 (PASS with corrections).
The 1D hazard numbers of this step (MC+CW 172.349/172.768 GeV) are historical rows of the convergence table; the
μ-variation item (+0.014/−0.045 GeV at P₁ = 0.5) entered PRL Table 1 (round 3) and is superseded by the run05
matching-scale scan of `09_claims` (`table1_rebuild.json`) if the rebuilt table is adopted.

**Run.**
```
python run.py --quick                 # ~1.5 min: checks + profiles + static
python run.py --full --workers 4      # ~10 min: + C138_run.py all (original ~5 min, 3 workers)
```
Optional: `C138_wolfram.wl` (γ(λ = −0.010) = 9.87118×10⁻⁴ in Wolfram vs 9.871149×10⁻⁴ here).
Original: `04_calculations/code/20261013_claude_v12_r2/higgs_matching/`; results `…/results/20261013_claude_v12_r2/
higgs_matching/run01`; derivation `04_calculations/derivations/20261013_claude_v12_r2/higgs_matching/
D110_HIGGS_MATCHING_CONSISTENCY_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:49:56Z | 9/9 | 0 | C138_run.py checks 5; C138_run.py profiles 30; C138_run.py static 25; matching_potential.py 0 |
| full | 2026-10-15T15:14:01Z | 10/10 | 0 | total (checks, static, all; 4 workers) 578 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRD Sec. V A, App. C 1 | T_B/2 = −0.1128925, c₂ = 0.0062594 | `r2/higgs_matching/run01/C138_checks.json` : `R14_fv_internal/TB_half`, `c2P` | same archived file |
| PRD Sec. V A, App. C 1 | γ_φ^C(μ_b) = +2.2×10⁻⁴ | `C138_checks.json` : `RG_one_loop_consistency_lam-0.010/gamma_phi_Coulomb` | same |
| report Sec. 6 (γ table) | γ(λ = −0.010, v11) MC+CW 9.871×10⁻⁴; MU 2.051×10⁻³ | `C138_static.json` : `table/v11/MC/-0.01/gamma`, `table/v11/MU/-0.01/gamma` | `r2/higgs_matching/run01/C138_static.json` |
| report Sec. 11.1 (Table 14) | MC+CW (D110) 172.349/172.768 (R14 mapping) | `C138_summary.json` : `["central_MC+CW_mu=1/L"]` (full) | `run01/C138_summary.json` |
| PRL Table I (round 3) 'one-loop matching' | μ ∈ [½, 2π]/L items | `C138_summary.json` : `error_budget_GeV` (full) | same |
| PRD Sec. V (effective slope); report Secs. 6–7 | one-loop matching lowers the local slope of the zero-point confinement by 10–33 % at r = 1–5 (λ = −0.010: 0.110–0.331; −0.013: 0.105–0.310) and roughly halves the barrier above the resonance level: radial potential U_ad 1.57 → 0.72 (λ = −0.010), 1.21 → 0.53 (−0.013); field-space potential U_g 1.55 → 0.70, 1.19 → 0.51 (units 1/L) | `w/matching_potential/matching_potential.json` : `headline`, `lambda/<λ>/slope`, `lambda/<λ>/barrier_above_Q` (`matching_potential.py`, R18; reads `figures/data/higgs/generated/potential_2Dex_lam010_013.csv` and the T14a resonance level) | `expected/w/matching_potential/matching_potential.json` |
