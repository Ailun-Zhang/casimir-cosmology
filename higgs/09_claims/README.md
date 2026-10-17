# 09_claims — claims layer at the run05 centre (D126 v2 + correction addendum; C161 / D132)

**Physics.** The measurable statement is the finite-volume threshold λ_c (model errors only); everything that depends
on the top mass is reported as a marginal: P̄₁ = ∫ P₁(m_t, α_s, M_h; δ_SM, δ_model) dP over the measured m_t (several
mass definitions), α_s ± 0.0009, M_h ± 0.11, the SM-mapping error (±0.102 GeV pole / ±0.070 m̄) and the model error
of Table 1 (asymmetric errors split with equal mass). Bayes factors B_ex = P̄₁ and B_sel = P̄₁/⟨P₁⟩_prior stay within
|ln B| ≤ 0.75: the data neither support nor disfavour the scenario. Exclusion: P₁ < 0.1 at the 95% level for
m̂^pole ≳ 173.7 GeV with FCC-ee precision (separation of 172.6 vs 173.3 at 5.3σ); HL-LHC is α_s-limited (1.7–2.5σ).

**Method.** `C154_core_v2.py` (+ inputs `C154_inputs_v2_run05.json`; `*_t1new.json` = rebuilt Table 1 composite;
`*_regress_run02.json` = round-3 centre for regression), `C154_correction_v2c.py` (equal-mass split, prior-support
Bayes, corrected HL-LHC inputs), `pbar_direct_mc.py` (independent seeded Monte Carlo, no C154 code),
`crit_variants.py` (P₂ = P₁²/w₀ and B3-w₀ critical masses at run05), `table1_rebuild.py` (Table 1 without double
counting: run05 matching-scale scan, D112/T14a overlap removed, w₀ by D124 bins), `mu_scan.py` (matching scale
μ = c/L, c ∈ {½, 1, 2, 2π}, top threshold c_t ∈ {½, 2}; 66 node jobs in the run05 pipeline), `compare_claims.py`,
`fcc_exclusion.py` (added for R18: strength of the FCC-ee exclusion — at the threshold m̂ = 173.68 GeV the central
P₁ = 0.0230 and the exclusion factor B_ex = P̄₁ = 0.0332 of the same C154 machinery, with the FCC-ee α_s, SM and M_h
errors and the asymmetric model error; the one-Gaussian estimate with σ_tot = 0.131 GeV gives 0.0302; the
construction reproduces the archived `future.FCCee_base` values to machine precision).
The JSON inputs were copied with their ROOT-relative result paths rewritten to work-tree paths (`r3/…`, `w/…`).

**Which composite does the paper quote?** The text quotes P̄₁ with the rebuilt Table 1 composite (`t1new/`,
P̄₁ direct = 0.7300); the round-3 composite (`C154_correction_summary.json`) gives 0.7302, the same value after
rounding (0.73). Both are reproduced here (checks `t1new_marginal` and `marginal_run05`); `\PbarOne` in
`numbers/numbers_v12.json` points to the rebuilt composite (`t1new/`).

**Status.** Numerically checked (quadrature vs independent MC ≤ 5.5×10⁻⁴; run02 regression reproduces the round-3
P̄₁ = 0.7271 and all marginal values exactly). Conditions: α_s ≠ 0.1180 curves from a quadratic map checked against
direct α_s columns (|ΔP₁| ≤ 4.4×10⁻⁴); P₁ = 0.1 model items extrapolated in the round-3 table; FCC-ee δα_s = 0.0002.

**Run.**
```
python run.py --quick                 # ~2.5 min: core + correction (run05, t1new, run02 regression), MC, variants
python run.py --full --workers 15     # + the 66-job matching-scale scan (mu_scan run/post) before the rebuild
```
Original code: `04_calculations/code/20261015_claude_v12_writing/claims_run05/` (copies of `…/20261014_claude_v12_r3/
claim_positioning/` with run05 inputs); results `…/results/20261015_claude_v12_writing/claims_run05/run01`;
round-3 baseline `…/results/20261014_claude_v12_r3/claim_positioning/run02` (archived as `expected/r3/…`);
derivations `…/derivations/20261014_claude_v12_r3/claim_positioning/D126_CLAIM_POSITIONING_v2_CN.md` (+ correction
addendum), `…/derivations/20261015_claude_v12_writing/claims_run05/D132_claims_run05_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:59:59Z | 16/16 | 0 | C154 core run05 19; C154 correction run05 12; fcc_exclusion.py 1; pbar_direct_mc.py 116; crit_variants.py 1; table1_rebuild.py 1; C154 core t1new 22; C154 correction t1new 13; C154 core regress run02 22; C154 correction regress run02 12; compare_claims.py 1 |
| full | 2026-10-15T20:18:15Z | 14/14 | 0 | mu_scan.py run (66 node jobs, 15 workers, first full attempt of this session) 689; mu_scan.py run (resume, all shards present) 25; mu_scan.py post 9; claims layer (core, correction, MC, variants, rebuild, t1new, regress, compare) 367 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number (run05) | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL claim sentence; PRD Sec. V F | P̄₁ direct 0.7300 (rebuilt Table 1 composite; round-3 composite 0.7302; run02 0.7271) | `w/claims_run05/run01/t1new/C154_correction_summary.json` : `marginal/equal_mass/D1_direct_pole/Pbar` | same archived file (C161) |
| PRL range; PRD Sec. V F | P̄₁ for σ-pole, MSR, m̄_t, ATLAS+CMS, CMS, BKPV/FJ variant | same : `marginal/equal_mass/*` | same |
| PRL one sentence; PRD Sec. V F | Bayes B_ex, B_sel | same : `bayes` | same |
| PRD Sec. V F | HL-LHC separation 1.7–2.5σ | same : `HLLHC` | same |
| PRL Fig. 2; PRD Sec. V F | critical masses incl. P₁ = 0.05; exclusion thresholds; FCC-ee 5.3σ | `w/claims_run05/run01/C154_summary.json` : `critical_mt`, `exclusion_mt`, `future`; `future_reach.csv`, `exclusion_mt.csv` | same |
| PRL End Matter B; PRD Sec. V F | FCC-ee exclusion strength: central P₁ = 0.023 and B_ex = 0.033 at m̂ = 173.68 GeV (`\PoneAtExclFCC`, `\BexFCC`) | `w/claims_run05/run01/fcc_exclusion.json` : `headline` | same archived file (R18) |
| PRL Fig. 2(a, b) data | P₁(m_t, α_s) grid, contours, ellipses, ξ curves | `fig2a_*.csv`, `fig2b_*` in `w/claims_run05/run01/` | same |
| PRL Table I (rebuilt) | model composite +0.0721/−0.0842 at P₁ = 0.5; matching row +0.012/−0.049 | `w/claims_run05/run01/table1_rebuild.json`, `table1_error_budget_run05.csv` | same |
| PRD Sec. V D, App. D 2 | P₂ critical 172.380/172.764/173.131; B3 172.486/172.953/173.414 | `w/claims_run05/run01/crit_variants.json` : `run05` | same |
| PRD Sec. V F | independent MC P̄₁ (0.7304 direct) | `w/claims_run05/run01/pbar_direct_mc.json` : `results` | same |
