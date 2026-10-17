# 08_robustness_w0 — background sensitivity, inflation-scale item, B3-rule w₀ (D124 / C152)

**Physics.** The background enters the Higgs chain only through δ_μ = Δ ln μ_b and δ_h = Δ ln(hL_b) (exact bounce
scaling): I(m_t) = ∫₀⁵ γ(m_t, ℓ(u) + δ_μ) du / (hL_b e^{δ_h} √cosh 2u), ℓ(u) = −½ ln cosh 2u. Content inputs (ξ,
N_C^eff, A_s, next zero-mode order, κ_E) move m_c by ≤ 0.009 GeV. The dominant continuous inputs are the transit
duration (local dm_c/d ln η = −0.19/−0.31/−0.47 GeV; e-fold finite differences −0.170/−0.270/−0.394 GeV) and the
inflation scale r ∈ [10⁻³, 10⁻²] (±0.035/±0.067/±0.112 GeV, row (vi) of PRL Table 1). The B3 rule (time-symmetric real
state of minimal energy variance in the centred Gaussian family) gives w₀ = 0.9945/0.9907/0.9852/0.9801 at
m_t = 172.2/172.6/173.0/173.4 and moves the critical masses by ≤ 0.016 GeV.

**Method.** `C152_core.py` (2D_ex + NNLO + cross-term roots at an arbitrary scale ℓ; `jobS` = `C149_direct.jobX`
with a shifted node scale), `C152_post.py` (sensitivities, (vi)), `C152_w0.py` (B3 w₀ and P = w₀P₁ critical masses),
`C152_account.py` (background bookkeeping). Base = the run02 node-direct pipeline (as in D124; run05 changes m_c by
+0.004…+0.007 GeV, see `09_claims/crit_variants` for the B3 critical masses at run05: 172.486/172.953/173.414).

**Run.**
```
python run.py --quick                  # ~2 min: method test at l = 0 + post/w0/accounting on archived shards
python run.py --full --workers 15      # + 47 scale-shifted node jobs (original 1403 s with 3 workers)
```
Original code: `04_calculations/code/20261014_claude_v12_r3/higgs_robustness/`; results `…/results/
20261014_claude_v12_r3/higgs_robustness/run01` (gauge profiles 168 MB not archived; recomputed); derivation
`04_calculations/derivations/20261014_claude_v12_r3/higgs_robustness/D124_HIGGS_ROBUSTNESS_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:56:18Z | 6/6 | 0 | C152_core.py test 45; C152_post.py 5; C152_w0.py 15; C152_account.py 1 |
| full | 2026-10-15T19:28:46Z | 7/7 | 0 | C152_core.py test 27; C152_core.py all -0.3147 -0.8803 0.2705 -0.014531 526; C152_post.py 4; C152_w0.py 11; C152_account.py 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL Table I row (vi) | ±0.035/±0.067/±0.112 GeV (r ∈ [10⁻³, 10⁻²]); r = 10⁻⁴ tail −0.089/−0.1725/−0.2795 | `r3/higgs_robustness/run01/C152_input_dmc.json` : `vi_inflation_scale` | same archived file |
| PRD Sec. V D–F | d ln I/dλ_b −805.4/−487.8/−319.2 | same : `slope_update/dlnI_dlam_b` | same |
| PRD Sec. V D, App. D 2 | e-fold finite differences of hL_b (−0.170/−0.270/−0.394) | `C152_post.json` | same |
| PRD Sec. V D, App. D 2 | w₀(B3) 0.9945/0.9907/0.9852/0.9801; critical 172.4819/172.9473/173.4076 (run02 base) | `C152_mc_w0.json` : `with_B3_w0_of_mt` ; `C152_w0_B3.json` | same |
| report Sec. 11.3 | h/M_P 6.054e-6 (pipeline) vs 6.116e-6 (v11): +0.0006/+0.0012/+0.0020 GeV | `C152_accounting.json` | same |
