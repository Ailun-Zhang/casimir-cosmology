# higgs/ — the Higgs zero-mode criterion chain (DNEC v12)

This part of the package reproduces, or contains in its archived outputs, every number of the papers that concerns
the Higgs zero mode at the turning point:
the finite-volume threshold λ_c(μ_b) ≈ −0.012 at μ_b = M̄_P/722 = 3.37×10¹⁵ GeV, the SM value λ(μ_b) = −0.0098,
the single-branch survival P₁(m_t, α_s) and the critical top masses 172.50/172.96/173.42 GeV (P₁ = 0.9/0.5/0.1,
α_s = 0.1180), the error budget, the marginal survival P̄₁ ≈ 0.73, the condition ξ_H(μ_b) ≳ 0.074–0.08, and the
supporting checks (full-dimensional DMC, universality, inflation-era stability, two-branch analysis).
**All results are conditional results of the reduced zero-mode model** (SU(2) conditional Gaussian + 2D (r, |w|) +
DMC shape + one-loop finite-volume matching + cross terms; single-branch hazard; Gamow/B3 preparation with w₀ ≈ 1;
explicit ξ_H; plateau cosh background); see the derivations D108–D132 named in each README.

## Dependency order (one physics sentence per step)
```
01_sm_mapping ............ SM couplings (λ, g2, gY, g3, y_t) at μ_b and along the turn μ(u) = μ_b/√cosh2u (NNLO + SMDR)
02_finite_volume_matching  consistent one-loop finite-volume matching dV_MC(r) (Coulomb gauge, MS-bar FV sums, μ = 1/L)
        │
03_zero_mode_2d .......... 2D (r,|w|) model; resonance widths at the 11 hazard nodes without cross terms (T14b)
        │
04_cross_terms ........... zero-mode/non-zero-mode cross terms (all SM fields, complete one-loop r²c² set): run05 widths γ(m_t,u)
        │
06_hazard_criterion ...... I = ∫γ du/(hL_b√cosh2u), P₁ = e^{−I}: critical m_t, λ_c, m̄_t, convergence table   ◄ PRL centre
        │
05_fulldim_validation .... walled 12-dim DMC + independent width estimates: reduced-model error Δm_c = +0.004/+0.007 GeV
07_realtime_xi ........... real-time evolution through the turn: ξ_c(m_t), universal ξ_c(P₁)  (run05 centre)
08_robustness_w0 ......... background sensitivity, inflation-scale item (vi), B3-rule w₀
09_claims ................ P̄₁, Bayes factors, exclusion, FCC-ee/HL-LHC reach, rebuilt Table 1  (run05 centre)
10_universality .......... |λ_c| scaling with transit duration and couplings; spin-structure and flux sectors
11_inflation_stability ... MMR2021-type all-mode bound ξ_inf during inflation
12_two_branch ............ both branches: GH interference, P₂ = P₁²/w₀, Fréchet bounds
```
`zmlib/` holds the shared modules (one canonical copy; code changes listed in `zmlib/PACKAGE_PATCHES.json`),
`data/` the few archived inputs produced outside this package (round-1 DMC shape spline `S3_prop.json`, round-2
C144 seeds, D129 `C157_xi_quick.json` used by the 08 bookkeeping).

## How to run
```
python higgs/run_higgs.py --quick              # every step: fast recomputation + comparison with expected/
python higgs/run_higgs.py --full --workers 15  # every step: full production recomputation in dependency order
python higgs/04_cross_terms/run.py --quick     # one step (archived upstream inputs are staged automatically)
python tools/verify_summary.py --update-readmes
```
Work tree: `output/higgs/<quick|full>/` (override `--work` or env `DNEC_HIGGS_WORK`); it mirrors the archive layout
`r1|r2|r3|w/<calculation>/<run>/…` of the research project (r1/r2/r3 = research rounds 1–3 on 2026-10-13/14,
w = writing round 2026-10-15), so a computed file and its archived copy in `<step>/expected/` share the same relative
path. `run.py` stages missing upstream inputs from the archive, never overwrites freshly computed files, and in
`--full` mode clears the step's own cached node shards first (the production code resumes from shard files).
Each `run.py` writes `VERIFY.json` (checks, tolerances, timings, machine). `--quick` always recomputes something
non-trivial with the production code (e.g. one 2D node job, one DMC shard, the hazard integral, the claims layer).

Environment: Python 3.12 with numpy 2.5.1, scipy 1.17.1, mpmath 1.4.1, sympy 1.14.0 (`environment.yml`); the
production code forces single-threaded BLAS and parallelises over node jobs (`--workers`). Optional: compiled SMDR
1.3 (`third_party/smdr`), Wolfram (`*.wl` cross-checks).

## Runtimes (Intel Core Ultra X9 378H, 16 cores, Windows 11; this packaging run)
| step | --quick | --full (workers) | original production run |
|---|---|---|---|
| 01 | 50 s | 303 s (8): 38 compiled SMDR refits | — |
| 02 | 104 s | 578 s (4) | ~5 min (3) |
| 03 | 3–5 min | 225 s (15): 2D main + als node jobs; 1D seeds archived | ~30 min (≤3) |
| 04 | 94 s | 701 s (11): 22 run05 node jobs from scratch | 2 × 8 min (3) |
| 05 | 93 s | 978 s (15): 66 DMC shards 337 s, CAP/WKB/3D | ~45 min (3) |
| 06 | 15 s | 12 s (on the 04 --full widths) | — |
| 07 | 8 s | 312 s (15): 145 nodes, calib, tables, 55 real-time runs | ~8 min (C161) |
| 08 | 114 s | 569 s (15): 47 scale-shifted node jobs 526 s | 1403 s (3) |
| 09 | 137 s | 689 s scan (66 node jobs, 15) + 367 s claims layer | ~3 min + scan |
| 10 | 82 s | 3543 s (U156_coup is capped at 3 workers inside) | — |
| 11 | 73 s | 276 s (15): 95-job scan 166 s | — |
| 12 | 28 s | 410 s (TDSE 359 s, 2 workers inside) | — |

Total `run_higgs.py --full` ≈ 2.6 h on this machine (dominated by 10, 05, 04, 09).

## Verification summary (generated by tools/verify_summary.py)
<!-- VERIFY-SUMMARY-START -->
| result directory | quick | reduced | full |
|---|---|---|---|
| higgs/01_sm_mapping | 14/15 (1 skipped) | - | 14/14 |
| higgs/02_finite_volume_matching | 9/9 | - | 10/10 |
| higgs/03_zero_mode_2d | 10/10 | - | 10/10 |
| higgs/04_cross_terms | 15/15 | - | 16/16 |
| higgs/05_fulldim_validation | 8/8 | - | 9/9 |
| higgs/06_hazard_criterion | 17/17 | - | 17/17 |
| higgs/07_realtime_xi | 3/3 | - | 4/4 |
| higgs/08_robustness_w0 | 6/6 | - | 7/7 |
| higgs/09_claims | 16/16 | - | 14/14 |
| higgs/10_universality | 7/7 | - | 8/8 |
| higgs/11_inflation_stability | 5/5 | - | 6/6 |
| higgs/12_two_branch | 3/3 | - | 4/4 |
<!-- VERIFY-SUMMARY-END -->

## Headline numbers → where they are reproduced
| paper | value | step : file : key |
|---|---|---|
| λ_c(μ_b), P₁ = 0.9/0.5/0.1 | −0.0091/−0.0121/−0.0152 | 04/06 : `r3/holonomy_cross/run05/C149_run05_summary.json` : `lam_c` |
| λ_SM(μ_b) (172.60, 0.1180, 125.13) | −0.0097741 | 01 : `r3/sm_mapping/package/sm_point.json` : `point_172p60_0p1180_125p13/lam_mu_b` |
| m_c^pole (α_s 0.1180) | 172.5024/172.9595/173.4184 | 04/06 : `C149_run05_summary.json` : `crit/0.1180` |
| m̄_t(m̄_t) at m_c | 162.7281/163.1637/163.6012 | 01/06 : `r3/sm_mapping/run02/R3_mbar_run05.json` : `new_central/0.1180` |
| P₁(172.60) | 0.8426 | 04/06 : `C149_run05_summary.json` : `P1/172.6` |
| P̄₁ (direct) | 0.7300 (run05, rebuilt Table 1 composite; round-3 composite 0.7302; run02 0.7271) | 09 : `w/claims_run05/run01/t1new/C154_correction_summary.json` : `marginal/equal_mass/D1_direct_pole/Pbar` |
| ξ_c(172.6) | 0.0738 ± 0.0025 (run05; run02 0.0741) | 07 : `w/claims_run05/run01/xi_run05_summary.json` |
| cross-term shift | −0.080/−0.118/−0.160 GeV | 04 : `C149_run05_summary.json` : `shift_vs_T14b/0.1180` |
| full-dimensional Δm_c | +0.0046(11)/+0.0075(19) GeV (run05 slopes; the C151 record with the earlier slopes: +0.0043(10)/+0.0070(17)) | 09 : `w/claims_run05/run01/table1_rebuild.json` : `t14a/central`; 05 : `r3/fulldim_dmc/run01/C151_summary.json` : `delta_m_c` |
| (vi) inflation scale | ±0.035/±0.067/±0.112 GeV | 08 : `r3/higgs_robustness/run01/C152_input_dmc.json` : `vi_inflation_scale` |
| ξ_inf (EW), m_t = 172.6 | 0.05512 | 11 : `r3/inflation_stability/run01/xi_inf_table.csv` |
| P₂ = P₁² critical (run02 base) | 172.376/172.759/173.124 | 12 : `r3/two_branch/run01/C159_summary.json` : `headline` |

## Not packaged / partial coverage (stated honestly)
- D129 run02 extras (SM/α_s bands of ξ_c, 2D real-time check, R15 items): archived results only (07/expected).
- Rounds 1–2 production runs (D108 MU model, D116/C144 final round-2 hazard) are historical; their outputs used as
  seeds/regressions are archived (`data/`), their raw recomputation is not packaged. The C144 gauge profiles (26 MB)
  needed by the `c144` 1D seed chains of `C150_twod.py oned` are not shipped (03 `--full` stages the 1D seeds).
- The independent reimplementations of the review rounds (blind keystone and V1–V5, semi-blind T19, compute-first
  R14/R15) are archived in `../validation/` (specifications, code with the SHA-256 recorded before the comparison,
  results); `python validation/run.py --quick` compares them with the main calculation. Their codes are archived
  outputs and are not rerun.
- C161 step 6 (field-by-field split of the run05 cross-term shift) is `04_cross_terms/cross_decomp.py`; its 66 node
  shards are archived and `04_cross_terms/run.py` recomputes the split from them (`post`, check `cross_decomp_post`).
- Large intermediate files (gauge profiles, node tables; 40–170 MB per run) are regenerated, not archived.
