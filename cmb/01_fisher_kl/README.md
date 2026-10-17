# 01_fisher_kl — torus T+E covariance, shape Fisher and exact KL (ideal full sky)

**Physics.** On a three-torus the CMB T and E fields have the covariance
C^{XY}_{ℓm,ℓ'm'} = (4π)²/V i^{ℓ−ℓ'} Σ_n P_ζ(k_n) W Δ^X_ℓ Δ^Y_ℓ' Y_ℓm Y_ℓ'm' + continuum, with k_n = (2π/L₀) M n
(M = exp(−β) for a shape β). The shape Fisher matrix and the exact Gaussian KL divergence D between a fossil-shaped
torus and the cube of equal volume measure readability (criterion D ≥ 4.5, i.e. 2D ≥ 9). The C59 Fisher gives
σ(β_E) = 0.023 / 0.073 / 0.25 at L₀ = 1.05 / 1.1 / 1.2 d_LSS (ℓ ≤ 30); C75 gives the v11 readability window of single
fossils (superseded for the ensemble statement by `../02_ensemble`).

**Method.** `C59_torus.py` (builder), `C59_transfer.py` + `C59_transfer_cutoff_scan.py` (CAMB 2.0.4 transfer with the
low-k continuation fix; shipped table used otherwise), `C59_run.py`, `C59_analyze.py`, `C59_validate.py`;
`C75_common.py`, `C75_produce.py`, `C75_analyze.py`, `C75_summary.py`; new `kl_quick.py` (end-to-end KL check).
Derivations D52 (C59, C69, C70), D57 (C75). Status: numerical, ideal full sky, no noise unless stated.

**Run.** `--quick` (~40 s): C59 unit/parity checks; fresh covariance builds of the free fossil and the cube at
L₀ = 1.07 d_LSS with the shipped transfer table, exact KL (T+E and T) compared with the archive; C75 summary.
`--full`: transfer (if CAMB is installed; otherwise the shipped table and its metadata), all C59 production stages
(conv, conv2, fine, main, full, exact; 1.3 GB cache), C59_analyze, continuum test, all C75 stages (0.7 GB), analyses.
`--full --c59-only`: the C59 part only (continuum test taken from the archive); used in packaging (2026-10-15/16).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:20:06Z | 5/5 | 0 | C59 validate units 3; C59 validate parity 1; C75 validate builds (fossil + cube at 1.07) 25; exact KL of the fresh pair 1; C75 summary 0 |
| full | 2026-10-16T00:24:21Z | 3/5 | 2 | C59 validate units 2; C59 validate parity 1; C59 run conv 0; C59 run conv2 825; C59 run fine 153; C59 run main 0; C59 run full 0; C59 run exact 0; C59 analyze 142 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VIII B; PRD App. F (covariance) | KL D_TE, D_T of the free fossil at 1.07 (end-to-end check) | `checks/kl_quick.json` : `D_TE_fresh_fresh`, `D_T_fresh_fresh` | `04_calculations/results/20261011_claude_followup/C75/C75_results.json` : `validation/fresh_vs_cached/s1.07_free_fossil` |
| PRD Sec. VIII B | Fisher σ(β_E) = 0.023 / 0.073 / 0.25 at L₀ = 1.05 / 1.1 / 1.2 (CMB-23, v11 values) | `r5/obs_fisher/C59_shape_fisher.json` (`--full`) | `.../20261009_claude_R5/obs_fisher/C59_shape_fisher.json` |
| PRD Sec. VIII B (v11 comparison) | v11 window 1.093–1.133 for the typical fossil, circle bound 1.0931 (CMB-06, 22) | `fu/C75/C75_summary.json` : `per_sigma/*` | `.../C75/C75_summary.json` |
| PRD App. F | transfer-table metadata, low-k fix | `r5/obs_fisher/C59_transfer.json` | `.../obs_fisher/C59_transfer.json` |
