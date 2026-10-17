# 02_ensemble — E-plane fossil ensemble, read edge and the P(read | L₀) window (C145/D117)

**Physics.** A fossil is β = A σ_E (cos θ X_d + sin θ X_d2) with A Rayleigh-distributed and θ in the E plane. For
55 fossils × a pool of 19 actual reference cubes the exact TE KL divergence (ℓ ≤ 80, x_max = 240, profiled over the
reference volume) shows that readability is set by the shortest side u = L₀ min_i e^{β_ii}: D_prof ≥ 4.5 for
u ≤ u_max ≈ 1.021–1.039 d_LSS (universal read edge). With the no-circle bound u ≥ 0.985 d_LSS the joint probability
peaks at P_joint = 0.344 at L₀ = 1.075 d_LSS, exceeds 0.1 for L₀ ∈ [1.018, 1.183] and integrates to
∫P_joint dln L₀ = 0.0422 = E[ln(u_max/0.985)]. Next order (σ_E = 0.094): 0.377 at 1.065, [1.015, 1.168], measure 0.0421.
Planck-143-like noise lowers the peak to 0.313 (measure −9 %), temperature alone to 0.286.

**Method.** `ens_core.py`, `run_ens.py` (S1–S4 stages; resumable; `--smoke` reduced configuration, `--refine`),
`postcheck_ens.py` (pool-min profile, convexity edge bounds → `S3_postcheck.json`, the main result),
`make_fossil_table_v2.py` (cleanup T12), `C153_ensemble.py` (other σ_E; D125 part B), `priors_ens.py` (conditional
size priors; reads the archived C79 table, shipped here as `expected/fu/C79/C79_results.json`). Derivations D117 + R15
correction addendum + T12 wording addendum. Status: numerical, conditional (ideal full sky, TE ℓ ≤ 80, D ≥ 4.5).

**Run.** `--quick` (~2 min): postcheck, fossil table, C153 ensemble and priors on the archived KL tables.
`--reduced` (27 s): `run_ens.py --smoke` from scratch (ℓ ≤ 12, x_max = 60; transfer rows from
`inputs/C59_transfer_l12.npz`, a derived subset of the snapshot table) — every build, cross-KL, profile, Brent, noise
and analysis path — compared with the archived smoke run (the Brent list is stored in completion order and is not
compared). `--full`: uses the ℓ ≤ 80 transfer table shipped in `data/C59_transfer.npz` (85 MB, SHA-256 `8d0ea21f…df8f`,
the hash recorded in `ens_core.py`; generated with CAMB 2.0.4 by `data/high_l_transfer.py`, settings in its header and
in `data/SOURCES.json`, metadata `data/C59_transfer.json`, log `data/transfer.log`; env `DNEC_CMB_TRANSFER` overrides)
and preferably the snapshot cache (3.3 GB, not shipped; only for the S1 regression); writes 15.5 GB. It was run once for the release on a copy of the package with identical code (2026-10-16, 12 workers, about 70 min: C145 run01 2374 s + refine 827 s; 15.5 GB cache): 13/13 full checks passed, all numbers bit for bit except the cross-KL table (max |diff| 2.3e-13); record in `VERIFY.json` (full entry). The flag list of `S3_postcheck.json` is now written in a fixed order (sorted by shape, A, s, u; the run tables are filled in completion order); the archived file was re-ordered, its content is unchanged.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:21:33Z | 12/12 | 0 | C145 postcheck (pool-min profile, edges, windows) 9; fossil_table_v2 2; C153 ensemble for other sigma_E 17; C145 size-prior experiments 58 |
| reduced | 2026-10-17T02:22:22Z | 3/3 | 0 | C145 smoke run (l up to 12) 23 |
| full | 2026-10-17T01:44:28Z | 13/13 | 0 | C145 run01 2374; C145 run01 refine 827; C145 postcheck (pool-min profile, edges, windows) 8; fossil_table_v2 2; C153 ensemble for other sigma_E 14; C145 size-prior experiments 48 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VIII B, Fig. 7 (P(read\|L₀)) | P_joint peak 0.3440 at 1.075 (CMB-02) | `ai/cmb_ensemble/run01/results/S3_postcheck.json` : `variants/TE80_grid/modes/mid/{Pj_max,L0_Pj_max}` | `04_calculations/results/20261013_argon_codex_intake/cmb_ensemble/run01/results/S3_postcheck.json` |
| PRD Sec. VIII B | P_joint ≥ 0.1 for [1.0175, 1.1825]; ≥ 0.3 / 0.2 / 0.05 windows (CMB-03, 04) | same : `.../win_joint/{0.1,0.3,0.2,0.05}` | same |
| PRD Sec. VIII B | integrated measure 0.04223 (CMB-05); representative window [1.0931, 1.1394] (CMB-06) | same : `.../mean_span`, `.../rep_window` | same |
| PRD Sec. VIII B | P_cond ≥ 0.5 up to 1.090 (CMB-07) | same : `.../win_cond/0.5` | same |
| PRD Sec. VIII B | read edge u_max 1.021–1.039 (CMB-01) | same : `variants/TE80_grid/edges` | same |
| PRD Sec. VIII B | noise 0.313 / 0.0384; T only 0.286 / 0.0352 (CMB-10, 11) | same : `variants/{noise_TE80_grid,T80_grid}/modes/mid/*` | same |
| PRD Sec. VIII B | next order 0.377 at 1.065; [1.015, 1.1675]; 0.0421; rep window [1.082, 1.128] (CMB-14..17) | `r3/shape_hardening/run01/ensemble/C153_ensemble.json` : `results/0.0940/variants/TE80_grid/mid/*` | `.../20261014_claude_v12_r3/shape_hardening/run01/ensemble/C153_ensemble.json` |
| PRD Sec. VIII B, Fig. 7 (mechanism: D_prof vs u) | 55 fossils (columns: xi = sin 3θ, s = L₀/d_LSS, u) | `r3/cleanup_t12/run01/fossil_table_v2.csv` | `.../cleanup_t12/run01/fossil_table_v2.csv` |
| PRD Sec. VIII B | near-window packet 1.7 %, α = 17.6 for P = 0.5 (CMB-12, 13) | `ai/cmb_ensemble/run01/priors/priors_summary.json`, `P4_size_prior_ens.json` | `.../cmb_ensemble/run01/priors/` |
| PRD App. F | KL table, profiles (TE30/40/80, T80, fixed/profiled), Brent, convergence | `ai/cmb_ensemble/run01/results/{kl_table,S2_profiles,S2_brent,S4_convergence,S4_noise,S3_analysis}.json` (archived; `--full` recomputes) | `.../cmb_ensemble/run01/results/` |
