# 01_sm_mapping — SM couplings at the bounce scale (D120 / C148, round 3, run01 + run02)

**Physics.** The zero-mode criterion compares the finite-volume threshold λ_c with the Standard-Model quartic
λ(μ_b) at μ_b = M̄_P/722.0 = 3.3726×10¹⁵ GeV (1/L_b of the v11 plateau background, L_b M̄_P = 722). This step maps
(M_t^pole, α_s(M_Z), M_h) to MS-bar couplings at μ_b and along the turn, μ(u) = μ_b/√cosh 2u.

**Method.** SMDR-1.3-type NNLO matching at μ₀ = M_t (scheme `SMDR25`; Python port `../zmlib/sm_provider.py`) and the
SMDR-port RGE (full three-loop, four-loop gauge, leading four/five-loop QCD; β functions generated from SMDR's
`betas.c` into `../zmlib/_smdr_betas.py`); four-loop pole ↔ MS-bar top mass (`../zmlib/qcd.py`).
Cross-check with the compiled SMDR 1.3 (`../../third_party/smdr/`): the driver `c148_fit.c` (a verbatim copy of
`SMDR_Fit_Inputs` with the evaluation scales as arguments) refits the MS-bar couplings with all scales × ξ
(ξ = 1/4 … 4); `run02_post.py analyze` compares compiled and Python results.

**Status.** Numerically checked (compiled SMDR, Python port, Wolfram W1–W3 in D120; blind V1 with `mr`, R16).
The SM-side theory error used in the paper (±0.07 GeV for m̄_t, ±0.10 GeV pole, four-loop truncated) is the V1
`DIAGNOSIS.json` recommendation (R16 review, not part of this package); the compiled matching-scale variation
(+0.033/−0.019 GeV at P₁ = 0.5) is reproduced here.

**Files.** `sm_point.py` (new; repeats the production calls of `C148_post.py`), `R3_mbar.py` (m̄_t of the critical
pole masses), `C148_selftest.py`, `C148_bench.py` (needs the SMDR benchmark data from `third_party/smdr`),
`C148_post.py` (needs the historical round-2 C144-pipeline hazard runs, archived in `expected/…/hazard_runs`; not
rerun), `run02_fits.py`, `run02_post.py`, `r02.py`, `r3util.py`, `c148_fit.c`, optional Wolfram cross-check
`mk_wolfram.py` + `C148_wolfram.wl` (env `WOLFRAMSCRIPT`). Shared modules: `../zmlib/{sm_provider,sm_rge,qcd,
_smdr_betas,R14_rge}.py`.

**Run.**
```
python run.py --quick            # ~1 min (+14 s compiled refit if third_party/smdr is built)
python run.py --full --workers 8 # ~5 min: C148_bench + 38 compiled refits + analysis
                                 # needs: python third_party/smdr/fetch_and_build.py all ; Windows: set DNEC_MINGW_BIN
```
Original: `04_calculations/code/20261014_claude_v12_r3/sm_mapping/`; results `…/results/20261014_claude_v12_r3/
sm_mapping/run01, run02`; derivation `04_calculations/derivations/20261014_claude_v12_r3/sm_mapping/
D120_SM_MAPPING_NNLO_CN.md`. Expected outputs: `expected/r3/sm_mapping/…` (SHA-256 in `expected/SOURCES.json`).

**Notes.** (i) INV contradiction 1: the critical m̄_t values 162.73/163.16/163.60 come from `R3_mbar_run05.json`
(run05 centre); `R3_mbar_newcentral.json` holds the run02-centre values 162.72/163.16/163.59 — both archived.
(ii) The stock SMDR applications are not used: `calc_fit` crashed (round 3) and `calc_all` did not finish within
4 h with MinGW gcc 5.3 (killed during packaging); `c148_fit` works and reproduces the archived fits bit for bit.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:48:55Z | 14/15 | 1 | sm_point.py 1; R3_mbar.py 3; C148_selftest.py 4; abs_stability m_stab m0 6; run02_post.py analyze 5 |
| full | 2026-10-15T15:19:21Z | 14/14 | 0 | sm_point.py 1; R3_mbar.py 3; C148_selftest.py 4; C148_bench.py 16; run02_fits.py 249; run02_post.py analyze 10; compiled refit xi1 172.60 20 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL core sentence; PRD App. D 3 | λ(μ_b) = −0.0097741 at (172.60, 0.1180, 125.13) | `r3/sm_mapping/package/sm_point.json` : `point_172p60_0p1180_125p13/lam_mu_b` | `r3/sm_mapping/run01/C148_summary.json` : same key |
| PRD App. D 3 | y_t(μ_b) = 0.4278267; g₂/g_Y/g₃ = 0.52776/0.44277/0.53565 | same file : `yt_mu_b`, `g2_mu_b`, `gY_mu_b`, `g3_mu_b` | same |
| PRL (μ_b ≈ M̄_P/722) | μ_b = 3.3726×10¹⁵ GeV | `sm_point.json` : `mu_b_GeV` | `zmlib/sm_provider.py` (`MPBAR`, `MU_B`) |
| PRL headline | λ_c = λ(μ_b; m_c) = −0.0091/−0.0121/−0.0152 | `sm_point.json` : `lam_mu_b_at_run05_critical_mt` | `r3/holonomy_cross/run05/C149_run05_summary.json` : `lam_c` |
| PRL Table I (m̄ form) | m̄_t(m̄_t) 162.7281/163.1637/163.6012 (α_s 0.1180; also 0.1171/0.1189) | `r3/sm_mapping/run02/R3_mbar_run05.json` : `new_central/<α_s>/P1=*/mbar_mbar_QCD4L` | same archived file |
| PRD App. D 3 | compiled SMDR − Python at Q = M_t; λ(μ_b) SMDR-RG / own RG / Python | `r3/sm_mapping/run02/C148r2_summary.json` : `A_match_at_Mt_compiled_minus_python`, `B_mu_b` | same archived file |
| PRD App. D 3 | compiled matching-scale variation +0.0331/−0.0186 GeV | `C148r2_summary.json` : `E_budget_P1=0.5/a_matching_scale_compiledSMDR/common_xi` | same |
| PRD App. D 3 | benchmarks (BKPV critical M_t 171.453; r2→r3 chain +0.063 GeV) | `r3/sm_mapping/run01/C148_bench.json` (full mode) | `run01/C148_bench.json`; `C148_summary.json` : `r2_to_r3_chain_at_172p6` |
| report Sec. 10.5 | absolute-stability bound m_stab = 170.99 GeV (m0 − m_stab = 0.10 GeV) | `w/abs_stability/abs_stability.json` : `central/m_stab_pole`, `central/m0_pole` | `abs_stability.json` (this directory; `abs_stability.py`, rep-B 2026-10-16) |
