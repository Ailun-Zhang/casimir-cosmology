# 05_fulldim_validation — full-dimensional check of the reduced zero-mode model (D123 / C151 / T14a)

**Physics.** A Dirichlet wall at the gauge-invariant radius r = |φ| = r_w is the only exact node of the 12-dimensional
zero-mode problem (SU(2)×U(1) constant modes + Higgs), so a walled diffusion Monte Carlo has no fixed-node error.
For λ = −0.010 and −0.013 the full DMC minus the same-wall 2D_ex energy is ΔQ = −0.0032(8)/−0.0034(9), independent
of the wall position and of λ (λ = 0: −0.0043(9)): the reduction error is a constant energy shift of ≈ −0.0035 inside
the well. Four independent width estimates give γ_full/γ_2D = 0.92–1.03; the critical mass moves by
+0.0043(10)/+0.0070(17) GeV (P₁ = 0.9/0.5) ⇒ the reduced-model error is credibly ≤ 0.1 GeV.

**Method.** `T14a_dmc.py` (walled DMC, 11 cases × 3 time steps × 2 shards, SHA-256-derived seeds; tau → 0
extrapolation), `T14a_2d.py` (same-wall 2D FD with Richardson), `T14a_gamma.py` (CAP d ln γ/dQ = 7.60(18)/7.12(18);
multi-dimensional adiabatic WKB), `T14a_3d.py` (3D (r, s, ζ) model with an SU(2) collective width coordinate),
`T14a_post.py` (C151 summary). `C140_beyond_bo_t14a.py` is the T14a copy of the D112 model (outputs redirected to the
T14a run; renamed so that it does not shadow `../zmlib/C140_beyond_bo.py`); `C72_common.py` = 12-dim potential pieces.
`dmc_one_shard.py` (new) reruns one production shard.

**Status.** Numerically checked with independent methods; conditions: v11 couplings, round-2 model (no cross terms),
ΔQ → γ conversion assumes an unchanged barrier (supported by WKB and tail amplitude). PRL Table 1 takes the one-sided
envelope +0.011/+0.018/+0.025 GeV (P₁ = 0.1 linearly extrapolated).

**Run.**
```
python run.py --quick                  # ~1.5 min: barrier/walls, one DMC shard (bit-identical), combine + post
python run.py --full --workers 15      # all 66 DMC shards + 2D FD + CAP + WKB + 3D + post
```
Original runtime: DMC ~30 min with 3 workers; CAP ~4 min; total ≈ 45 min.
Original code: `04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/`; results `…/results/20261014_claude_v12_r3/
fulldim_dmc/run01`; derivation `04_calculations/derivations/20261014_claude_v12_r3/fulldim_dmc/
D123_FULLDIM_ZERO_MODE_CHECK_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:54:39Z | 8/8 | 0 | T14a_profile.py 5; DMC shard m010_rw5.12 tau0.04 k0 29; T14a_dmc.py combine 1; T14a_post.py 1 |
| full | 2026-10-15T19:19:17Z | 9/9 | 0 | T14a_profile.py 4; T14a_dmc.py run all 15 337; T14a_2d.py 58; T14a_gamma.py cap 106; T14a_gamma.py wkb 33; T14a_3d.py run 440; T14a_post.py 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRD Sec. V B | ΔQ (same wall) −0.0032(8) (λ = −0.010), −0.0034(9) (−0.013) | `r3/fulldim_dmc/run01/C151_summary.json` : `lam_neg/-0.01/dQ_ex_main`, `lam_neg/-0.013/dQ_ex_main` | same archived file |
| PRD Sec. V B | λ = 0: Q_full 4.44659(94) vs C72 4.4478(13) | same : `lam0_regression` | same |
| PRD Sec. V B | tail amplitude A = 0.92(16)/1.03(7); WKB 0.937/0.973; CAP d ln γ/dQ | same : `lam_neg/*/tail_fit`, `wkb`, `conversion` | same |
| PRD Sec. V B | 3D collective coordinate γ_3D/γ_2D 0.9992–1.0000 | same : `model_3D_zeta` | same |
| PRL Table I 'full-dimensional validation' | Δm_c +0.0046(11)/+0.0075(19) GeV with the run05 slopes (`09_claims`: `w/claims_run05/run01/table1_rebuild.json` : `t14a/central`); C151 with the earlier slopes +0.0043(10)/+0.0070(17) | same : `delta_m_c` | same |
| PRD Sec. V B | DMC energies per wall and tau | `r3/fulldim_dmc/run01/T14a_dmc_combined.json`, `shards/*.json` | same archived files |
