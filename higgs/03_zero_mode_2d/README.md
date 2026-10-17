# 03_zero_mode_2d — 2D (r, |w|) zero-mode model and the direct 2D hazard (D112 / C140; D122 / C150 = T14b)

**Physics.** The SU(2) conditional Gaussian factorises exactly, Ψ = φ(x|w) χ(w); the Born–Huang term B_w is almost
entirely spurious near r → 0. Replacing χ(w) by χ(r, w) and solving the Higgs radius r and the hypercharge zero mode
|w| jointly gives the 2D model 2D_ex:
H_2D = −½∂_r² + 3/(8r²) − ½Δ_w + C₀ + B_x + ½Ω_eff²|w|² + λr⁴/4 + ΔV_MC(r) + δE_DMC(r).
At λ = 0 its ground state agrees with the full-dimensional DMC to +0.003; for λ < 0 the zero mode is a resonance
E = Q − iγ/2 whose width is 0.39–0.48 of the 1D value (stabilising). D122 solves 2D_ex directly at every hazard node
with SM running couplings (no 1D U_eff bump, no coupling transfer).

**Method.** `../zmlib/C140_beyond_bo.py` (decomposition, 2D Siegert solver: sine-DVR in s = |w|, matrix Numerov in r,
2nd-order WKB outgoing condition, local-window Muller); `../zmlib/C150_twod.py` (node-direct hazard
I = ∫₀⁵ γ du/(hL_b√cosh 2u), P₁ = e^{−I}, critical m_t; coupling source env `C150_COUPLINGS`, run env `C150_RUN`):
run02 = NNLO mapping (baseline of the PRL centre), run01 = R14 mapping (convergence table).
`c150_one_node.py` (new) recomputes one production node job with the production function `job2d`.

**Status.** Decomposition identities and projections proven (Wolfram id1–id6); SU(2) conditional Gaussian + r-only
δE_DMC conditional (δE_DMC still the v11 spline, effect estimated ≤ 0.002 GeV); widths numerically checked (CAP, box
checks); keystone blind check PASS (2D_ex′ 172.521/173.017/173.520 vs 172.518/173.014/173.515).

**Run.**
```
python run.py --quick                   # ~3-5 min: C140 lam0, one T14b node (u = 0), post for run02 and run01
python run.py --full --workers 15       # 2D main + als node jobs of T14b run02 from scratch (+ gauge profiles), post
```
Original runtime: C150 ~30 min with ≤3 workers (all modes), C140 res2d ~13 min. In `--full` the 1D seed chains
(`C150_twod.py oned`) are staged from the archive: their `c144` variants need the round-2 C144 gauge profiles
(`r2/higgs_final/run01/profiles`, 37 files, 26 MB, produced by round-2 `C144_model.py`, not packaged); the 2D roots
do not depend on the seeds (recomputed node shards are bit-identical). Gauge profiles of T14b (220 files, 92 MB) are
not archived; they are recomputed (deterministic) by `C150_twod.get_profile`.
Original code: `04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/`, `…/20261014_claude_v12_r3/
twod_direct_sm/`; derivations `…/derivations/20261013_claude_v12_r2/higgs_beyond_bo/D112_HIGGS_BEYOND_BO_CN.md`,
`…/20261014_claude_v12_r3/twod_direct_sm/D122_TWOD_DIRECT_HAZARD_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:53:10Z | 10/10 | 0 | C140 lam0 100; C150 node u=0 (run02) 86; C150 post run02 4; C150 post run01 4 |
| full | 2026-10-15T19:02:59Z | 10/10 | 0 | C140 lam0 60; C150 two main run02 79; C150 two als run02 78; C150 post run02 4; C150 post run01 3 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRD Sec. V B | λ = 0: 2D_ex 4.45086, 2D_G 4.5077 vs DMC 4.4478(13) | `r2/higgs_beyond_bo/run01/C140_lam0.json` : `Q`, `test/Q_2D_ex`, `test/minus_C72` | `r2/higgs_beyond_bo/run01/C140_lam0.json` |
| report Sec. 11.1 (Table 14) (2D direct, R14) | 172.5185/173.0140/173.5155 | `r3/twod_direct_sm/run01/C150_summary.json` : `main_2D_direct_as0.1180/critical_mt` | same archived file |
| report Sec. 11.1 (Table 14) (2D_G) | 172.4671/172.9412 | `…/run01/C150_summary.json` : `twoD_G_direct/critical_mt` | same |
| report Sec. 11.1 (Table 14) (rows MU → 2D direct) | MU 172.2099/172.5837 … | `…/run01/C150_summary.json` : `convergence_table` | same |
| report Sec. 11.1 (Table 14) (+ NNLO) | 172.5826/173.0775/173.5784 | `r3/twod_direct_sm/run02/C150_summary.json` : `main_2D_direct_as0.1180/critical_mt` | same archived file |
| PRD Sec. V B–D (node widths, no cross terms) | γ(m_t, u) at 11 nodes | `r3/twod_direct_sm/run02/shards2d/2Dex_*.json` : `cols/*/gamma2` | same archived shards |
| PRD Sec. V C; report Sec. 8 (R19 N4) | Higgs-only inversion of the soft hypercharge direction for r < 0.774/0.776/0.776 (λ = −0.008/−0.010/−0.0143; D112 sec. 6.1) | archive only | `r2/higgs_beyond_bo/run01/C140_cross_note.json` : `<λ>/r_X_inverted_below` (with `C140_crossX.json`) |
