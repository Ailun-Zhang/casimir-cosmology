# shape/ — emptiness gate, quantum shape at the turn, fossil, nonlinear survival, spectra (DNEC v12)

This chain reproduces PRD Secs. VI (`sec:gate`), VII (`sec:shape`, `sec:quantum`) and VIII A (spectra), App. E (`app:theorem`)
and report Secs. 13 and 15 (gate, photon pairs, nonlinear survival): the quanta budget Σ|n| < 0.2666 (1−ξ) N_C ("about five" lowest
bosonic quanta), the reheating coupling 10⁹⁴, the latest natural preparation ≈ 7 e-folds before the turn, the
closed-form floor σ_floor² = K(κ_E)/X (σ_E ≈ 0.104; 0.0934 at next order), the real-moment theorem and G(t_x), the
conditional theorem t_*(99%) = 0.237/h and t_*(90%) ≈ 0.35/h, the survival thresholds s₉₉ ≈ 2.8–2.95, the plateau
background check, the SM Moyal correction, the B1 Wheeler–DeWitt test and n_s, r.

## Dependency order
```
shapelib/ ................ C60 Ewald + integrator, C76 (xi = 0.30 model), C142 (D114), C153_nl, C153_run02, helpers
01_emptiness_gate ........ budget, R = 10^94, T_max, 3-dof crossing, photon branch            (C55, C56, C153_emptiness, C80)
02_realtime_theorem ...... G(t_x), floor closed form, conditional theorem t_*                 (C137, C153_linear, C153 runs)
03_fossil ................ floor, natural state, sigma_E vs xi, side difference, NL ratio      (C58, C62, C63, C77, C82, C143, C60, C76)
04_nonlinear_survival .... s50/s90/s99, R1 states, plateau background, next order            (C142, C160_plateau, C153 run02)
05_moyal_wdw ............. SM Moyal, shear criterion with quantum volume, B1 WDW, B3 lemma     (C160, C141)
06_primordial_spectra .... n_s, r                                                              (C64)
```
Cross-chain inputs: the next-order κ_E comes from `casimir/03`; `02` stages the ensemble layer of `cmb/02` for
`C153_summary.json`.  `_shape_paths.py` is the path shim (work tree `output/shape/<mode>`, mirror prefixes `r5/` R5
2026-10-09, `fu/` follow-up 2026-10-11, `r1/ r2/ r3/` v12 research rounds 1–3 2026-10-13/14).
`PACKAGE_PATCHES.json` lists every code change (paths, worker caps lifted, package note in docstrings). The one-off
builder that assembled casimir/, shape/ and cmb/ from the research tree is not part of the release (it is not needed
to run anything; its substitutions are recorded in each chain's `PACKAGE_PATCHES.json`).

## Trajectory archives (shipped, zipped)
The nonlinear results are statistics over per-trajectory JSON records. They are shipped as zip archives in the
`expected/` directories (unpacked into the work tree by `--quick`):
| archive | records | size | used by |
|---|---|---|---|
| `02/expected/r3/shape_hardening/run01/draws.zip` | 17 920 (C153 run01: grid, phase, never, adiab, next, width) | 19.8 MB | 02, 04 |
| `02/expected/r3/shape_hardening/run02/draws.zip` | 28 672 (C153 run02, N = 2048) | 33.6 MB | 02, 04 |
| `03/expected/r5/nonlinear_SM/draws.zip` | 2 592 (C60 free fields) | 2.4 MB | 03 |
| `03/expected/fu/C76/draws.zip` | 4 224 (C76, ξ = 0.30) | 3.9 MB | 03, 04 |
| `04/expected/r2/shape_nonlinear_threshold/run01/draws.zip` | 6 656 (C142 / D114) | 6.6 MB | 04 |
| `04/expected/r3/shape_tier2/run01/plateau/draws.zip` | 2 560 (C160 plateau) | 2.8 MB | 04 |
Every `--quick` re-integrates a few archived trajectories with the production integrator and compares the records
key by key (`shapelib/redraw_check.py`); `--full` recomputes all of them (resumable, `--fresh` restarts).

## How to run
```
python shape/02_realtime_theorem/run.py --quick
python shape/04_nonlinear_survival/run.py --full --workers 15
```

## Runtimes (Intel Core Ultra X9 378H, 16 cores, Windows 11)
| step | --quick (this run) | --full | original production (research rounds) |
|---|---|---|---|
| 01 | 60–70 s | 1760 s | C55 68 + 61 s; C56 NC100 1235 s, SM_AAA 1510 s (11 processes) |
| 02 | 99 s | 5424 s (46 592 trajectories from scratch) | C153 run01 3438 s + width 221 s + run02 ~2 h + smoke 615 s (4 processes) |
| 03 | 36–58 s | 531 s (6 816 trajectories) | C76 ind 247 s (2048 draws); C60 not recorded |
| 04 | 89 s | 1354 s (C142 6 656 + plateau 2 560 trajectories) | C142 ≈ 40 min (5 processes); plateau 1095 + 2443 s (2 processes) |
| 05 | 10–16 min | 2483 s | Moyal 111 s, 1D 1057 s, shear 235 s, B1 z_s = 0 1106 s |
| 06 | 1–6 min | 137 s | — |
Full runs: 2026-10-15, `--workers 10`, concurrently with `cmb/01 --full` on 5 workers. Every recomputed trajectory
reproduces its archived record and every analysis its archived summary (tolerance 1e-9 relative); the full-mode
comparisons exclude wall-time statistics (`runtime_mean_s`, `runtime_max_s`) and were re-evaluated from the full
outputs once this rule was added (no recomputation; noted in VERIFY.json).

## Verification summary
<!-- VERIFY-SUMMARY-START -->
| result directory | quick | reduced | full |
|---|---|---|---|
| shape/01_emptiness_gate | 11/11 | - | 15/15 |
| shape/02_realtime_theorem | 19/19 | - | 16/16 |
| shape/03_fossil | 16/16 | - | 12/12 |
| shape/04_nonlinear_survival | 13/13 | - | 9/9 |
| shape/05_moyal_wdw | 9/9 | - | 23/23 |
| shape/06_primordial_spectra | 5/5 | - | 5/5 |
<!-- VERIFY-SUMMARY-END -->

## Notation warnings
- `r`, `r_SM`, `R_ZM` in C76/C142/C153/C160 code = zero-mode fraction ξ (0.299897), not the tensor ratio.
- `s` = turning-point amplitude multiple (s₉₉ etc.); in cmb/ tables `s` = L₀/d_LSS.
- The H convention of the floor: `C153_linear.json` `floor_conventions` lists σ_floor = 0.1037 (H_eff, canonical),
  0.10378 (D109), 0.1040 (C60/C76 sampling convention, δ = 2.077e-4) and 0.1031 (v11 Table I δ) — the same quantity in
  different H/L_b conventions (≤ 1 %).

## Not packaged / partial coverage
- `C60_twisted_ewald.py` stand-alone validation (agreement with the round-1 C40 kernel) needs the research tree; its
  archived output is shipped (`03/expected/r5/nonlinear_SM/C60_validation_ewald.json`).
- B1 z_s = 2.5 production run was stopped during case B4 (contaminated, D131 §4.2(d)); `05/b1_cases.py` recomputes the
  valid cases one by one.
- The independent reimplementations of the shape side (R14 state-selection check, R16 V4 blind, T19 semi-blind) are
  archived in `../validation/` (`early_rounds/R14/`, `shape/`, `T19_semiblind/`); `validation/run.py` compares V4
  with this chain.
- Wolfram cross-checks (`02/*.wl`) are optional (env `WOLFRAMSCRIPT`).
- The archived C142 `smoke.json` (D114 pre-production smoke) used a deeper pinch stop (eig G < 1e-6) than the
  production runs (1e-3); `04 --full` runs the smoke as a path check only and does not compare it.
- The archived C55 production file (`C55_window_gate.json`) predates the meta keys `nq/qmin/qmax/rtol` that the
  packaged code writes; all numbers agree, the key `nq` is not compared by `01`.
