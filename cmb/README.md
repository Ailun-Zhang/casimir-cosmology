# cmb/ — reading the shape fossil in the CMB (DNEC v12)

This chain reproduces PRD Sec. VIII B (s10_sky.tex) and the appendices on the CMB covariance and the conditional evidence:
the T+E harmonic covariance of a three-torus, the shape Fisher information and exact Gaussian KL divergences
(ideal full sky), the E-plane fossil ensemble with the universal read edge u_max ≈ 1.02–1.04 d_LSS, the joint
readability P(read ∧ no circles | L₀) (peak 0.344 at L₀ = 1.075 d_LSS, ≥ 0.1 for [1.018, 1.183], integrated measure
0.042; next order 0.377 at 1.065), and the joint inflaton–shape state (p(N) ∝ 1/N, readability ≲ 1.5 %).
**All results are conditional**: ideal full sky, TE ℓ ≤ 80, D ≥ 4.5 reading criterion, no masks/foregrounds/τ.

## Dependency order
```
01_fisher_kl ..... C59 covariance builder + transfer table (CAMB 2.0.4, l <= 40), Fisher, exact KL; C75 v11 window
02_ensemble ...... C145 E-plane ensemble (l <= 80): KL table -> profiles -> postcheck/windows; fossil table; C153 sigma_E variants; priors
03_joint_state ... C147 joint inflaton-shape state, size distribution, readability
```
`_cmb_paths.py` = path shim (work tree `output/cmb/<mode>`; mirror prefixes `r5/`, `fu/`, `r3/`, `ai/` = Argon/Codex
intake 2026-10-13). Code changes: `PACKAGE_PATCHES.json`.

## Large files: what is shipped and how the rest is regenerated
| item | size | shipped? | generation |
|---|---|---|---|
| CAMB transfer table l ≤ 40 (`01/expected/r5/obs_fisher/C59_transfer.npz`) | 34 MB | yes | `C59_transfer.py` with CAMB 2.0.4 (`pip install camb==2.0.4`) |
| C59 covariance cache (`r5/obs_fisher/cache`) | 1.3 GB, 128 files | no | `01 --full` (`C59_run.py conv/main/full/exact`, 113 tasks, 130–1050 s each on 8 processes) |
| C75 covariance cache (`fu/C75/cache`) | 668 MB, 83 files | no | `01 --full` (`C75_produce.py` all stages) |
| snapshot transfer table l ≤ 80 (`02_ensemble/data/C59_transfer.npz`) | 85 MB | yes (R18; its l ≤ 12 rows also as `02/inputs/C59_transfer_l12.npz`) | `02_ensemble/data/high_l_transfer.py` with CAMB 2.0.4 (settings in `data/SOURCES.json`) |
| snapshot covariance cache (`ai/snapshot/P5_CMB/cache`) | 3.3 GB, 21 files | no | Codex snapshot P5 (`high_cov.py`); needed only for the S1 regression of `02 --full` |
| C145 covariance cache (`ai/cmb_ensemble/run01/cache`) | 15.5 GB, 163 files | no | `02 --full` (`run_ens.py`, `run_ens.py --refine`) |
| C145 reduced run (`--reduced`, l ≤ 12, x_max = 60) | 6.9 MB, 77 files | regenerated in 27 s | `02 --reduced` |

## How to run
```
python cmb/01_fisher_kl/run.py --quick
python cmb/02_ensemble/run.py --quick        # all paper numbers of the ensemble from the archived KL tables
python cmb/02_ensemble/run.py --reduced      # covariance -> KL -> profile -> analysis path from scratch (l <= 12)
python cmb/03_joint_state/run.py --quick
```

## Runtimes (Intel Core Ultra X9 378H, 16 cores, Windows 11)
| step | --quick | --reduced | --full |
|---|---|---|---|
| 01 | 11–40 s (two covariance builds 34 s) | — | `--full --c59-only`: C59 production from scratch (conv 1357 s, main 1088 s, full 121 s, exact 287 s; 5 workers under load; conv2 826 s, fine 154 s on 12 workers on 2026-10-16) + analyze 142–188 s: all 128 covariance tasks recomputed, C59_shape_fisher.json reproduced to 3e-13; C75 production not rerun |
| 02 | 161 s | 25–27 s | about 70 min on 12 workers (run once on a copy of the release, 13/13; 15.5 GB cache) |
| 03 | 45–64 s | — | 22 s (= quick) |

## Verification summary
<!-- VERIFY-SUMMARY-START -->
| result directory | quick | reduced | full |
|---|---|---|---|
| cmb/01_fisher_kl | 5/5 | - | 3/5 (2 skipped) |
| cmb/02_ensemble | 12/12 | 3/3 | 13/13 |
| cmb/03_joint_state | 6/6 | - | 6/6 |
<!-- VERIFY-SUMMARY-END -->

## Notation warnings
- `fossil_table_v2.csv`: column `xi` = sin 3θ (not the zero-mode fraction), column `s` = L₀/d_LSS, `u` = shortest side.
- Outputs record input file paths and creation times of the machine that ran them (`path`, `source_*`,
  `generated_utc`); these fields are not compared.
