# 06_primordial_spectra — n_s and r of the plateau after the turn

**Physics.** After the turn the universe inflates on the plateau V = 3M_P²h²(1 − e^{−√(2/3)φ/M_P})²; the torus modes
are deep inside the Hubble radius at the turn (2π/(hL_b) ≈ 1300), so the spectra are the standard plateau spectra:
n_s = 0.96534, r = 0.00346 for instant reheating (N_* = 55.72) and n_s = 0.96174, r = 0.00420 for the weak visible decay
(N_* = 50.33). Initial-state tests (adiabatic order 0/2, 6–10 e-folds of preparation) change n_s by < 1e-7.

**Method.** `C64_plateau_ns_r.py`: scalar (Mukhanov–Sasaki) and tensor mode equations on the plateau background,
second-order adiabatic initial states, n_s by symmetric differences. Derivations D51, D30, D35, D36. Status: numerical
(standard).

**Run.** `python run.py --quick` (= `--full`; 1–6 min).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:19:37Z | 5/5 | 0 | C64 n_s and r 207 |
| full | 2026-10-15T23:19:45Z | 5/5 | 0 | C64 n_s and r 137 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VIII A | n_s = 0.9653407, r = 0.0034624 (N_* = 55.722) (PRIM-01, 02) | `r5/C64_plateau_ns_r/C64_results.json` : `rows/6/{ns,r}` | `04_calculations/results/20261009_claude_R5/C64_plateau_ns_r/C64_results.json` |
| PRD Sec. VIII A | n_s = 0.9617386, r = 0.0042022 (N_* = 50.328) (PRIM-03, 04) | same : `rows/3/{ns,r}` | same |
| report Fig. 22 (n_s, r vs N_*) | 12 values of N_* | same : `rows[*]` | same |
| PRD Sec. VIII A | initial-state tests | same : `initial_state_tests` | same |
