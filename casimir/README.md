# casimir/ — Standard-Model Casimir weight, zero-mode energy and the shape Hessian (DNEC v12)

This chain reproduces the numbers of PRD Sec. IV (`sec:casimir`), Sec. VII (`sec:shape`) and the appendices
`app:smcasimir`, `app:zeromodes`, `app:shapelaplacian`: the signed Casimir weight of the Standard Model on a cubic
three-torus (N_C = +25.515 for all-antiperiodic fermions, −30 for periodic ones; holonomy sectors), the energy of the
gauge/Higgs zero modes (fraction ξ = Q/C_free = 0.2999, band 0.258–0.308, next order ≈ 0.28), the trace identity
Tr_sh Hess E_C = (8/3) E_C with its finite-temperature / zero-mode generalisations, the zero-momentum gain
𝒢(κ) = √2 cosh(π√(4κ−1)/4) (guaranteed ≥ 4.500465) and the stiffnesses κ_E = 19.32 (ξ = 0.30) and 18.0 (next order).

## Dependency order
```
01_sm_casimir_weight ... twisted Ewald sums: N_C of 8 spin structures, holonomy sectors, cube Hessian, kappa (C57, C68)
02_zero_mode_energy .... DMC energies of the constant-mode Hamiltonians -> xi = 0.2999; next order dQ (C72, C78, C81)
        │  (fu/C78, fu/C81 results)
03_shape_hessian ....... shape Laplacian identities, gain, no-go generalisations, next-order Hessian (C73, C83)
```
Shared path shim: `_casimir_paths.py` (work tree `output/casimir/<mode>`, mirror prefixes `r5/` = round R5 2026-10-09,
`fu/` = follow-up 2026-10-11). Code changes relative to the research tree: `PACKAGE_PATCHES.json` (paths, worker counts,
one package note at the top of each module docstring; algorithms and numerics unchanged).

## How to run
```
python casimir/01_sm_casimir_weight/run.py --quick
python casimir/02_zero_mode_energy/run.py --full --workers 15
python casimir/03_shape_hessian/run.py --full
```
Every `run.py` writes `VERIFY.json` (checks, tolerances, timings, machine).

## Runtimes (Intel Core Ultra X9 378H, 16 cores, Windows 11; this packaging run)
| step | --quick | --full | original production |
|---|---|---|---|
| 01 | 6 s | = quick | — |
| 02 | 22 s | 447–475 s (DMC 435–459 s, 15 workers) | ~10 min on 8 workers (C72_dmc docstring) |
| 03 | 17–22 s | 122–194 s (mpmath dps 40–60, 14–15 workers) | — |

## Verification summary
<!-- VERIFY-SUMMARY-START -->
| result directory | quick | reduced | full |
|---|---|---|---|
| casimir/01_sm_casimir_weight | 9/9 | - | 9/9 |
| casimir/02_zero_mode_energy | 9/9 | - | 9/9 |
| casimir/03_shape_hessian | 11/11 | - | 16/16 |
<!-- VERIFY-SUMMARY-END -->

## Notes
- All quick and full checks reproduce the archived JSON bit for bit (|diff| = 0) except the DMC rows, which are stored
  in completion order: the full check compares the per-case extrapolations (identical seeds → identical numbers).
- Fields that store the module docstring (`description`) differ textually because of the package note; they are not
  compared.
- In C72/C76/C142 code the symbol `r` (or `r_SM`) is the zero-mode fraction ξ (called f_Q or ξ in the paper), not the
  tensor-to-scalar ratio.
