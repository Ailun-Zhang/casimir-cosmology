# V7 — blind recalculation of the abelian Higgs zero-mode resonance (report v1.1, Sec. 19.13)

Level: **blind** (report Sec. 19.1). The check received the specification below and general literature on methods; it had
no access to the code, results or text of the main calculation (report Sec. 18.9; `prl_supplement/c171`,
`prl_supplement/results/C171`), to the Letter or to the report, and it recorded its results before any comparison.
Internal paths and workflow instructions of the original brief are removed here.

## Problem
Abelian Higgs model on a spatial three-torus of edge L, units 1/L. Constant modes only: a complex scalar zero mode with
two real canonical coordinates (phi1, phi2) and three constant gauge-field modes w = (w1, w2, w3):

    H = 1/2 (p1^2 + p2^2) + 1/2 |p_w|^2 + 1/2 e^2 (phi1^2 + phi2^2) |w|^2 + (lambda/4) (phi1^2 + phi2^2)^2,

charge e > 0, lambda < 0. Gauss's law for constant gauge transformations: zero U(1) charge (zero angular momentum in the
(phi1, phi2) plane); s-wave in w. With r = (phi1^2 + phi2^2)^(1/2), s = |w|:

    H = -1/2 (d_r^2 + r^-1 d_r) - 1/2 (d_s^2 + 2 s^-1 d_s) + 1/2 e^2 r^2 s^2 + (lambda/4) r^4,  measure r dr s^2 ds,

regular at r = 0 and s = 0. The lowest state is a resonance E = eps - i Gamma/2 (outgoing at large r).

## Required
1. eps and Gamma for e = 1 and lambda = -1, -0.11174, -0.03, -0.01003, to 1e-4 relative in Gamma and 1e-6 in eps, with a
   demonstration of the accuracy (refinement, extrapolation, variation of the complex-scaling or absorbing-potential
   parameters and of the box).
2. The scaling law: eps and Gamma for e = 0.873, lambda = -0.03 x 0.873^2, against e^(2/3) times the e = 1 values.
3. Optional: further lambda between -0.11174 and -0.01003 and the local slope -d ln Gamma / d|lambda|^(-1/2).
Two genuinely different numerical approaches for at least two lambda.

## Files
- `code/`: the check's code (`python code/run_all.py --fresh` reruns everything, about 25 min; a rerun writes its JSON
  files next to `code/`, not into `results/`).
- `results/`: `blind_results.json` (final values), `methodA_results.json`, `methodB_results.json`,
  `slope_results.json`, `METHOD.md` (formulation, methods, convergence tables), `convergence_tables.md`.
- `post_comparison/`: comparison with the main calculation, compiled after the blind results were recorded
  (`COMPARISON.md`); the two couplings given rounded to five digits were re-evaluated with the blind solver at the exact
  grid couplings of the main calculation (`blind_at_c171_exact.py`). Result: widths agree to 1e-10, energies to 1e-11.
