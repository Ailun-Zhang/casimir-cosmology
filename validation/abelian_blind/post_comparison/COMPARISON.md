# Blind recalculation of the abelian Higgs resonance against C171 (compiled after the blind results were recorded)

Blind results: `../results/blind_results.json` (Method B values; Methods A and B agree to 2e-9 in Gamma, 6e-11 in eps).
C171 values: `prl_supplement/results/C171/C171_results.json#table` of this package (full precision).
The specification (`../SPEC.md`) quoted two couplings rounded to five digits (-0.11174, -0.01003). After the
blind results were delivered, the blind Method-B solver was evaluated at the exact C171 grid couplings
(`blind_at_c171_exact.py`, run 2026-10-22 ~06:45Z; bases 150x70 (beta = gamma = 1) for j10 and 200x90 (0.7) for j21,
theta = pi/8, 40-digit Rayleigh quotient). `blind_at_c171_grid.py` and `.json` are the same evaluation at the couplings
rounded to 8 decimals as printed in C171_gamma_table.csv (differences there, up to 1.8e-6 in Gamma at j21, are the
effect of that rounding: d ln Gamma / d|Lambda| = 1945 at j21).

| C171 point | Lambda (exact) | eps blind | eps C171 | d eps | Gamma blind | Gamma C171 | rel. diff. |
|---|---|---|---|---|---|---|---|
| j00 | -1 | 1.6688420440548240 | 1.6688420440494924 | +5.3e-12 | 0.61228826631903297 | 0.6122882663752914 | -9.2e-11 |
| j10 | -0.11173797183976203 | 1.8506521094818 | 1.8506521094818096 | +1.5e-14 | 0.00611592740232 | 0.006115927402293394 | +4.2e-12 |
| j16 | -0.03 | 1.8907275360559994 | 1.8907275360560785 | -7.9e-14 | 4.5653523704362915e-7 | 4.5653523704322273e-7 | +8.9e-13 |
| j21 | -0.010028169057997869 | 1.8987196506819 | 1.8987196506819004 | -2.6e-14 | 5.07904289105e-14 | 5.079042891085601e-14 | -6.5e-12 |
| e = 0.873, Lambda = -0.03 | (lambda = -0.0228639) | 1.72705079778190 | e^(2/3) x C171 j16: 1.727050797782 | -7.1e-14 | 4.17013837433421e-7 | 4.170138374330e-7 | +8.9e-13 |

Scaling law within the blind results: Gamma(0.873) / [e^(2/3) Gamma(1, -0.03)] - 1 = -5e-16 (Method B).
Local slope -d ln Gamma / d|Lambda|^(-1/2) (blind, dense scan): 3.6506 at -0.03, 3.9083 at -0.01003 (C171 grid: 3.648, 3.907).
State at Lambda = 0 (blind, direct): bound, eps_0 = 1.90256784984 (C171 extrapolation eps_0 + c x^-2: 1.90264).
Conclusion: the blind values reproduce C171 to 1e-10 in Gamma and 1e-11 in eps; no correction of C171 is needed.
