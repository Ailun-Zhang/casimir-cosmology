# PROVENANCE: 04_calculations/code/20261009_claude_R5/C63_rolling_fossil.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Linear shape transfer and fossil on the A_s-calibrated rolling plateau history (free fields: sigma_E plateau 0.0587). Paper: PRD Sec. VIII D (tab:s-fossil).

C63: linear shape transfer and quantum fossil on the actual (A_s-calibrated) rolling plateau history.

Background from C62 (instant reheating, N_* = 55.72234537): phi_b, A = Vhat(phi_b), h/M_P = 6.1159966044e-6.
Units t -> h t, L_b = 1. Linear shape mode about the cube (kappa = 3 lambda / Z_eff):
  beta' = C32 Pi / L^3,     Pi' = (kappa A / 3) / C32 * beta / L,     C32 = 3/2^{3/2},
  (cosh limit A = 3 reproduces C58). [beta, Pi] = 2 i delta, delta = sqrt2/(3 M_P^2 h L_b^3).
Time symmetry of the central history: T = M P M^{-1} P with M = transfer from the turning point to the end
of inflation (shape frozen long before). Natural width at the turning point: omega^2 = -kappa (A/3) h^2,
Var beta_0 = (3/(2 sqrt2)) delta / sqrt(|kappa| A/3).
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
C62 = json.loads(WP.wfile('r5/C62_SM_pivot/C62_results.json').read_text())
OUT = WP.wdir('r5/C63_rolling_fossil')
OUT.mkdir(parents=True, exist_ok=True)
S23 = np.sqrt(2 / 3); C32 = 3 / 2 ** 1.5


def Vp(p):
    e = np.exp(-S23 * p); return 6 * S23 * e * (1 - e)


def run(label, kappas):
    cal = C62[label]
    phib, A, delta = cal['phi_b'], cal['A'], cal['delta']

    def rhs(t, y, kappa):
        n, H, p, pd, b, pi_ = y
        L = np.exp(n)
        return [H, -0.5 * pd * pd + (2 / 3) * A * np.exp(-4 * n), pd, -3 * H * pd - Vp(p),
                C32 * pi_ / L ** 3, (kappa * A / 3) / C32 * b / L]

    def end(t, y, kappa=None):
        return 0.5 * y[3] ** 2 - y[1] ** 2 if y[1] > 0.05 else -1.0
    end.terminal = True; end.direction = 1
    out = {'phi_b': phib, 'A': A, 'delta': delta, 'hLb': cal['hLb'], 'modes': {}}
    for name, kappa in kappas:
        cols = []
        for b0, p0 in ((1.0, 0.0), (0.0, 1.0)):
            s = solve_ivp(rhs, (0, 300), [0, 0, phib, 0, b0, p0], args=(kappa,), method='DOP853', rtol=1e-12,
                          atol=1e-14, events=[end])
            cols.append(s.y[4:, -1])
            tend = s.t[-1]
        M = np.array(cols).T
        P = np.diag([1.0, -1.0])
        T = M @ P @ np.linalg.inv(M) @ P
        m11, m12 = M[0]
        floor = abs(T[0, 1]) * delta
        s_nat = (3 / (2 * np.sqrt(2))) * delta / np.sqrt(abs(kappa) * A / 3)
        var_nat = m11 ** 2 * s_nat + m12 ** 2 * delta ** 2 / s_nat
        out['modes'][name] = {'kappa': kappa, 'M': M.tolist(), 'T': T.tolist(), 'det_M': float(np.linalg.det(M)),
                              'gain_T11': float(T[0, 0]), 'T12': float(T[0, 1]), 'floor_rms': float(np.sqrt(floor)),
                              's_star_over_delta': float(abs(m12 / m11)), 's_nat_over_delta': float(s_nat / delta),
                              'natural_rms': float(np.sqrt(var_nat)), 'natural_over_floor_var': float(var_nat / floor),
                              't_end_inflation': float(tend)}
        print(label, name, f"kappa={kappa:.6f} gain={T[0,0]:.5f} T12={T[0,1]:.5f} detM={np.linalg.det(M):.12f} "
              f"floor rms={np.sqrt(floor):.6f} natural rms={np.sqrt(var_nat):.6f} (var ratio {var_nat/floor:.5f})")
    tot_nat = 2 * out['modes']['aspect']['natural_rms'] ** 2 + 3 * out['modes']['angle']['natural_rms'] ** 2
    out['five_mode_natural_rms'] = float(np.sqrt(tot_nat))
    print(label, 'five-mode natural rms', np.sqrt(tot_nat))
    return out


def main():
    res = {'NC100_toy': run('NC100_check', [('aspect', 5.997708596), ('angle', -1.3318057307)]),
           'SM_AAA': run('SM_AAA', [('aspect', 13.528578536527492), ('angle', -6.352386155663664)])}
    (OUT / 'C63_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
