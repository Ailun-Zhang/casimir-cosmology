# PROVENANCE: 04_calculations/code/20261009_claude_R5/C58_turning_point_states.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Shape states prepared at the turning point and the quantum floor (D49): transfer M, time symmetry T = M P M^-1 P, gain, floor |T12| delta, natural state. Paper: PRD Sec. VIII A (eq:s-Tsym, eq:s-T12, eq:s-floor, eq:s-family).

C58: shape states prepared at the turning point and the quantum floor of shape memory (D49).

Leading constant-vacuum Casimir bounce, units h = L_b = 1:
  beta'' + 3 tanh(2t) beta' - kappa sech^2(2t) beta = 0,
  Pi = P/p0 with p0 = 3 M_P^2 h L_b^3 / 2^{3/2}, so beta' = (3/2^{3/2}) Pi / cosh^{3/2}(2t),
  [beta, Pi] = 2 i delta, delta = 1/(2 p0).
M = transfer (beta_0, Pi_0) at t = 0 -> (beta_+, Pi_+) at t = +infinity.
Time symmetry gives the full transfer T = M P M^{-1} P, P = diag(1,-1), so that
T11 = T22 = 1 + 2 m12 m21 and T12 = 2 m11 m12.
A real (uncorrelated) Gaussian at the turning point with Var beta_0 = s has
Var beta_+ = m11^2 s + m12^2 delta^2 / s >= 2 |m11 m12| delta = |T12| delta, the all-time optimum (E1).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from mpmath import mp, hyp2f1, gamma as mgamma, sqrt as msqrt, cos as mcos, cosh as mcosh, pi as mpi

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C58_turning_point_states')
OUT.mkdir(parents=True, exist_ok=True)
C32 = 3.0 / 2 ** 1.5


def rhs(t, y, kappa):
    b, pi_ = y
    c = np.cosh(2 * t)
    db = C32 * pi_ / c ** 1.5
    # P' = M_P^2 d(L^3 beta')/dt = -(dV/dbeta): L^3 beta'' + 3 L^2 L' beta' = kappa sech^2 L^3 beta
    # => d(L^3 beta')/dt = kappa sech^2(2t) L^3 beta ; with L^3 = cosh^{3/2}:  Pi' = (1/C32) kappa sech^2 cosh^{3/2} beta
    dpi = kappa * c ** (-2) * c ** 1.5 * b / C32
    return [db, dpi]


def transfer_from(t0, t1, kappa):
    cols = []
    for y0 in ([1.0, 0.0], [0.0, 1.0]):
        s = solve_ivp(rhs, (t0, t1), y0, args=(kappa,), method='DOP853', rtol=1e-12, atol=1e-14)
        cols.append(s.y[:, -1])
    return np.array(cols).T


def gain_closed(kappa):
    mp.dps = 30
    d = 1 - 4 * kappa
    if d >= 0:
        return float(msqrt(2) * mcos(mpi / 4 * msqrt(d)))
    return float(msqrt(2) * mcosh(mpi / 4 * msqrt(-d)))


def analyze(kappa, delta=None, label=''):
    tf = 14.0
    M = transfer_from(0.0, tf, kappa)
    Tm = transfer_from(-tf, tf, kappa)
    P = np.diag([1.0, -1.0])
    Tsym = M @ P @ np.linalg.inv(M) @ P
    m11, m12 = M[0]
    out = {'label': label, 'kappa': kappa, 'M_bounce_to_out': M.tolist(), 'T_full': Tm.tolist(),
           'T_from_M_time_symmetry': Tsym.tolist(), 'max_diff_T_vs_symmetry': float(np.max(np.abs(Tm - Tsym))),
           'gain_closed_form': gain_closed(kappa), 'gain_numeric': float(Tm[0, 0]),
           'detT': float(np.linalg.det(Tm)), 'floor_coefficient_T12': float(abs(Tm[0, 1])),
           's_star_over_delta': float(abs(m12 / m11))}
    if delta is not None:
        out['delta'] = delta
        out['floor_var_beta_plus'] = float(abs(Tm[0, 1]) * delta)
        out['floor_rms_beta_plus'] = float(np.sqrt(abs(Tm[0, 1]) * delta))
        # instantaneous ground / inverted-oscillator natural width at the turning point:
        # m = M_P^2 L_b^3, |omega| = sqrt(|kappa|) h  ->  Var beta = 1/(2 m |omega|) = (3/(2 sqrt2)) delta / sqrt|kappa|
        s_nat = (3 / (2 * np.sqrt(2))) * delta / np.sqrt(abs(kappa))
        out['s_natural_over_delta'] = float(s_nat / delta)
        out['var_beta_plus_natural'] = float(m11 ** 2 * s_nat + m12 ** 2 * delta ** 2 / s_nat)
        out['rms_beta_plus_natural'] = float(np.sqrt(out['var_beta_plus_natural']))
        out['natural_over_floor_rms'] = float(np.sqrt(out['var_beta_plus_natural'] / out['floor_var_beta_plus']))
    return out


def delta_of(Zeff_total, h_over_MP):
    """delta = sqrt2/(3 M_P^2 h L_b^3), L_b^4 = Zeff_total/(3 pi^2 M_P^2 h^2) (leading cosh model)."""
    hLb = (Zeff_total / (3 * np.pi ** 2)) ** 0.25 * np.sqrt(h_over_MP)
    return np.sqrt(2) / (3 * hLb ** 3 / h_over_MP ** 2), hLb


def main():
    Z2 = 16.532315959761668
    hMP = 6.115996604389969e-06          # instant-reheating pivot calibration (C37/D38)
    res = {}
    dM, hLbM = delta_of(100 * Z2, hMP)
    res['Maxwell_NC100'] = {'delta': dM, 'hLb': hLbM,
                            'aspect': analyze(5.997708596, dM, 'SC aspect, periodic Maxwell+graviton'),
                            'angle': analyze(-1.3318057307, dM, 'SC angle, periodic Maxwell+graviton')}
    Zaaa = 421.827111                      # C57: 15 Z2(P) - 45 Z2(AAA)
    dS, hLbS = delta_of(Zaaa, hMP)
    eig_E, eig_T = 1902.2404, -893.2029     # C57 Hessian eigenvalues of Zeff at the cube
    res['SM_AAA'] = {'delta': dS, 'hLb': hLbS, 'Zeff': Zaaa, 'N_C': Zaaa / Z2,
                     'aspect': analyze(3 * eig_E / Zaaa, dS, 'SC aspect E, SM all-antiperiodic'),
                     'angle': analyze(3 * eig_T / Zaaa, dS, 'SC angle T2, SM all-antiperiodic')}
    res['universal_8_5'] = analyze(8 / 5, None, 'universal mode kappa=8/5')
    for key in ['Maxwell_NC100', 'SM_AAA']:
        r = res[key]
        tot = 2 * r['aspect']['floor_var_beta_plus'] + 3 * r['angle']['floor_var_beta_plus']
        totn = 2 * r['aspect']['var_beta_plus_natural'] + 3 * r['angle']['var_beta_plus_natural']
        r['total_floor_rms_five_modes'] = float(np.sqrt(tot))
        r['total_natural_rms_five_modes'] = float(np.sqrt(totn))
        print(key, 'delta', r['delta'], 'hLb', r['hLb'])
        for m in ['aspect', 'angle']:
            a = r[m]
            print(f"  {m:6s} kappa={a['kappa']:.6f} gain={a['gain_numeric']:.6f} (closed {a['gain_closed_form']:.6f}) "
                  f"T12={a['floor_coefficient_T12']:.6f} floor rms={a['floor_rms_beta_plus']:.6f} "
                  f"natural rms={a['rms_beta_plus_natural']:.6f} (x{a['natural_over_floor_rms']:.3f}) "
                  f"sym-check {a['max_diff_T_vs_symmetry']:.2e} det {a['detT']:.12f}")
        print('  five-mode floor rms', r['total_floor_rms_five_modes'], 'natural', r['total_natural_rms_five_modes'])
    print('universal 8/5:', res['universal_8_5']['gain_numeric'], res['universal_8_5']['floor_coefficient_T12'])
    (OUT / 'C58_results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
