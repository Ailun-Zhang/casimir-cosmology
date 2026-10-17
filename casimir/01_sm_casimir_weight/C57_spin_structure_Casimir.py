# PROVENANCE: 04_calculations/code/20261009_claude_R5/C57_spin_structure_Casimir.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Signed Casimir weight N_C of the Standard Model on the cubic T^3 for the eight fermion spin structures (twisted Ewald sums, s = 2), cube Hessian of Z_eff on the five shape modes, trace identity tr Hess Z_eff = (8/3) Z_eff, normalised stiffnesses kappa_A = 3 lambda_A / Z_eff and zero-momentum gains. Units: Z_2 dimensionless, E_C = -Z_eff/(pi^2 L). Paper: PRD Sec. IV A-B (sec:pa-image, sec:pa-spin, Eq. eq:pa-NC), App. app:smcasimir Table tab:pa-spin, Fig. fig:pa-casimir-shape(a). Output <work>/r5/C57_spin_structure/C57_spin_structures.json. Derivation D50 Secs. 1-3.

C57: signed Casimir weight and cubic shape Hessian for Standard-Model content on T^3
with the eight fermion spin structures (round R5).

Image form (free massless fields, flat twist theta):
    E_C = -(1/(pi^2 L)) sum_{n != 0} w_n (n^T G n)^{-2},
    w_n = sum_fields d_f s_f cos(2 pi theta_f . n),  s = +1 bosons, -1 fermions,
with d = 1 (Maxwell, Weyl, graviton static weight), 1/2 (real conformal scalar).
Twisted Epstein sums Z_2(G,theta) = sum' cos(2 pi theta.n) (n^T G n)^-2 by an
Ewald split (independent implementation, s = 2):
  Z = sum'_n c_n (1+pi Q) e^{-pi Q}/Q^2
      + pi^2 detG^{-1/2} [ sum_m (pi P_m)^{1/2} Gamma(-1/2, pi P_m) + 2 delta_{theta in Z^3} ]
      - pi^2/2,      P_m = (m-theta)^T G^{-1} (m-theta), m = 0 excluded only if P = 0.
Checks: periodic cube Z_2 = 16.5323159598; direct summation for the alternating sum;
trace identity tr Hess Z_eff = (8/3) Z_eff for any weights.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from scipy.special import erfc
from scipy.linalg import expm

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C57_spin_structure')
OUT.mkdir(parents=True, exist_ok=True)
R_EW = 7
_grid = np.array(list(itertools.product(range(-R_EW, R_EW + 1), repeat=3)), dtype=float)
_nz = _grid[np.any(_grid != 0, axis=1)]


def gamma_m12(x):
    """Upper incomplete Gamma(-1/2, x) = 2 (x^{-1/2} e^{-x} - sqrt(pi) erfc(sqrt x))."""
    return 2.0 * (np.exp(-x) / np.sqrt(x) - np.sqrt(np.pi) * erfc(np.sqrt(x)))


def Z2(G, theta):
    theta = np.asarray(theta, float)
    Q = np.einsum('ni,ij,nj->n', _nz, G, _nz)
    c = np.cos(2 * np.pi * _nz @ theta)
    direct = np.sum(c * (1 + np.pi * Q) * np.exp(-np.pi * Q) / Q ** 2)
    Gi = np.linalg.inv(G)
    mm = _grid - theta
    P = np.einsum('ni,ij,nj->n', mm, Gi, mm)
    mask = P > 1e-14
    recip = np.sum(np.sqrt(np.pi * P[mask]) * gamma_m12(np.pi * P[mask]))
    periodic = 2.0 if np.all(np.abs(theta - np.round(theta)) < 1e-12) else 0.0
    return direct + np.pi ** 2 / np.sqrt(np.linalg.det(G)) * (recip + periodic) - np.pi ** 2 / 2.0


def Z2_direct(theta, R=60):
    """Brute-force check (absolutely convergent; tail ~ 4 pi / R for theta = 0)."""
    r = np.arange(-R, R + 1)
    n1, n2, n3 = np.meshgrid(r, r, r, indexing='ij')
    Q = (n1 ** 2 + n2 ** 2 + n3 ** 2).astype(float)
    Q[R, R, R] = np.inf
    c = np.cos(2 * np.pi * (theta[0] * n1 + theta[1] * n2 + theta[2] * n3))
    return float(np.sum(c / Q ** 2))


def basis5():
    Xs = []
    Xs.append(np.diag([1, -1, 0]) / np.sqrt(2))          # E (aspect)
    Xs.append(np.diag([1, 1, -2]) / np.sqrt(6))          # E (aspect)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:               # T2 (angle)
        X = np.zeros((3, 3)); X[i, j] = X[j, i] = 1 / np.sqrt(2); Xs.append(X)
    return Xs


def Zeff(G, content):
    """content: list of (weight, theta)."""
    return sum(w * Z2(G, th) for w, th in content)


def hessian(content, h=2e-3):
    Xs = basis5()
    H = np.zeros((5, 5))
    f0 = Zeff(np.eye(3), content)
    for a in range(5):
        for b in range(a, 5):
            def f(x, y):
                return Zeff(expm(2 * (x * Xs[a] + y * Xs[b])), content)
            if a == b:
                H[a, a] = (-f(2 * h, 0) + 16 * f(h, 0) - 30 * f0 + 16 * f(-h, 0) - f(-2 * h, 0)) / (12 * h * h)
            else:
                H[a, b] = H[b, a] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    grad = np.array([(Zeff(expm(2 * h * X), content) - Zeff(expm(-2 * h * X), content)) / (2 * h) for X in Xs])
    return f0, grad, H


def gain(kappa):
    d = 1 - 4 * kappa
    if d >= 0:
        return float(np.sqrt(2) * np.cos(np.pi / 4 * np.sqrt(d)))
    return float(np.sqrt(2) * np.cosh(np.pi / 4 * np.sqrt(-d)))


def main():
    th0 = np.zeros(3)
    out = {}
    Zp = Z2(np.eye(3), th0)
    out['Z2_periodic_cube'] = Zp
    out['Z2_periodic_direct_R60_plus_tail'] = Z2_direct(th0, 60) + 4 * np.pi / 60.5
    spin = {}
    for bits in itertools.product([0, 1], repeat=3):
        th = 0.5 * np.array(bits, float)
        spin[''.join('A' if b else 'P' for b in bits)] = th
    alt = Z2(np.eye(3), spin['AAA'])
    out['Z2_AAA_cube'] = alt
    out['Z2_AAA_direct_R60'] = Z2_direct(spin['AAA'], 60)
    out['ratio_AAA_over_periodic'] = alt / Zp
    # Standard Model free high-energy content in Maxwell-equivalent weights
    B = 12 + 4 * 0.5 + 1          # 12 gauge vectors, 4 real Higgs, static graviton
    F = 45                        # Weyl fermions (3 generations, no nu_R)
    rows = {}
    for name, th in spin.items():
        content = [(B, th0), (-F, th)]
        z, g, H = hessian(content)
        ev, evec = np.linalg.eigh(H)
        NC = z / Zp
        row = {'theta': th.tolist(), 'Zeff': z, 'N_C_signed': NC, 'grad_norm': float(np.linalg.norm(g)),
               'Hess_eigs': ev.tolist(), 'trace_over_8_3_Z': float(np.trace(H) / (8 / 3 * z))}
        if z > 0:
            kap = 3 * ev / z
            row['kappa'] = kap.tolist()
            row['sum_kappa'] = float(kap.sum())
            row['gains_zero_momentum'] = [gain(k) for k in kap]
        rows[name] = row
        print(name, f'Zeff={z:.6f} N_C={NC:.4f} |grad|={np.linalg.norm(g):.2e} eigs={np.round(ev,4)} '
              f'tr/(8/3 Z)={row["trace_over_8_3_Z"]:.8f}', row.get('gains_zero_momentum'))
    out['SM_spin_structures'] = rows
    # Maxwell-only check of the archived SC numbers
    z, g, H = hessian([(1.0, th0)])
    out['Maxwell_check'] = {'Z2': z, 'eigs': np.linalg.eigvalsh(H).tolist()}
    print('Maxwell check', z, np.round(np.linalg.eigvalsh(H), 6))
    # AAA fermion alone (per Weyl): its own Hessian structure
    z, g, H = hessian([(-1.0, spin['AAA'])])
    out['AAA_Weyl_alone'] = {'Zeff': z, 'eigs': np.linalg.eigvalsh(H).tolist()}
    print('AAA Weyl alone', z, np.round(np.linalg.eigvalsh(H), 6))
    (OUT / 'C57_spin_structures.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))


if __name__ == '__main__':
    main()
