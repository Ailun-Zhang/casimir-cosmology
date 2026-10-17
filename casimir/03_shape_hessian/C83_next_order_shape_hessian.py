# PROVENANCE: 04_calculations/code/20261011_claude_followup/C83_next_order_shape_hessian.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Shape Hessian of the next-order zero-mode terms (holonomy matching energy and Higgs zero-mode mass) and the next-order stiffnesses kappa_E = 18.04, kappa_T2 = -8.34 at xi = 0.2826 (D59 Sec. 1); LO control kappa_E = 19.324. Reads <work>/fu/C78, fu/C81 (casimir/02). Paper: PRD Sec. IV C, App. app:zeromodes, App. app:shapelaplacian, Sec. VIII (next-order floor). Output <work>/fu/C83/C83_results.json.

C83: shape Hessian of the next-order zero-mode terms and its effect on the fossil (R10 METHODOLOGY-1, DA-1, DOM-1).

At O(g^{4/3}) the zero-mode sector adds energies that depend on the shape through twisted Epstein sums at s = 1:
- the matching (holonomy-curvature) energy E_m(G) = (1/6 pi^2 L) sum_groups <c^2>_G sum_f d_f s_f t_f Z_G(1;theta_f)
  (App. D/F; the local ground state is an O(3) singlet in x = S^{-1}c, so <c_i c_j> = v G_ij with v shape independent);
- the induced mass of the Higgs zero mode, L E_H(G) = <Phi^dag Phi> [ (9/4 g2^2 + 3/4 gY^2 + 6 lam) Z_G(1;0)
  - 6 yt^2 Z_G(1;AAA) ] / (4 pi^2)  (C81; the tadpoles T_B, T_F are Z_G(1;theta)/(4 pi^2 L^2) for any shape).
Their Laplacian is -(4/3) times their value (App. D), but their anisotropic Hessian at the cube also shifts the shape
stiffnesses: lambda_A -> lambda_A - pi^2 Hess_A(L E), with lambda_A the Hessian eigenvalues of Z_eff = -pi^2 L E_C.

This script
1. evaluates Z_G(1;theta) for a general unimodular G by an own Ewald split (t = 1) and its Hessians at the cube along
   the aspect (E) and angle (T2) geodesics G(t) = exp(2 t X_A) by Richardson finite differences; checks the trace
   against -(4/3) Z and, as a control, the s = 2 Hessians against Table III (lambda_E = 1902.24, lambda_T2 = -893.20);
2. builds Hess(L E_m) and Hess(L E_H) with the C78/C81 spreads (Gaussian; SU(3) virial variant);
3. computes kappa_E, kappa_T2 and sigma_E (leading background, natural state; C82 transfer) for:
   LO; energy shift only; Hessian only; both; and reports the plateau side difference and the typical-fossil circle
   bound / window edge (shortest-cycle upper edge interpolated between the C75 values 1.0168 at sigma_E = 0.0589 and
   1.0213 at 0.1041).

    python -B C83_next_order_shape_hessian.py
Output: results/20261011_claude_followup/C83/C83_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import erfc

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C83")
C78 = json.loads(WP.wfile("fu/C78/C78_results.json").read_text())
C81 = json.loads(WP.wfile("fu/C81/C81_results.json").read_text())

S2, S6 = math.sqrt(2), math.sqrt(6)
XE = np.diag([1.0, -1.0, 0.0]) / S2
XT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float) / S2
TH0, THA = np.zeros(3), np.array([0.5, 0.5, 0.5])
NMAX = 6
GRID = np.array(list(product(range(-NMAX, NMAX + 1), repeat=3)), float)
Q_ZM, C_FREE = 0.299897 * 42.7400221506, 42.7400221506
Z2CUBE, NC = 16.5323159598, 25.515
LAM_E, LAM_T = 1902.24, -893.20
C32 = 3 / 2 ** 1.5
KE0 = 13.528581
DELTA0 = 1.5897991e-4
PLATEAU = 0.0587 / 0.058891


def zeta_s(G, th, s):
    """Ewald split (t = 1) of sum'_n cos(2 pi th.n) (n^T G n)^{-s}, det G = 1, for s = 1 or 2."""
    Gi = np.linalg.inv(G)
    Qn = np.einsum("ni,ij,nj->n", GRID, G, GRID)
    ph = np.cos(2 * math.pi * GRID @ th)
    m = GRID - th
    Pm = np.einsum("ni,ij,nj->n", m, Gi, m)
    nz, mz = Qn > 0, Pm > 1e-14
    x, y = math.pi * Qn[nz], math.pi * Pm[mz]
    if s == 1:
        direct = np.sum(ph[nz] * np.exp(-x) / x)
        recip = np.sum(np.sqrt(math.pi) * erfc(np.sqrt(y)) / np.sqrt(y))
        const = (-2.0 if not mz.all() else 0.0) - 1.0       # 1/(s-3/2) from the m = 0 term if present, -1/s
        return math.pi * (direct + recip + const)
    # s = 2: Gamma(2, x) = (1 + x) e^{-x}; Gamma(-1/2, y) = 2 e^{-y}/sqrt(y) - 2 sqrt(pi) erfc(sqrt(y))
    direct = np.sum(ph[nz] * (1 + x) * np.exp(-x) / x ** 2)
    recip = np.sum(y ** 0.5 * (2 * np.exp(-y) / np.sqrt(y) - 2 * math.sqrt(math.pi) * erfc(np.sqrt(y))))
    const = (2.0 if not mz.all() else 0.0) - 0.5            # 1/(s-3/2) = 2 if the m = 0 term is present; -1/s
    return math.pi ** 2 * (direct + recip + const)


def hess(f, X, h=2e-3):
    """Second derivative along G(t) = exp(2 t X), Richardson from steps h and h/2."""
    def d2(hh):
        return (f(expm(2 * hh * X)) - 2 * f(np.eye(3)) + f(expm(-2 * hh * X))) / hh ** 2
    return (4 * d2(h / 2) - d2(h)) / 3


def M_from(kappa, t1=12.0):
    def f(t, y):
        l3 = math.cosh(2 * t) ** 1.5
        s2 = 1.0 / math.cosh(2 * t) ** 2
        return [C32 * y[1] / l3, (kappa / C32) * l3 * s2 * y[0], C32 * y[3] / l3, (kappa / C32) * l3 * s2 * y[2]]
    y = solve_ivp(f, (0.0, t1), [1, 0, 0, 1], method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    return y[0], y[2]


def sigma(kappa, delta):
    m11, m12 = M_from(kappa)
    s_star = abs(m12 / m11) * delta
    s_nat = 3 / (2 * math.sqrt(2)) * delta / math.sqrt(abs(kappa))
    x = s_nat / s_star
    return math.sqrt(abs(2 * m11 * m12) * delta * (x + 1 / x) / 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    # 1. Hessians at the cube
    H = {}
    for name, th in (("P", TH0), ("AAA", THA)):
        z1 = zeta_s(np.eye(3), th, 1)
        hE = hess(lambda G: zeta_s(G, th, 1), XE)
        hT = hess(lambda G: zeta_s(G, th, 1), XT)
        H[name] = dict(Z1=z1, HessE=hE, HessT2=hT, trace_over_Z=(2 * hE + 3 * hT) / z1)
    zeff = lambda G: 15 * zeta_s(G, TH0, 2) - 45 * zeta_s(G, THA, 2)
    res["control_s2"] = dict(lambda_E=hess(zeff, XE), lambda_T2=hess(zeff, XT), Zeff_cube=zeff(np.eye(3)),
                             table=dict(lambda_E=LAM_E, lambda_T2=LAM_T))
    res["hessians_s1"] = H
    print(json.dumps(res, indent=1), flush=True)
    # 2. next-order terms: coefficient vectors over (Z(1;0), Z(1;AAA)) per group, times spreads
    pref = 1 / (6 * math.pi ** 2)
    groups = {"SU3": (3.0, -6.0), "SU2": (2.5, -6.0), "U1Y": (0.5, -10.0)}
    sp = {"gauss": {"SU3": C78["c2_SU3"]["gauss"], "SU2": C78["c2_SU2"]["coupled_gauss"], "U1Y": C78["gaussian_EW"]["u2"]},
          "virial_SU3": {"SU3": C78["c2_SU3"]["virial"], "SU2": C78["c2_SU2"]["coupled_gauss"], "U1Y": C78["gaussian_EW"]["u2"]}}
    run = C81["runs"]["landau_3pol"]
    g2, gY, lam, yt, z = 0.526764, 0.44146, 0.01, C81["yt"], C81["z"]
    hig = ((9 / 4) * g2 ** 2 + (3 / 4) * gY ** 2 + 6 * lam, -6 * yt ** 2)
    out = {}
    for label, spreads in sp.items():
        E0 = sum(pref * spreads[g] * (a * H["P"]["Z1"] + b * H["AAA"]["Z1"]) for g, (a, b) in groups.items())
        hE = sum(pref * spreads[g] * (a * H["P"]["HessE"] + b * H["AAA"]["HessE"]) for g, (a, b) in groups.items())
        hT = sum(pref * spreads[g] * (a * H["P"]["HessT2"] + b * H["AAA"]["HessT2"]) for g, (a, b) in groups.items())
        EH = (hig[0] * H["P"]["Z1"] + hig[1] * H["AAA"]["Z1"]) / (4 * math.pi ** 2 * z)
        hHE = (hig[0] * H["P"]["HessE"] + hig[1] * H["AAA"]["HessE"]) / (4 * math.pi ** 2 * z)
        hHT = (hig[0] * H["P"]["HessT2"] + hig[1] * H["AAA"]["HessT2"]) / (4 * math.pi ** 2 * z)
        zq0 = math.pi ** 2 * Q_ZM
        zeff0 = NC * Z2CUBE - zq0
        cases = {}
        for case, dQ, dHE, dHT in (("LO", 0.0, 0.0, 0.0),
                                    ("energy_only", E0 + EH, 0.0, 0.0),
                                    ("hessian_only", 0.0, hE + hHE, hT + hHT),
                                    ("both", E0 + EH, hE + hHE, hT + hHT),
                                    ("both_without_Higgs", E0, hE, hT)):
            zeff = zeff0 - math.pi ** 2 * dQ
            kE = 3 * (LAM_E - math.pi ** 2 * dHE) / zeff
            kT = 3 * (LAM_T - math.pi ** 2 * dHT) / zeff
            xi = (Q_ZM + dQ) / C_FREE
            delta = DELTA0 * (NC * Z2CUBE / zeff) ** 0.75
            sE = sigma(kE, delta)
            sE_pl = sE * PLATEAU
            side = math.exp(1.5878 * sE_pl) - 1
            l_up = 1.0168 + (1.0213 - 1.0168) * (sE - 0.0589) / (0.1041 - 0.0589)
            cases[case] = dict(xi=xi, kappa_E=kE, kappa_T2=kT, delta=delta, sigma_E_leading=sE, sigma_E_plateau=sE_pl,
                               typical_side_difference=side, circle_bound=0.985 * math.exp(sE),
                               window_upper_edge=l_up * math.exp(sE), shortest_cycle_edge=l_up)
        lo = cases["LO"]["sigma_E_leading"]
        for v in cases.values():
            v["sigma_ratio_to_LO"] = v["sigma_E_leading"] / lo
        out[label] = dict(E_m=E0, HessE_m=hE, HessT2_m=hT, E_H=EH, HessE_H=hHE, HessT2_H=hHT, cases=cases)
    res["next_order"] = out
    (OUT / "C83_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
