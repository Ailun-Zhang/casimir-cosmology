# PROVENANCE: 04_calculations/code/20261011_claude_followup/C78_next_order_zero_mode.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] First-order O(g^{4/3}) holonomy-curvature correction to the zero-mode energy (van Baal / Luscher kappa_1 term): dQ and dr for the SM content (D58 Sec. 4). Paper: PRD Sec. IV C (next order: xi -> 0.28), App. app:zeromodes. Output <work>/fu/C78/C78_results.json.

C78: first-order estimate of the O(g^{4/3}) correction to the SM zero-mode energy Q (follow-up to C72/D54).

At next order the constant modes feel the curvature of the one-loop holonomy potential of the nonzero modes
(van Baal 1988, Eq. (38); Luscher's kappa_1 term). In coupling-absorbed variables (D = d - i c.T - i Y u) a field
with weight d_f, statistics s_f (+1 boson, -1 fermion), twist w_f(n) and generator trace tr_R(T^a T^b) = t_R delta^ab
contributes, from the image sum E_C = -(1/pi^2 L) sum' w(n) cos(c.n ...)/|n|^4,
    L d^2E/dc^a_i dc^b_j |_0 = delta^ab delta_ij (1/(3 pi^2)) sum_f d_f s_f t_R Z_wf(1),
with Z(1) = sum' |n|^-2 (periodic, analytically continued = zero-mode cusp removed) and Z_AAA(1) = sum' (-1)^{n1+n2+n3}|n|^-2.
Check: SU(N) gauge + n_f Dirac fundamentals reproduces van Baal's (1/4)(N M^(0) - n_f M^(1/2)) c^2 with M = 2 Z/(3 pi^2).

SM content per group (AAA fermions, periodic bosons; Higgs zero mode explicit, so its nonzero modes use Z(1)):
    SU(3): gluons (t = 3), 12 Weyl triplets (= 6 Dirac, t = 1/2 each Weyl -> sum d t = 6).
    SU(2): W (t = 2), 12 Weyl doublets (sum d t = 6), Higgs doublet (complex, d = 1, t = 1/2).
    U(1)_Y: Weyl fermions sum Y^2 = 10, Higgs doublet sum Y^2 = 1/2 (two complex components, Y = 1/2).
First-order energy shift: dQ = sum_G (1/2) E''_G <c^2>_G, with <c^2> from
    (i) the optimal Gaussian of the coupled SU(2) x U(1) x Higgs model (D54 formula) and the SU(3) Gaussian, and
    (ii) for SU(3) also a virial-scaled spread (exact eps_3 = 12.5868 fixes <pi^2/2> = 2E/3; alpha_eff = 4<T>/n).
Not included: the induced mass of the Higgs zero mode (depends on what stabilizes the Higgs), O(g^2) cubic terms
(vanish in the ground-state expectation by symmetry), and higher orders.

    python C78_next_order_zero_mode.py
Output: results/20261011_claude_followup/C78/C78_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C78")
C_FREE = 42.7400221506
G2, G3, GY, LAM = 0.526764, 0.539605, 0.441460, 0.01      # one-loop couplings at 1/L_b (C72), stabilized quartic
EPS2, EPS3 = 4.116720, 12.5868


def zeta1(twisted, nmax=7):
    """Ewald split (t = 1) of sum' w(n)/|n|^2 on the cubic lattice; w = 1 or (-1)^{n1+n2+n3}."""
    th = np.array([0.5, 0.5, 0.5]) if twisted else np.zeros(3)
    direct, recip = 0.0, 0.0
    for n in product(range(-nmax, nmax + 1), repeat=3):
        n = np.array(n, float)
        q = n @ n
        if q > 0:
            sgn = (-1) ** int(abs(n).sum()) if twisted else 1.0
            direct += sgn * np.exp(-np.pi * q) / (np.pi * q)
        m = n - th
        p = np.sqrt(m @ m)
        if p > 0:
            recip += erfc(np.sqrt(np.pi) * p) / p
    const = -1.0 if twisted else -3.0          # -1/s, plus 1/(s - 3/2) from the m = 0 term when untwisted
    return np.pi * (direct + recip + const)


def gauss_ew(g2, gy, lam):
    """Optimal Gaussian of the coupled SU(2) x U(1) x Higgs zero-mode model (D54 Sec. 2), psi ~ exp[-(a c^2 + b u^2 + z phi^2)/2]."""
    def E(x):
        a, b, z = np.exp(x)
        return (9 * g2 ** 2 * a / 4 + 3 * gy ** 2 * b / 4 + z + 9 / (4 * g2 ** 2 * a ** 2)
                + 9 / (8 * a * z) + 3 / (8 * b * z) + 3 * lam / (2 * z ** 2))
    best = min((minimize(E, x0, method="BFGS", options=dict(gtol=1e-12)) for x0 in
                ([1.2, 0.6, -0.3], [0.5, 0.0, 0.0], [2.0, 1.0, -1.0])), key=lambda r: r.fun)
    a, b, z = np.exp(best.x)
    return dict(E=float(best.fun), a=float(a), b=float(b), z=float(z), c2=float(9 / (2 * a)), u2=float(3 / (2 * b)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Z1, ZA = zeta1(False), zeta1(True)
    Z1b, ZAb = zeta1(False, 9), zeta1(True, 9)
    pref = 1 / (3 * np.pi ** 2)
    # curvature L E'' per component (delta^ab delta_ij), coupling-absorbed variables
    curv = {"SU3": pref * (3 * Z1 - 6 * ZA),
            "SU2": pref * (2 * Z1 - 6 * ZA + 0.5 * Z1),
            "SU2_without_Higgs": pref * (2 * Z1 - 6 * ZA),
            "U1Y": pref * (-10 * ZA + 0.5 * Z1)}
    vb = {"M0": 2 * Z1 / (3 * np.pi ** 2), "Mhalf": 2 * ZA / (3 * np.pi ** 2),
          "SU3_coeff_(3M0-6Mhalf)/4": (3 * 2 * Z1 - 6 * 2 * ZA) / (4 * 3 * np.pi ** 2)}
    ew = gauss_ew(G2, GY, LAM)
    # SU(3): Gaussian (alpha = 3^(1/3)) and virial-scaled spreads, x = c / g^(2/3)
    n3 = 24
    x2_gauss = n3 / (2 * 3 ** (1 / 3))
    x2_vir = n3 / (2 * (4 * (2 * EPS3 / 3) / n3))
    c2_su3 = {"gauss": G3 ** (4 / 3) * x2_gauss, "virial": G3 ** (4 / 3) * x2_vir}
    # SU(2): coupled-model Gaussian and pure-SU(2) Gaussian (alpha = 2^(1/3))
    c2_su2 = {"coupled_gauss": ew["c2"], "pure_gauss": G2 ** (4 / 3) * 9 / (2 * 2 ** (1 / 3))}
    shifts = {}
    for k3, c3 in c2_su3.items():
        for k2, c2 in c2_su2.items():
            d3 = 0.5 * curv["SU3"] * c3
            d2 = 0.5 * curv["SU2"] * c2
            dy = 0.5 * curv["U1Y"] * ew["u2"]
            tot = d3 + d2 + dy
            shifts["%s_%s" % (k3, k2)] = dict(dQ_SU3=d3, dQ_SU2=d2, dQ_U1Y=dy, dQ_total=tot, dr=tot / C_FREE,
                                              dr_nonabelian=(d3 + d2) / C_FREE)
    res = dict(couplings=dict(g2=G2, g3=G3, gY=GY, lam=LAM), C_free=C_FREE,
               zeta=dict(Z1=Z1, Z_AAA1=ZA, Z1_nmax9=Z1b, Z_AAA1_nmax9=ZAb,
                         L25_reference=dict(Z1=-8.913632917585, Z_AAA1=-2.519356152089)),
               vanBaal_check=vb, curvature=curv, gaussian_EW=ew, c2_SU3=c2_su3, c2_SU2=c2_su2, shifts=shifts,
               leading_order_r=0.299897,
               leading_plus_first_order_r={k: 0.299897 + v["dr"] for k, v in shifts.items()})
    (OUT / "C78_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
