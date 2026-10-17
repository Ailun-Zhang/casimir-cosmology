# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_cross.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D110 round-2 indicative zero-mode cross-term estimate (superseded by D121, 04_cross_terms); kept for the history table.

C138 cross-term estimate (D110 Sec. 3.4): first-order zero-mode/nonzero-mode cross term for the Higgs
fluctuations,  E_X(r) = (1/2) sum_i kappa_i <M_ii>_Gauss(r),  kappa_i = sum'_k [1/(2w) - k_theta^2/(2w^3)] ~ (2/3)<phi_i^2>_FV
(isotropic twist average, light-mode limit), M = sum_j (g x_j.T + gY Y w_j)^2 from the conditional Gaussian:
    M_vev = M_h = M_pi3 = (3/8)[2g^2/a + (g^2 b - 2 g gY c + gY^2 k)/D],  M_pi+- = same with +2 g gY c.
Static gamma(lambda) for MC + E_X (and the seagull-only bound, factor 3/2) -> shift of lambda_c."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C138_core as C  # noqa: E402
from C138_core import R, R14_rge  # noqa: E402
from C138_run import wjson, LAMS  # noqa: E402


def cross_term(gp_full, r, lam, g2, gY, factor=2.0 / 3):
    rf, P = gp_full["rf"], gp_full["P"]
    a, k, b, c = (np.interp(r, rf, P[i]) for i in range(4))
    D = k * b - c * c
    Mv = 3.0 / 8 * (2 * g2 ** 2 / a + (g2 ** 2 * b - 2 * g2 * gY * c + gY ** 2 * k) / D)
    Mp = 3.0 / 8 * (2 * g2 ** 2 / a + (g2 ** 2 * b + 2 * g2 * gY * c + gY ** 2 * k) / D)
    m = C.masses(r, lam, g2, gY, 0.0)
    eps = 1e-6
    h2 = (C.estd(m["mh2"] + eps) - C.estd(m["mh2"] - eps)) / eps           # <h^2> = 2 dE/dm^2
    pW = 2 * C.coarse(lambda x: C.ddelta_db0(g2 * g2 / 4 * x * x), r)
    pZ = 2 * C.coarse(lambda x: C.ddelta_db0((g2 * g2 + gY * gY) / 4 * x * x), r)
    return 0.5 * factor * (Mv * (h2 + pZ) + Mp * 2 * pW), Mv, Mp


def main():
    cal, _ = R14_rge.calibration()
    sm = R14_rge.couplings(172.6, 0.1180, cal=cal)
    sets = {"v11": dict(C.V11), "SM": dict(g2=sm["g2"], gY=sm["gY"], yt=sm["yt"])}
    r, m = R.grid(16.0)
    out = {}
    for tag, cp in sets.items():
        gp = R.gauss_profile(cp["g2"], cp["gY"])
        s, al = C.spar_xi(gp)
        cols, keys = [], []
        # quadratic holonomy expansion fails at small r (U(1)_Y zero-mode width k/D ~ 1/r diverges): freeze E_X
        # below r_s (constant shift there) -- estimate only.
        for fac, rs, name in ((0.0, 0, "MC"), (2.0 / 3, 1.0, "MCX_rs1.0"), (2.0 / 3, 1.5, "MCX_rs1.5"),
                              (1.0, 1.0, "MCXseagull_rs1.0")):
            for lam in LAMS:
                U = C.U_col(gp, "MC", lam, cp["g2"], cp["gY"], cp["yt"], r)
                if fac > 0:
                    U = U + cross_term(gp, np.maximum(r, rs), lam, cp["g2"], cp["gY"], fac)[0]
                cols.append(U)
                keys.append((name, lam))
        nl = len(LAMS)
        E0, it, rs = R.solve_set(cols[:nl], r, m, np.full(nl, s), np.full(nl, al))      # MC (default guesses)
        E = [E0]
        for jm in range(1, len(cols) // nl):                                           # MCX: guess shifted by E_X(1.7)
            cj = cols[jm * nl:(jm + 1) * nl]
            gs = np.array([E0[i] + (cj[i] - cols[i])[int(round((1.7 - r[0]) / R.H_N))] for i in range(nl)])
            Ej, _, _ = R.solve_set(cj, r, m, np.full(nl, s), np.full(nl, al), guess=gs)
            E.append(Ej)
        E = np.concatenate(E)
        tab = {}
        for (name, lam), e in zip(keys, E):
            tab.setdefault(name, {})[str(lam)] = dict(Q=float(e.real), gamma=float(-2 * e.imag))
        lc = {}
        for name in tab:
            ls = np.array(sorted(LAMS))
            sp = CubicSpline(ls, [math.log(tab[name][str(x)]["gamma"]) for x in ls])
            lc[name] = {}
            for kk, v in (("P1=0.9", -math.log(0.9)), ("P1=0.5", math.log(2))):
                try:
                    lc[name][kk] = float(brentq(lambda x: sp(x) - math.log(v / R.ETA_TOT), ls[0], ls[-1]))
                except ValueError:
                    lc[name][kk] = None
            print(tag, name, [(x, round(tab[name][str(x)]["Q"], 5), "%.3e" % tab[name][str(x)]["gamma"])
                              for x in (-0.004, -0.010, -0.020)], flush=True)
        rr = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        EX, Mv, Mp = cross_term(gp, rr, -0.010, cp["g2"], cp["gY"])
        out[tag] = dict(couplings=cp, table=tab, lambda_c=lc,
                        profile_r=dict(r=rr, E_X=EX, M_vev=Mv, M_pm=Mp), muller_iter=int(it))
        print(tag, lc, "E_X(r)=", np.round(EX, 5).tolist(), flush=True)
    wjson("C138_cross.json", out)


if __name__ == "__main__":
    main()
