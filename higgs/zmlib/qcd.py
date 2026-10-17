# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/qcd.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261014_claude_v12_r3/twod_direct_sm/qcd.py on 2026-10-14 by agent holonomy-cross (C149 run02).
# PROVENANCE: copied UNCHANGED from 04_calculations/code/20261014_claude_v12_r3/sm_mapping/qcd.py on 2026-10-14 by twod-direct.
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] QCD helpers: alpha_s running/decoupling, top pole <-> MS-bar conversion to four loops (Marquard et al. 2015, z4 = -859.96 +- 21.5), mbar(mbar).  Paper: PRD App. H5 (m_t definitions).  Units: GeV.

C148 (sm-mapping): pure-QCD pieces of the SM mapping (own implementation; RunDec was not obtainable).
  * beta_n(nf) (a_s = alpha_s/4pi; 1-5 loops: van Ritbergen-Vermaseren-Larin, Czakon, Baikov-Chetyrkin-Kuhn);
    checked against the nf=6 numbers used by SMDR betas.c (2472.28 = beta3, 271.428 = beta4) and Buttazzo (98).
  * gamma_m,n(nf) (a = alpha_s/pi, dln m/dln mu^2 = -sum g_n a^(n+1); 4-loop Chetyrkin / VLR 1997);
    gamma_4(nf=6) taken from SMDR betas.c (yt 5-loop QCD 19988.8).
  * pole <-> MSbar: z_m(mu) = m(mu)/M = sum a(mu)^k P_k(l), l = ln(mu^2/M^2), a = alpha_s^(6)(mu)/pi; the logs P_k(l)
    follow from RG (recursion below); constants z1 = -4/3, z2, z3 from the classic c2 = 13.443433-1.041367 nl,
    c3 = 190.595-26.655 nl+0.6527 nl^2 (M = m(m) sum c_n a(m)^n), z4(nl=5) = -859.96 +- 21.5 (Marquard et al.
    1502.01030 Eq. (11)).
  * top decoupling alpha_s^(5) = alpha_s^(6) Z(aS, ln m_t(Q)^2/Q^2) through 4 loops: SMDR QCDQED_match.c
    (Martin 1812.04100 Eqs. (3.4)-(3.9)), MSbar top mass in the 6-flavour theory.
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import math
import json
import numpy as np
from numpy.polynomial import polynomial as NP
from scipy.integrate import solve_ivp

PI = math.pi
Z3 = 1.2020569031595942
Z4 = PI ** 4 / 90
Z5 = 1.0369277551433699
HERE = os.path.dirname(os.path.abspath(__file__))
MZ = 91.1876
Z4_NL5, Z4_NL5_ERR = -859.96, 21.5


def beta_as(nf):
    b0 = 11 - 2 * nf / 3
    b1 = 102 - 38 * nf / 3
    b2 = 2857 / 2 - 5033 * nf / 18 + 325 * nf ** 2 / 54
    b3 = ((149753 / 6 + 3564 * Z3) - (1078361 / 162 + 6508 * Z3 / 27) * nf + (50065 / 162 + 6472 * Z3 / 81) * nf ** 2
          + 1093 * nf ** 3 / 729)
    b4 = (8157455 / 16 + 621885 / 2 * Z3 - 88209 / 2 * Z4 - 288090 * Z5
          + nf * (-336460813 / 1944 - 4811164 / 81 * Z3 + 33935 / 6 * Z4 + 1358995 / 27 * Z5)
          + nf ** 2 * (25960913 / 1944 + 698531 / 81 * Z3 - 10526 / 9 * Z4 - 381760 / 81 * Z5)
          + nf ** 3 * (-630559 / 5832 - 48722 / 243 * Z3 + 1618 / 27 * Z4 + 460 / 9 * Z5)
          + nf ** 4 * (1205 / 2916 - 152 / 81 * Z3))
    return [b0, b1, b2, b3, b4]


def beta_api(nf):
    """a = alpha_s/pi : da/dln mu^2 = -sum_n bp_n a^(n+2)"""
    return [b / 4 ** (n + 1) for n, b in enumerate(beta_as(nf))]


def gamma_m(nf):
    g0 = 1.0
    g1 = (202 / 3 - 20 * nf / 9) / 16
    g2 = (1249 + (-2216 / 27 - 160 * Z3 / 3) * nf - 140 * nf ** 2 / 81) / 64
    g3 = (4603055 / 162 + 135680 * Z3 / 27 - 8800 * Z5
          + (-91723 / 27 - 34192 * Z3 / 9 + 880 * Z4 + 18400 * Z5 / 9) * nf
          + (5242 / 243 + 800 * Z3 / 9 - 160 * Z4 / 3) * nf ** 2 + (-332 / 243 + 64 * Z3 / 27) * nf ** 3) / 256
    g4 = -19988.8 / 2048 if nf == 6 else 0.0
    return [g0, g1, g2, g3, g4]


def run_a(a0, mu0, mu1, nf, nloop=5):
    """alpha_s/pi from mu0 to mu1 (nf fixed), nloop-loop beta."""
    if abs(math.log(mu1 / mu0)) < 1e-14:
        return a0
    bp = beta_api(nf)[:nloop]

    def f(t, y):
        a = y[0]
        return [-2 * sum(b * a ** (n + 2) for n, b in enumerate(bp))]
    s = solve_ivp(f, (math.log(mu0), math.log(mu1)), [a0], method="DOP853", rtol=1e-13, atol=1e-18)
    return float(s.y[0, -1])


# ------------------------------------------------------------------------------------------------ pole <-> MSbar
def _c_nl(nl):
    c2 = 13.443433 - 1.041367 * nl
    c3 = 190.595 - 26.655 * nl + 0.6527 * nl ** 2
    return c2, c3


def _polys(zc, nf=6, N=4):
    bp, gm = beta_api(nf), gamma_m(nf)
    Ps = [np.array([1.0])]
    for n in range(1, N + 1):
        dP = np.zeros(1)
        for k in range(1, n):
            dP = NP.polyadd(dP, k * bp[n - 1 - k] * Ps[k])
        for j in range(0, n):
            dP = NP.polysub(dP, gm[n - 1 - j] * Ps[j])
        Pn = NP.polyint(dP)
        Pn[0] = zc[n]
        Ps.append(Pn)
    return Ps


def zm_to_cm(zc, nf=6, N=4):
    """numerical series inversion: given z constants -> c_n of M = m(m) sum c_n a(m)^n (a = a^(6)(m))."""
    Ps = _polys(zc, nf, N)
    # x(a) solves x = sum a^k P_k(2 ln x); fit Taylor coefficients by exact recursion via sympy-free fixed point
    import sympy as sp
    a = sp.Symbol("a")
    xs = sp.symbols("x1:%d" % (N + 1))
    X = 1 + sum(xs[i] * a ** (i + 1) for i in range(N))
    ell = sp.expand(sp.series(2 * sp.log(X), a, 0, N + 1).removeO())
    expr = -X
    for k, P in enumerate(Ps):
        expr += a ** k * sum(sp.Float(float(c), 30) * ell ** j for j, c in enumerate(P))
    ser = sp.expand(sp.series(sp.expand(expr), a, 0, N + 1).removeO())
    sol = {}
    for n in range(1, N + 1):
        eq = sp.expand(ser.coeff(a, n).subs(sol))
        sol[xs[n - 1]] = sp.solve(eq, xs[n - 1])[0]
    Xv = sp.expand(X.subs(sol))
    inv = sp.expand(sp.series(1 / Xv, a, 0, N + 1).removeO())
    return [float(inv.coeff(a, n)) for n in range(N + 1)]


_ZC = {}


def zconst(nl=5, z4=Z4_NL5):
    """z_m constants at mu = M: z1=-4/3, z2,z3 fitted to c2,c3 (exact linear solve), z4 from Marquard."""
    key = (nl, round(z4, 6))
    if key in _ZC:
        return _ZC[key]
    c2, c3 = _c_nl(nl)
    z = {0: 1.0, 1: -4.0 / 3, 2: 0.0, 3: 0.0, 4: z4}
    # c_n depends linearly on z_n (coefficient -1) at fixed lower z's: two-step solve
    cm = zm_to_cm(z, N=2)
    z[2] = z[2] - (c2 - cm[2])
    cm = zm_to_cm(z, N=3)
    z[3] = z[3] - (c3 - cm[3])
    cm = zm_to_cm(z, N=4)
    _ZC[key] = dict(z=z, c=cm)
    return _ZC[key]


def zm(a, ell, N=4, z4=Z4_NL5):
    Ps = _polys(zconst(5, z4)["z"], 6, N)
    return float(sum(a ** k * NP.polyval(ell, Ps[k]) for k in range(N + 1)))


def zm_terms(a, ell, N=4, z4=Z4_NL5):
    Ps = _polys(zconst(5, z4)["z"], 6, N)
    return [float(a ** k * NP.polyval(ell, Ps[k])) for k in range(N + 1)]


# ------------------------------------------------------------------------------------------------ decoupling
def Zas(aS, L, nloop, nq=5):
    """alpha_s^(5)/alpha_s^(6) at Q; aS = alpha_s^(6)(Q)/(4 pi); L = ln(m_t(Q)^2/Q^2) (MSbar, 6-flavour). SMDR."""
    TF, CF = 0.5, 4.0 / 3
    Z = 1.0
    if nloop >= 1:
        Z += (4.0 / 3) * TF * aS * L
    if nloop >= 2:
        Z += (16.0 / 9) * TF * TF * aS ** 2 * L ** 2 - TF * aS * (CF * aS) * (4 * L + 13.0 / 3) + TF * aS ** 2 * (
            20 * L + 32.0 / 3)
    if nloop >= 3:
        Z += aS ** 3 * (8 * L ** 3 / 27 + (53 / 9 - 16 * nq / 9) * L ** 2 + (955 / 9 - 67 * nq / 9) * L
                        + 62.211628 - 2633 * nq / 486)
    if nloop >= 4:
        Z += aS ** 4 * (16 * L ** 4 / 81 + (3766 / 81 + 508 * nq / 81 - 64 * nq ** 2 / 81) * L ** 3
                        + (4354 / 27 - 2966 * nq / 81 - 77 * nq ** 2 / 81) * L ** 2
                        + (2157.863053 - 335.316171 * nq - 6865 * nq ** 2 / 729) * L
                        + 1323.608830 - 258.542470 * nq - 5.62646 * nq ** 2)
    return Z


def as6_at(Mt, as5_MZ, mu_dec=None, mu_out=None, nrun=5, ndec=4, zN=4):
    """alpha_s^(6)(mu_out) from alpha_s^(5)(MZ): nrun-loop running, ndec-loop decoupling at mu_dec (default Mt)."""
    mu_dec = Mt if mu_dec is None else mu_dec
    mu_out = Mt if mu_out is None else mu_out
    a5 = run_a(as5_MZ / PI, MZ, mu_dec, 5, nrun)
    a6 = a5
    for _ in range(60):
        m = Mt * zm(a6, 2 * math.log(mu_dec / Mt), zN)
        a6n = a5 / Zas(a6 / 4, math.log(m * m / mu_dec ** 2), ndec)
        if abs(a6n - a6) < 1e-16:
            a6 = a6n
            break
        a6 = a6n
    return PI * run_a(a6, mu_dec, mu_out, 6, nrun)


def mbar_mbar(Mt, as6_Mt, N=4, nrun=5):
    """scale-invariant MSbar mass mbar(mbar) (pure QCD, 6-flavour alpha_s) from the pole mass."""
    m = Mt * 0.94
    for _ in range(60):
        a = run_a(as6_Mt / PI, Mt, m, 6, nrun)
        mn = Mt * zm(a, 2 * math.log(m / Mt), N)
        if abs(mn - m) < 1e-11:
            return mn
        m = mn
    return m


def pole_from_mbar(mb, as6_of_Mt, N=4, guess=173.0):
    """invert mbar_mbar (as6_of_Mt: callable Mt -> alpha_s^(6)(Mt))."""
    from scipy.optimize import brentq
    return brentq(lambda M: mbar_mbar(M, as6_of_Mt(M), N) - mb, guess - 15, guess + 15, xtol=1e-10)


def selfcheck():
    out = {}
    out["beta_nf6"] = beta_as(6)
    out["beta_nf5"] = beta_as(5)
    out["check_beta3_nf6_vs_Buttazzo_2472.28"] = beta_as(6)[3]
    out["check_beta4_nf6_vs_SMDR_271.428"] = beta_as(6)[4]
    gm = gamma_m(6)
    out["gamma_m_nf6"] = gm
    out["check_-128*g2_vs_Buttazzo_-619.35"] = -128 * gm[2]
    out["check_-512*g3_vs_SMDR_2308.18"] = -512 * gm[3]
    zc = zconst()
    out["z_const"] = zc["z"]
    out["c_from_z"] = zc["c"]
    out["check_c4_vs_Marquard_8.49pi^4=%.1f" % (8.49 * PI ** 4)] = zc["c"][4]
    P4 = _polys(zc["z"])[4]
    out["P4_logs_vs_Marquard(-328.94,-50.856,-6.4922,-0.33203)"] = [float(x) for x in P4[1:]]
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(selfcheck(), indent=1))
