# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_twist.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128, v12 round 3) on 2026-10-14 (no copied code).
r"""[v12 package, 2026-10-15] Twisted image sums for the P/A spin-structure sectors and 't Hooft flux sectors (Z6: |Phi_Y| >= 2 pi, classical flux energy 100.7/L = 2.36 x AAA Casimir).  Paper: PRD Sec. IV D. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C156 T2/T3: twisted Epstein sums for the other spin structures and the 't Hooft flux bound.
Z_s(a, th) = sum'_n cos(2 pi n.th) / (sum_i a_i n_i^2)^s, diagonal metric a (det a = 1 on the shape slice), by the Mellin
representation with a Poisson split at t = 1; the leading dual exponential exp(-c tau) of the t < 1 part is integrated in
closed form, c^(s-3/2) Gamma(3/2-s, c) (c = 0: 1/(s-3/2)), which carries the n = 0 cusp of periodic fields exactly.
Checks: Z_2(1,0) = 16.5323159598, Z_2(1,(1/2,1/2,1/2)) = -3.8631638072 (v11), Z_1(1,0) = -8.913633, Z_1(AAA) = -2.519356
(D121); direct lattice sums for the absolutely convergent twisted Z_2.
Outputs C156_twist.json:
  * N_C for PPP/PPA/PAA/AAA, tetragonal shape scan Z_eff(beta), linear force F = 3 Z'/Z, Hessian kappa, shape displacement
    over the cosh bounce (linear response, beta'' + 3 tanh(2tau) beta' = sech^2(2tau)(kappa beta + F));
  * per-direction curvature sums Y_i(th) = 3 sum' n_i^2 cos/|n|^4 = -3 dZ_1/da_i, holonomy curvatures kappa_Y,i,
    kappa_2,i, kappa_3,i (D121 normalisation), fermion tadpoles Z_1(th_F);
  * U(1)_Y holonomy potential along one direction (nonzero modes; periodic-boson n = 0 cusps removed) for each structure:
    curvature, global minimum delta*, depth, Higgs twist at delta*;
  * L_b, hL_b, mu_b, eta_inf from N_C^eff = N_C - Q/gamma (Q = 12.8 fixed: conditional);
  * 't Hooft flux: Z_6 consistency -> minimal hypercharge flux 2 pi, E_flux = 2 pi^2/(g_Y^2 L) vs C = N_C gamma; Higgs lowest
    Landau mode and WKB lambda_c of the twisted Higgs mode.
Usage: python -B U156_twist.py
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import time
import numpy as np
import mpmath as mp
from scipy.integrate import quad, solve_ivp

OUT = _zp.w("r3/universality/run01")
PI = math.pi
GAM = 1.6750738                                      # Z_2(1)/pi^2 (v11)


def th1(t, a, th):
    N = int(math.ceil(math.sqrt(46.0 / (PI * a * t)))) + 1
    n = np.arange(-N, N + 1)
    return float(np.sum(np.exp(-PI * a * n * n * t) * np.cos(2 * PI * n * th)))


def thd(tau, a, th):
    N = int(math.ceil(math.sqrt(46.0 * a / (PI * tau)))) + 2
    m = np.arange(-N, N + 1) + round(th)
    return float(np.sum(np.exp(-PI * (m - th) ** 2 * tau / a)))


def Zs(s, a, th):
    a = [float(x) for x in a]
    th = [float(x) for x in th]
    da = a[0] * a[1] * a[2]
    ms = [round(x) for x in th]
    c = PI * sum((m - x) ** 2 / ai for m, x, ai in zip(ms, th, a))

    def f1(t):
        return t ** (s - 1) * (th1(t, a[0], th[0]) * th1(t, a[1], th[1]) * th1(t, a[2], th[2]) - 1.0)

    def f2(tau):
        return tau ** (0.5 - s) * (thd(tau, a[0], th[0]) * thd(tau, a[1], th[1]) * thd(tau, a[2], th[2]) - math.exp(-c * tau))
    amin = min(a)
    T1 = 60.0 / (PI * amin) + 5
    I1 = quad(f1, 1.0, T1, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    # remainder of the dual sum decays at least like exp(-pi tau (1/4)/a_max) beyond the leading term
    T2 = 60.0 * max(a) / PI * 4 + 5
    I2 = quad(f2, 1.0, T2, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    J = 1.0 / (s - 1.5) if c < 1e-300 else float(mp.power(c, s - 1.5) * mp.gammainc(1.5 - s, c))
    return PI ** s / math.gamma(s) * (I1 + (I2 + J) / math.sqrt(da) - 1.0 / s)


def direct_Z2(th, N=40):
    n = np.arange(-N, N + 1)
    n1, n2, n3 = np.meshgrid(n, n, n, indexing="ij")
    r2 = (n1 * n1 + n2 * n2 + n3 * n3).astype(float)
    r2[N, N, N] = np.inf
    return float(np.sum(np.cos(2 * PI * (n1 * th[0] + n2 * th[1] + n3 * th[2])) / r2 ** 2))


SPIN = {"PPP": (0.0, 0.0, 0.0), "PPA": (0.0, 0.0, 0.5), "PAA": (0.0, 0.5, 0.5), "AAA": (0.5, 0.5, 0.5)}
# SM Weyl fermions (left-handed basis): (name, multiplicity, Y, T3-doublet?)
WEYL = [("Q", 6, 1 / 6), ("uc", 3, -2 / 3), ("dc", 3, 1 / 3), ("L", 2, -1 / 2), ("ec", 1, 1.0)]


def zeff(a, thF):
    return 15.0 * Zs(2, a, (0, 0, 0)) - 45.0 * Zs(2, a, thF)


def tet(beta, axis):
    e = 2 * beta / math.sqrt(6)
    a = [math.exp(e)] * 3
    a[axis] = math.exp(-2 * e)
    return a


def Ydir(thF, i, h=2e-4):
    ap, am = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    ap[i] += h
    am[i] -= h
    return -3 * (Zs(1, ap, thF) - Zs(1, am, thF)) / (2 * h)


def wrap(x):
    return x - round(x)


def VY(delta, i, thF, cusp=True):
    """U(1)_Y one-loop energy (L = 1) along direction i, delta = twist of a Y = 1 field; image-sum weights d_f s_f."""
    e = [0.0, 0.0, 0.0]
    e[i] = 1.0
    tot = 0.0
    # Higgs doublet: 2 complex scalars, d = 1 each, Y = 1/2, periodic
    thH = [0.5 * delta * x for x in e]
    tot += -(2.0 / PI ** 2) * Zs(2, (1, 1, 1), thH)
    if cusp:
        tot -= 2.0 * 2 * PI * abs(wrap(0.5 * delta))           # remove the Higgs n = 0 zero-point (slow coordinate)
    for nm, mult, Y in WEYL:
        thf = [thF[k] + Y * delta * e[k] for k in range(3)]
        tot += +(3.0 * mult / PI ** 2) * Zs(2, (1, 1, 1), thf)  # fermions s = -1, d = 1, 3 generations
    return tot


def shape_ode(kap, F, tmax=8.0):
    def rhs(t, y):
        return [y[1], -3 * math.tanh(2 * t) * y[1] + (kap * y[0] + F) / math.cosh(2 * t) ** 2]
    sym = solve_ivp(rhs, (0, tmax), [0.0, 0.0], rtol=1e-11, atol=1e-13).y[0][-1]
    full = solve_ivp(rhs, (-tmax, tmax), [0.0, 0.0], rtol=1e-11, atol=1e-13).y[0][-1]
    return sym, full


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    out = {}
    one = (1.0, 1.0, 1.0)
    Z20, Z2A = Zs(2, one, (0, 0, 0)), Zs(2, one, (0.5, 0.5, 0.5))
    Z10, Z1A = Zs(1, one, (0, 0, 0)), Zs(1, one, (0.5, 0.5, 0.5))
    out["checks"] = dict(Z2_0=Z20, Z2_AAA=Z2A, Z1_0=Z10, Z1_AAA=Z1A, ref=[16.5323159598, -3.8631638072, -8.913633, -2.519356],
                         direct_Z2_N40={k: direct_Z2(v) for k, v in SPIN.items() if k != "PPP"},
                         mellin_Z2={k: Zs(2, one, v) for k, v in SPIN.items()})
    print("checks", json.dumps(out["checks"]), "%.1fs" % (time.time() - t0), flush=True)
    # ---- N_C, Z_1 (tadpole), directional curvature sums
    sp = {}
    for k, thF in SPIN.items():
        Z2F = Zs(2, one, thF)
        Zeff = 15 * Z20 - 45 * Z2F
        Yd = [Ydir(thF, i) for i in range(3)]
        Y0 = [Ydir((0, 0, 0), i) for i in range(1)]
        kY = [(Z10 - 20 * y) / (6 * PI ** 2) for y in Yd]
        k2 = [(5 * Z10 - 12 * y) / (6 * PI ** 2) for y in Yd]
        k3 = [(Z10 - 2 * y) / PI ** 2 for y in Yd]
        sp[k] = dict(Z2F=Z2F, Zeff=Zeff, N_C=Zeff / Z20, Z1F=Zs(1, one, thF), Ydir=Yd, Ydir_periodic_check=Y0,
                     kappa_Y=kY, kappa_2=k2, kappa_3=k3, tadpole_ratio_vs_AAA=Zs(1, one, thF) / Z1A)
        print(k, json.dumps(sp[k]), flush=True)
    out["spin"] = sp
    # ---- tetragonal shape scans (PPA: special axis = the A axis 2; PAA: special axis = the P axis 0)
    shp = {}
    for k, axis in (("PPA", 2), ("PAA", 0)):
        thF = SPIN[k]
        bs = np.round(np.arange(-1.5, 1.5001, 0.125), 4)
        zz = [zeff(tet(b, axis), thF) for b in bs]
        h = 1e-3
        zp = (zeff(tet(h, axis), thF) - zeff(tet(-h, axis), thF)) / (2 * h)
        z0 = zeff(tet(0.0, axis), thF)
        zpp = (zeff(tet(h, axis), thF) - 2 * z0 + zeff(tet(-h, axis), thF)) / h ** 2
        dz = np.gradient(np.array(zz), bs)
        stat = [float(0.5 * (bs[j] + bs[j + 1])) for j in range(len(bs) - 1) if dz[j] * dz[j + 1] < 0]
        F, kap = 3 * zp / z0, 3 * zpp / z0
        res = dict(axis=axis, beta=bs.tolist(), Zeff=zz, Zp=zp, Zpp=zpp, F=F, kappa_tet=kap, stationary_points=stat,
                   note="beta: unit shape-metric coordinate, G = exp(2 beta X), X = diag(1,1,-2)/sqrt6 with -2 on the special axis")
        for xi_lab, xi in (("xi0", 0.0), ("Q12.8", 12.8 / (z0 / PI ** 2))):
            Fe, ke = F / (1 - xi), kap / (1 - xi)
            sym, full = shape_ode(ke, Fe)
            lin_sym, lin_full = shape_ode(0.0, 1.0)
            res[xi_lab] = dict(xi=xi, F=Fe, kappa=ke, dbeta_half_sym=sym, dbeta_from_rest=full,
                               c_sym_unitF=lin_sym, c_full_unitF=lin_full)
        shp[k] = res
        print("shape", k, json.dumps({x: res[x] for x in ("Zp", "Zpp", "F", "kappa_tet", "stationary_points", "xi0", "Q12.8")}),
              flush=True)
    out["shape"] = shp
    # ---- U(1)_Y holonomy potential along one direction
    hol = {}
    dgrid = np.round(np.arange(0.0, 6.0001, 0.05), 4)
    for k, dirs in (("AAA", [0]), ("PPP", [0]), ("PPA", [0, 2]), ("PAA", [0, 1])):
        for i in dirs:
            v = np.array([VY(d, i, SPIN[k]) for d in dgrid])
            j = int(np.argmin(v))
            # refine minimum
            dd = np.linspace(max(dgrid[j] - 0.05, 0), min(dgrid[j] + 0.05, 6.0), 41)
            vv = np.array([VY(d, i, SPIN[k]) for d in dd])
            jm = int(np.argmin(vv))
            h = 1e-3
            curv = (VY(h, i, SPIN[k]) - 2 * VY(0.0, i, SPIN[k]) + VY(-h, i, SPIN[k])) / h ** 2
            dstar = float(dd[jm])
            twH = abs(wrap(0.5 * dstar))
            hol["%s_dir%d" % (k, i)] = dict(V0=float(v[0]), Vmin=float(vv[jm]), delta_star=dstar, depth=float(v[0] - vv[jm]),
                                            curv0=curv, kappa_Y=curv / (4 * PI ** 2), higgs_twist=twH,
                                            higgs_gap=2 * PI * twH, V_grid=v.tolist())
            print("hol", k, i, json.dumps({x: hol["%s_dir%d" % (k, i)][x] for x in ("delta_star", "depth", "kappa_Y",
                                                                                     "higgs_twist", "higgs_gap")}), flush=True)
    out["holonomy_Y"] = dict(delta_grid=dgrid.tolist(), res=hol)
    # ---- background quantities, conditional on Q = 12.8 (shape-independent leading-order zero-mode energy)
    bg = {}
    ref = sp["AAA"]
    NC_AAA_eff = ref["N_C"] - 12.8 / GAM
    for k in ("PPA", "PAA", "AAA"):
        NC = sp[k]["N_C"]
        NCe = NC - 12.8 / GAM
        fac = (NCe / NC_AAA_eff) ** 0.25
        hlb = 0.00437243 * fac
        bg[k] = dict(N_C=NC, xi=12.8 / (NC * GAM), N_C_eff=NCe, LbMP=722.0 * fac, hLb=hlb, eta_inf=1.31102877714606 / hlb,
                     mu_b_GeV=2.435e18 / (722.0 * fac), dln_eta_vs_AAA=math.log(1.31102877714606 / hlb * 0.00437243 / 1.31102877714606))
    out["background"] = bg
    # ---- 't Hooft flux sector
    gY = 0.4427
    Phi = 2 * PI
    Ef = Phi ** 2 / (2 * gY ** 2)
    om0 = math.sqrt(0.5 * Phi)                         # Higgs LLL: |q B| = Y Phi / L^2 = pi
    betaA = 1.1803                                     # Abrikosov parameter (square), int|f|^4 for the single LLL state
    flux = dict(Phi_Y_min=Phi, E_flux=Ef, C_AAA=sp["AAA"]["N_C"] * GAM, ratio=Ef / (sp["AAA"]["N_C"] * GAM),
                N_C_needed=(Ef + 0.0) / GAM, omega0_higgs=om0, S_wkb_coeff=4 * om0 ** 3 / (3 * betaA))
    lc = {}
    for lab, It in (("P1=0.9", -math.log(0.9)), ("P1=0.5", math.log(2)), ("P1=0.1", math.log(10))):
        Sc = math.log(om0 / (2 * PI) * 250.0 / It)
        lc[lab] = dict(S_c=Sc, lam_c=-flux["S_wkb_coeff"] / Sc)
    flux["lam_c_wkb"] = lc
    # Z_6 consistency table: minimal |Phi_Y| for (k3, k2) center twists
    z6 = {}
    for k3 in range(3):
        for k2 in range(2):
            best = None
            for nph in range(-12, 13):
                ph = nph * PI
                ok = True
                for (Y, c3, c2) in ((1 / 6, 1, 1), (2 / 3, 1, 0), (-1 / 3, 1, 0), (-1 / 2, 0, 1), (-1.0, 0, 0), (0.5, 0, 1)):
                    z = Y * ph + c3 * 2 * PI * k3 / 3 + c2 * PI * k2
                    if abs(wrap(z / (2 * PI))) > 1e-9:
                        ok = False
                if ok and (best is None or abs(ph) < abs(best)):
                    best = ph
            z6["k3=%d,k2=%d" % (k3, k2)] = best / PI if best is not None else None
    flux["Z6_min_flux_over_pi"] = z6
    out["flux"] = flux
    out["time_s"] = time.time() - t0
    with open(os.path.join(OUT, "C156_twist.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"background": bg, "flux": flux}, indent=1))


if __name__ == "__main__":
    main()
