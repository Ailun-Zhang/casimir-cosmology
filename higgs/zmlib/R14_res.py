# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/R14_res.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/R14_res.py on 2026-10-14 by agent holonomy-cross (C149/D121).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_matching/R14_res.py on 2026-10-13 by agent higgs-beyond-bo (C140/D112).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12/higgs_survival/xi_scan/R14_res.py on 2026-10-13 by agent higgs-matching (C138/D110).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Conditional-Gaussian gauge profile of the Higgs zero mode (D108 functional, Newton marching in r) and the 1D radial resonance solver (Jost function, 2nd-order WKB outgoing/Siegert condition, complex Muller).  Paper: PRD App. H1-H2. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

R14 checks 2+3: reduced gauge-invariant radial problem of the EW zero modes (model definitions from D108 Sec.1),
own implementation: sympy-generated conditional-Gaussian energy, marching + vectorised Newton minimisation,
implicit-derivative Born-Huang term, own finite-volume matching (R14_fv), Numerov integration with a
2nd-order WKB outgoing (Siegert) condition and a vectorised complex Muller root search.
Usage:  python -B R14_res.py bench|hazard|all
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import math
import time
import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, brentq
from scipy.integrate import quad, simpson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import R14_fv  # noqa: E402
import R14_rge  # noqa: E402

PI = math.pi
HLB = 0.00437243
ETA_TOT = 1.31102877714606 / HLB
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
H_N = 0.001
R0 = 0.01


def merge_json(name, key, obj):
    R14_rge.merge_json(_zp.w("r1/R14_checks/" + name), key, obj)


# ------------------------------------------------------------------ conditional Gaussian
def _build():
    a, k, b, c, r, g2, gY = sp.symbols("a k b c r g2 gY", real=True)
    D = k * b - c ** 2
    E = (sp.Rational(3, 4) * (2 * a + k + b)
         + sp.Rational(3, 16) * r ** 2 * (2 * g2 ** 2 / a + (g2 ** 2 * b - 2 * g2 * gY * c + gY ** 2 * k) / D)
         + sp.Rational(3, 4) * g2 ** 2 / a ** 2 + sp.Rational(3, 2) * g2 ** 2 * b / (a * D)
         + 3 * (b * (k - a) ** 2 - 2 * c ** 2 * (k - a) + k * c ** 2) / (a * r ** 2 * D))
    P = [a, k, b, c]
    G = [sp.diff(E, v) for v in P]
    H = [[sp.diff(G[i], P[j]) for j in range(4)] for i in range(4)]
    Gr = [sp.diff(Gi, r) for Gi in G]
    args = (a, k, b, c, r, g2, gY)
    return (sp.lambdify(args, E, "numpy"), sp.lambdify(args, G, "numpy"),
            sp.lambdify(args, H, "numpy"), sp.lambdify(args, Gr, "numpy"))


FE, FG, FH, FGR = _build()


def _vec(lst, n):
    return np.array([np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in lst])


def _mat(lst, n):
    return np.array([[np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in row] for row in lst])


def feasible(p):
    return p[0] > 0 and p[1] > 0 and p[2] > 0 and p[1] * p[2] - p[3] ** 2 > 0


def newton1(p, r, g2, gY, maxit=200):
    p = np.array(p, float)
    E0 = FE(*p, r, g2, gY)
    for _ in range(maxit):
        g = _vec(FG(*p, r, g2, gY), 1)[:, 0]
        Hm = _mat(FH(*p, r, g2, gY), 1)[:, :, 0]
        w, V = np.linalg.eigh(Hm)
        w = np.where(w > 0, w, np.maximum(np.abs(w), 1e-8 * np.abs(w).max()))
        step = V @ ((V.T @ g) / w)
        dec = float(g @ step)
        t = 1.0
        while True:
            pn = p - t * step
            if feasible(pn):
                En = FE(*pn, r, g2, gY)
                if En <= E0 + 1e-15 * abs(E0):
                    break
            t *= 0.5
            if t < 1e-14:
                pn, En = p, E0
                break
        p, E0 = pn, En
        if dec < 1e-28:
            break
    return p, E0


def global_check(r, g2, gY, Emarch, ntry=10, seed=7):
    rng = np.random.default_rng(seed)

    def f(q):
        a, k, b = np.exp(q[:3])
        c = np.tanh(q[3]) * np.sqrt(k * b)
        return FE(a, k, b, c, r, g2, gY)
    best = np.inf
    for _ in range(ntry):
        q0 = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-5, 2), rng.uniform(-1.5, 1.5)])
        rr = minimize(f, q0, method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-14, maxiter=40000,
                                                                 maxfev=80000))
        rr = minimize(f, rr.x, method="BFGS", options=dict(gtol=1e-11))
        best = min(best, float(rr.fun))
    return best - Emarch


def gauss_profile(g2, gY, rmax=18.2, hc=0.01, hf=0.002, rf0=0.004, check_global=False):
    rc = np.arange(0.01, rmax + 1e-9, hc)
    a0 = (2 * g2 * g2) ** (1.0 / 3)
    n = len(rc)
    S = np.zeros((n, 4))
    Ec = np.zeros(n)
    for i, r in enumerate(rc):
        if i == 0:
            guess = np.array([a0, a0, gY * r / 2, 0.0])
        elif i == 1:
            guess = S[0].copy()
        else:
            guess = 2 * S[i - 1] - S[i - 2]
            if not feasible(guess):
                guess = S[i - 1].copy()
        S[i], Ec[i] = newton1(guess, r, g2, gY)
    rf = np.arange(rf0, rmax - 0.05, hf)
    nf = len(rf)
    P = np.empty((4, nf))
    lo = rf < rc[0]
    hi = ~lo
    for j in (0, 1, 2):
        P[j, hi] = np.exp(np.interp(rf[hi], rc, np.log(S[:, j])))
    P[3, hi] = np.interp(rf[hi], rc, S[:, 3])
    P[0, lo], P[1, lo] = S[0, 0], S[0, 1]
    P[2, lo] = S[0, 2] * rf[lo] / rc[0]
    P[3, lo] = S[0, 3] * (rf[lo] / rc[0]) ** 4
    dec = None
    for _ in range(20):
        g = _vec(FG(*P, rf, g2, gY), nf)
        Hm = _mat(FH(*P, rf, g2, gY), nf).transpose(2, 0, 1)
        step = np.linalg.solve(Hm, g.T[:, :, None])[:, :, 0].T
        P = P - step
        dec = np.abs(np.einsum("ij,ij->j", g, step))
        if dec.max() < 1e-28:
            break
    feas = (P[0] > 0) & (P[1] > 0) & (P[2] > 0) & (P[1] * P[2] - P[3] ** 2 > 0)
    if not feas.all():
        raise RuntimeError("infeasible Gaussian parameters")
    Et = FE(*P, rf, g2, gY)
    Hm = _mat(FH(*P, rf, g2, gY), nf).transpose(2, 0, 1)
    eigmin = np.linalg.eigvalsh(Hm).min(axis=1)
    Gr = _vec(FGR(*P, rf, g2, gY), nf)
    dP = -np.linalg.solve(Hm, Gr.T[:, :, None])[:, :, 0].T
    a, k, b, c = P
    ap, kp, bp, cp = dP
    D = k * b - c * c
    M11 = (b * kp - c * cp) / D
    M12 = (b * cp - c * bp) / D
    M21 = (-c * kp + k * cp) / D
    M22 = (-c * cp + k * bp) / D
    trM2 = M11 ** 2 + 2 * M12 * M21 + M22 ** 2
    Br = 3.0 / 16 * (2 * (ap / a) ** 2 + trM2)
    Ug = Et + Br
    spl = CubicSpline(rf, rf * rf * Ug)
    sel = rf <= 0.06
    A = np.vstack([np.ones(sel.sum()), rf[sel], rf[sel] ** 2, rf[sel] ** 3]).T
    coef = np.linalg.lstsq(A, (rf * rf * Ug)[sel], rcond=None)[0]
    common = np.searchsorted(rf, rc[(rc > rf[0] + 1e-9) & (rc < rf[-1])])
    out = dict(rf=rf, P=P, Et=Et, Br=Br, spl=spl, beta_g=float(coef[0]), alpha_g=float(coef[1]),
               maxdec=float(dec.max()), min_hess_eig=float(eigmin.min()),
               march_vs_fine=float(np.max(np.abs(np.interp(rf[common], rc, Ec) - Et[common]))),
               g2=g2, gY=gY)
    if check_global:
        gc = {}
        for rr in (0.3, 1.0, 3.0, 6.0, 10.0):
            i = int(np.argmin(np.abs(rf - rr)))
            gc[f"{rf[i]:.3f}"] = global_check(rf[i], g2, gY, Et[i])
        out["global_minus_march"] = gc
    return out


# ------------------------------------------------------------------ potentials
FVF = R14_fv.FV()
TB = 2 * FVF.a[0][1]


def dV(model, r, lam, g2, gY, yt):
    if model == "L0":
        return np.zeros_like(r)
    mW2 = (g2 * r / 2) ** 2
    mZ2 = (g2 * g2 + gY * gY) * r * r / 4
    mt2 = yt * yt * r * r / 2
    out = 6 * FVF.DP(mW2) + 3 * FVF.DP(mZ2) - 12 * FVF.DA(mt2)
    if model == "M0":
        return out
    if model == "LQ":
        return out + 3 * lam * TB * r * r
    if model == "MU":
        return out + FVF.DP(3 * lam * r * r)
    if model == "ML":
        return out + FVF.DP(3 * lam * r * r) + 3 * FVF.DP(lam * r * r)
    raise ValueError(model)


def U_radial(gp, model, lam, g2, gY, yt, r, aterm=0.0):
    return 3.0 / (8 * r * r) + gp["spl"](r) / (r * r) + lam * r ** 4 / 4 + dV(model, r, lam, g2, gY, yt) \
        - aterm * r * r


def grid(rout, h=H_N):
    m = int(round((rout - R0) / h))
    return R0 + h * np.arange(m + 3), m


def jost(E, Ut, r, m, s, alpha, h=H_N):
    """Numerov for u''=2(U-E)u (columns = problems); returns f=u'-y u at r[m] and u(r[m])."""
    E = np.asarray(E, complex)
    hh = h * h / 12.0
    u_prev = (r[0] ** s * (1 + alpha / s * r[0])) + 0j * E
    u_cur = (r[1] ** s * (1 + alpha / s * r[1])) + 0j * E
    w_prev = 1 - hh * 2 * (Ut[0] - E)
    w_cur = 1 - hh * 2 * (Ut[1] - E)
    store = {}
    for i in range(1, m + 2):
        w_next = 1 - hh * 2 * (Ut[i + 1] - E)
        u_next = ((12 - 10 * w_cur) * u_cur - w_prev * u_prev) / w_next
        if i + 1 >= m - 2:
            store[i + 1] = u_next
        u_prev, u_cur, w_prev, w_cur = u_cur, u_next, w_cur, w_next
    if m - 2 <= 1:
        raise ValueError("grid too short")
    um = store[m]
    up = (store[m - 2] - 8 * store[m - 1] + 8 * store[m + 1] - store[m + 2]) / (12 * h)
    Um = Ut[m]
    U1 = (Ut[m - 2] - 8 * Ut[m - 1] + 8 * Ut[m + 1] - Ut[m + 2]) / (12 * h)
    U2 = (-Ut[m - 2] + 16 * Ut[m - 1] - 30 * Ut[m] + 16 * Ut[m + 1] - Ut[m + 2]) / (12 * h * h)
    kk = np.sqrt(2 * (E - Um))
    k1 = -U1 / kk
    k2 = -U2 / kk - U1 ** 2 / kk ** 3
    y = 1j * kk - k1 / (2 * kk) + (1j / (2 * kk)) * (3 * k1 ** 2 / (4 * kk ** 2) - k2 / (2 * kk))
    return up - y * um, um


def muller(fun, x0, x1, x2, tol=1e-13, maxit=80):
    f0, f1, f2 = fun(x0), fun(x1), fun(x2)
    act = np.ones(x0.shape, bool)
    it = 0
    with np.errstate(all="ignore"):
        for it in range(maxit):
            h1 = x1 - x0
            h2 = x2 - x1
            d1 = (f1 - f0) / h1
            d2 = (f2 - f1) / h2
            A = (d2 - d1) / (h2 + h1)
            B = A * h2 + d2
            disc = np.sqrt(B * B - 4 * A * f2)
            den = np.where(np.abs(B + disc) >= np.abs(B - disc), B + disc, B - disc)
            dx = -2 * f2 / den
            dx = np.where(act & np.isfinite(dx), dx, 0)
            x3 = x2 + dx
            x0, x1, x2 = x1, x2, x3
            f0, f1 = f1, f2
            f2 = fun(x3)
            act = act & (np.abs(dx) > tol * np.maximum(1, np.abs(x3)))
            if not act.any():
                break
    return x2, it


def spar(gp):
    beta = 3.0 / 8 + gp["beta_g"]
    return 0.5 + math.sqrt(0.25 + 2 * beta), gp["alpha_g"], beta


def solve_set(Ucols, r, m, s, alpha, guess=None, h=H_N):
    Ut = np.array(Ucols).T.copy()
    n = Ut.shape[1]
    fun = lambda E: jost(E, Ut, r, m, s, alpha, h)[0]  # noqa: E731
    if guess is None:
        x0, x1, x2 = (np.full(n, 4.30 - 0.002j), np.full(n, 4.40 - 0.01j), np.full(n, 4.45 - 0.0005j))
    else:
        x0, x1, x2 = guess - 0.03, guess + 0.03, guess + 0.001j * 0
    E, it = muller(fun, x0, x1, x2)
    resid = np.abs(fun(E)) / np.maximum(np.abs(jost(E, Ut, r, m, s, alpha, h)[1]), 1e-300)
    return E, it, resid


# ------------------------------------------------------------------ check 2
PROD = {("MU", -0.004): 9.80e-7, ("MU", -0.006): 4.68e-5, ("MU", -0.008): 4.57e-4, ("MU", -0.010): 2.051e-3,
        ("MU", -0.012): 5.87e-3, ("MU", -0.0143): 1.390e-2, ("MU", -0.016): 2.24e-2, ("MU", -0.020): 4.98e-2,
        ("M0", -0.0143): 0.01917044614, ("M0", -0.010): 0.002920102916, ("M0", -0.020): 0.06573984,
        ("M0", -0.005): 1.27745872e-5, ("L0", -0.0143): 3.237181e-4, ("L0", -0.010): 6.0844953e-6,
        ("LQ", -0.0143): 0.01020686, ("LQ", -0.010): 0.00146751, ("ML", -0.010): 1.433e-3,
        ("ML", -0.0143): 9.95e-3}
PROD_Q = {("M0", -0.0143): 4.339945141, ("MU", -0.004): 4.4191, ("MU", -0.010): 4.3890,
          ("MU", -0.0143): 4.3626, ("MU", -0.020): 4.3286}


def check2():
    t0 = time.time()
    gp = gauss_profile(V11["g2"], V11["gY"], check_global=True)
    s, alpha, beta = spar(gp)
    res = dict(gauss=dict(beta_total=beta, s=s, alpha=alpha, maxdec=gp["maxdec"], min_hess_eig=gp["min_hess_eig"],
                          march_vs_fine=gp["march_vs_fine"], global_minus_march=gp["global_minus_march"],
                          Etot_r0p01=float(gp["Et"][3]), Etot0_analytic=27 / 8 * (2 * V11["g2"] ** 2) ** (1 / 3)))
    # lambda=0 local ground state (Dirichlet at r=14)
    r, mD = grid(14.0)
    Ub = U_radial(gp, "L0", 0.0, V11["g2"], V11["gY"], V11["yt"], r)
    Utb = Ub[:, None].copy()

    def gD(E):
        return float(jost(np.array([E + 0j]), Utb, r, mD, s, np.array([alpha]))[1][0].real)
    Es = np.linspace(4.2, 5.0, 17)
    vals = [gD(E) for E in Es]
    i = next(j for j in range(len(Es) - 1) if vals[j] * vals[j + 1] < 0)
    Q0 = brentq(gD, Es[i], Es[i + 1], xtol=1e-13)
    res["lam0_local_Q"] = dict(R14=Q0, D108=4.5985676, Codex=4.598545)
    # resonances
    keys = list(PROD.keys())
    r, m = grid(16.0)
    cols = [U_radial(gp, mod, lam, V11["g2"], V11["gY"], V11["yt"], r) for (mod, lam) in keys]
    E, it, rs = solve_set(cols, r, m, s, alpha)
    tab = {}
    for j, key in enumerate(keys):
        g = -2 * E[j].imag
        tab[f"{key[0]} {key[1]}"] = dict(Q=float(E[j].real), gamma=float(g), D108=PROD[key],
                                         ratio=float(g / PROD[key]), D108_Q=PROD_Q.get(key),
                                         resid=float(rs[j]))
    res["table_v11_couplings"] = tab
    res["muller_iter"] = int(it)
    # convergence: r_out=13, and h/2
    sub = [("MU", -0.010), ("MU", -0.008), ("M0", -0.0143), ("MU", -0.0143)]
    r2, m2 = grid(13.0)
    cols2 = [U_radial(gp, mod, lam, V11["g2"], V11["gY"], V11["yt"], r2) for (mod, lam) in sub]
    E2, _, _ = solve_set(cols2, r2, m2, s, alpha)
    r3, m3 = grid(16.0, h=H_N / 2)
    cols3 = [U_radial(gp, mod, lam, V11["g2"], V11["gY"], V11["yt"], r3) for (mod, lam) in sub]
    E3, _, _ = solve_set(cols3, r3, m3, s, alpha, h=H_N / 2)
    conv = {}
    for j, key in enumerate(sub):
        g1 = -2 * E[keys.index(key)].imag
        conv[f"{key[0]} {key[1]}"] = dict(rout13_rel=float(-2 * E2[j].imag / g1 - 1),
                                          h_half_rel=float(-2 * E3[j].imag / g1 - 1),
                                          dQ_h_half=float(E3[j].real - E[keys.index(key)].real))
    res["convergence"] = conv
    # SM couplings at mu_b (2-loop, 172.60/0.1180) at lambda=-0.010: coupling sensitivity of gamma
    c = R14_rge.couplings(172.60, 0.1180)
    gps = gauss_profile(c["g2"], c["gY"])
    ss, als, _ = spar(gps)
    colsS = [U_radial(gps, "MU", -0.010, c["g2"], c["gY"], c["yt"], r),
             U_radial(gps, "MU", -0.010, c["g2"], c["gY"], V11["yt"], r),
             U_radial(gp, "MU", -0.010, V11["g2"], V11["gY"], c["yt"], r)]
    ES, _, _ = solve_set(colsS, r, m, ss, als)
    g_v11 = -2 * E[keys.index(("MU", -0.010))].imag
    res["coupling_effect_lam-0.010"] = dict(
        gamma_SMcouplings=float(-2 * ES[0].imag), ratio_all=float(-2 * ES[0].imag / g_v11),
        ratio_g_only=float(-2 * ES[1].imag / g_v11),
        ratio_yt_only_with_v11_gauss=float(-2 * ES[2].imag / g_v11), couplings=c)
    res["time_s"] = time.time() - t0
    merge_json("results.json", "check2_resonance", res)
    merge_json("STATUS.json", "check2", dict(done=True, MU_m0p010=tab["MU -0.01"]["gamma"],
                                             MU_m0p008=tab["MU -0.008"]["gamma"],
                                             MU_m0p0143=tab["MU -0.0143"]["gamma"], lam0_Q=Q0))
    return res


# ------------------------------------------------------------------ check 3
def eta_rem(u):
    f = lambda v: math.sqrt(2.0) * math.exp(-v) / math.sqrt(1.0 + math.exp(-4 * v))  # = 1/sqrt(cosh 2v)  # noqa: E731
    return quad(f, u, np.inf, epsabs=1e-13, epsrel=1e-12)[0] / HLB


def check3():
    t0 = time.time()
    cal, offs = R14_rge.calibration()
    un = np.round(np.arange(0, 4.0001, 0.25), 6)
    mus = R14_rge.MU_B / np.sqrt(np.cosh(2 * un))
    combos = [(mt, 0.1180) for mt in (171.8, 172.0, 172.2, 172.4, 172.5, 172.6, 172.7, 172.8, 173.0)] + \
             [(172.6, 0.1171), (172.6, 0.1189)]
    Y = {}
    for cb in combos:
        sol = R14_rge.solve(cb[0], cb[1], 125.13)
        Y[cb] = sol.sol(np.log(mus))
    Yc = Y[(172.6, 0.1180)]
    profs = [gauss_profile(float(Yc[1, i]), float(Yc[0, i])) for i in range(len(un))]
    r, m = grid(16.0)
    cols, meta, sl, al = [], [], [], []
    for cb in combos:
        for i, u in enumerate(un):
            for with_a in ((True, False) if cb == (172.6, 0.1180) else (True,)):
                at = HLB ** 2 * math.cosh(2 * u) if with_a else 0.0
                lam = float(Y[cb][4, i] + cal)
                cols.append(U_radial(profs[i], "MU", lam, float(Yc[1, i]), float(Yc[0, i]), float(Y[cb][3, i]),
                                     r, at))
                s_, a_, _ = spar(profs[i])
                sl.append(s_)
                al.append(a_)
                meta.append((cb, i, with_a, lam))
    E, it, rs = solve_set(cols, r, m, np.array(sl), np.array(al))
    gam = -2 * E.imag
    out = dict(cal=cal, u_nodes=un.tolist(), mu_nodes=mus.tolist(), muller_iter=int(it),
               max_resid=float(np.max(rs)), Qrange=[float(E.real.min()), float(E.real.max())])
    er = {u: eta_rem(u) for u in (0.0, 3.0, 4.0)}
    out["eta_rem"] = {str(k): v for k, v in er.items()}
    uf = np.linspace(0, 4.0, 8001)
    wf = 1.0 / (HLB * np.sqrt(np.cosh(2 * uf)))
    P = {}
    for cb in combos:
        for with_a in ((True, False) if cb == (172.6, 0.1180) else (True,)):
            idx = [j for j, mm in enumerate(meta) if mm[0] == cb and mm[2] == with_a]
            g = gam[idx]
            lams = [meta[j][3] for j in idx]
            spl = CubicSpline(un, np.log(g))
            gf = np.exp(spl(uf))
            sel3 = uf <= 3.0 + 1e-12
            I3 = simpson(gf[sel3] * wf[sel3], x=uf[sel3]) + g[un.tolist().index(3.0)] * er[3.0]
            I4 = simpson(gf * wf, x=uf) + g[-1] * er[4.0]
            Ist = g[0] * ETA_TOT
            P[f"Mt={cb[0]},as={cb[1]},aterm={with_a}"] = dict(
                lam_mu_b=lams[0], gamma_nodes=g.tolist(), lam_nodes=lams,
                I_static=float(Ist), P1_static=float(math.exp(-Ist)),
                I_u3_frozen=float(I3), P1_u3_frozen=float(math.exp(-I3)), P2_u3=float(math.exp(-2 * I3)),
                I_u4_frozen=float(I4), P1_u4_frozen=float(math.exp(-I4)))
    out["P"] = P
    # critical masses (alpha_s = 0.1180, with a''/a)
    mts = np.array([cb[0] for cb in combos if cb[1] == 0.1180])
    crit = {}
    for tag in ("I_u3_frozen", "I_u4_frozen", "I_static"):
        I = np.array([P[f"Mt={mt},as=0.118,aterm=True"][tag] for mt in mts])
        sp_ = CubicSpline(mts, np.log(I))
        cc = {}
        for name, target in (("P1=0.9", -math.log(0.9)), ("P1=0.5", math.log(2.0)),
                             ("P2=0.9", -math.log(0.9) / 2), ("P2=0.5", math.log(2.0) / 2)):
            try:
                cc[name] = float(brentq(lambda x: sp_(x) - math.log(target), mts[0], mts[-1]))
            except ValueError:
                cc[name] = None
        crit[tag] = cc
    out["critical_mt"] = crit
    out["time_s"] = time.time() - t0
    merge_json("results.json", "check3_hazard", out)
    k0 = "Mt=172.6,as=0.118,aterm=True"
    merge_json("STATUS.json", "check3", dict(done=True, P1_u3=P[k0]["P1_u3_frozen"], P1_u4=P[k0]["P1_u4_frozen"],
                                             P1_static=P[k0]["P1_static"], crit=crit["I_u3_frozen"]))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("bench", "all"):
        r2 = check2()
        print(json.dumps({k: v for k, v in r2.items()}, indent=1, default=float))
    if mode in ("hazard", "all"):
        r3 = check3()
        print(json.dumps(r3, indent=1, default=float))
