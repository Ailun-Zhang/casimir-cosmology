# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C138_core.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/C138_core.py on 2026-10-14 by agent holonomy-cross (C149/D121).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_core.py on 2026-10-13 by agent higgs-beyond-bo (C140/D112).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Consistent one-loop finite-volume matching (D110): Coulomb-gauge spectrum (2 transverse + 1 Coulomb mode per massive vector, Goldstone mixing), MS-bar finite-volume sums sum[V_CW^inf + D_theta] at mu = 1/L; model 'MC' (+CW) is the production matching dV_MC(r); MU/ML are round-1 regressions.  Paper: PRD App. H1 (matching potential). Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C138 core -- DNEC v12 round 2, agent higgs-matching (D110): consistent one-loop finite-volume matching.

Builds on the R14 independent implementation (R14_fv / R14_res / R14_rge copied UNCHANGED next to this file,
provenance line added).  Units: L = 1 (energies in 1/L, field r in 1/L, momenta k = 2 pi n).

Per real d.o.f. the MS-bar-renormalised finite-volume one-loop energy of a periodic scalar is (H1)
    E_std(m2; mu) = D_P(m2) + V_CW(m2; 3/2, mu),   V_CW(m2; c, mu) = m2^2 [ln(|m2|/mu^2) - c] / (64 pi^2),
D_P = (k!=0 sum) - continuum (R14_fv); the -m2^2 ln|m2|/(64pi^2) of D_P cancels the CW log => E_std analytic.
Vectors (MS-bar, 2-2eps transverse + 1 Coulomb/longitudinal polarisation): 2 x c=1/2 + 1 x c=3/2 = 3 x c=5/6.
Coulomb-gauge (MC) physical modes per massive vector V with Goldstone mass b = m_G^2 = V'_tree/r = lam r^2:
    2 transverse  w^2 = k^2 + a,    1 Coulomb mode  w_C^2 = (k^2 + a)(k^2 + b)/k^2,   a = m_V^2.
MS-bar FV energy of the Coulomb mode:  E_C(a, b) = E_std(a) + Delta(a, b),
    Delta(a, b) = R(a, b) + E_std(b) + (2b/a) E_std(a) - (b/a) E_std(2a),
    R(a, b) = sum_{k!=0} 1/2 [w_C - w_a - w_b - (2b/a) w_a + (b/a) w_2a + (1 + b/a) k]   (absolutely convergent,
    summand ~ (5 a^2 b - a b^2)/(32 k^5); the subtraction scalars reproduce the k^-1 and k^-3 terms of w_C - w_a).
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
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import R14_fv  # noqa: E402
import R14_res as R  # noqa: E402
import R14_rge  # noqa: E402

PI = math.pi
FVF = R.FVF
TBH = float(FVF.a[0][1])                 # T_B/2 = Z_P(1/2)/(8 pi) = -0.11289...
K2 = 4 * PI ** 2 * FVF.qP                 # shells |k|^2, k = 2 pi n, |n| <= RL
KK = np.sqrt(K2)
NQ = FVF.NP
TAILF = (2 * PI) ** -5 * 2 * PI / R14_fv.RL ** 2      # sum_{|n|>RL} |2 pi n|^-5 (continuum estimate)
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r2/higgs_matching/run01"))
SCR = _zp.w("scratch")  # [v12-package] was a session scratch directory (profile cache)


def vcw(m2, c, mu2=1.0):
    m2 = np.atleast_1d(np.asarray(m2, float))
    out = np.zeros_like(m2)
    nz = m2 != 0
    out[nz] = m2[nz] ** 2 * (np.log(np.abs(m2[nz]) / mu2) - c) / (64 * PI ** 2)
    return out


def DP(m2):
    return FVF.DP(np.atleast_1d(np.asarray(m2, float)))


def DA(m2):
    return FVF.DA(np.atleast_1d(np.asarray(m2, float)))


def estd(m2, mu2=1.0):
    return DP(m2) + vcw(m2, 1.5, mu2)


def R_delta(a, b, chunk=300):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    out = np.empty(len(a))
    for st in range(0, len(a), chunk):
        A = a[st:st + chunk, None]
        B = b[st:st + chunk, None]
        wa = np.sqrt(K2 + A)
        wb = np.sqrt(K2 + B)
        w2 = np.sqrt(K2 + 2 * A)
        wc = np.sqrt((K2 + A) * (K2 + B)) / KK
        dC = (A + B + A * B / K2) / (wc + KK)
        da = A / (wa + KK)
        db = B / (wb + KK)
        d2 = 2 * A / (w2 + KK)
        rho = B / A
        s = dC - da - db - 2 * rho * da + rho * d2
        A1, B1 = A[:, 0], B[:, 0]
        out[st:st + chunk] = 0.5 * (s * NQ).sum(1) + 0.5 * (5 * A1 ** 2 * B1 - A1 * B1 ** 2) / 16 * TAILF
    return out


def delta(a, b, mu2=1.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    rho = b / a
    return R_delta(a, b) + estd(b, mu2) + 2 * rho * estd(a, mu2) - rho * estd(2 * a, mu2)


def ddelta_db0(a, mu2=1.0, chunk=300):
    """d Delta / d b at b = 0 (Coulomb-weighted FV Goldstone tadpole, = T_B/2 at a -> 0)."""
    a = np.atleast_1d(np.asarray(a, float))
    S = np.empty(len(a))
    for st in range(0, len(a), chunk):
        A = a[st:st + chunk, None]
        wa = np.sqrt(K2 + A)
        w2 = np.sqrt(K2 + 2 * A)
        s = A / (2 * K2 * (wa + KK)) - 2 / (wa + KK) + 2 / (w2 + KK)
        S[st:st + chunk] = 0.5 * (s * NQ).sum(1) + 0.5 * 5 * A[:, 0] ** 2 / 16 * TAILF
    return S + TBH + (2 * estd(a, mu2) - estd(2 * a, mu2)) / a


def masses(r, lam, g2, gY, yt):
    r2 = r * r
    return dict(aW=g2 * g2 / 4 * r2, aZ=(g2 * g2 + gY * gY) / 4 * r2, mt2=yt * yt / 2 * r2, mh2=3 * lam * r2,
                b=lam * r2)


def C_shift(r, g2, gY, mu2=1.0):
    """rho_0 - |Phi_0| = sum_a <pi_a^2>_FV / (2r) (Coulomb-weighted, MS-bar FV), so that V'_tree C = MC1 - MU."""
    def f(x):
        m = masses(x, 0.0, g2, gY, 0.0)
        return 2 * ddelta_db0(m["aW"], mu2) + ddelta_db0(m["aZ"], mu2)
    return coarse(f, r) / np.atleast_1d(r)


MODELS = ("MU", "ML", "MU_CW", "ML_CW", "MC", "MC1", "MC_v32", "MUPS")


def coarse(fun, r, h=0.02):
    """evaluate a smooth function of r on a coarse grid and spline it onto r (used for the shell sums)."""
    r = np.atleast_1d(np.asarray(r, float))
    if len(r) < 2000:
        return fun(r)
    from scipy.interpolate import CubicSpline
    rc = np.arange(r.min(), r.max() + h, h)
    return CubicSpline(rc, fun(rc))(r)


def dV(model, r, lam, g2, gY, yt, mu2=1.0, yt_top=None, mu2_top=None):
    """one-loop FV matching energy (nonzero modes) as a function of the slow coordinate r (array)."""
    r = np.atleast_1d(np.asarray(r, float))
    ytt = yt if yt_top is None else yt_top
    mut = mu2 if mu2_top is None else mu2_top
    m = masses(r, lam, g2, gY, ytt)
    if model in ("MU", "ML"):                    # D108 / R14 regression (no infinite-volume CW)
        out = 6 * DP(m["aW"]) + 3 * DP(m["aZ"]) - 12 * DA(m["mt2"]) + DP(m["mh2"])
        return out + (3 * DP(m["b"]) if model == "ML" else 0.0)
    vec = 6 * (DP(m["aW"]) + vcw(m["aW"], 5.0 / 6, mu2)) + 3 * (DP(m["aZ"]) + vcw(m["aZ"], 5.0 / 6, mu2))
    top = -12 * (DA(m["mt2"]) + vcw(m["mt2"], 1.5, mut))
    base = vec + top + estd(m["mh2"], mu2)       # = MU_CW (H1-complete unitary counting)
    if model == "MU_CW":
        return base
    if model == "ML_CW":
        return base + 3 * estd(m["b"], mu2)
    if model in ("MC", "MC_v32"):
        def fC(x):
            mm = masses(x, lam, g2, gY, ytt)
            return 2 * delta(mm["aW"], mm["b"], mu2) + delta(mm["aZ"], mm["b"], mu2)
        out = base + coarse(fC, r)
        if model == "MC_v32":                    # inconsistent constant 3/2 for all vector polarisations
            out = out - 2 * (2 * m["aW"] ** 2 + m["aZ"] ** 2) / (64 * PI ** 2)
        return out
    if model == "MC1":
        def f1(x):
            mm = masses(x, lam, g2, gY, ytt)
            return 2 * ddelta_db0(mm["aW"], mu2) + ddelta_db0(mm["aZ"], mu2)
        return base + m["b"] * coarse(f1, r)
    raise ValueError(model)


def U_col(gp, model, lam, g2, gY, yt, r, mu2=1.0, xi=1.0, yt_top=None, mu2_top=None, Cr=None):
    """radial potential on grid r.  xi: Higgs field normalisation r_mu = xi r (gamma_phi); MUPS: MU_CW with the
    tree + matching part evaluated at rho = r + C(r) (gauge zero modes and centrifugal term kept at r)."""
    z = xi * r
    U = 3.0 / (8 * r * r) + gp["spl"](z) / (z * z)
    if model == "MUPS":
        rho = z + Cr * z ** 4 / (z ** 4 + 1.0)          # shift switched off below r ~ 1 (where rho_0 -> 0)
        return U + lam * rho ** 4 / 4 + dV("MU_CW", rho, lam, g2, gY, yt, mu2, yt_top, mu2_top)
    return U + lam * z ** 4 / 4 + dV(model, z, lam, g2, gY, yt, mu2, yt_top, mu2_top)


def spar_xi(gp, xi=1.0):
    beta = 3.0 / 8 + gp["beta_g"] / xi ** 2
    return 0.5 + math.sqrt(0.25 + 2 * beta), gp["alpha_g"] / xi


def gamma_phi(g2, gY, yt, gauge="C"):
    """one-loop Higgs field anomalous dimension; 'L' Landau, 'C' Coulomb (= Feynman at one loop, derived in D110)."""
    if gauge == "L":
        return (3 * yt * yt - 2.25 * g2 * g2 - 0.75 * gY * gY) / (16 * PI ** 2)
    return (3 * yt * yt - 1.5 * g2 * g2 - 0.5 * gY * gY) / (16 * PI ** 2)


def bo_corrections(r, g2, gY, yt, lam):
    """Born-Huang diagonal term sum (1/16)(w'/w)^2 and BO induced mass sum (w'/w)^2/(8w) over the MC fast modes
    (bosons; top estimated with the same formula), w' = d/dr.  Coulomb mode uses Z w = sqrt((k^2+b)k^2/(k^2+a))."""
    out = {}
    for rr in np.atleast_1d(r):
        m = masses(rr, lam, g2, gY, yt)
        bh = dm = 0.0
        for (n, key) in ((4, "aW"), (2, "aZ"), (12, "mt2"), (1, "mh2")):     # transverse (2 per W, x2; 2 Z) etc.
            a = m[key]
            w2 = K2 + a
            dlw = (a / rr) / w2                    # d ln w / dr = (da/dr)/(2 w^2), da/dr = 2a/r
            bh += n * np.sum(NQ * dlw ** 2) / 16
            dm += n * np.sum(NQ * dlw ** 2 / (8 * np.sqrt(w2)))
        for key in ("aW", "aW", "aZ"):              # Coulomb modes: oscillator frequency w_C, width Z w_C
            a, b = m[key], m["b"]
            dlzw = (b / rr) / (K2 + b) - (a / rr) / (K2 + a)   # d ln (Z w_C)/dr, Z w_C = k sqrt(k^2+b)/sqrt(k^2+a)
            wc = np.sqrt((K2 + a) * (K2 + b)) / KK
            bh += np.sum(NQ * dlzw ** 2) / 16
            dm += np.sum(NQ * dlzw ** 2 / (8 * wc))
        out[float(rr)] = dict(E_BH=float(bh), dM_induced=float(dm))
    return out
