# PROVENANCE: copied from D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\keystone_blind\work\oneloop.py on 2026-10-15 by ui-blind (unmodified below this line)
# keystone-blind (T3) : MS-bar finite-volume one-loop energies of nonzero modes on the unit cubic torus (L=1, mu=1/L).
# All functions return (1/2) sum'_k omega_k per real mode (bosons) in MS-bar, finite-volume representation
#   F = CW_inf(MS-bar, mu=1) + [discrete nonzero-mode sum - continuum integral],
# implemented through an analytic proper-time representation valid for complex/negative m^2 (m^2 > -4 pi^2).
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.special import exp1, erfcx, erf, kv
from numpy.polynomial.legendre import leggauss
from scipy.integrate import trapezoid as _trap

PI = np.pi
T0 = 1.0 / (4.0 * PI)
EG = np.euler_gamma

def _lattice(nmax=6, shift=0.0):
    r = np.arange(-nmax, nmax + 1) + shift
    X, Y, Z = np.meshgrid(r, r, r, indexing="ij")
    return X.ravel(), Y.ravel(), Z.ravel()

# periodic nonzero lattice n (for direct sums, k=2 pi n), and windings l for Poisson side
_x, _y, _z = _lattice(6)
_n2 = _x**2 + _y**2 + _z**2
_P_N2 = _n2[_n2 > 0]                         # nonzero periodic |n|^2
_L2 = _P_N2.copy()                            # windings |l|^2 (nonzero)
_LSIGN = ((-1.0) ** (_x + _y + _z))[_n2 > 0]  # (-1)^{l1+l2+l3} for AAA
_xa, _ya, _za = _lattice(6, 0.5)
_A_N2 = _xa**2 + _ya**2 + _za**2              # AAA |n|^2 (half-integers)
# compress into shells
def _shells(v, w=None):
    v = np.round(v * 4).astype(np.int64)
    if w is None: w = np.ones_like(v, dtype=float)
    u, inv = np.unique(v, return_inverse=True)
    m = np.zeros(len(u)); np.add.at(m, inv, w)
    return u / 4.0, m
P_SH, P_MULT = _shells(_P_N2)
P_SH, P_MULT = P_SH[P_SH <= 30], P_MULT[P_SH <= 30]
L_SH, L_MULT = P_SH, P_MULT
LA_SH, LA_MULT = _shells(_L2, _LSIGN); LA_SH, LA_MULT = LA_SH[LA_SH <= 30], LA_MULT[LA_SH <= 30]
A_SH, A_MULT = _shells(_A_N2); A_SH, A_MULT = A_SH[A_SH <= 30], A_MULT[A_SH <= 30]

_gx, _gw = leggauss(80)

def _R(m2):
    """MS-bar[int_0^T0 t^-3 e^{-t m2} dt], entire in m2 (mu=1)."""
    m2 = np.asarray(m2, dtype=complex)
    x = m2 * T0
    out = np.empty_like(m2)
    small = np.abs(x) <= 4.0
    xs = x[small]
    ser = np.zeros_like(xs)
    import math
    for j in range(3, 80):
        ser = ser + ((-1.0) ** j) * xs ** (j - 2) / (math.factorial(j) * (j - 2))
    m2s = m2[small]
    out[small] = m2s**2 * ser - 1.0 / (2 * T0**2) + m2s / T0 + 0.5 * m2s**2 * (np.log(T0) + EG)
    big = ~small
    if np.any(big):
        mb = m2[big].real; xb = mb * T0
        g2 = 0.5 * (exp1(xb) + np.exp(-xb) * (1.0 / xb**2 - 1.0 / xb))
        out[big] = -(mb**2 / 2) * (np.log(mb) - 1.5) - mb**2 * g2
    return out

def _W(m2, shells, mult):
    """int_0^T0 t^-3 e^{-t m2} sum_l w_l e^{-l^2/(4t)} dt."""
    m2 = np.asarray(m2, dtype=complex)
    t = 0.5 * T0 * (_gx + 1.0); w = 0.5 * T0 * _gw
    lsum = (mult[None, :] * np.exp(-shells[None, :] / (4 * t[:, None]))).sum(1)   # (nt,)
    f = t**-3 * lsum * w
    return (np.exp(-np.outer(m2, t)) * f[None, :]).sum(1)

def _Z0(m2):
    """dim-reg int_0^T0 t^-3/2 e^{-t m2} dt."""
    m2 = np.asarray(m2, dtype=complex)
    tau = 0.5 * np.sqrt(T0) * (_gx + 1.0); w = 0.5 * np.sqrt(T0) * _gw
    J = 2.0 * (np.exp(-np.outer(m2, tau**2)) * w[None, :]).sum(1)
    return -2.0 / np.sqrt(T0) * np.exp(-m2 * T0) - 2.0 * m2 * J

def _Igt(m2, shells, mult):
    """int_T0^inf t^-3/2 e^{-t m2} sum_n e^{-4 pi^2 n^2 t} dt (closed form per shell)."""
    m2 = np.asarray(m2, dtype=complex)
    a = m2[:, None] + 4 * PI**2 * shells[None, :]
    sa = np.sqrt(a * T0)
    if np.all(np.abs(a.imag) < 1e-300):
        ar = a.real; sr = np.sqrt(ar * T0)
        val = 2.0 / np.sqrt(T0) * np.exp(-ar * T0) - 2.0 * np.sqrt(PI * ar) * erfcx(sr) * np.exp(-ar * T0)
    else:
        from scipy.special import erfc
        val = 2.0 / np.sqrt(T0) * np.exp(-a * T0) - 2.0 * np.sqrt(PI * a) * erfc(sa)
    return (val * mult[None, :]).sum(1)

def FP(m2, C=1.5):
    """(1/2) sum_{k != 0} sqrt(k^2+m2), periodic, MS-bar with constant C (3/2 scalar, 1/2 transverse)."""
    m2a = np.atleast_1d(np.asarray(m2, dtype=complex))
    val = (-(1 / (32 * PI**2)) * (_R(m2a) + _W(m2a, L_SH, L_MULT)) + _Z0(m2a) / (4 * np.sqrt(PI))
           - _Igt(m2a, P_SH, P_MULT) / (4 * np.sqrt(PI)))
    val = val + (1.5 - C) * m2a**2 / (64 * PI**2)
    return val.real if np.all(np.abs(m2a.imag) == 0) else val

def FA(m2):
    """(1/2) sum_{k in 2pi(Z+1/2)^3} sqrt(k^2+m2), MS-bar C=3/2 (per real fermionic dof, sign NOT included)."""
    m2a = np.atleast_1d(np.asarray(m2, dtype=complex))
    val = (-(1 / (32 * PI**2)) * (_R(m2a) + _W(m2a, LA_SH, LA_MULT)) - _Igt(m2a, A_SH, A_MULT) / (4 * np.sqrt(PI)))
    return val.real if np.all(np.abs(m2a.imag) == 0) else val

# ---- Coulomb (longitudinal-Goldstone) mode: omega^2 = (k^2+A)(k^2+B)/k^2, A=m_G^2, B=m_V^2 ----
_ulog = np.linspace(np.log(1e-9), np.log(4.0), 3000)

def _bracket(u):
    """Theta(u) - 1 - (4 pi u)^{-3/2}, Theta(u)=sum_{n in Z^3} e^{-4pi^2 n^2 u}."""
    out = np.empty_like(u)
    sm = u < 0.08
    us = u[sm]
    th = 1.0 + 2.0 * sum(np.exp(-l * l / (4 * us)) for l in range(1, 8))
    out[sm] = (4 * PI * us) ** -1.5 * (th**3 - 1.0) - 1.0
    ub = u[~sm]
    th3 = 1.0 + 2.0 * sum(np.exp(-4 * PI**2 * n * n * ub) for n in range(1, 8))
    out[~sm] = (th3**3 - 1.0) - (4 * PI * ub) ** -1.5
    return out

def DT1(S):
    """(1/4)[sum'_k - int d^3k/(2pi)^3] k^-2 (k^2+S)^-1/2 for S>0 (vectorised)."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    res = np.empty_like(S)
    for i, s in enumerate(S):
        U1 = max(60.0 / s, 4.0)
        v = np.linspace(np.log(1e-9), np.log(U1), 4000)
        u = np.exp(v)
        wfun = erf(np.sqrt(u * s)) / np.sqrt(s)
        f = wfun * _bracket(u) * u
        integ = _trap(f, v)
        # tail u>U1: bracket = -(4 pi u)^-3/2 (+ e^{-4pi^2 U1} negligible), erf=1
        tail = -(1 / np.sqrt(s)) * (4 * PI) ** -1.5 * 2.0 / np.sqrt(U1)
        res[i] = 0.25 * (integ + tail)
    return res

def T1(S):
    S = np.atleast_1d(np.asarray(S, dtype=float))
    return (2.0 - np.log(S)) / (16 * PI**2) + DT1(S)

_KS = 4 * PI**2 * P_SH   # k^2 shells (|n|^2<=30)
# direct-sum shells for the convergent remainder: use bigger lattice
_xx, _yy, _zz = _lattice(14)
_nn = _xx**2 + _yy**2 + _zz**2
_RS, _RM = _shells(_nn[_nn > 0].astype(float))
_RS, _RM = _RS[_RS <= 196], _RM[_RS <= 196]

def RC(A, B):
    """(1/2) sum'_k [omega_C - omega_S - P/(2 k^2 omega_S)], convergent; direct sum |n|^2<=196 + k^-7 tail."""
    A = np.atleast_1d(np.asarray(A, dtype=float)); B = np.atleast_1d(np.asarray(B, dtype=float))
    S = (A + B)[:, None]; P = (A * B)[:, None]
    k2 = 4 * PI**2 * _RS[None, :]
    wS = np.sqrt(k2 + S); wC = np.sqrt(k2 + S + P / k2)
    summ = -(P**2 / k2**2) / (2 * wS * (wC + wS) ** 2)
    val = 0.5 * (summ * _RM[None, :]).sum(1)
    # tail: summand ~ -P^2/(8 k^7); sum_{|n|>14} ~ 4 pi int_14^inf n^2 dn (2 pi n)^-7
    tail = 0.5 * (-(P[:, 0] ** 2) / 8.0) * 4 * PI * (2 * PI) ** -7 / (4 * 14.0**4)
    return val + tail

def FC(A, B):
    A = np.atleast_1d(np.asarray(A, dtype=float)); B = np.atleast_1d(np.asarray(B, dtype=float))
    S = A + B; P = A * B
    return FP(S) + P * T1(S) + RC(A, B)

def EC_inf_msbar(A, B):
    """closed-form infinite-volume MS-bar energy of the Coulomb mode (A,B>=0), mu=1."""
    sA, sB = np.sqrt(A + 0j), np.sqrt(B + 0j)
    return ((A + B) ** 2 - 2 * (A + B) * sA * sB + (A - B) ** 2 * (np.log((sA + sB) ** 2) - 2.5)) / (64 * PI**2)

# ---------------- full one-loop matching potential -----------------
def V1loop(r, lam, g2, gY, yt, subtract0=True, parts=False):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    mW2 = g2**2 * r**2 / 4; mZ2 = (g2**2 + gY**2) * r**2 / 4
    mG2 = lam * r**2; mh2 = 3 * lam * r**2; mt2 = yt**2 * r**2 / 2
    W = 2 * (2 * FP(mW2, C=0.5) + FC(mG2, mW2))
    Z = 2 * FP(mZ2, C=0.5) + FC(mG2, mZ2)
    H = FP(mh2)
    T = -12 * FA(mt2)
    tot = W + Z + H + T
    if subtract0:
        z0 = 3 * (3 * FP(np.array([0.0]))[0]) + FP(np.array([0.0]))[0] - 12 * FA(np.array([0.0]))[0]
        tot = tot - z0
    if parts:
        return tot, dict(W=W, Z=Z, H=H, T=T)
    return tot

def V1_const0():
    return 3 * (3 * FP(np.array([0.0]))[0]) + FP(np.array([0.0]))[0] - 12 * FA(np.array([0.0]))[0]

# ---------------- self tests -----------------
def _bessel_FP(m2, C=1.5, lmax=14):
    m = np.sqrt(m2)
    x, y, z = _lattice(lmax); l2 = x**2 + y**2 + z**2; l = np.sqrt(l2[l2 > 0])
    D = -(m2 / (4 * PI**2)) * np.sum(kv(2, m * l) / l**2)
    return m2**2 / (64 * PI**2) * (np.log(m2) - C) + D - m / 2

def _bessel_FA(m2, lmax=14):
    m = np.sqrt(m2)
    x, y, z = _lattice(lmax); l2 = x**2 + y**2 + z**2; sgn = (-1.0) ** (x + y + z)
    msk = l2 > 0; l = np.sqrt(l2[msk])
    D = -(m2 / (4 * PI**2)) * np.sum(sgn[msk] * kv(2, m * l) / l**2)
    return m2**2 / (64 * PI**2) * (np.log(m2) - 1.5) + D

if __name__ == "__main__":
    import sys, json
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for m2 in [1.0, 4.0, 16.0, 60.0]:
        a, b = FP(np.array([m2]))[0], _bessel_FP(m2)
        c, d = FA(np.array([m2]))[0], _bessel_FA(m2)
        print(f"m2={m2:6.1f} FP proper={a:.12f} bessel={b:.12f} diff={a-b:.2e} | FA proper={c:.12f} bessel={d:.12f} diff={c-d:.2e}")
        out[f"FP_vs_bessel_m2_{m2}"] = a - b; out[f"FA_vs_bessel_m2_{m2}"] = c - d
    h = 1e-4
    dP = (FP(np.array([h]))[0] - FP(np.array([-h]))[0]) / (2 * h)
    dA = (FA(np.array([h]))[0] - FA(np.array([-h]))[0]) / (2 * h)
    print("tadpole dFP/dm2(0) =", dP, " expected Z(1/2)/(8pi) =", -2.8372974794806 / (8 * PI))
    print("tadpole dFA/dm2(0) =", dA, " expected -2.519356/(8pi^2) =", -2.519356 / (8 * PI**2))
    print("FP(0) =", FP(np.array([0.0]))[0], " expected -Z(2)/(2pi^2) =", -16.532315959761669 / (2 * PI**2))
    out.update(tadP=dP, tadA=dA, FP0=FP(np.array([0.0]))[0], FA0=FA(np.array([0.0]))[0])
    # negative m2 continuity / analyticity: compare with Taylor from small positive side
    for m2 in [-1.0, -5.0, -20.0]:
        print("FP(neg)", m2, FP(np.array([m2]))[0])
    # Coulomb checks: (i) A=B -> F_P(0) + A Z(1/2)/(4 pi), linear in A
    for A in [0.3, 2.0, 8.0]:
        fc = FC(np.array([A]), np.array([A]))[0]
        ex = FP(np.array([0.0]))[0] + A * (-2.8372974794806) / (4 * PI)
        print(f"FC(A=B={A}) = {fc:.12f}  exact = {ex:.12f}  diff={fc-ex:.2e}")
        out[f"FC_AeqB_{A}"] = fc - ex
    # (ii) A=0 -> FP(B)
    print("FC(0,3)-FP(3) =", FC(np.array([0.0]), np.array([3.0]))[0] - FP(np.array([3.0]))[0])
    # (iii) negative A small: smoothness
    for A in [-0.5, -0.1, 0.0, 0.1]:
        print("FC(A,2.0)", A, FC(np.array([A]), np.array([2.0]))[0])
    # (iv) infinite volume closed form expansion consistency: d/dP at P=0 equals (2-ln S)/(16pi^2)
    S = 3.0; eps = 1e-4
    A1 = 0.5 * (S - np.sqrt(S * S - 4 * eps)); B1 = S - A1
    print("EC_inf P-derivative", (EC_inf_msbar(A1, B1) - EC_inf_msbar(0, S)).real / eps, (2 - np.log(S)) / (16 * PI**2))
    # small-r expansion of V1 vs v11 tadpole formula
    g2, gY, yt, lam = 0.5268, 0.4415, 0.4585, 0.01
    r = np.array([1e-2, 2e-2])
    v = V1loop(r, lam, g2, gY, yt)
    TB = -8.913633 / (4 * PI**2); TF = -2.519356 / (4 * PI**2)
    mphi2 = (9 / 4 * g2**2 + 3 / 4 * gY**2 + 6 * lam) * TB - 6 * yt**2 * TF
    print("V1(r)/(r^2/2) at small r:", v / (r**2 / 2), " v11 m_Phi^2 =", mphi2)
    out["mPhi2_numeric"] = float(v[0] / (r[0]**2 / 2)); out["mPhi2_v11"] = mphi2
    rr = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15])
    for lam in [0.0, -0.01]:
        vv, parts = V1loop(rr, lam, g2, gY, yt, parts=True)
        print("lam", lam, "V1:", np.array2string(vv, precision=5))
    print("V1 const at r=0 (not included):", V1_const0())
    json.dump({k: float(np.real(v)) for k, v in out.items()}, open(os.path.join(os.path.dirname(__file__), "..", "results", "oneloop_selftest.json"), "w"), indent=1)
