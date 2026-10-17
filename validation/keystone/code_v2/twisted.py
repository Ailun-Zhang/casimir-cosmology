# keystone-blind V2: direction-averaged twisted finite-volume one-loop functions on the unit cubic torus.
# Per REAL degree of freedom: f(m2, th) = (1/2) sum_n omega(2 pi (n + theta)), MS-bar, averaged over the direction of theta
# (|theta| = th), periodic: n = 0 (slow mode) excluded; AAA: n in (Z+1/2)^3.  Proper-time split at T0 = 1/(4 pi):
#   f = -(1/32pi^2)[R(m2) + W(m2,th)] + Z0(m2 + 4pi^2 th^2)/(4 sqrt(pi)) - I_>(m2,th)/(4 sqrt(pi))
#   W  = sum'_l c_l sinc(2 pi |l| th) int_0^T0 t^-3 e^{-t m2 - l^2/4t},     (direction average of cos(2 pi l.theta))
#   I_>= sum_n int_T0^inf t^-3/2 e^{-t m2} e^{-4pi^2(n^2+th^2)t} sinhc(8 pi^2 t |n| th)  (closed form via Gamma(-3/2,x))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
from scipy.special import erfcx, erf, kv
from numpy.polynomial.legendre import leggauss
import oneloop

PI = np.pi; T0 = 1.0/(4*PI); SQPI = np.sqrt(PI)

def _lat(nmax, shift=0.0):
    a = np.arange(-nmax, nmax + 1) + shift
    X, Y, Z = np.meshgrid(a, a, a, indexing="ij"); return X.ravel(), Y.ravel(), Z.ravel()
def _shells(n2, w, cut=40.0):
    key = np.round(n2*4).astype(np.int64); u, inv = np.unique(key, return_inverse=True)
    m = np.zeros(len(u)); np.add.at(m, inv, w); s = u/4.0
    k = (s <= cut) & (np.abs(m) > 0); return s[k], m[k]
_x, _y, _z = _lat(7); _n2 = _x*_x + _y*_y + _z*_z; _nz = _n2 > 0
P_SH, P_M = _shells(_n2[_nz], np.ones(_nz.sum()))                 # periodic n != 0 (direct) == winding shells
LA_SH, LA_M = _shells(_n2[_nz], ((-1.0)**(_x + _y + _z))[_nz])    # signed winding shells for AAA
_xa, _ya, _za = _lat(7, 0.5); A_SH, A_M = _shells(_xa**2 + _ya**2 + _za**2, np.ones(len(_xa)))

_gx, _gw = leggauss(80); _t = 0.5*T0*(_gx + 1); _wt = 0.5*T0*_gw
def _J(m2, L2):
    """J[i,s] = int_0^T0 t^-3 exp(-t m2_i - L2_s/(4t)) dt."""
    base = (_wt*_t**-3)[None, :]*np.exp(-L2[:, None]/(4*_t[None, :]))     # (S, nt)
    return np.exp(-np.outer(m2, _t)) @ base.T                           # (N, S)

def _sinc(x): return np.sinc(x/PI)

def _G(b):
    """int_T0^inf t^-5/2 e^{-b t} dt  (b>0)."""
    x = b*T0; sx = np.sqrt(x)
    return (2/3)*np.exp(-x)*(T0**-1.5 - 2*b*T0**-0.5 + 2*SQPI*b**1.5*erfcx(sx))

def _Hz(a):
    """int_T0^inf t^-3/2 e^{-a t} dt (a>0)."""
    x = a*T0
    return np.exp(-x)*(2/np.sqrt(T0) - 2*np.sqrt(PI*a)*erfcx(np.sqrt(x)))

def _Igt(m2, th, SH, M):
    n = np.sqrt(SH)[None, :]; m2c = m2[:, None]; thc = th[:, None]
    small = thc < 1e-7
    bm = m2c + 4*PI**2*(n - thc)**2; bp = m2c + 4*PI**2*(n + thc)**2
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (_G(bm) - _G(bp))/(16*PI**2*n*np.where(small, 1.0, thc))
    val0 = _Hz(m2c + 4*PI**2*n*n)
    val = np.where(small, val0, val)
    return (val*M[None, :]).sum(1)

def fP(m2, th):
    m2 = np.atleast_1d(np.asarray(m2, float)); th = np.broadcast_to(np.atleast_1d(np.asarray(th, float)), m2.shape).copy()
    W = (_J(m2, P_SH)*(P_M[None, :]*_sinc(2*PI*np.sqrt(P_SH)[None, :]*th[:, None]))).sum(1)
    R = oneloop._R(m2).real
    Z0 = oneloop._Z0(m2 + 4*PI**2*th**2).real
    return -(R + W)/(32*PI**2) + Z0/(4*SQPI) - _Igt(m2, th, P_SH, P_M)/(4*SQPI)

def fA(m2, th):
    m2 = np.atleast_1d(np.asarray(m2, float)); th = np.broadcast_to(np.atleast_1d(np.asarray(th, float)), m2.shape).copy()
    W = (_J(m2, LA_SH)*(LA_M[None, :]*_sinc(2*PI*np.sqrt(LA_SH)[None, :]*th[:, None]))).sum(1)
    R = oneloop._R(m2).real
    return -(R + W)/(32*PI**2) - _Igt(m2, th, A_SH, A_M)/(4*SQPI)

_uv = np.linspace(np.log(1e-9), np.log(60.0), 3000)
def dT1(S, th):
    """Delta T1(S,theta) = T1(S,theta) - T1(S,0), direction-averaged twisted (n=0 excluded)."""
    S = np.atleast_1d(np.asarray(S, float)); th = np.broadcast_to(np.atleast_1d(np.asarray(th, float)), S.shape)
    u = np.exp(_uv); out = np.zeros(len(S))
    sm = u < 0.08; us, ub = u[sm], u[~sm]
    for i, (s, t) in enumerate(zip(S, th)):
        if t == 0: continue
        dB = np.empty_like(u)
        ell = np.sqrt(P_SH)
        dB[sm] = (4*PI*us)**-1.5*((P_M[None, :]*np.exp(-P_SH[None, :]/(4*us[:, None])))*(_sinc(2*PI*ell*t)[None, :] - 1)).sum(1) - (np.exp(-4*PI**2*t*t*us) - 1)
        n = np.sqrt(P_SH)[None, :]; ubc = ub[:, None]
        two = (np.exp(-4*PI**2*(n - t)**2*ubc) - np.exp(-4*PI**2*(n + t)**2*ubc))/(16*PI**2*n*t*ubc)
        dB[~sm] = ((two - np.exp(-4*PI**2*n*n*ubc))*P_M[None, :]).sum(1)
        w = erf(np.sqrt(u*s))/np.sqrt(s)
        out[i] = 0.25*np.trapezoid(w*dB*u, _uv) if hasattr(np, "trapezoid") else 0.25*np.trapz(w*dB*u, _uv)
    return out

def kappa_q(fun, m2, d=0.01):
    """curvature per real dof per unit q^2 (theta = q/2pi), Richardson from d and 2d."""
    m2 = np.atleast_1d(np.asarray(m2, float)); z = np.zeros_like(m2)
    f0, f1, f2 = fun(m2, z), fun(m2, z + d), fun(m2, z + 2*d)
    A = (16*(f1 - f0) - (f2 - f0))/(12*d*d)
    return 2*A/(4*PI**2)

if __name__ == "__main__":
    import json
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    # (1) theta=0 reduces to T3 functions
    for m2 in [-5.0, 0.0, 3.0, 20.0]:
        print("theta=0 check m2=%g: fP-FP=%.2e fA-FA=%.2e" % (m2, fP(m2, 0.0)[0] - oneloop.FP(np.array([m2]))[0], fA(m2, 0.0)[0] - oneloop.FA(np.array([m2]))[0]))
    # (2) massive twisted vs Poisson-Bessel with sinc (full sum incl n=0)
    xs, ys, zs = _lat(16); l2 = (xs*xs + ys*ys + zs*zs).astype(float); msk = l2 > 0; l = np.sqrt(l2[msk]); sg = ((-1.0)**(xs + ys + zs))[msk]
    for m2 in [4.0, 9.0]:
        m = np.sqrt(m2); cw = m2*m2/(64*PI**2)*(np.log(m2) - 1.5)
        for th in [0.1, 0.3, 0.45]:
            DP = -(m2/(4*PI**2))*np.sum(_sinc(2*PI*l*th)*kv(2, m*l)/l**2)
            DA = -(m2/(4*PI**2))*np.sum(sg*_sinc(2*PI*l*th)*kv(2, m*l)/l**2)
            eP = fP(m2, th)[0] + 0.5*np.sqrt(4*PI**2*th*th + m2) - (cw + DP)
            eA = fA(m2, th)[0] - (cw + DA)
            print("m2=%g th=%.2f: periodic diff %.2e  AAA diff %.2e" % (m2, th, eP, eA)); out[f"bessel_{m2}_{th}"] = [eP, eA]
    # (3) massless curvatures per real dof per unit q^2
    kP, kA = kappa_q(fP, 0.0)[0], kappa_q(fA, 0.0)[0]
    print("kappa_P(0) = %.6f (Z(1/2)/6pi = %.6f), kappa_A(0) = %.6f (-2.519356/(6 pi^2) = %.6f)" % (kP, -2.8372974794806/(6*PI), kA, -2.519356/(6*PI**2)))
    # (4) SM curvatures at r=0 (coupling-absorbed c): per real dof units
    k3 = 6*kP + 12*(-kA)                                  # gluons 2 pol x tr_adj 3 ; 12 Weyl triplets x 2 rd x tr_f 1/2
    k2 = 4*kP + 4*0.25*kP + 3*16*0.25*(-kA)               # W: 2 pol x 2 rd x t3^2=1 ; Higgs 4 rd x 1/4 ; 16 rd/gen x 1/4
    Ysum = 2*(3*2*(1/6)**2 + 3*(2/3)**2 + 3*(1/3)**2 + 2*(1/2)**2 + 1.0)     # real dof x Y^2 per generation = 2 x 10/3
    kY = 4*0.25*kP + 3*Ysum*(-kA)
    print("SM r=0 curvatures: SU(3) %.5f (v11 -0.393), SU(2) %.5f (v11 -0.242), U(1)_Y %.5f (v11 +0.700)" % (k3, k2, kY))
    print("first-order dQ with v11 spreads: %.4f %.4f %.4f (v11 -0.72 -0.17 +0.29)" % (0.5*k3*3.66, 0.5*k2*1.40, 0.5*kY*0.82))
    # (5) mass dependence of curvatures
    for m2 in [0.5, 1.0, 2.0, 4.0, 9.0]:
        print("m2=%4.1f kappa_P=%.6f kappa_A=%.6f" % (m2, kappa_q(fP, m2)[0], kappa_q(fA, m2)[0]))
    # (6) Delta T1 small
    print("dT1(S=5, th=0.1,0.3) =", dT1(np.array([5.0, 5.0]), np.array([0.1, 0.3])))
    out.update(kappa_P0=kP, kappa_A0=kA, SU3=k3, SU2=k2, UY=kY, dQ=[0.5*k3*3.66, 0.5*k2*1.40, 0.5*kY*0.82])
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "twisted_selftest.json"), "w"), indent=1)
