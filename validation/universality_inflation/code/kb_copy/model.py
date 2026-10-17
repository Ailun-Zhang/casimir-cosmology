# PROVENANCE: copied from D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\keystone_blind\work\model.py on 2026-10-15 by ui-blind (unmodified below this line)
# keystone-blind (T3): body-frame zero-mode model.
# H = -(1/2)(d_r^2 + 3/r d_r) + (2/r^2) J^2 - (1/2)nabla_x^2 - (1/2)nabla_w^2 + V_YM + V_mass + lam r^4/4 + V_1(r)
# Conditional Gaussian for SU(2) zero modes x given (r,w): per spatial direction i,
#   x_i^{1,2} ~ N(0, 1/a), x_i^3 ~ N(m w_i, 1/k)   (|psi|^2 precisions a, k; mean slope m)
# Exact projections (derived in report):
#   U0n = 3/8(2a+k) + 3 r^2 g^2/8 (2/a+1/k) + g^2(6/(a k)+3/a^2) + (3/r^2)(a-k)^2/(a k) + 3/8 k m^2
#   U2n = r^2/8 (g m - gY)^2 + 2 g^2 m^2/a + a m^2/r^2          (coefficient of s^2 = |w|^2)
#   Gaussian energy (w marginal optimal, beta = 2 Omega): E_G = U0n + (3/2) Omega, Omega = sqrt(2 U2n)
#   Born-Huang (r-dependence of conditional Gaussian): (3/8)(a'/a)^2 + (3/16)(k'/k)^2 + (k/8) m'^2 s^2
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import oneloop

def _parts(p, r, g, gY):
    al, ka, m = p
    a, k = np.exp(al), np.exp(ka)
    U0 = 3/8*(2*a+k) + 3*r*r*g*g/8*(2/a+1/k) + g*g*(6/(a*k)+3/a**2) + 3/r**2*(a/k+k/a-2) + 3/8*k*m*m
    U2 = r*r/8*(g*m-gY)**2 + 2*g*g*m*m/a + a*m*m/r**2
    return U0, U2

def _EG_and_grad(p, r, g, gY):
    al, ka, m = p
    a, k = np.exp(al), np.exp(ka)
    U0, U2 = _parts(p, r, g, gY)
    Om = np.sqrt(2*U2)
    E = U0 + 1.5*Om
    dU0a = 3/4 - 3*r*r*g*g/4/a**2 - g*g*(6/(a*a*k) + 6/a**3) + 3/r**2*(1/k - k/a**2)
    dU0k = 3/8 - 3*r*r*g*g/8/k**2 - 6*g*g/(a*k*k) + 3/r**2*(-a/k**2 + 1/a) + 3/8*m*m
    dU0m = 3/4*k*m
    dU2a = -2*g*g*m*m/a**2 + m*m/r**2
    dU2m = r*r/4*(g*m-gY)*g + 4*g*g*m/a + 2*a*m/r**2
    f = 1.5/np.sqrt(2*U2)
    gr = np.array([a*(dU0a + f*dU2a), k*dU0k, dU0m + f*dU2m])
    return E, gr

class Gauss:
    """Per-r optimal conditional-Gaussian parameters on a fine grid, with smooth derivatives."""
    def __init__(self, g, gY, rmin=0.002, rmax=27.0, dr=0.004):
        self.g, self.gY = g, gY
        rr = np.arange(rmin, rmax + dr/2, dr)
        a0 = (16*g*g)**(1/3)
        p = np.array([np.log(a0), np.log(a0), 0.0])
        P = np.empty((len(rr), 3)); E = np.empty(len(rr))
        for i, r in enumerate(rr):
            res = minimize(_EG_and_grad, p, args=(r, g, gY), jac=True, method="BFGS", options=dict(gtol=1e-13, maxiter=2000))
            if not res.success and res.status != 2:
                res = minimize(_EG_and_grad, res.x, args=(r, g, gY), jac=True, method="L-BFGS-B", options=dict(gtol=1e-14, ftol=1e-16, maxiter=5000))
            p = res.x; P[i] = p; E[i] = res.fun
        self.r = rr; self.P = P; self.E = E
        self.sp = [CubicSpline(rr, P[:, j]) for j in range(3)]
        self.spE = CubicSpline(rr, E)

    def terms(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        al, ka, m = [s(r) for s in self.sp]
        dal, dka, dm = [s(r, 1) for s in self.sp]
        g, gY = self.g, self.gY
        a, k = np.exp(al), np.exp(ka)
        U0, U2 = _parts((al, ka, m), r, g, gY)
        Om = np.sqrt(2*U2)
        EG = U0 + 1.5*Om
        BH0 = 3/8*dal**2 + 3/16*dka**2          # s-independent BH
        BH2 = k/8*dm**2                          # coefficient of s^2 in BH
        dOm = np.gradient(Om, r) if len(r) > 3 else np.zeros_like(r)
        return dict(a=a, k=k, m=m, U0n=U0, U2n=U2, Om=Om, EG=EG, BH0=BH0, BH2=BH2, dOm=dOm)

    def potentials(self, r):
        """Gaussian-sector pieces of the three reduced models on grid r (must be fine & sorted for dOm)."""
        t = self.terms(r)
        U0 = t["U0n"] + t["BH0"]                 # 2D_G: s-independent part
        U2 = t["U2n"] + t["BH2"]                 # 2D_G: coefficient of s^2
        Om_n = t["Om"]
        V1D = t["EG"] + t["BH0"] + t["BH2"]*3/(2*Om_n) + 3/16*(t["dOm"]/Om_n)**2   # 1D BO (variational)
        Vlow = U0 + 1.5*np.sqrt(2*U2)          # BO lower bound for 2D_G (no w-BH)
        return dict(U0=U0, U2=U2, V1D=V1D, Vlow=Vlow, terms=t)

def onel(r, lam, g2, gY, yt):
    return oneloop.V1loop(r, lam, g2, gY, yt)

if __name__ == "__main__":
    import sys, json
    sys.stdout.reconfigure(encoding="utf-8")
    G = Gauss(0.5268, 0.4415)
    for r in [0.01, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25]:
        t = G.terms(np.array([r]))
        print(f"r={r:5.2f} a={t['a'][0]:.5f} k={t['k'][0]:.5f} m={t['m'][0]:.5f} EG={t['EG'][0]:.6f} Om={t['Om'][0]:.5f} BH0={t['BH0'][0]:.2e} BH2={t['BH2'][0]:.2e}")
    a0 = (16*0.5268**2)**(1/3); print("pure YM Gaussian 27/16 a0 =", 27/16*a0)
