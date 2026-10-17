# keystone-blind V2: zero-mode / nonzero-mode cross term in the 2D (r, s=|w|) model.
# Background (body frame, per spatial direction): Cartan c3 = g2 x^3, u = gY w  -> shift q_f = t3 c3 + y u per field component
# (twist approximation with the T3 r-dependent masses; Coulomb modes carry the Goldstone charges; fermions chirality-wise).
# Parts:  SU2  = curvature of x-fluctuations (x^1,x^2 with kappa_1 ~= kappa_33, delta x^3 with kappa_33) in the Gaussian functional
#         UY   = exact (direction-averaged) twisted sums along the conditional-mean background (m(r) w, w) -> V_Y(r,s) in 2D;
#                its quadratic part kappa_mean(m) in the Gaussian functional (for the optimisation)
#         SU3t = SU(3) zero-mode Gaussian energy with r-dependent curvature (top mass) -> V_3t(r)
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "work")); sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, RectBivariateSpline
import scipy.sparse as sp, scipy.sparse.linalg as spla
import model, solvers, oneloop
import twisted as tw

PI = np.pi
LIGHT = [("uL", 12, 0.5, 1/6), ("dL", 12, -0.5, 1/6), ("uR", 12, 0.0, 2/3), ("dR", 12, 0.0, -1/3), ("nuL", 4, 0.5, -0.5), ("eL", 4, -0.5, -0.5),
         ("eR", 4, 0.0, -1.0), ("bL", 6, -0.5, 1/6), ("bR", 6, 0.0, -1/3), ("ntL", 2, 0.5, -0.5), ("taL", 2, -0.5, -0.5), ("taR", 2, 0.0, -1.0)]
TOP = [("tL", 6, 0.5, 1/6), ("tR", 6, 0.0, 2/3)]

def fields(r, g2, gY, yt, lam, which="all"):
    mW2 = g2*g2*r*r/4; mZ2 = (g2*g2 + gY*gY)*r*r/4; mt2 = yt*yt*r*r/2; mh2 = 3*lam*r*r; mG2 = lam*r*r
    F = [("WT", 4, 1, "P", mW2, 1.0, 0.0), ("WC", 2, 1, "C", (mG2, mW2), 0.5, 0.5), ("h", 1, 1, "P", mh2, -0.5, 0.5), ("ZC", 1, 1, "C", (mG2, mZ2), -0.5, 0.5)]
    if which == "higgs":
        return [f for f in F if f[0] in ("WC", "h", "ZC")]
    F += [(nm, n, -1, "A", 0.0, t3, y) for nm, n, t3, y in LIGHT] + [(nm, n, -1, "A", mt2, t3, y) for nm, n, t3, y in TOP]
    return F

# ---- tabulated Delta T1(S, theta) for Coulomb modes ----
_S_GRID = np.concatenate([[1e-3], np.geomspace(0.01, 80.0, 34)]); _TH_GRID = np.linspace(0, 0.5, 26)
_DT1 = None
def _dt1_table():
    global _DT1
    f = os.path.join(HERE, "cache_dT1.npz")
    if _DT1 is None:
        if os.path.exists(f): T = np.load(f)["T"]
        else:
            SS, TT = np.meshgrid(_S_GRID, _TH_GRID, indexing="ij"); T = tw.dT1(SS.ravel(), TT.ravel()).reshape(SS.shape); np.savez(f, T=T)
        _DT1 = RectBivariateSpline(np.log(_S_GRID), _TH_GRID, T, kx=3, ky=3)
    return _DT1

def dfield(f, th):
    """Delta f (per real dof) for field f at twist |theta| = th (array)."""
    kind = f[3]; th = np.atleast_1d(th).astype(float)
    if kind == "A":
        m2 = np.full(th.shape, float(f[4])); return tw.fA(m2, th) - tw.fA(m2, 0*th)
    thb = np.minimum(th, 0.5)                    # bosons: freeze beyond theta=1/2 (local zero-mode picture ends)
    if kind == "P":
        m2 = np.full(th.shape, float(f[4])); return tw.fP(m2, thb) - tw.fP(m2, 0*thb)
    A, B = f[4]; S = A + B; P = A*B; m2 = np.full(th.shape, S)
    return tw.fP(m2, thb) - tw.fP(m2, 0*thb) + P*_dt1_table().ev(np.full(th.shape, np.log(max(S, 1e-3))), thb)

def kappa_field(f, d=0.01):
    """curvature per real dof per unit q^2."""
    z = np.array([0.0]); a = dfield(f, z + d)[0]; b = dfield(f, z + 2*d)[0]
    return 2*(16*a - b)/(12*d*d)/(4*PI**2)

def cartan_kappa(r, g2, gY, yt, lam, which="all"):
    """kappa_AB in coupling-absorbed units for (c3, u)."""
    K = np.zeros((2, 2))
    for f in fields(r, g2, gY, yt, lam, which):
        k = f[2]*f[1]*kappa_field(f); q = np.array([f[5], f[6]]); K += k*np.outer(q, q)
    return K

def su3_kappa(r, yt):
    mt2 = yt*yt*r*r/2
    kP0 = tw.kappa_q(tw.fP, 0.0)[0]; kA0 = tw.kappa_q(tw.fA, 0.0)[0]; kAt = tw.kappa_q(tw.fA, mt2)[0]
    return 6*kP0 - 10*kA0 - 2*kAt

def su3_energy(kap, g3):
    """SU(3) zero-mode Gaussian: E(a) = 3a + 36 g^2/a^2 + 12 kappa g^2/a (coupling-absorbed curvature)."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda la: 3*np.exp(la) + 36*g3*g3*np.exp(-2*la) + 12*kap*g3*g3*np.exp(-la), bounds=(-3, 3), method="bounded", options=dict(xatol=1e-12))
    a = np.exp(res.x); return res.fun, 24*g3*g3/a

def _zero(r): return 0*np.asarray(r, float)

# ---- Gaussian functional with cross terms ----
def _EGx(p, r, g, gY, k1, k33, k3w, kww, su2, uy):
    """Gaussian energy with cross terms and its analytic gradient in (ln a, ln k, m)."""
    al, ka, m = p; a, k = np.exp(al), np.exp(ka)
    U0 = 3/8*(2*a + k) + 3*r*r*g*g/8*(2/a + 1/k) + g*g*(6/(a*k) + 3/a**2) + 3/r**2*(a/k + k/a - 2) + 3/8*k*m*m
    U2 = r*r/8*(g*m - gY)**2 + 2*g*g*m*m/a + a*m*m/r**2
    s2, su = (1.0 if su2 else 0.0), (1.0 if uy else 0.0)
    dU0 = s2*0.5*g*g*(k1*6/a + k33*3/k)
    dU2 = su*0.5*(k33*g*g*m*m + 2*k3w*g*gY*m + kww*gY*gY)
    X = max(U2 + dU2, 1e-14); f = 1.5/np.sqrt(2*X)
    dU0a = 3/4 - 3*r*r*g*g/4/a**2 - g*g*(6/(a*a*k) + 6/a**3) + 3/r**2*(1/k - k/a**2) - s2*3*g*g*k1/a**2
    dU0k = 3/8 - 3*r*r*g*g/8/k**2 - 6*g*g/(a*k*k) + 3/r**2*(-a/k**2 + 1/a) + 3/8*m*m - s2*1.5*g*g*k33/k**2
    dU2a = -2*g*g*m*m/a**2 + m*m/r**2
    dU2m = r*r/4*(g*m - gY)*g + 4*g*g*m/a + 2*a*m/r**2 + su*(k33*g*g*m + k3w*g*gY)
    grad = np.array([a*(dU0a + f*dU2a), k*dU0k, 3/4*k*m + f*dU2m])
    return U0 + dU0 + 1.5*np.sqrt(2*X), grad

class XGauss:
    def __init__(self, g2, gY, yt, lam, g3, parts, rmin=0.002, rmax=27.0, dr=0.004):
        self.g, self.gY, self.yt, self.lam, self.g3, self.parts = g2, gY, yt, lam, g3, set(parts)
        rk = np.concatenate([np.linspace(0.002, 3, 31), np.linspace(3.25, 27, 96)])
        Ks = np.array([cartan_kappa(r, g2, gY, yt, lam) for r in rk])
        self.kap = dict(r=rk, k33=Ks[:, 0, 0], k3w=Ks[:, 0, 1], kww=Ks[:, 1, 1])
        sk = {n: CubicSpline(rk, self.kap[n]) for n in ("k33", "k3w", "kww")}
        rr = np.arange(rmin, rmax + dr/2, dr)
        su2, uy = "su2" in self.parts, "uy" in self.parts
        a0 = (16*g2*g2)**(1/3); p = np.array([np.log(a0), np.log(a0), 0.0]); P = np.empty((len(rr), 3))
        k33a, k3wa, kwwa = sk["k33"](rr), sk["k3w"](rr), sk["kww"](rr)
        for i, r in enumerate(rr):
            res = minimize(_EGx, p, args=(r, g2, gY, k33a[i], k33a[i], k3wa[i], kwwa[i], su2, uy), jac=True, method="BFGS",
                           options=dict(gtol=1e-13, maxiter=3000))
            p = res.x; P[i] = p
        self.r = rr; self.P = P; self.k33, self.k3w, self.kww = k33a, k3wa, kwwa
        self.sp = [CubicSpline(rr, P[:, j]) for j in range(3)]
        # SU(3)-top
        if "su3t" in self.parts:
            r3 = np.linspace(0.0, 27.0, 109); E3 = np.array([su3_energy(su3_kappa(r, yt), g3)[0] for r in r3])
            self.V3t = CubicSpline(r3, E3 - E3[0])
        else:
            self.V3t = _zero
        self.VY = None
        if uy: self._build_VY()

    def terms(self, r):
        r = np.atleast_1d(np.asarray(r, float)); g, gY = self.g, self.gY
        al, ka, m = [s(r) for s in self.sp]; dal, dka, dm = [s(r, 1) for s in self.sp]
        a, k = np.exp(al), np.exp(ka)
        k33 = np.interp(r, self.r, self.k33); k3w = np.interp(r, self.r, self.k3w); kww = np.interp(r, self.r, self.kww)
        su2, uy = "su2" in self.parts, "uy" in self.parts
        U0 = 3/8*(2*a + k) + 3*r*r*g*g/8*(2/a + 1/k) + g*g*(6/(a*k) + 3/a**2) + 3/r**2*(a/k + k/a - 2) + 3/8*k*m*m
        U2 = r*r/8*(g*m - gY)**2 + 2*g*g*m*m/a + a*m*m/r**2
        dU0 = 0.5*g*g*(k33*6/a + k33*3/k) if su2 else 0*r
        dU2 = 0.5*(k33*g*g*m*m + 2*k3w*g*gY*m + kww*gY*gY) if uy else 0*r
        Om = np.sqrt(2*np.maximum(U2 + dU2, 1e-14))
        return dict(a=a, k=k, m=m, U0n=U0, U2n=U2, dU0=dU0, dU2=dU2, Om=Om, BH0=3/8*dal**2 + 3/16*dka**2, BH2=k/8*dm**2,
                    EG=U0 + dU0 + 1.5*Om, dOm=np.gradient(Om, r) if len(r) > 3 else 0*r)

    def _build_VY(self):
        rV = np.concatenate([np.linspace(0.01, 2.0, 24), np.linspace(2.1, 21.0, 64)]); sV = np.linspace(0, 16, 65)
        t = self.terms(rV); V = np.zeros((len(rV), len(sV)))
        for i, r in enumerate(rV):
            for f in fields(r, self.g, self.gY, self.yt, self.lam):
                q = abs(self.g*f[5]*t["m"][i] + self.gY*f[6])
                if q == 0: continue
                V[i] += f[2]*f[1]*dfield(f, q*sV/(2*PI))
        self.VYgrid = (rV, sV, V)
        self.VY = RectBivariateSpline(rV, sV, V, kx=3, ky=3)

    def potentials(self, r):
        t = self.terms(r)
        U0 = t["U0n"] + t["dU0"] + t["BH0"]; U2 = t["U2n"] + t["BH2"]
        V3 = self.V3t(r)
        # 1D BO (variational, Gaussian w-marginal with precision 2*Om): replace quadratic dU2 by exact <V_Y>
        Om = t["Om"]
        if self.VY is not None:
            ss = np.linspace(0, 16, 321); wgt = ss[None, :]**2*np.exp(-Om[:, None]*ss[None, :]**2)
            VYm = np.array([self.VY.ev(np.full(ss.shape, min(max(ri, 0.01), 21.0)), ss) for ri in r])
            VYavg = (wgt*VYm).sum(1)/wgt.sum(1)
        else:
            VYavg = 0*r
        V1D = t["U0n"] + t["dU0"] + 1.5*Om - t["dU2"]*3/(2*Om) + VYavg + t["BH0"] + t["BH2"]*3/(2*Om) + 3/16*(t["dOm"]/Om)**2
        return dict(U0=U0 + V3, U2=U2, V1D=V1D + V3, terms=t)

    def vlow(self, r, Ns=24, Om0=0.35):
        """BO lower bound: U0 + lowest eigenvalue of the s-problem (DVR) at each r."""
        t = self.terms(r); sig, Ts = solvers.sdvr(Ns, Om0); s = np.sqrt(sig)
        U2 = t["U2n"] + t["BH2"]
        VYs = np.array([self.VY.ev(np.full(s.shape, min(max(ri, 0.01), 21.0)), np.minimum(s, 16.0)) for ri in r]) if self.VY is not None else 0*np.outer(r, s)
        H = Ts[None, :, :] + np.einsum("r,j->rj", U2, sig)[:, :, None]*np.eye(Ns)[None] + VYs[:, :, None]*np.eye(Ns)[None]
        e0 = np.linalg.eigvalsh(H)[:, 0]
        return t["U0n"] + t["dU0"] + t["BH0"] + e0 + self.V3t(r)

# ---- 2D with extra V_Y(r,s) ----
def build2d_x(Vr, U2, VY, Rmax, dr, Ns, Om0, eta=0.0, Rc=None):
    N = int(round(Rmax/dr)) - 1; r = dr*np.arange(1, N + 1)
    sig, Ts = solvers.sdvr(Ns, Om0); s = np.sqrt(sig)
    H = sp.kron(solvers.kin_r(N, dr), sp.identity(Ns)) + sp.kron(sp.identity(N), sp.csr_matrix(Ts))
    diag = (Vr(r) + 3/(8*r*r))[:, None] + U2(r)[:, None]*sig[None, :]
    if VY is not None:
        diag = diag + VY(np.minimum(r, 21.0)[:, None]*np.ones(Ns)[None, :], np.minimum(s, 16.0)[None, :]*np.ones(N)[:, None])
    H = H + sp.diags(diag.ravel())
    if eta: H = H.astype(complex) - 1j*eta*sp.diags(np.repeat(solvers.cap_W(r, Rc, Rmax), Ns))
    return H.tocsc(), r, sig

def cap2d_x(Vr, U2, VY, Rc, Rmax, dr, Ns, Om0, E0, etas):
    traj = []
    for eta in etas:
        H, r, sig = build2d_x(Vr, U2, VY, Rmax, dr, Ns, Om0, eta, Rc)
        vals, vecs = spla.eigs(H, k=6, sigma=E0); i = np.argmin(np.abs(vals - E0)); E = vals[i]; v = vecs[:, i]
        W = np.repeat(solvers.cap_W(r, Rc, Rmax), len(sig)); dE = -1j*np.sum(v*v*W)/np.sum(v*v)
        traj.append((eta, E, E - eta*dE, abs(eta*dE))); E0 = E
    j = int(np.argmin([t[3] for t in traj])); return dict(traj=traj, E=traj[j][1], Ecorr=traj[j][2], eta=traj[j][0])

def VYfun(XG):
    if XG.VY is None: return None
    return lambda R, S: XG.VY.ev(R.ravel(), S.ravel()).reshape(R.shape)
