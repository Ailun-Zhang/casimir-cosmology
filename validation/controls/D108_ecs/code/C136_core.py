r"""C136 core library -- DNEC v12, branch higgs_survival (Claude, 2026-10-13).

Independent implementation (no Codex code imported or copied) of the gauge-invariant reduced radial problem for the
SM Higgs zero mode held by the electroweak gauge zero modes on the AAA cube (v11 Eq. (11)), with finite-volume
matching of the nonzero modes, outgoing-wave (Siegert) and exterior-complex-scaling (ECS) pole solvers.

Conventions (C72, used by convention only): unit-mass coordinates c = g2 x, u = gY w, Higgs Phi=(phi0+i phi1,
phi2+i phi3)/sqrt2, r = |phi|, L*H = 1/2 p^2 + g2^2/4 F[x]^2 + sum_i Phi^+(g2 x_i.T + gY Y w_i)^2 Phi + lam r^4/4.
Physical reduction (derivation D108 Sec. 2): body frame N = 2 Phi^+ sigma Phi / r^2 = e3, residual U(1)_em,
Higgs angular kinetic energy on Gauss-law states = (2/r^2) L_gauge^2.  Conditional Gaussian, per space direction
y_i = (x^1,x^2,x^3,w)_i, precision K = diag(a, a, K2), K2 = [[k,c],[c,b]]: the most general SO(3)_space x U(1)_em
invariant centred Gaussian.  Radial operator (u = r^{3/2} f):
    H = -1/2 d^2/dr^2 + 3/(8 r^2) + E_tot(p*(r); r) + B_r(r) + lam r^4/4 + DeltaV(r) [+ options]
E_tot = <kin + mass + YM> + B_ang (+ holonomy curvature option);  B_r = (3/16) tr[(K^-1 K')^2].
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import math
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import special, sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import brentq, minimize
from scipy.sparse.linalg import eigs

ROOT = Path(r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology")
CODE = ROOT / "04_calculations" / "code" / "20261013_claude_v12" / "higgs_survival"
RES = ROOT / "04_calculations" / "results" / "20261013_claude_v12" / "higgs_survival"

COUP_V11 = {"g2": 0.5268, "gY": 0.4415, "yt": 0.4585, "k2": 0.0, "kY": 0.0}
HOL_V11 = {"k2": -0.242, "kY": 0.700}          # v11 App. F: L d2E/dc2 at r = 0 (SU(2), hypercharge)
TB_HALF_REF, TF_HALF_REF = -0.1128924797204, -0.0319080184183   # C81 / v11 App. F small-mass coefficients
GAMMA_Z2 = 1.6750738213                        # Z(2)/pi^2 (C72) ; F_P(0) = -GAMMA_Z2/2 per real periodic boson
HL_B = 0.00437243
ETA_ONE = 1.31102877714606 / HL_B              # one-sided conformal time of the turn, 299.84
CODEX = {"local": {-0.005: 8.21e-11, -0.010: 6.08450e-6, -0.0143: 3.23718e-4, -0.020: 5.19101e-3},
         "matched": {-0.005: 1.27746e-5, -0.010: 2.92010e-3, -0.0143: 1.917044e-2, -0.020: 6.57398e-2},
         "matched_Q_0143": 4.33994514, "local_lam0_bound": 4.598545,
         "LQ": {-0.0143: 1.020685e-2, -0.010: 1.46751e-3}}

# =====================================================================================================  Gaussian family
_S = sp.symbols("a k c b r g gy k2 kY")


def _build_family():
    a, k, c, b, r, g, gy, k2, kY = _S
    R = sp.Rational
    D = k * b - c ** 2
    parts = {
        "kin": R(3, 4) * (2 * a + k + b),                                                     # <p^2/2>
        "mass": R(3, 16) * r ** 2 * (2 * g ** 2 / a + (g ** 2 * b - 2 * g * gy * c + gy ** 2 * k) / D),  # 3/4 tr(M K^-1)
        "ym": R(3, 4) * g ** 2 / a ** 2 + R(3, 2) * g ** 2 * b / (a * D),                     # g^2/4 <F^2>
        "ang": 3 * (b * (k - a) ** 2 - 2 * c ** 2 * (k - a) + k * c ** 2) / (a * r ** 2 * D),  # (2/r^2)<L^2>
        "hol": R(3, 4) * (k2 * g ** 2 * (2 / a + b / D) + kY * gy ** 2 * k / D),              # holonomy curvature
    }
    E = sum(parts.values())
    p = [a, k, c, b]
    grad = [sp.diff(E, v) for v in p]
    hess = [[sp.diff(gi, v) for v in p] for gi in grad]
    dgr = [sp.diff(gi, r) for gi in grad]
    lam = lambda ex: sp.lambdify(_S, ex, "numpy")
    return lam(E), {kk: lam(vv) for kk, vv in parts.items()}, lam(grad), lam(hess), lam(dgr)


F_E, F_PARTS, F_GRAD, F_HESS, F_DGR = _build_family()


def _cargs(p, r, cp):
    return (p[0], p[1], p[2], p[3], r, cp["g2"], cp["gY"], cp["k2"], cp["kY"])


def _vec(vals, shape, dtype):
    return np.array([np.broadcast_to(np.asarray(v, dtype=dtype), shape) for v in vals])


def feasible(p):
    a, k, c, b = p
    return a > 0 and k > 0 and b > 0 and k * b - c * c > 0


def newton(p0, r, cp, tol=1e-14, maxit=80):
    cplx = np.iscomplexobj(p0) or isinstance(r, complex) or np.iscomplexobj(r)
    dt = complex if cplx else float
    p = np.array(p0, dtype=dt)
    for it in range(maxit):
        g = np.array(F_GRAD(*_cargs(p, r, cp)), dtype=dt)
        H = np.array(F_HESS(*_cargs(p, r, cp)), dtype=dt)
        dp = np.linalg.solve(H, -g)
        t = 1.0
        if not cplx:
            while not feasible(p + t * dp) and t > 1e-10:
                t *= 0.5
        p = p + t * dp
        if np.max(np.abs(t * dp)) <= tol * (1.0 + np.max(np.abs(p))):
            return p, it + 1
    return p, -1


def dp_dr(p, r, cp):
    dt = complex if (np.iscomplexobj(p) or isinstance(r, complex)) else float
    H = np.array(F_HESS(*_cargs(p, r, cp)), dtype=dt)
    d = np.array(F_DGR(*_cargs(p, r, cp)), dtype=dt)
    return -np.linalg.solve(H, d)


def _q2p(q):
    a, k, b = np.exp(q[0]), np.exp(q[1]), np.exp(q[3])
    return np.array([a, k, np.tanh(q[2]) * np.sqrt(k * b), b])


def global_min(r, cp, starts=24, seed=7):
    rng = np.random.default_rng(seed)
    old = np.seterr(all="ignore")

    def f(q):
        v = float(F_E(*_cargs(_q2p(q), r, cp)))
        return v if np.isfinite(v) else 1e30

    best = None
    for _ in range(starts):
        q0 = np.array([rng.uniform(-3, 2), rng.uniform(-3, 2), rng.uniform(-1.5, 1.5), rng.uniform(-4, 2)])
        res = minimize(f, q0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=40000, maxfev=80000))
        res = minimize(f, res.x, method="BFGS", options=dict(gtol=1e-11, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    np.seterr(**old)
    p, it = newton(_q2p(best.x), r, cp)
    return p, float(best.fun), it


def family_eval(P, r, cp):
    """E_tot, B_r and parts for parameter rows P (n,4) at radii r (n,) [real or complex, holomorphic]."""
    P = np.asarray(P)
    dt = complex if (np.iscomplexobj(P) or np.iscomplexobj(r)) else float
    r = np.asarray(r, dtype=dt)
    n = len(r)
    args = _cargs(P.T, r, cp)
    E = np.asarray(F_E(*args), dtype=dt)
    H = np.array([[np.broadcast_to(np.asarray(v, dtype=dt), (n,)) for v in row] for row in F_HESS(*args)])
    H = np.moveaxis(H, (0, 1), (1, 2))                     # (n,4,4)
    d = _vec(F_DGR(*args), (n,), dt).T                     # (n,4)
    dP = -np.linalg.solve(H, d[..., None])[..., 0]
    a, k, c, b = P.T
    da, dk, dc, db = dP.T
    D = k * b - c * c
    m11 = (b * dk - c * dc) / D
    m12 = (b * dc - c * db) / D
    m21 = (-c * dk + k * dc) / D
    m22 = (-c * dc + k * db) / D
    trM2 = m11 * m11 + 2 * m12 * m21 + m22 * m22
    Br = 3.0 / 16.0 * (2 * (da / a) ** 2 + trM2)
    parts = {kk: np.broadcast_to(np.asarray(f(*args), dtype=dt), (n,)) for kk, f in F_PARTS.items()}
    return E, Br, parts, dP


def gauge_table(rgrid, cp, r_start=1.0):
    """March the conditional-Gaussian optimum over a real grid (global start at r_start)."""
    n = len(rgrid)
    i0 = int(np.argmin(np.abs(rgrid - r_start)))
    P = np.empty((n, 4))
    P[i0], fstart, _ = global_min(rgrid[i0], cp)
    for idx in (range(i0 + 1, n), range(i0 - 1, -1, -1)):
        p = P[i0].copy()
        rp = rgrid[i0]
        for i in idx:
            pred = p + dp_dr(p, rp, cp) * (rgrid[i] - rp)
            if not feasible(pred):
                pred = p
            p, it = newton(pred, rgrid[i], cp)
            if it < 0 or not feasible(p):
                raise RuntimeError(f"gauge_table: Newton failure at r={rgrid[i]}")
            P[i] = p
            rp = rgrid[i]
    E, Br, parts, dP = family_eval(P, rgrid, cp)
    return {"r": rgrid, "P": P, "E": E, "Br": Br, "parts": parts, "dP": dP}


def gauge_on_ray(p_start, R0, theta, rho, cp):
    """Analytic continuation of the stationary Gaussian parameters along z = R0 + rho e^{i theta} (rho[0] = 0)."""
    Pz = np.empty((len(rho), 4), complex)
    p = np.asarray(p_start, complex)
    zprev = complex(R0)
    e = np.exp(1j * theta)
    for j, rh in enumerate(rho):
        z = R0 + rh * e
        if j > 0:
            pred = p + dp_dr(p, zprev, cp) * (z - zprev)
            p, it = newton(pred, z, cp, tol=1e-14)
            if it < 0:
                raise RuntimeError(f"gauge_on_ray: Newton failure at z={z}")
        Pz[j] = p
        zprev = z
    z = R0 + rho * e
    E, Br, _, _ = family_eval(Pz, z, cp)
    return z, E + Br, Pz


# =====================================================================================================  matching D(mu^2)
class HeatD:
    r"""D_theta(mu^2) = -(1/4 sqrt pi) int_0^inf ds s^{-3/2} (e^{-mu^2 s}-1) [K_theta(s) - delta_{theta,0} - (4 pi s)^{-3/2}],
    K_theta(s) = sum_n exp(-4 pi^2 |n+theta|^2 s): nonzero-mode energy minus continuum (L units, per real dof).
    kind 'P' (periodic, constant mode removed) or 'A' (AAA).  Split at s_c: Poisson images below, lattice
    shells + closed-form continuum above.  Holomorphic in mu^2; branch 'pos' (log mu^2) or 'neg' (log(-mu^2):
    the real-part prescription for tachyonic masses, holomorphic around the negative axis).  Valid for
    Re mu^2 > -4 pi^2 (all nonzero modes real)."""

    def __init__(self, kind, sc=0.1, nquad=120, nmax=8):
        self.kind, self.sc = kind, sc
        rng = np.arange(-nmax, nmax + 1)
        n1, n2, n3 = np.meshgrid(rng, rng, rng, indexing="ij")
        if kind == "P":
            q4 = 4 * (n1 ** 2 + n2 ** 2 + n3 ** 2)
        else:
            ra = np.arange(-nmax, nmax)
            a1, a2, a3 = np.meshgrid(ra, ra, ra, indexing="ij")
            q4 = (2 * a1 + 1) ** 2 + (2 * a2 + 1) ** 2 + (2 * a3 + 1) ** 2
        q4 = q4.ravel()
        q4 = q4[q4 > 0]
        vals, mult = np.unique(q4, return_counts=True)
        self.kap = np.pi ** 2 * vals.astype(float)
        self.mult = mult.astype(float)
        msq = (n1 ** 2 + n2 ** 2 + n3 ** 2).ravel()
        w = np.ones(msq.shape) if kind == "P" else (-1.0) ** ((n1 + n2 + n3).ravel())
        keep = msq > 0
        mv, inv = np.unique(msq[keep], return_inverse=True)
        self.msq = mv.astype(float)
        self.wimg = np.bincount(inv, weights=w[keep])
        x, wq = np.polynomial.legendre.leggauss(nquad)
        lo, hi = math.log(2e-3), math.log(sc)
        u = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        self.s = np.exp(u)
        self.ws = 0.5 * (hi - lo) * wq * self.s
        Ps = (4 * np.pi * self.s) ** -1.5 * (self.wimg[None, :] * np.exp(-self.msq[None, :] / (4 * self.s[:, None]))).sum(1)
        self.wsmall = self.s ** -1.5 * Ps * self.ws
        self.G0 = self._G(self.kap.astype(complex))

    def _G(self, alpha):
        sc = self.sc
        sa = np.sqrt(alpha * sc)
        return np.exp(-alpha * sc) * (2 / math.sqrt(sc) - 2 * math.sqrt(math.pi) * (sa / math.sqrt(sc)) * special.erfcx(sa))

    @staticmethod
    def _Ein(z):
        term = np.ones_like(z)
        s = np.zeros_like(z)
        for k in range(1, 110):
            term = term * (-z) / k
            s = s - term / k
        return s

    def _E1(self, z, branch):
        if branch == "pos":
            out = np.empty_like(z)
            big = np.abs(z) > 8
            out[big] = special.exp1(z[big])
            zs = z[~big]
            out[~big] = -np.euler_gamma - np.log(zs) + self._Ein(zs)
            return out
        if np.any(np.abs(z) > 8):
            raise ValueError("neg branch outside series range")
        return -np.euler_gamma - np.log(-z) + self._Ein(z)

    def __call__(self, mu2, branch="pos", chunk=1500):
        mu2 = np.atleast_1d(np.asarray(mu2, dtype=complex))
        out = np.zeros(mu2.shape, complex)
        flat_in, flat_out = mu2.ravel(), out.ravel()
        sc = self.sc
        for st in range(0, len(flat_in), chunk):
            m = flat_in[st:st + chunk]
            nz = m != 0
            mm = m[nz]
            if len(mm) == 0:
                continue
            I1 = ((np.exp(-np.outer(mm, self.s)) - 1.0) * self.wsmall[None, :]).sum(1)
            if self.kind == "P":
                mu = np.sqrt(mm)
                J = -2 * (np.exp(-mm * sc) - 1) / math.sqrt(sc) - 2 * math.sqrt(math.pi) * mu * special.erf(mu * math.sqrt(sc))
                I1 = I1 - J
            I2 = (self.mult[None, :] * (self._G(self.kap[None, :] + mm[:, None]) - self.G0[None, :])).sum(1)
            z = mm * sc
            C = 0.5 * (np.exp(-z) / sc ** 2 - mm * np.exp(-z) / sc + mm * mm * self._E1(z, branch)) - 0.5 / sc ** 2
            tot = I1 + I2 - (4 * np.pi) ** -1.5 * C
            res = np.zeros(len(m), complex)
            res[nz] = -tot / (4 * math.sqrt(math.pi))
            flat_out[st:st + chunk] = res
        return flat_out.reshape(mu2.shape)


DP = HeatD("P")
DA = HeatD("A")


def dP_c(coef, z2):
    return DP(coef * z2, "pos" if coef > 0 else "neg")


def bessel_F(x, kind, Nb=90):
    """Independent image-sum F_theta(x) = -(x^2/4pi^2) sum_{m!=0} w_m K_2(x|m|)/|m|^2 (real x>0; includes constant mode)."""
    qmax = Nb * Nb
    W = np.zeros(3 * qmax + 1)
    rr = np.arange(-Nb, Nb + 1)
    m2, m3 = np.meshgrid(rr, rr, indexing="ij")
    for m1 in rr:
        q = (m1 * m1 + m2 * m2 + m3 * m3).ravel()
        w = np.ones(q.shape) if kind == "P" else (-1.0) ** ((m1 + m2 + m3).ravel())
        W += np.bincount(q, weights=w, minlength=len(W))
    q = np.arange(1, qmax + 1)
    Wq = W[1:qmax + 1]
    sel = Wq != 0
    q, Wq = q[sel].astype(float), Wq[sel]
    xs = np.atleast_1d(x)
    return np.array([-(xx ** 2 / (4 * np.pi ** 2)) * np.sum(Wq * special.kv(2, xx * np.sqrt(q)) / q) for xx in xs])


# =====================================================================================================  potential pieces
def extra_terms(z, lam, cp, opt):
    """Holomorphic non-gauge part: 3/(8 z^2) + lam_eff z^4/4 - mc2 z^2/2 + finite-volume matching (+ CW).
    opt keys: match in {'none','gt'}; higgs in {'none','U','L','F','LQ'}; cw (bool); rg_beta (float or None); mc2."""
    z = np.asarray(z)
    z2 = z * z
    g2, gY, yt = cp["g2"], cp["gY"], cp["yt"]
    gZ2 = g2 * g2 + gY * gY
    rgb = opt.get("rg_beta")
    lam_eff = lam + (0.5 * rgb * np.log(1 + z2) if rgb is not None else 0.0)
    V = 3.0 / (8.0 * z2) + lam_eff * z2 * z2 / 4.0 - 0.5 * opt.get("mc2", 0.0) * z2
    match = opt.get("match", "none")
    hig = opt.get("higgs", "none")
    if match == "gt":
        if hig == "F":
            V = V + 2 * (2 * dP_c(g2 * g2 / 4, z2) + dP_c(g2 * g2 / 4 + lam, z2)) \
                + (2 * dP_c(gZ2 / 4, z2) + dP_c(gZ2 / 4 + lam, z2))
        else:
            V = V + 6 * dP_c(g2 * g2 / 4, z2) + 3 * dP_c(gZ2 / 4, z2)
        V = V - 12 * DA(yt * yt / 2 * z2, "pos")
        if hig in ("U", "L", "F"):
            V = V + dP_c(3 * lam, z2)
        if hig == "L":
            V = V + 3 * dP_c(lam, z2)
        if hig == "LQ":
            V = V + 3 * lam * (2 * TB_HALF_REF) * z2
    if opt.get("cw", False):
        lmu = np.log(1 + z2) if rgb is not None else 0.0
        terms = [(6, g2 * g2 / 4, 5 / 6), (3, gZ2 / 4, 5 / 6), (-12, yt * yt / 2, 1.5)]
        if hig in ("U", "L"):
            terms.append((1, 3 * lam, 1.5))
        if hig == "L":
            terms.append((3, lam, 1.5))
        for n, c, C in terms:
            m2 = c * z2
            LOG = np.log(m2) if c > 0 else np.log(-m2)
            V = V + n * m2 * m2 / (64 * np.pi ** 2) * (LOG - C - lmu)
    return V


def higgs_rmax(lam, opt, cap=26.0):
    if opt.get("match") == "gt" and opt.get("higgs", "none") in ("U", "L", "F") and lam < 0:
        return min(cap, 0.9 * 2 * np.pi / math.sqrt(3 * abs(lam)))
    return cap


# =====================================================================================================  radial solvers
class Radial:
    def __init__(self, r, U, fit_rmax=0.1):
        self.r, self.U = np.asarray(r, float), np.asarray(U, float)
        self.W = CubicSpline(self.r, self.r ** 2 * self.U)
        sel = self.r <= fit_rmax
        x = self.r[sel]
        A = np.vstack([x ** j for j in range(6)]).T
        coef = np.linalg.lstsq(A, (x * x * self.U[sel]), rcond=None)[0]
        self.beta, self.alpha, self.U0 = coef[0], coef[1], coef[2]

    def Uf(self, x):
        return self.W(x) / (x * x)

    def Uder(self, x):
        W, W1, W2 = self.W(x), self.W(x, 1), self.W(x, 2)
        return W / x ** 2, W1 / x ** 2 - 2 * W / x ** 3, W2 / x ** 2 - 4 * W1 / x ** 3 + 6 * W / x ** 4

    def start(self, r0, E):
        s = 0.5 + math.sqrt(0.25 + 2 * self.beta)
        c1 = self.alpha / s
        c2 = (self.alpha * c1 + self.U0 - E) / (2 * s + 1)
        u = r0 ** s * (1 + c1 * r0 + c2 * r0 ** 2)
        du = s * r0 ** (s - 1) + c1 * (s + 1) * r0 ** s + c2 * (s + 2) * r0 ** (s + 1)
        return u, du

    def shoot(self, E, r0, rout, rtol=1e-12, start="frob"):
        u, du = self.start(r0, E) if start == "frob" else (0.0, 1.0)
        W = self.W

        def f(x, y):
            return np.array([y[1], 2.0 * (W(x) / (x * x) - E) * y[0]])

        sol = solve_ivp(f, (r0, rout), np.array([u, du], complex), method="DOP853", rtol=rtol, atol=1e-300)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[0, -1], sol.y[1, -1]

    def w_out(self, x, E):
        U, U1, U2 = self.Uder(x)
        k = np.sqrt(2 * (E - U) + 0j)
        k1 = -U1 / k
        k2 = -(U2 + k1 * k1) / k
        return 1j * k - k1 / (2 * k) + (1j / (2 * k)) * (0.75 * k1 * k1 / (k * k) - 0.5 * k2 / k)

    def pole(self, E0, r0=0.02, rout=14.0, tol=1e-13, maxit=60, dE=None, rtol=1e-12, start="frob", resid="H"):
        # resid 'H': u' - w u (entire in E, linear near the pole on the level-spacing scale; used for all
        # results after 2026-10-13 03:00); 'F': u'/u - w (Moebius-like on the scale gamma; first repro run).
        def F(E):
            u, du = self.shoot(E, r0, rout, rtol, start)
            return (du - self.w_out(rout, E) * u) if resid == "H" else (du / u - self.w_out(rout, E))

        Ea = complex(E0)
        Eb = Ea + (dE if dE is not None else (1e-4 - 0.3e-4j) * max(1.0, abs(Ea)))
        Fa, Fb = F(Ea), F(Eb)
        for it in range(maxit):
            if Fb == Fa:
                break
            Ec = Eb - Fb * (Eb - Ea) / (Fb - Fa)
            Ea, Fa = Eb, Fb
            Eb = Ec
            Fb = F(Eb)
            if abs(Eb - Ea) < tol * max(1.0, abs(Eb)):
                return Eb, it + 1
        raise RuntimeError(f"pole: no convergence from {E0}")

    def dirichlet(self, r_wall, h=0.004, r0=0.01):
        x = np.arange(r0, r_wall, h)[1:]
        d = 1.0 / h ** 2 + self.Uf(x)
        e = -0.5 / h ** 2 * np.ones(len(x) - 1)
        return float(eigvalsh_tridiagonal(d, e, select="i", select_range=(0, 0))[0])

    def dirichlet_rich(self, r_wall, h=0.004, r0=0.01):
        e1, e2, e3 = (self.dirichlet(r_wall, hh, r0) for hh in (h, h / 2, h / 4))
        return (64 * e3 - 20 * e2 + e1) / 45, (e1, e2, e3)

    def landmarks(self, rlo=0.3):
        r, U = self.r, self.U
        sel = r >= rlo
        rs, Us = r[sel], U[sel]
        im = int(np.argmin(Us[rs < 4.0]))
        r_min, U_min = rs[im], Us[im]
        after = rs > r_min
        it = int(np.argmax(Us[after]))
        r_top, U_top = rs[after][it], Us[after][it]
        if it == len(rs[after]) - 1:
            return dict(r_min=r_min, U_min=U_min, r_top=None, U_top=None, r_wall=None)
        beyond = (rs > r_top) & (Us < U_min)
        r_wall = rs[beyond][0] if beyond.any() else rs[-1]
        return dict(r_min=float(r_min), U_min=float(U_min), r_top=float(r_top), U_top=float(U_top), r_wall=float(r_wall))

    def turning_points(self, Q, rlo=0.005):
        r = self.r
        f = self.U - Q
        idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
        g = lambda x: float(self.Uf(x)) - Q
        return [brentq(g, r[i], r[i + 1], xtol=1e-13) for i in idx if r[i] >= rlo]

    def wkb(self, Q):
        tp = self.turning_points(Q)
        if len(tp) < 3:
            return dict(S=None, T=None, gamma=None, tp=tp)
        ra, r1, r2 = tp[0], tp[1], tp[2]
        x, w = np.polynomial.legendre.leggauss(400)
        phi = 0.5 * np.pi * (x + 1)
        rr = ra + (r1 - ra) * (1 - np.cos(phi)) / 2
        T = 2 * np.sum(0.5 * np.pi * w * (r1 - ra) * np.sin(phi) / 2 / np.sqrt(np.maximum(2 * (Q - self.Uf(rr)), 1e-300)))
        rr2 = r1 + (r2 - r1) * (1 - np.cos(phi)) / 2
        S = 2 * np.sum(0.5 * np.pi * w * (r2 - r1) * np.sin(phi) / 2 * np.sqrt(np.maximum(2 * (self.Uf(rr2) - Q), 0)))
        return dict(S=float(S), T=float(T), gamma=float(np.log1p(np.exp(-S)) / T), tp=[ra, r1, r2])

    def guess(self):
        lm = self.landmarks()
        if lm["r_wall"] is None:
            return complex(self.dirichlet(min(self.r[-1], 20.0))), lm, None
        Q = self.dirichlet(lm["r_wall"])
        wk = self.wkb(Q)
        g = wk["gamma"] if wk["gamma"] is not None else 0.05
        return complex(Q, -max(g, 1e-14) / 2), lm, wk


# =====================================================================================================  ECS
def ecs_eigs(U_on_nodes, z, sigma, k=6):
    """Second-order FD on the (complex) contour nodes z (Dirichlet ends); returns eigenvalues nearest sigma."""
    dm = z[1:-1] - z[:-2]
    dp = z[2:] - z[1:-1]
    D = z[2:] - z[:-2]
    main = 1 / (dp * D) + 1 / (dm * D) + U_on_nodes[1:-1]
    up = -1 / (dp * D)
    lo = -1 / (dm * D)
    A = sparse.diags([lo[1:], main, up[:-1]], [-1, 0, 1], format="csc")
    vals = eigs(A, k=k, sigma=sigma, which="LM", return_eigenvectors=False)
    return vals[np.argsort(np.abs(vals - sigma))]


def richardson3(e1, e2, e3):
    return (64 * e3 - 20 * e2 + e1) / 45
