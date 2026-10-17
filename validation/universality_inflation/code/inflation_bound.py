# inflation_bound.py -- ui-blind (V5), own implementation, 2026-10-15
# <N> = int over past light cone of Gamma_HM = (R/12)^2 exp(-384 pi^2 DeltaV/R^2)  (MMR 2021 eqs. 2.2-2.3, 3.2-3.10)
# backgrounds: (a) MMR-type slow-roll start, pivot at N=60, post-inflation radius 3.21/(a0 H0);
#              (b) v11 plateau from the Casimir turning point (H=0), finite cubic torus T^3.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import logsumexp

MP_GEV = 2.435e18
SQ23 = np.sqrt(2.0 / 3.0)


# ---------------- potentials (MP=1) ----------------
def V_plateau(phi, A):            # A = 3 h^2  (v11)  or (3/4) alpha^2 (MMR); V = A (1-e^{-sqrt(2/3) phi})^2
    e = np.exp(-SQ23 * phi)
    return A * (1 - e)**2


def dV_plateau(phi, A):
    e = np.exp(-SQ23 * phi)
    return 2 * A * (1 - e) * SQ23 * e


def V_quad(phi, m2):
    return 0.5 * m2 * phi**2


def dV_quad(phi, m2):
    return m2 * phi


# ---------------- slow-roll normalisation (MMR eq. 3.16) ----------------
def norm_starobinsky(Pz=2.1e-9, N=60.0):
    xe = np.log(1 + 2 / np.sqrt(3))
    # N = (3/4)[e^x - x]_{xe}^{x}
    x = brentq(lambda x: 0.75 * ((np.exp(x) - x) - (np.exp(xe) - xe)) - N, xe + 1e-6, 50)
    phi = x / SQ23
    eps = (4.0 / 3.0) / (np.exp(x) - 1)**2
    # P = V/(24 pi^2 eps), V = A (1-e^{-x})^2
    A = Pz * 24 * np.pi**2 * eps / (1 - np.exp(-x))**2
    alpha = np.sqrt(4 * A / 3)
    return A, alpha, phi


def norm_quadratic(Pz=2.1e-9, N=60.0):
    phi2 = 4 * (N + 0.5)
    eps = 2 / phi2
    m2 = Pz * 24 * np.pi**2 * eps / (0.5 * phi2)
    return m2, np.sqrt(phi2)


# ---------------- background integration ----------------
def bg_from_slowroll(Vf, dVf, par, phi_i):
    V0 = Vf(phi_i, par)
    H0 = np.sqrt(V0 / 3)
    pi0 = -dVf(phi_i, par) / (3 * H0)

    def rhs(t, y):
        phi, p, lna = y
        H = np.sqrt((0.5 * p * p + Vf(phi, par)) / 3)
        return [p, -3 * H * p - dVf(phi, par), H]

    def ev(t, y):     # addot/a = H^2 - p^2/2 /... : addot/a = -(p^2 - V)/3 ... use H^2(1-eps)=0 <-> p^2 = V
        return y[1]**2 - Vf(y[0], par)
    ev.terminal = True; ev.direction = 1
    T = 400.0 / H0
    sol = solve_ivp(rhs, (0, T), [phi_i, pi0, 0.0], method="DOP853", rtol=1e-11, atol=1e-14,
                    events=ev, dense_output=True)
    te = sol.t_events[0][0]
    return sol, te


def bg_v11(A, phi_b):
    """Casimir turning point: H=0, phidot=0 at t=0; rho_C = -V(phi_b) a^-4 (a_b=1)."""
    Vb = V_plateau(phi_b, A)

    def rhs(t, y):
        phi, p, lna, H = y
        rhoC = -Vb * np.exp(-4 * lna)
        return [p, -3 * H * p - dV_plateau(phi, A), H, -0.5 * (p * p + 4.0 / 3.0 * rhoC)]

    def ev(t, y):     # addot/a = Hdot + H^2 = 0
        phi, p, lna, H = y
        rhoC = -Vb * np.exp(-4 * lna)
        return -0.5 * (p * p + 4.0 / 3.0 * rhoC) + H * H
    ev.terminal = True; ev.direction = -1
    h = np.sqrt(A / 3)
    sol = solve_ivp(rhs, (0, 200.0 / h), [phi_b, 0.0, 0.0, 0.0], method="DOP853", rtol=1e-11,
                    atol=1e-16, events=ev, dense_output=True)
    # skip the trivial t=0 root of the event (addot>0 at start, so direction=-1 is fine)
    te = sol.t_events[0][0]
    return sol, te, Vb


# ---------------- finite cubic torus: |ball(rho) cap unit cube| ----------------
def _disk_square(s):
    if s <= 0.5:
        return np.pi * s * s
    if s >= np.sqrt(0.5):
        return 1.0
    return np.pi * s * s - 4 * (s * s * np.arccos(0.5 / s) - 0.5 * np.sqrt(s * s - 0.25))


def ball_cube(rho):
    if rho <= 0:
        return 0.0
    if rho >= np.sqrt(3) / 2:
        return 1.0
    zm = min(rho, 0.5)
    val, _ = quad(lambda z: _disk_square(np.sqrt(max(rho * rho - z * z, 0.0))), -zm, zm, limit=200,
                  epsabs=1e-13, epsrel=1e-11)
    return val


_rg = np.linspace(0, 0.8661, 1733)
_fg = np.array([ball_cube(r) for r in _rg])
_ball_sp = CubicSpline(_rg, _fg)


def ball_cube_fast(rho):
    rho = np.asarray(rho, dtype=float)
    out = np.where(rho >= np.sqrt(3) / 2, 1.0, _ball_sp(np.clip(rho, 0, 0.8661)))
    return np.where(rho <= 0.5, 4 * np.pi / 3 * rho**3, out)


# ---------------- B(R) table ----------------
class BTable:
    def __init__(self, veff, Rmin, Rmax, nodes=36):
        self.lR = np.linspace(np.log(Rmin), np.log(Rmax), nodes)
        out = [veff.B_HM(np.exp(x)) for x in self.lR]
        self.B = np.array([o[0] for o in out]); self.hb = np.array([o[1] for o in out])
        self.mus = np.array([o[2] for o in out])
        self.sp = CubicSpline(self.lR, self.B)

    def __call__(self, R):
        return self.sp(np.log(R))


# ---------------- <N> for MMR-type backgrounds ----------------
class MMRBackground:
    def __init__(self, kind, Pz=2.1e-9, Npiv=60.0, kstar_over_H0=0.05 * 299792.458 / 67.4, c_post=3.21):
        self.kind = kind
        if kind == "starobinsky":
            A, alpha, _ = norm_starobinsky(Pz, Npiv)
            self.par = A; self.alpha = alpha
            Vf, dVf = V_plateau, dV_plateau
            _, _, phi_i = norm_starobinsky(Pz, Npiv + 15)
        elif kind == "quadratic":
            m2, _ = norm_quadratic(Pz, Npiv)
            self.par = m2; self.m = np.sqrt(m2)
            Vf, dVf = V_quad, dV_quad
            _, phi_i = norm_quadratic(Pz, Npiv + 15)
        else:
            raise ValueError(kind)
        sol, te = bg_from_slowroll(Vf, dVf, self.par, phi_i)
        lna_e = sol.sol(te)[2]
        tt = np.linspace(0, te, 400001)
        Y = sol.sol(tt)
        phi, p, lna = Y
        Vv = Vf(phi, self.par)
        H = np.sqrt((0.5 * p * p + Vv) / 3)
        N = lna_e - lna                               # e-folds before end (addot=0)
        R = 4 * Vv - p * p                            # MP=1
        m = N <= Npiv + 1.0
        Nm, Hm, Rm = N[m][::-1], H[m][::-1], R[m][::-1]
        self.N, self.H, self.R = Nm, Hm * MP_GEV, Rm * MP_GEV**2
        Hp = np.interp(Npiv, self.N, self.H)
        # physical past-light-cone radius at N (GeV^-1)
        integ = np.concatenate([[0.0], np.cumsum(0.5 * (np.exp(self.N[1:]) / self.H[1:] + np.exp(self.N[:-1]) / self.H[:-1])
                                                 * np.diff(self.N))])
        self.rho = c_post * kstar_over_H0 * np.exp(Npiv - self.N) / Hp + np.exp(-self.N) * integ
        self.Npiv = Npiv
        self.Rrange = (self.R.min(), np.interp(Npiv, self.N, self.R))

    def lnN(self, btab, Nstart=60.0):
        m = self.N <= Nstart
        N, H, R, rho = self.N[m], self.H[m], self.R[m], self.rho[m]
        lg = np.log(4 * np.pi / 3) + 3 * np.log(rho) - np.log(H) + 2 * np.log(R / 12) - btab(R)
        w = np.zeros_like(N); dN = np.diff(N); w[:-1] += 0.5 * dN; w[1:] += 0.5 * dN
        tot = logsumexp(lg, b=w)
        ipk = int(np.argmax(lg))
        return float(tot), dict(N_peak=float(N[ipk]), H_peak=float(H[ipk]), R_peak=float(R[ipk]),
                                B_peak=float(btab(R[ipk])))


class V11Background:
    def __init__(self, h_over_MP=6.116e-6, phi_b=5.5794, LbMP=790.0, Npost_v11=68.25,
                 r_after_over_L=3.21 / 6.0, torus=True, dlnLe=0.0):
        """Physical torus side at the end of inflation fixed to the v11 value L_e = LbMP*exp(Npost_v11)
        (790 e^{68.25} = 722 e^{68.34}: identical for free fields and with the zero-mode energy, since
        L_0 is fixed); the background shape H(t), R(t) is integrated here from phi_b.
        dlnLe: optional shift of ln L_e (e.g. ln(L0/0.95 d_LSS))."""
        A = 3 * h_over_MP**2
        sol, te, Vb = bg_v11(A, phi_b)
        self.te = te
        tt = np.unique(np.concatenate([np.linspace(0, 1, 2001)**2 * min(5.0 / h_over_MP, te),
                                       np.linspace(min(5.0 / h_over_MP, te), te, 400001)]))
        Y = sol.sol(tt)
        phi, p, lna, H = Y
        self.Npost_raw = float(sol.sol(te)[2])
        Vv = V_plateau(phi, A)
        R = 4 * Vv - p * p
        inva = np.exp(-lna)
        eta = np.concatenate([[0.0], np.cumsum(0.5 * (inva[1:] + inva[:-1]) * np.diff(tt))])
        # comoving torus side (a=1 at this integration's turning point) reproducing L_e
        Lb = LbMP * np.exp(Npost_v11 + dlnLe - self.Npost_raw)
        r = r_after_over_L * Lb + (eta[-1] - eta)
        self.t, self.lna, self.H, self.R, self.r, self.Lb = tt, lna, H, R, r, Lb
        rho = r / Lb
        if torus:
            frac = ball_cube_fast(rho)
        else:
            frac = 4 * np.pi / 3 * rho**3
        self.frac_end = float(frac[-1])
        self.lnVcom = 3 * np.log(Lb) + np.log(frac)
        self.Rrange = (R[-1] * MP_GEV**2 * 0.999, R.max() * MP_GEV**2 * 1.001)
        self.phi, self.p = phi, p
        # constraint check: 3H^2 = p^2/2 + V + rhoC
        rhoC = -Vb * np.exp(-4 * lna)
        self.constraint_max = float(np.max(np.abs(3 * H * H - (0.5 * p * p + Vv + rhoC))) / Vb)

    def lnN(self, btab):
        RG = self.R * MP_GEV**2
        lg = 3 * self.lna + self.lnVcom + 2 * np.log(self.R / 12) - btab(RG)   # MP units for volume*Gamma
        w = np.zeros_like(self.t); dt = np.diff(self.t); w[:-1] += 0.5 * dt; w[1:] += 0.5 * dt
        tot = logsumexp(lg, b=w)
        ipk = int(np.argmax(lg))
        # fraction from the first 10 e-folds after the turn
        m10 = self.lna < 10
        f10 = float(np.exp(logsumexp(lg[m10], b=w[m10]) - tot)) if m10.any() else 0.0
        Nb = self.lna[-1] - self.lna[ipk]
        return float(tot), dict(N_before_end_peak=float(Nb), H_peak_GeV=float(self.H[ipk] * MP_GEV),
                                R_peak=float(RG[ipk]), B_peak=float(btab(RG[ipk])), frac_first10=f10)
