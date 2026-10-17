# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/inflation_stability/infl_core.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] All-mode Higgs stability during inflation (D127): RG-improved potential V = xi R h^2/2 + lam h^4/4 (+ alpha R^2/144), Hawking-Moss rate Gamma = (R/12)^2 e^-B, past-light-cone count <N> = 1, Starobinsky-shaped plateau (h/M_P = 6.116e-6), xi running to mu_b.  Paper: PRD Sec. VII; PRL one sentence.  Units: GeV / M_P.

C155 / D127 (inflation-stability): SM Higgs stability during plateau inflation.

Ingredients (all literature-anchored, see D127 Sec. 2):
  * 1-loop curved-space SM effective potential in the dS form, 31 degrees of freedom
    (MNRS 2018 arXiv:1804.02020 Eq. 5.2 + Tables 1-2; MMR 2021 arXiv:2011.03763 Eqs. 2.9-2.10), gauge zeta_W = zeta_Z = zeta.
    De Sitter form used with H^2 -> R/12 (MMR 2021 p.7).
  * RG improvement: mu* from DeltaV_loops(h, mu*, R) = 0 (MMR 2021 Eq. 2.12), V_RGI = xi R h^2/2 + lam h^4/4 + alpha R^2/144
    (Eq. 2.13). Variants: "her" mu^2 = h^2 + R with the loop term kept (Herranen 2014 Eq. 10), "tree" same mu without loops.
  * xi(mu) = 1/6 + (xi_EW - 1/6) Xi(mu), 16 pi^2 d ln Xi/d ln mu = 12 lam + 6 yt^2 - 3/2 g'^2 - 9/2 g^2 (MMR 2021 Eq. 2.15);
    16 pi^2 d alpha/d ln mu = 288 xi^2 - 96 xi - 1751/30 (Eq. 2.16); xi_EW is the MSbar value at mu = m_t (MMR 2022 p.17).
  * Hawking-Moss: Gamma = (R/12)^2 exp(-B), B = 384 pi^2 DeltaV / R^2 (MMR 2022 Eqs. 4.6-4.7).
  * Past light cone: <N> = int dN 4pi/(3H) (a_inf [eta0 - eta])^3 e^{-3N} Gamma (MMR 2022 Eqs. 4.5, 4.11-4.12), eta0 = 3.21/H0.
  * Optional R^2-gravity extras (MMR 2022 Eqs. 4.16-4.18) for the literature reproduction only.
Units: GeV; inflaton background in reduced-Planck units internally. SM couplings from sm_provider (function call, replaceable).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import math
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True  # do not write __pycache__ into the sm_mapping directory of T1
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = _zp.WORK
SMDIR = os.path.join(ROOT, _zp.ZMLIB)
if SMDIR not in sys.path:
    sys.path.insert(0, SMDIR)
import sm_provider as SP  # noqa: E402
import sm_rge as RG  # noqa: E402

PI = math.pi
K = 1.0 / (16.0 * PI ** 2)
MPBAR = 2.435e18
MU_B = MPBAR / 722.0
A23 = math.sqrt(2.0 / 3.0)
GEV_PER_INV_MPC = 1.97327e-16 / 3.08568e22  # hbar c / Mpc
KSTAR = 0.05 * GEV_PER_INV_MPC            # pivot 0.05 Mpc^-1 in GeV (a0 = 1)
IDX = dict(g3=0, g2=1, gY=2, yt=3, yb=4, ytau=5, lam=6, v=7)

# MNRS 2018 Table 1 (Higgs-coupled) and Table 2 (not coupled): (n_i, d_i, n'_i, label); masses built in dV_loops
MMR_TABLE1 = dict(lam=0.126249, yt=0.934843, g2=0.648382, gY=0.358729)  # MMR 2021 Table 1, mu_EW = m_t = 172.76


class SMCouplings:
    """MSbar SM couplings on ln(mu) in [ln Mt, ln mu_max] + xi/alpha running machinery."""

    def __init__(self, Mt, als, Mh=125.13, variant="central", override=None, loops=3, mu_max=1.2e19, n=900,
                 xi_yuk="top"):
        self.Mt, self.als, self.Mh, self.variant = Mt, als, Mh, variant
        self.override, self.loops_override = override, loops
        t = np.linspace(math.log(Mt), math.log(mu_max), n)
        if override is None:
            c = SP.couplings(Mt, als, Mh, np.exp(t), variant)
            Y = np.array([c[k] for k in ("g3", "g2", "gY", "yt", "yb", "ytau", "lam")])
        else:
            c0 = SP.coup_point(Mt, als, Mh, Mt, variant)
            y0 = [c0["g3"], c0["g2"], c0["gY"], c0["yt"], c0["yb"], c0["ytau"], c0["lam"], c0["v"]]
            for k, v in override.items():
                y0[IDX[k]] = v
            sol = RG.solve(y0, Mt, mu_max * 1.001, loops=loops)
            Y = sol.sol(t)[:7]
        g3, g2, gY, yt, yb, ytau, lam = Y
        gx = K * (12 * lam + 6 * yt ** 2 - 1.5 * gY ** 2 - 4.5 * g2 ** 2)
        if xi_yuk == "all":
            gx = gx + K * (6 * yb ** 2 + 2 * ytau ** 2)
        lnXi = CubicSpline(t, gx).antiderivative()(t)
        I2 = CubicSpline(t, np.exp(2 * lnXi)).antiderivative()(t)
        self.t, self.t0, self.tmin, self.tmax = t, t[0], t[0], t[-1]
        self.sp = CubicSpline(t, np.vstack([lam, g2, gY, yt, yb, ytau, lnXi, I2]).T)
        self.lam_raw = lam

    def at(self, mu):
        c = self.sp(np.log(np.atleast_1d(mu)))
        return dict(lam=c[:, 0], g2=c[:, 1], gY=c[:, 2], yt=c[:, 3], Xi=np.exp(c[:, 6]))

    def xi_at(self, xi_ew, mu):
        return 1.0 / 6 + (xi_ew - 1.0 / 6) * float(np.exp(self.sp(math.log(mu))[6]))

    def xi_ew_from(self, xi_mu, mu):
        return 1.0 / 6 + (xi_mu - 1.0 / 6) / float(np.exp(self.sp(math.log(mu))[6]))

    # ---------------------------------------------------------------- 1-loop curved-space potential
    def dV_loops(self, h, t, HR2, delta, zeta=1.0, C=None):
        if C is None:
            C = self.sp(t)
        lam, g2, gY, yt, yb, ytau = C[..., 0], C[..., 1], C[..., 2], C[..., 3], C[..., 4], C[..., 5]
        Xi = np.exp(C[..., 6])
        h2 = h * h
        mu2 = np.exp(2.0 * t)
        H4 = HR2 * HR2
        mW = 0.25 * g2 * g2 * h2
        mZ = 0.25 * (g2 * g2 + gY * gY) * h2
        mtq, mbq, mta = 0.5 * yt * yt * h2, 0.5 * yb * yb * h2, 0.5 * ytau * ytau * h2
        th = 12.0 * delta * Xi * HR2
        one = np.ones_like(h2)

        def T(n, d, npr, M2):
            L = np.log(np.abs(M2) / mu2)
            return n * M2 * M2 * (L - d) + npr * H4 * L

        S = (T(2, 1.5, -34 / 15, mW + HR2) + T(6, 5 / 6, -34 / 5, mW + HR2) + T(-2, 1.5, 4 / 15, mW - 2 * HR2)
             + T(1, 1.5, -17 / 15, mZ + HR2) + T(3, 5 / 6, -17 / 5, mZ + HR2) + T(-1, 1.5, 2 / 15, mZ - 2 * HR2)
             + T(-12, 1.5, 38 / 5, mtq + HR2) + T(-12, 1.5, 38 / 5, mbq + HR2) + 4 * T(-12, 1.5, 38 / 5, HR2 * one)
             + T(-4, 1.5, 38 / 15, mta + HR2) + 2 * T(-4, 1.5, 38 / 15, HR2 * one)
             + T(1, 1.5, -2 / 15, 3 * lam * h2 + th)
             + T(2, 1.5, -4 / 15, lam * h2 + zeta * mW + th) + T(1, 1.5, -2 / 15, lam * h2 + zeta * mZ + th)
             + T(-2, 1.5, 4 / 15, zeta * mW - 2 * HR2) + T(-1, 1.5, 2 / 15, zeta * mZ - 2 * HR2)
             # Table 2: photon, gluons, neutrinos, photon and gluon ghosts
             + T(1, 1.5, -17 / 15, HR2 * one) + T(3, 5 / 6, -17 / 5, HR2 * one) + T(-1, 1.5, 2 / 15, -2 * HR2 * one)
             + T(8, 1.5, -136 / 15, HR2 * one) + T(24, 5 / 6, -136 / 5, HR2 * one)
             + T(-8, 1.5, 16 / 15, -2 * HR2 * one)
             + 3 * T(-2, 1.5, 19 / 15, HR2 * one) + T(-1, 1.5, 2 / 15, -2 * HR2 * one)
             + T(-8, 1.5, 16 / 15, -2 * HR2 * one))
        return S / (64 * PI ** 2)

    def t_star(self, h, HR2, delta, zeta=1.0, span=5.0, ng=41, nb=48):
        """mu* from DeltaV_loops = 0 (vectorised); root closest to ln sqrt(h^2+R). NaN if no root in window."""
        h = np.atleast_1d(np.asarray(h, float))
        tc = 0.5 * np.log(h * h + 12 * HR2)
        off = np.linspace(-span, span, ng)
        tg = np.clip(tc[:, None] + off[None, :], self.tmin, self.tmax)
        hh = np.broadcast_to(h[:, None], tg.shape)
        F = self.dV_loops(hh.ravel(), tg.ravel(), HR2, delta, zeta).reshape(tg.shape)
        s = np.sign(F)
        ch = (s[:, :-1] * s[:, 1:]) < 0
        out = np.full(h.shape, np.nan)
        if not ch.any():
            return out
        jc = ng // 2
        dist = np.where(ch, np.abs(np.arange(ng - 1)[None, :] + 0.5 - jc), np.inf)
        j = np.argmin(dist, axis=1)
        ok = np.isfinite(dist[np.arange(len(h)), j])
        a = tg[np.arange(len(h)), j].copy()
        b = tg[np.arange(len(h)), j + 1].copy()
        fa = F[np.arange(len(h)), j].copy()
        for _ in range(nb):
            m = 0.5 * (a + b)
            fm = self.dV_loops(h, m, HR2, delta, zeta)
            left = np.sign(fm) == np.sign(fa)
            a = np.where(left, m, a)
            fa = np.where(left, fm, fa)
            b = np.where(left, b, m)
        out[ok] = (0.5 * (a + b))[ok]
        return out

    def V_rgi(self, h, HR2, delta, presc="mmr", zeta=1.0, r2=None):
        h = np.atleast_1d(np.asarray(h, float))
        if presc == "mmr":
            t = self.t_star(h, HR2, delta, zeta)
        elif presc in ("her", "tree"):
            t = np.clip(0.5 * np.log(h * h + 12 * HR2), self.tmin, self.tmax)
        elif presc in ("her10", "tree10"):  # mu^2 = 10 (h^2 + R)  (Herranen fn. 20 range)
            t = np.clip(0.5 * np.log(10 * (h * h + 12 * HR2)), self.tmin, self.tmax)
        elif presc in ("her01", "tree01"):  # mu^2 = 0.1 (h^2 + R)
            t = np.clip(0.5 * np.log(0.1 * (h * h + 12 * HR2)), self.tmin, self.tmax)
        else:
            raise ValueError(presc)
        bad = ~np.isfinite(t)
        t = np.where(bad, self.tmin, t)
        C = self.sp(t)
        lam = C[:, 0]
        xi = 1.0 / 6 + delta * np.exp(C[:, 6])
        al = K * (-(1991.0 / 30) * (t - self.t0) + 288 * delta * delta * C[:, 7])
        R = 12 * HR2
        if r2 is None:
            meff2, leff = xi * R, lam
        else:  # MMR 2022 Eqs. 4.17-4.18 (R^2 gravity, Einstein frame); r2: M (dimensionless), e=exp(-sqrt(2/3)phi/MP), eps
            Xq = xi - 1.0 / 6
            M2MP2 = (r2["M"] * MPBAR) ** 2
            e, eps, H2 = r2["e"], r2["eps"], r2["H2"]
            base = xi * R + 3 * M2MP2 * Xq * (1 - e) * e
            meff2 = base + Xq * H2 * 2 * eps
            leff = (lam + 3 * r2["M"] ** 2 * Xq ** 2 * e * e + 4 * Xq ** 3 * H2 * 2 * eps / MPBAR ** 2
                    + 4 * Xq ** 2 / MPBAR ** 2 * base)
        V = 0.5 * meff2 * h * h + 0.25 * leff * h ** 4 + al * HR2 * HR2
        if presc.startswith("her"):
            V = V + self.dV_loops(h, t, HR2, delta, zeta, C)
        V = np.where(bad, np.nan, V)
        return V, t, xi, lam

    def barrier(self, HR2, delta, presc="mmr", zeta=1.0, r2=None, hmax=MPBAR, want_curv=False):
        """Barrier height above the false vacuum h=0 of V_RGI(h) at fixed R=12 HR2."""
        H = math.sqrt(HR2)
        V0 = self.V_rgi([0.0], HR2, delta, presc, zeta, r2)[0][0]
        hg = H * np.logspace(-1.0, math.log10(min(hmax / H, 3e4)), 150)
        V, t, xi, lam = self.V_rgi(hg, HR2, delta, presc, zeta, r2)
        res = dict(V0=V0, flag="ok")
        if not np.isfinite(V0) or np.isnan(V).any():
            res["flag"] = "nan_mu*"
            Vf = np.where(np.isnan(V), -np.inf, V)
        else:
            Vf = V
        if Vf[0] <= V0:  # potential decreasing at the origin: false vacuum displaced (xi(mu*) < 0 or tachyonic)
            res.update(B=0.0, flag="no_fv", hbar=0.0, dV=0.0)
            return res
        i = int(np.argmax(Vf))
        if i >= len(hg) - 1:
            res.update(B=np.inf, flag="no_barrier", hbar=np.inf, dV=np.inf)
            return res
        hr = np.exp(np.linspace(math.log(hg[max(i - 1, 0)]), math.log(hg[i + 1]), 41))
        Vr = self.V_rgi(hr, HR2, delta, presc, zeta, r2)[0]
        j = int(np.nanargmax(Vr))
        Vm, hb = Vr[j], hr[j]
        if 0 < j < len(hr) - 1:
            y0, y1, y2 = Vr[j - 1], Vr[j], Vr[j + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                Vm = y1 - 0.125 * (y2 - y0) ** 2 / den
        dV = Vm - V0
        res.update(B=(8 * PI ** 2 / 3) * dV / HR2 ** 2, hbar=hb, dV=dV, t_bar=float(t[i]), xi_bar=float(xi[i]),
                   lam_bar=float(lam[i]))
        if want_curv:
            hs = 0.05 * H
            Vs = self.V_rgi([hs, 2 * hs], HR2, delta, presc, zeta, r2)[0]
            m2_0 = (8 * (Vs[0] - V0) - (Vs[1] - V0)) / (3 * hs * hs)  # quadratic+quartic elimination
            dh = 0.02 * hb
            Vb = self.V_rgi([hb - dh, hb, hb + dh], HR2, delta, presc, zeta, r2)[0]
            res.update(m2_0=m2_0, m2_bar=(Vb[0] - 2 * Vb[1] + Vb[2]) / dh ** 2)
        return res


# -------------------------------------------------------------------- plateau background
def plateau_background(V0, phi0, n_span=200.0, slowroll=True):
    """V = V0 (1 - exp(-sqrt(2/3) phi))^2 (M_P = 1). Forward e-folds n; stops at R = 0 (eps = 2)."""
    def V(p):
        return V0 * (1 - math.exp(-A23 * p)) ** 2

    def dVV(p):
        e = math.exp(-A23 * p)
        return 2 * A23 * e / (1 - e)

    def rhs(n, y):
        p, q = y
        eps = 0.5 * q * q
        return [q, -(3 - eps) * (q + dVV(p))]

    def ev(level):
        f = lambda n, y: 0.5 * y[1] ** 2 - level  # noqa: E731
        f.terminal = level >= 2.0
        f.direction = 1
        return f

    q0 = -dVV(phi0) if slowroll else 0.0
    evs = [ev(1.0 / 32), ev(0.25), ev(1.0), ev(2.0)]
    sol = solve_ivp(rhs, (0.0, n_span), [phi0, q0], method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True,
                    events=evs)
    n_ev = {lv: float(sol.t_events[k][0]) for k, lv in enumerate((1.0 / 32, 0.25, 1.0, 2.0))
            if len(sol.t_events[k])}

    def state(n):
        y = sol.sol(n)
        p, q = y[0], y[1]
        eps = 0.5 * q * q
        Vv = V0 * (1 - np.exp(-A23 * p)) ** 2
        H = np.sqrt(Vv / (3 - eps))  # M_P units
        return p, q, eps, H

    return dict(sol=sol, n_ev=n_ev, state=state, V0=V0, phi0=phi0, Vfun=V)


def a_inf_mmr(bg, n_ref, H0=1.5e-42):
    """MMR 2022 Eq. 4.2: a_inf = (H0 e^60 / 1e16 GeV) V^{1/4}(phi_inf) / H_inf at the reference point."""
    p, q, eps, H = bg["state"](n_ref)
    Vg = bg["Vfun"](float(p)) * MPBAR ** 4
    return (H0 * math.exp(60.0) / 1e16) * Vg ** 0.25 / (float(H) * MPBAR)


def a_inf_pivot(bg, n_ref, Nstar, n_end):
    """Pivot k* = 0.05/Mpc exits Nstar e-folds before n_end (eps=1): a(n_end) = k* e^Nstar / H(n_end - Nstar)."""
    Hs = float(bg["state"](n_end - Nstar)[3]) * MPBAR
    a_end = KSTAR * math.exp(Nstar) / Hs
    return a_end * math.exp(n_ref - n_end)


def ball_cube_fraction(r, L, nmc=0):
    """Volume of (ball radius r) intersect (cube side L, centred) / ball volume; analytic for r < L/sqrt(2)."""
    if r <= L / 2:
        return 1.0
    if r < L / math.sqrt(2):
        hc = r - L / 2
        cap = PI * hc * hc * (3 * r - hc) / 3
        return 1.0 - 6 * cap / (4 * PI / 3 * r ** 3)
    raise ValueError("r >= L/sqrt2 not needed here")


class Count:
    """<N> over the past light cone for a given background, reference point and B(N)."""

    def __init__(self, bg, n_ref, a_inf, H0, N_start, N_end_list, fT=1.0, dN_fine=0.02, N_fine=16.0):
        self.bg, self.n_ref, self.a_inf, self.H0, self.fT = bg, n_ref, a_inf, H0, fT
        Nmin = min(N_end_list)
        Ng = np.unique(np.concatenate([np.arange(Nmin, Nmin + N_fine, dN_fine), np.linspace(Nmin + N_fine, N_start, 160)]))
        Ng = Ng[Ng <= N_start]
        self.N = Ng
        p, q, eps, H = bg["state"](n_ref - Ng)
        self.eps, self.Hg = eps, H * MPBAR
        self.R = 6 * self.Hg ** 2 * (2 - eps)
        self.HR2 = self.R / 12
        self.e = np.exp(-A23 * p)
        # eta_tilde: d eta~/dN = -eta~ - 1/(a_inf H), eta~(0)=0 at the reference point (N = 0)
        Nz = np.linspace(0.0, N_start, 4001)
        Hz = bg["state"](n_ref - Nz)[3] * MPBAR
        integ = np.exp(Nz) / (a_inf * Hz)
        cum = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(Nz))])
        eta_t = -np.exp(-Nz) * cum
        self.eta_t = np.interp(Ng, Nz, eta_t)
        r = a_inf * (3.21 * np.exp(-Ng) / H0 - self.eta_t)
        self.lngeo = math.log(4 * PI / 3) + 3 * np.log(r) - np.log(self.Hg) + math.log(fT)
        self.N_end_list = list(N_end_list)
        self.N_start = N_start

    def lnN(self, B, pref=None):
        """B on self.N grid -> dict N_end -> (ln<N>, N_peak). Gamma = HR2^2 e^-B (HM) times optional prefactor."""
        lng = self.lngeo + 2 * np.log(self.HR2) - B
        if pref is not None:
            lng = lng + np.log(pref)
        out = {}
        for Ne in self.N_end_list:
            m = self.N >= Ne - 1e-12
            x, y = self.N[m], lng[m]
            ymax = np.max(y)
            w = np.exp(y - ymax)
            I = np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(x))
            out[Ne] = (ymax + math.log(max(I, 1e-300)), float(x[np.argmax(y)]))
        return out


def B_on_grid(sm, cnt, delta, presc="mmr", zeta=1.0, r2M=None, nR=26, want_pref=False, gauss=False):
    """B(N) on the Count grid. Field-theory model: B depends on R only -> spline in ln R.
    R^2 model: direct evaluation on a thinned N grid, linear interpolation.
    gauss=True: Gaussian stochastic exponent B_G = (4 pi^2/3) V''(0) h_bar^2 / H^4 (Espinosa 2015 Eqs. 36-38 type;
    quadratic part only) instead of the HM exponent."""
    if r2M is None:
        lnR = np.log(cnt.HR2)
        Rg = np.linspace(lnR.min(), lnR.max(), nR)
        Bg, P = [], []
        for x in Rg:
            b = sm.barrier(math.exp(x), delta, presc, zeta, want_curv=(want_pref or gauss))
            if gauss and b.get("flag") == "ok" and b.get("m2_0", -1) > 0:
                Bg.append(min((4 * PI ** 2 / 3) * b["m2_0"] * b["hbar"] ** 2 / math.exp(2 * x), 1e5))
            else:
                Bg.append(min(b["B"], 1e5))
            if want_pref:
                P.append(_kramers(b, math.exp(x)))
        Bg = np.array(Bg)
        Bi = CubicSpline(Rg, Bg)(lnR) if np.all(np.isfinite(Bg)) and np.all(Bg > 0) else np.interp(lnR, Rg, Bg)
        if want_pref:
            return Bi, np.exp(np.interp(lnR, Rg, np.log(np.maximum(P, 1e-300))))
        return Bi
    # R^2: evaluate at thinned grid
    idx = np.unique(np.concatenate([np.arange(0, len(cnt.N), 10), [len(cnt.N) - 1]]))
    Bs = []
    for k in idx:
        r2 = dict(M=r2M, e=float(cnt.e[k]), eps=float(cnt.eps[k]), H2=float(cnt.Hg[k] ** 2))
        b = sm.barrier(float(cnt.HR2[k]), delta, presc, zeta, r2=r2)
        Bs.append(min(b["B"], 1e5))
    return np.interp(cnt.N, cnt.N[idx], np.array(Bs))


def _kramers(b, HR2):
    """Overdamped stochastic (Fokker-Planck) escape prefactor relative to the HM (R/12)^2:
    Gamma_FP = 2 * H^2 sqrt(V''_0 |V''_bar|)/(8 pi^2) e^-B (both signs of h); returns Gamma_FP/(HR2^2 e^-B)."""
    if b.get("flag") != "ok" or "m2_0" not in b or b["m2_0"] <= 0 or b["m2_bar"] >= 0:
        return 1.0
    return 2 * HR2 * math.sqrt(b["m2_0"] * abs(b["m2_bar"])) / (8 * PI ** 2) / HR2 ** 2


def xi_bound(sm, cnt, presc="mmr", zeta=1.0, r2M=None, lo=0.0, hi=0.16, xtol=2e-5, kramers=False, gauss=False):
    """Lower bound xi_EW with <N> = 1 for each N_end in cnt.N_end_list."""
    cache = {}

    def lnN_all(xi):
        if xi not in cache:
            d = xi - 1.0 / 6
            if kramers:
                B, P = B_on_grid(sm, cnt, d, presc, zeta, r2M, want_pref=True)
                cache[xi] = cnt.lnN(B, P)
            else:
                cache[xi] = cnt.lnN(B_on_grid(sm, cnt, d, presc, zeta, r2M, gauss=gauss))
        return cache[xi]

    out = {}
    for Ne in cnt.N_end_list:
        f = lambda x: lnN_all(x)[Ne][0]  # noqa: E731
        flo, fhi = f(lo), f(hi)
        if fhi > 0:
            out[Ne] = dict(xi_ew=None, note=">%g" % hi, lnN_hi=fhi)
            continue
        if flo < 0:
            out[Ne] = dict(xi_ew=lo, note="<=%g" % lo, lnN_lo=flo)
            continue
        x = brentq(f, lo, hi, xtol=xtol)
        out[Ne] = dict(xi_ew=x, N_peak=lnN_all(x)[Ne][1])
    return out
