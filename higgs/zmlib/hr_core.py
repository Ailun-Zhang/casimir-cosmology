# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/realtime_2d/hr_core.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations/code/20261013_claude_v12_r2/higgs_final/hr_core.py on 2026-10-14 by realtime-2d (C157/D129); code unchanged unless a later '# R2D:' comment says otherwise.
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_realtime/hr_core.py on 2026-10-13 by agent higgs-final (C144/D116).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Real-time zero-mode TDSE through the turn (D111): conformal frame, exact lens transform to the physical frame, Wigner/TWA hand-over, survival measures.  Paper: PRD App. H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C139 / D111 core: real-time TDSE for the SM-Higgs zero mode along the v11 Casimir turn.

r-frame (conformal, time eta, d eta = du/hL, hL = hL_b sqrt(cosh 2u)):
    i d_eta psi = [-1/2 d_r^2 + U_nc(r;u) - (1-6 xi)(hL)^2 r^2 - i CAP(r)] psi
y-frame (exact lens transform, y = r/(hL) = phi/h, time u = ht, M = (hL)^3):
    Phi(y) = (hL)^{1/2} exp(-i (L'/L) r^2/2) psi(hL y),  L'/L = hL tanh 2u
    i d_u Phi = [-d_y^2/(2M) + U_nc(hL y;u)/hL + 6 xi M y^2 - i CAP(y,u)] Phi
    (= physical-frame zero mode: L_phys = L^3[phidot^2/2 - 6 xi h^2 phi^2 - V], V = U_nc(L phi)/L^4)
Space: 4th-order 5-point Laplacian, uniform x_j = j dx, Dirichlet at 0 (odd ghost) -> regular r^s behaviour.
Time: Crank-Nicolson, H at the midpoint, constant/updated energy shift (global phase only).
Beyond the absorber onset the real potential is smoothly flattened (tanh) so the outgoing wave number stays bounded;
everything beyond the onset is classically certain runaway (onset >= 2 x exit point in r, >= 1.5 x barrier top in y).
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
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from hr_potential import PotentialTable, HLB, H_GEV, MU_B  # noqa: F401

ETA_INF = 1.31102877714606 / HLB
_UF = np.linspace(0.0, 9.0, 360001)
_FF = 1.0 / (HLB * np.sqrt(np.cosh(2 * _UF)))
_ETAF = np.concatenate([[0.0], np.cumsum(0.5 * (_FF[1:] + _FF[:-1]) * np.diff(_UF))])


def hL(u):
    return HLB * np.sqrt(np.cosh(2.0 * np.asarray(u, float)))


def eta_of_u(u):
    return np.interp(u, _UF, _ETAF)


def u_of_eta(eta):
    return np.interp(eta, _ETAF, _UF)


def eta_rem(u):
    return ETA_INF - float(eta_of_u(u))


class Grid:
    def __init__(self, dx, xmax):
        self.dx = float(dx)
        self.N = int(round(xmax / dx))
        self.x = self.dx * np.arange(1, self.N + 1)
        c = 1.0 / (12.0 * dx * dx)
        self.t0 = np.full(self.N, 30.0 * c)
        self.t0[0] = 29.0 * c
        self.t1 = np.full(self.N - 1, -16.0 * c)
        self.t2 = np.full(self.N - 2, c)


def cn_step(psi, g, pref, V, dt):
    a = 0.5j * dt
    D = pref * g.t0 + V
    o1 = (a * pref) * g.t1
    o2 = (a * pref) * g.t2
    rhs = (1.0 - a * D) * psi
    rhs[:-1] -= o1 * psi[1:]
    rhs[1:] -= o1 * psi[:-1]
    rhs[:-2] -= o2 * psi[2:]
    rhs[2:] -= o2 * psi[:-2]
    ab = np.zeros((5, g.N), complex)
    ab[0, 2:] = o2
    ab[1, 1:] = o1
    ab[2] = 1.0 + a * D
    ab[3, :-1] = o1
    ab[4, :-2] = o2
    return solve_banded((2, 2), ab, rhs, overwrite_ab=True, overwrite_b=True, check_finite=False)


def h_apply(psi, g, pref, V):
    out = (pref * g.t0 + V) * psi
    out[:-1] += pref * g.t1 * psi[1:]
    out[1:] += pref * g.t1 * psi[:-1]
    out[:-2] += pref * g.t2 * psi[2:]
    out[2:] += pref * g.t2 * psi[:-2]
    return out


def h_sparse(g, pref, V):
    return sps.diags([pref * g.t2, pref * g.t1, pref * g.t0 + V, pref * g.t1, pref * g.t2], [-2, -1, 0, 1, 2],
                     format="csc")


def landmarks(x, U, Q):
    """well minimum, barrier top, outer turning point (U = Q) of a real potential."""
    i0 = int(np.searchsorted(x, 0.3))
    d = np.diff(U)
    imin = next(i for i in range(i0, len(d) - 1) if d[i] < 0 <= d[i + 1]) + 1
    itop = next(i for i in range(imin, len(d) - 1) if d[i] > 0 >= d[i + 1]) + 1
    j = itop + int(np.argmax(U[itop:] < Q))
    rex = x[j - 1] + (Q - U[j - 1]) * (x[j] - x[j - 1]) / (U[j] - U[j - 1])
    return dict(i_min=imin, r_min=float(x[imin]), U_min=float(U[imin]), i_top=itop, r_top=float(x[itop]),
                U_top=float(U[itop]), r_exit=float(rex))


class RFrame:
    def __init__(self, tab, g, xi, r_c=10.0, cap0=20.0, ell=1.0):
        self.tab, self.g, self.xi, self.r_c, self.cap0, self.ell = tab, g, float(xi), float(r_c), float(cap0), float(ell)
        x = g.x
        self.ic = int(np.searchsorted(x, r_c))
        w = x[-1] - x[self.ic]
        self.cap = np.where(x > x[self.ic], cap0 * ((x - x[self.ic]) / w) ** 2, 0.0)

    def U_real(self, u):
        x = self.g.x
        return self.tab.U_on(x, u) - (1.0 - 6.0 * self.xi) * float(hL(u)) ** 2 * x * x

    def V(self, u):
        U = self.U_real(u)
        i, x = self.ic, self.g.x
        Fc = (U[i + 1] - U[i - 1]) / (2 * self.g.dx)
        U[i + 1:] = U[i] + Fc * self.ell * np.tanh((x[i + 1:] - x[i]) / self.ell)
        return U - 1j * self.cap


def cnorm(v, dx):
    v = v / np.sqrt(np.sum(v * v) * dx)
    i = int(np.argmax(np.abs(v)))
    return -v if v[i].real < 0 else v


def resonance(g, V, sigma, k=1, inner=None):
    H = h_sparse(g, 0.5, V)
    vals, vecs = spla.eigs(H, k=k, sigma=sigma, which="LM", tol=1e-13, maxiter=20000)
    if inner is None:
        j = int(np.argmin(np.abs(vals - sigma)))
    else:
        fr = [np.sum(np.abs(vecs[:inner, j]) ** 2) / np.sum(np.abs(vecs[:, j]) ** 2) for j in range(len(vals))]
        j = int(np.argmax(np.array(fr) - 10 * np.abs(vals.imag)))
    return complex(vals[j]), cnorm(vecs[:, j], g.dx)


def evolve_r(psi, fr, u0, u1, deta, E0, checks=(), on_check=None):
    """CN in eta from u0 to u1; on_check(u, psi) at the step closest to each u in checks."""
    e0, e1 = float(eta_of_u(u0)), float(eta_of_u(u1))
    n = max(1, int(math.ceil((e1 - e0) / deta)))
    h = (e1 - e0) / n
    ck = {}
    for uc in checks:
        k = int(round((float(eta_of_u(uc)) - e0) / h))
        if 1 <= k <= n:
            ck.setdefault(k, []).append(uc)
    for k in range(1, n + 1):
        um = float(u_of_eta(e0 + (k - 0.5) * h))
        psi = cn_step(psi, fr.g, 0.5, fr.V(um) - E0, h)
        if on_check is not None and k in ck:
            on_check(float(ck[k][0]), psi)
    return psi


class YFrame:
    def __init__(self, tab, g, xi, ytop0, fac=1.5, yfloor=10.0, cap0=10.0, ell=1.0, capmin=20.0, ell_r=2.0):
        # absorber kept 'conformally gentle': flattening length max(ell, ell_r/hL) (>= a few conformal wavelengths
        # early on), CAP strength cap0/hL, CAP width >= capmin (smoke test showed reflection with a sharp absorber)
        self.tab, self.g, self.xi = tab, g, float(xi)
        self.fac, self.yfloor, self.cap0, self.ell, self.capmin, self.ell_r = fac, yfloor, cap0, ell, capmin, ell_r
        self.ytop = float(ytop0)

    def W_real(self, u):
        hl = float(hL(u))
        y = self.g.x
        return self.tab.U_on(hl * y, u) / hl + 6.0 * self.xi * hl ** 3 * y * y, hl ** 3, hl

    def V(self, u):
        W, M, hl = self.W_real(u)
        y = self.g.x
        ilo = int(np.searchsorted(y, 0.3))
        ihi = min(int(np.searchsorted(y, 1.3 * self.ytop)), len(y) - 2)
        imin = ilo + int(np.argmin(W[ilo:ihi]))
        itop = imin + int(np.argmax(W[imin:ihi + 1]))
        self.ytop = float(y[itop])
        ycap = min(max(self.fac * self.ytop, self.yfloor), y[-1] - self.capmin)
        i = int(np.searchsorted(y, ycap))
        Wm = W.copy()
        Fc = (W[i + 1] - W[i - 1]) / (2 * self.g.dx)
        ell = max(self.ell, self.ell_r / hl)
        Wm[i + 1:] = W[i] + Fc * ell * np.tanh((y[i + 1:] - y[i]) / ell)
        cap = np.zeros_like(y)
        cap[i + 1:] = (self.cap0 / hl) * ((y[i + 1:] - y[i]) / (y[-1] - y[i])) ** 2
        self.ycap, self.icap, self.itop, self.imin = float(y[i]), i, itop, imin
        return Wm - 1j * cap, M, W


def map_r_to_y(psi, gr, gy, u, r_cut):
    hl = float(hL(u))
    lpl = hl * math.tanh(2 * u)
    ps = np.where(gr.x <= r_cut, psi, 0.0)
    xr = np.concatenate([[0.0], gr.x])
    sre = CubicSpline(xr, np.concatenate([[0.0], ps.real]))
    sim = CubicSpline(xr, np.concatenate([[0.0], ps.imag]))
    r = hl * gy.x
    m = r <= gr.x[-1]
    Phi = np.zeros(gy.N, complex)
    Phi[m] = math.sqrt(hl) * np.exp(-0.5j * lpl * r[m] ** 2) * (sre(r[m]) + 1j * sim(r[m]))
    return Phi


def evolve_y(Phi, yf, u0, u1, checks=(), on_check=None, du_max=0.002, cE=0.25, every=20):
    """CN in u with adaptive step du = min(du_max, cE/DeltaE), DeltaE = max |W - E0| over y < y_cap."""
    g = yf.g
    u = float(u0)
    targets = sorted(set([c for c in checks if u0 < c <= u1 + 1e-12] + [u1]))
    E0, dE, it = None, None, 0
    for tgt in targets:
        while u < tgt - 1e-12:
            if it % every == 0 or E0 is None:
                Vc, M, W = yf.V(u)
                ic = yf.icap
                a2 = np.abs(Phi[:ic]) ** 2
                nrm = np.sum(a2)
                E0 = float(np.real(np.vdot(Phi[:ic], h_apply(Phi, g, 0.5 / M, W)[:ic])) / max(nrm, 1e-300))
                pop = a2 > 1e-6 * a2.max()
                pop[:yf.imin] = False          # classically forbidden core: tiny amplitude, CN is stable there
                dE = float(np.max(np.abs(W[:ic][pop] - E0))) if pop.any() else 1.0
            du = min(du_max, cE / max(dE, 1e-9), tgt - u)
            Vm, M, _ = yf.V(u + 0.5 * du)
            Phi = cn_step(Phi, g, 0.5 / M, Vm - E0, du)
            u += du
            it += 1
        if on_check is not None and tgt in checks:
            on_check(tgt, Phi)
    return Phi


def wigner_cells(Phi, g, ycap, ny=320, npc=200, qcut=1e-7):
    """p-cell-averaged Wigner weights P[i,k] on rows y_i (subset of a coarse grid) and momentum cells p_k."""
    y, dy = g.x, g.dx
    Pin = Phi * 0.5 * (1.0 - np.tanh((y - (ycap - 1.5)) / 0.4))
    Nf = 1 << int(math.ceil(math.log2(4 * len(Pin))))
    F = np.fft.fft(Pin, Nf)
    p = 2 * np.pi * np.fft.fftfreq(Nf, d=dy)
    o = np.argsort(p)
    p, rp = p[o], np.abs(F[o]) ** 2
    c = np.cumsum(rp) / rp.sum()
    plo, phi = p[np.searchsorted(c, qcut)], p[min(np.searchsorted(c, 1 - qcut), len(p) - 1)]
    pad = 0.1 * (phi - plo) + 1e-3
    plo, phi = plo - pad, phi + pad
    m = max(1, int(math.floor(math.pi / (2 * 1.2 * max(abs(plo), abs(phi))) / dy)))
    dyc = m * dy
    Pc = Pin[m - 1::m]
    yc = dyc * np.arange(1, len(Pc) + 1)
    rc = np.abs(Pc) ** 2
    cy = np.cumsum(rc) / rc.sum()
    ilo, ihi = int(np.searchsorted(cy, qcut)), int(np.searchsorted(cy, 1 - qcut))
    slo, shi = int(np.searchsorted(cy, 1e-12)), int(min(np.searchsorted(cy, 1 - 1e-12), len(Pc) - 1))
    idx = np.unique(np.round(np.linspace(ilo, ihi, ny)).astype(int))
    yw = yc[idx]
    wy = np.gradient(yw) if len(yw) > 1 else np.array([dyc])
    J = shi - slo + 2
    Pp = np.concatenate([np.zeros(J), Pc, np.zeros(J)])
    jj = np.arange(-J, J + 1)
    A = np.conj(Pp[idx[:, None] + J + jj[None, :]]) * Pp[idx[:, None] + J - jj[None, :]]
    dp = (phi - plo) / npc
    pk = plo + (np.arange(npc) + 0.5) * dp
    s = jj * dyc
    sw = np.where(s == 0, dp, np.sin(dp * s) / np.where(s == 0, 1.0, s))
    E = (dyc / math.pi) * np.exp(2j * pk[None, :] * s[:, None]) * sw[:, None]
    P = np.real(A @ E) * wy[:, None]
    vm = np.imag(np.conj(Pc) * np.gradient(Pc, dyc)) / np.maximum(rc, 1e-300)
    return dict(y=yw, p=pk, P=P, dy_rows=wy, dyc=dyc, norm_in=float(np.sum(np.abs(Pin) ** 2) * dy),
                mad=dict(y=yc[slo:shi + 1], pm=vm[slo:shi + 1], w=rc[slo:shi + 1] * dyc))


def classical_fates(y0, v0, u0, xi, tab, u_end=60.0, rg=False, y_core=0.3):
    """Physical-frame zero mode (units of h, time u = ht): y'' + 3 tanh(2u) y' + 12 xi y + U_nc'(hL y)/(hL)^3 [+RG] = 0.
    fate: 1 bounded (core or trapped below the late-time barrier with E < 0.9 V_b), 2 runaway, 0 undecided at u_end."""
    lamL = tab.lam(tab.umax)
    yb = math.sqrt(12 * xi / abs(lamL)) if (xi > 0 and lamL < 0) else 0.0
    Vb = 36 * xi * xi / abs(lamL) if lamL < 0 else np.inf
    y_run = max(3 * yb, 30.0)
    y = np.array(y0, float)
    v = np.array(v0, float)
    fate = np.zeros(len(y), int)
    tdec = np.full(len(y), np.nan)
    act = np.arange(len(y))

    def acc(yy, vv, uu):
        hl = float(hL(uu))
        a = -3 * math.tanh(2 * uu) * vv - 12 * xi * yy - tab.dU_on(hl * np.maximum(yy, 1e-9), uu) / hl ** 3
        if rg:
            ph = yy * H_GEV
            mu = np.sqrt(ph * ph + H_GEV ** 2 + (MU_B / math.sqrt(math.cosh(2 * min(uu, 8.0)))) ** 2)
            lr, br = tab.lam_of_mu(mu)
            a = a - (lr - tab.lam(uu)) * yy ** 3 - br * (ph * ph / (mu * mu)) * yy ** 3 / 4
        return a
    u = float(u0)
    while act.size and u < u_end - 1e-12:
        du = min(0.01 if u < 8 else 0.04, u_end - u)
        ya, va = y[act], v[act]
        k1y, k1v = va, acc(ya, va, u)
        k2y, k2v = va + 0.5 * du * k1v, acc(ya + 0.5 * du * k1y, va + 0.5 * du * k1v, u + 0.5 * du)
        k3y, k3v = va + 0.5 * du * k2v, acc(ya + 0.5 * du * k2y, va + 0.5 * du * k2v, u + 0.5 * du)
        k4y, k4v = va + du * k3v, acc(ya + du * k3y, va + du * k3v, u + du)
        ya = ya + du / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        va = va + du / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        u += du
        y[act], v[act] = ya, va
        run = (ya > y_run) | ~np.isfinite(ya)
        bnd = ya < y_core
        if xi > 0 and u > max(u0 + 2.0, 7.5):
            E = 0.5 * va * va + 6 * xi * ya * ya + lamL * ya ** 4 / 4
            bnd |= (ya < yb) & (E < 0.9 * Vb)
        fate[act[run]] = 2
        fate[act[bnd & ~run]] = 1
        tdec[act[(run | bnd)]] = u
        act = act[fate[act] == 0]
    return fate, tdec, dict(yb=yb, Vb=Vb, y_run=y_run, lamL=lamL, u_end=u)
