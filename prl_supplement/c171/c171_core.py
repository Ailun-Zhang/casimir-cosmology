"""C171 core: exact zero-mode resonance of the abelian Higgs model in a small periodic box.

Written from scratch; it uses no code of the code release v1.0.0.

Scaled units (e = 1, Lambda = lambda/e^2; Gamma(e) = e^{2/3} F(Lambda), lengths x e^{-1/3}):
    H = -1/2 Lap_2D(phi) - 1/2 Lap_3D(w) + 1/2 r^2 s^2 + (Lambda/4) r^4,   r = |phi|, s = |w|,
restricted to zero U(1) charge (m = 0 in the phi plane, Gauss law) and the s-wave in w.

Method A (primary):
  r : psi-form (measure r dr) second-order cell-centred finite differences in flux form,
      nodes rho_i = (i - 1/2) h, flux points rho = j h; smooth exterior complex scaling
      r(rho) = rho + (e^{i theta} - 1) * delta * I((rho - R0)/delta), I = integral of a C2 smoothstep.
  s : sine DVR on (0, S) for v = s * psi (v(0) = v(S) = 0).
  eigenvalue: shift-invert Arnoldi (ARPACK) with an exact block-tridiagonal LU of (H - sigma),
      polished by inverse iteration; Gamma from Im E (Rayleigh quotient) and from the exact
      discrete flux identity Gamma = -2 H_{n,n+1} Im(Psi_n^dag Psi_{n+1}) / P_n in the unscaled region.

1D models (Born-Oppenheimer):  -1/2 F'' + U(r) F = E F,  U = sigma0 r + c2 / r^2 + (Lambda/4) r^4,
  c2 = -1/8 (plain BO) or c2 = -1/8 + 3/16 = 1/16 (BO + diagonal BO correction).
  Solved by complex-plane shooting: Frobenius series at r -> 0, DOP853 along the real axis to R1 and
  then along the ray R1 + t e^{i phi}; resonance condition F(t_max) = 0 (no incoming component), Newton.
WKB: Langer-corrected Gamow/Kemble formula and the deep-well asymptotic exponent (2 pi sqrt2/3) sigma0 / |Lambda|^{1/2}.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import math
import numpy as np
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

SIGMA0 = 1.5                                   # BO slope 3e/2 at e = 1
B_ASYM = 2.0 * math.pi * math.sqrt(2.0) / 3.0 * SIGMA0   # 4.44288...


# ----------------------------------------------------------------------------------------------
# complex path (smooth exterior complex scaling)
# ----------------------------------------------------------------------------------------------
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def _smoothstep_int(x):
    x = np.maximum(np.asarray(x, float), 0.0)
    xc = np.minimum(x, 1.0)
    val = 2.5 * xc ** 4 - 3.0 * xc ** 5 + xc ** 6
    return np.where(x > 1.0, 0.5 + (x - 1.0), val)


def ecs_path(rho, R0, theta, delta):
    rho = np.asarray(rho, float)
    x = (rho - R0) / delta
    q = np.exp(1j * theta) - 1.0
    r = rho + q * delta * _smoothstep_int(x)
    rp = 1.0 + q * _smoothstep(x)
    return r, rp


def radial_operator(N, h, R0, theta, delta):
    """Symmetrized 2D-radial kinetic operator (tridiagonal, complex symmetric)."""
    j = np.arange(N + 1)
    rf, rpf = ecs_path(j * h, R0, theta, delta)
    w = rf / rpf                                   # r / r' at flux points; w[0] = 0 since r(0) = 0
    rho = (np.arange(N) + 0.5) * h
    rn, rpn = ecs_path(rho, R0, theta, delta)
    m = rn * rpn * h                               # lumped mass  r r' h
    Kd = (w[:-1] + w[1:]) / (2.0 * h)              # Dirichlet beyond the last node
    Ko = -w[1:-1] / (2.0 * h)
    sq = np.sqrt(m)
    return dict(rho=rho, r=rn, rp=rpn, m=m, Hd=Kd / m, Ho=Ko / (sq[:-1] * sq[1:]), h=h, N=N)


def sine_dvr(Ns, S):
    n = np.arange(1, Ns + 1)
    s = S / (Ns + 1) * n
    U = np.sqrt(2.0 / (Ns + 1)) * np.sin(np.pi * np.outer(n, n) / (Ns + 1))
    T = (U * (0.5 * (np.pi * n / S) ** 2)) @ U.T
    return s, T, U


def tail_matrix(Ns, S, sc):
    """B_nm = int_sc^S phi_n phi_m ds for phi_n = sqrt(2/S) sin(n pi s / S)."""
    n = np.arange(1, Ns + 1, dtype=float)
    k = math.pi / S
    nn, mm = np.meshgrid(n, n, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        dmn = nn - mm
        off = (-np.sin(dmn * k * sc) / (dmn * k) + np.sin((nn + mm) * k * sc) / ((nn + mm) * k)) / S
    diag = ((S - sc) + np.sin(2 * n * k * sc) / (2 * n * k)) / S
    B = np.where(dmn == 0, 0.0, off)
    B[np.diag_indices(Ns)] = diag
    return B


# ----------------------------------------------------------------------------------------------
# 2D problem
# ----------------------------------------------------------------------------------------------
class Problem2D:
    def __init__(self, Lam, h, rmax, R0, theta, delta, S, Ns, e=1.0):
        self.Lam, self.e = Lam, e
        self.N = int(round(rmax / h))
        self.rad = radial_operator(self.N, h, R0, theta, delta)
        self.s, self.T, self.U = sine_dvr(Ns, S)
        self.Ns, self.S = Ns, S
        r = self.rad["r"]
        self.V = 0.5 * e ** 2 * np.outer(r ** 2, self.s ** 2) + (Lam / 4.0) * (r ** 4)[:, None]
        self.Vd = self.V + self.rad["Hd"][:, None]
        self.c = self.rad["Ho"]
        self.Tc = self.T.astype(complex)
        self.params = dict(Lam=Lam, h=h, rmax=rmax, R0=R0, theta=theta, delta=delta, S=S, Ns=Ns, e=e,
                           N=self.N, dim=self.N * Ns)

    def matvec(self, x):
        X = np.asarray(x).reshape(self.N, self.Ns)
        Y = X @ self.T + self.Vd * X
        Y[1:] += self.c[:, None] * X[:-1]
        Y[:-1] += self.c[:, None] * X[1:]
        return Y.ravel()

    def factor(self, sigma):
        N, Ns = self.N, self.Ns
        Ginv = np.empty((N, Ns, Ns), complex)
        dg = np.arange(Ns)
        G = self.Tc.copy()
        G[dg, dg] += self.Vd[0] - sigma
        Ginv[0] = np.linalg.inv(G)
        c2 = self.c ** 2
        for i in range(1, N):
            G = self.Tc - c2[i - 1] * Ginv[i - 1]
            G[dg, dg] += self.Vd[i] - sigma
            Ginv[i] = np.linalg.inv(G)
        self.Ginv = Ginv
        self.sigma = sigma

    def solve(self, b):
        N, Ginv, c = self.N, self.Ginv, self.c
        B = np.asarray(b, complex).reshape(N, self.Ns)
        Y = np.empty_like(B)
        Y[0] = B[0]
        for i in range(1, N):
            Y[i] = B[i] - c[i - 1] * (Ginv[i - 1] @ Y[i - 1])
        X = np.empty_like(B)
        X[N - 1] = Ginv[N - 1] @ Y[N - 1]
        for i in range(N - 2, -1, -1):
            X[i] = Ginv[i] @ (Y[i] - c[i] * X[i + 1])
        return X.ravel()


def _extrema(U, dU, rhi):
    """Local minimum (if any) and maximum of U on (0, rhi) by sign changes of dU on a log grid."""
    rr = np.geomspace(1e-3, rhi, 4000)
    d = np.array([dU(r) for r in rr])
    roots = []
    for i in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        roots.append(brentq(dU, rr[i], rr[i + 1], xtol=1e-14))
    rmin = rmax = None
    for r0 in roots:
        if dU(r0 * (1 - 1e-6)) < 0 < dU(r0 * (1 + 1e-6)):
            rmin = r0
        elif dU(r0 * (1 - 1e-6)) > 0 > dU(r0 * (1 + 1e-6)):
            rmax = r0
    return rmin, rmax


def bo_landmarks(Lam, E, sigma0=SIGMA0, c2=1.0 / 16.0):
    """Barrier top and outer turning point of U = sigma0 r + c2/r^2 + Lam r^4/4 (Lam < 0)."""
    U = lambda r: sigma0 * r + c2 / r ** 2 + Lam / 4.0 * r ** 4
    dU = lambda r: sigma0 - 2 * c2 / r ** 3 + Lam * r ** 3
    rb0 = (sigma0 / abs(Lam)) ** (1.0 / 3.0)
    rmin, rb = _extrema(U, dU, 3.0 * rb0)
    Umax = U(rb)
    if E < Umax:
        r3 = brentq(lambda r: U(r) - E, rb, 4 * rb, xtol=1e-14)
        above = False
    else:
        r3 = rb
        above = True
    return dict(rb=rb, Umax=Umax, r3=r3, above=above, rmin=rmin)


def exterior_end(Lam, E, R0, theta, delta, D=40.0, sigma0=SIGMA0):
    """rho at which the outgoing BO-channel WKB wave has decayed by exp(-D) along the ECS path."""
    rho = R0 + np.linspace(0.0, 60.0, 60001)
    r, rp = ecs_path(rho, R0, theta, delta)
    k = np.sqrt(2.0 * (E - sigma0 * r - Lam / 4.0 * r ** 4) + 0j)
    k = np.where(k.real < 0, -k, k)
    integrand = k * rp
    ph = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rho))])
    idx = np.argmax(ph.imag >= D)
    if idx == 0:
        idx = len(rho) - 1
    return rho[idx]


def default_grid(Lam, E_guess, h=0.02, theta=0.45, delta=0.6, S=10.0, ds=None, dR=1.2, D=40.0,
                 extra_ext=0.0, sigma0=SIGMA0):
    lm = bo_landmarks(Lam, E_guess, sigma0)
    r3 = lm["r3"] if not lm["above"] else lm["rb"] + 0.5
    R0 = r3 + dR
    R0 = h * round(R0 / h)
    rend = exterior_end(Lam, E_guess, R0, theta, delta, D, sigma0) + 0.3 + extra_ext
    rmax = h * math.ceil(rend / h)
    rabs = abs(R0 + delta / 2 + np.exp(1j * theta) * (rmax - R0 - delta / 2))
    if ds is None:
        ds = min(0.16, 0.5 / math.sqrt(rabs))
    Ns = int(round(S / ds)) - 1
    return dict(Lam=Lam, h=h, rmax=rmax, R0=R0, theta=theta, delta=delta, S=S, Ns=Ns), lm


def solve_2d(grid, E_guess, k=8, offset=0.05, r_well=None, r3=None, tol=1e-13, n_polish=4,
             hol_e=(1.0, 0.8733, 0.5), want_vec=False, verbose=False, frac_min=0.5,
             refine_offset=1e-3):
    """Find the well resonance of the 2D problem on a given grid. Returns a dict of results."""
    t0 = time.time()
    P = Problem2D(**grid)
    sigma = E_guess - 1j * offset
    P.factor(sigma)
    t_fac = time.time() - t0
    n = P.N * P.Ns
    A = spla.LinearOperator((n, n), matvec=P.matvec, dtype=complex)
    OPinv = spla.LinearOperator((n, n), matvec=P.solve, dtype=complex)
    rng = np.random.default_rng(1234)
    v0 = np.zeros((P.N, P.Ns), complex)
    rho = P.rad["rho"]
    if r_well is None:
        r_well = 0.5 * grid["R0"]
    # start vector: smooth bump in the well
    v0 += np.exp(-((rho[:, None] - 1.0) ** 2) - (P.s[None, :] - 1.0) ** 2) * rho[:, None] * P.s[None, :]
    v0 += 1e-3 * rng.standard_normal(v0.shape)
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, OPinv=OPinv, which="LM", tol=tol,
                           ncv=max(2 * k + 2, 24), v0=v0.ravel(), maxiter=5000)
    inwell = rho < r_well
    fr = []
    for j in range(len(vals)):
        X = vecs[:, j].reshape(P.N, P.Ns)
        w = np.sum(np.abs(X) ** 2, axis=1)
        fr.append(np.sum(w[inwell]) / np.sum(w))
    fr = np.array(fr)
    cand = np.where(fr > frac_min)[0]
    if len(cand) == 0:
        jsel = int(np.argmax(fr))
    else:
        jsel = int(cand[np.argmin(vals[cand].real)])     # ground resonance: lowest Re E localized in the well
    x = vecs[:, jsel].copy()
    # refine: re-factor next to the selected eigenvalue and inverse-iterate
    P.factor(vals[jsel] - 1j * refine_offset)
    for _ in range(n_polish):
        x = P.solve(x)
        x /= np.linalg.norm(x)
    P.Ginv = None
    Hx = P.matvec(x)
    E_rq = (x @ Hx) / (x @ x)
    resid = np.linalg.norm(Hx - E_rq * x) / np.linalg.norm(x)
    X = x.reshape(P.N, P.Ns)
    imax = np.unravel_index(np.argmax(np.abs(X) * inwell[:, None]), X.shape)
    X = X * (abs(X[imax]) / X[imax])
    # flux identity in the unscaled region beyond the outer turning point
    R0 = grid["R0"]
    wnode = np.sum(np.abs(X) ** 2, axis=1)
    Pcum = np.cumsum(wnode)
    lo = (r3 + 0.4) if r3 is not None else 0.75 * R0
    sel = np.where((rho > lo) & (rho + P.rad["h"] < R0 - 1e-9))[0]
    gam_flux = np.array([-2.0 * P.c[i].real * np.imag(np.vdot(X[i], X[i + 1])) / Pcum[i] for i in sel])
    out = dict(params=P.params, E_re=float(E_rq.real), E_im=float(E_rq.imag), Gamma_rq=float(-2 * E_rq.imag),
               arpack_vals=[[float(v.real), float(v.imag)] for v in vals], well_frac=fr.tolist(), jsel=jsel,
               resid=float(resid), t_factor=t_fac, t_total=None)
    if len(gam_flux):
        out.update(Gamma_flux=float(np.median(gam_flux)), Gamma_flux_min=float(gam_flux.min()),
                   Gamma_flux_max=float(gam_flux.max()), n_flux=int(len(gam_flux)),
                   flux_window=[float(rho[sel[0]]), float(rho[sel[-1]])])
    else:
        out.update(Gamma_flux=None)
    # probability in the well region and holonomy tails
    wsel = rho < r_well
    Pw = np.sum(wnode[wsel])
    out["P_well_over_unscaled"] = float(Pw / np.sum(wnode[rho < R0]))
    A_fbr = X[wsel] @ P.U                      # FBR (sine) coefficients per node
    hol = {}
    unscaled = abs(P.e - 1.0) > 1e-12
    for ee in ((P.e,) if unscaled else hol_e):
        for frac, lab in ((0.5, "pi/2"), (1.0, "pi")):
            # scaled problem (e=1): e s_phys > frac*pi <=> S_scaled > frac*pi*e^{-2/3}; unscaled: s > frac*pi/e
            sc = frac * math.pi / P.e if unscaled else frac * math.pi * ee ** (-2.0 / 3.0)
            if sc >= P.S:
                val = 0.0
            else:
                B = tail_matrix(P.Ns, P.S, sc)
                val = float(np.real(np.einsum("in,nm,im->", A_fbr.conj(), B, A_fbr)) / Pw)
            hol[f"e={ee:g}|ew|>{lab}"] = val
    out["holonomy_tail"] = hol
    out["r_well"] = float(r_well)
    out["t_total"] = time.time() - t0
    if want_vec:
        out["_X"] = X
        out["_rho"] = rho
        out["_s"] = P.s
    if verbose:
        print(f"  dim={n} E={E_rq:.12g} G_rq={-2*E_rq.imag:.6e} G_flux={out.get('Gamma_flux')} "
              f"res={resid:.1e} t={out['t_total']:.1f}s", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# 1D Born-Oppenheimer model: complex-plane shooting
# ----------------------------------------------------------------------------------------------
def frob_series(E, Lam, c2, sigma0=SIGMA0, nterms=90):
    a = 0.5 * (1.0 + math.sqrt(1.0 + 8.0 * c2))
    co = np.zeros(nterms, complex)
    co[0] = 1.0
    for n in range(1, nterms):
        num = 0.0
        if n >= 3:
            num += 2.0 * sigma0 * co[n - 3]
        if n >= 2:
            num += -2.0 * E * co[n - 2]
        if n >= 6:
            num += 0.5 * Lam * co[n - 6]
        co[n] = num / (n * (2.0 * a + n - 1.0))
    return a, co


def shoot_1d(E, Lam, c2, R1, phi, tmax, r_start=0.05, sigma0=SIGMA0, rtol=1e-12):
    a, co = frob_series(E, Lam, c2, sigma0)
    nn = np.arange(len(co))
    F0 = np.sum(co * r_start ** (a + nn))
    dF0 = np.sum(co * (a + nn) * r_start ** (a + nn - 1))

    def rhs_real(r, y):
        return [y[1], 2.0 * (sigma0 * r + c2 / r ** 2 + Lam / 4.0 * r ** 4 - E) * y[0]]

    s1 = solve_ivp(rhs_real, (r_start, R1), np.array([F0, dF0], complex), method="DOP853",
                   rtol=rtol, atol=1e-300)
    y1 = s1.y[:, -1]
    d = np.exp(1j * phi)

    def rhs_ray(t, y):
        r = R1 + t * d
        return [d * y[1], d * 2.0 * (sigma0 * r + c2 / r ** 2 + Lam / 4.0 * r ** 4 - E) * y[0]]

    s2 = solve_ivp(rhs_ray, (0.0, tmax), y1, method="DOP853", rtol=rtol, atol=1e-300)
    return s2.y[0, -1], y1


def ray_tmax(Lam, E, R1, phi, D=35.0, sigma0=SIGMA0):
    t = np.linspace(0.0, 80.0, 80001)
    r = R1 + t * np.exp(1j * phi)
    k = np.sqrt(2.0 * (E - sigma0 * r - Lam / 4.0 * r ** 4) + 0j)
    k = np.where(k.real < 0, -k, k)
    integ = k * np.exp(1j * phi)
    ph = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(t))])
    idx = np.argmax(ph.imag >= D)
    return t[idx] if idx > 0 else t[-1]


def resonance_1d(Lam, c2, E0, sigma0=SIGMA0, phi=0.45, dR=1.0, D=35.0, rtol=1e-12, maxit=40,
                 r_start=0.05, verbose=False):
    lm = bo_landmarks(Lam, E0.real, sigma0, c2=max(c2, 1e-3))
    R1 = (lm["r3"] if not lm["above"] else lm["rb"] + 0.5) + dR
    tmax = ray_tmax(Lam, E0.real, R1, phi, D, sigma0)
    f = lambda E: shoot_1d(E, Lam, c2, R1, phi, tmax, r_start, sigma0, rtol)[0]
    E = complex(E0)
    hE = 1e-6
    hist = []
    for it in range(maxit):
        fE = f(E)
        df = (f(E + hE) - fE) / hE
        dE = -fE / df
        E = E + dE
        hist.append(abs(dE))
        if abs(dE) < 1e-13 * max(abs(E.imag), 1e-300) + 1e-300 or (it > 3 and abs(dE) < 1e-15 and abs(dE) <= 1e-9 * abs(E.imag)):
            break
        if it > 25 and abs(dE) < 1e-14 * abs(E):
            break
    return dict(E_re=E.real, E_im=E.imag, Gamma=-2.0 * E.imag, R1=R1, tmax=tmax, phi=phi, iters=it + 1,
                last_step=hist[-1])


# ----------------------------------------------------------------------------------------------
# WKB
# ----------------------------------------------------------------------------------------------
def _int_between(fun, a, b):
    """int_a^b fun(r) dr with cosine substitution (removes inverse-sqrt endpoint singularities)."""
    g = lambda u: fun(a + (b - a) * 0.5 * (1.0 - math.cos(u))) * (b - a) * 0.5 * math.sin(u)
    val, err = quad(g, 0.0, math.pi, limit=400, epsabs=0.0, epsrel=1e-10)
    return val


def wkb_bo(Lam, c2L, sigma0=SIGMA0, n=0):
    """Langer-corrected WKB for U_L = sigma0 r + c2L/r^2 + Lam r^4/4 (c2L = 0: plain BO, 3/16: BO+DBOC).
    E_BS from int_{r1}^{r2} p dr = (n + 1/2) pi (r1 = 0 if c2L = 0); T = 2 int dr/p;
    S = 2 int_{r2}^{r3} |p| dr; Gamma_Gamow = e^{-S}/T, Gamma_Kemble = ln(1 + e^{-S})/T."""
    U = lambda r: sigma0 * r + c2L / r ** 2 + Lam / 4.0 * r ** 4
    dU = lambda r: sigma0 - 2 * c2L / r ** 3 + Lam * r ** 3
    rb0 = (sigma0 / abs(Lam)) ** (1.0 / 3.0)
    rmin, rb = _extrema(U, dU, 3.0 * rb0)
    Umax = U(rb)
    Umin = U(rmin) if (c2L > 0 and rmin is not None) else 0.0

    def tps(E):
        r1 = brentq(lambda r: U(r) - E, 1e-10, rmin, xtol=1e-15) if c2L > 0 else 0.0
        r2 = brentq(lambda r: U(r) - E, (rmin if c2L > 0 else 1e-12), rb, xtol=1e-15)
        return r1, r2

    def p_of(E):
        return lambda r: math.sqrt(max(2.0 * (E - U(r)), 0.0)) if r > 0 else math.sqrt(2.0 * E)

    def action(E):
        r1, r2 = tps(E)
        return _int_between(p_of(E), r1, r2)

    target = (n + 0.5) * math.pi
    Etop = Umax - 1e-10 * max(1.0, abs(Umax))
    if action(Etop) < target:
        return dict(E_BS=float("nan"), T=float("nan"), S=float("nan"), Gamma_Kemble=float("nan"),
                    Gamma_Gamow=float("nan"), rb=rb, Umax=Umax, above=True)
    E_BS = brentq(lambda E: action(E) - target, Umin + 1e-10, Etop, xtol=1e-14)
    r1, r2 = tps(E_BS)
    invp = lambda r: 1.0 / math.sqrt(max(2.0 * (E_BS - U(r)), 1e-300)) if r > 0 else 1.0 / math.sqrt(2.0 * E_BS)
    T = 2.0 * _int_between(invp, r1, r2)
    r3 = brentq(lambda r: U(r) - E_BS, rb, 4 * rb, xtol=1e-14)
    kap = lambda r: math.sqrt(max(2.0 * (U(r) - E_BS), 0.0))
    S = 2.0 * _int_between(kap, r2, r3)
    return dict(E_BS=E_BS, T=T, S=S, Gamma_Kemble=math.log1p(math.exp(-S)) / T, Gamma_Gamow=math.exp(-S) / T,
                rb=rb, Umax=Umax, r1=r1, r2=r2, r3=r3, above=False)


def barrier_action(Lam, E, c2L, sigma0=SIGMA0):
    """S(E) = 2 int |p| dr across the barrier of the Langer potential (nan if E is above the top)."""
    U = lambda r: sigma0 * r + c2L / r ** 2 + Lam / 4.0 * r ** 4
    dU = lambda r: sigma0 - 2 * c2L / r ** 3 + Lam * r ** 3
    rb0 = (sigma0 / abs(Lam)) ** (1.0 / 3.0)
    rmin, rb = _extrema(U, dU, 3.0 * rb0)
    if E >= U(rb):
        return float("nan")
    r2 = brentq(lambda r: U(r) - E, (rmin if (c2L > 0 and rmin) else 1e-12), rb, xtol=1e-15)
    r3 = brentq(lambda r: U(r) - E, rb, 4 * rb, xtol=1e-14)
    kap = lambda r: math.sqrt(max(2.0 * (U(r) - E), 0.0))
    return 2.0 * _int_between(kap, r2, r3)
