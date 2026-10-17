# -*- coding: utf-8 -*-
"""
R15B_main.py -- agent r15-beyond-bo, R15 independent check of D112 (2D (r,|w|) beyond-BO Higgs zero mode).

Written BEFORE opening any D112 code or result JSON.  Own implementation:
  * D108 conditional Gaussian solved in reduced variables (a, k, beta) with b~ = Omega eliminated analytically;
    r-derivatives by the implicit-function theorem; B_r cross-checked against (3/16) tr[(K^-1 K')^2].
  * Symbolic identities (sympy): E_tot decomposition, B_r = Bx + Bw + Bbeta, conditional projections.
  * 2D: H = -1/2 d_r^2 + 3/(8 r^2) - 1/2 d_s^2 (v = s chi, v(0)=0) + C0 + Bx + 1/2 Omega_eff^2 s^2
          + lam r^4/4 + dV_MC + dE_DMC,
    degree-7 B-spline Galerkin in r (graded knots, Dirichlet) x odd harmonic-oscillator basis in s
    (sparse, exact s^2 matrix) or sine-DVR in s (explicit |w| box S_max).
  * Resonances: complex absorbing potential -i eta W(r), W = ((r-Rc)/(Rmax-Rc))^3, eigenvalue trajectory by
    shift-invert Arnoldi, resonance picked by localisation P(r<4); Riss-Meyer first-order corrected
    E_c = E - eta dE/deta (c-product Hellmann-Feynman); stationary point in eta.  Second method: real-box
    stabilisation (Dirichlet wall R_w scan).
Inputs (read-only): S3_prop.json (rot_knots / rot_dEtot), C138_core.dV('MC') (D110 matching, imported from the
D110 code directory, not from D112), C138_hazard.json (P1(m_t), lambda_b(m_t), gamma nodes along u).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import brentq

ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
OUT = os.path.join(ROOT, r"07_reviews\R15_v12_round2\higgs_beyond_bo")
S3P = os.path.join(ROOT, r"04_calculations\results\20261013_claude_v12\higgs_survival\run02_stage3\S3_prop.json")
MATCH_CODE = os.path.join(ROOT, r"04_calculations\code\20261013_claude_v12_r2\higgs_matching")
C138RES = os.path.join(ROOT, r"04_calculations\results\20261013_claude_v12_r2\higgs_matching\run01")
G2, GY, YT = 0.5268, 0.4415, 0.4585
LAMS = [-0.007, -0.008, -0.009, -0.010, -0.011, -0.012, -0.013, -0.0143]
SB0 = ("HO", 30, 0.35, 14.0)


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=conv, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------- symbolic
def sym_Om2(a, k, b, r, g2, gY):
    return r ** 2 / 4 * (gY - g2 * b) ** 2 + 2 * g2 ** 2 * b ** 2 / a + 4 * a * b ** 2 / r ** 2


def sym_C0(a, k, b, r, g2, gY):
    import sympy as sy
    R = sy.Rational
    return (R(3, 4) * (2 * a + k) + R(3, 4) * k * b ** 2 + R(3, 16) * r ** 2 * (2 * g2 ** 2 / a + g2 ** 2 / k)
            + R(3, 4) * g2 ** 2 / a ** 2 + R(3, 2) * g2 ** 2 / (a * k) + 3 * (k - a) ** 2 / (a * k * r ** 2))


def gauss_expect(expr, sig2):
    import sympy as sy
    vs = list(sig2.keys())
    poly = sy.Poly(sy.expand(expr), *vs)
    tot = 0
    for mon, coef in poly.terms():
        term = coef
        for v, n in zip(vs, mon):
            if n % 2:
                term = 0
                break
            term = term * sig2[v] ** (n // 2) * sy.factorial2(n - 1)
        tot += term
    return tot


def identities():
    import sympy as sy
    R = sy.Rational
    a, k, r, g2, gY = sy.symbols("a k r g2 gY", positive=True)
    b, c = sy.symbols("b c", positive=True)
    out = {}
    D = k * b - c ** 2
    E108 = (R(3, 4) * (2 * a + k + b) + R(3, 16) * r ** 2 * (2 * g2 ** 2 / a + (g2 ** 2 * b - 2 * g2 * gY * c + gY ** 2 * k) / D)
            + 3 * g2 ** 2 / (4 * a ** 2) + 3 * g2 ** 2 * b / (2 * a * D)
            + 3 / (a * r ** 2) * (b * (k - a) ** 2 - 2 * c ** 2 * (k - a) + k * c ** 2) / D)
    bt = b - c ** 2 / k
    Edec = sym_C0(a, k, c / k, r, g2, gY) + R(3, 4) * (bt + sym_Om2(a, k, c / k, r, g2, gY) / bt)
    out["id1_Etot_D108_minus_[C0+3/4(bt+Om2/bt)]"] = str(sy.cancel(sy.together(E108 - Edec)))
    ap, kp, bp, cp = sy.symbols("ap kp bp cp", real=True)
    K2 = sy.Matrix([[k, c], [c, b]])
    K2p = sy.Matrix([[kp, cp], [cp, bp]])
    Mm = K2.inv() * K2p
    Br = R(3, 16) * (2 * (ap / a) ** 2 + (Mm * Mm).trace())
    betap = cp / k - c * kp / k ** 2
    btp = bp - 2 * c * cp / k + c ** 2 * kp / k ** 2
    Bsum = R(3, 16) * (2 * (ap / a) ** 2 + (kp / k) ** 2) + R(3, 16) * (btp / bt) ** 2 + 3 * k * betap ** 2 / (8 * bt)
    out["id2_Br_minus_(Bx+Bw+Bbeta)"] = str(sy.cancel(sy.together(Br - Bsum)))
    x1, x2, z, w, be, bep = sy.symbols("x1 x2 z w beta betap", real=True)
    sig = {x1: 1 / (2 * a), x2: 1 / (2 * a), z: 1 / (2 * k)}
    dlr = ap / (2 * a) + kp / (4 * k) - R(1, 2) * (ap * (x1 ** 2 + x2 ** 2) + kp * z ** 2 + 2 * k * bep * w * z)
    id3 = R(1, 2) * gauss_expect(dlr ** 2, sig) - (R(1, 16) * (2 * (ap / a) ** 2 + (kp / k) ** 2) + k / 4 * bep ** 2 * w ** 2)
    out["id3_half<|d_r phi|^2>_per_dir_minus_(Bx/3+k/4 beta'^2 w^2)"] = str(sy.simplify(id3))
    id4 = R(1, 2) * gauss_expect((k * be * z) ** 2, sig) - k * be ** 2 / 4
    out["id4_half<|d_w phi|^2>_per_dir_minus_k beta^2/4"] = str(sy.simplify(id4))
    X1 = sy.symbols("x1_0:3", real=True)
    X2 = sy.symbols("x2_0:3", real=True)
    Z = sy.symbols("z_0:3", real=True)
    Wv = sy.symbols("w_0:3", real=True)
    sig3 = {}
    for i in range(3):
        sig3[X1[i]] = 1 / (2 * a)
        sig3[X2[i]] = 1 / (2 * a)
        sig3[Z[i]] = 1 / (2 * k)
    X3 = [Z[i] - be * Wv[i] for i in range(3)]
    kin = sum(R(1, 2) * (a ** 2 * X1[i] ** 2 + a ** 2 * X2[i] ** 2 + k ** 2 * Z[i] ** 2) for i in range(3))
    mass = sum(r ** 2 / 8 * (g2 ** 2 * (X1[i] ** 2 + X2[i] ** 2) + (g2 * X3[i] + gY * Wv[i]) ** 2) for i in range(3))
    vec = [sy.Matrix([X1[i], X2[i], X3[i]]) for i in range(3)]
    ym = 0
    for i in range(3):
        for j in range(i + 1, 3):
            cr = vec[i].cross(vec[j])
            ym += g2 ** 2 / 2 * cr.dot(cr)
    L1 = sum(X2[i] * ((a - k) * Z[i] - a * be * Wv[i]) for i in range(3))
    L2 = sum(X1[i] * ((k - a) * Z[i] + a * be * Wv[i]) for i in range(3))
    ang = 2 / r ** 2 * (L1 ** 2 + L2 ** 2)
    Econd = gauss_expect(kin + mass + ym + ang, sig3)
    target = (sym_C0(a, k, be, r, g2, gY) - R(3, 4) * k * be ** 2
              + R(1, 2) * sym_Om2(a, k, be, r, g2, gY) * sum(wi ** 2 for wi in Wv))
    out["id5_<H_x|w>_minus_(C0-3/4 k beta^2 + 1/2 Om2 |w|^2)"] = str(sy.cancel(sy.together(sy.expand(Econd - target))))
    return out


_FUN = None


def funs():
    global _FUN
    if _FUN is not None:
        return _FUN
    import sympy as sy
    a, k, be, r, g2, gY = sy.symbols("a k beta r g2 gY", real=True)
    Om = sy.sqrt(sym_Om2(a, k, be, r, g2, gY))
    F = sym_C0(a, k, be, r, g2, gY) + sy.Rational(3, 2) * Om
    P = (a, k, be)
    grad = [sy.diff(F, v) for v in P]
    hess = [[sy.diff(gi, v) for v in P] for gi in grad]
    gr = [sy.diff(gi, r) for gi in grad]
    dOm = [sy.diff(Om, v) for v in P] + [sy.diff(Om, r)]
    args = (a, k, be, r, g2, gY)

    def L(e, mod):
        try:
            return sy.lambdify(args, e, mod, cse=True)
        except TypeError:
            return sy.lambdify(args, e, mod)
    _FUN = dict(F_m=L(F, "math"), G_m=L(grad, "math"), H_m=L(hess, "math"),
                F=L(F, "numpy"), H=L(hess, "numpy"), Gr=L(gr, "numpy"), Om=L(Om, "numpy"), dOm=L(dOm, "numpy"))
    return _FUN


def newton(p, r, F, maxit=60):
    p = np.array(p, float)
    for _ in range(maxit):
        g = np.array(F["G_m"](p[0], p[1], p[2], r, G2, GY), float)
        H = np.array(F["H_m"](p[0], p[1], p[2], r, G2, GY), float)
        dg = np.sqrt(np.abs(np.diag(H)))
        d = 1.0 / dg
        dp = -d * np.linalg.solve(H * np.outer(d, d), g * d)
        t = 1.0
        while p[0] + t * dp[0] <= 0 or p[1] + t * dp[1] <= 0:
            t *= 0.5
        p = p + t * dp
        if t == 1.0 and np.max(np.abs(dp * dg)) < 1e-14:
            break
    return p


def anchor(F, r0=1.0):
    from scipy.optimize import minimize
    best = None
    for a0, k0, b0 in [(0.8, 0.8, 0.0), (0.9, 1.0, 0.05), (0.6, 0.7, 0.2), (1.2, 1.5, 0.4), (0.7, 0.9, 0.6), (1.0, 0.8, 0.1)]:
        f = lambda q: F["F_m"](math.exp(q[0]), math.exp(q[1]), q[2], r0, G2, GY)
        res = minimize(f, [math.log(a0), math.log(k0), b0], method="Nelder-Mead",
                       options=dict(xatol=1e-11, fatol=1e-15, maxiter=40000, maxfev=40000))
        if best is None or res.fun < best.fun:
            best = res
    return newton(np.array([math.exp(best.x[0]), math.exp(best.x[1]), best.x[2]]), r0, F)


def profile(rs):
    F = funs()
    rs = np.asarray(rs, float)
    order = np.argsort(rs)
    rsort = rs[order]
    p1 = anchor(F, 1.0)
    Ps = np.zeros((len(rs), 3))
    i0 = int(np.searchsorted(rsort, 1.0))
    for rng in (range(i0, len(rsort)), range(i0 - 1, -1, -1)):
        p = p1.copy()
        rp = 1.0
        for i in rng:
            rr = rsort[i]
            dmax = min(0.02, 0.2 * min(rr, rp))
            nsub = max(1, int(math.ceil(abs(rr - rp) / dmax)))
            for j in range(1, nsub + 1):
                p = newton(p, rp + (rr - rp) * j / nsub, F)
            rp = rr
            Ps[order[i]] = p
    return Ps


def _bc(L, sh):
    return np.array([np.broadcast_to(np.asarray(v, float), sh) for v in L])


def derived(P, r):
    F = funs()
    r = np.asarray(r, float)
    a, k, be = P[:, 0], P[:, 1], P[:, 2]
    sh = r.shape
    Hl = F["H"](a, k, be, r, G2, GY)
    H = np.array([[np.broadcast_to(np.asarray(v, float), sh) for v in row] for row in Hl]).transpose(2, 0, 1)
    gr = _bc(F["Gr"](a, k, be, r, G2, GY), sh).T
    d = 1.0 / np.sqrt(np.abs(np.einsum("nii->ni", H)))
    Hs = H * d[:, :, None] * d[:, None, :]
    pp = np.linalg.solve(Hs, (-gr * d)[:, :, None])[:, :, 0] * d
    ap, kp, bep = pp.T
    Om = np.broadcast_to(np.asarray(F["Om"](a, k, be, r, G2, GY), float), sh)
    dOm = _bc(F["dOm"](a, k, be, r, G2, GY), sh)
    Omp = dOm[0] * ap + dOm[1] * kp + dOm[2] * bep + dOm[3]
    Etot = np.broadcast_to(np.asarray(F["F"](a, k, be, r, G2, GY), float), sh)
    C0 = Etot - 1.5 * Om
    Bx = 3 / 16 * (2 * (ap / a) ** 2 + (kp / k) ** 2)
    Bw = 3 / 16 * (Omp / Om) ** 2
    Bb = 3 * k * bep ** 2 / (8 * Om)
    c = k * be
    bb = Om + k * be ** 2
    cp = kp * be + k * bep
    bbp = Omp + kp * be ** 2 + 2 * k * be * bep
    det = k * bb - c * c
    m11 = (bb * kp - c * cp) / det
    m12 = (bb * cp - c * bbp) / det
    m21 = (-c * kp + k * cp) / det
    m22 = (-c * cp + k * bbp) / det
    Br = 3 / 16 * (2 * (ap / a) ** 2 + m11 ** 2 + 2 * m12 * m21 + m22 ** 2)
    Oe2 = Om ** 2 + k / 2 * bep ** 2
    return dict(r=r, a=a, k=k, beta=be, Om=Om, ap=ap, kp=kp, betap=bep, Omp=Omp, Etot=Etot, C0=C0, Bx=Bx, Bw=Bw,
                Bb=Bb, Br=Br, Oe2=Oe2)


# ----------------------------------------------------------------------------------------------- inputs
_DV = None


def dV_fun(lam):
    global _DV
    if _DV is None:
        z = np.load(os.path.join(OUT, "R15B_dV_MC.npz"))
        _DV = {kk: z[kk] for kk in z.files}
    return CubicSpline(_DV["r"], _DV["lam%.5f" % lam])


SIG_ROT = np.array([5.53e-4, 42e-4, 24e-4, 17e-4, 14e-4, 12e-4, 9e-4, 9e-4, 7e-4, 4e-4, 4e-4])
TAU = np.array([1, 1, 1, 1, 0.5, 0, -1, -1, -1, -1, -1.0])


def dE_fun(tilt=0.0, endbc="not-a-knot"):
    with open(S3P, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    kn = np.array(d["rot_knots"], float)
    y = np.array(d["rot_dEtot"], float) + tilt * SIG_ROT * TAU
    cs = CubicSpline(kn, y, bc_type=((1, 0.0), (2, 0.0) if endbc == "natural" else endbc))

    def f(r):
        r = np.asarray(r, float)
        return np.where(r <= kn[-1], cs(np.minimum(r, kn[-1])), y[-1])
    return f


# ----------------------------------------------------------------------------------------------- bases
def make_basis(Rmax=14.0, h=0.05, kdeg=7, ng=10):
    grade = [0.0, 0.004, 0.012, 0.025, 0.045, 0.07, 0.1]
    bp = grade + list(np.arange(0.15, Rmax - 0.5 * h, h)) + [Rmax]
    bp = np.array(bp, float)
    t = np.r_[[0.0] * kdeg, bp, [Rmax] * kdeg]
    n = len(t) - kdeg - 1
    xg, wg = np.polynomial.legendre.leggauss(ng)
    xs, ws = [], []
    for lo, hi in zip(bp[:-1], bp[1:]):
        xs.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wg)
    x = np.concatenate(xs)
    w = np.concatenate(ws)
    spl = BSpline(t, np.eye(n), kdeg)
    B = spl(x)[:, 1:-1]
    dB = spl.derivative()(x)[:, 1:-1]
    return dict(x=x, w=w, B=B, dB=dB, n=n - 2, Rmax=Rmax, t=t, kdeg=kdeg, h=h)


def basis_at(bas, r):
    n = len(bas["t"]) - bas["kdeg"] - 1
    return BSpline(bas["t"], np.eye(n), bas["kdeg"])(np.asarray(r, float))[:, 1:-1]


def s_basis(kind="HO", n=30, om0=0.35, Smax=14.0):
    if kind == "HO":
        N = 2 * np.arange(n) + 1.0
        off = np.sqrt((N[:-1] + 1) * (N[:-1] + 2)) / (2 * om0)
        X2 = np.diag((2 * N + 1) / (2 * om0)) + np.diag(off, 1) + np.diag(off, -1)
        K = np.diag((N + 0.5) * om0) - 0.5 * om0 ** 2 * X2
        return K, X2
    j = np.arange(1, n + 1)
    s = j * Smax / (n + 1)
    U = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(j, j) * np.pi / (n + 1))
    K = (U * (0.5 * (j * np.pi / Smax) ** 2)) @ U.T
    return K, np.diag(s ** 2)


_SETUPS = {}


def get_setup(Rmax=14.0, h=0.05):
    key = (round(Rmax, 6), round(h, 6))
    if key not in _SETUPS:
        bas = make_basis(Rmax, h)
        x = bas["x"]
        prof = derived(profile(x), x)
        B, dB, w = bas["B"], bas["dB"], bas["w"]
        S = B.T @ (w[:, None] * B)
        T = 0.5 * (dB.T @ (w[:, None] * dB))
        _SETUPS[key] = dict(bas=bas, prof=prof, S=S, T=T)
    return _SETUPS[key]


def Vmat(st, f):
    B, w = st["bas"]["B"], st["bas"]["w"]
    return B.T @ ((w * f)[:, None] * B)


def pot(st, model, lam, match=True, tilt=0.0, endbc="not-a-knot", extraV=None):
    x = st["bas"]["x"]
    pr = st["prof"]
    V = 3.0 / (8 * x * x) + lam * x ** 4 / 4
    if match:
        V = V + dV_fun(lam)(x)
    Oe2 = None
    if model.startswith("1D"):
        if model == "1D_sep":
            V = V + pr["C0"] + pr["Bx"] + 0.45
        else:
            V = V + pr["Etot"] + pr["Br"]
    else:
        V = V + pr["C0"] + pr["Bx"]
        Oe2 = pr["Oe2"]
        if model == "2D_sep":
            Oe2 = 0.09 * np.ones_like(x)
        if model == "2D_ex_noBx":
            V = V - pr["Bx"]
    if model in ("1D_ex", "2D_ex", "2D_exX", "2D_ex_noBx"):
        V = V + dE_fun(tilt, endbc)(x)
    if extraV is not None:
        V = V + extraV
    return V, Oe2


def matrices(st, model, lam, match, sb, Rc, tilt=0.0, endbc="not-a-knot", extraV=None):
    V, Oe2 = pot(st, model, lam, match, tilt, endbc, extraV)
    H1 = st["T"] + Vmat(st, V)
    S1 = st["S"]
    x = st["bas"]["x"]
    Rmax = st["bas"]["Rmax"]
    Wf = np.where(x > Rc, ((x - Rc) / (Rmax - Rc)) ** 3, 0.0) if Rc is not None else np.zeros_like(x)
    W1 = Vmat(st, Wf)
    if Oe2 is None:
        return sp.csc_matrix(H1), sp.csc_matrix(S1), sp.csc_matrix(W1), 1, None
    Ks, X2 = s_basis(*sb)
    ns = Ks.shape[0]
    Is = sp.identity(ns, format="csr")
    M1 = Vmat(st, Oe2)
    H = (sp.kron(sp.csr_matrix(H1), Is) + sp.kron(sp.csr_matrix(S1), sp.csr_matrix(Ks))
         + 0.5 * sp.kron(sp.csr_matrix(M1), sp.csr_matrix(X2)))
    S = sp.kron(sp.csr_matrix(S1), Is)
    W = sp.kron(sp.csr_matrix(W1), Is)
    return H.tocsc(), S.tocsc(), W.tocsc(), ns, X2


def eig_near(H, S, sigma, nev):
    A = (H - sigma * S).astype(complex).tocsc()
    lu = spla.splu(A)
    Sc = S.astype(complex).tocsr()
    op = spla.LinearOperator(H.shape, matvec=lambda v: lu.solve(np.asarray(Sc @ v, complex)), dtype=complex)
    nev = min(nev, H.shape[0] - 3)
    ncv = min(H.shape[0] - 1, max(2 * nev + 1, 40))
    rng = np.random.default_rng(12345)
    for attempt in range(3):
        v0 = None if attempt == 0 else (rng.standard_normal(H.shape[0]) + 0j)
        try:
            vals, vecs = spla.eigs(op, k=nev, which="LM", tol=1e-13, maxiter=20000, ncv=ncv, v0=v0)
            break
        except spla.ArpackNoConvergence as e:
            vals, vecs = e.eigenvalues, e.eigenvectors
            ncv = min(H.shape[0] - 1, 2 * ncv)
    return sigma + 1.0 / vals, vecs


def p_in(st, vec, ns, Rin=4.0):
    B, w, x = st["bas"]["B"], st["bas"]["w"], st["bas"]["x"]
    psi = B @ vec.reshape(-1, ns)
    rho = (np.abs(psi) ** 2).sum(1)
    return float((w * rho)[x < Rin].sum() / (w * rho).sum())


def cap_scan(st, H, S, W, ns, sig_ref, etas, nev=10, Rin=4.0, win=0.02):
    """sig_ref: ground quasi-bound energy with a Dirichlet wall inside the barrier (unambiguous lowest state).
    Resonance = most localised eigenvalue with |Re E - sig_ref| < win; shift kept fixed at sig_ref."""
    E0, V0 = eig_near(H, S, sig_ref, 12)
    P0 = np.array([p_in(st, V0[:, j], ns, Rin) for j in range(V0.shape[1])])
    cand = [j for j in range(len(E0)) if abs(E0[j].real - sig_ref) < win]
    j = max(cand, key=lambda q: P0[q]) if cand else int(np.argmax(P0))
    box = dict(Q=float(E0[j].real), Pin=float(P0[j]), sig_ref=sig_ref, ncand=len(cand))
    rows = []
    for eta in etas:
        E, Vv = eig_near(H - 1j * eta * W, S, sig_ref, nev)
        P = np.array([p_in(st, Vv[:, i], ns, Rin) for i in range(len(E))])
        cand = [q for q in range(len(E)) if abs(E[q].real - sig_ref) < win]
        if not cand:
            continue
        i = max(cand, key=lambda q: P[q])
        v = Vv[:, i]
        dEde = -1j * (v @ (W @ v)) / (v @ (S @ v))
        Ec = E[i] - eta * dEde
        rows.append(dict(eta=float(eta), E=complex(E[i]), Ec=complex(Ec), dEde=complex(dEde), Pin=float(P[i]),
                         P2=float(np.sort(P)[-2]) if len(P) > 1 else 0.0, ncand=len(cand)))
    if len(rows) < 3:
        return dict(box=box, rows=rows, ok=False)
    Ecs = np.array([r["Ec"] for r in rows])
    Es = np.array([r["E"] for r in rows])
    good = np.array([r["Pin"] > 0.5 for r in rows])
    best = None
    for i in range(1, len(rows) - 1):
        if good[i - 1] and good[i] and good[i + 1]:
            dd = abs(Ecs[i + 1] - Ecs[i - 1])
            if best is None or dd < best[0]:
                best = (dd, i)
    out = dict(box=box, rows=rows)
    if best is None:
        out["ok"] = False
        return out
    i = best[1]
    lo, hi = max(0, i - 2), min(len(rows), i + 3)
    gs = -2 * Ecs[lo:hi].imag
    gs = gs[good[lo:hi]]
    g = -2 * Ecs[i].imag
    jr = int(np.argmin([abs(r["eta"] * r["dEde"]) if r["Pin"] > 0.5 else 1e99 for r in rows]))
    out.update(ok=True, eta_opt=rows[i]["eta"], Q=float(Ecs[i].real), gamma=float(g),
               gamma_spread=float(np.max(np.abs(gs - g)) / g), Q_spread=float(np.max(np.abs(Ecs[lo:hi].real - Ecs[i].real))),
               Pin=rows[i]["Pin"], raw_Q=float(Es[jr].real), raw_gamma=float(-2 * Es[jr].imag), raw_eta=rows[jr]["eta"])
    return out


def EX_array(st):
    z = np.load(os.path.join(OUT, "R15B_lam0_s2.npz"))
    rf, s2f, rhof = z["r"], z["s2"], z["rho"]
    TB = float(np.load(os.path.join(OUT, "R15B_dV_MC.npz"))["TB"][0])
    rcut = float(rf[rhof > 1e-8 * rhof.max()].max())
    m = rf <= rcut
    cs = CubicSpline(rf[m], s2f[m])
    pr = st["prof"]
    x = pr["r"]
    OeS = CubicSpline(x, np.sqrt(pr["Oe2"]))
    sc = float(cs(rcut)) / (1.5 / float(OeS(rcut)))
    s2 = np.where(x <= rcut, cs(np.clip(x, rf[m].min(), rcut)), sc * 1.5 / np.sqrt(pr["Oe2"]))
    EX = TB / 2 * G2 ** 2 * (2 / pr["a"] + 1 / pr["k"]) + TB / 3 * (GY ** 2 + G2 ** 2 * pr["beta"] ** 2) * s2
    return EX, rcut, TB


# ----------------------------------------------------------------------------------------------- tasks
def task_lam0(task):
    t0 = time.time()
    st = get_setup(task.get("Rmax", 12.0), task.get("h", 0.05))
    sb = tuple(task.get("sb", SB0))
    H, S, W, ns, X2 = matrices(st, task["model"], 0.0, False, sb, None, tilt=task.get("tilt", 0.0),
                               endbc=task.get("endbc", "not-a-knot"))
    E, Vv = eig_near(H, S, 3.5, 6)
    i = int(np.argmin(E.real))
    out = dict(task=task, E0=float(E[i].real), Eim=float(E[i].imag))
    if task.get("detail"):
        v = Vv[:, i]
        v = v / np.sqrt(v @ (S @ v))
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        v = v.real
        C = v.reshape(-1, ns)
        rf = np.linspace(0.01, 8.0, 800)
        Bf = basis_at(st["bas"], rf)
        psi = Bf @ C
        rho = (psi ** 2).sum(1)
        s2 = np.einsum("rn,nm,rm->r", psi, X2, psi) / np.maximum(rho, 1e-300)
        np.savez(os.path.join(OUT, "R15B_lam0_s2.npz"), r=rf, s2=s2, rho=rho)
        rt = np.array([0.4, 0.6, 1.0, 1.22, 1.5, 2.0, 3.0])
        prt = derived(profile(rt), rt)
        Bt = basis_at(st["bas"], rt)
        pst = Bt @ C
        rht = (pst ** 2).sum(1)
        s2t = np.einsum("rn,nm,rm->r", pst, X2, pst) / rht
        B, w, x = st["bas"]["B"], st["bas"]["w"], st["bas"]["x"]
        pq = B @ C
        rq = (pq ** 2).sum(1)
        s2q = np.einsum("rn,nm,rm->r", pq, X2, pq)
        pr = st["prof"]
        out["detail"] = dict(r=rt, s2_2D=s2t, s2_BO=1.5 / prt["Om"], rho=rht,
                             s2_global_2D=float((w * s2q).sum()), s2_global_BO=float((w * rq * 1.5 / pr["Om"]).sum()),
                             r_mean=float((w * rq * x).sum()), P_r_lt_065=float((w * rq)[x < 0.65].sum()),
                             norm=float((w * rq).sum()))
    out["time"] = time.time() - t0
    return out


def task_res(task):
    t0 = time.time()
    st = get_setup(task.get("Rmax", 14.0), task.get("h", 0.05))
    sb = tuple(task.get("sb", SB0))
    model = task["model"]
    extraV = None
    if model == "2D_exX":
        extraV, rcut, TB = EX_array(st)
    H, S, W, ns, X2 = matrices(st, model, task["lam"], True, sb, task.get("Rc", 9.0), tilt=task.get("tilt", 0.0),
                               endbc=task.get("endbc", "not-a-knot"), extraV=extraV)
    # reference: lowest eigenvalue with a Dirichlet wall at r = 4.5 (inside the barrier for all lambda used)
    stw = get_setup(task.get("Rwall", 4.5), task.get("h", 0.05))
    exw = EX_array(stw)[0] if model == "2D_exX" else None
    Hw, Sw, Ww, nsw, X2w = matrices(stw, model, task["lam"], True, sb, None, tilt=task.get("tilt", 0.0),
                                    endbc=task.get("endbc", "not-a-knot"), extraV=exw)
    Ew, _ = eig_near(Hw, Sw, 3.5, 4)
    sig_ref = float(np.min(Ew.real))
    etas = np.geomspace(task.get("eta_min", 10.0), task.get("eta_max", 2000.0), task.get("n_eta", 13))
    out = cap_scan(st, H, S, W, ns, sig_ref, etas, nev=task.get("nev", 10))
    out["task"] = task
    out["dim"] = H.shape[0]
    out["time"] = time.time() - t0
    return out


def task_stab(task):
    t0 = time.time()
    sb = tuple(task.get("sb", SB0))
    rows = []
    for Rw in task["Rws"]:
        st = get_setup(Rw, task.get("h", 0.05))
        H, S, W, ns, X2 = matrices(st, task["model"], task["lam"], True, sb, None)
        E, Vv = eig_near(H, S, task["Qref"], 12)
        P = [p_in(st, Vv[:, j], ns) for j in range(len(E))]
        o = np.argsort(E.real)
        rows.append(dict(Rw=Rw, E=[float(E[j].real) for j in o], P=[P[j] for j in o]))
    return dict(task=task, rows=rows, time=time.time() - t0)


def pmap(fn, tasks, nproc=3):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(nproc) as pool:
        return pool.map(fn, tasks, chunksize=1)


# ----------------------------------------------------------------------------------------------- main
def cmd_prep():
    t0 = time.time()
    out = {}
    sys.path.insert(0, MATCH_CODE)
    import C138_core as CC
    rg = np.arange(0.005, 16.6 + 1e-9, 0.005)
    d = {"r": rg, "TB": np.array([2 * CC.TBH])}
    for lam in [-0.005, -0.006] + LAMS:
        d["lam%.5f" % lam] = CC.dV("MC", rg, lam, G2, GY, YT)
    np.savez(os.path.join(OUT, "R15B_dV_MC.npz"), **d)
    out["TB"] = 2 * CC.TBH
    # profile checks (item 1)
    r3 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7, 4.5, 5.5, 7.0, 9.0])
    E3 = [2.98637, 3.28884, 3.66714, 4.10464, 4.58780, 5.10626, 5.87612, 6.79986, 7.99677, 9.84383, 12.35837]
    p3 = derived(profile(r3), r3)
    out["Etot_vs_Stage3"] = dict(r=r3, mine=p3["Etot"], stage3=E3, diff=p3["Etot"] - np.array(E3))
    rt = np.array([0.004, 0.01, 0.1, 0.3, 0.65, 1.0, 1.22, 1.5, 2.0, 3.0, 4.5, 5.44, 9.0, 14.0])
    pt = derived(profile(rt), rt)
    out["Br_decomp"] = dict(r=rt, Om=pt["Om"], gYr2=GY * rt / 2, a=pt["a"], k=pt["k"], beta=pt["beta"], Bx=pt["Bx"],
                            Bw=pt["Bw"], Bb=pt["Bb"], Br_tr=pt["Br"], r2Bw=rt ** 2 * pt["Bw"],
                            rel_tr_minus_sum=(pt["Br"] - pt["Bx"] - pt["Bw"] - pt["Bb"]) / pt["Br"],
                            Oe2=pt["Oe2"], C0=pt["C0"], Etot=pt["Etot"])
    # optimum check: b~ = Omega minimises; compare F with direct 4-parameter D108 E_tot at perturbed b~
    out["time"] = time.time() - t0
    jdump(out, "R15B_prep.json")
    np.set_printoptions(linewidth=200, precision=6)
    print("TB", out["TB"])
    print("Etot - Stage3:", np.round(out["Etot_vs_Stage3"]["diff"], 6))
    bd = out["Br_decomp"]
    for i, r in enumerate(rt):
        print("r=%6.3f Om=%.4f gYr/2=%.4f Bx=%.5f Bw=%.5f Bb=%.5f r2Bw=%.6f rel(tr-sum)=%.1e beta=%.4f Oe2=%.5f" % (
            r, bd["Om"][i], bd["gYr2"][i], bd["Bx"][i], bd["Bw"][i], bd["Bb"][i], bd["r2Bw"][i],
            bd["rel_tr_minus_sum"][i], bd["beta"][i], bd["Oe2"][i]))
    print("prep time", out["time"])


def cmd_lam0():
    tasks = [dict(model="2D_ex", detail=True)]
    for m in ("1D_G", "1D_ex", "2D_G", "2D_ex_noBx", "2D_sep", "1D_sep"):
        tasks.append(dict(model=m))
    tasks += [dict(model="2D_ex", tilt=1.0), dict(model="2D_ex", tilt=-1.0), dict(model="2D_ex", endbc="natural"),
              dict(model="2D_ex", h=0.035), dict(model="2D_ex", sb=("HO", 40, 0.35, 14.0)),
              dict(model="2D_ex", sb=("HO", 30, 0.25, 14.0)), dict(model="2D_ex", Rmax=10.0),
              dict(model="2D_ex", sb=("SINE", 48, 0.0, 14.0)), dict(model="1D_G", h=0.035), dict(model="1D_ex", tilt=1.0),
              dict(model="1D_ex", tilt=-1.0), dict(model="2D_G", h=0.035)]
    res = pmap(task_lam0, tasks)
    jdump(res, "R15B_lam0.json")
    for r in res:
        print("%-55s E0=%.7f  (-C72 %+.5f)  t=%.1fs" % (json.dumps(r["task"])[:55], r["E0"], r["E0"] - 4.4478, r["time"]))
        if "detail" in r:
            d = r["detail"]
            print("   r      ", np.round(d["r"], 3))
            print("   s2_2D  ", np.round(d["s2_2D"], 3))
            print("   s2_BO  ", np.round(d["s2_BO"], 3))
            print("   rho    ", np.round(d["rho"], 3))
            print("   global s2 2D/BO %.3f / %.3f  <r>=%.3f P(r<.65)=%.4f norm=%.6f" % (
                d["s2_global_2D"], d["s2_global_BO"], d["r_mean"], d["P_r_lt_065"], d["norm"]))


def summarize_res(res, tag):
    for r in res:
        t = r["task"]
        if r.get("ok"):
            print("%-6s %-10s lam=%8.4f Q=%.6f gamma=%.5e spr=%.1e Qspr=%.1e Pin=%.4f eta*=%.3g raw=%.5e box=%.5f dim=%d t=%.0fs %s" % (
                tag, t["model"], t["lam"], r["Q"], r["gamma"], r["gamma_spread"], r["Q_spread"], r["Pin"], r["eta_opt"],
                r["raw_gamma"], r["box"]["Q"], r["dim"], r["time"],
                json.dumps({kk: vv for kk, vv in t.items() if kk not in ("model", "lam")})))
        else:
            print(tag, t, "NOT OK", r["box"])


def cmd_res():
    tasks = []
    for lam in LAMS:
        for m in ("2D_ex", "2D_G"):
            tasks.append(dict(model=m, lam=lam))
    for lam in LAMS:
        for m in ("1D_G", "1D_ex"):
            tasks.append(dict(model=m, lam=lam))
    res = pmap(task_res, tasks)
    jdump(res, "R15B_res.json")
    summarize_res(res, "res")


def cmd_conv():
    tasks = []
    for lam in (-0.010, -0.0143):
        base = dict(model="2D_ex", lam=lam, eta_min=20.0, eta_max=1000.0, n_eta=9)
        for extra in (dict(sb=("HO", 22, 0.35, 14.0)), dict(sb=("HO", 40, 0.35, 14.0)), dict(sb=("HO", 30, 0.25, 14.0)),
                      dict(sb=("SINE", 32, 0.0, 10.0)), dict(sb=("SINE", 44, 0.0, 14.0)),
                      dict(sb=("SINE", 56, 0.0, 18.0)), dict(Rmax=13.0, Rc=8.0), dict(Rmax=15.5, Rc=10.0), dict(h=0.04),
                      dict(tilt=1.0), dict(tilt=-1.0)):
            t = dict(base)
            t.update(extra)
            tasks.append(t)
        for extra in (dict(Rmax=13.0, Rc=8.0), dict(Rmax=15.5, Rc=10.0), dict(h=0.04)):
            t = dict(model="1D_G", lam=lam, eta_min=20.0, eta_max=1000.0, n_eta=9)
            t.update(extra)
            tasks.append(t)
    tasks.sort(key=lambda t: -(t.get("sb", SB0)[1] if t.get("sb", SB0)[0] == "SINE" else 0))
    res = pmap(task_res, tasks)
    jdump(res, "R15B_conv.json")
    summarize_res(res, "conv")


def cmd_cross():
    tasks = [dict(model="2D_exX", lam=lam) for lam in LAMS]
    res = pmap(task_res, tasks)
    jdump(res, "R15B_cross.json")
    summarize_res(res, "cross")


def cmd_stab():
    with open(os.path.join(OUT, "R15B_res.json"), "r", encoding="utf-8") as fh:
        rr = json.load(fh)
    Q = {(r["task"]["model"], r["task"]["lam"]): r["Q"] for r in rr if r.get("ok")}
    tasks = []
    for lam in (-0.0143, -0.010):
        Rws = list(np.round(np.arange(7.0, 10.01, 0.2), 3))
        tasks.append(dict(model="2D_ex", lam=lam, Qref=Q[("2D_ex", lam)], Rws=Rws[0::2]))
        tasks.append(dict(model="2D_ex", lam=lam, Qref=Q[("2D_ex", lam)], Rws=Rws[1::2]))
    res = pmap(task_stab, tasks)
    jdump(res, "R15B_stab.json")
    for r in res:
        t = r["task"]
        for row in r["rows"]:
            sel = [(round(e, 5), round(p, 3)) for e, p in zip(row["E"], row["P"]) if abs(e - t["Qref"]) < 0.03]
            print("stab lam=%.4f Qref=%.5f Rw=%.2f" % (t["lam"], t["Qref"], row["Rw"]), sel)


def cmd_post():
    with open(os.path.join(OUT, "R15B_res.json"), "r", encoding="utf-8") as fh:
        rr = json.load(fh)
    G = {}
    Qd = {}
    for r in rr:
        if r.get("ok"):
            G[(r["task"]["model"], r["task"]["lam"])] = r["gamma"]
            Qd[(r["task"]["model"], r["task"]["lam"])] = r["Q"]
    out = {"ratios": {}, "lambda_c": {}, "hazard": {}}
    lams = sorted(LAMS)
    for num, den in (("2D_ex", "1D_G"), ("2D_ex", "1D_ex"), ("2D_G", "1D_G"), ("1D_ex", "1D_G")):
        out["ratios"]["%s/%s" % (num, den)] = {str(l): G[(num, l)] / G[(den, l)] for l in lams}
    with open(os.path.join(C138RES, "C138_static.json"), "r", encoding="utf-8") as fh:
        gc = json.load(fh)["gamma_c"]
    gcs = {"P1=0.9": gc["P1=0.9"], "P1=0.5": gc["P1=0.5"]}
    for m in ("1D_G", "1D_ex", "2D_G", "2D_ex"):
        cs = CubicSpline(lams, [math.log(G[(m, l)]) for l in lams])
        out["lambda_c"][m] = {kk: brentq(lambda l: float(cs(l)) - math.log(v), lams[0], lams[-1]) for kk, v in gcs.items()}
    for m in ("1D_ex", "2D_G", "2D_ex"):
        out["lambda_c"][m]["dmt_static"] = {kk: (out["lambda_c"][m][kk] - out["lambda_c"]["1D_G"][kk]) / (-0.00656)
                                            for kk in gcs}
    with open(os.path.join(C138RES, "C138_hazard.json"), "r", encoding="utf-8") as fh:
        hz = json.load(fh)
    u = np.array(hz["meta"]["u_main"], float)
    rows = hz["sets"]["MC|c=1|gphi=None|ct=None|as=0.1180|main"]["rows"]
    crit = hz["sets"]["MC|c=1|gphi=None|ct=None|as=0.1180|main"]["crit"]
    mt = np.array([r["mt"] for r in rows])
    lnI = CubicSpline(mt, np.log([r["I"] for r in rows]))
    lamb = CubicSpline(mt, [r["lam_b"] for r in rows])
    gn = np.array([r["gamma_nodes"] for r in rows])  # (nm, nu)
    wu = np.zeros_like(u)
    wu[1:] += 0.5 * np.diff(u)
    wu[:-1] += 0.5 * np.diff(u)
    wu = wu / np.sqrt(np.cosh(2 * u))
    dlam_u = -(-2.97e-4) / 2 * np.log(np.cosh(2 * u))
    lng = [CubicSpline(mt, np.log(np.maximum(gn[:, j], 1e-300))) for j in range(len(u))]
    P1_172p6_orig = math.exp(-math.exp(float(lnI(172.6))))
    for m in ("1D_ex", "2D_G", "2D_ex"):
        lrho = CubicSpline(lams, [math.log(G[(m, l)] / G[("1D_G", l)]) for l in lams])

        def rho_eff(mm):
            gam = np.array([math.exp(float(f(mm))) for f in lng])
            lb = float(lamb(mm))
            rr_ = np.exp(lrho(np.clip(lb + dlam_u, lams[0], lams[-1])))
            return float((wu * gam * rr_).sum() / (wu * gam).sum())
        res = {}
        for P in (0.9, 0.5):
            target = math.log(-math.log(P))
            m0 = brentq(lambda mm: float(lnI(mm)) - target, mt[0], mt[-1])
            slope = float(lnI(m0, 1))
            lin = m0 - float(lrho(float(lamb(m0)))) / slope
            # bracket extended 0.4 GeV beyond the last C138 row (cubic extrapolation of ln I, lambda_b: nearly linear)
            exact = brentq(lambda mm: float(lnI(mm)) + float(lrho(float(lamb(mm)))) - target, mt[0], mt[-1] + 0.4)
            uint = brentq(lambda mm: float(lnI(mm)) + math.log(rho_eff(mm)) - target, mt[0], mt[-1] + 0.4)
            res["P1=%.1f" % P] = dict(m0=m0, m0_C138=crit["P1=%.1f" % P], dlnI_dm=slope, lam_b_m0=float(lamb(m0)),
                                      rho_at_lamb_m0=math.exp(float(lrho(float(lamb(m0))))),
                                      mc_linear=lin, mc_exact_rho_lamb=exact, mc_u_integrated=uint,
                                      shift_linear=lin - m0, shift_exact=exact - m0, shift_uint=uint - m0)
        I6 = math.exp(float(lnI(172.6)))
        res["P1(172.60)"] = dict(orig=P1_172p6_orig, rho_lamb=math.exp(-I6 * math.exp(float(lrho(float(lamb(172.6)))))),
                                 rho_uint=math.exp(-I6 * rho_eff(172.6)), lam_b=float(lamb(172.6)))
        out["hazard"][m] = res
    try:
        with open(os.path.join(OUT, "R15B_cross.json"), "r", encoding="utf-8") as fh:
            cr = json.load(fh)
        GX = {r["task"]["lam"]: r["gamma"] for r in cr if r.get("ok")}
        lx = sorted(GX)
        lrx = CubicSpline(lx, [math.log(GX[l] / G[("2D_ex", l)]) for l in lx])
        cx = {"ratio_X": {str(l): GX[l] / G[("2D_ex", l)] for l in lx}}
        for P in (0.9, 0.5):
            hzm = out["hazard"]["2D_ex"]["P1=%.1f" % P]
            mnew = hzm["mc_exact_rho_lamb"]
            lb = float(lamb(mnew))
            cx["P1=%.1f" % P] = dict(lam_b=lb, ratio=math.exp(float(lrx(lb))),
                                     dm_hazard=-float(lrx(lb)) / float(lnI(mnew, 1)))
        out["cross"] = cx
    except FileNotFoundError:
        pass
    out["Q"] = {"%s|%s" % (m, l): q for (m, l), q in Qd.items()}
    out["gamma"] = {"%s|%s" % (m, l): g for (m, l), g in G.items()}
    jdump(out, "R15B_post.json")
    print(json.dumps({kk: out[kk] for kk in ("ratios", "lambda_c", "hazard")}, indent=1))
    if "cross" in out:
        print(json.dumps(out["cross"], indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ids":
        t0 = time.time()
        o = identities()
        o["time"] = time.time() - t0
        jdump(o, "R15B_ids.json")
        print(json.dumps(o, indent=1, ensure_ascii=False))
    elif cmd == "prep":
        cmd_prep()
    elif cmd == "lam0":
        cmd_lam0()
    elif cmd == "res":
        cmd_res()
    elif cmd == "conv":
        cmd_conv()
    elif cmd == "cross":
        cmd_cross()
    elif cmd == "stab":
        cmd_stab()
    elif cmd == "post":
        cmd_post()
