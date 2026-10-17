# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_common.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Shared C73 code (D55): own twisted Ewald and Chowla-Selberg sums (mpmath), shape metric geodesics, Richardson extrapolation. Paper: PRD Sec. VII (sec:shape), App. app:shapelaplacian (shape Laplacian).

C73 common helpers (independent implementation; no Codex code is used or imported).

Conventions
-----------
* Shape basis X_A (A=1..5), tr(X_A X_B) = delta_AB:
  X1 = diag(1,-1,0)/sqrt2, X2 = diag(1,1,-2)/sqrt6, X3,X4,X5 = (E_ij+E_ji)/sqrt2 for ij = 12,13,23.
* Geodesic through G0 in direction X:  G(t) = S exp(2 t X) S,  S = G0^{1/2} (symmetric root).
  The shape Laplacian is  Delta_sh f(G0) = sum_A d^2/dt^2 f(G(t))|_{t=0}.
* Twisted Epstein zeta (theta = 2 pi a):
      Z_G(s; a) = sum'_{n in Z^3} cos(2 pi a.n) (n^T G n)^{-s}      (analytic continuation in s)
  - ewald_Z : theta-function (Ewald) split at Mellin parameter lam (result must be lam-independent)
      pi^{-s} Gamma(s) Z = sum'_n cos(2pi a.n) (pi Q)^{-s} Gamma(s, pi lam Q)
                          + detG^{-1/2} [ sum_{m != a} (pi P_m)^{s-3/2} Gamma(3/2-s, pi P_m/lam)
                                          + delta_{a in Z^3} lam^{s-3/2}/(s-3/2) ] - lam^s/s,
      Q = n^T G n,  P_m = (m-a)^T G^{-1} (m-a).
  - cs_Z : Chowla-Selberg (Poisson resummation in n1, Bessel-K series), recursive 3D -> 2D.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
from pathlib import Path
import numpy as np
import mpmath as mp

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
RES = WP.wdir("fu/C73")
RES.mkdir(parents=True, exist_ok=True)

# non-cubic test shape: all five shape directions switched on (gives off-diagonal G)
BETA_NC = ["0.21", "-0.13", "0.08", "-0.06", "0.11"]


# ----------------------------------------------------------------------------- basis / geodesics
def basis_mp():
    s2, s6 = mp.sqrt(2), mp.sqrt(6)
    return [mp.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / s2,
            mp.matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / s6,
            mp.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / s2,
            mp.matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / s2,
            mp.matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / s2]


def basis_np():
    s2, s6 = np.sqrt(2.0), np.sqrt(6.0)
    return [np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], float) / s2,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], float) / s6,
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float) / s2,
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], float) / s2,
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float) / s2]


def symfun_mp(A, f):
    E, Q = mp.eigsy(A)
    n = A.rows
    D = mp.zeros(n, n)
    for i in range(n):
        D[i, i] = f(E[i])
    M = Q * D * Q.T
    return (M + M.T) / 2


def shape_metric_mp(beta):
    X = basis_mp()
    Y = mp.zeros(3, 3)
    for b, Xa in zip(beta, X):
        Y += mp.mpf(b) * Xa
    return symfun_mp(Y, lambda y: mp.exp(2 * y))


class GeodesicMP:
    """Points S exp(2 t X_A) S on the five orthonormal geodesics through G0 = S^2."""

    def __init__(self, G0):
        self.G0 = G0
        self.S = symfun_mp(G0, mp.sqrt)
        self.Sinv = mp.inverse(self.S)
        self.X = basis_mp()
        self.eig = [mp.eigsy(Xa) for Xa in self.X]

    def point(self, A, t):
        E, Q = self.eig[A]
        D = mp.zeros(3, 3)
        for i in range(3):
            D[i, i] = mp.exp(2 * mp.mpf(t) * E[i])
        M = self.S * (Q * D * Q.T) * self.S
        return (M + M.T) / 2


def shape_metric_np(beta):
    X = basis_np()
    Y = sum(float(b) * Xa for b, Xa in zip(beta, X))
    w, V = np.linalg.eigh(Y)
    return (V * np.exp(2 * w)) @ V.T


class GeodesicNP:
    def __init__(self, G0):
        w, V = np.linalg.eigh(G0)
        self.S = (V * np.sqrt(w)) @ V.T
        self.Sinv = (V / np.sqrt(w)) @ V.T
        self.X = basis_np()
        self.eig = [np.linalg.eigh(Xa) for Xa in self.X]

    def point(self, A, t):
        w, V = self.eig[A]
        M = self.S @ ((V * np.exp(2 * t * w)) @ V.T) @ self.S
        return 0.5 * (M + M.T)


def fd_laplacian(f, geo, eps_list, center_val=None):
    """Second differences along the five geodesics, for each step in eps_list.
    Returns list of D(eps) = sum_A [f(+eps)-2f(0)+f(-eps)]/eps^2 (works for mp or float)."""
    f0 = f(geo.G0) if center_val is None else center_val
    out = []
    for eps in eps_list:
        tot = 0
        for A in range(5):
            tot += (f(geo.point(A, eps)) - 2 * f0 + f(geo.point(A, -eps))) / eps ** 2
        out.append(tot)
    return out


def richardson(vals, ratio=2, order=2):
    """Richardson table for errors ~ eps^2, eps^4, ... with eps_k = eps_0/ratio^k."""
    table = [list(vals)]
    p = order
    while len(table[-1]) > 1:
        prev = table[-1]
        fac = ratio ** p
        table.append([(fac * prev[i + 1] - prev[i]) / (fac - 1) for i in range(len(prev) - 1)])
        p += 2
    return table


# ----------------------------------------------------------------------------- Ewald (mpmath)
def _ginc(z, x):
    """Upper incomplete gamma Gamma(z, x) with closed forms for the cases used most."""
    if z == 1:
        return mp.exp(-x)
    if z == 2:
        return (1 + x) * mp.exp(-x)
    if z == mp.mpf(1) / 2:
        return mp.sqrt(mp.pi) * mp.erfc(mp.sqrt(x))
    if z == -mp.mpf(1) / 2:
        return 2 * (mp.exp(-x) / mp.sqrt(x) - mp.sqrt(mp.pi) * mp.erfc(mp.sqrt(x)))
    return mp.gammainc(z, x)


def _as_list(G):
    return [[mp.mpf(G[i, j]) for j in range(3)] for i in range(3)]


def _quad(g, v):
    return (g[0][0] * v[0] ** 2 + g[1][1] * v[1] ** 2 + g[2][2] * v[2] ** 2
            + 2 * (g[0][1] * v[0] * v[1] + g[0][2] * v[0] * v[2] + g[1][2] * v[1] * v[2]))


def _is_int(x):
    return x == mp.floor(x)


def ewald_Z(s, G, a=(0, 0, 0), lam=1):
    s = mp.mpf(s)
    lam = mp.mpf(lam)
    a = [mp.mpf(x) for x in a]
    g = _as_list(G)
    Gm = mp.matrix(g)
    gi = _as_list(mp.inverse(Gm))
    det = mp.det(Gm)
    xmax = (mp.mp.dps + 10) * mp.log(10) + 20
    # direct (image) sum
    Qmax = xmax / (mp.pi * lam)
    b = [int(mp.floor(mp.sqrt(Qmax * gi[i][i]))) + 1 for i in range(3)]
    direct = mp.mpf(0)
    for n in itertools.product(range(-b[0], b[0] + 1), range(-b[1], b[1] + 1), range(-b[2], b[2] + 1)):
        if n == (0, 0, 0):
            continue
        Q = _quad(g, n)
        if Q > Qmax:
            continue
        ph = mp.cospi(2 * (a[0] * n[0] + a[1] * n[1] + a[2] * n[2]))
        direct += ph * (mp.pi * Q) ** (-s) * _ginc(s, mp.pi * lam * Q)
    # dual (reciprocal) sum
    Pmax = xmax * lam / mp.pi
    rng = []
    for i in range(3):
        h = mp.sqrt(Pmax * g[i][i])
        rng.append(range(int(mp.ceil(a[i] - h)) - 1, int(mp.floor(a[i] + h)) + 2))
    dual = mp.mpf(0)
    for m in itertools.product(*rng):
        v = [m[i] - a[i] for i in range(3)]
        P = _quad(gi, v)
        if P == 0 or P > Pmax:
            continue
        dual += (mp.pi * P) ** (s - mp.mpf(3) / 2) * _ginc(mp.mpf(3) / 2 - s, mp.pi * P / lam)
    pole = lam ** (s - mp.mpf(3) / 2) / (s - mp.mpf(3) / 2) if all(_is_int(x) for x in a) else 0
    val = direct + (dual + pole) / mp.sqrt(det) - lam ** s / s
    return val * mp.pi ** s / mp.gamma(s)


# ----------------------------------------------------------------------------- Chowla-Selberg (mpmath)
def _cosser(sig, a):
    """sum_{k>=1} cos(2 pi a k) k^{-sig} (analytically continued)."""
    a = mp.mpf(a)
    fa = a - mp.floor(a)
    if fa == 0:
        return mp.zeta(sig)
    if fa == mp.mpf(1) / 2:
        return -(1 - mp.mpf(2) ** (1 - sig)) * mp.zeta(sig)
    return mp.re(mp.polylog(sig, mp.expjpi(2 * a)))


def _mindist(a):
    fa = a - mp.floor(a)
    if fa == 0:
        return mp.mpf(1)
    return min(fa, 1 - fa)


def cs_Z2D(w, B, a2, a3):
    w = mp.mpf(w)
    b11, b12, b22 = B[0][0], B[0][1], B[1][1]
    c2 = b12 / b11
    A2 = b22 - b12 ** 2 / b11
    res = 2 * b11 ** (-w) * _cosser(2 * w, a2)
    if _is_int(a2):
        res += (mp.sqrt(mp.pi / b11) * mp.gamma(w - mp.mpf(1) / 2) / mp.gamma(w)
                * A2 ** (mp.mpf(1) / 2 - w) * 2 * _cosser(2 * w - 1, a3))
    zmax = (mp.mp.dps + 10) * mp.log(10) + 20
    r = mp.sqrt(A2 / b11)
    d2 = _mindist(a2)
    n3max = int(mp.floor(zmax / (2 * mp.pi * d2 * r))) + 1
    nu = w - mp.mpf(1) / 2
    acc = mp.mpf(0)
    for n3 in range(-n3max, n3max + 1):
        if n3 == 0:
            continue
        be = r * abs(n3)
        kspan = zmax / (2 * mp.pi * be)
        for k in range(int(mp.floor(a2 - kspan)) - 1, int(mp.ceil(a2 + kspan)) + 2):
            x = abs(k - a2)
            if x == 0:
                continue
            z = 2 * mp.pi * be * x
            if z > zmax:
                continue
            ph = mp.cospi(2 * a3 * n3 + 2 * (k - a2) * c2 * n3)
            acc += ph * (mp.pi * x / be) ** nu * mp.besselk(nu, z)
    return res + b11 ** (-w) * 2 * mp.sqrt(mp.pi) / mp.gamma(w) * acc


def cs_Z_raw(s, G, a):
    s = mp.mpf(s)
    a1, a2, a3 = [mp.mpf(x) for x in a]
    g = _as_list(G)
    g11 = g[0][0]
    B = [[g[1][1] - g[0][1] ** 2 / g11, g[1][2] - g[0][1] * g[0][2] / g11],
         [g[1][2] - g[0][1] * g[0][2] / g11, g[2][2] - g[0][2] ** 2 / g11]]
    res = 2 * g11 ** (-s) * _cosser(2 * s, a1)
    if _is_int(a1):
        res += mp.sqrt(mp.pi / g11) * mp.gamma(s - mp.mpf(1) / 2) / mp.gamma(s) * cs_Z2D(s - mp.mpf(1) / 2, B, a2, a3)
    zmax = (mp.mp.dps + 10) * mp.log(10) + 20
    d1 = _mindist(a1)
    beta_max = zmax / (2 * mp.pi * d1)
    Amax = g11 * beta_max ** 2
    Bm = mp.matrix(B)
    Bi = mp.inverse(Bm)
    b2 = int(mp.floor(mp.sqrt(Amax * Bi[0, 0]))) + 1
    b3 = int(mp.floor(mp.sqrt(Amax * Bi[1, 1]))) + 1
    nu = s - mp.mpf(1) / 2
    acc = mp.mpf(0)
    for n2 in range(-b2, b2 + 1):
        for n3 in range(-b3, b3 + 1):
            if n2 == 0 and n3 == 0:
                continue
            A = B[0][0] * n2 ** 2 + 2 * B[0][1] * n2 * n3 + B[1][1] * n3 ** 2
            if A > Amax:
                continue
            be = mp.sqrt(A / g11)
            c = (g[0][1] * n2 + g[0][2] * n3) / g11
            kspan = zmax / (2 * mp.pi * be)
            for k in range(int(mp.floor(a1 - kspan)) - 1, int(mp.ceil(a1 + kspan)) + 2):
                x = abs(k - a1)
                if x == 0:
                    continue
                z = 2 * mp.pi * be * x
                if z > zmax:
                    continue
                ph = mp.cospi(2 * (a2 * n2 + a3 * n3) + 2 * (k - a1) * c)
                acc += ph * (mp.pi * x / be) ** nu * mp.besselk(nu, z)
    return res + g11 ** (-s) * 2 * mp.sqrt(mp.pi) / mp.gamma(s) * acc


def cs_Z(s, G, a, delta=None):
    """Chowla-Selberg value; if the first characteristic is integral and s is such that the
    internal 2D decomposition has cancelling poles (s = 1), average s = 1 +/- delta."""
    if delta is None:
        return cs_Z_raw(s, G, a)
    s = mp.mpf(s)
    d = mp.mpf(delta)
    return (cs_Z_raw(s + d, G, a) + cs_Z_raw(s - d, G, a)) / 2


# ----------------------------------------------------------------------------- numpy Ewald, s = 2
_RNP = 8
_GRID = np.array(list(itertools.product(range(-_RNP, _RNP + 1), repeat=3)), dtype=float)
_NZ = _GRID[np.any(_GRID != 0, axis=1)]


def ewald_Z2_np(G, a=(0.0, 0.0, 0.0)):
    from scipy.special import erfc
    a = np.asarray(a, float)
    Q = np.einsum('ni,ij,nj->n', _NZ, G, _NZ)
    ph = np.cos(2 * np.pi * (_NZ @ a))
    direct = np.sum(ph * (1 + np.pi * Q) * np.exp(-np.pi * Q) / (np.pi * Q) ** 2)
    Gi = np.linalg.inv(G)
    V = _GRID - a
    P = np.einsum('ni,ij,nj->n', V, Gi, V)
    mask = P > 1e-14
    x = np.pi * P[mask]
    g12 = 2 * (np.exp(-x) / np.sqrt(x) - np.sqrt(np.pi) * erfc(np.sqrt(x)))
    dual = np.sum(np.sqrt(x) * g12)
    pole = 2.0 if np.all(np.abs(a - np.round(a)) < 1e-15) else 0.0
    return np.pi ** 2 * (direct + (dual + pole) / np.sqrt(np.linalg.det(G)) - 0.5)


# ----------------------------------------------------------------------------- SM Casimir coefficient
AAA = (mp.mpf(1) / 2, mp.mpf(1) / 2, mp.mpf(1) / 2)
PER = (0, 0, 0)


def C_free_mp(G):
    """C_free(G) = (15 Z_2(G,0) - 45 Z_2(G,AAA))/pi^2  (SM, AAA fermions, static graviton)."""
    return (15 * ewald_Z(2, G, PER) - 45 * ewald_Z(2, G, AAA)) / mp.pi ** 2


def C_free_np(G):
    return (15 * ewald_Z2_np(G, (0, 0, 0)) - 45 * ewald_Z2_np(G, (0.5, 0.5, 0.5))) / np.pi ** 2


def mp2str(x, n=25):
    return mp.nstr(x, n)
