# ewald_v4.py -- twisted Epstein sums Z_2(G, theta) on T^3 by an Ewald split, with analytic gradient.
# Written from scratch by agent shape-blind (R16 V4), 2026-10-15, from the v11 formulas
# (App. "Twisted image sums": split of int_0^inf t e^{-tQ} dt at t = pi, Poisson resummation of the small-t part).
#
#   Z_2(G,th) = sum'_n cos(2 pi th.n) Q_n^-2,  Q_n = n^T G n
#             = sum'_n c_n (1+pi Q)e^{-pi Q}/Q^2 - pi^2/2
#               + pi^2/sqrt(det G) [ sum_{P_m>0} g(P_m) + 2 delta_{th in Z^3} ],
#   g(P) = sqrt(pi P) Gamma(-1/2, pi P) = 2 e^{-pi P} - 2 pi sqrt(P) erfc(sqrt(pi P)),
#   P_m = (m-th)^T G^-1 (m-th).
#   dZ/dG (without the det-G term, which only adds a multiple of G^-1 and drops out of the
#   trace-free shape force):  real: sum' c_n f'(Q) n n^T,  f'(Q) = -e^{-pi Q}(2/Q^3 + 2pi/Q^2 + pi^2/Q)
#                             recip: pi^3/sqrt(detG) sum erfc(sqrt(pi P))/sqrt(P) k k^T,  k = G^-1 (m-th).
# Field content (v11 benchmark AAA SM + static graviton): Z_eff = 15 Z_2(G,0) - 45 Z_2(G,(1/2,1/2,1/2)).
import numpy as np
from scipy.special import erfc

PI = np.pi
SQPI = np.sqrt(np.pi)
RC = 3.0                    # every term with Q <= RC^2 (real) or P <= RC^2 (reciprocal) lies inside the box
NB_SM, NF_SM = 15.0, 45.0   # periodic bosonic weight, AAA Weyl fermions

_cache = {}


def _grid(N, kind):
    """kind 'r': integer box without 0 (+ parity sign), 'k': integer box without 0, 'h': half-integer box."""
    key = (kind, N[0], N[1], N[2])
    v = _cache.get(key)
    if v is not None:
        return v
    if kind == 'h':
        axes = [np.arange(-k, k, dtype=float) + 0.5 for k in N]
    else:
        axes = [np.arange(-k, k + 1, dtype=float) for k in N]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    if kind != 'h':
        g = g[np.any(g != 0.0, axis=1)]
    sgn = (1.0 - 2.0 * np.mod(g.sum(1), 2.0)) if kind == 'r' else None
    v = (np.ascontiguousarray(g), sgn)
    if len(_cache) < 3000:
        _cache[key] = v
    return v


def zeff(G, B=NB_SM, F=NF_SM, grad=True):
    """Return (Z, dZ/dG) for Z = B*Z_2(G,0) - F*Z_2(G,(1/2,1/2,1/2)).  dZ/dG omits the det-G term."""
    G = np.asarray(G, dtype=float)
    Gi = np.linalg.inv(G)
    Gi = 0.5 * (Gi + Gi.T)
    sq = 1.0 / np.sqrt(np.linalg.det(G))
    g = np.zeros((3, 3)) if grad else None
    # ---- real space, combined weights w_n = B - F (-1)^{n1+n2+n3}
    Nr = tuple(int(x) for x in np.maximum(1, np.ceil(RC * np.sqrt(np.diag(Gi)))))
    n, sgn = _grid(Nr, 'r')
    w = B - F * sgn
    Q = np.einsum('ij,ij->i', n @ G, n)
    iQ = 1.0 / Q
    e = np.exp(-PI * Q)
    Z = float(np.dot(w, (iQ + PI) * iQ * e))
    if grad:
        c = w * (-e * iQ * (2.0 * iQ * iQ + 2.0 * PI * iQ + PI * PI))
        g += (n * c[:, None]).T @ n
    Z += -0.5 * PI * PI * (B - F)
    # ---- reciprocal, periodic fields
    if B != 0.0:
        Nk = tuple(int(x) for x in np.maximum(1, np.ceil(RC * np.sqrt(np.diag(G)))))
        m, _ = _grid(Nk, 'k')
        k = m @ Gi
        P = np.einsum('ij,ij->i', k, m)
        sP = np.sqrt(P)
        ec = erfc(SQPI * sP)
        Z += B * PI * PI * sq * (float(np.sum(2.0 * np.exp(-PI * P) - 2.0 * PI * sP * ec)) + 2.0)
        if grad:
            g += (B * PI ** 3 * sq) * ((k * (ec / sP)[:, None]).T @ k)
    # ---- reciprocal, AAA (half-integer) fields
    if F != 0.0:
        Nh = tuple(int(x) for x in np.maximum(1, np.floor(RC * np.sqrt(np.diag(G)) + 0.5)))
        m, _ = _grid(Nh, 'h')
        k = m @ Gi
        P = np.einsum('ij,ij->i', k, m)
        sP = np.sqrt(P)
        ec = erfc(SQPI * sP)
        Z -= F * PI * PI * sq * float(np.sum(2.0 * np.exp(-PI * P) - 2.0 * PI * sP * ec))
        if grad:
            g -= (F * PI ** 3 * sq) * ((k * (ec / sP)[:, None]).T @ k)
    return Z, g


# orthonormal trace-free basis (v11 Sec. Setup): X_d, X_d2 (E doublet), X_12, X_13, X_23 (T2)
def shape_basis():
    X = np.zeros((5, 3, 3))
    X[0] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    X[1] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    for a, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        X[2 + a, i, j] = X[2 + a, j, i] = 1.0 / np.sqrt(2.0)
    return X


def expm_sym(Bm):
    b, O = np.linalg.eigh(Bm)
    return (O * np.exp(b)) @ O.T


def direct_sum(theta, N):
    """Brute-force sum'_{|n_i|<=N} cos(2 pi theta.n)/|n|^4 at the cube (check only)."""
    r = np.arange(-N, N + 1, dtype=float)
    n1, n2, n3 = np.meshgrid(r, r, r, indexing='ij')
    q = n1 * n1 + n2 * n2 + n3 * n3
    q[N, N, N] = 1.0
    c = np.cos(2 * PI * (theta[0] * n1 + theta[1] * n2 + theta[2] * n3))
    t = c / q ** 2
    t[N, N, N] = 0.0
    return float(t.sum())
