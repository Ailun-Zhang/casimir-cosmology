"""Extended-precision c-product Rayleigh quotient for Method B (HO basis, complex rotation).

H_theta is complex symmetric.  For an approximate eigenvector x = x0 + d of H (H x0 = E x0),
    R(x) = x^T H x / x^T x = E + d^T (H - E) d / x^T x,
so the error is quadratic in the eigenvector error d.  The eigenvector is computed in double
precision (ARPACK shift-invert, |d| ~ 1e-13 or better); R(x) is then evaluated with mpmath at
40 digits, with the matrix entries also generated in mpmath.  This gives the eigenvalue of the
truncated matrix to ~1e-25, far below the double-precision rounding floor (~1e-16) that
otherwise limits Im E.  Basis-truncation error is then measured by varying the basis.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import mpmath as mp
import ho_basis as hb


def _band_r(N, beta):
    """Exact band representation {offset: object array} of T_r, R2, R4 (2D m=0 HO basis)."""
    b = mp.mpf(beta)
    M = N + 2
    # X (size M): diag 2n+1, off -(n+1)
    Xd = [mp.mpf(2 * n + 1) for n in range(M)]
    Xo = [mp.mpf(-(n + 1)) for n in range(M - 1)]
    Xfull = {0: Xd, 1: Xo, -1: Xo}
    # X^2 band (offsets -2..2), truncated to N
    def Xel(i, j):
        if i == j:
            return Xd[i]
        if j == i + 1:
            return Xo[i]
        if i == j + 1:
            return Xo[j]
        return mp.mpf(0)
    X2 = {}
    for d in range(-2, 3):
        arr = []
        for i in range(N):
            j = i + d
            if j < 0 or j >= N:
                continue
            s = mp.mpf(0)
            for k in range(max(0, i - 1), min(M, i + 2)):
                s += Xel(i, k) * Xel(k, j)
            arr.append(s)
        X2[d] = np.array(arr, dtype=object)
    R4 = {d: X2[d] / b**2 for d in X2}
    R2 = {0: np.array(Xd[:N], dtype=object) / b, 1: np.array(Xo[:N - 1], dtype=object) / b,
          -1: np.array(Xo[:N - 1], dtype=object) / b}
    Td = np.array([b * (mp.mpf(n) + mp.mpf(1) / 2) for n in range(N)], dtype=object)
    To = np.array([b * mp.mpf(n + 1) / 2 for n in range(N - 1)], dtype=object)
    T = {0: Td, 1: To, -1: To}
    return T, R2, R4


def _band_s(N, gamma):
    g = mp.mpf(gamma)
    n = [2 * k + 1 for k in range(N)]
    diag = np.array([mp.mpf(2 * m + 1) for m in n], dtype=object)
    off = np.array([mp.sqrt(mp.mpf((m + 1) * (m + 2))) for m in n[:-1]], dtype=object)
    S2 = {0: diag / (2 * g), 1: off / (2 * g), -1: off / (2 * g)}
    T = {0: diag * g / 4, 1: -off * g / 4, -1: -off * g / 4}
    return T, S2


def _apply_left(B, X):
    """(B X) for band matrix B acting on rows of object 2D array X."""
    N = X.shape[0]
    Y = np.empty_like(X)
    Y[:] = mp.mpf(0)
    for d, arr in B.items():
        if d >= 0:
            # Y[i] += B[i, i+d] X[i+d], B[i,i+d] = arr[i], i = 0..N-1-d
            Y[:N - d] = Y[:N - d] + arr[:, None] * X[d:]
        else:
            dd = -d
            # Y[i] += B[i, i-dd] X[i-dd], i = dd..N-1, B[i,i-dd] = arr[i-dd]
            Y[dd:] = Y[dd:] + arr[:, None] * X[:N - dd]
    return Y


def _apply_right(X, B):
    """X B^T (B symmetric) acting on columns."""
    return _apply_left(B, X.T).T


def _bil(XA, XB_, A=None, B=None):
    """sum_{ik} XA_ik (A XB B^T)_ik  (A acts on r index, B on s index)."""
    Y = XB_
    if A is not None:
        Y = _apply_left(A, Y)
    if B is not None:
        Y = _apply_right(Y, B)
    return mp.fsum((XA * Y).ravel())


def rayleigh_mp(e, lam, Nr, beta, Ns, gamma, theta, phi, x, dps=40):
    mp.mp.dps = dps
    X = np.asarray(x, dtype=complex).reshape(Nr, Ns)
    Xr = np.vectorize(mp.mpf, otypes=[object])(X.real)
    Xi = np.vectorize(mp.mpf, otypes=[object])(X.imag)
    Tr, R2, R4 = _band_r(Nr, beta)
    Ts, S2 = _band_s(Ns, gamma)

    def q(A, B):
        rr = _bil(Xr, Xr, A, B)
        ii = _bil(Xi, Xi, A, B)
        ri = _bil(Xr, Xi, A, B)
        return mp.mpc(rr - ii, 2 * ri)

    th = mp.mpf(theta)
    ph = mp.mpf(phi)
    E_mp = mp.mpf(e)
    L_mp = mp.mpf(lam)
    num = (mp.expj(-2 * th) * q(Tr, None) + mp.expj(-2 * ph) * q(None, Ts)
           + E_mp**2 / 2 * mp.expj(2 * (th + ph)) * q(R2, S2)
           + L_mp / 4 * mp.expj(4 * th) * q(R4, None))
    den = q(None, None)
    return num / den


def resonance_mp(e, lam, Nr, beta, Ns, gamma, theta, phi=0.0, sigma=1.9, dps=40):
    """Return (E_double, E_mp) for the eigenvalue of H_theta closest to sigma."""
    H = hb.build_H(e, lam, Nr, beta, Ns, gamma, theta, phi)
    vals, vecs = hb.eig_near(H, sigma, k=3, return_vec=True)
    x = vecs[:, 0]
    # one step of inverse iteration with a shift at the computed eigenvalue (polish)
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    lu = sla.splu((H - vals[0] * sp.identity(H.shape[0], format="csc")).tocsc())
    for _ in range(2):
        x = lu.solve(x)
        x = x / np.sqrt(np.sum(x * x))
    E_mp = rayleigh_mp(e, lam, Nr, beta, Ns, gamma, theta, phi, x, dps=dps)
    return vals[0], E_mp


if __name__ == "__main__":
    import sys, time
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else -0.01003
    for (Nr, Ns, th) in [(60, 30, np.pi / 8), (80, 40, np.pi / 8), (80, 40, np.pi / 6)]:
        t = time.time()
        Ed, Em = resonance_mp(1.0, lam, Nr, 1.0, Ns, 1.0, th)
        print(Nr, Ns, round(th, 4), "double:", Ed, " mp:", mp.nstr(Em.real, 18), mp.nstr(Em.imag, 18),
              f"{time.time()-t:.1f}s", flush=True)
