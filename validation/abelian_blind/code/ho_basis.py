"""Method B: product harmonic-oscillator basis with complex rotation.

H = -1/2 Delta_r(2D, m=0) - 1/2 Delta_s(3D, l=0) + e^2/2 r^2 s^2 + lam/4 r^4

r-basis : 2D isotropic HO functions with m = 0,
          phi_n(r) = sqrt(2 beta) L_n(beta r^2) exp(-beta r^2/2),  <phi_n|phi_m> = delta  (measure r dr)
s-basis : 3D isotropic HO s-states (frequency gamma); in v = s*psi form these are the
          odd 1D HO functions h_{2k+1}, measure ds.
All operators (T_r, r^2, r^4, T_s, s^2) are exact sparse (band) matrices.

Complex rotation: r -> r exp(i theta)  (dissociation coordinate), optionally s -> s exp(i phi).
  H_theta = e^{-2i theta} T_r + e^{-2i phi} T_s + e^2/2 e^{2i(theta+phi)} r^2 s^2 + lam/4 e^{4 i theta} r^4
The resonance is a discrete eigenvalue E = eps - i Gamma/2 of the complex-symmetric H_theta.

Complex absorbing potential (CAP) variant: theta = phi = 0, add -i eta W(r) with
  W(r) = (r - rc)^nW for r > rc, matrix elements by Gauss-Laguerre quadrature in x = beta r^2.
  For a CAP the interior problem stays real, Gamma = 2 eta <W> is a positive expectation value.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")  # conda MKL threading layer crashes in SuperLU here
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def r_matrices(N, beta):
    """T_r, r^2, r^4 in the 2D m=0 HO basis (N functions), exact."""
    M = N + 2
    n = np.arange(M, dtype=float)
    off = -(n[:-1] + 1.0)
    X = sp.diags([off, 2 * n + 1.0, off], [-1, 0, 1], format="csr")  # matrix of x = beta r^2
    X2 = (X @ X).tocsr()[:N, :N]
    X1 = X[:N, :N]
    nn = np.arange(N, dtype=float)
    offT = 0.5 * (nn[:-1] + 1.0)
    T = beta * sp.diags([offT, nn + 0.5, offT], [-1, 0, 1], format="csr")
    R2 = X1 / beta
    R4 = X2 / beta**2
    return T.tocsr(), R2.tocsr(), R4.tocsr()


def s_matrices(N, gamma):
    """T_s, s^2 in the odd 1D HO basis n = 2k+1, k = 0..N-1 (3D s-wave), exact."""
    k = np.arange(N, dtype=float)
    n = 2 * k + 1
    diag = (2 * n + 1.0)
    off = np.sqrt((n[:-1] + 1.0) * (n[:-1] + 2.0))
    S2 = sp.diags([off, diag, off], [-1, 0, 1], format="csr") / (2 * gamma)
    T = (gamma / 4.0) * sp.diags([-off, diag, -off], [-1, 0, 1], format="csr")
    return T.tocsr(), S2.tocsr()


def r_cap_matrix(N, beta, rc, nW=4, nquad=None):
    """Matrix of W(r) = (r-rc)^nW Theta(r-rc) in the r basis via Gauss-Laguerre quadrature."""
    from scipy.special import roots_laguerre, eval_laguerre
    if nquad is None:
        nquad = max(4 * N, 400)
    x, w = roots_laguerre(nquad)
    r = np.sqrt(x / beta)
    W = np.where(r > rc, (r - rc) ** nW, 0.0)
    # phi_n phi_m r dr = 2 beta L_n L_m e^{-x} dx/(2 beta)
    L = np.empty((N, nquad))
    L[0] = 1.0
    if N > 1:
        L[1] = 1.0 - x
    for m in range(1, N - 1):
        L[m + 1] = ((2 * m + 1 - x) * L[m] - m * L[m - 1]) / (m + 1)
    A = L * np.sqrt(w)[None, :]
    return (A * W[None, :]) @ A.T


def build_H(e, lam, Nr, beta, Ns, gamma, theta=0.0, phi=0.0, cap=None):
    Tr, R2, R4 = r_matrices(Nr, beta)
    Ts, S2 = s_matrices(Ns, gamma)
    Ir = sp.identity(Nr, format="csr")
    Is = sp.identity(Ns, format="csr")
    c = np.exp
    H = (c(-2j * theta) * sp.kron(Tr, Is) + c(-2j * phi) * sp.kron(Ir, Ts)
         + 0.5 * e**2 * c(2j * (theta + phi)) * sp.kron(R2, S2)
         + 0.25 * lam * c(4j * theta) * sp.kron(R4, Is))
    H = H.astype(complex)
    if cap is not None:
        eta, rc, nW = cap
        W = r_cap_matrix(Nr, beta, rc, nW)
        H = H - 1j * eta * sp.kron(sp.csr_matrix(W), Is)
    return H.tocsc()


def eig_near(H, sigma, k=6, return_vec=False, tol=0.0):
    vals, vecs = sla.eigs(H, k=k, sigma=sigma, which="LM", tol=tol)
    idx = np.argsort(np.abs(vals - sigma))
    vals = vals[idx]
    vecs = vecs[:, idx]
    if return_vec:
        return vals, vecs
    return vals


def resonance(e, lam, Nr, beta, Ns, gamma, theta, phi=0.0, sigma=2.0, k=4):
    H = build_H(e, lam, Nr, beta, Ns, gamma, theta, phi)
    vals = eig_near(H, sigma, k=k)
    return vals


if __name__ == "__main__":
    import sys, time
    e = 1.0
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else -1.0
    for (Nr, Ns) in [(60, 30), (100, 40), (140, 50)]:
        t = time.time()
        v = resonance(e, lam, Nr, 1.0, Ns, 1.0, theta=np.pi / 8, sigma=2.0, k=6)
        print(Nr, Ns, np.round(v, 10), f"{time.time()-t:.1f}s")
