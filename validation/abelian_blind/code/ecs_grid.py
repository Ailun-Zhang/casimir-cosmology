"""Method A: smooth exterior complex scaling (ECS) in r on a 2D finite-difference grid,
plus the flux formula Gamma = J(R)/N(R) evaluated on the unscaled part of the grid.

Unknown: Phi(r, s) = s * psi(r, s)   (3D s-wave in s -> Dirichlet at s = 0, Phi odd in s)
psi kept as a function of r (2D m = 0): even in r, regular at r = 0; the operator
    -1/2 (d_r^2 + r^-1 d_r)  is discretised with central 2p-order stencils using mirror
    values psi(-r) = psi(r); at r = 0 the operator is 2 psi''(0).
Grid: r_i = i h_r (i = 0..Nr-1, includes r = 0), s_j = j h_s (j = 1..Ns).
H Phi = -1/2 [A(r) Phi_rr + B(r) Phi_r] - 1/2 Phi_ss + V(R(r), s) Phi
ECS contour: R(r) = r for r <= R0, R'(r) = 1 + (e^{i theta} - 1) sw((r - R0)/w), sw = C3 smoothstep.
    A = 1/R'^2,  B = -R''/R'^3 + 1/(R R')
Measure for norms: r dr ds (Phi = s psi).
Flux through r = R (unscaled): J = R * int ds Im(conj(Phi) dPhi/dr);  N(R) = int_0^R r dr int ds |Phi|^2.
For the Siegert state with E = eps - i Gamma/2:  J(R) = Gamma N(R) exactly (continuity equation).
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def fornberg(z, x, m):
    n = len(x) - 1
    c = np.zeros((n + 1, m + 1))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def central_weights(p):
    offs = np.arange(-p, p + 1)
    c = fornberg(0.0, offs.astype(float), 2)
    return offs, c[:, 1], c[:, 2]


def deriv_matrices_r(N, h, p):
    """D1, D2 on r_i = i h, i = 0..N-1, psi even about r = 0, zero beyond the far end."""
    offs, w1, w2 = central_weights(p)
    rows, cols, v1, v2 = [], [], [], []
    for i in range(N):
        for m, a1, a2 in zip(offs, w1, w2):
            j = i + m
            if j < 0:
                j = -j  # even mirror
            if j >= N:
                continue
            rows.append(i); cols.append(j); v1.append(a1 / h); v2.append(a2 / h**2)
    D1 = sp.csr_matrix((v1, (rows, cols)), shape=(N, N))
    D2 = sp.csr_matrix((v2, (rows, cols)), shape=(N, N))
    D1.sum_duplicates(); D2.sum_duplicates()
    return D1, D2


def deriv_matrix_s(N, h, p):
    """D2 on s_j = j h, j = 1..N, Phi odd about s = 0 (Phi(0)=0), zero beyond the far end."""
    offs, w1, w2 = central_weights(p)
    rows, cols, v2 = [], [], []
    for jj in range(N):
        j = jj + 1
        for m, a2 in zip(offs, w2):
            k = j + m
            sign = 1.0
            if k == 0:
                continue
            if k < 0:
                k = -k
                sign = -1.0
            if k > N:
                continue
            rows.append(jj); cols.append(k - 1); v2.append(sign * a2 / h**2)
    D2 = sp.csr_matrix((v2, (rows, cols)), shape=(N, N))
    D2.sum_duplicates()
    return D2


def contour(r, R0, w, theta, kind="erf", a=6.0):
    """Smooth ECS contour R(r), R'(r), R''(r).
    kind="c3" : C3 smoothstep switch on [R0, R0+w] (exactly real for r <= R0)
    kind="erf": C-infinity switch sw = (1+erf(a(t-1/2)))/2, t=(r-R0)/w; Im R ~ erfc tail below R0
    """
    from scipy.special import erf
    z = np.exp(1j * theta) - 1.0
    t = (r - R0) / w
    if kind == "c3":
        t = np.clip(t, 0.0, None)
        tc = np.minimum(t, 1.0)
        sw = np.where(t < 1, tc**4 * (35 - 84 * tc + 70 * tc**2 - 20 * tc**3), 1.0)
        dsw = np.where(t < 1, 140 * tc**3 - 420 * tc**4 + 420 * tc**5 - 140 * tc**6, 0.0) / w
        Sw = np.where(t < 1, 7 * tc**5 - 14 * tc**6 + 10 * tc**7 - 2.5 * tc**8, 0.5 + (t - 1.0))
    else:
        u = t - 0.5
        sw = 0.5 * (1 + erf(a * u))
        dsw = a / np.sqrt(np.pi) * np.exp(-(a * u) ** 2) / w
        Sw = 0.5 * (u + u * erf(a * u) + np.exp(-(a * u) ** 2) / (a * np.sqrt(np.pi)))
    R = r + z * w * Sw
    R1 = 1.0 + z * sw
    R2 = z * dsw
    return R, R1, R2


class ECSGrid:
    def __init__(self, e, lam, hr, Rmax, hs, Smax, R0, w, theta, p=4, mask_c=None, kind="erf", a=6.0):
        self.e, self.lam, self.p = e, lam, p
        self.Nr = int(round(Rmax / hr)) + 1
        self.hr = hr
        self.r = np.arange(self.Nr) * hr
        self.Ns = int(round(Smax / hs))
        self.hs = hs
        self.s = np.arange(1, self.Ns + 1) * hs
        self.R0, self.w, self.theta = R0, w, theta
        R, R1, R2 = contour(self.r, R0, w, theta, kind, a)
        self.R, self.R1, self.R2 = R, R1, R2
        D1, D2 = deriv_matrices_r(self.Nr, hr, p)
        self.D1r = D1
        A = 1.0 / R1**2
        B = np.zeros_like(R)
        B[1:] = -R2[1:] / R1[1:] ** 3 + 1.0 / (R[1:] * R1[1:])
        Lr = (sp.diags(A) @ D2 + sp.diags(B) @ D1).tolil()
        Lr[0, :] = 2.0 * D2[0, :]   # r = 0: (psi'' + psi'/r) -> 2 psi''(0)
        Lr = Lr.tocsr()
        D2s = deriv_matrix_s(self.Ns, hs, p)
        Ir = sp.identity(self.Nr, format="csr")
        Is = sp.identity(self.Ns, format="csr")
        RR, SS = np.meshgrid(R, self.s, indexing="ij")
        V = 0.5 * e**2 * RR**2 * SS**2 + 0.25 * lam * RR**4
        H = -0.5 * sp.kron(Lr, Is) - 0.5 * sp.kron(Ir, D2s) + sp.diags(V.ravel())
        H = H.tocsr().astype(complex)
        # optional mask: keep points where the wave function can be non-negligible
        rr, ss = np.meshgrid(self.r, self.s, indexing="ij")
        if mask_c is not None:
            c = mask_c
            keep = (ss <= np.sqrt(2 * c / (e * np.maximum(rr, 1.0)))) | \
                   (rr <= np.sqrt(2 * c / (e * np.maximum(ss, 1.0))))
        else:
            keep = np.ones_like(rr, dtype=bool)
        self.keep = keep.ravel()
        self.idx = np.nonzero(self.keep)[0]
        self.H = H[self.idx][:, self.idx].tocsc()
        self.n = self.H.shape[0]

    def full(self, x):
        X = np.zeros(self.Nr * self.Ns, dtype=complex)
        X[self.idx] = x
        return X.reshape(self.Nr, self.Ns)

    def solve(self, sigma, nit=6, x0=None, permc="COLAMD"):
        n = self.n
        M = (self.H - sigma * sp.identity(n, format="csc")).tocsc()
        lu = sla.splu(M, permc_spec=permc)
        if x0 is None:
            # start: Gaussian bump in the well
            X0 = np.exp(-(self.r[:, None] ** 2) / 2 - (self.s[None, :] ** 2) / 2) * self.s[None, :]
            x = X0.ravel()[self.idx].astype(complex)
        else:
            x = x0
        lam_hist = []
        for it in range(max(nit, 40)):
            y = lu.solve(x)
            # eigenvalue estimate: E = sigma + (x.x)/(x.y) (exact for converged x)
            mu = np.vdot(x, y) / np.vdot(x, x)
            lam_hist.append(sigma + 1.0 / mu)
            x = y / np.linalg.norm(y)
            if it >= nit and abs(lam_hist[-1] - lam_hist[-2]) < 1e-15 * abs(lam_hist[-1]):
                break
        E = lam_hist[-1]
        # Rayleigh quotient with the converged vector
        E = np.vdot(x, self.H @ x) / np.vdot(x, x)
        # residual check
        res = np.linalg.norm(self.H @ x - E * x)
        return E, x, lam_hist, res

    def flux_gamma(self, x, Rf_list):
        """Gamma_flux(R) = J(R)/N(R) at grid points R in the unscaled region."""
        X = self.full(x)
        hs, hr = self.hs, self.hr
        p = self.p
        offs, w1, w2 = central_weights(p)
        # f(r) = int ds |Phi|^2 (trapezoid on [0,S] for even integrand: spectral)
        f = hs * np.sum(np.abs(X) ** 2, axis=1)
        g = self.r * f
        out = []
        for Rf in Rf_list:
            K = int(round(Rf / hr))
            if self.r[K] > self.R0 - 1e-12:
                raise ValueError("flux radius must be in the unscaled region")
            # dPhi/dr at r_K (central, interior point)
            dX = sum(a * X[K + m] for m, a in zip(offs, w1)) / hr
            J = self.r[K] * hs * np.sum(np.imag(np.conj(X[K]) * dX))
            # N(R) = int_0^{r_K} g dr: trapezoid + Euler-Maclaurin end corrections
            T = hr * (0.5 * g[0] + np.sum(g[1:K]) + 0.5 * g[K])
            # derivatives of g at r = 0 from f (even): g'(0)=f(0), g'''(0)=3 f''(0), g5(0)=5 f''''(0)
            fe = np.concatenate([f[p:0:-1], f[: 2 * p + 1]])  # mirrored, centre at index p
            c4 = fornberg(0.0, np.arange(-p, p + 1).astype(float), 4)
            f0 = f[0]
            f2 = np.dot(c4[:, 2], fe[: 2 * p + 1]) / hr**2
            f4 = np.dot(c4[:, 4], fe[: 2 * p + 1]) / hr**4
            gp0, g3p0, g5p0 = f0, 3 * f2, 5 * f4
            # derivatives of g at r_K (central)
            cK = fornberg(0.0, np.arange(-p, p + 1).astype(float), 5)
            seg = g[K - p: K + p + 1]
            gpK = np.dot(cK[:, 1], seg) / hr
            g3K = np.dot(cK[:, 3], seg) / hr**3
            g5K = np.dot(cK[:, 5], seg) / hr**5
            N = (T - hr**2 / 12 * (gpK - gp0) + hr**4 / 720 * (g3K - g3p0)
                 - hr**6 / 30240 * (g5K - g5p0))
            out.append((self.r[K], J, N, J / N))
        return out


def run_case(e, lam, hr, Rmax, hs, Smax, R0, w, theta, Rf_list, sigma=1.9, p=4, mask_c=None,
             nit=5, verbose=True, kind="erf", a=6.0):
    import time
    t0 = time.time()
    G = ECSGrid(e, lam, hr, Rmax, hs, Smax, R0, w, theta, p=p, mask_c=mask_c, kind=kind, a=a)
    t1 = time.time()
    E, x, hist, res = G.solve(sigma, nit=nit)
    t2 = time.time()
    fl = G.flux_gamma(x, Rf_list)
    if verbose:
        print(f"  n={G.n} build {t1-t0:.1f}s solve {t2-t1:.1f}s  E={E.real:.12f} {E.imag:+.6e}  res={res:.1e}",
              flush=True)
        for (R, J, N, Gm) in fl:
            print(f"    R={R:.3f}  J={J:.6e}  N={N:.10f}  Gamma_flux={Gm:.8e}", flush=True)
    return dict(E=E, hist=hist, res=res, flux=fl, n=G.n, grid=G)


if __name__ == "__main__":
    import sys
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else -0.11174
    run_case(1.0, lam, 0.05, 7.0, 0.05, 10.0, R0=4.5, w=1.0, theta=0.5, Rf_list=[3.6, 4.0, 4.4])
