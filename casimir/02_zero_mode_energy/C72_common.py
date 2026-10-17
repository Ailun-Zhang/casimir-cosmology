# PROVENANCE: 04_calculations/code/20261011_claude_followup/C72/C72_common.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Conventions and constant-mode Hamiltonians of the SM gauge/Higgs zero modes on T^3 (D54): SU(2) x U(1)_Y x Higgs block, SU(3) block, Gaussian trial energies, one-loop couplings at mu = 1/L_b. Energies in units of 1/L. Paper: PRD Sec. IV C (sec:pa-zeromodes, Eq. eq:pa-H0) and App. app:zeromodes.

C72 common definitions: conventions, constant-mode potentials, Gaussian trial energies, one-loop couplings.

Conventions (stated once, used everywhere in C72):
  * Torus coordinates x^i in [0,1), metric h_ij = L^2 G_ij, det G = 1, volume L^3. Mostly-plus signature.
  * Geometric (coupling-absorbed) connections: D = d - i (A^a T_a + Y B), tr(T_a T_b) = delta_ab/2,
    [T_a, T_b] = i f_abc T_c (SU(2): f = epsilon). Gauge action -(1/4g^2) int sqrt(-g) F^a F^a.
    Constant modes: A^a_i = c^a_i (SU(2)), B_i = u_i (U(1)_Y), with coordinate components (period-1 coordinates),
    i.e. c^a_i / L in Cartesian components of the cube.
  * Higgs doublet H = Phi / L, Phi = (phi0 + i phi1, phi2 + i phi3)/sqrt2, Y = 1/2, potential lambda (H^dag H)^2,
    no mass term. H^dag H = phi^2 / (2 L^2).
  * Temporal gauge; Gauss law imposed on states (constant gauge transformations only; local domain R^n).
  Result (derivation in D54): L*H = (g2^2/2) G_ij p_c,i.p_c,j + (gY^2/2) G_ij p_u,i p_u,j + (1/2) p_phi^2
        + (1/4 g2^2) G^ik G^jl F^a_ij F^a_kl + G^ij Phi^dag (c_i.T + Y u_i)(c_j.T + Y u_j) Phi + (lambda/4) (phi^2)^2
        (+ the analogous SU(3) block, which does not couple to the colour-singlet Higgs).
  Unit-mass coordinates used by the DMC: c = g2 x, u = gY w (so all kinetic terms are p^2/2).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
from pathlib import Path

import numpy as np

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
RES = WP.wdir("fu/C72")

# ------------------------------------------------------------------ project / input numbers
C_FREE = 42.7400221506            # SM (AAA) + static graviton, cube
GAMMA = 1.6750738213              # Z2(1)/pi^2
NC = C_FREE / GAMMA               # 25.5153...
KAPPA_E, KAPPA_T2 = 13.528581, -6.352387
H_OVER_MP = 6.116e-6              # paper Sec. IV C (A_s calibrated, instant reheating)
MP_GEV = 2.435e18                 # reduced Planck mass
MZ = 91.1876
ALPHA_S_MZ, INV_ALPHA_EM_MZ, SIN2W = 0.1184, 127.95, 0.23122
B1_GUT, B1_SM, B2, B3 = 41.0 / 10.0, 41.0 / 6.0, -19.0 / 6.0, -7.0
EPS_SU2_VB = 4.116720             # van Baal 1988 Table 2 (input, literature check by another agent)
EPS_SU3_PAVEL = 12.5868           # Pavel 2021 Table 2 (variational, input)
EPS_QUARTIC4 = 2.1407             # paper App.: 4D quartic H = p^2/2 + (q^2)^2/4 (certified there)


# ------------------------------------------------------------------ Lie algebra
def su_basis(n):
    """Hermitian traceless generators with tr(T_a T_b) = delta_ab / 2 (generalised Gell-Mann / 2)."""
    gens = []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.zeros((n, n), complex); m[i, j] = m[j, i] = 0.5; gens.append(m)
            m = np.zeros((n, n), complex); m[i, j] = -0.5j; m[j, i] = 0.5j; gens.append(m)
    for k in range(1, n):
        m = np.zeros((n, n), complex)
        m[:k, :k] = np.eye(k); m[k, k] = -k
        gens.append(m / np.sqrt(2 * k * (k + 1)))
    return gens


def structure_constants(n):
    """f_abc = -2 i tr([T_a, T_b] T_c), real and totally antisymmetric."""
    T = su_basis(n)
    d = len(T)
    f = np.zeros((d, d, d))
    for a in range(d):
        for b in range(d):
            cm = T[a] @ T[b] - T[b] @ T[a]
            for c in range(d):
                f[a, b, c] = (-2j * np.trace(cm @ T[c])).real
    return f


PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex)]


# ------------------------------------------------------------------ vectorised potentials (unit-mass coordinates)
def v_sun(x, f):
    """(1/4) F^a_ij F^a_ij with F^a_ij = f_abc x^b_i x^c_j; x shape (W, d, 3) [colour, space]."""
    d = f.shape[0]
    fm = f.transpose(1, 0, 2).reshape(d, d * d)       # fm[b, a*d+c] = f[a,b,c]
    v = np.zeros(x.shape[0])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (x[:, :, i] @ fm).reshape(-1, d, d)       # m[w,a,c] = sum_b f_abc x^b_i
        F = np.einsum("wac,wc->wa", m, x[:, :, j])
        v += (F * F).sum(1)
    return 0.5 * v                                     # (1/4) * 2 * sum_{i<j}


def v_su2(x):
    """Same for SU(2) (f = epsilon) via cross products; x shape (W, 3, 3) [colour, space]."""
    v = np.zeros(x.shape[0])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        F = np.cross(x[:, :, i], x[:, :, j])
        v += (F * F).sum(1)
    return 0.5 * v


def higgs_bilinears(phi):
    """phi (W,4) -> Phi^dag Phi and n = Phi^dag sigma Phi for Phi = (phi0 + i phi1, phi2 + i phi3)/sqrt2."""
    p0, p1, p2, p3 = phi[:, 0], phi[:, 1], phi[:, 2], phi[:, 3]
    norm = 0.5 * (phi * phi).sum(1)
    n = np.stack([p0 * p2 + p1 * p3, p0 * p3 - p1 * p2, 0.5 * (p0 ** 2 + p1 ** 2 - p2 ** 2 - p3 ** 2)], axis=1)
    return norm, n


def v_higgs_gradient(x, w, phi, g2, gY, Y=0.5):
    """sum_i Phi^dag (g2 x_i.T + gY Y w_i)^2 Phi, using (A.sigma/2 + b)^2 = |A|^2/4 + b^2 + b A.sigma.
    x (W,3,3) [colour, space] unit-mass SU(2) coords, w (W,3) unit-mass U(1) coords, phi (W,4)."""
    norm, n = higgs_bilinears(phi)
    A2 = g2 ** 2 * (x * x).sum(1)                      # (W,3): |A_i|^2 per space index
    b = gY * Y * w                                     # (W,3)
    An = g2 * np.einsum("wai,wa->wi", x, n)            # A_i . n
    return ((A2 / 4 + b * b) * norm[:, None] + b * An).sum(1)


def v_higgs_gradient_matrix(x, w, phi, g2, gY, Y=0.5):
    """Direct matrix evaluation (no Pauli identity), for verification; loops over walkers."""
    out = np.empty(x.shape[0])
    for k in range(x.shape[0]):
        Phi = np.array([phi[k, 0] + 1j * phi[k, 1], phi[k, 2] + 1j * phi[k, 3]]) / np.sqrt(2)
        s = 0.0
        for i in range(3):
            M = sum(g2 * x[k, a, i] * PAULI[a] / 2 for a in range(3)) + gY * Y * w[k, i] * np.eye(2)
            v = M @ Phi
            s += (np.conj(v) @ v).real
        out[k] = s
    return out


def v_ew(y, g2, gY, lam):
    """Full electroweak-Higgs constant-mode potential in unit-mass coordinates y = (x[9], w[3], phi[4]),
    x stored colour-major (a*3+i)."""
    x = y[:, :9].reshape(-1, 3, 3)
    w = y[:, 9:12]
    phi = y[:, 12:16]
    r2 = (phi * phi).sum(1)
    return g2 ** 2 * v_su2(x) + v_higgs_gradient(x, w, phi, g2, gY) + 0.25 * lam * r2 * r2


# ------------------------------------------------------------------ Gaussian trial energies (derived in D54)
def e_gauss_sun(alpha, N):
    """<(1/2)p^2 + (1/4)F^2> for psi = exp(-alpha x^2/2), x in R^{3(N^2-1)}."""
    return 3 * (N * N - 1) * alpha / 4 + 3 * N * (N * N - 1) / (8 * alpha ** 2)


def e_gauss_ew(a, b, z, g2, gY, lam):
    """L<H> for psi = exp[-(a c^2 + b u^2 + z phi^2)/2] in the ORIGINAL coordinates (c, u, phi)."""
    return (9 * g2 ** 2 * a / 4 + 3 * gY ** 2 * b / 4 + z + 9 / (4 * g2 ** 2 * a ** 2)
            + 9 / (8 * a * z) + 3 / (8 * b * z) + 3 * lam / (2 * z ** 2))


def e_gauss_ew_grad_log(s, g2, gY, lam):
    """Gradient of e_gauss_ew w.r.t. s = (ln a, ln b, ln z)."""
    a, b, z = np.exp(s)
    t1, t2, t3 = 9 * g2 ** 2 * a / 4, 3 * gY ** 2 * b / 4, z
    t4, t5, t6, t7 = 9 / (4 * g2 ** 2 * a ** 2), 9 / (8 * a * z), 3 / (8 * b * z), 3 * lam / (2 * z ** 2)
    return np.array([t1 - 2 * t4 - t5, t2 - t6, t3 - t5 - t6 - 2 * t7])


def minimize_ew(g2, gY, lam, n_starts=12, seed=1):
    """Global minimum of the (log-convex) coupled Gaussian energy; several random starts, BFGS + Newton polish."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    f = lambda s: e_gauss_ew(*np.exp(s), g2, gY, lam)
    jac = lambda s: e_gauss_ew_grad_log(s, g2, gY, lam)
    runs = []
    for k in range(n_starts):
        s0 = rng.uniform(-3, 3, size=3)
        r = minimize(f, s0, jac=jac, method="BFGS", options=dict(gtol=1e-13, maxiter=10000))
        runs.append((float(r.fun), r.x))
    runs.sort(key=lambda t: t[0])
    s = runs[0][1].copy()
    for _ in range(30):                                    # Newton polish with a finite-difference Hessian
        g = jac(s)
        Hs = np.array([(jac(s + 1e-6 * e) - jac(s - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
        Hs = 0.5 * (Hs + Hs.T)
        s = s - np.linalg.solve(Hs, g)
    g = jac(s)
    Hs = np.array([(jac(s + 1e-6 * e) - jac(s - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
    a, b, z = np.exp(s)
    return dict(E=float(f(s)), a=float(a), b=float(b), z=float(z), grad_max=float(np.abs(g).max()),
                hess_log_eigs=[float(v) for v in np.sort(np.linalg.eigvalsh(0.5 * (Hs + Hs.T)))[::-1]],
                spread_over_starts=float(runs[-1][0] - runs[0][0]),
                b_analytic=float(1 / np.sqrt(2 * gY ** 2 * z)))


# ------------------------------------------------------------------ background scale and one-loop couplings
def mu_bounce_gev(C):
    """1/L_b in GeV for 3 M_P^2 h^2 = C / L_b^4 with h = H_OVER_MP * M_P."""
    Lb_times_mp = (C / (3 * H_OVER_MP ** 2)) ** 0.25
    return MP_GEV / Lb_times_mp


def couplings_one_loop(mu):
    """One-loop SM running from MZ: 1/alpha_i(mu) = 1/alpha_i(MZ) - b_i/(2 pi) ln(mu/MZ).
    gY is the SM-normalised hypercharge coupling (D = d - i gY Y B, Y_H = 1/2); g1 = sqrt(5/3) gY (GUT)."""
    t = np.log(mu / MZ)
    a_em = 1 / INV_ALPHA_EM_MZ
    a2 = a_em / SIN2W
    aY = a_em / (1 - SIN2W)
    a1 = 5 / 3 * aY
    inv_a1 = 1 / a1 - B1_GUT / (2 * np.pi) * t
    inv_aY = 1 / aY - B1_SM / (2 * np.pi) * t
    inv_a2 = 1 / a2 - B2 / (2 * np.pi) * t
    inv_a3 = 1 / ALPHA_S_MZ - B3 / (2 * np.pi) * t
    g = lambda inv: float(np.sqrt(4 * np.pi / inv))
    return dict(mu_GeV=float(mu), ln_mu_over_MZ=float(t), g1_GUT=g(inv_a1), gY_from_GUT=float(np.sqrt(3 / 5) * g(inv_a1)),
                gY_SM=g(inv_aY), g2=g(inv_a2), g3=g(inv_a3), inv_alpha1_GUT=float(inv_a1), inv_alphaY=float(inv_aY),
                inv_alpha2=float(inv_a2), inv_alpha3=float(inv_a3),
                at_MZ=dict(g2=g(1 / a2), gY=g(1 / aY), g3=g(1 / ALPHA_S_MZ)))
