# twa1d.py -- truncated-Wigner (TWA) error estimate: exact 1D quantum vs TWA on the X_D2 (slab/rod) invariant line
# (T19, unverified-check, 2026-10-14).  Fixed leading cosh background, xi = 0.30, delta from v11 calibration.
# H = (CN/2) e^{-3 lr} Pi^2 - (vb/(CN Z0)) e^{-lr} [Z(beta) - Z(0)],  [beta,Pi] = i hb,  hb = 2 delta.
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
m = D.Model('lead')
CN, VB, Z0 = D.CN, 3.0, m.Z0
d_plat = 1.565e-4 / (1 - D.R_ZM) ** 0.75
DELTA = d_plat * (3.0 / D.vpot(D.PHIB)) ** 0.75
HB = 2 * DELTA
X2 = D.XB[1]

bg = np.linspace(-2.6, 2.6, 26001)
t0c = time.time()
Zg = np.array([D.zeff(D.expm_sym(2 * b * X2), False) for b in bg])
Zg -= D.zeff(np.eye(3), False)
spZ = CubicSpline(bg, Zg)
spZp = spZ.derivative()
kap_line = 3 * spZ(0.0, 2) / Z0
CU = VB / (CN * Z0)


def lr(t):
    return 0.5 * D.logcosh(2 * t)


def quantum(t0, sb, L, N, cap0, t1, dt):
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    psi = np.exp(-x * x / (4 * sb)).astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    Zx = spZ(np.clip(x, -2.6, 2.6))
    capw = np.where(np.abs(x) > cap0, ((np.abs(x) - cap0) / (L - cap0)) ** 2, 0.0) * 2000.0
    t = t0
    nsteps = int(round((t1 - t0) / dt))
    for _ in range(nsteps):
        e1 = math.exp(-lr(t))
        psi *= np.exp(1j * CU * e1 * Zx * dt / (2 * HB) - capw * dt / 2)
        tm = t + dt / 2
        kin = (CN / 2) * math.exp(-3 * lr(tm)) * HB * k * k * dt
        psi = np.fft.ifft(np.exp(-1j * kin) * np.fft.fft(psi))
        t += dt
        e1 = math.exp(-lr(t))
        psi *= np.exp(1j * CU * e1 * Zx * dt / (2 * HB) - capw * dt / 2)
    rho = np.abs(psi) ** 2 * dx
    pk = np.abs(np.fft.fft(psi)) ** 2
    pk /= pk.sum()
    return x, rho, float(np.max(np.abs(k)) * HB), float(pk[np.abs(k) * HB > 0.8 * np.max(np.abs(k)) * HB].sum())


def twa(t0, beta, Pi, t1, dt, cap0):
    b, p = beta.copy(), Pi.copy()
    dead = np.zeros(len(b), bool)
    t = t0
    nsteps = int(round((t1 - t0) / dt))

    def f(t, b, p):
        return CN * math.exp(-3 * lr(t)) * p, CU * math.exp(-lr(t)) * spZp(np.clip(b, -2.6, 2.6))
    for _ in range(nsteps):
        k1b, k1p = f(t, b, p)
        k2b, k2p = f(t + dt / 2, b + dt / 2 * k1b, p + dt / 2 * k1p)
        k3b, k3p = f(t + dt / 2, b + dt / 2 * k2b, p + dt / 2 * k2p)
        k4b, k4p = f(t + dt, b + dt * k3b, p + dt * k3p)
        b = b + dt / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        t += dt
        newd = np.abs(b) > cap0
        dead |= newd
        b = np.where(dead, np.sign(b) * 2.0, b)
        p = np.where(dead, 0.0, p)
    return b, dead


def sopt_of(t0):
    M = m.lin_transfer(kap_line, t0)
    return abs(M[0, 1] / M[0, 0]) * DELTA, M


out = {'delta': DELTA, 'hbar_eff': HB, 'kappa_line': kap_line, 'kappa_E_ref': 19.32374589, 'Zgrid_cpu_s': time.time() - t0c}
bands = [0.2, 0.5, 0.8, 1.1]
# --- case A: natural turning-point state; Moyal correction to the outgoing variance
snat = 3 * DELTA / (2 * D.S2 * math.sqrt(kap_line))
M0 = m.lin_transfer(kap_line, 0.0)
var_lin = M0[0, 0] ** 2 * snat + M0[0, 1] ** 2 * DELTA ** 2 / snat
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60)
B0, P0 = np.meshgrid(gh_x * math.sqrt(snat), gh_x * DELTA / math.sqrt(snat), indexing='ij')
Wq = np.outer(gh_w, gh_w) / (2 * math.pi)
bT, deadA = twa(0.0, B0.ravel(), P0.ravel(), 6.0, 1e-3, 5.0)
var_twa = float(np.sum(Wq.ravel() * bT ** 2))
caseA = {'s_nat': snat, 'var_lin': var_lin, 'var_twa_GH60': var_twa}
for (L, N, dt) in ((1.0, 2 ** 14, 1e-3), (1.0, 2 ** 15, 5e-4)):
    tq = time.time()
    x, rho, pmax, ptail = quantum(0.0, snat, L, N, 0.85, 6.0, dt)
    vq = float(np.sum(rho * x * x) / np.sum(rho))
    caseA['quantum_L%g_N%d_dt%g' % (L, N, dt)] = {'var': vq, 'norm': float(rho.sum()), 'ratio_q_over_twa': vq / var_twa,
                                                  'Pi_grid_max': pmax, 'p_tail_k': ptail, 'cpu_s': time.time() - tq}
caseA['twa_over_lin'] = var_twa / var_lin
out['A_natural_t0'] = caseA
print(json.dumps(out, indent=1), flush=True)
# --- cases B/C: R1 real Gaussians prepared before the turn (pure states); absorbed fraction and distribution
rng = np.random.default_rng(2026101931)
for t0 in (-0.5, -1.093):
    so, M = sopt_of(t0)
    Ns = 100000
    z = rng.standard_normal((Ns, 2))
    bI, pI = z[:, 0] * math.sqrt(so), z[:, 1] * DELTA / math.sqrt(so)
    tt = time.time()
    bT, dead = twa(t0, bI, pI, 6.0, 1e-3, 1.2)
    case = {'s_opt': so, 'sigma_lin_out': math.sqrt(2 * abs(M[0, 0] * M[0, 1]) * DELTA), 'twa_N': Ns,
            'twa_absorbed': float(dead.mean()), 'twa_absorbed_se': float(math.sqrt(dead.mean() * (1 - dead.mean()) / Ns)),
            'twa_P_in': {str(b): float(np.mean((~dead) & (np.abs(bT) < b))) for b in bands}, 'twa_cpu_s': time.time() - tt}
    for (L, N, dt) in ((1.8, 2 ** 16, 5e-4), (1.8, 2 ** 17, 2.5e-4)):
        tq = time.time()
        x, rho, pmax, ptail = quantum(t0, so, L, N, 1.2, 6.0, dt)
        case['quantum_N%d_dt%g' % (N, dt)] = {'absorbed': float(1 - rho.sum()),
                                              'P_in': {str(b): float(rho[np.abs(x) < b].sum()) for b in bands},
                                              'Pi_grid_max': pmax, 'p_tail_k': ptail, 'cpu_s': time.time() - tq}
    out['R1_%g' % t0] = case
    print(json.dumps(case, indent=1), flush=True)
with open(os.path.join(RES, 'D114_twa1d_own.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('done', flush=True)
