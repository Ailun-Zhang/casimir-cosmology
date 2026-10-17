# d114_own.py -- independent re-implementation for T19 (agent unverified-check), 2026-10-14.
# Written from the model definition only (v11 Sec. VI/VII + App. shape/SM/nonlinear, D114 sec.1 method text).
# No producer code was read.  Units h = L_b = 1, M_P = 1 for the inflaton.
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
from scipy.special import erfc
from scipy.integrate import solve_ivp

PI = math.pi
S2 = math.sqrt(2.0)
S6 = math.sqrt(6.0)
XB = np.zeros((5, 3, 3))
XB[0] = np.diag([1.0, -1.0, 0.0]) / S2      # X_D  (E; 'mid')
XB[1] = np.diag([1.0, 1.0, -2.0]) / S6      # X_D2 (E; +: slab = one short side, -: rod)
for _a, (_i, _j) in enumerate([(0, 1), (0, 2), (1, 2)]):
    XB[2 + _a, _i, _j] = XB[2 + _a, _j, _i] = 1.0 / S2   # T2
_I0, _I1 = np.triu_indices(3)
EYE = np.eye(3)
CN = 3.0 / (2.0 * S2)          # Y = CN * G^1/2 (Pi.X) G^1/2 ;  Pi = (2 sqrt2/3) L^3 beta_dot
R_ZM = 0.299897                # zero-mode fraction (xi = 0.30)
PHIB = 5.587                   # reference plateau history (v11 App. spectra)
AINF = math.sqrt(2.0 / 3.0)


def pack(M):
    return M[_I0, _I1].copy()


def unpack(v):
    M = np.empty((3, 3))
    M[_I0, _I1] = v
    M[_I1, _I0] = v
    return M


def expm_sym(A):
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T


def sqrtm_spd(G):
    w, V = np.linalg.eigh(G)
    return (V * np.sqrt(w)) @ V.T


def logm_spd(G):
    w, V = np.linalg.eigh(G)
    return (V * np.log(w)) @ V.T


def vpot(phi):
    e = math.exp(-AINF * phi)
    return 3.0 * (1.0 - e) ** 2


def dvpot(phi):
    e = math.exp(-AINF * phi)
    return 6.0 * AINF * e * (1.0 - e)


def logcosh(x):
    ax = abs(x)
    return ax + math.log1p(math.exp(-2.0 * ax)) - math.log(2.0)


# ---------------------------------------------------------------- Ewald sums
QMAX = 12.5
_cint, _chalf = {}, {}


def _ibox(b):
    key = (int(b[0]), int(b[1]), int(b[2]))
    if (2 * key[0] + 1) * (2 * key[1] + 1) * (2 * key[2] + 1) > 3_000_000:
        raise RuntimeError('ewald_box_overflow')
    g = _cint.get(key)
    if g is None:
        r = [np.arange(-k, k + 1, dtype=float) for k in key]
        g = np.stack(np.meshgrid(*r, indexing='ij'), -1).reshape(-1, 3)
        if len(_cint) > 600:
            _cint.clear()
        _cint[key] = g
    return g


def _hbox(b):
    key = (int(b[0]), int(b[1]), int(b[2]))
    g = _chalf.get(key)
    if g is None:
        r = [np.arange(-k, k, dtype=float) + 0.5 for k in key]
        if min(key) <= 0:
            g = np.zeros((0, 3))
        else:
            g = np.stack(np.meshgrid(*r, indexing='ij'), -1).reshape(-1, 3)
        if len(_chalf) > 600:
            _chalf.clear()
        _chalf[key] = g
    return g


def fp_enum(A, Q, s):
    """Fincke-Pohst enumeration (vectorized): all k = m + s*(1,1,1), m integer, with k^T A k <= Q.
    The coordinate with the largest (A^-1)_ii is enumerated outermost, so needle/pancake ellipsoids stay cheap."""
    dinv = np.diag(np.linalg.inv(A))
    p = np.argsort(dinv)
    Ap = A[np.ix_(p, p)]
    R = np.linalg.cholesky(Ap).T
    r00, r01, r02, r11, r12, r22 = R[0, 0], R[0, 1], R[0, 2], R[1, 1], R[1, 2], R[2, 2]
    sq = math.sqrt(Q)
    m2 = np.arange(math.ceil(-sq / r22 - s - 1e-12), math.floor(sq / r22 - s + 1e-12) + 1)
    k2 = m2 + s
    Q2 = Q - (r22 * k2) ** 2
    ok = Q2 >= 0
    k2, Q2 = k2[ok], Q2[ok]
    c1 = -r12 * k2 / r11
    w1 = np.sqrt(Q2) / r11
    lo1 = np.ceil(c1 - w1 - s - 1e-12)
    hi1 = np.floor(c1 + w1 - s + 1e-12)
    cnt = np.maximum(0, (hi1 - lo1 + 1)).astype(np.int64)
    ix = np.repeat(np.arange(len(k2)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    k1 = lo1[ix] + off + s
    k2e = k2[ix]
    Q1 = Q2[ix] - (r11 * k1 + r12 * k2e) ** 2
    ok = Q1 >= 0
    k1, k2e, Q1 = k1[ok], k2e[ok], Q1[ok]
    c0 = -(r01 * k1 + r02 * k2e) / r00
    w0 = np.sqrt(Q1) / r00
    lo0 = np.ceil(c0 - w0 - s - 1e-12)
    hi0 = np.floor(c0 + w0 - s + 1e-12)
    cnt = np.maximum(0, (hi0 - lo0 + 1)).astype(np.int64)
    ix = np.repeat(np.arange(len(k1)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    K = np.stack([lo0[ix] + off + s, k1[ix], k2e[ix]], 1)
    out = np.empty_like(K)
    out[:, p] = K
    return out


def lat_points(A, ainv_diag, half):
    """lattice (half=False) or half-integer (half=True) vectors with k^T A k <= QMAX (superset)."""
    if half:
        b = np.floor(np.sqrt(QMAX * ainv_diag) + 0.5)
        size = float(np.prod(2 * b))
    else:
        b = np.floor(np.sqrt(QMAX * ainv_diag) + 1e-12)
        size = float(np.prod(2 * b + 1))
    if size <= 6000:
        return _hbox(b) if half else _ibox(b)
    return fp_enum(A, QMAX * (1 + 1e-9), 0.5 if half else 0.0)


def z2_single(G, theta):
    """Z_2(G,theta) for theta in {0, 1/2} (all components equal) -- test helper."""
    Gi = np.linalg.inv(G)
    sd = math.sqrt(np.linalg.det(G))
    n = _ibox(np.floor(np.sqrt(QMAX * np.diag(Gi)) + 1e-12))
    Q = ((n @ G) * n).sum(1)
    m = (Q <= QMAX) & (Q > 1e-12)
    n, Q = n[m], Q[m]
    c = np.cos(2 * PI * theta * n.sum(1))
    Zr = np.dot(c, (1 + PI * Q) * np.exp(-PI * Q) / Q ** 2) - PI ** 2 / 2
    if theta == 0:
        k = _ibox(np.floor(np.sqrt(QMAX * np.diag(G)) + 1e-12))
        extra = 2.0
    else:
        k = _hbox(np.floor(np.sqrt(QMAX * np.diag(G)) + 0.5))
        extra = 0.0
    P = ((k @ Gi) * k).sum(1)
    mk = (P <= QMAX) & (P > 1e-12)
    P = P[mk]
    g = 2 * np.exp(-PI * P) - 2 * PI * np.sqrt(P) * erfc(np.sqrt(PI * P))
    return Zr + PI ** 2 / sd * (g.sum() + extra)


def zeff(G, grad=True):
    """Z_eff = 15 Z_2(G,0) - 45 Z_2(G,(1/2,1/2,1/2)) and dZ/dG (full symmetric gradient)."""
    Gi = np.linalg.inv(G)
    sd = math.sqrt(np.linalg.det(G))
    n = lat_points(G, np.diag(Gi), False)
    Q = ((n @ G) * n).sum(1)
    m = (Q <= QMAX) & (Q > 1e-12)
    n, Q = n[m], Q[m]
    odd = (np.rint(n.sum(1)).astype(np.int64) & 1).astype(bool)
    w = np.where(odd, 60.0, -30.0)
    eQ = np.exp(-PI * Q)
    Zr = np.dot(w, (1.0 + PI * Q) * eQ / Q ** 2)
    dg = np.diag(G)
    k = lat_points(Gi, dg, False)
    P = ((k @ Gi) * k).sum(1)
    mk = (P <= QMAX) & (P > 1e-12)
    k, P = k[mk], P[mk]
    kh = lat_points(Gi, dg, True)
    Ph = ((kh @ Gi) * kh).sum(1)
    mh = Ph <= QMAX
    kh, Ph = kh[mh], Ph[mh]
    sP, sPh = np.sqrt(P), np.sqrt(Ph)
    eP, ePh = erfc(np.sqrt(PI * P)), erfc(np.sqrt(PI * Ph))
    gP = 2 * np.exp(-PI * P) - 2 * PI * sP * eP
    gPh = 2 * np.exp(-PI * Ph) - 2 * PI * sPh * ePh
    Zk = (PI ** 2 / sd) * (15.0 * (gP.sum() + 2.0) - 45.0 * gPh.sum())
    Z = Zr + Zk + 15.0 * PI ** 2
    if not grad:
        return Z
    fp = -eQ * (PI ** 2 * Q ** 2 + 2 * PI * Q + 2) / Q ** 3
    Dr = (n.T * (w * fp)) @ n
    kk = np.concatenate([k, kh])
    hk = np.concatenate([15.0 * eP / sP, -45.0 * ePh / sPh])
    Km = (kk.T * hk) @ kk
    Dk = (PI ** 3 / sd) * (Gi @ Km @ Gi)
    D = Dr + Dk - 0.5 * Zk * Gi
    return Z, D


# ---------------------------------------------------------------- backgrounds
class PlateauBG:
    def __init__(self, phib=PHIB):
        self.phib = phib
        self.vb = vpot(phib)

        def f(t, y):
            lr, H, ph, dph = y
            return [H, -2 * H * H - dph * dph / 6 + 2 * vpot(ph) / 3, dph, -3 * H * dph - dvpot(ph)]
        y0 = [0.0, 0.0, phib, 0.0]
        self.fw = solve_ivp(f, (0, 12.5), y0, method='DOP853', rtol=1e-12, atol=1e-13, dense_output=True)
        self.bw = solve_ivp(f, (0, -3.5), y0, method='DOP853', rtol=1e-12, atol=1e-13, dense_output=True)

    def __call__(self, t):
        return self.fw.sol(t) if t >= 0 else self.bw.sol(t)

    def constraint(self, t):
        lr, H, ph, dph = self(t)
        return 3 * H * H - (0.5 * dph * dph + vpot(ph) - self.vb * math.exp(-4 * lr))


class Model:
    def __init__(self, bg='lead', r=R_ZM, phib=PHIB):
        self.bg = bg
        self.Zf = zeff(EYE, grad=False)
        self.r = r
        self.Z0 = (1.0 - r) * self.Zf
        if bg == 'lead':
            self.vb = 3.0
            self.pbg = None
        else:
            self.pbg = PlateauBG(phib)
            self.vb = self.pbg.vb
        self.nfev = 0
        self.maxfev = 400000

    # ---- linear transfer for a single mode (kappa defined w.r.t. Z0) from t0 to T
    def lin_transfer(self, kappa, t0, T=12.0):
        a0 = CN
        b0 = (1.0 / CN) * kappa * (self.vb / 3.0)
        if self.bg == 'lead':
            def f(t, y):
                lr = 0.5 * logcosh(2 * t)
                a = a0 * math.exp(-3 * lr)
                b = b0 * math.exp(-lr)
                return [a * y[2], a * y[3], b * y[0], b * y[1]]
            y0 = [1.0, 0.0, 0.0, 1.0]
            s = solve_ivp(f, (t0, T), y0, method='DOP853', rtol=1e-12, atol=1e-14)
            return s.y[:, -1].reshape(2, 2)
        bgy = self.pbg(t0)

        def f(t, y):
            lr, H, ph, dph = y[:4]
            a = a0 * math.exp(-3 * lr)
            b = b0 * math.exp(-lr)
            return [H, -2 * H * H - dph * dph / 6 + 2 * vpot(ph) / 3, dph, -3 * H * dph - dvpot(ph),
                    a * y[6], a * y[7], b * y[4], b * y[5]]
        y0 = list(bgy) + [1.0, 0.0, 0.0, 1.0]
        s = solve_ivp(f, (t0, T), y0, method='DOP853', rtol=1e-12, atol=1e-14)
        return s.y[4:, -1].reshape(2, 2)

    def zeta(self, G):
        return (zeff(G, grad=False) - self.r * self.Zf) / self.Z0

    # ---- full nonlinear RHS
    def rhs(self, t, y):
        self.nfev += 1
        if self.nfev > self.maxfev:
            raise RuntimeError('nfev cap')
        lr, H = y[0], y[1]
        G = unpack(y[2:8])
        Y = unpack(y[8:14])
        Gi = np.linalg.inv(G)
        e3 = math.exp(-3.0 * lr)
        A = Gi @ Y
        S = np.trace(A @ A) * e3 * e3
        dy = np.empty_like(y)
        dy[0] = H
        if self.bg == 'lead':
            dy[1] = 2.0 - 2.0 * H * H - S / 6.0
        else:
            ph, dph = y[14], y[15]
            dy[1] = -2.0 * H * H - dph * dph / 6.0 + 2.0 * vpot(ph) / 3.0 - S / 6.0
            dy[14] = dph
            dy[15] = -3.0 * H * dph - dvpot(ph)
        dy[2:8] = pack(2.0 * e3 * Y)
        Yd = 2.0 * e3 * (Y @ A)
        e1 = math.exp(-lr)
        if e1 ** 4 > 1e-14:
            _, D = zeff(G)
            DG = D @ G
            TF = DG - (np.trace(DG) / 3.0) * EYE
            Yd = Yd + (2.0 * self.vb * e1 / self.Z0) * (G @ TF)
        dy[8:14] = pack(Yd)
        return dy

    def gates(self, y):
        lr, H = y[0], y[1]
        G = unpack(y[2:8])
        Y = unpack(y[8:14])
        Gi = np.linalg.inv(G)
        e3 = math.exp(-3.0 * lr)
        A = Gi @ Y
        S = np.trace(A @ A) * e3 * e3
        z = self.zeta(G) if 4 * lr < 60 else 1.0
        if self.bg == 'lead':
            rhs_c = 3.0 - 3.0 * z * math.exp(-4 * lr) + S / 2.0
        else:
            ph, dph = y[14], y[15]
            rhs_c = 0.5 * dph * dph + vpot(ph) - self.vb * z * math.exp(-4 * lr) + S / 2.0
        return {'constraint': float(3 * H * H - rhs_c), 'detG_m1': float(np.linalg.det(G) - 1.0),
                'trGiY': float(np.trace(A))}

    # ---- initial data
    def initial(self, t0, beta, Pi):
        Bm = np.tensordot(beta, XB, 1)
        Pm = np.tensordot(Pi, XB, 1)
        G0 = expm_sym(2.0 * Bm)
        R = expm_sym(Bm)
        Y0 = CN * (R @ Pm @ R)
        A = np.linalg.solve(G0, Y0)
        s0 = float(np.trace(A @ A))
        z = self.zeta(G0)
        vb = self.vb
        extra = []
        if t0 == 0.0:
            if z <= 0 or z < 3.0 * (s0 / (4.0 * vb)) ** (2.0 / 3.0):
                return None
            rts = np.roots([1.0, 0.0, -z, s0 / (2.0 * vb)])
            rts = rts[np.abs(rts.imag) < 1e-10].real
            u = rts.max()
            lr0, H0 = 0.5 * math.log(u), 0.0
            if self.bg != 'lead':
                extra = [self.pbg.phib, 0.0]
        else:
            if self.bg == 'lead':
                lr0 = 0.5 * logcosh(2 * t0)
                S0 = s0 * math.exp(-6 * lr0)
                arg = 1.0 - z * math.exp(-4 * lr0) + S0 / 6.0
            else:
                lrb, Hb, ph0, dph0 = self.pbg(t0)
                lr0 = lrb
                S0 = s0 * math.exp(-6 * lr0)
                arg = (0.5 * dph0 * dph0 + vpot(ph0) - vb * z * math.exp(-4 * lr0) + S0 / 2.0) / 3.0
                extra = [ph0, dph0]
            if arg < 0:
                return None
            H0 = -math.sqrt(arg)
        y0 = np.concatenate([[lr0, H0], pack(G0), pack(Y0), extra])
        return y0

    def run(self, t0, beta, Pi, T=12.0, pinch=1e-3, rtol=1e-9, atol=1e-11, stop_on_down=True):
        y0 = self.initial(t0, beta, Pi)
        if y0 is None:
            return {'fate': 'no_turning'}
        g0 = self.gates(y0)
        ev_pinch = lambda t, y: np.linalg.eigvalsh(unpack(y[2:8]))[0] - pinch
        ev_pinch.terminal, ev_pinch.direction = True, -1
        ev_coll = lambda t, y: y[0] + 1.5
        ev_coll.terminal, ev_coll.direction = True, -1
        ev_down = lambda t, y: y[1]
        ev_down.terminal, ev_down.direction = bool(stop_on_down), -1
        ev_up = lambda t, y: y[1]
        ev_up.terminal, ev_up.direction = False, 1
        self.nfev = 0
        t_start = time.time()
        try:
            sol = solve_ivp(self.rhs, (t0, T), y0, method='DOP853', rtol=rtol, atol=atol,
                            events=[ev_pinch, ev_coll, ev_down, ev_up])
        except RuntimeError as e:
            return {'fate': 'error_' + str(e), 'nfev': self.nfev}
        yend = sol.y[:, -1]
        tend = float(sol.t[-1])
        ups = [float(x) for x in sol.t_events[3] if x > t0 + 1e-9]
        bounced = (t0 == 0.0) or len(ups) > 0
        tb = 0.0 if t0 == 0.0 else (ups[0] if ups else None)
        downs = [float(x) for x in sol.t_events[2] if x > t0 + 1e-9]
        if sol.status == 1:
            if len(sol.t_events[0]):
                fate = 'pinch_after_bounce' if bounced else 'no_bounce'
            elif len(sol.t_events[1]):
                fate = 'collapse_after_bounce' if bounced else 'no_bounce'
            else:
                fate = 'recollapse'
        elif sol.status == 0:
            if bounced and not downs and yend[1] > 0.9:
                fate = 'survive'
            else:
                fate = 'recollapse' if downs else 'weak'
        else:
            fate = 'solver_fail'
        G = unpack(yend[2:8])
        Gh = G / np.linalg.det(G) ** (1.0 / 3.0)
        try:
            Lg = logm_spd(Gh)
            bout = [0.5 * float(np.trace(XB[a] @ Lg)) for a in range(5)]
        except Exception:
            bout = None
        out = {'fate': fate, 't_end': tend, 't_bounce': tb, 'H_end': float(yend[1]), 'nfev': self.nfev,
               'H_down_times': downs,
               'beta_out': bout, 'gates_end': self.gates(yend) if fate == 'survive' else None,
               'gates_0': g0, 'cpu_s': time.time() - t_start,
               'mineig_end': float(np.linalg.eigvalsh(G)[0])}
        return out


# ---------------------------------------------------------------- selftest
def selftest():
    sys.stdout.reconfigure(encoding='utf-8')
    res = {}
    z0 = z2_single(EYE, 0.0)
    zh = z2_single(EYE, 0.5)
    res['Z2_cube_per'] = z0
    res['Z2_cube_AAA'] = zh
    mlead = Model('lead')
    res['Zfree'] = mlead.Zf
    res['NC'] = mlead.Zf / z0
    # Hessian eigenvalues along X_A at the cube
    lam = []
    h = 2e-3
    for a in range(5):
        zp = zeff(expm_sym(2 * h * XB[a]), False)
        zm = zeff(expm_sym(-2 * h * XB[a]), False)
        lam.append((zp + zm - 2 * mlead.Zf) / h ** 2)
    res['lambda_A_free'] = lam
    res['kappa_free'] = [3 * l / mlead.Zf for l in lam]
    res['kappa_xi'] = [3 * l / mlead.Z0 for l in lam]
    res['sum_kappa_free'] = sum(res['kappa_free'])
    # gradient finite-difference check on a generic shape
    rng = np.random.default_rng(1)
    Bg = np.tensordot(rng.normal(0, 0.15, 5), XB, 1)
    Gg = expm_sym(2 * Bg)
    Z, D = zeff(Gg)
    errs = []
    for _ in range(4):
        dG = rng.normal(size=(3, 3)); dG = dG + dG.T
        e = 1e-5
        fd = (zeff(Gg + e * dG, False) - zeff(Gg - e * dG, False)) / (2 * e)
        errs.append(abs(fd - np.trace(D @ dG)) / max(1.0, abs(fd)))
    res['grad_fd_relerr_max'] = max(errs)
    # existence criterion p_c along Pi = p X_D2 (beta=0)
    res['p_c_pred'] = math.sqrt(12 * (1 / 3) ** 1.5 * 8 / 9)
    # linear transfers
    kE_f, kT_f = res['kappa_free'][1], res['kappa_free'][3]
    kE, kT = res['kappa_xi'][1], res['kappa_xi'][3]
    out = {}
    mplat = Model('plat')
    res['vb_plateau'] = mplat.vb
    res['plateau_constraint_t3'] = mplat.pbg.constraint(3.0)
    for name, mdl in (('lead', mlead), ('plat', mplat)):
        for lab, kap in (('E_free', kE_f), ('E_xi', kE), ('T_free', kT_f), ('T_xi', kT), ('toy6', 6.0)):
            M = mdl.lin_transfer(kap, 0.0)
            Tm = M @ np.diag([1, -1]) @ np.linalg.inv(M) @ np.diag([1, -1])
            out[name + '_' + lab] = {'m11': M[0, 0], 'm12': M[0, 1], 'm21': M[1, 0], 'm22': M[1, 1],
                                     'T11': Tm[0, 0], 'T12': Tm[0, 1], 'T21': Tm[1, 0], 'detM': np.linalg.det(M)}
    res['transfers'] = out
    # closed forms on leading background
    from scipy.special import loggamma
    def T12_closed(k):
        nu = math.sqrt(4 * k - 1)
        z = complex(1.25, nu / 4)
        return math.exp(math.lgamma(1.75) + math.lgamma(0.75) - 2 * loggamma(z).real)
    def T11_closed(k):
        return S2 * math.cosh(PI * math.sqrt(4 * k - 1) / 4)
    res['T12_closed'] = {'E_free': T12_closed(kE_f), 'E_xi': T12_closed(kE), 'toy6': T12_closed(6.0)}
    res['T11_closed'] = {'E_free': T11_closed(kE_f), 'E_xi': T11_closed(kE), 'toy6': T11_closed(6.0)}
    # delta calibration and natural-state linear widths
    d_plat = 1.565e-4 / (1 - R_ZM) ** 0.75
    d_lead = d_plat * (3.0 / mplat.vb) ** 0.75
    res['delta_plat_xi'] = d_plat
    res['delta_lead_xi'] = d_lead
    for name, mdl, dl in (('lead', mlead, d_lead), ('plat', mplat, d_plat)):
        for lab, kap in (('E', kE), ('T', kT)):
            M = mdl.lin_transfer(kap, 0.0)
            snat = 3 * dl / (2 * S2 * math.sqrt(abs(kap) * mdl.vb / 3))
            sstar = abs(M[0, 1] / M[0, 0]) * dl
            var = M[0, 0] ** 2 * snat + M[0, 1] ** 2 * dl ** 2 / snat
            res['nat_' + name + '_' + lab] = {'s_nat': snat, 'x_nat': snat / sstar, 'sigma_lin': math.sqrt(var),
                                              'floor_sigma': math.sqrt(2 * abs(M[0, 0] * M[0, 1]) * dl)}
    # small-amplitude nonlinear vs linear, and timing at natural amplitude
    beta = rng.normal(size=5) * 1e-5
    Pi = rng.normal(size=5) * 1e-5
    r1 = mlead.run(0.0, beta, Pi)
    ks = [kE, kE, kT, kT, kT]
    blin = []
    for a in range(5):
        M = mlead.lin_transfer(ks[a], 0.0)
        blin.append(M[0, 0] * beta[a] + M[0, 1] * Pi[a])
    res['small_amp'] = {'fate': r1['fate'], 'beta_out': r1['beta_out'], 'beta_lin': blin,
                        'rel_dev': [abs(r1['beta_out'][a] / blin[a] - 1) for a in range(5)],
                        'gates_end': r1['gates_end'], 'cpu_s': r1['cpu_s'], 'nfev': r1['nfev']}
    sn = [res['nat_lead_E']['s_nat']] * 2 + [res['nat_lead_T']['s_nat']] * 3
    zz = rng.normal(size=10)
    b1 = zz[:5] * np.sqrt(sn)
    p1 = zz[5:] * d_lead / np.sqrt(sn)
    for s in (1.0, 10.0):
        rr = mlead.run(0.0, s * b1, s * p1)
        res['nat_run_s%g' % s] = {k: rr.get(k) for k in ('fate', 'cpu_s', 'nfev', 'beta_out', 'gates_end', 't_end')}
    rr = mplat.run(0.0, b1, p1)
    res['plat_run_s1'] = {k: rr.get(k) for k in ('fate', 'cpu_s', 'nfev', 'beta_out', 'gates_end')}
    rr = mlead.run(-1.093, b1, p1)
    res['lead_run_tm1093'] = {k: rr.get(k) for k in ('fate', 'cpu_s', 'nfev', 'beta_out', 'gates_end', 't_bounce', 'gates_0')}
    print(json.dumps(res, indent=1, default=float))
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'd114_selftest.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    if sys.argv[1] == 'selftest':
        selftest()
