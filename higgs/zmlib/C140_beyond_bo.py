# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C140_beyond_bo.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/C140_beyond_bo.py on 2026-10-14 by agent holonomy-cross (C149/D121).
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Two-dimensional (r, |w|) zero-mode model beyond Born-Oppenheimer (D112): exact conditional-Gaussian decomposition (C0, B_x, Om_eff^2), 2D Siegert resonance solver (sine-DVR in s = |w|, matrix Numerov in r, WKB outgoing condition, local-window Muller), lam = 0 ground-state regression against the full-dimensional DMC.  Paper: PRD App. H2. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C140 -- DNEC v12 round 2, agent higgs-beyond-bo (direction A-II, derivation D112).  Claude, 2026-10-13.

Beyond-Born-Oppenheimer treatment of the soft U(1)_Y / photon zero-mode direction of the reduced Higgs zero-mode problem.

2D model (D112 Sec. 1).  The D108 conditional Gaussian for y_i = (x^1, x^2, x^3, w)_i, K = diag(a, a, [[k, c], [c, b]]),
factorises EXACTLY as  Psi = phi(x | r, w) chi(w):  phi = SU(2) Gaussian with K_x = diag(a, a, k) and mean
<x^3_i> = -beta w_i (beta = c/k);  chi = w-marginal with precision bt = b - c^2/k.  Promoting chi(w) -> chi(r, w)
(joint exact dynamics of the Higgs radius r and the U(1)_Y 3-vector w, s-wave in w) gives the exact projection
    H_2D = -1/2 d2/dr2 + 3/(8 r^2) - 1/2 Lap_w + C0(r) + B_x(r) + 1/2 Om_eff(r)^2 |w|^2 + lam r^4/4 + dV(r) [+ dE_dmc(r)]
    D108 E_tot = C0 + (3/4)(bt + Om^2/bt)   (= C0 + 3/2 Om at the D108 optimum bt = Om)
    Om^2  = (r^2/4)(gY - g2 beta)^2 + 2 g2^2 beta^2/a + 4 a beta^2/r^2
    C0    = (3/4)(2a + k) + (3/4) k beta^2 + (3/16) r^2 (2 g2^2/a + g2^2/k) + (3/4) g2^2/a^2 + (3/2) g2^2/(a k)
            + 3 (k - a)^2/(a k r^2)
    D108 B_r = B_x + B_w + B_beta:  B_x = (3/16)[2 (a'/a)^2 + (k'/k)^2],  B_w = (3/16)(bt'/bt)^2 (-> 3/(16 r^2)),
            B_beta = 3 k beta'^2/(8 bt)  [= <(k/4) beta'^2 |w|^2>, kept as Om_eff^2 = Om^2 + (k/2) beta'^2].
B_w is the Born-Huang term that Stage 3 identified as the residual error; in H_2D it is replaced by exact dynamics.
dE_dmc = Stage-3 rotating-frame DMC correction E_tot^ex - E_tot^G (S3_prop.json rot_knots/rot_dEtot, same spline).
Numerics: sine-DVR in s = |w| (v = s chi), batched renormalised matrix Numerov in r from the Frobenius start, Siegert
condition (2nd-order WKB outgoing/decaying) in the local adiabatic w-channels at R_out; det(u'_m - Y u_m) is entire in E
(log det F accumulated); vectorised complex Muller.  Independent check: sparse 2D finite differences (fd2d).
Matching: MC+CW (D110/C138, code copied UNCHANGED with provenance line), comparison MU (D108).  Couplings v11.
Usage: python -B C140_beyond_bo.py prep | lam0 | res | fd2d | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r2/higgs_beyond_bo/run01"))
S3DIR = os.path.join(ROOT, _zp.w("r1/higgs_survival/run02_stage3"))
C138SUM = os.path.join(ROOT, _zp.w("r2/higgs_matching/run01/C138_summary.json"))
PROFILE = os.path.join(RES, "C140_profile_v11.npz")
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
C72_LAM0, C72_SE = 4.4478058, 0.0013
HLB = 0.00437243
ETA = 1.31102877714606 / HLB
DLAM_DMT = -0.00656
R0 = 0.01
LAMS = [-0.007, -0.008, -0.009, -0.010, -0.011, -0.012, -0.013, -0.0143]
LAM_MU = [-0.008, -0.010, -0.0143]
LAM_REP = [-0.008, -0.010, -0.0143]
# Stage-3 1-sigma errors of E_tot^ex (D108_STAGE3_DMC_BRACKET_CN.md Sec. 2 table; r = 0: E_g^ex)
S3_ERR = {0.0: 0.00055, 1.0: 0.0042, 1.5: 0.0024, 2.0: 0.0017, 2.5: 0.0014, 3.0: 0.0012, 3.7: 0.0009, 4.5: 0.0009,
          5.5: 0.0007, 7.0: 0.0004, 9.0: 0.0004}
# C138 static MC+CW (v11) gamma (C138_summary.json) -- used for guesses and regression only
G_MC = {-0.006: 1.8555370621051292e-05, -0.008: 2.0101243146834757e-04, -0.009: 4.7904014377768184e-04,
        -0.010: 9.871149132904924e-04, -0.011: 1.8160803541031922e-03, -0.012: 3.052774754412369e-03,
        -0.0143: 7.803079543827287e-03, -0.016: 1.3208301236758787e-02}
Q_MC_010 = 4.409885198394229
G_MU = {-0.008: 4.566e-4, -0.010: 2.0507511357905073e-3, -0.0143: 1.3905e-2}
Q_MU = {-0.008: 4.400, -0.010: 4.3890, -0.0143: 4.3626}
RTAB = [0.3, 0.5, 0.65, 1.0, 1.22, 1.5, 2.0, 2.65, 3.0, 3.7, 4.5, 5.44, 7.0, 9.0, 12.0, 16.0]


# ------------------------------------------------------------------------------------------------ io
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, (complex, np.complexfloating)):
        return [clean(o.real), clean(o.imag)]
    return o


def wjson(name, obj):
    p = os.path.join(RES, name)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def jload(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def say(msg):
    print(msg, flush=True)
    with open(os.path.join(RES, "C140_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def status(upd):
    p = os.path.join(RES, "STATUS.json")
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:  # noqa: BLE001
        st = {}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(st.get(k), dict):
            st[k].update(v)
        else:
            st[k] = v
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(st), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------------------------------------------ model
_C = None


def core():
    global _C
    if _C is None:
        import C138_core as C  # copied unchanged from higgs_matching
        _C = C
    return _C


def derived(rf, P, dP, g2, gY):
    a, k, b, c = P                       # R14 order [a, k, b, c]
    ap, kp, bp, cp = dP
    r = rf
    beta = c / k
    betap = (cp * k - c * kp) / k ** 2
    D = k * b - c * c
    bt = D / k
    Dp = kp * b + k * bp - 2 * c * cp
    btp = (Dp * k - D * kp) / k ** 2
    Om2 = r * r / 4 * (gY - g2 * beta) ** 2 + 2 * g2 * g2 * beta ** 2 / a + 4 * a * beta ** 2 / (r * r)
    C0 = (0.75 * (2 * a + k) + 0.75 * k * beta ** 2 + 3.0 / 16 * r * r * (2 * g2 * g2 / a + g2 * g2 / k)
          + 0.75 * g2 * g2 / a ** 2 + 1.5 * g2 * g2 / (a * k) + 3 * (k - a) ** 2 / (a * k * r * r))
    Bx = 3.0 / 16 * (2 * (ap / a) ** 2 + (kp / k) ** 2)
    Bw = 3.0 / 16 * (btp / bt) ** 2
    Bb = 3 * k * betap ** 2 / (8 * bt)
    Om2e = Om2 + 0.5 * k * betap ** 2
    return dict(a=a, k=k, beta=beta, betap=betap, bt=bt, Om2=Om2, Om2e=Om2e, C0=C0, Bx=Bx, Bw=Bw, Bb=Bb)


class Model:
    def __init__(self):
        d = np.load(PROFILE)
        rf, P, dP, Et, Br = d["rf"], d["P"], d["dP"], d["Et"], d["Br"]
        q = derived(rf, P, dP, V11["g2"], V11["gY"])
        self.rf, self.q, self.Et, self.Br = rf, q, Et, Br
        r2 = rf * rf
        self.s_bo = CubicSpline(rf, r2 * (Et + Br))          # identical to R14 gauss_profile 'spl'
        self.s_c0 = CubicSpline(rf, r2 * q["C0"])
        self.s_bx = CubicSpline(rf, r2 * q["Bx"])
        self.s_om2 = CubicSpline(rf, q["Om2e"])
        with open(os.path.join(S3DIR, "S3_prop.json"), encoding="utf-8") as f:
            pj = json.load(f)
        self.kn = np.array(pj["rot_knots"], float)
        self.kv = np.array(pj["rot_dEtot"], float)
        self.ke = np.array([S3_ERR[round(float(x), 1)] for x in self.kn])

    def gR(self, x, tilt=0.0):
        """Stage-3 R_rot correction E_tot^ex - E_tot^G (S3_bracket.ph_prop spline: dE/dr(0) = 0, natural, frozen > 9)."""
        vals = self.kv + tilt * np.where(self.kn < 3.0, 1.0, -1.0) * self.ke
        f = CubicSpline(self.kn, vals, bc_type=((1, 0.0), "natural"))
        x = np.asarray(x, float)
        return np.where(x <= self.kn[-1], f(np.minimum(x, self.kn[-1])), vals[-1])


def dVm(match, r, lam):
    if match == "L0":
        return np.zeros_like(r)
    return core().dV(match, r, lam, V11["g2"], V11["gY"], V11["yt"])


def build(M, var, match, lam, r, tilt=0.0):
    """radial channel potential U1(r) (incl. 3/(8r^2)) and, for 2D variants, Om_eff^2(r)."""
    r2 = r * r
    base = 3.0 / (8 * r2) + lam * r2 * r2 / 4 + dVm(match, r, lam)
    corr = M.gR(r, tilt) if "ex" in var else 0.0
    c0 = M.s_c0(r) / r2
    bx = M.s_bx(r) / r2
    if var in ("1D_G", "1D_ex"):
        return base + M.s_bo(r) / r2 + corr, None
    if var in ("2D_G", "2D_ex"):
        return base + c0 + bx + corr, M.s_om2(r)
    if var == "2D_ex_noBx":
        return base + c0 + corr, M.s_om2(r)
    if var in ("1D_BO2Dex", "1D_BO2Dex_lo"):
        o2 = M.s_om2(r)
        U = base + c0 + bx + corr + 1.5 * np.sqrt(o2)
        if var == "1D_BO2Dex":
            U = U + 3.0 / 16 * (M.s_om2(r, 1) / (2 * o2)) ** 2
        return U, None
    if var == "2D_exX":          # + first-order cross term with the 2D conditional <|w|^2>(r) (indicative, D112 Sec. 6)
        EXx, omX2, _ = ex_parts(M, r, lam)
        return base + c0 + bx + corr + EXx + 0.5 * omX2 * s2c_fun(M)(r), M.s_om2(r)
    if var == "1D_GX_rs1":       # D110-style: BO cross term E_X^BO frozen below r_s = 1 (comparison)
        return base + M.s_bo(r) / r2 + ex_parts(M, np.maximum(r, 1.0), lam)[2], None
    if var == "SEP1D":
        return base + c0 + bx, None
    if var == "SEP2D":
        return base + c0 + bx, np.full_like(r, 0.09)
    raise ValueError(var)


def frob(U1, r):
    sel = r <= 0.06
    A = np.vstack([np.ones(sel.sum()), r[sel], r[sel] ** 2, r[sel] ** 3]).T
    coef = np.linalg.lstsq(A, (r * r * U1)[sel], rcond=None)[0]
    beta, alpha = float(coef[0]), float(coef[1])
    return 0.5 + math.sqrt(0.25 + 2 * beta), alpha


def rgrid(Rout, h):
    m = int(round((Rout - R0) / h))
    return R0 + h * np.arange(m + 3), m


class SGrid:
    """sine DVR on (0, Smax) for v = s chi (3D s-wave), Dirichlet at both ends."""

    def __init__(self, N, Smax):
        self.N, self.Smax = int(N), float(Smax)
        j = np.arange(1, self.N + 1)
        self.s = j * self.Smax / (self.N + 1)
        self.s2 = self.s ** 2
        U = math.sqrt(2.0 / (self.N + 1)) * np.sin(np.pi * np.outer(j, j) / (self.N + 1))
        self.T = (U * (0.5 * (np.pi * j / self.Smax) ** 2)[None, :]) @ U.T


# ------------------------------------------------------------------------------------------------ solver
def bc_data(U1, om2, sg, h, m, order=2):
    """Siegert/decaying 2nd-order WKB log-derivative in the local adiabatic w-channels at r_m -> Y(E) (P, N, N)."""
    P = U1.shape[0]
    if sg is None:
        eps = np.zeros((P, 5, 1))
        Phi = np.ones((P, 1, 1))
    else:
        N = sg.N
        eps = np.empty((P, 5, N))
        Phi = np.empty((P, N, N))
        for p in range(P):
            for j, i in enumerate(range(m - 2, m + 3)):
                w, V = np.linalg.eigh(sg.T + np.diag(0.5 * om2[p, i] * sg.s2))
                eps[p, j] = w
                if i == m:
                    Phi[p] = V
    Uc = U1[:, m - 2:m + 3][:, :, None] + eps
    Um_ = Uc[:, 2]
    U1d = (Uc[:, 0] - 8 * Uc[:, 1] + 8 * Uc[:, 3] - Uc[:, 4]) / (12 * h)
    U2d = (-Uc[:, 0] + 16 * Uc[:, 1] - 30 * Uc[:, 2] + 16 * Uc[:, 3] - Uc[:, 4]) / (12 * h * h)

    def Y(E):
        d = E[:, None] - Um_
        kk = np.where(d.real > 0, np.sqrt(2 * d + 0j), 1j * np.sqrt(-2 * d + 0j))
        if order == 0:
            y = 1j * kk
        else:
            k1 = -U1d / kk
            k2 = -U2d / kk - U1d ** 2 / kk ** 3
            y = 1j * kk - k1 / (2 * kk) + (1j / (2 * kk)) * (3 * k1 ** 2 / (4 * kk ** 2) - k2 / (2 * kk))
        return np.einsum("pab,pb,pcb->pac", Phi, y, Phi)
    return Y


def prop(E, Dd, Tn, h, m, ph0, ph1, Yf):
    """batched renormalised matrix Numerov u'' = 2(V - E)u, F_i = W_i u_i, R_i = F_{i+1} F_i^{-1};
    returns (complex sign, log|.|) of det(u'_m - Y u_m) = det(A) det(F_m) (entire in E)."""
    P, n, N = Dd.shape
    dg = np.arange(N)
    cc = h * h / 6.0
    Eb = E[:, None]

    def Wm(i):
        A = np.empty((P, N, N), complex)
        A[:] = Tn
        A[:, dg, dg] += 1.0 - cc * (Dd[:, i, :] - Eb)
        return A
    W0 = Wm(0)
    Rp = (ph1 / ph0)[:, None, None] * (Wm(1) @ np.linalg.inv(W0))
    s0, l0 = np.linalg.slogdet(W0)
    sgn = s0.astype(complex)
    la = l0 + N * np.log(ph0)
    s_, l_ = np.linalg.slogdet(Rp)
    sgn, la = sgn * s_, la + l_
    kW, kR, kRi = {}, {}, {}
    for i in range(1, m + 2):
        Wi = np.linalg.inv(Wm(i))
        Rpi = np.linalg.inv(Rp)
        Rc = 12.0 * Wi - Rpi
        Rc[:, dg, dg] -= 10.0
        if i <= m - 1:
            s_, l_ = np.linalg.slogdet(Rc)
            sgn, la = sgn * s_, la + l_
        if i >= m - 2:
            kW[i] = Wi
            kR[i] = Rc
        if i - 1 >= m - 2:
            kRi[i - 1] = Rpi
        Rp = Rc
    kW[m + 2] = np.linalg.inv(Wm(m + 2))
    G1 = kR[m]
    G2 = kR[m + 1] @ kR[m]
    Gm1 = kRi[m - 1]
    Gm2 = kRi[m - 2] @ kRi[m - 1]
    Up = (kW[m - 2] @ Gm2 - 8 * (kW[m - 1] @ Gm1) + 8 * (kW[m + 1] @ G1) - kW[m + 2] @ G2) / (12 * h)
    A = Up - Yf(E) @ kW[m]
    sA, lA = np.linalg.slogdet(A)
    return sgn * sA, la + lA


def muller(fun, x0, x1, x2, tol=1e-10, maxit=60):
    f0, f1, f2 = fun(x0), fun(x1), fun(x2)
    act = np.ones(x0.shape, bool)
    it = 0
    last = np.full(x0.shape, np.inf)
    with np.errstate(all="ignore"):
        for it in range(maxit):
            h1 = x1 - x0
            h2 = x2 - x1
            d1 = (f1 - f0) / h1
            d2 = (f2 - f1) / h2
            A = (d2 - d1) / (h2 + h1)
            B = A * h2 + d2
            disc = np.sqrt(B * B - 4 * A * f2)
            den = np.where(np.abs(B + disc) >= np.abs(B - disc), B + disc, B - disc)
            dx = -2 * f2 / den
            ok = act & np.isfinite(dx)
            dx = np.where(ok, dx, 0)
            last = np.where(ok, np.abs(dx), last)
            x3 = x2 + dx
            x0, x1, x2 = x1, x2, x3
            f0, f1 = f1, f2
            f2 = fun(x3)
            act = act & (np.abs(dx) > tol * np.maximum(1, np.abs(x3)))
            if not act.any():
                break
    return x2, it, act, last


def job(spec):
    t0 = time.time()
    M = Model()
    cfg = spec["cfg"]
    N, h, Rout = int(cfg["N"]), float(cfg["h"]), float(cfg["Rout"])
    Smax, order = float(cfg.get("Smax", 14.0)), int(cfg.get("order", 2))
    r, m = rgrid(Rout, h)
    sg = SGrid(N, Smax) if N > 1 else None
    probs = spec["probs"]
    P = len(probs)
    U1 = np.empty((P, len(r)))
    om2 = np.empty((P, len(r))) if sg is not None else None
    ph0, ph1, fr = np.empty(P), np.empty(P), []
    for j, p in enumerate(probs):
        u, o = build(M, p["var"], p["match"], p["lam"], r, p.get("tilt", 0.0))
        if (o is None) != (sg is None):
            raise ValueError("dimension mismatch " + p["var"])
        U1[j] = u
        if sg is not None:
            om2[j] = o
        s, al = frob(u, r)
        ph0[j] = r[0] ** s * (1 + al / s * r[0])
        ph1[j] = r[1] ** s * (1 + al / s * r[1])
        fr.append([s, al])
    if sg is None:
        Dd = U1[:, :, None].copy()
        Tn = np.zeros((1, 1), complex)
    else:
        Dd = U1[:, :, None] + 0.5 * om2[:, :, None] * sg.s2[None, None, :]
        Tn = (-(h * h / 6.0) * sg.T).astype(complex)
    Yf = bc_data(U1, om2, sg, h, m, order)
    cn = {}

    def fun(E):
        s_, la = prop(np.asarray(E, complex), Dd, Tn, h, m, ph0, ph1, Yf)
        if "c" not in cn:
            cn["c"] = la.copy()
        return s_ * np.exp(np.clip(la - cn["c"], -700.0, 700.0))

    scan = None
    if spec.get("scan") is not None:
        Es = np.asarray(spec["scan"], float)
        vals = np.array([fun(np.full(P, e + 0j)).real for e in Es])
        g = np.full(P, np.nan + 0j)
        nsc = []
        for j in range(P):
            v = vals[:, j]
            idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
            nsc.append(int(len(idx)))
            if len(idx):
                i = idx[0]
                g[j] = Es[i] - v[i] * (Es[i + 1] - Es[i]) / (v[i + 1] - v[i])
        scan = dict(E=Es, n_sign_changes=nsc)
        d = 0.004
    elif spec.get("window") is not None:
        # narrow resonance: min |f| on a real-E window around the BO-bracket prediction, then Muller with small spacing
        # (the summed outgoing-wave phase of ~N open channels rotates ~N T_out rad per unit E: wide Muller starts fail)
        half, step = spec["window"]
        offs = np.arange(-half, half + 0.5 * step, step)
        cen = np.array([p["center"] for p in probs], float)
        av = np.array([np.abs(fun(cen + o + 0j)) for o in offs])
        i0 = int(np.argmin(np.abs(offs)))
        jmin = np.empty(P, int)
        for jj in range(P):         # interior local minimum of |f| closest to the predicted centre (not the global one:
            a_ = av[:, jj]          # a sloping background put the global minimum on the window edge for broad poles)
            loc = [i for i in range(1, len(offs) - 1) if a_[i] < a_[i - 1] and a_[i] < a_[i + 1]]
            jmin[jj] = min(loc, key=lambda i: abs(i - i0)) if loc else i0
        g = cen + offs[jmin] - 0.5j * np.array([p["gimag"] for p in probs])
        scan = dict(offsets=offs, centers=cen, E_min=cen + offs[jmin],
                    depth=[float(av[jmin[j], j] / np.median(av[:, j])) for j in range(P)],
                    edge=[bool(jmin[j] in (0, len(offs) - 1)) for j in range(P)],
                    log_abs_f=np.log(av + 1e-300).T)
        d = float(spec.get("d", 0.001))
    else:
        g = np.array([complex(p["guess"][0], p["guess"][1]) for p in probs])
        d = float(spec.get("d", 0.02))
    E, it, act, last = muller(fun, g - d, g + d, g + 0j)
    rows = []
    for j, p in enumerate(probs):
        conv = bool((not act[j]) and np.isfinite(E[j]))
        rows.append(dict(var=p["var"], match=p["match"], lam=p["lam"], tilt=p.get("tilt", 0.0), Q=float(E[j].real),
                         gamma=float(-2 * E[j].imag), converged=conv, last_dx=float(last[j]),
                         guess=[float(g[j].real), float(g[j].imag)], frob=fr[j]))
    out = dict(name=spec.get("name"), cfg=cfg, rows=rows, muller_iter=int(it), time_s=time.time() - t0, scan=scan)
    say("job %s done in %.1fs: %s" % (spec.get("name"), time.time() - t0,
                                       "; ".join("%s/%s/%g/t%g Q=%.6f g=%.4e%s" % (x["var"], x["match"], x["lam"],
                                                                                   x["tilt"], x["Q"], x["gamma"],
                                                                                   "" if x["converged"] else " NC")
                                                 for x in rows)))
    return out


def run_specs(specs, nw=5):
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(specs)))) as ex:
        return list(ex.map(job, specs))


def g_mc(lam):
    ls = np.array(sorted(G_MC))
    return float(np.exp(np.interp(lam, ls, np.log([G_MC[x] for x in ls]))))


BASE = dict(N=32, Smax=14.0, h=0.002, Rout=16.0, order=2)
ONE = dict(N=1, h=0.002, Rout=16.0)


# ------------------------------------------------------------------------------------------------ phases
def prep():
    t0 = time.time()
    C = core()
    R = C.R
    g2, gY = V11["g2"], V11["gY"]
    if not os.path.exists(PROFILE):
        gp = R.gauss_profile(g2, gY)
        rf, P = gp["rf"], gp["P"]
        nf = len(rf)
        Hm = R._mat(R.FH(*P, rf, g2, gY), nf).transpose(2, 0, 1)
        Gr = R._vec(R.FGR(*P, rf, g2, gY), nf)
        dP = -np.linalg.solve(Hm, Gr.T[:, :, None])[:, :, 0].T
        np.savez(PROFILE, rf=rf, P=P, dP=dP, Et=gp["Et"], Br=gp["Br"], beta_g=gp["beta_g"], alpha_g=gp["alpha_g"],
                 maxdec=gp["maxdec"], min_hess_eig=gp["min_hess_eig"], march_vs_fine=gp["march_vs_fine"])
        say("profile computed in %.1fs" % (time.time() - t0))
    M = Model()
    q, rf = M.q, M.rf
    Om = np.sqrt(q["Om2"])
    sel = rf >= 0.05
    bo2d = (q["C0"] + q["Bx"] + 1.5 * np.sqrt(q["Om2e"])
            + 3.0 / 16 * (M.s_om2(rf, 1) / (2 * q["Om2e"])) ** 2)
    chk = dict(max_abs_Et_minus_C0_1p5Om=float(np.max(np.abs(M.Et - (q["C0"] + 1.5 * Om)))),
               max_abs_Et_minus_identity=float(np.max(np.abs(M.Et - (q["C0"] + 0.75 * (q["bt"] + q["Om2"] / q["bt"]))))),
               max_rel_bt_over_Om_minus_1=float(np.max(np.abs(q["bt"] / Om - 1))),
               max_abs_Br_minus_Bx_Bw_Bb=float(np.max(np.abs(M.Br - (q["Bx"] + q["Bw"] + q["Bb"])))),
               max_abs_BO_of_2D_minus_D108_r_ge_0p05=float(np.max(np.abs(bo2d - (M.Et + M.Br))[sel])),
               r2Bw_small_r={f"{rf[i]:.4f}": float((rf * rf * q["Bw"])[i]) for i in (0, 3, 10, 25, 48)})
    tab = {}
    for x in RTAB:
        i = int(np.argmin(np.abs(rf - x)))
        tab[f"{rf[i]:.3f}"] = dict(Et=M.Et[i], Br=M.Br[i], C0=q["C0"][i], Om=Om[i], Om_eff=math.sqrt(q["Om2e"][i]),
                                   bt=q["bt"][i], Bx=q["Bx"][i], Bw=q["Bw"][i], Bb=q["Bb"][i], beta=q["beta"][i],
                                   a=q["a"][i], k=q["k"][i], gR=float(M.gR(rf[i])), gYr_over_2=gY * rf[i] / 2)
    r, m = rgrid(16.0, 0.002)
    fro = {v: frob(build(M, v, "L0", 0.0, r)[0], r) for v in ("1D_G", "2D_ex", "2D_G")}
    dvs = {f"{x}": float(dVm("MC", np.array([x]), -0.010)[0]) for x in (1.0, 3.0, 5.0)}
    say("prep checks: " + json.dumps(clean(chk)))
    reg = job(dict(name="regression_1D", cfg=ONE, probs=[
        dict(var="1D_G", match="MC", lam=-0.010, guess=[Q_MC_010 + 0.01, -0.6 * G_MC[-0.010]]),
        dict(var="1D_G", match="MU", lam=-0.010, guess=[Q_MU[-0.010] + 0.01, -0.6 * G_MU[-0.010]])]))
    rg = {x["match"]: dict(Q=x["Q"], gamma=x["gamma"], conv=x["converged"]) for x in reg["rows"]}
    rg["MC"]["C138_gamma"], rg["MC"]["C138_Q"] = G_MC[-0.010], Q_MC_010
    rg["MU"]["C138_gamma"] = G_MU[-0.010]
    rg["MC"]["rel"] = rg["MC"]["gamma"] / G_MC[-0.010] - 1
    rg["MU"]["rel"] = rg["MU"]["gamma"] / G_MU[-0.010] - 1
    out = dict(checks=chk, table=tab, frobenius=fro, dV_MC_lam_m0p010=dvs, regression=rg, time_s=time.time() - t0)
    wjson("C140_prep.json", out)
    status(dict(stage="prep done", key_numbers=dict(identity_checks=chk, regression_1D=rg)))
    say("prep regression: " + json.dumps(clean(rg)))


def lam0():
    sc = list(np.round(np.arange(4.36, 4.701, 0.02), 4))
    specs = [
        dict(name="1D", cfg=ONE, scan=list(np.round(np.arange(3.50, 4.801, 0.02), 4)),
             probs=[dict(var=v, match="L0", lam=0.0) for v in ("1D_G", "1D_ex", "1D_BO2Dex", "1D_BO2Dex_lo", "SEP1D")]
             + [dict(var="1D_ex", match="L0", lam=0.0, tilt=t) for t in (1.0, -1.0)]),
        dict(name="2D_base", cfg=BASE, scan=sc,
             probs=[dict(var=v, match="L0", lam=0.0) for v in ("2D_G", "2D_ex", "2D_ex_noBx")]
             + [dict(var="2D_ex", match="L0", lam=0.0, tilt=t) for t in (1.0, -1.0)]),
        dict(name="SEP2D", cfg=BASE, scan=list(np.round(np.arange(3.90, 4.901, 0.025), 4)),
             probs=[dict(var="SEP2D", match="L0", lam=0.0)]),
        dict(name="2D_N48_S16", cfg=dict(BASE, N=48, Smax=16.0), scan=sc, probs=[dict(var="2D_ex", match="L0", lam=0.0)]),
        dict(name="2D_N24_S12", cfg=dict(BASE, N=24, Smax=12.0), scan=sc, probs=[dict(var="2D_ex", match="L0", lam=0.0)]),
        dict(name="2D_h0p001", cfg=dict(BASE, h=0.001), scan=sc, probs=[dict(var="2D_ex", match="L0", lam=0.0)]),
    ]
    jobs = run_specs(specs)
    rows = {}
    for jb in jobs:
        for x in jb["rows"]:
            rows[f"{jb['name']}|{x['var']}|t{x['tilt']:g}"] = x
    Q = {k: v["Q"] for k, v in rows.items()}
    q2 = Q["2D_base|2D_ex|t0"]
    test = dict(C72=C72_LAM0, C72_se=C72_SE, Q_2D_ex=q2, minus_C72=q2 - C72_LAM0, pass_0p01=abs(q2 - C72_LAM0) <= 0.01,
                Q_1D_G=Q["1D|1D_G|t0"], Q_1D_ex=Q["1D|1D_ex|t0"], Q_2D_G=Q["2D_base|2D_G|t0"],
                Q_2D_ex_noBx=Q["2D_base|2D_ex_noBx|t0"], BO_bracket_2Dex=[Q["1D|1D_BO2Dex_lo|t0"], Q["1D|1D_BO2Dex|t0"]],
                sep_test=Q["SEP2D|SEP2D|t0"] - Q["1D|SEP1D|t0"] - 1.5 * 0.3,
                tilt=[Q["2D_base|2D_ex|t1"] - q2, Q["2D_base|2D_ex|t-1"] - q2],
                conv=dict(N48_S16=Q["2D_N48_S16|2D_ex|t0"] - q2, N24_S12=Q["2D_N24_S12|2D_ex|t0"] - q2,
                          h0p001=Q["2D_h0p001|2D_ex|t0"] - q2),
                regress_D108_1D_G=Q["1D|1D_G|t0"] - 4.5985676, regress_S3_Rrot=Q["1D|1D_ex|t0"] - 4.54340)
    wjson("C140_lam0.json", dict(jobs=jobs, Q=Q, test=test))
    status(dict(stage="lam0 done", key_numbers=dict(lam0_test=test)))
    say("lam0 test: " + json.dumps(clean(test)))


def res():
    t0 = time.time()
    L = jload("C140_lam0.json")["Q"]
    off = {"1D_G": 0.0, "2D_G": L["2D_base|2D_G|t0"] - L["1D|1D_G|t0"], "2D_ex": L["2D_base|2D_ex|t0"] - L["1D|1D_G|t0"],
           "2D_ex_noBx": L["2D_base|2D_ex_noBx|t0"] - L["1D|1D_G|t0"], "1D_ex": L["1D|1D_ex|t0"] - L["1D|1D_G|t0"],
           "1D_BO2Dex": L["1D|1D_BO2Dex|t0"] - L["1D|1D_G|t0"], "1D_BO2Dex_lo": L["1D|1D_BO2Dex_lo|t0"] - L["1D|1D_G|t0"]}
    p1 = job(dict(name="1D_G", cfg=ONE, probs=[dict(var="1D_G", match="MC", lam=l, guess=[Q_MC_010 + 5.5 * (l + 0.010),
                                                                                      -0.5 * g_mc(l)]) for l in LAMS]
                  + [dict(var="1D_G", match="MU", lam=l, guess=[Q_MU[l], -0.5 * G_MU[l]]) for l in LAM_MU]))
    base = {(x["match"], x["lam"]): x for x in p1["rows"]}

    def P(var, match, lam, tilt=0.0, fg=0.7):
        b = base[(match, lam)]
        return dict(var=var, match=match, lam=lam, tilt=tilt, guess=[b["Q"] + off[var], -0.5 * fg * b["gamma"]])
    specs = [
        dict(name="1D_rest", cfg=ONE, probs=[P(v, "MC", l) for v in ("1D_ex", "1D_BO2Dex", "1D_BO2Dex_lo") for l in LAMS]
             + [P("1D_ex", "MU", l) for l in LAM_MU] + [P("1D_ex", "MC", l, t) for l in (-0.009, -0.011) for t in (1.0, -1.0)]),
        dict(name="2D_G", cfg=BASE, probs=[P("2D_G", "MC", l) for l in LAMS]),
        dict(name="2D_ex", cfg=BASE, probs=[P("2D_ex", "MC", l, fg=0.5) for l in LAMS]),
        dict(name="2D_ex_noBx", cfg=BASE, probs=[P("2D_ex_noBx", "MC", l, fg=0.4) for l in LAMS]),
        dict(name="MU2D_tilt", cfg=BASE, probs=[P(v, "MU", l, fg=0.5) for v in ("2D_G", "2D_ex") for l in LAM_MU]
             + [P("2D_ex", "MC", l, t, fg=0.5) for l in (-0.009, -0.011) for t in (1.0, -1.0)]),
    ]
    for nm, c in (("conv_N48_S16", dict(BASE, N=48, Smax=16.0)), ("conv_N24_S12", dict(BASE, N=24, Smax=12.0)),
                  ("conv_h0p001", dict(BASE, h=0.001)), ("conv_Rout14", dict(BASE, Rout=14.0)),
                  ("conv_wkb0", dict(BASE, order=0))):
        specs.append(dict(name=nm, cfg=c, probs=[P("2D_ex", "MC", l, fg=0.5) for l in (-0.010, -0.0143)]))
    specs.append(dict(name="conv_1D_h0p001", cfg=dict(ONE, h=0.001), probs=[P("1D_G", "MC", l) for l in (-0.010, -0.0143)]))
    jobs = [p1] + run_specs(specs)
    wjson("C140_res.json", dict(jobs=jobs, offsets_lam0=off, time_s=time.time() - t0))
    status(dict(stage="res done", key_numbers=dict(res_time_s=time.time() - t0)))


def res2d():
    """2D resonances (rerun): windows centred on the BO-bracket interpolation of the lambda = 0 position."""
    t0 = time.time()
    Z = jload("C140_res.json")
    L = jload("C140_lam0.json")["Q"]
    rows = {(x["var"], x["match"], round(x["lam"], 6), x["tilt"]): x for jb in Z["jobs"]
            if jb["name"] in ("1D_G", "1D_rest") for x in jb["rows"]}
    lo0, hi0, ex0 = L["1D|1D_BO2Dex_lo|t0"], L["1D|1D_BO2Dex|t0"], L["2D_base|2D_ex|t0"]
    phi = (ex0 - lo0) / (hi0 - lo0)
    dG, dN = L["2D_base|2D_G|t0"] - ex0, L["2D_base|2D_ex_noBx|t0"] - ex0

    def R_(var, match, lam, tilt=0.0):
        return rows[(var, match, round(lam, 6), tilt)]

    def P(var, match, lam, tilt=0.0):
        lo, hi = R_("1D_BO2Dex_lo", "MC", lam), R_("1D_BO2Dex", "MC", lam)
        c = lo["Q"] + phi * (hi["Q"] - lo["Q"])
        g = math.exp(math.log(lo["gamma"]) + phi * (math.log(hi["gamma"]) - math.log(lo["gamma"])))
        if match == "MU":
            c += R_("1D_ex", "MU", lam)["Q"] - R_("1D_ex", "MC", lam)["Q"]
            g *= R_("1D_ex", "MU", lam)["gamma"] / R_("1D_ex", "MC", lam)["gamma"]
        c += {"2D_G": dG, "2D_ex_noBx": dN}.get(var, 0.0)
        if tilt:
            c += R_("1D_ex", "MC", lam, tilt)["Q"] - R_("1D_ex", "MC", lam)["Q"]
        return dict(var=var, match=match, lam=lam, tilt=tilt, center=c, gimag=g)
    W = (0.05, 0.0025)
    specs = [dict(name="2D_G", cfg=BASE, window=W, probs=[P("2D_G", "MC", l) for l in LAMS]),
             dict(name="2D_ex", cfg=BASE, window=W, probs=[P("2D_ex", "MC", l) for l in LAMS]),
             dict(name="2D_ex_noBx", cfg=BASE, window=W, probs=[P("2D_ex_noBx", "MC", l) for l in LAMS]),
             dict(name="MU2D_tilt", cfg=BASE, window=W, probs=[P(v, "MU", l) for v in ("2D_G", "2D_ex") for l in LAM_MU]
                  + [P("2D_ex", "MC", l, t) for l in (-0.009, -0.011) for t in (1.0, -1.0)])]
    for nm, c in (("conv_N48_S16", dict(BASE, N=48, Smax=16.0)), ("conv_N24_S12", dict(BASE, N=24, Smax=12.0)),
                  ("conv_h0p001", dict(BASE, h=0.001)), ("conv_Rout14", dict(BASE, Rout=14.0)),
                  ("conv_wkb0", dict(BASE, order=0))):
        specs.append(dict(name=nm, cfg=c, window=W, probs=[P("2D_ex", "MC", l) for l in (-0.010, -0.0143)]))
    jobs = run_specs(specs)
    wjson("C140_res2d.json", dict(jobs=jobs, phi_lam0=phi, offsets=dict(dG=dG, dN=dN), time_s=time.time() - t0))
    status(dict(stage="res2d done", key_numbers=dict(res2d_time_s=time.time() - t0)))


def fd_task(args):
    var, h, Rw, Sw, want_vec = args
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from scipy.linalg import eigvalsh_tridiagonal
    t0 = time.time()
    M = Model()
    nr = int(round(Rw / h)) - 1
    r = h * np.arange(1, nr + 1)
    U1, om2 = build(M, var, "L0", 0.0, r)
    if om2 is None:
        e = eigvalsh_tridiagonal(1.0 / h ** 2 + U1, np.full(nr - 1, -0.5 / h ** 2), select="i", select_range=(0, 1))
        return dict(var=var, h=h, Rw=Rw, E=e.tolist(), time_s=time.time() - t0)
    ns = int(round(Sw / h)) - 1
    s = h * np.arange(1, ns + 1)

    def lap(n):
        return sparse.diags([np.full(n - 1, -0.5 / h ** 2), np.full(n, 1.0 / h ** 2), np.full(n - 1, -0.5 / h ** 2)],
                            [-1, 0, 1])
    H = (sparse.kron(lap(nr), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), lap(ns))
         + sparse.diags((U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2).ravel())).tocsc()
    vals, vecs = eigsh(H, k=3, sigma=4.30, which="LM")
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    out = dict(var=var, h=h, Rw=Rw, Sw=Sw, n=nr * ns, E=vals.tolist(), time_s=time.time() - t0)
    if want_vec:
        p2 = vecs[:, 0].reshape(nr, ns) ** 2
        rho = p2.sum(1)
        s2c = (p2 * s[None, :] ** 2).sum(1) / np.maximum(rho, 1e-300)
        omb = np.sqrt(om2)
        dg = {}
        for x in (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
            i = int(np.argmin(np.abs(r - x)))
            dg[f"{r[i]:.2f}"] = dict(s2_cond=float(s2c[i]), s2_BO=float(1.5 / omb[i]),
                                     radial_density=float(rho[i] / (rho.sum() * h)))
        out.update(cond_s2=dg, mean_s2=float((p2 * s[None, :] ** 2).sum() / p2.sum()),
                   mean_r=float((p2 * r[:, None]).sum() / p2.sum()), P_r_lt_0p65=float(rho[r < 0.65].sum() / rho.sum()),
                   mean_s2_BO=float((rho * 1.5 / omb).sum() / rho.sum()))
    return out


def fd2d():
    t0 = time.time()
    tasks = [(v, hh, 9.0, 11.0, (v == "2D_ex" and hh == 0.04)) for v in ("2D_ex", "2D_G") for hh in (0.08, 0.04, 0.02)]
    tasks += [("1D_ex", hh, 16.0, None, False) for hh in (0.004, 0.002, 0.001)]
    with ProcessPoolExecutor(max_workers=3) as ex:
        outs = list(ex.map(fd_task, tasks))
    rich = {}
    for v in ("2D_ex", "2D_G", "1D_ex"):
        e = [min(o["E"]) for o in sorted([o for o in outs if o["var"] == v], key=lambda o: -o["h"])]
        rich[v] = dict(e_h=e, rich_h2h4=(64 * e[2] - 20 * e[1] + e[0]) / 45, rich_h2=(4 * e[2] - e[1]) / 3)
    try:
        L = jload("C140_lam0.json")["Q"]
        rich["2D_ex"]["numerov"] = L["2D_base|2D_ex|t0"]
        rich["2D_G"]["numerov"] = L["2D_base|2D_G|t0"]
        rich["1D_ex"]["numerov"] = L["1D|1D_ex|t0"]
        for v in rich:
            rich[v]["fd_minus_numerov"] = rich[v]["rich_h2h4"] - rich[v]["numerov"]
    except Exception as e:  # noqa: BLE001
        rich["note"] = repr(e)
    wjson("C140_fd2d.json", dict(tasks=outs, richardson=rich, time_s=time.time() - t0))
    status(dict(stage="fd2d done", key_numbers=dict(fd2d={k: v.get("rich_h2h4") for k, v in rich.items()
                                                          if isinstance(v, dict)})))
    say("fd2d: " + json.dumps(clean(rich)))


def lamc(lams, gams):
    o = np.argsort(lams)
    ls, lg = np.asarray(lams)[o], np.log(np.asarray(gams)[o])
    sp = CubicSpline(ls, lg)
    out = {}
    for k, v in (("P1=0.9", -math.log(0.9) / ETA), ("P1=0.5", math.log(2) / ETA)):
        try:
            out[k] = float(brentq(lambda x: sp(x) - math.log(v), ls[0], ls[-1]))
        except ValueError:
            out[k] = None
    return out, sp


def post():
    L = jload("C140_lam0.json")
    Z = jload("C140_res.json")
    jobs_all = {jb["name"]: jb for jb in Z["jobs"]}
    if os.path.exists(os.path.join(RES, "C140_res2d.json")):          # 2D rerun supersedes the first 2D attempt
        jobs_all.update({jb["name"]: jb for jb in jload("C140_res2d.json")["jobs"]})
    rows = [dict(x, job=jb["name"]) for jb in jobs_all.values() for x in jb["rows"]]

    def get(job_, var, match, lam, tilt=0.0):
        for x in rows:
            if x["job"] == job_ and x["var"] == var and x["match"] == match and abs(x["lam"] - lam) < 1e-12 \
                    and abs(x["tilt"] - tilt) < 1e-12:
                return x
        return None
    src = {"1D_G": "1D_G", "1D_ex": "1D_rest", "1D_BO2Dex": "1D_rest", "1D_BO2Dex_lo": "1D_rest", "2D_G": "2D_G",
           "2D_ex": "2D_ex", "2D_ex_noBx": "2D_ex_noBx"}
    T = {v: {l: get(j, v, "MC", l) for l in LAMS} for v, j in src.items()}
    TMU = {v: {l: get("1D_G" if v == "1D_G" else ("1D_rest" if v == "1D_ex" else "MU2D_tilt"), v, "MU", l) for l in LAM_MU}
           for v in ("1D_G", "1D_ex", "2D_G", "2D_ex")}
    flags = [f"{x['job']}|{x['var']}|{x['match']}|{x['lam']}|{x['tilt']}" for x in rows if not x["converged"]
             or not (x["gamma"] > 0)]
    gam = {v: {l: T[v][l]["gamma"] for l in LAMS} for v in T}
    Qs = {v: {l: T[v][l]["Q"] for l in LAMS} for v in T}
    ratios = {}
    for num, den in (("2D_ex", "1D_ex"), ("2D_ex", "1D_G"), ("2D_G", "1D_G"), ("1D_ex", "1D_G"), ("2D_ex_noBx", "1D_G"),
                     ("1D_BO2Dex", "1D_ex"), ("1D_BO2Dex_lo", "1D_G")):
        ratios[f"{num}/{den}"] = {l: gam[num][l] / gam[den][l] for l in LAMS}
    ratios_MU = {f"{a}/{b}": {l: TMU[a][l]["gamma"] / TMU[b][l]["gamma"] for l in LAM_MU}
                 for a, b in (("2D_ex", "1D_ex"), ("2D_ex", "1D_G"), ("2D_G", "1D_G"), ("1D_ex", "1D_G"))}
    lc, spl = {}, {}
    for v in T:
        lc[v], spl[v] = lamc(LAMS, [gam[v][l] for l in LAMS])
    # lambda = 0 calibrated interpolation of the remaining Gaussian Born-Huang term B_x
    Q0 = L["Q"]
    qa, qb = Q0["2D_base|2D_ex|t0"], Q0["2D_base|2D_ex_noBx|t0"]
    sx = (C72_LAM0 - qb) / (qa - qb) if abs(qa - qb) > 1e-9 else None
    if sx is not None:
        gcal = {l: math.exp(math.log(gam["2D_ex_noBx"][l]) + sx * (math.log(gam["2D_ex"][l]) - math.log(gam["2D_ex_noBx"][l])))
                for l in LAMS}
        lc["2D_cal"], spl["2D_cal"] = lamc(LAMS, [gcal[l] for l in LAMS])
        gam["2D_cal"] = gcal
    shifts = {}
    for v in lc:
        shifts[v] = {k: dict(lam_c=lc[v][k], dlam_c=lc[v][k] - lc["1D_G"][k], dmt_static=(lc[v][k] - lc["1D_G"][k]) / DLAM_DMT)
                     for k in lc[v] if lc[v][k] is not None and lc["1D_G"][k] is not None}
    # hazard-consistent translation with the C138 MC+CW P1(m_t) table (alpha_s = 0.1180, xi_H = 1/6, u <= 5)
    with open(C138SUM, encoding="utf-8") as f:
        cs = json.load(f)
    mts = np.array(sorted(float(k) for k in cs["P1_MC_vs_mt"]))
    lnI = np.log([-math.log(cs["P1_MC_vs_mt"][f"{x}"]) for x in mts])
    lamb = np.array([cs["lambda_b_vs_mt"][f"{x}"] for x in mts])
    spI = CubicSpline(mts, lnI)
    haz = {}
    for k, mc in (("P1=0.9", cs["central_MC+CW_mu=1/L"]["P1=0.9"]), ("P1=0.5", cs["central_MC+CW_mu=1/L"]["P1=0.5"])):
        slope = float(spI(mc, 1))
        lb = float(np.interp(mc, mts, lamb))
        haz[k] = dict(mc_C138=mc, dlnI_dmt=slope, lam_b_at_mc=lb)
        for v in ("2D_ex", "2D_G", "1D_ex", "2D_ex_noBx", "2D_cal"):
            if v not in spl:
                continue
            lnrho = float(spl[v](lb) - spl["1D_G"](lb))
            haz[k][v] = dict(ln_ratio=lnrho, ratio=math.exp(lnrho), dmt_hazard=-lnrho / slope, mc_new=mc - lnrho / slope)
    # convergence and tilt (ratios relative to base 2D_ex)
    conv = {}
    for nm in ("conv_N48_S16", "conv_N24_S12", "conv_h0p001", "conv_Rout14", "conv_wkb0"):
        conv[nm] = {}
        for l in (-0.010, -0.0143):
            x = get(nm, "2D_ex", "MC", l)
            if x:
                conv[nm][l] = dict(rel_gamma=x["gamma"] / gam["2D_ex"][l] - 1, dQ=x["Q"] - Qs["2D_ex"][l])
    conv["1D_h0p001"] = {l: get("conv_1D_h0p001", "1D_G", "MC", l)["gamma"] / gam["1D_G"][l] - 1 for l in (-0.010, -0.0143)}
    tilt = {}
    for l in (-0.009, -0.011):
        for t in (1.0, -1.0):
            a2 = get("MU2D_tilt", "2D_ex", "MC", l, t)
            a1 = get("1D_rest", "1D_ex", "MC", l, t)
            tilt[f"{l}|{t:+g}"] = dict(rel_2D=a2["gamma"] / gam["2D_ex"][l] - 1, rel_1D=a1["gamma"] / gam["1D_ex"][l] - 1,
                                       ratio_2D_1Dex=a2["gamma"] / a1["gamma"])
    reg = dict(gamma_1D_G_vs_C138={l: gam["1D_G"][l] / G_MC[l] - 1 for l in LAMS if l in G_MC},
               MU_vs_D108={l: TMU["1D_G"][l]["gamma"] / G_MU[l] - 1 for l in LAM_MU})
    out = dict(lam0=L["test"], gamma=gam, Q=Qs, ratios=ratios, ratios_MU=ratios_MU, lam_c=lc, shifts=shifts,
               hazard=haz, calibration=dict(s_x=sx, Q0_2D_ex=qa, Q0_2D_ex_noBx=qb), convergence=conv, tilt=tilt,
               regression=reg, MU=dict(gamma={v: {l: TMU[v][l]["gamma"] for l in LAM_MU} for v in TMU},
                                       Q={v: {l: TMU[v][l]["Q"] for l in LAM_MU} for v in TMU}),
               nonconverged=flags, eta=ETA, dlam_dmt=DLAM_DMT)
    wjson("C140_summary.json", out)
    status(dict(stage="post done", key_numbers=dict(
        ratios_2Dex_over_1Dex={l: ratios["2D_ex/1D_ex"][l] for l in LAM_REP},
        ratios_2Dex_over_1DG={l: ratios["2D_ex/1D_G"][l] for l in LAM_REP}, lam_c=lc,
        shifts={v: {k: s["dmt_static"] for k, s in shifts[v].items()} for v in shifts},
        hazard={k: {v: haz[k][v]["dmt_hazard"] for v in haz[k] if isinstance(haz[k][v], dict)} for k in haz},
        nonconverged=flags)))
    say("post: ratios " + json.dumps(clean({k: {l: v[l] for l in LAM_REP} for k, v in ratios.items()})))
    say("post: lam_c " + json.dumps(clean(lc)))
    say("post: shifts " + json.dumps(clean({v: {k: s["dmt_static"] for k, s in shifts[v].items()} for v in shifts})))
    say("post: hazard " + json.dumps(clean(haz)))
    say("post: MU ratios " + json.dumps(clean(ratios_MU)))
    say("post: conv " + json.dumps(clean(conv)) + " tilt " + json.dumps(clean(tilt)) + " reg " + json.dumps(clean(reg))
        + " flags " + json.dumps(flags))


def _bracket_P(rows1, L):
    lo0, hi0, ex0 = L["1D|1D_BO2Dex_lo|t0"], L["1D|1D_BO2Dex|t0"], L["2D_base|2D_ex|t0"]
    phi = (ex0 - lo0) / (hi0 - lo0)
    dG, dN = L["2D_base|2D_G|t0"] - ex0, L["2D_base|2D_ex_noBx|t0"] - ex0

    def R_(var, match, lam, tilt=0.0):
        return rows1[(var, match, round(lam, 6), tilt)]

    def P(var, match, lam, tilt=0.0):
        lo, hi = R_("1D_BO2Dex_lo", "MC", lam), R_("1D_BO2Dex", "MC", lam)
        c = lo["Q"] + phi * (hi["Q"] - lo["Q"])
        g = math.exp(math.log(lo["gamma"]) + phi * (math.log(hi["gamma"]) - math.log(lo["gamma"])))
        if match == "MU":
            c += R_("1D_ex", "MU", lam)["Q"] - R_("1D_ex", "MC", lam)["Q"]
            g *= R_("1D_ex", "MU", lam)["gamma"] / R_("1D_ex", "MC", lam)["gamma"]
        c += {"2D_G": dG, "2D_ex_noBx": dN}.get(var, 0.0)
        if tilt:
            c += R_("1D_ex", "MC", lam, tilt)["Q"] - R_("1D_ex", "MC", lam)["Q"]
        return dict(var=var, match=match, lam=lam, tilt=tilt, center=c, gimag=g)
    return P


def ex_parts(M, r, lam):
    """D110 Sec. 3.6 cross term in the 2D frame: E_X = E_X^x(r) + 1/2 Om_X^2(r) |w|^2 (see C140_cross_note.py);
    returns (E_X^x, Om_X^2, E_X^BO = E_X^x + (3/4) Om_X^2 / bt)."""
    C = core()
    g2, gY, f = V11["g2"], V11["gY"], 2.0 / 3
    q = M.q
    r = np.asarray(r, float)
    a, k = np.interp(r, M.rf, q["a"]), np.interp(r, M.rf, q["k"])
    beta, bt = np.interp(r, M.rf, q["beta"]), np.interp(r, M.rf, q["bt"])
    m = C.masses(r, lam, g2, gY, 0.0)
    eps = 1e-6
    h2 = (C.estd(m["mh2"] + eps) - C.estd(m["mh2"] - eps)) / eps
    pW = 2 * C.coarse(lambda x: C.ddelta_db0(g2 * g2 / 4 * x * x), r)
    pZ = 2 * C.coarse(lambda x: C.ddelta_db0((g2 * g2 + gY * gY) / 4 * x * x), r)
    half = f / 8 * ((h2 + pZ) * (gY - g2 * beta) ** 2 + 2 * pW * (gY + g2 * beta) ** 2)
    EXx = 0.5 * f * (h2 + pZ + 2 * pW) * 3.0 / 8 * (2 * g2 * g2 / a + g2 * g2 / k)
    return EXx, 2 * half, EXx + half * 1.5 / bt


_S2C = {}


def s2c_profile(M, h=0.04, Rw=9.0, Sw=11.0):
    """conditional <|w|^2>(r) of the lambda = 0 2D_ex ground state (sparse FD); BO-shaped continuation where rho < 1e-8."""
    if "p" in _S2C:
        return _S2C["p"]
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    nr, ns = int(round(Rw / h)) - 1, int(round(Sw / h)) - 1
    r, s = h * np.arange(1, nr + 1), h * np.arange(1, ns + 1)
    U1, om2 = build(M, "2D_ex", "L0", 0.0, r)

    def lap(n):
        return sparse.diags([np.full(n - 1, -0.5 / h ** 2), np.full(n, 1.0 / h ** 2), np.full(n - 1, -0.5 / h ** 2)],
                            [-1, 0, 1])
    H = (sparse.kron(lap(nr), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), lap(ns))
         + sparse.diags((U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2).ravel())).tocsc()
    vals, vecs = eigsh(H, k=1, sigma=4.30, which="LM")
    p2 = vecs[:, 0].reshape(nr, ns) ** 2
    rho = p2.sum(1)
    s2c = (p2 * s[None, :] ** 2).sum(1) / np.maximum(rho, 1e-300)
    good = rho / rho.max() > 1e-8
    icut = int(np.where(good)[0].max())
    bo = 1.5 / np.sqrt(om2)
    s2c[icut + 1:] = s2c[icut] * bo[icut + 1:] / bo[icut]
    _S2C["p"] = dict(r=r, s2c=s2c, rho=rho, r_cut=float(r[icut]), E0=float(vals[0]))
    return _S2C["p"]


def s2c_fun(M):
    p = s2c_profile(M)
    sp = CubicSpline(p["r"], p["s2c"])
    r0, rN = p["r"][0], p["r"][-1]
    return lambda x: sp(np.clip(np.asarray(x, float), r0, rN))


def res_X():
    """indicative first-order cross term in the 2D frame (2D_exX) and the D110-style 1D comparison (1D_GX_rs1)."""
    t0 = time.time()
    S = jload("C140_summary.json")
    M = Model()
    p = s2c_profile(M)
    w = p["rho"] / p["rho"].sum()
    probs2, probs1, first = [], [], {}
    for l in LAMS:
        EXx, omX2, EXbo = ex_parts(M, p["r"], l)
        d2 = float((w * (EXx + 0.5 * omX2 * p["s2c"])).sum())
        d1 = float((w * ex_parts(M, np.maximum(p["r"], 1.0), l)[2]).sum())
        first[l] = dict(dQ_first_2D=d2, dQ_first_1D_rs1=d1)
        probs2.append(dict(var="2D_exX", match="MC", lam=l, center=S["Q"]["2D_ex"][f"{l}"] + d2,
                           gimag=0.8 * S["gamma"]["2D_ex"][f"{l}"]))
        probs1.append(dict(var="1D_GX_rs1", match="MC", lam=l, guess=[S["Q"]["1D_G"][f"{l}"] + d1,
                                                                    -0.4 * S["gamma"]["1D_G"][f"{l}"]]))
    jobs = run_specs([dict(name="2D_exX", cfg=BASE, window=(0.05, 0.0025), probs=probs2),
                      dict(name="1D_GX_rs1", cfg=ONE, probs=probs1)], nw=2)
    out = dict(first_order_lam0_density=first, s2c_profile=dict(r_cut=p["r_cut"], E0=p["E0"]), jobs=jobs)
    for jb in jobs:
        rows = [x for x in jb["rows"] if x["converged"] and x["gamma"] > 0]
        ref = "2D_ex" if jb["name"] == "2D_exX" else "1D_G"
        ls = [x["lam"] for x in rows]
        if len(ls) >= 4:
            lc, spx = lamc(ls, [x["gamma"] for x in rows])
            lc0 = S["lam_c"][ref]
            out[jb["name"]] = dict(lam_c=lc, ratio_vs_ref={x["lam"]: x["gamma"] / S["gamma"][ref][f"{x['lam']}"] for x in rows},
                                   dlam_c={k: lc[k] - lc0[k] for k in lc if lc[k] is not None},
                                   dmt_static={k: (lc[k] - lc0[k]) / DLAM_DMT for k in lc if lc[k] is not None})
            hz = {}
            xr = sorted(float(x) for x in S["gamma"][ref])
            _, spr = lamc(xr, [S["gamma"][ref][f"{x}"] for x in xr])
            for k in ("P1=0.9", "P1=0.5"):
                hzk = S["hazard"][k]
                lb = hzk["lam_b_at_mc"]
                lnr = float(spx(lb)) - float(spr(lb))
                hz[k] = dict(ln_ratio=lnr, dmt_hazard=-lnr / hzk["dlnI_dmt"])
            out[jb["name"]]["hazard_vs_ref"] = hz
    wjson("C140_crossX.json", out)
    status(dict(stage="crossX done"))
    say("crossX: " + json.dumps(clean({k: v for k, v in out.items() if k != "jobs"})))


def res2d_fix():
    """re-solve the rows of C140_res2d.json that did not converge (window rule fixed) + lambda -> 0- trend points."""
    t0 = time.time()
    D = jload("C140_res2d.json")
    Z = jload("C140_res.json")
    L = jload("C140_lam0.json")["Q"]
    rows1 = {(x["var"], x["match"], round(x["lam"], 6), x["tilt"]): x for jb in Z["jobs"]
             if jb["name"] in ("1D_G", "1D_rest") for x in jb["rows"]}
    tl = [-0.005, -0.006]
    t1 = job(dict(name="1D_trend", cfg=ONE, probs=[
        dict(var=v, match="MC", lam=l, guess=[rows1[(v, "MC", -0.007, 0.0)]["Q"] + 3.0 * (l + 0.007),
                                              -0.5 * rows1[(v, "MC", -0.007, 0.0)]["gamma"] * (0.04 if l == -0.005 else 0.27)])
        for v in ("1D_G", "1D_ex", "1D_BO2Dex", "1D_BO2Dex_lo") for l in tl]))
    for x in t1["rows"]:
        rows1[(x["var"], x["match"], round(x["lam"], 6), x["tilt"])] = x
    P = _bracket_P(rows1, L)
    specs = []
    for jb in D["jobs"]:
        bad = [x for x in jb["rows"] if not (x["converged"] and x["gamma"] > 0)]
        if bad:
            specs.append(dict(name=jb["name"], cfg=jb["cfg"], window=(0.05, 0.0025),
                              probs=[P(x["var"], x["match"], x["lam"], x["tilt"]) for x in bad]))
    specs.append(dict(name="2D_trend", cfg=BASE, window=(0.05, 0.0025), probs=[P("2D_ex", "MC", l) for l in tl]))
    outs = run_specs(specs)
    byname = {jb["name"]: jb for jb in D["jobs"]}
    for o in outs:
        if o["name"] == "2D_trend":
            D["jobs"].append(o)
            continue
        jb = byname[o["name"]]
        for x in o["rows"]:
            for i, y in enumerate(jb["rows"]):
                if y["var"] == x["var"] and y["match"] == x["match"] and abs(y["lam"] - x["lam"]) < 1e-12 \
                        and abs(y["tilt"] - x["tilt"]) < 1e-12:
                    jb["rows"][i] = dict(x, refit="res2d_fix (local-minimum window rule)")
        jb.setdefault("fix_scans", []).append(o["scan"])
    D["jobs"].append(t1)
    D["fix_time_s"] = time.time() - t0
    wjson("C140_res2d.json", D)
    status(dict(stage="res2d_fix done"))


def post2():
    """robust post-processing: only converged rows with gamma > 0 enter (non-converged ones are listed in flags)."""
    L = jload("C140_lam0.json")
    Z = jload("C140_res.json")
    jobs_all = {jb["name"]: jb for jb in Z["jobs"]}
    if os.path.exists(os.path.join(RES, "C140_res2d.json")):
        jobs_all.update({jb["name"]: jb for jb in jload("C140_res2d.json")["jobs"]})
    rows = [dict(x, job=jb["name"]) for jb in jobs_all.values() for x in jb["rows"]]

    def ok(x):
        return x is not None and x["converged"] and x["gamma"] > 0

    def get(job_, var, match, lam, tilt=0.0):
        for x in rows:
            if x["job"] == job_ and x["var"] == var and x["match"] == match and abs(x["lam"] - lam) < 1e-12 \
                    and abs(x["tilt"] - tilt) < 1e-12 and ok(x):
                return x
        return None
    src = {"1D_G": "1D_G", "1D_ex": "1D_rest", "1D_BO2Dex": "1D_rest", "1D_BO2Dex_lo": "1D_rest", "2D_G": "2D_G",
           "2D_ex": "2D_ex", "2D_ex_noBx": "2D_ex_noBx"}
    T = {v: {l: get(j, v, "MC", l) for l in LAMS} for v, j in src.items()}
    T = {v: {l: x for l, x in d.items() if x} for v, d in T.items()}
    mu_src = {"1D_G": "1D_G", "1D_ex": "1D_rest", "2D_G": "MU2D_tilt", "2D_ex": "MU2D_tilt"}
    TMU = {v: {l: get(j, v, "MU", l) for l in LAM_MU} for v, j in mu_src.items()}
    TMU = {v: {l: x for l, x in d.items() if x} for v, d in TMU.items()}
    flags = [f"{x['job']}|{x['var']}|{x['match']}|{x['lam']}|{x['tilt']}" for x in rows if not ok(x)]
    gam = {v: {l: T[v][l]["gamma"] for l in T[v]} for v in T}
    Qs = {v: {l: T[v][l]["Q"] for l in T[v]} for v in T}
    gMU = {v: {l: TMU[v][l]["gamma"] for l in TMU[v]} for v in TMU}

    def rat(a, b, G, lams):
        return {l: G[a][l] / G[b][l] for l in lams if l in G[a] and l in G[b]}
    pairs = (("2D_ex", "1D_ex"), ("2D_ex", "1D_G"), ("2D_G", "1D_G"), ("1D_ex", "1D_G"), ("2D_ex_noBx", "1D_G"),
             ("1D_BO2Dex", "1D_ex"), ("1D_BO2Dex_lo", "1D_G"), ("2D_G", "1D_ex"))
    ratios = {f"{a}/{b}": rat(a, b, gam, LAMS) for a, b in pairs}
    ratios_MU = {f"{a}/{b}": rat(a, b, gMU, LAM_MU) for a, b in pairs[:4]}
    lc, spl = {}, {}
    for v in gam:
        ls = sorted(gam[v])
        if len(ls) >= 4:
            lc[v], spl[v] = lamc(ls, [gam[v][l] for l in ls])
    Q0 = L["Q"]
    qa, qb = Q0["2D_base|2D_ex|t0"], Q0["2D_base|2D_ex_noBx|t0"]
    sx = (C72_LAM0 - qb) / (qa - qb)
    common = [l for l in LAMS if l in gam["2D_ex"] and l in gam["2D_ex_noBx"]]
    if len(common) >= 4:
        gcal = {l: math.exp(math.log(gam["2D_ex_noBx"][l]) + sx * (math.log(gam["2D_ex"][l])
                                                                     - math.log(gam["2D_ex_noBx"][l]))) for l in common}
        gam["2D_cal"] = gcal
        lc["2D_cal"], spl["2D_cal"] = lamc(common, [gcal[l] for l in common])
        ratios["2D_cal/1D_G"] = rat("2D_cal", "1D_G", gam, LAMS)
    shifts = {v: {k: dict(lam_c=lc[v][k], dlam_c=lc[v][k] - lc["1D_G"][k], dmt_static=(lc[v][k] - lc["1D_G"][k]) / DLAM_DMT)
                  for k in lc[v] if lc[v][k] is not None and lc["1D_G"].get(k) is not None} for v in lc}
    with open(C138SUM, encoding="utf-8") as f:
        cs = json.load(f)
    mts = np.array(sorted(float(k) for k in cs["P1_MC_vs_mt"]))
    spI = CubicSpline(mts, np.log([-math.log(cs["P1_MC_vs_mt"][f"{x}"]) for x in mts]))
    lamb = np.array([cs["lambda_b_vs_mt"][f"{x}"] for x in mts])
    haz = {}
    for k in ("P1=0.9", "P1=0.5"):
        mc = cs["central_MC+CW_mu=1/L"][k]
        slope, lb = float(spI(mc, 1)), float(np.interp(mc, mts, lamb))
        haz[k] = dict(mc_C138=mc, dlnI_dmt=slope, lam_b_at_mc=lb)
        for v in ("2D_ex", "2D_G", "1D_ex", "2D_ex_noBx", "2D_cal", "1D_BO2Dex_lo"):
            if v in spl:
                lnr = float(spl[v](lb) - spl["1D_G"](lb))
                haz[k][v] = dict(ln_ratio=lnr, ratio=math.exp(lnr), dmt_hazard=-lnr / slope, mc_new=mc - lnr / slope)
    conv = {}
    for nm in ("conv_N48_S16", "conv_N24_S12", "conv_h0p001", "conv_Rout14", "conv_wkb0"):
        conv[nm] = {}
        for l in (-0.010, -0.0143):
            x = get(nm, "2D_ex", "MC", l)
            if x and l in gam["2D_ex"]:
                conv[nm][l] = dict(rel_gamma=x["gamma"] / gam["2D_ex"][l] - 1, dQ=x["Q"] - Qs["2D_ex"][l])
    y1, y2 = get("conv_1D_h0p001", "1D_G", "MC", -0.010), get("conv_1D_h0p001", "1D_G", "MC", -0.0143)
    conv["1D_h0p001"] = [y1["gamma"] / gam["1D_G"][-0.010] - 1 if y1 else None,
                         y2["gamma"] / gam["1D_G"][-0.0143] - 1 if y2 else None]
    tilt = {}
    for l in (-0.009, -0.011):
        for t in (1.0, -1.0):
            a2, a1 = get("MU2D_tilt", "2D_ex", "MC", l, t), get("1D_rest", "1D_ex", "MC", l, t)
            if a2 and a1 and l in gam["2D_ex"]:
                tilt[f"{l}|{t:+g}"] = dict(rel_2D=a2["gamma"] / gam["2D_ex"][l] - 1, rel_1D=a1["gamma"] / gam["1D_ex"][l] - 1,
                                           ratio_2D_1Dex=a2["gamma"] / a1["gamma"])
    trend = {}
    for l in (-0.005, -0.006, -0.007):
        a = get("2D_trend", "2D_ex", "MC", l) or get("2D_ex", "2D_ex", "MC", l)
        b = get("1D_trend", "1D_G", "MC", l) or get("1D_G", "1D_G", "MC", l)
        c = get("1D_trend", "1D_ex", "MC", l) or get("1D_rest", "1D_ex", "MC", l)
        if a and b:
            trend[l] = dict(Q_2D=a["Q"], g_2D=a["gamma"], g_1DG=b["gamma"], g_1Dex=c["gamma"] if c else None,
                            ratio_2D_1DG=a["gamma"] / b["gamma"], ratio_2D_1Dex=a["gamma"] / c["gamma"] if c else None)
    out = dict(lam0=L["test"], gamma=gam, Q=Qs, ratios=ratios, ratios_MU=ratios_MU, lam_c=lc, shifts=shifts,
               hazard=haz, calibration=dict(s_x=sx, Q0_2D_ex=qa, Q0_2D_ex_noBx=qb), convergence=conv, tilt=tilt,
               trend_lam_to_0=trend, MU=dict(gamma=gMU, Q={v: {l: TMU[v][l]["Q"] for l in TMU[v]} for v in TMU}),
               regression=dict(gamma_1D_G_vs_C138={l: gam["1D_G"][l] / G_MC[l] - 1 for l in LAMS if l in G_MC and l in gam["1D_G"]}),
               nonconverged_superseded_or_open=flags, eta=ETA, dlam_dmt=DLAM_DMT)
    wjson("C140_summary.json", out)
    status(dict(stage="post done", key_numbers=dict(
        ratios_2Dex_over_1Dex={l: ratios["2D_ex/1D_ex"].get(l) for l in LAM_REP},
        ratios_2Dex_over_1DG={l: ratios["2D_ex/1D_G"].get(l) for l in LAM_REP}, lam_c=lc,
        shifts={v: {k: s["dmt_static"] for k, s in shifts[v].items()} for v in shifts},
        hazard={k: {v: haz[k][v]["dmt_hazard"] for v in haz[k] if isinstance(haz[k][v], dict)} for k in haz},
        note="first-pass 2D rows (C140_res.json) were non-converged broad box states and are superseded by C140_res2d.json")))
    for k, v in (("ratios", {a: {l: b.get(l) for l in LAM_REP} for a, b in ratios.items()}), ("MU", ratios_MU), ("lam_c", lc),
                 ("shifts", {v: {k2: s["dmt_static"] for k2, s in shifts[v].items()} for v in shifts}), ("hazard", haz),
                 ("conv", conv), ("tilt", tilt), ("trend", trend), ("calib_sx", sx), ("flags", flags)):
        say("post2 %s: %s" % (k, json.dumps(clean(v))))


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "prep"
    t_start = time.time()
    {"prep": prep, "lam0": lam0, "res": res, "res2d": res2d, "res2d_fix": res2d_fix, "fd2d": fd2d, "post": post2,
     "post_old": post, "crossX": res_X}[ph]()
    say("phase %s finished in %.1fs" % (ph, time.time() - t_start))
