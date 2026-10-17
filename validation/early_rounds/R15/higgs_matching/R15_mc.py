# -*- coding: utf-8 -*-
"""R15 independent check of D110 (MC+CW consistent one-loop finite-volume matching).  Own code.

New parts written here (nothing taken from the D110 producer code, which was not opened before these numbers):
  * Coulomb-gauge non-zero-mode spectrum: per massive vector 2 transverse modes omega^2 = k^2 + a and one
    Coulomb/Goldstone mode omega_C^2 = (k^2 + a)(k^2 + b)/k^2 (A_0 eliminated by the Gauss law), b = V'_tree/r.
  * MS-bar finite-volume energy of the Coulomb mode by a basis-free decomposition
        E_C(a,b) = R_C(a,b) + (T_B/2)(a+b) + K3(mu) (a-b)^2,
    R_C = sum'_k 1/2 [omega_C - k - (a+b)/2k + (a-b)^2/8k^3]  (absolutely convergent; cancellation-free form),
    K3 = c2 - (ln mu^2 L^2 + 3/2)/(64 pi^2) = m^4 coefficient of E_std = D_theta + V_CW(m^2; 3/2, mu).
  * mu = c/L variation (couplings at c mu, Higgs rescaling xi = exp(-gamma_phi^C ln c)); RG residual check.
  * Born-Huang (adiabatic) estimate inside MC with bosonic and Dirac-pair (fermionic) matrix elements.
Reused read-only (independent of the D110 producer): R14_fv (D_theta), R14_res (conditional Gaussian, Numerov +
Siegert, Muller), R14_rge (2-loop SM RGE + 3-loop calibration) from the round-1 xi_scan folder.
Usage:  python -B R15_mc.py checks | static | hazard
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
sys.dont_write_bytecode = True
import json
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
R14DIR = (r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\04_calculations\code"
          r"\20261013_claude_v12\higgs_survival\xi_scan")
sys.path.insert(0, R14DIR)
import R14_fv  # noqa: E402
import R14_rge  # noqa: E402
import R14_res as R  # noqa: E402

PI = math.pi
FVF = R.FVF
TB2 = FVF.ZP12 / (8 * PI)
TF2 = FVF.ZA12 / (8 * PI)
C2P = FVF.c2[0]
C64 = 1.0 / (64 * PI ** 2)
NSH = 60
QP, NPm = R14_fv.shells_int(NSH)
KP = 2 * PI * np.sqrt(QP)
NEFFP = (3 * (NPm.sum() + 1) / (4 * PI)) ** (1 / 3)
QA, NAm = R14_fv.shells_half(NSH)
KA = 2 * PI * np.sqrt(QA)
NEFFA = (3 * NAm.sum() / (4 * PI)) ** (1 / 3)
TAIL4P = (2 * PI) ** -4 * 4 * PI / NEFFP
TAIL4A = (2 * PI) ** -4 * 4 * PI / NEFFA
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
OUT = os.path.join(HERE, "R15_results.json")
RCG = np.arange(0.005, 16.2, 0.01)          # coarse grid for the matching potentials (spline to Numerov grid)
GC1 = -math.log(0.9) / R.ETA_TOT
GC5 = math.log(2.0) / R.ETA_TOT


def merge(key, obj):
    R14_rge.merge_json(OUT.replace(".json", f"_{key}.json"), key, obj)


# ------------------------------------------------------------------ MS-bar finite-volume building blocks
def vcw(m2, c, l2=0.0):
    m2 = np.asarray(m2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(m2 != 0, np.log(np.abs(m2)), 0.0)
    return m2 * m2 * (lg - l2 - c) * C64


def estd(m2, l2=0.0):
    m2 = np.atleast_1d(np.asarray(m2, float))
    return FVF.DP(m2) + vcw(m2, 1.5, l2)


def estdA(m2, l2=0.0):
    m2 = np.atleast_1d(np.asarray(m2, float))
    return FVF.DA(m2) + vcw(m2, 1.5, l2)


def K3(l2=0.0):
    return C2P - (l2 + 1.5) * C64


def _g(x, k):
    """k*[sqrt(1+x) - 1 - x/2 + x^2/8] without cancellation."""
    s = np.sqrt(1.0 + x)
    return k * x ** 3 * (s + 3) / (8 * (1 + s) ** 3)


def RC(a, b, chunk=96):
    """sum'_k 1/2 [omega_C - k - (a+b)/2k + (a-b)^2/8k^3], periodic k = 2 pi n, n != 0."""
    a = np.atleast_1d(np.asarray(a, float)).ravel()
    b = np.broadcast_to(np.asarray(b, float), a.shape).ravel()
    out = np.empty(a.size)
    k = KP[None, :]
    k2 = k * k
    for i in range(0, a.size, chunk):
        aa = a[i:i + chunk, None]
        bb = b[i:i + chunk, None]
        x = (aa + bb) / k2 + aa * bb / (k2 * k2)
        t = -(aa + bb) * aa * bb / (4 * k2 * k2 * k) - (aa * bb) ** 2 / (8 * k2 ** 3 * k) + _g(x, k)
        A_, B_ = aa[:, 0], bb[:, 0]
        tail = (A_ + B_) * (A_ - B_) ** 2 / 16.0 * (2 * PI) ** -5 * 2 * PI / NEFFP ** 2
        out[i:i + chunk] = 0.5 * (t @ NPm + tail)
    return out


def EC(a, b, l2=0.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    return RC(a, b) + TB2 * (a + b) + K3(l2) * (a - b) ** 2


def delta_basis(a, b, p=2.0, l2=0.0):
    """D110-type cross-check: scalar basis (b; a; p a) with weights fixed by the k^-1, k^-3 moments."""
    wp = -2 * b / (a * p * (p - 1))
    wa = 2 * b / (a * (p - 1))
    k = KP
    xC = (a + b) / k ** 2 + a * b / k ** 4
    tC = -(a + b) * a * b / (4 * k ** 5) - (a * b) ** 2 / (8 * k ** 7) + _g(xC, k)
    t = tC - (1 + wa) * _g(a / k ** 2, k) - _g(b / k ** 2, k) - wp * _g(p * a / k ** 2, k)
    lead = (a + b) * (a - b) ** 2 / 16 - (1 + wa) * a ** 3 / 16 - b ** 3 / 16 - wp * (p * a) ** 3 / 16
    tail = lead * (2 * PI) ** -5 * 2 * PI / NEFFP ** 2
    Rp = 0.5 * (float(t @ NPm) + tail)
    return Rp + estd(b, l2)[0] + wa * estd(a, l2)[0] + wp * estd(p * a, l2)[0]


def dbDelta(a, l2=0.0, h=1e-5):
    return (EC(a, h, l2) - EC(a, -h, l2)) / (2 * h)


def masses(r, lam, g2, gY, yt):
    r2 = r * r
    return g2 * g2 * r2 / 4, (g2 * g2 + gY * gY) * r2 / 4, lam * r2, 3 * lam * r2, yt * yt * r2 / 2


def bo(r, lam, g2, gY, yt, chunk=96):
    """Born-Huang diagonal term sum (1/16)(d ln(Z omega)/dr)^2 over k != 0 fast modes (MC spectrum).
    Returns bosonic part, top with the bosonic formula (D110 style), top with the Dirac-pair matrix element
    (per real dof (m/r)^2 k^2/(16 omega^4), r-independent UV constant removed)."""
    r = np.atleast_1d(np.asarray(r, float))
    aW, aZ, b, mh2, mt2 = masses(r, lam, g2, gY, yt)
    out = dict(bos=np.empty(r.size), topb=np.empty(r.size), topf=np.empty(r.size))
    kp2 = (KP ** 2)[None, :]
    ka2 = (KA ** 2)[None, :]
    for i in range(0, r.size, chunk):
        sl = slice(i, i + chunk)
        rr = r[sl, None]
        W, Z, B, H, T = aW[sl, None], aZ[sl, None], b[sl, None], mh2[sl, None], mt2[sl, None]

        def sq(m2):
            return ((m2 / rr) / (kp2 + m2)) ** 2

        def dc(a):
            return (B / rr) / (kp2 + B) - (a / rr) / (kp2 + a)
        s_bos = 4 * sq(W) + 2 * sq(Z) + 2 * dc(W) ** 2 + dc(Z) ** 2 + sq(H)
        lead = (4 * W ** 2 + 2 * Z ** 2 + 2 * (B - W) ** 2 + (B - Z) ** 2 + H ** 2)[:, 0] / rr[:, 0] ** 2
        out["bos"][sl] = (s_bos @ NPm + lead * TAIL4P) / 16
        tb = ((T / rr) / (ka2 + T)) ** 2
        out["topb"][sl] = 12 * (tb @ NAm + (T[:, 0] / rr[:, 0]) ** 2 * TAIL4A) / 16
        tf = (T / rr ** 2) * (ka2 / (ka2 + T) ** 2 - 1.0 / ka2)
        out["topf"][sl] = 12 * (tf @ NAm + (T[:, 0] / rr[:, 0] ** 2) * (-2 * T[:, 0]) * TAIL4A) / 16
    return out


def dV(model, r, lam, g2, gY, yt, l2=0.0):
    r = np.atleast_1d(np.asarray(r, float))
    aW, aZ, b, mh2, mt2 = masses(r, lam, g2, gY, yt)
    if model == "MU":
        return 6 * FVF.DP(aW) + 3 * FVF.DP(aZ) - 12 * FVF.DA(mt2) + FVF.DP(mh2)
    base = estd(mh2, l2) - 12 * estdA(mt2, l2)

    def vu(a):
        return 3 * estd(a, l2) + 2 * a * a * C64
    if model == "MU_CW":
        return 2 * vu(aW) + vu(aZ) + base
    if model == "ML_CW":
        return 2 * vu(aW) + vu(aZ) + base + 3 * estd(b, l2)
    if model == "MC1":
        return 2 * (vu(aW) + b * dbDelta(aW, l2)) + vu(aZ) + b * dbDelta(aZ, l2) + base

    def vc(a):
        return 2 * estd(a, l2) + 2 * a * a * C64 + EC(a, b, l2)
    v = 2 * vc(aW) + vc(aZ) + base
    if model == "MC":
        return v
    B = bo(r, lam, g2, gY, yt)
    if model == "MC_BHd":
        return v + B["bos"] + B["topb"]
    if model == "MC_BHf":
        return v + B["bos"] + B["topf"]
    raise ValueError(model)


def gam_phi(g2, gY, yt):
    return ((3 * yt ** 2 - 1.5 * g2 ** 2 - 0.5 * gY ** 2) / (16 * PI ** 2),
            (3 * yt ** 2 - 2.25 * g2 ** 2 - 0.75 * gY ** 2) / (16 * PI ** 2))


# ------------------------------------------------------------------ radial problem helpers
def prof_task(args):
    g2, gY = args
    gp = R.gauss_profile(g2, gY)
    return gp["rf"], gp["rf"] ** 2 * (gp["Et"] + gp["Br"]), gp["beta_g"], gp["alpha_g"]


def dv_task(specs):
    return np.array([dV(s[0], s[6] * RCG, s[1], s[2], s[3], s[4], s[5]) for s in specs])


class Prof:
    def __init__(self, t):
        self.rf, vals, self.bg, self.ag = t
        self.spl = CubicSpline(self.rf, vals)

    def col(self, r, lam, dvc, xi=1.0):
        rx = xi * r
        return 3.0 / (8 * r * r) + self.spl(rx) / (rx * rx) + lam * rx ** 4 / 4 + CubicSpline(RCG, dvc)(r)

    def sa(self, xi=1.0):
        beta = 3.0 / 8 + self.bg / xi ** 2
        return 0.5 + math.sqrt(0.25 + 2 * beta), self.ag / xi


# ------------------------------------------------------------------ mode: checks
def checks():
    t0 = time.time()
    res = {}
    res["consts"] = dict(TB2=TB2, TF2=TF2, c2P=C2P, c2A=FVF.c2[1], K3_mu_1overL=K3(0.0), minus_1_64pi2=-C64,
                         muStarL=math.exp((64 * PI ** 2 * C2P - 1.5) / 2), R14_fit_resid=FVF.fitres,
                         NEFFP=NEFFP, NEFFA=NEFFA)
    m2s = [-25.0, -10.0, -4.0, -1.0, -0.25, 0.25, 1.0, 2.56, 4.0, 6.25, 9.0, 16.0, 25.0, 49.0, 100.0, 196.0]
    k3 = [float((estd(m)[0] - RC(m, 0.0)[0] - TB2 * m) / m ** 2) for m in m2s]
    xs = [4.0, 6.25, 9.0, 16.0, 25.0, 49.0, 100.0, 196.0]
    ys = [float((FVF.DP(m)[0] - RC(m, 0.0)[0] - TB2 * m) / m ** 2) for m in xs]
    sl, ic = np.polyfit(np.log(xs), ys, 1)
    fit_res = float(np.max(np.abs(np.polyval([sl, ic], np.log(xs)) - ys)))
    res["H1_log_cancellation"] = dict(
        m2=m2s, K3_extracted=k3, K3_expected=K3(0.0), max_abs_dev=float(np.max(np.abs(np.array(k3) - K3(0.0)))),
        Dtheta_bessel_region_log_slope=float(sl), expected_slope=-C64, rel_err_slope=float(sl / -C64 - 1),
        c2_from_intercept=float(ic), c2_R14=C2P, fit_resid=fit_res)
    # Coulomb mode tests
    pts = [(0.5, -0.02), (2.0, -0.09), (2.9637, -0.25), (6.0, -0.5), (1.0, 0.3), (12.0, -1.0)]
    tab = []
    for a, b in pts:
        dm = float(EC(a, b)[0] - estd(a)[0])
        h = 1e-3
        dmu = float((EC(a, b, h)[0] - estd(a, h)[0] - EC(a, b, -h)[0] + estd(a, -h)[0]) / (2 * h))
        tab.append(dict(a=a, b=b, Delta=dm, Delta_basis_p2=float(delta_basis(a, b, 2.0)),
                        Delta_basis_p3=float(delta_basis(a, b, 3.0)), Delta_basis_p1p5=float(delta_basis(a, b, 1.5)),
                        dDelta_dlnmu2=dmu, expected=-(b * b - 2 * a * b) * C64,
                        EC_b0_minus_Estd=float(EC(a, 0.0)[0] - estd(a)[0])))
    res["coulomb_mode"] = tab
    c = R14_rge.couplings(172.60, 0.1180)
    res["SM_couplings_mu_b"] = c
    g2, gY, yt = c["g2"], c["gY"], c["yt"]
    rc = {}
    for r in (0.01, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        aW, aZ, _, _, _ = masses(r, -0.01, g2, gY, yt)
        rc[str(r)] = float(2 * dbDelta(aW)[0] + dbDelta(aZ)[0])
    res["rC_SM"] = dict(values=rc, limit_3TB2=3 * TB2)
    # RG residual (lambda=-0.010, SM couplings, mu = 1/L)
    lam = -0.010
    gC, gL = gam_phi(g2, gY, yt)
    b1 = float(R14_rge.beta(np.array([gY, g2, c["g3"], yt, lam]), nl=1)[4])
    rr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rg = {}
    for model, gname, gv in (("MC", "C", gC), ("MC", "L", gL), ("ML_CW", "L", gL), ("MU_CW", "L", gL),
                             ("MU_CW", "C", gC)):
        h = 1e-3
        mud = 2 * (dV(model, rr, lam, g2, gY, yt, h) - dV(model, rr, lam, g2, gY, yt, -h)) / (2 * h)
        resid = mud + b1 * rr ** 4 / 4 - gv * lam * rr ** 4
        rg[f"{model}+gamma_{gname}"] = dict(resid_over_lam_r4_4=(resid / (lam * rr ** 4 / 4)).tolist())
    res["RG_residual"] = dict(gamma_C=gC, gamma_L=gL, beta1_lambda=b1, table=rg,
                              expected_MU_CW_L=3 * lam / (8 * PI ** 2))
    # masses at r=5 vs 2 pi (MU_RG claim)
    aW, aZ, b, mh2, mt2 = masses(5.0, -0.01, g2, gY, yt)
    res["masses_r5_SM"] = dict(mW=math.sqrt(aW), mZ=math.sqrt(aZ), mt=math.sqrt(mt2), r_mZ_eq_2pi=4 * PI /
                               math.sqrt(g2 * g2 + gY * gY), two_pi=2 * PI)
    # Born-Huang estimate inside MC
    B = bo(np.array([1.5, 3.0, 5.0]), lam, g2, gY, yt)
    res["BO_estimate_SM_lam-0.010"] = dict(
        r=[1.5, 3.0, 5.0], bosonic=B["bos"].tolist(), top_bosonic_formula=B["topb"].tolist(),
        top_dirac_pairs=B["topf"].tolist(), D110_style_total=(B["bos"] + B["topb"]).tolist(),
        fermionic_total=(B["bos"] + B["topf"]).tolist(),
        D110_claim=[2.6e-4, 1.0e-3, 2.8e-3])
    # Goldstone first-order coefficient for comparison (b * sum dbDelta / r^2)
    aW, aZ, _, _, _ = masses(1.0, lam, g2, gY, yt)
    res["goldstone_first_order_coef_r2"] = float(lam * (2 * dbDelta(aW)[0] + dbDelta(aZ)[0]))
    res["time_s"] = time.time() - t0
    merge("checks", res)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ mode: static
def static():
    t0 = time.time()
    c = R14_rge.couplings(172.60, 0.1180)
    sets = {"v11": (V11["g2"], V11["gY"], V11["yt"]), "SM": (c["g2"], c["gY"], c["yt"])}
    lams = [-0.006, -0.007, -0.008, -0.009, -0.010, -0.011, -0.012, -0.013]
    models = ["MC", "MU", "MU_CW", "ML_CW", "MC1"]
    r, m = R.grid(16.0)
    cols, meta, S, A = [], [], [], []
    for key, (g2, gY, yt) in sets.items():
        P = Prof(prof_task((g2, gY)))
        s_, a_ = P.sa()
        for model in models:
            for lam in lams:
                cols.append(P.col(r, lam, dV(model, RCG, lam, g2, gY, yt)))
                meta.append((key, model, lam))
                S.append(s_)
                A.append(a_)
    E, it, rs = R.solve_set(cols, r, m, np.array(S), np.array(A))
    gam = -2 * E.imag
    out = dict(muller_iter=int(it), max_resid=float(np.max(rs)), gamma={}, Q={}, lam_c={})
    for key in sets:
        for model in models:
            idx = [j for j, mm in enumerate(meta) if mm[0] == key and mm[1] == model]
            g = gam[idx]
            out["gamma"][f"{key}|{model}"] = dict(zip([str(x) for x in lams], g.tolist()))
            out["Q"][f"{key}|{model}"] = dict(zip([str(x) for x in lams], E.real[idx].tolist()))
            la = np.array(lams)[::-1]
            spl = CubicSpline(la, np.log(g[::-1]))
            lc = {}
            for name, gc in (("P1=0.9", GC1), ("P1=0.5", GC5)):
                try:
                    lc[name] = float(brentq(lambda x: spl(x) - math.log(gc), la[0], la[-1]))
                except ValueError:
                    lc[name] = None
            out["lam_c"][f"{key}|{model}"] = lc
    out["time_s"] = time.time() - t0
    merge("static", out)
    for k in out["gamma"]:
        print(k, "gamma(-0.010)=%.6e" % out["gamma"][k]["-0.01"], "lam_c", out["lam_c"][k])
    print("time", out["time_s"])


# ------------------------------------------------------------------ mode: hazard
MT_C = [171.8, 172.0, 172.2, 172.4, 172.6, 172.8, 173.0]
MT_LO = [171.6, 171.8, 172.0, 172.2, 172.4, 172.6]
MT_HI = [172.4, 172.6, 172.8, 173.0, 173.2, 173.4]
CVALS = [0.5, 2.0, 2 * PI]


def hazard_specs():
    sp = []
    for model in ("MC", "ML_CW", "MU_CW", "MU", "MC_BHd", "MC_BHf"):
        for mt in MT_C:
            sp.append(dict(group=f"{model}|0.1180|c=1", model=model, mt=mt, als=0.1180, c=1.0))
    for cc in CVALS:
        for mt in MT_C:
            sp.append(dict(group=f"MC|0.1180|c={cc:.4f}", model="MC", mt=mt, als=0.1180, c=cc))
    for mt in MT_LO:
        sp.append(dict(group="MC|0.1171|c=1", model="MC", mt=mt, als=0.1171, c=1.0))
    for mt in MT_HI:
        sp.append(dict(group="MC|0.1189|c=1", model="MC", mt=mt, als=0.1189, c=1.0))
    return sp


def hazard():
    import multiprocessing as mp
    t0 = time.time()
    cal, _ = R14_rge.calibration()
    un = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
    mus = R14_rge.MU_B / np.sqrt(np.cosh(2 * un))
    specs = hazard_specs()
    combos = sorted(set((s["mt"], s["als"]) for s in specs))
    sols = {cb: R14_rge.solve(cb[0], cb[1], 125.13) for cb in combos}
    solc = R14_rge.solve(172.6, 0.1180, 125.13)
    cs = [1.0] + CVALS
    Yc = {cc: solc.sol(np.log(cc * mus)) for cc in cs}
    Ys = {(cb, cc): sols[cb].sol(np.log(cc * mus)) for cb in combos for cc in cs}
    ptasks = [(float(Yc[cc][1, i]), float(Yc[cc][0, i])) for cc in cs for i in range(len(un))]
    ctx = mp.get_context("spawn")
    with ctx.Pool(5) as pool:
        praw = pool.map(prof_task, ptasks)
        PR = {}
        j = 0
        for cc in cs:
            for i in range(len(un)):
                PR[(cc, i)] = Prof(praw[j])
                j += 1
        print("profiles done", time.time() - t0, flush=True)
        node_specs, node_par = [], []
        for i in range(len(un)):
            lst, par = [], []
            for s in specs:
                cb, cc = (s["mt"], s["als"]), s["c"]
                lam = float(Ys[(cb, cc)][4, i] + cal)
                yt = float(Ys[(cb, cc)][3, i])
                g2, gY = float(Yc[cc][1, i]), float(Yc[cc][0, i])
                gC = gam_phi(float(Yc[1.0][1, i]), float(Yc[1.0][0, i]), float(Ys[(cb, 1.0)][3, i]))[0]
                xi = math.exp(-gC * math.log(cc))
                l2 = 2 * math.log(cc)
                lst.append((s["model"], lam, g2, gY, yt, l2, xi))
                par.append((lam, xi))
            node_specs.append(lst)
            node_par.append(par)
        dvs = pool.map(dv_task, node_specs)
    print("matching potentials done", time.time() - t0, flush=True)
    r, m = R.grid(16.0)
    ns = len(specs)
    G = np.zeros((ns, len(un)))
    Q = np.zeros_like(G)
    guess, prev = None, None
    diag = []
    for i in range(len(un)):
        cols, S, A = [], [], []
        for j, s in enumerate(specs):
            lam, xi = node_par[i][j]
            P = PR[(s["c"], i)]
            cols.append(P.col(r, lam, dvs[i][j], xi))
            s_, a_ = P.sa(xi)
            S.append(s_)
            A.append(a_)
        cols = np.array(cols)
        S, A = np.array(S), np.array(A)
        E, it, rs = R.solve_set(list(cols), r, m, S, A, guess=guess)
        if guess is None:
            bad = ~(np.isfinite(E) & (E.real > 4.25) & (E.real < 4.5) & (-2 * E.imag > 0) & (-2 * E.imag < 0.2))
            if bad.any():
                E2, _, _ = R.solve_set(list(cols[bad]), r, m, S[bad], A[bad], guess=np.full(bad.sum(), 4.39 - 5e-4j))
                E[bad] = E2
                diag.append(dict(u=float(un[i]), node0_retry=int(bad.sum())))
        else:
            g_new = np.maximum(-2 * E.imag, 1e-300)
            g_old = np.maximum(-2 * guess.imag, 1e-300)
            bad = ~(np.isfinite(E) & (np.abs(E.real - guess.real) < 0.25) & (-2 * E.imag > -1e-9)
                    & (np.abs(np.log(g_new / g_old)) < 3.5))
            if bad.any():
                Eh = guess[bad].copy()
                for t in np.linspace(0.125, 1.0, 8):
                    colsh = (1 - t) * prev[bad] + t * cols[bad]
                    Eh, _, _ = R.solve_set(list(colsh), r, m, S[bad], A[bad], guess=Eh)
                diag.append(dict(u=float(un[i]), homotopy=int(bad.sum()), first=[str(x) for x in E[bad][:3]],
                                 after=[str(x) for x in Eh[:3]]))
                E[bad] = Eh
        guess = E.copy()
        prev = cols
        G[:, i] = -2 * E.imag
        Q[:, i] = E.real
        print("node u=%.3f done  t=%.0f  it=%d  maxresid=%.1e" % (un[i], time.time() - t0, it, np.max(rs)), flush=True)
    uf = np.linspace(0, 5.0, 10001)
    wf = 1.0 / (R.HLB * np.sqrt(np.cosh(2 * uf)))
    Iv = np.zeros(ns)
    for j in range(ns):
        spl = CubicSpline(un, np.log(np.maximum(G[j], 1e-300)))
        Iv[j] = simpson(np.exp(spl(uf)) * wf, x=uf)
    out = dict(cal=cal, u=un.tolist(), diag=diag, specs=specs, gamma=G.tolist(), Q=Q.tolist(), I=Iv.tolist(),
               P1=np.exp(-Iv).tolist())
    groups = sorted(set(s["group"] for s in specs))
    crit = {}
    for gname in groups:
        idx = [j for j, s in enumerate(specs) if s["group"] == gname]
        mts = np.array([specs[j]["mt"] for j in idx])
        lI = np.log(Iv[idx])
        sp_ = CubicSpline(mts, lI)
        cc = dict(P1_at={f"{mt:.2f}": float(math.exp(-Iv[j])) for mt, j in zip(mts, idx)})
        for name, target in (("P1=0.9", -math.log(0.9)), ("P1=0.5", math.log(2.0))):
            try:
                cc[name] = float(brentq(lambda x: sp_(x) - math.log(target), mts[0], mts[-1]))
            except ValueError:
                cc[name] = None
        crit[gname] = cc
    out["critical"] = crit
    out["time_s"] = time.time() - t0
    merge("hazard", out)
    for g_, v in crit.items():
        print(g_, "m_c(0.9)=", v["P1=0.9"], "m_c(0.5)=", v["P1=0.5"], "P1(172.6)=", v["P1_at"].get("172.60"))
    print("time", out["time_s"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "checks"
    if mode == "checks":
        checks()
    elif mode == "static":
        static()
    elif mode == "hazard":
        hazard()
    elif mode == "wcomp":   # Python numbers to compare with R15_wolfram.wl section 4
        cs_ = R14_rge.couplings(172.60, 0.1180)
        o = dict(Delta_2p9637=float(EC(2.9637, -0.25)[0] - estd(2.9637)[0]))
        for rr_ in (1.5, 3.0, 5.0):
            aW_, aZ_, b_, _, _ = masses(rr_, -0.01, cs_["g2"], cs_["gY"], cs_["yt"])
            o[f"sumDelta_r{rr_}"] = float(2 * (EC(aW_, b_)[0] - estd(aW_)[0]) + EC(aZ_, b_)[0] - estd(aZ_)[0])
            o[f"dVMC-dVMUCW_r{rr_}"] = float(dV("MC", rr_, -0.01, cs_["g2"], cs_["gY"], cs_["yt"])[0]
                                             - dV("MU_CW", rr_, -0.01, cs_["g2"], cs_["gY"], cs_["yt"])[0])
        merge("wcomp", o)
        print(json.dumps(o, indent=1))
