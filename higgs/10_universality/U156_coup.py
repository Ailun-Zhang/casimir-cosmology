# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_coup.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: copied (functions base_parts, etot_g, cross_increments, V_light_c, make_funW, jobX) from
# 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct.py on 2026-10-14 by agent universality (C156/D128),
# then modified: coupling multipliers (g2, gY, yt, lambda) at every node, fresh gauge profiles for the varied (g2, gY),
# seeds = run02 root + 1D_ex root shift (wide real-axis scan), retry with the wide D112 window.  The holonomy_cross modules
# (C150_twod, cross_core, cross_kappa, cross_2d, cross_var, c150_nnlo_central, sm_provider) are imported UNCHANGED from
# their directory (python -B); nothing is written there.
r"""[v12 package, 2026-10-15] Coupling dependence of lam_c: 2D direct hazard with g2, gY, y_t scaled by +-10% at fixed lam(mu) trajectory -> dln|lam_c|/dln(g2, gY, y_t) = 0.66/0.16/0.63 (P1 = 0.5).  Paper: PRD Sec. IV D. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C156 T1b: lambda_c(g2, gY, yt) in the round-3 central model (C149 run02 pipeline) by direct node recomputation.
Variant k = (f_g2, f_gY, f_yt, f_lam) multiplies the node couplings of every column (m_t, alpha_s = 0.1180) at every node.
Hazard of a variant: gamma_var(u, m_t) = gamma_run02(u, m_t) * rho(u, m_t), rho computed at nodes U and columns MTV,
ln rho cubic in u (and in m_t) -> I(m_t) on the run02 11-node rule -> m_c, lambda_c = lambda(mu_b, m_c) (lambda unvaried).
Usage: python -B U156_coup.py test | run [nw] | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = _zp.WORK
HX = os.path.join(ROOT, _zp.ZMLIB)
os.environ["C150_COUPLINGS"] = "module:" + os.path.join(HX, "c150_nnlo_central.py")
sys.path.insert(0, HX)
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.integrate import simpson  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import C150_twod as W  # noqa: E402
import cross_core as X  # noqa: E402
import cross_kappa as K  # noqa: E402
import cross_2d as T  # noqa: E402
import cross_var as V  # noqa: E402
import sm_provider as SP  # noqa: E402

T14RES = os.path.join(ROOT, _zp.w("r3/twod_direct_sm/run02"))
X02 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/shards"))
RES = os.path.join(ROOT, _zp.w("r3/universality/run01"))
SH = os.path.join(RES, "shards")
MYPROF = os.path.join(RES, "profiles")
W.RES, W.PROFDIR, W.SH1, W.SH2 = RES, MYPROF, os.path.join(RES, "sh1"), os.path.join(RES, "sh2")
PI = math.pi
PV11 = np.load(K.PROFILE)
WINDOW = (0.02, 0.0015)
WWIDE = (0.05, 0.0025)
NODES = [0.0, 1.0, 2.0, 3.5, 5.0]
MTV = [172.0, 172.4, 172.8, 173.2, 173.6]
VAR = {"g2+": (1.1, 1.0, 1.0, 1.0), "g2-": (0.9, 1.0, 1.0, 1.0), "gY+": (1.0, 1.1, 1.0, 1.0), "gY-": (1.0, 0.9, 1.0, 1.0),
       "yt+": (1.0, 1.0, 1.1, 1.0), "yt-": (1.0, 1.0, 0.9, 1.0)}
VAR_U0 = {"scal": (1.1, 1.1, 1.0, 1.21), "scalt": (1.1, 1.1, 1.1, 1.21)}   # zero-mode scaling tests (u = 0 only)


def g3_at(mt, als, mu):
    return float(SP.coup_point(mt, als, W.MH, mu, "central")["g3"])


def base_parts(mt, als, u, var, g2, gY, fg):
    """unvaried (g2, gY): T14b run02 profile (read only); varied: fresh D108 Gaussian profile in my run01/profiles."""
    if fg[0] == 1.0 and fg[1] == 1.0:
        fn = os.path.join(T14RES, "profiles", "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
    else:
        fn = os.path.join(MYPROF, "prof_%s_mt%.2f_as%.4f_u%.4f.npz" % (var, mt, als, u))
        if not os.path.exists(fn):
            d0 = W.compute_profile(g2, gY)
            os.makedirs(MYPROF, exist_ok=True)
            tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
            np.savez(tmp, **{k: np.asarray(d0[k]) for k in ("rf", "C0", "Bx", "Om2e", "EtBr", "g2", "gY", "chk_Et")})
            os.replace(tmp, fn)
    d = dict(np.load(fn))
    return W.Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]), dict(g2=float(d["g2"]), gY=float(d["gY"]), file=fn)


def etot_g(p, r, g2, gY):
    a, k, b, c = p
    D = k * b - c * c
    kin = 0.75 * (2 * a + k + b)
    mass = 3 * r * r / 16 * (2 * g2 * g2 / a + (g2 * g2 * b - 2 * g2 * gY * c + gY * gY * k) / D)
    ym = 3 * g2 * g2 / (4 * a * a) + 3 * g2 * g2 * b / (2 * a * D)
    ang = 3.0 / (a * r * r) * (b * (k - a) ** 2 - 2 * c * c * (k - a) + k * c * c) / D
    return kin + mass + ym + ang


def newton_safe(fun, p0, iters=8):
    """cross_var.newton (verbatim) except a Levenberg shift 1e-10*max|diag| on point-wise singular FD Hessians
    (a varied-coupling profile at u = 2 produced an exactly singular point; regular points are unchanged)."""
    p = p0.copy()
    n = p.shape[0]
    for it in range(iters):
        h = 1e-4 * np.maximum(np.abs(p), 1e-3)
        f0 = fun(p)
        g = np.empty_like(p)
        Hs = np.empty((n, n) + p.shape[1:])
        fp, fm = [], []
        for i in range(n):
            e = np.zeros_like(p)
            e[i] = h[i]
            fp.append(fun(p + e))
            fm.append(fun(p - e))
            g[i] = (fp[i] - fm[i]) / (2 * h[i])
            Hs[i, i] = (fp[i] - 2 * f0 + fm[i]) / h[i] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros_like(p)
                ej = np.zeros_like(p)
                ei[i] = h[i]
                ej[j] = h[j]
                v = (fun(p + ei + ej) - fun(p + ei - ej) - fun(p - ei + ej) + fun(p - ei - ej)) / (4 * h[i] * h[j])
                Hs[i, j] = Hs[j, i] = v
        M = np.moveaxis(Hs, (0, 1), (-2, -1)).copy()
        dt = np.abs(np.linalg.det(M))
        sc = np.max(np.abs(np.diagonal(M, axis1=-2, axis2=-1)), axis=-1)
        bad = ~(dt > 1e-30 * np.maximum(sc, 1e-300) ** n) | ~np.isfinite(dt)
        if bad.any():
            M[bad] += (1e-10 * np.maximum(sc[bad], 1e-12))[:, None, None] * np.eye(n)[None]
        dp = np.linalg.solve(M, np.moveaxis(g, 0, -1)[..., None])[..., 0]
        dp[~np.isfinite(dp)] = 0.0
        p = p - np.moveaxis(dp, -1, 0)
    return p, np.max(np.abs(g), axis=0)


def cross_increments(c, g3):
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = PV11["rf"]
    rg = T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows])(rr)
    P0, _ = newton_safe(lambda p: etot_g(p, rf, g2, gY), PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    Pst = P0.copy()
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    P1, gres = newton_safe(lambda p: etot_g(p, rf, g2, gY) + V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
    q0 = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    q1 = W.derived(rf, P1, np.gradient(P1, rf, axis=1, edge_order=2), g2, gY)
    fsu2 = V.fcross(P1, Hs, "SU2")
    beta = q1["beta"]
    omx_nl = Hs["Hww_nl"] - 2 * beta * Hs["H3w_nl"] + beta ** 2 * Hs["H33_nl"]
    kap = np.array([x["kappa3"] for x in rows])
    k0 = (6 * X.curv(0.0, "P")[0] - 12 * X.curv(0.0, "A")[0]) / (4 * PI ** 2)
    E0, s0 = 12.980 * g3 ** (2 / 3), 3.66 * (g3 / 0.5396) ** (2 / 3)
    Ev, _ = V.su3_var(kap, E0, s0)
    Ev0, _ = V.su3_var(np.array([k0]), E0, s0)
    r2 = rf * rf
    out = dict(dC0=CubicSpline(rf, r2 * (q1["C0"] + fsu2 - q0["C0"])), dBx=CubicSpline(rf, r2 * (q1["Bx"] - q0["Bx"])),
               dOm2e=CubicSpline(rf, q1["Om2e"] - q0["Om2e"]), omx_nl=CubicSpline(rf, omx_nl),
               beta=CubicSpline(rf, beta), dU3=CubicSpline(rg, Ev - Ev0[0]))
    return out, dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])))


def V_light_c(r, s, beta, g2, gY):
    f, E0 = T.ebar_A()
    out = np.zeros((len(r), len(s)))
    for nm, mult, t3, y in K.LIGHT:
        q = np.abs(gY * y - g2 * beta * t3)
        out += -2.0 * mult * (f(np.minimum(q[:, None] * s[None, :] / (2 * PI), 2.5)) - E0)
    return out


def make_funW(U1, OM, Wl, r, m, h, sg, order, ph0, ph1):
    Dd = U1[:, :, None] + 0.5 * OM[:, :, None] * sg.s2[None, None, :]
    for j in range(len(Wl)):
        if Wl[j] is not None:
            Dd[j] += Wl[j]
    Tn = (-(h * h / 6.0) * sg.T).astype(complex)
    Yf = T.bc2(U1, OM, Wl, sg, h, m, order)
    cn = {"n": 0}

    def fun(E):
        cn["n"] += 1
        s_, la = W.prop(np.asarray(E, complex), Dd, Tn, h, m, ph0, ph1, Yf)
        if "c" not in cn:
            cn["c"] = la.copy()
        return s_ * np.exp(np.clip(la - cn["c"], -700.0, 700.0))
    return fun, cn


def x02_root(mt, u):
    d = W.jload(os.path.join(X02, "X_main_u%.4f.json" % u))
    for x in d["cols"]:
        if abs(x["mt"] - mt) < 1e-6:
            return complex(*x["E"])
    raise KeyError((mt, u))


def scan_root1d(cols, r, m, qc, gsd, half=0.40, step=0.002):
    """1D_ex root: detrended deepest real-axis |f| minimum in qc +- half, then Muller (W.solve_res)."""
    Ut = np.array(cols).T.copy()
    s, al = W._fr(cols, r)
    offs = np.arange(-half, half + 0.5 * step, step)
    lv = np.array([np.log(np.abs(W.R.jost(qc + o + 0j, Ut, r, m, s, al)[0]) + 1e-300) for o in offs])
    g = np.empty(len(qc), complex)
    for j in range(len(qc)):
        y = lv[:, j]
        y = y - np.polyval(np.polyfit(offs, y, 1), offs)
        i = int(np.argmin(y[1:-1])) + 1
        g[j] = qc[j] + offs[i] - 0.5j * gsd[j]
    E, rs = W.solve_res(cols, r, m, g, spread=0.002)
    return E, rs


def jobV(spec):
    fn = os.path.join(SH, spec["tag"] + ".json")
    if os.path.exists(fn):
        return W.jload(fn)
    t0 = time.time()
    u, var, fg = float(spec["u"]), spec["var"], spec["f"]
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    r1, m1 = W.R.grid(16.0)
    cols = spec["cols"]
    P = len(cols)
    U1, OM = np.empty((P, len(r2))), np.empty((P, len(r2)))
    ph0, ph1 = np.empty(P), np.empty(P)
    Wl, rows, EX = [], [], np.empty(P, complex)
    c1b, c1v = [], []
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    tprof = 0.0
    for j, mt in enumerate(cols):
        als = 0.1180
        mu = W.mu_node(u)
        c0 = CS(mt, als, mu)
        c = dict(c0, g2=c0["g2"] * fg[0], gY=c0["gY"] * fg[1], yt=c0["yt"] * fg[2], lam=c0["lam"] * fg[3])
        g3 = g3_at(mt, als, mu)
        Pb, _ = base_parts(mt, als, u, "base", c0["g2"], c0["gY"], (1.0, 1.0))
        t1 = time.time()
        Pp, pinfo = base_parts(mt, als, u, var, c["g2"], c["gY"], fg)
        tprof += time.time() - t1
        U0, OM0 = W.build2d(Pp, dmc, r2, c, "2D_ex")
        inc, info = cross_increments(c, g3)
        rr2 = r2 * r2
        U1[j] = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        Wl.append(V_light_c(r2, sg.s, inc["beta"](r2), c["g2"], c["gY"]))
        ph0[j], ph1[j], _, _ = W.phases(U1[j], r2)
        EX[j] = x02_root(mt, u)
        c1b.append(W.U1d("1D_ex", Pb, dmc, r1, c0, None))
        c1v.append(W.U1d("1D_ex", Pp, dmc, r1, c, None))
        rows.append(dict(mt=mt, u=u, var=var, f=list(fg), g2=c["g2"], gY=c["gY"], yt=c["yt"], lam=c["lam"], g3=g3,
                         prof=pinfo["file"], info=info, E_X02=[EX[j].real, EX[j].imag], gamma_X02=-2 * EX[j].imag))
    # 1D_ex roots, base and variant (same scan rule) -> seed shift
    gX = -2 * EX.imag
    E1b, rsb = scan_root1d(c1b, r1, m1, EX.real + 0.015, 1.25 * gX)
    E1v, rsv = scan_root1d(c1v, r1, m1, E1b.real, -2 * E1b.imag)
    dQ = E1v.real - E1b.real
    rg = np.where((E1b.imag < 0) & (E1v.imag < 0), E1v.imag / E1b.imag, 1.0)
    cen, gim = EX.real + 0.9 * dQ, gX * rg          # 0.9: 2D/1D shift ratio of the test job (g2+10%: 0.191/0.211)
    fun, cnt = make_funW(U1, OM, Wl, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    if spec.get("seed") is not None:            # fix pass: Muller from a continuation seed (neighbouring good node)
        E0 = np.array([complex(a, b) for a, b in spec["seed"]])
        cen, gim = E0.real.copy(), -2 * E0.imag
        E, it, act, _ = W.muller2(fun, E0 - 1e-4, E0 + 1e-4, E0 + 0j)
        wd = dict(Emin=E0.real.copy(), act=act, noloc=np.zeros(P, bool))
    else:
        E, wd = W.window_roots(fun, cen, gim, WINDOW)
    edge = np.abs(wd["Emin"] - cen) >= WINDOW[0] - 1e-9
    bad = edge | wd["act"] | wd["noloc"] | ~(-2 * E.imag > 0)
    retried = np.zeros(P, bool)
    if bad.any() and spec.get("seed") is None:
        jj = np.where(bad)[0]
        f2, _ = make_funW(U1[jj], OM[jj], [Wl[j] for j in jj], r2, m2, h, sg, int(cfg["order"]), ph0[jj], ph1[jj])
        E2, wd2 = W.window_roots(f2, cen[jj], gim[jj], WWIDE)
        for k, j in enumerate(jj):
            E[j] = E2[k]
            wd["Emin"][j], wd["act"][j], wd["noloc"][j] = wd2["Emin"][k], wd2["act"][k], wd2["noloc"][k]
            retried[j] = True
    for j in range(P):
        win = WWIDE if retried[j] else WINDOW
        rows[j].update(E=[float(E[j].real), float(E[j].imag)], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                       center=float(cen[j]), gim=float(gim[j]), Emin=float(wd["Emin"][j]), noloc=bool(wd["noloc"][j]),
                       nc=bool(wd["act"][j]), retried=bool(retried[j]),
                       edge=bool(abs(wd["Emin"][j] - cen[j]) >= win[0] - 1e-9),
                       rho=float(E[j].imag / EX[j].imag), dQ2=float(E[j].real - EX[j].real),
                       E1b=[float(E1b[j].real), float(E1b[j].imag)], E1v=[float(E1v[j].real), float(E1v[j].imag)],
                       rho1=float(E1v[j].imag / E1b[j].imag), dQ1=float(dQ[j]))
    out = dict(tag=spec["tag"], u=u, var=var, f=list(fg), cols=rows, cfg=cfg, time_s=time.time() - t0, t_prof=tprof,
               n_eval=cnt["n"])
    W.wjson(fn, out)
    print("[V %s] %.0fs (prof %.0fs) %s" % (spec["tag"], time.time() - t0, tprof, "; ".join(
        "%.1f rho=%.4f rho1=%.4f dQ=%.4f/%.4f%s%s%s" % (x["mt"], x["rho"], x["rho1"], x["dQ2"], x["dQ1"],
                                                     " R" if x["retried"] else "", " NC" if x["nc"] else "",
                                                     " EDGE" if x["edge"] else "") for x in rows)), flush=True)
    return out


def okcol(x):
    return x["gamma"] > 0 and abs(math.log(max(x["gamma"], 1e-300) / x["gim"])) < 1.2 and not x["nc"]


def fix(nw=3):
    """re-solve shards with a spurious root (gamma <= 0 or far from the seed width) from a continuation seed:
    Q_X02(u) + dQ2 and gamma_X02(u) * rho of the nearest all-good node of the same variant."""
    by = {}
    for f in sorted(glob.glob(os.path.join(SH, "V_*.json"))):
        if f.endswith("_bad.json"):
            continue
        d = W.jload(f)
        by.setdefault(d["var"], {})[d["u"]] = (f, d)
    todo = []
    for v, D in by.items():
        good = [u for u, (f, d) in D.items() if all(okcol(x) for x in d["cols"])]
        for u, (f, d) in D.items():
            if all(okcol(x) for x in d["cols"]) or not good:
                continue
            dg = D[min(good, key=lambda z: abs(z - u))][1]
            seed = []
            for x in d["cols"]:
                xg = [y for y in dg["cols"] if abs(y["mt"] - x["mt"]) < 1e-6][0]
                EX = complex(*x["E_X02"])
                seed.append([EX.real + xg["dQ2"], EX.imag * xg["rho"]])
            os.replace(f, f[:-5] + "_bad.json")
            todo.append(dict(tag=d["tag"], var=v, f=tuple(d["f"]), u=u, cols=[x["mt"] for x in d["cols"]], seed=seed))
    print("fix: %d jobs %s" % (len(todo), [t["tag"] for t in todo]), flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(nw, 3)) as ex:
            list(ex.map(jobV, todo))


def specs():
    out = []
    for v, f in VAR.items():
        for u in NODES:
            out.append(dict(tag="V_%s_u%.4f" % (v, u), var=v, f=f, u=u, cols=MTV))
    for v, f in VAR_U0.items():
        out.append(dict(tag="V_%s_u%.4f" % (v, 0.0), var=v, f=f, u=0.0, cols=MTV))
    return out


def run(nw=3):
    t0 = time.time()
    os.makedirs(SH, exist_ok=True)
    sp = specs()
    with ProcessPoolExecutor(max_workers=min(nw, 3)) as ex:
        list(ex.map(jobV, sp))
    print("run done: %d jobs %.0fs" % (len(sp), time.time() - t0), flush=True)


def test():
    o = jobV(dict(tag="TEST_base_u0.0000", var="base", f=(1.0, 1.0, 1.0, 1.0), u=0.0, cols=[172.6, 173.0]))
    o2 = jobV(dict(tag="TEST_g2p_u0.0000", var="g2+", f=VAR["g2+"], u=0.0, cols=[172.6, 173.0]))
    for oo in (o, o2):
        print(json.dumps(W.clean({k: oo[k] for k in ("tag", "time_s", "t_prof", "n_eval")}), indent=None))
        print(json.dumps(W.clean([{k: x[k] for k in ("mt", "gamma_X02", "gamma", "rho", "rho1", "dQ2", "dQ1", "edge",
                                                     "retried", "nc", "info")} for x in oo["cols"]]), indent=None))


# ------------------------------------------------------------------------------------------------ post
def integ(u, g, umax=5.0):
    uf = np.linspace(0.0, umax, 20001)
    wf = 1.0 / (W.HLB * np.sqrt(np.cosh(2 * uf)))
    spl = CubicSpline(u, np.log(np.maximum(g, 1e-300)))
    return float(simpson(np.exp(spl(uf)) * wf, x=uf))


def crit(x, I, tg):
    sp = CubicSpline(x, np.log(I))
    try:
        return float(brentq(lambda z: sp(z) - math.log(tg), x[0], x[-1]))
    except ValueError:
        e = 0 if math.log(tg) < sp(x[0]) else -1
        return float(x[e] + (math.log(tg) - sp(x[e])) / float(sp(x[e], 1)))


def post():
    UN11 = list(W.UN11)
    G = {}
    LAMB = {}
    for u in UN11:
        d = W.jload(os.path.join(X02, "X_main_u%.4f.json" % u))
        for x in d["cols"]:
            G[(round(x["mt"], 2), round(u, 4))] = x["gamma"]
            if u == 0.0:
                LAMB[round(x["mt"], 2)] = x["lam"]
    mall = sorted(LAMB)
    lam_b = CubicSpline(mall, [LAMB[m] for m in mall])
    out = dict(nodes=NODES, cols=MTV, variants={}, base={})
    Ib = np.array([integ(UN11, [G[(m, round(u, 4))] for u in UN11]) for m in MTV])
    base = {k: crit(MTV, Ib, t) for k, t in W.TGT.items()}
    out["base"] = dict(mc_5col=base, lam_c_5col={k: float(lam_b(v)) for k, v in base.items()})
    for f in sorted(glob.glob(os.path.join(SH, "V_*.json"))):
        if f.endswith("_bad.json"):
            continue
        d = W.jload(f)
        v = d["var"]
        out["variants"].setdefault(v, dict(f=d["f"], rho={}, rho1={}, flags=[]))
        for x in d["cols"]:
            out["variants"][v]["rho"]["%.2f|%.4f" % (x["mt"], x["u"])] = x["rho"]
            out["variants"][v]["rho1"]["%.2f|%.4f" % (x["mt"], x["u"])] = x["rho1"]
            if x["edge"] or x["nc"] or not okcol(x):
                out["variants"][v]["flags"].append([x["mt"], x["u"]])
    # 1D-guided substitution for 2D roots that converged to real-axis zeros (flagged): rho = rho1 * F(u), F = rho/rho1
    # at the good nodes of the same (variant, m_t), linear in u (constant if one good node)
    for v, D in out["variants"].items():
        D["substituted"] = []
        bad = set((round(m, 2), round(u, 4)) for m, u in D["flags"])
        for m in MTV:
            us_all = sorted(float(k.split("|")[1]) for k in D["rho"] if abs(float(k.split("|")[0]) - m) < 1e-6)
            good = [u for u in us_all if (round(m, 2), round(u, 4)) not in bad]
            if len(good) == len(us_all) or not good:
                continue
            Fg = [D["rho"]["%.2f|%.4f" % (m, u)] / D["rho1"]["%.2f|%.4f" % (m, u)] for u in good]
            cf = np.polyfit(good, Fg, 1) if len(good) >= 2 else [0.0, Fg[0]]
            for u in us_all:
                if (round(m, 2), round(u, 4)) in bad:
                    k = "%.2f|%.4f" % (m, u)
                    D["rho"][k] = D["rho1"][k] * float(np.polyval(cf, u))
                    D["substituted"].append([m, u, D["rho"][k]])
    wfr = {}
    for m in MTV:
        g_ = [G[(m, round(u, 4))] for u in UN11]
        wfr["%.2f" % m] = 1 - integ([u for u in UN11 if u <= 2.0], [g for u, g in zip(UN11, g_) if u <= 2.0], umax=2.0) / integ(UN11, g_)
    out["hazard_fraction_u_gt_2"] = wfr
    for v, D in out["variants"].items():
        us = sorted(set(float(k.split("|")[1]) for k in D["rho"]))
        st = {}
        lr0 = np.array([math.log(D["rho"]["%.2f|%.4f" % (m, 0.0)]) for m in MTV])
        st["s_eff_u0"] = (lr0 / math.log(D["f"][0] * D["f"][1] * D["f"][2] if v not in VAR_U0 else 1.1)).tolist()
        g0 = np.array([G[(m, 0.0)] for m in MTV]) * np.exp(lr0)
        st["static_u0"] = {}
        sp0 = CubicSpline(MTV, np.log(g0))
        sp0b = CubicSpline(MTV, np.log([G[(m, 0.0)] for m in MTV]))
        for k, t in W.TGT.items():
            mcv = float(brentq(lambda z: sp0(z) - math.log(t / W.ETA), MTV[0] - 0.6, MTV[-1] + 0.6))
            mcb = float(brentq(lambda z: sp0b(z) - math.log(t / W.ETA), MTV[0] - 0.6, MTV[-1] + 0.6))
            st["static_u0"][k] = dict(lam_c=float(lam_b(mcv)), lam_c_base=float(lam_b(mcb)), dmc=mcv - mcb)
        if len(us) >= 4:
            Iv = []
            for m in MTV:
                lr = CubicSpline(us, [math.log(D["rho"]["%.2f|%.4f" % (m, u)]) for u in us])
                Iv.append(integ(UN11, [G[(m, round(u, 4))] * math.exp(lr(u)) for u in UN11]))
            Iv = np.array(Iv)
            mc = {k: crit(MTV, Iv, t) for k, t in W.TGT.items()}
            st["hazard"] = {k: dict(mc=mc[k], dmc=mc[k] - base[k], lam_c=float(lam_b(mc[k])),
                                    dlam_c=float(lam_b(mc[k]) - lam_b(base[k]))) for k in W.TGT}
            st["dlnI_at_fixed_mt"] = (np.log(Iv / Ib)).tolist()
        D.update(st)
    der = {}
    for g in ("g2", "gY", "yt"):
        a, b = out["variants"].get(g + "+"), out["variants"].get(g + "-")
        if a and b and "hazard" in a and "hazard" in b:
            dl = math.log(1.1) - math.log(0.9)
            der[g] = {k: dict(dlamc_dlng=(a["hazard"][k]["lam_c"] - b["hazard"][k]["lam_c"]) / dl,
                              dmc_dlng=(a["hazard"][k]["mc"] - b["hazard"][k]["mc"]) / dl,
                              asym=[a["hazard"][k]["dlam_c"], b["hazard"][k]["dlam_c"]]) for k in W.TGT}
            der[g]["s_hazard_mean"] = float(np.mean((np.array(a["dlnI_at_fixed_mt"]) - np.array(b["dlnI_at_fixed_mt"])) / dl))
    out["derivatives"] = der
    W.wjson(os.path.join(RES, "C156_coup.json"), out)
    print(json.dumps(W.clean({"base": out["base"], "derivatives": der,
                              "static": {v: D.get("static_u0") for v, D in out["variants"].items()},
                              "s_u0": {v: D.get("s_eff_u0") for v, D in out["variants"].items()},
                              "flags": {v: D["flags"] for v, D in out["variants"].items()}}), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "test":
        test()
    elif md == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif md == "fix":
        fix()
    elif md == "post":
        post()
