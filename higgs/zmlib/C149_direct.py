# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Node-by-node 2D hazard WITH the cross terms (D121 run02, CCA): adds the cross-term increments to the T14b 2D_ex model at each node/column and re-solves the 2D Siegert roots; post() integrates the hazard and finds the critical m_t.  Base of run04/run05 (04_cross_terms).  Paper: PRD App. H2', H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 run02 (holonomy-cross, D121 appendix run02): node-by-node 2D hazard WITH the zero-mode/nonzero-mode cross terms.

Baseline = T14b C150 run02 (2D_ex direct, T1 NNLO 'central' couplings, 11 nodes, FAST cfg); this module adds at every node
and column (m_t, alpha_s) with that node's couplings (g2, gY, yt, lam, g3):
  * one-loop Hessians H11, H33, H3w, Hww(r) (cross_kappa.hessian with node couplings, CCA);
  * SU(2)/U(1) variational: D108 Gaussian functional + F_SU2 + F_Y re-minimised (Newton from the node's base optimum);
    increments dC0 = C0(p') + F_SU2(p') - C0(p0), dBx, dOm2e computed with the SAME discretisation for p0 and p'
    (np.gradient), added to T14b's exact-derivative base parts -> no cross term reproduces T14b's 2D_ex exactly;
  * w-potential: + Om_X^2(beta') of W, Higgs, top (quadratic) + EXACT light-fermion twist V_L(r, s; beta', g2, gY);
  * SU(3)-top variational: E(s) = alpha/s + beta s^2 + kappa3(r) s/2, E0 = 12.98 g3^(2/3), s0 = 3.66 (g3/0.5396)^(2/3).
2D Siegert roots: C150 solver pieces (verbatim C140 prop) + channel term W(r, s); window around the T14b root + run01 shift;
box check (N, S_max) = (28, 14).  Hazard and critical m_t exactly as C150/C144.
Writes only to results/20261014_claude_v12_r3/holonomy_cross/run02/ (T14b/T1 directories are read only).
Usage: python -B C149_direct.py setup | test | run main|als | post
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
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["C150_COUPLINGS"] = "module:" + os.path.join(HERE, "c150_nnlo_central.py")
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import glob  # noqa: E402
import shutil  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

sys.path.insert(0, HERE)
ROOT = _zp.WORK
T14RES = os.path.join(ROOT, _zp.w("r3/twod_direct_sm/run02"))
RES2 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02"))
SH = os.path.join(RES2, "shards")
MYPROF = os.path.join(RES2, "profiles")
# run01 central (VAR_all|SU3var+YL vs 2D_ex, v11 couplings): Q shift and gamma ratio vs lambda (seeds only)
RUN01 = {-0.008: (0.055050, 1.62855), -0.010: (0.054135, 1.56552), -0.012: (0.053110, 1.51121),
         -0.0143: (0.051878, 1.45604), -0.0165: (0.050749, 1.40988)}
MT_ALS = {0.1171: [171.8, 172.2, 172.6, 173.0, 173.4], 0.1189: [172.4, 172.8, 173.2, 173.6, 174.0]}
WINDOW = (0.02, 0.0015)


# [v12-package] the original "setup" step (copying the T14b modules next to this file) is dropped: they are in zmlib.

import C150_twod as W  # noqa: E402  (local unchanged copy)
import cross_core as X  # noqa: E402
import cross_kappa as K  # noqa: E402
import cross_2d as T  # noqa: E402
import cross_var as V  # noqa: E402
import sm_provider as SP  # noqa: E402

W.RES, W.PROFDIR, W.SH1, W.SH2 = RES2, MYPROF, os.path.join(RES2, "sh1"), os.path.join(RES2, "sh2")
PI = math.pi
PV11 = np.load(K.PROFILE)


def g3_at(mt, als, mu):
    return float(SP.coup_point(mt, als, W.MH, mu, "central")["g3"])


def base_parts(mt, als, u):
    """T14b run02 profile parts (read only) -> C150 Parts; fallback: compute into my run02/profiles."""
    fn = os.path.join(T14RES, "profiles", "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
    if not os.path.exists(fn):
        fn = os.path.join(MYPROF, "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
        if not os.path.exists(fn):
            CS = W.Couplings(W.KIND)
            c = CS(mt, als, W.mu_node(u))
            d0 = W.compute_profile(c["g2"], c["gY"])
            os.makedirs(MYPROF, exist_ok=True)
            np.savez(fn, **{k: np.asarray(d0[k]) for k in ("rf", "C0", "Bx", "Om2e", "EtBr", "g2", "gY", "chk_Et")})
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


def cross_increments(c, g3):
    """node couplings -> splines on rf of dC0, dBx, dOm2e (variational), OmX2_nonlight(beta'), beta', dU3var."""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = PV11["rf"]
    rg = T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows])(rr)
    P0, _ = V.newton(lambda p: etot_g(p, rf, g2, gY), PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    Pst = P0.copy()                          # start with the BO w-width bt = sqrt(Om^2 + Om_X^2) (fast Newton convergence)
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    P1, gres = V.newton(lambda p: etot_g(p, rf, g2, gY) + V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
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
    info = dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])), chk_base_Et=None, E0_su3=E0, s0_su3=s0)
    return out, info


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


def t14_root(mt, als, u):
    d = W.jload(os.path.join(T14RES, "shards2d", ("2Dex_main_u%.4f.json" if abs(als - 0.1180) < 1e-9 else
                                                   "2Dex_als_u%.4f.json") % u))
    for x in d["cols"]:
        if abs(x["mt"] - mt) < 1e-6 and abs(x["als"] - als) < 1e-9:
            return complex(*x["E2"]), x
    raise KeyError((mt, als, u))


def seed_shift(lam):
    ls = sorted(RUN01)
    dq = np.polyfit(ls, [RUN01[l][0] for l in ls], 1)
    lr = np.polyfit(ls, [math.log(RUN01[l][1]) for l in ls], 1)
    return float(np.polyval(dq, lam)), float(math.exp(np.polyval(lr, lam)))


def jobX(spec):
    fn = os.path.join(SH, spec["tag"] + ".json")
    if os.path.exists(fn):
        return W.jload(fn)
    t0 = time.time()
    u = float(spec["u"])
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    cols = spec["cols"]
    P = len(cols)
    U1, OM, U0, OM0 = (np.empty((P, len(r2))) for _ in range(4))
    ph0, ph1, ph00, ph10 = (np.empty(P) for _ in range(4))
    Wl, rows, cen, gim, E14 = [], [], np.empty(P), np.empty(P), np.empty(P, complex)
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    for j, cc in enumerate(cols):
        mt, als = cc["mt"], cc["als"]
        mu = W.mu_node(u)
        c = CS(mt, als, mu)
        g3 = g3_at(mt, als, mu)
        Pp, pinfo = base_parts(mt, als, u)
        U0[j], OM0[j] = W.build2d(Pp, dmc, r2, c, "2D_ex")
        inc, info = cross_increments(c, g3)
        rr2 = r2 * r2
        U1[j] = U0[j] + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0[j] + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        Wl.append(V_light_c(r2, sg.s, inc["beta"](r2), c["g2"], c["gY"]))
        ph0[j], ph1[j], s, al = W.phases(U1[j], r2)
        ph00[j], ph10[j], _, _ = W.phases(U0[j], r2)
        e14, x14 = t14_root(mt, als, u)
        dq, rat = seed_shift(c["lam"])
        cen[j], gim[j], E14[j] = e14.real + dq, -2 * e14.imag * rat, e14
        rows.append(dict(mt=mt, als=als, u=u, g2=c["g2"], gY=c["gY"], yt=c["yt"], lam=c["lam"], g3=g3, prof=pinfo["file"],
                         prof_g2=pinfo["g2"], info=info, E_T14b=[e14.real, e14.imag], gamma_T14b=-2 * e14.imag))
    fun, cnt = make_funW(U1, OM, Wl, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    E, wd = W.window_roots(fun, cen, gim, WINDOW)
    sgb = W.SGrid(W.BOXCHK["N"], W.BOXCHK["Smax"])
    Wlb = [V_light_c(r2, sgb.s, cross_increments.cache[j] if False else None, 0, 0) if False else None for j in range(P)]
    # box check: rebuild the light-fermion term on the (28, 14) s-grid
    Wlb = []
    for j, cc in enumerate(cols):
        Wlb.append(None)
    out = dict(tag=spec["tag"], u=u, cols=rows, cfg=cfg, window=WINDOW)
    # regression (no cross term) on the first column: reproduce T14b's root with this code path
    if spec.get("regress"):
        f0, _ = make_funW(U0[:1], OM0[:1], [None], r2, m2, h, sg, int(cfg["order"]), ph00[:1], ph10[:1])
        E0r, wd0 = W.window_roots(f0, np.array([E14[0].real]), np.array([-2 * E14[0].imag]), (0.004, 0.0004))
        out["regression_noX"] = dict(E=[float(E0r[0].real), float(E0r[0].imag)], E_T14b=[E14[0].real, E14[0].imag],
                                     dQ=float(E0r[0].real - E14[0].real), rel_gamma=float(E0r[0].imag / E14[0].imag - 1))
    for j in range(P):
        rows[j].update(E=[float(E[j].real), float(E[j].imag)], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                       center=float(cen[j]), Emin=float(wd["Emin"][j]), noloc=bool(wd["noloc"][j]),
                       nc=bool(wd["act"][j]), ratio_vs_T14b=float(E[j].imag / E14[j].imag),
                       edge=bool(abs(wd["Emin"][j] - cen[j]) >= WINDOW[0] - 1e-9))
    out["time_s"], out["n_eval"] = time.time() - t0, cnt["n"]
    W.wjson(fn, out)
    print("[X %s] %.0fs %s" % (spec["tag"], time.time() - t0, "; ".join(
        "%.2f/%.4f r=%.4f%s%s" % (x["mt"], x["als"], x["ratio_vs_T14b"], " NC" if x["nc"] else "",
                                  " EDGE" if x["edge"] else "") for x in rows)), flush=True)
    return out


def specs(which):
    out = []
    for u in W.UN11:
        if which == "main":
            out.append(dict(tag="X_main_u%.4f" % u, u=float(u), cols=[dict(mt=mt, als=0.1180) for mt in W.MT_MAIN],
                            regress=bool(u in (0.0, 2.5, 5.0))))
        elif which == "als":
            out.append(dict(tag="X_als_u%.4f" % u, u=float(u),
                            cols=[dict(mt=mt, als=a) for a in (0.1171, 0.1189) for mt in MT_ALS[a]]))
    return out


def run(which, nw=int(os.environ.get("DNEC_WORKERS", "3"))):
    t0 = time.time()
    os.makedirs(SH, exist_ok=True)
    sp = specs(which)
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(sp)))) as ex:  # [v12-package] was min(nw, 3)
        outs = list(ex.map(jobX, sp))
    print("run %s done: %d node jobs %.0fs" % (which, len(outs), time.time() - t0), flush=True)


def test():
    o = jobX(dict(tag="TEST_u0.0000", u=0.0, cols=[dict(mt=172.6, als=0.1180), dict(mt=173.0, als=0.1180)], regress=True))
    print(json.dumps(W.clean({k: o[k] for k in ("regression_noX", "time_s")}), indent=1))
    print(json.dumps(W.clean([{k: x[k] for k in ("mt", "lam", "g3", "gamma_T14b", "gamma", "ratio_vs_T14b", "Q", "center",
                                                 "edge", "nc", "info")} for x in o["cols"]]), indent=1))


def post():
    R = {}
    for f in glob.glob(os.path.join(SH, "X_*.json")):
        d = W.jload(f)
        for x in d["cols"]:
            R[(round(x["als"], 4), round(x["mt"], 2), round(x["u"], 4))] = x
    out = dict(crit={}, P1={}, flags=[k for k, x in R.items() if x["nc"] or x["edge"] or not x["gamma"] > 0])
    T14S = W.jload(os.path.join(T14RES, "C150_summary.json"))
    for als, mts in ((0.1180, W.MT_MAIN), (0.1171, MT_ALS[0.1171]), (0.1189, MT_ALS[0.1189])):
        I, I0, ok = [], [], []
        for mt in mts:
            g = [R.get((round(als, 4), round(mt, 2), round(float(u), 4))) for u in W.UN11]
            if any(x is None for x in g):
                continue
            I.append(W.integ(W.UN11, np.array([x["gamma"] for x in g])))
            I0.append(W.integ(W.UN11, np.array([x["gamma_T14b"] for x in g])))
            ok.append(mt)
        if len(ok) < 3:
            continue
        cX, spX = W.crit(ok, I)
        c0, sp0 = W.crit(ok, I0)
        out["crit"]["%.4f" % als] = dict(with_cross=cX, T14b_recomputed=c0,
                                         shift={k: (cX[k] - c0[k]) if cX[k] and c0[k] else None for k in cX})
        if abs(als - 0.1180) < 1e-9:
            out["P1"] = {f"{m}": float(math.exp(-math.exp(spX(m)))) for m in (172.2, 172.52, 172.6, 173.0)}
            out["P1_T14b"] = {f"{m}": float(math.exp(-math.exp(sp0(m)))) for m in (172.2, 172.52, 172.6, 173.0)}
            out["P1_grid"] = {f"{m}": float(math.exp(-i)) for m, i in zip(ok, I)}
            CS = W.Couplings(W.KIND)
            out["lam_c_hazard_equiv"] = {k: CS(v, als, W.mu_node(0.0))["lam"] for k, v in cX.items() if v}
    out["ratio_method_run01"] = {"P1=0.9": -0.0933, "P1=0.5": -0.1346, "P1=0.1": -0.1796}
    out["T14b_run02_published"] = {"P1=0.9": 172.5826, "P1=0.5": 173.0775, "P1=0.1": 173.5784}
    W.wjson(os.path.join(RES2, "C149_run02_summary.json"), out)
    print(json.dumps(W.clean(out), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "test":
        test()
    elif md == "run":
        run(sys.argv[2])
    elif md == "post":
        post()
