# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_fix2.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128 run02, coordinator W3 04:58Z) on 2026-10-15; the problem build
# is copied from U156_coup.jobV (same directory, unchanged); run01 files are read only, all output goes to run02/.
r"""[v12 package, 2026-10-15] D128 run02 repair: unbounded cross-term functional at large r for gY/g2 <~ 0.75 (model applicability); coupling derivatives re-evaluated (change <= 0.2%).  Paper: PRD Sec. IV D (footnote).

C156 run02: re-solve the 2D roots that run01 left on real-axis zeros (g2+10% u = 3.5, 5; gY-10% u = 2, 3.5, 5).
Root rule (D112 Sec. 2): fine real-axis |f| scan in center +- 0.05 (step 0.001), interior local minima ordered by distance
to the center, Muller with spacing 0.001 from the closest one; a converged root with |Im E| < 1e-13 (real zero) is deflated,
f -> f/(E - z), and Muller restarted (up to 4 passes, then the next local minimum).  Acceptance: gamma > 0,
|ln(gamma/gim)| < 1.5, |Q - center| < 0.01; box check on the (N, S_max) = (28, 14) sine-DVR (C150 criterion
|dE| < max(2e-7, 2e-3 gamma)).  Seeds (center, gim) = the continuation seeds stored in the run01 shards.
Then: hazards of all variants with the corrected rho (no 1D substitution), dln|lam_c|/dln g, WKB decomposition, run01 diff.
Usage: python -B U156_fix2.py run | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import time
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import U156_coup as UC  # noqa: E402  (sets the holonomy_cross imports; W.RES etc. point to run01 but nothing is written)

W = UC.W
R1 = UC.RES
R2 = os.path.join(os.path.dirname(R1), "run02")
SH1, SH2 = os.path.join(R1, "shards"), os.path.join(R2, "shards")
JOBS = [("g2+", 3.5), ("g2+", 5.0), ("gY-", 2.0), ("gY-", 3.5), ("gY-", 5.0)]


def cross_increments_rep(c, g3):
    """UC.cross_increments with a repair of the SU(2)/U(1) variational Newton at large r: diag (run02) showed that for
    g_Y/g_2 <~ 0.75 the Newton of the cross-term functional diverges for r >~ 12-17 (|grad| up to 1e12, D = kb - c^2 < 0,
    beta = c/k runs away, Om2 + Om_X^2 -> -1e16..-1e46): the term (3/4) H33 b/D with H33 < 0 is unbounded as D -> 0+.
    Repair: first r > 0.05 where (non-finite | |grad| > 1e-6 | D <= 0 | Om2' + Om_X'^2 <= 0) = r_b; for r >= r_b the
    parameters are P1 = P0 + [P1 - P0](r_b^-) (constant cross offset; r_b is in the outgoing region)."""
    from scipy.interpolate import CubicSpline
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = UC.PV11["rf"]
    rg = UC.T.RX
    rows = [UC.K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows])(rr)
    P0, _ = UC.newton_safe(lambda p: UC.etot_g(p, rf, g2, gY), UC.PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    Pst = P0.copy()
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    with np.errstate(all="ignore"):
        P1, gres = UC.newton_safe(lambda p: UC.etot_g(p, rf, g2, gY) + UC.V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
        D1 = P1[1] * P1[2] - P1[3] ** 2
        b1 = P1[3] / P1[1]
        om1 = (rf * rf / 4 * (gY - g2 * b1) ** 2 + 2 * g2 * g2 * b1 ** 2 / P1[0] + 4 * P1[0] * b1 ** 2 / (rf * rf)
               + Hs["Hww"] - 2 * b1 * Hs["H3w"] + b1 ** 2 * Hs["H33"])
        bad = ~np.isfinite(P1).all(axis=0) | ~np.isfinite(gres) | (gres > 1e-6) | ~(D1 > 0) | ~(om1 > 0)
    bad &= rf > 0.05
    rb = None
    if bad.any():
        ib = int(np.argmax(bad))
        rb = float(rf[ib])
        P1[:, ib:] = P0[:, ib:] + (P1[:, ib - 1] - P0[:, ib - 1])[:, None]
    q0 = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    q1 = W.derived(rf, P1, np.gradient(P1, rf, axis=1, edge_order=2), g2, gY)
    fsu2 = UC.V.fcross(P1, Hs, "SU2")
    beta = q1["beta"]
    omx_nl = Hs["Hww_nl"] - 2 * beta * Hs["H3w_nl"] + beta ** 2 * Hs["H33_nl"]
    kap = np.array([x["kappa3"] for x in rows])
    k0 = (6 * UC.X.curv(0.0, "P")[0] - 12 * UC.X.curv(0.0, "A")[0]) / (4 * math.pi ** 2)
    E0, s0 = 12.980 * g3 ** (2 / 3), 3.66 * (g3 / 0.5396) ** (2 / 3)
    Ev, _ = UC.V.su3_var(kap, E0, s0)
    Ev0, _ = UC.V.su3_var(np.array([k0]), E0, s0)
    r2 = rf * rf
    out = dict(dC0=CubicSpline(rf, r2 * (q1["C0"] + fsu2 - q0["C0"])), dBx=CubicSpline(rf, r2 * (q1["Bx"] - q0["Bx"])),
               dOm2e=CubicSpline(rf, q1["Om2e"] - q0["Om2e"]), omx_nl=CubicSpline(rf, omx_nl),
               beta=CubicSpline(rf, beta), dU3=CubicSpline(rg, Ev - Ev0[0]))
    return out, dict(r_b=rb, n_bad=int(bad.sum()), grad_max_below_rb=float(np.max(gres[(rf > 0.05) & (rf < (rb or 99))])))


def build(spec, sg):
    u, var, fg = float(spec["u"]), spec["var"], spec["f"]
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    P = len(spec["cols"])
    U1, OM = np.empty((P, len(r2))), np.empty((P, len(r2)))
    ph0, ph1 = np.empty(P), np.empty(P)
    Wl, betas, cs, reps, EX = [], [], [], [], np.empty(P, complex)
    for j, mt in enumerate(spec["cols"]):
        mu = W.mu_node(u)
        c0 = CS(mt, 0.1180, mu)
        c = dict(c0, g2=c0["g2"] * fg[0], gY=c0["gY"] * fg[1], yt=c0["yt"] * fg[2], lam=c0["lam"] * fg[3])
        g3 = UC.g3_at(mt, 0.1180, mu)
        Pp, _ = UC.base_parts(mt, 0.1180, u, var, c["g2"], c["gY"], fg)
        U0, OM0 = W.build2d(Pp, dmc, r2, c, "2D_ex")
        inc, rep = cross_increments_rep(c, g3)
        reps.append(rep)
        rr2 = r2 * r2
        U1[j] = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        betas.append(inc["beta"](r2))
        cs.append(c)
        Wl.append(UC.V_light_c(r2, sg.s, betas[-1], c["g2"], c["gY"]))
        ph0[j], ph1[j], _, _ = W.phases(U1[j], r2)
        EX[j] = UC.x02_root(mt, u)
    return dict(U1=U1, OM=OM, Wl=Wl, betas=betas, cs=cs, r2=r2, m2=m2, h=h, cfg=cfg, ph0=ph0, ph1=ph1, EX=EX, reps=reps)


def deflated(fun, zs):
    def g(E):
        v = fun(E)
        for z in zs:
            v = np.where(np.isfinite(z), v / (E - np.where(np.isfinite(z), z, 0.0)), v)
        return v
    return g


def job(key):
    var, u = key
    tag = "V_%s_u%.4f" % (var, u)
    fn = os.path.join(SH2, tag + ".json")
    if os.path.exists(fn):
        return json.load(open(fn, encoding="utf-8"))
    t0 = time.time()
    old = json.load(open(os.path.join(SH1, tag + ".json"), encoding="utf-8"))
    spec = dict(var=var, u=u, f=tuple(old["f"]), cols=[x["mt"] for x in old["cols"]])
    cen = np.array([x["center"] for x in old["cols"]])
    gim = np.array([x["gim"] for x in old["cols"]])
    cfg = dict(W.FAST)
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    B = build(spec, sg)
    P = len(cen)
    fun, cnt = UC.make_funW(B["U1"], B["OM"], B["Wl"], B["r2"], B["m2"], B["h"], sg, int(cfg["order"]), B["ph0"], B["ph1"])
    offs = np.round(np.arange(-0.05, 0.05 + 1e-3, 0.0025), 6)
    av = np.array([np.abs(fun(cen + o + 0j)) for o in offs])            # (n_off, P)
    i0 = int(np.argmin(np.abs(offs)))
    mins = []
    for j in range(P):
        a_ = av[:, j]
        loc = [i for i in range(1, len(offs) - 1) if a_[i] < a_[i - 1] and a_[i] < a_[i + 1]]
        mins.append(sorted(loc, key=lambda i_: abs(i_ - i0)))
    E = np.full(P, np.nan + 0j)
    ok = np.zeros(P, bool)
    zs, log = [], [[] for _ in range(P)]
    for rank in range(3):                                   # local minimum rank (closest first)
        if ok.all():
            break
        start = np.array([cen[j] + offs[mins[j][min(rank, len(mins[j]) - 1)]] - 0.5j * gim[j] if mins[j] else
                          cen[j] - 0.5j * gim[j] for j in range(P)])
        zs = []
        for ps in range(4):
            g = deflated(fun, zs)
            Ei, it, act, _ = W.muller2(g, start - 0.001, start + 0.001, start + 0j)
            gam = -2 * Ei.imag
            real0 = np.isfinite(Ei) & (np.abs(Ei.imag) < 1e-13)
            good = (np.isfinite(Ei) & ~act & (gam > 0) & ~real0 & (np.abs(np.log(np.maximum(gam, 1e-300) / gim)) < 1.5)
                    & (np.abs(Ei.real - cen) < 0.01))
            for j in range(P):
                if not ok[j]:
                    log[j].append(dict(rank=rank, pass_=ps, E=[float(Ei[j].real), float(Ei[j].imag)], real_zero=bool(real0[j]),
                                       accepted=bool(good[j])))
                    if good[j]:
                        E[j], ok[j] = Ei[j], True
            if ok.all():
                break
            zs.append(np.where(real0 & ~ok, Ei.real, np.nan))  # deflate the real zeros of the still-open columns
    # box check (28, 14)
    sgb = W.SGrid(W.BOXCHK["N"], W.BOXCHK["Smax"])
    Wlb = [UC.V_light_c(B["r2"], sgb.s, B["betas"][j], B["cs"][j]["g2"], B["cs"][j]["gY"]) for j in range(P)]
    fb, cb = UC.make_funW(B["U1"], B["OM"], Wlb, B["r2"], B["m2"], B["h"], sgb, int(cfg["order"]), B["ph0"], B["ph1"])
    Eok = np.where(ok, E, cen - 0.5j * gim)
    Eb, itb, actb, _ = W.muller2(fb, Eok - 1e-4, Eok + 1e-4, Eok + 0j, maxit=25)
    rows = []
    for j in range(P):
        gam = float(-2 * E[j].imag) if ok[j] else None
        bdE = float(abs(Eb[j] - E[j])) if ok[j] else None
        rows.append(dict(mt=spec["cols"][j], u=u, var=var, ok=bool(ok[j]), E=[float(E[j].real), float(E[j].imag)] if ok[j] else None,
                         gamma=gam, rho=float(E[j].imag / B["EX"][j].imag) if ok[j] else None,
                         dQ2=float(E[j].real - B["EX"][j].real) if ok[j] else None, center=float(cen[j]), gim=float(gim[j]),
                         box_dE=bdE, box_ok=bool(ok[j] and not actb[j] and bdE < max(2e-7, 2e-3 * gam)),
                         n_local_min=len(mins[j]), first_min_offset=float(offs[mins[j][0]]) if mins[j] else None,
                         repair=B["reps"][j], log=log[j]))
    out = dict(tag=tag, var=var, u=u, cols=rows, time_s=time.time() - t0, n_eval=cnt["n"] + cb["n"], scan=dict(half=0.05, step=0.0025))
    os.makedirs(SH2, exist_ok=True)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(W.clean(out), f, indent=1)
    print("[F %s] %.0fs %s" % (tag, time.time() - t0, "; ".join(
        "%.1f %s rho=%s box=%s" % (x["mt"], "OK" if x["ok"] else "FAIL", "%.4f" % x["rho"] if x["rho"] else "-",
                                    "ok" if x["box_ok"] else "x") for x in rows)), flush=True)
    return out


def rb_table():
    """repair onset r_b for every column of every node of the two affected variants (cheap; profile-free)."""
    CS = W.Couplings(W.KIND)
    rbt = {}
    for var in ("g2+", "gY-"):
        fg = UC.VAR[var]
        for u in UC.NODES:
            for mt in UC.MTV:
                c0 = CS(mt, 0.1180, W.mu_node(u))
                c = dict(c0, g2=c0["g2"] * fg[0], gY=c0["gY"] * fg[1], yt=c0["yt"] * fg[2], lam=c0["lam"] * fg[3])
                rbt["%s|%.4f|%.2f" % (var, u, mt)] = cross_increments_rep(c, UC.g3_at(mt, 0.1180, W.mu_node(u)))[1]
    with open(os.path.join(R2, "repair_rb.json"), "w", encoding="utf-8") as f:
        json.dump(rbt, f, indent=1)
    print("r_b:", json.dumps({k: v["r_b"] for k, v in rbt.items()}), flush=True)


def run():
    os.makedirs(SH2, exist_ok=True)
    rb_table()
    with ProcessPoolExecutor(max_workers=3) as ex:
        list(ex.map(job, JOBS))


def post():
    UN11 = list(W.UN11)
    G, LAMB = {}, {}
    for u in UN11:
        d = W.jload(os.path.join(UC.X02, "X_main_u%.4f.json" % u))
        for x in d["cols"]:
            G[(round(x["mt"], 2), round(u, 4))] = x["gamma"]
            if u == 0.0:
                LAMB[round(x["mt"], 2)] = x["lam"]
    from scipy.interpolate import CubicSpline
    mall = sorted(LAMB)
    lam_b = CubicSpline(mall, [LAMB[m] for m in mall])
    MTV = UC.MTV
    rho = {}
    for f in sorted(glob.glob(os.path.join(SH1, "V_*.json"))):
        if f.endswith("_bad.json"):
            continue
        d = W.jload(f)
        for x in d["cols"]:
            rho[(d["var"], round(x["mt"], 2), round(x["u"], 4))] = x["rho"] if UC.okcol(x) else None
    fixed = {}
    for var, u in JOBS:
        d = W.jload(os.path.join(SH2, "V_%s_u%.4f.json" % (var, u)))
        for x in d["cols"]:
            k = (var, round(x["mt"], 2), round(u, 4))
            fixed[k] = dict(rho=x["rho"], ok=x["ok"], box_ok=x["box_ok"])
            rho[k] = x["rho"] if x["ok"] else None
    r01 = json.load(open(os.path.join(R1, "C156_coup.json"), encoding="utf-8"))
    for k in fixed:                                         # run01 1D-guided substitute for comparison
        sub = [s for s in r01["variants"][k[0]]["substituted"] if abs(s[0] - k[1]) < 1e-6 and abs(s[1] - k[2]) < 1e-6]
        fixed[k]["rho_run01_1D_guided"] = sub[0][2] if sub else None
        fixed[k]["rel_diff"] = (fixed[k]["rho"] / fixed[k]["rho_run01_1D_guided"] - 1) if (sub and fixed[k]["rho"]) else None
    Ib = np.array([UC.integ(UN11, [G[(m, round(u, 4))] for u in UN11]) for m in MTV])
    base = {k: UC.crit(MTV, Ib, t) for k, t in W.TGT.items()}
    haz, dlnI = {}, {}
    for v in ("g2+", "g2-", "gY+", "gY-", "yt+", "yt-"):
        Iv = []
        for m in MTV:
            us = [u for u in UC.NODES if rho.get((v, round(m, 2), round(u, 4))) is not None]
            lr = CubicSpline(us, [math.log(rho[(v, round(m, 2), round(u, 4))]) for u in us])
            Iv.append(UC.integ(UN11, [G[(m, round(u, 4))] * math.exp(lr(u)) for u in UN11]))
        Iv = np.array(Iv)
        mc = {k: UC.crit(MTV, Iv, t) for k, t in W.TGT.items()}
        haz[v] = {k: dict(mc=mc[k], lam_c=float(lam_b(mc[k])), dlam_c=float(lam_b(mc[k]) - lam_b(base[k]))) for k in W.TGT}
        dlnI[v] = np.log(Iv / Ib)
    eta = json.load(open(os.path.join(R1, "C156_eta.json"), encoding="utf-8"))
    summ = json.load(open(os.path.join(R1, "C156_summary.json"), encoding="utf-8"))
    lv = np.array([abs(LAMB[round(m, 2)]) for m in MTV])
    dl = math.log(1.1) - math.log(0.9)
    der = {}
    for g in ("g2", "gY", "yt"):
        s = (dlnI[g + "+"] - dlnI[g + "-"]) / dl
        A = np.vstack([np.ones_like(lv), -lv ** -0.5]).T
        (alpha, beta), *_ = np.linalg.lstsq(A, s, rcond=None)
        der[g] = dict(alpha=float(alpha), beta=float(beta), s_columns=s.tolist())
        for k in W.TGT:
            be, lc = eta["b_eff_hazard"][k]["b_eff"], abs(eta["b_eff_hazard"][k]["lam_c"])
            Sc = be / math.sqrt(lc)
            dlam = (haz[g + "+"][k]["lam_c"] - haz[g + "-"][k]["lam_c"]) / dl
            d01 = summ["couplings"]["wkb_check"][g][k]
            der[g][k] = dict(dlamc_dlng=dlam, dlnlamc_dlng=-dlam / lc, dlnlamc_dlng_wkb=2 * beta / be - 2 * alpha / Sc,
                             dlnb_dlng=beta / be, dmc_dlng=(haz[g + "+"][k]["mc"] - haz[g + "-"][k]["mc"]) / dl,
                             run01_direct=d01["dlnlamc_dlng_direct"], run01_wkb=d01["dlnlamc_dlng_wkb"],
                             diff_direct=-dlam / lc - d01["dlnlamc_dlng_direct"],
                             asym=[haz[g + "+"][k]["dlam_c"], haz[g + "-"][k]["dlam_c"]])
    out = dict(fixed_nodes={"%s|%.2f|%.4f" % k: v for k, v in fixed.items()}, hazard=haz, derivatives=der,
               base_5col=base, note="run01 untouched; rho at the 5 re-solved (variant, node) jobs replace the 1D-guided values")
    with open(os.path.join(R2, "C156_run02.json"), "w", encoding="utf-8") as f:
        json.dump(W.clean(out), f, indent=1)
    print(json.dumps(W.clean(dict(fixed={"%s|%.2f|%.4f" % k: [v["rho"], v["rho_run01_1D_guided"], v["rel_diff"], v["box_ok"]]
                                         for k, v in fixed.items()},
                                  der={g: {k: {x: der[g][k][x] for x in ("dlnlamc_dlng", "run01_direct", "diff_direct",
                                                                           "dlnlamc_dlng_wkb", "run01_wkb", "dmc_dlng")}
                                           for k in W.TGT} for g in der})), indent=1))


def diag():
    """where does the SU(2)/U(1) variational Newton of cross_increments go non-finite for the failing variants?"""
    CS = W.Couplings(W.KIND)
    rf = UC.PV11["rf"]
    out = {}
    for var, u, mt in (("g2+", 2.0, 172.0), ("g2+", 3.5, 172.0), ("gY-", 1.0, 172.0), ("gY-", 2.0, 172.0), ("gY-", 5.0, 173.6)):
        fg = UC.VAR[var]
        c0 = CS(mt, 0.1180, W.mu_node(u))
        c = dict(c0, g2=c0["g2"] * fg[0], gY=c0["gY"] * fg[1], yt=c0["yt"] * fg[2], lam=c0["lam"] * fg[3])
        g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
        rg = UC.T.RX
        rows = [UC.K.hessian(float(x), lam, g2, gY, yt) for x in rg]
        rr = np.minimum(rf, rg[-1])
        from scipy.interpolate import CubicSpline
        Hs = {}
        for kk in ("H11", "H33", "H3w", "Hww"):
            Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])(rr)
        P0, g0 = UC.newton_safe(lambda p: UC.etot_g(p, rf, g2, gY), UC.PV11["P"].astype(float))
        qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
        omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
        Pst = P0.copy()
        Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
        P1, g1 = UC.newton_safe(lambda p: UC.etot_g(p, rf, g2, gY) + UC.V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
        bad0 = ~np.isfinite(P0).all(axis=0)
        bad1 = ~np.isfinite(P1).all(axis=0) | ~np.isfinite(g1)
        D1 = P1[1] * P1[2] - P1[3] ** 2
        out["%s|u%.1f|%.1f" % (var, u, mt)] = dict(
            gY_over_g2=gY / g2, n_rf=len(rf), n_bad_P0=int(bad0.sum()), n_bad_P1=int(bad1.sum()),
            r_bad_P1=[float(rf[bad1].min()), float(rf[bad1].max())] if bad1.any() else None,
            min_D1_finite=float(np.nanmin(np.where(np.isfinite(D1), D1, np.nan))),
            max_grad_r_gt_0p05=float(np.nanmax(np.where(rf > 0.05, g1, np.nan))),
            omx_min=float(np.min(omx)), Om2_plus_omx_min=float(np.min(qq["Om2"] + omx)),
            r_at_Om2omx_min=float(rf[int(np.argmin(qq["Om2"] + omx))]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "diag":
        diag()
    if md == "run":
        run()
    elif md == "post":
        post()
