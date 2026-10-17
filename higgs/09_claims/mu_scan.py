# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/mu_scan.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.  It IMPORTS UNCHANGED (read
#   only; always run with python -B so nothing is written there) 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/
#   C149_direct5.py and, through it, C149_direct4 / C149_direct / C150_twod / C138_core / cross_* / C149_coulomb.
#   job() is C149_direct.jobX line by line except: (a) node RG scale mu = c * mu_b/sqrt(cosh 2u) for ALL couplings
#   (tree lambda, one-loop dV, gauge profile, cross terms, g3); (b) the conditional-Gaussian gauge profile is recomputed
#   with g(c mu) by C150_twod.compute_profile (cached in my results dir; also for c = 1, regression-tested against the
#   T14b profile files used by run05); (c) the MC+CW matching dV is MS-bar subtracted at (mu L)^2 = c^2 and the Higgs
#   field normalisation xi = exp(-gamma_phi^C ln c) is applied to lam r^4/4 and dV (r sector, D110 Sec. 3.1-3.2);
#   (d) top-threshold variants ct: y_t in all top loops (dV top + cross-term top pieces) moved to ct*mu, explicit logs
#   kept at mu (D110 Sec. 3.4 extended to the cross terms); (e) root seeds = run05 roots (same window as run05).
#   All output goes to results/20261015_claude_v12_writing/claims_run05/run01/mu_scan/.
r"""[v12 package, 2026-10-15] Matching-scale scan in the run05 node pipeline: mu = c/L with c in {1/2, 1, 2, 2 pi} (top threshold c_t in {1/2, 2}); 66 node jobs; c = 1 reproduces run05.  Feeds the Table 1 'one-loop matching' row.  Paper: PRL Table 1. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

Consistent matching-scale scan of the run05 node-direct 2D hazard (Table 1 item 'one-loop matching').

r^2 c^2 sector (Higgs - gauge-zero-mode terms): the one-loop logs of vev mixing + Coulomb mismatch + top chiral close to
2 gamma_phi^C x tree (D121 run05, >= 97 %), so moving the cross-term subtraction scale AND the Higgs normalisation of the
tree r^2 c^2 term together changes nothing at one loop; here neither is moved (identical at one loop, residual <= 3 % x
2 gamma_phi^C ln c x tree, negligible).  The r sector (lam r^4/4 + dV) is moved consistently as in D110.
Usage: python -B mu_scan.py regress | run <var> [<var> ...] | post        vars: c1 c0.5 c2 c2pi ct0.5 ct2
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
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402

ROOT = _zp.WORK
HX = os.path.join(ROOT, _zp.ZMLIB)
OUT = os.path.join(ROOT, _zp.w("w/claims_run05/run01/mu_scan"))
SH = os.path.join(OUT, "shards")
PROF = os.path.join(OUT, "profiles")
RUN05 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run05"))
T14PROF = os.path.join(ROOT, _zp.w("r3/twod_direct_sm/run02/profiles"))
sys.path.insert(0, HX)
sys.path.insert(0, _zp.step("04_cross_terms"))  # [v12-package] C149_direct5
os.environ.setdefault("C149_V", "A")
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import C149_direct5 as D5  # noqa: E402

D, W, K, X, V, D4 = D5.D, D5.W, D5.K, D5.X, D5.V, D5.D4
C = W.C
# redirect every module-level output path of the imported pipeline to my directory (nothing may be written to old runs)
for _m, _a in ((W, ("RES", "PROFDIR", "SH1", "SH2")), (D, ("SH", "RES2", "MYPROF")), (X, ("RES",)), (D.T.B, ("RES",))):
    for _x in _a:
        if hasattr(_m, _x):
            setattr(_m, _x, OUT)
VARS = {"c1": (1.0, None), "c0.5": (0.5, None), "c2": (2.0, None), "c2pi": (2 * math.pi, None),
        "ct0.5": (1.0, 0.5), "ct2": (1.0, 2.0)}
_SEEDS = {}


def seeds():
    if not _SEEDS:
        for f in glob.glob(os.path.join(RUN05, "shards", "X_*.json")):
            for x in W.jload(f)["cols"]:
                _SEEDS[(round(x["als"], 4), round(x["mt"], 2), round(x["u"], 4))] = x
    return _SEEDS


def profile(g2, gY, key):
    fn = os.path.join(PROF, key + ".npz")
    if os.path.exists(fn):
        d = dict(np.load(fn))
    else:
        d0 = W.compute_profile(g2, gY)
        d = {k: np.asarray(d0[k]) for k in ("rf", "C0", "Bx", "Om2e", "EtBr", "g2", "gY")}
        os.makedirs(PROF, exist_ok=True)
        tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
        np.savez(tmp, **d)
        os.replace(tmp, fn)
    return W.Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]), d


def build2d_c(Pp, dmc, r, cc, c, xi, yt_top=None):
    z = xi * r
    U = (3.0 / (8 * r * r) + Pp.c0bx(r) + cc["lam"] * z ** 4 / 4
         + C.dV("MC", z, cc["lam"], cc["g2"], cc["gY"], cc["yt"], mu2=c * c, yt_top=yt_top) + dmc(r))
    return U, Pp.om2(r)


def job(spec):
    fn = os.path.join(SH, spec["tag"] + ".json")
    if os.path.exists(fn):
        return W.jload(fn)
    t0 = time.time()
    u, c, ct = float(spec["u"]), float(spec["c"]), spec.get("ct")
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    cols = spec["cols"]
    P = len(cols)
    U1, OM = (np.empty((P, len(r2))) for _ in range(2))
    ph0, ph1 = np.empty(P), np.empty(P)
    Wl, rows, cen, gim = [], [], np.empty(P), np.empty(P)
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    S5 = seeds()
    for j, col in enumerate(cols):
        mt, als = col["mt"], col["als"]
        mu = c * W.mu_node(u)
        cc = CS(mt, als, mu)
        g3 = D.g3_at(mt, als, mu)
        gam = C.gamma_phi(cc["g2"], cc["gY"], cc["yt"], "C")
        xi = math.exp(-gam * math.log(c))
        Pp, pd = profile(cc["g2"], cc["gY"], "prof_c%.6f_mt%.2f_as%.4f_u%.4f" % (c, mt, als, u))
        prof_chk = None
        t14 = os.path.join(T14PROF, "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
        if abs(c - 1.0) < 1e-12 and os.path.exists(t14):
            dt = dict(np.load(t14))
            prof_chk = {k: float(np.max(np.abs(dt[k] - pd[k]))) for k in ("C0", "Bx", "Om2e", "EtBr")}
        cct, yt_top = dict(cc), None
        if ct is not None:
            yt_top = float(CS(mt, als, ct * mu)["yt"])
            cct["yt"] = yt_top
        U0, OM0 = build2d_c(Pp, dmc, r2, cc, c, xi, yt_top)
        inc, info = D.cross_increments(cct, g3)
        rr2 = r2 * r2
        U1[j] = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        Wl.append(D.V_light_c(r2, sg.s, inc["beta"](r2), cc["g2"], cc["gY"]))
        ph0[j], ph1[j], _, _ = W.phases(U1[j], r2)
        s5 = S5[(round(als, 4), round(mt, 2), round(u, 4))]
        cen[j], gim[j] = s5["Q"], s5["gamma"]
        rows.append(dict(mt=mt, als=als, u=u, c=c, ct=ct, mu=mu, xi=xi, gamma_phi_C=gam, g2=cc["g2"], gY=cc["gY"], yt=cc["yt"],
                         yt_top=yt_top, lam=cc["lam"], g3=g3, info=info, prof_maxdiff_vs_T14b=prof_chk,
                         Q_run05=s5["Q"], gamma_run05=s5["gamma"]))
    fun, cnt = D.make_funW(U1, OM, Wl, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    E, wd = W.window_roots(fun, cen, gim, D.WINDOW)
    for j in range(P):
        rows[j].update(E=[float(E[j].real), float(E[j].imag)], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                       nc=bool(wd["act"][j]), noloc=bool(wd["noloc"][j]),
                       edge=bool(abs(wd["Emin"][j] - cen[j]) >= D.WINDOW[0] - 1e-9),
                       dQ_vs_run05=float(E[j].real - rows[j]["Q_run05"]),
                       ratio_vs_run05=float(-2 * E[j].imag / rows[j]["gamma_run05"]))
    out = dict(tag=spec["tag"], var=spec["var"], u=u, c=c, ct=ct, cols=rows, cfg=cfg, window=D.WINDOW,
               time_s=time.time() - t0, n_eval=cnt["n"])
    W.wjson(fn, out)
    print("[%s] %.0fs %s" % (spec["tag"], time.time() - t0, "; ".join(
        "%.2f r=%.5f dQ=%+.2e%s%s" % (x["mt"], x["ratio_vs_run05"], x["dQ_vs_run05"], " NC" if x["nc"] else "",
                                      " EDGE" if x["edge"] else "") for x in rows)), flush=True)
    return out


def specs(var, us=None):
    c, ct = VARS[var]
    us = W.UN11 if us is None else us
    return [dict(tag="%s_u%.4f" % (var, u), var=var, u=float(u), c=c, ct=ct,
                 cols=[dict(mt=mt, als=0.1180) for mt in W.MT_MAIN]) for u in us]


def run(vs, nw=15, us=None):
    t0 = time.time()
    os.makedirs(SH, exist_ok=True)
    sp = [s for v in vs for s in specs(v, us)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(job, sp))
    print("run %s done: %d jobs %.0fs" % (vs, len(sp), time.time() - t0), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def post():
    S5 = W.jload(os.path.join(RUN05, "C149_run05_summary.json"))
    R = {}
    for f in glob.glob(os.path.join(SH, "*.json")):
        d = W.jload(f)
        for x in d["cols"]:
            R[(d["var"], round(x["mt"], 2), round(x["u"], 4))] = x
    out = dict(task="C161 mu_scan (consistent matching-scale variation in the run05 node pipeline)", vars={}, crit={}, P1={},
               flags=[list(k) for k, x in R.items() if x["nc"] or x["edge"] or not x["gamma"] > 0])
    for var in VARS:
        I, ok = [], []
        for mt in W.MT_MAIN:
            g = [R.get((var, round(mt, 2), round(float(u), 4))) for u in W.UN11]
            if any(x is None for x in g):
                continue
            I.append(W.integ(W.UN11, np.array([x["gamma"] for x in g])))
            ok.append(mt)
        if len(ok) < 4:
            continue
        cX, spX = W.crit(ok, I)
        out["crit"][var] = cX
        out["P1"][var] = {f"{m}": float(math.exp(-math.exp(spX(m)))) for m in (172.2, 172.6, 173.0)}
        out["vars"][var] = dict(c=VARS[var][0], ct=VARS[var][1], n_mt=len(ok),
                                max_abs_dQ_vs_run05=max(abs(R[(var, round(m, 2), round(float(u), 4))]["dQ_vs_run05"])
                                                        for m in ok for u in W.UN11),
                                max_abs_lnratio_vs_run05=max(abs(math.log(R[(var, round(m, 2), round(float(u), 4))]["ratio_vs_run05"]))
                                                             for m in ok for u in W.UN11))
    base = out["crit"].get("c1")
    if base:
        out["c1_minus_run05"] = {k: base[k] - S5["crit"]["0.1180"][k] for k in base}
        out["dmc_vs_c1"] = {v: {k: out["crit"][v][k] - base[k] for k in base} for v in out["crit"]}
        d = out["dmc_vs_c1"]
        if all(v in d for v in ("c0.5", "c2", "c2pi")):
            out["mu_envelope"] = {k: dict(plus_half_2=max(0.0, d["c0.5"][k], d["c2"][k]), minus_half_2=min(0.0, d["c0.5"][k], d["c2"][k]),
                                          plus_half_2pi=max(0.0, d["c0.5"][k], d["c2"][k], d["c2pi"][k]),
                                          minus_half_2pi=min(0.0, d["c0.5"][k], d["c2"][k], d["c2pi"][k])) for k in base}
        if all(v in d for v in ("ct0.5", "ct2")):
            out["top_threshold"] = {k: dict(ct0p5=d["ct0.5"][k], ct2=d["ct2"][k],
                                            half_spread=0.5 * abs(d["ct0.5"][k] - d["ct2"][k])) for k in base}
    out["D110_reference_P1_0.9_0.5"] = dict(mu_c_half_2=[[0.002, -0.008], [0.005, -0.011]], mu_c_half_2pi=[[0.002, -0.034], [0.005, -0.043]],
                                            top_threshold=[0.009, 0.012], src="D110 Sec. 3.3-3.4 / Sec. 5 table (1D MC+CW, no cross terms)")
    out["code_sha256"] = {"mu_scan.py": sha(os.path.abspath(__file__)),
                          **{f: sha(os.path.join(HX if os.path.exists(os.path.join(HX, f)) else _zp.step("04_cross_terms"), f)) for f in ("C149_direct5.py", "C149_direct4.py", "C149_direct.py",
                                                                   "C150_twod.py", "C138_core.py", "C149_coulomb.py")}}
    W.wjson(os.path.join(OUT, "mu_scan_summary.json"), out)
    print(json.dumps(W.clean({k: v for k, v in out.items() if k != "code_sha256"}), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "regress":
        run(["c1"], nw=2, us=[0.0, 2.5])
    elif md == "run":
        run(sys.argv[2:])
    elif md == "post":
        post()
    else:
        raise SystemExit(__doc__)
