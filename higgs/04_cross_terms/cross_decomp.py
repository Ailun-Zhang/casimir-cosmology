# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/cross_decomp.py (copied 2026-10-16 by rev-code, R18 R1/MH-02; original SHA-256 711244c5328d13c39e4e8d568a1ba9ab519014482b36919ac168ff496bf243cd); only the path block is changed ([v12-package] lines).
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.  Imports UNCHANGED (read only,
#   python -B) holonomy_cross/C149_direct5.py (-> C149_direct4/C149_direct/C150_twod/...); cross_increments_var() below is
#   C149_direct5.cross_increments5 copied line by line with switches added; job() is C149_direct.jobX with T14b base
#   profiles (read only), run05 roots as seeds, output only to results/20261015_claude_v12_writing/claims_run05/run01/cross_decomp/.
r"""Step 6 (optional): zero-mode-field decomposition of the run05 cross-term shift of m_c (P1 = 0.9/0.5/0.1, alpha_s = 0.1180).
Switch definitions (by the gauge factor of the ZERO MODE that the Higgs fluctuation loops couple to):
  noY   : U(1)_Y zero mode w off: variational uses fcross 'SU2' only (no Hww, H3w, no c^2 H33/k correlation term),
          Om_X^2(w) = 0, exact light-fermion twist V_L(w) off;
  noSU2 : SU(2) zero mode x off: H11 = H33 = 0 everywhere (also in the Y-part correlation term and in Om_X^2);
  noSU3 : SU(3)-top variational dU3 off;
  novevC: new-vertex transverse vev mixing + Coulomb mismatch (ext pieces) off, top chiral kept;
  notop : exact top chiral bubble (ext top_G pieces) off, vev mixing + Coulomb kept.
All-off = T14b (no cross terms; run05 post 'T14b_recomputed').  Two removal orders: (a) Y -> SU2 -> SU3, (b) SU3 -> SU2 -> Y.
Usage: python -B cross_decomp.py run [var ...] | post        vars: noY noY_noSU2 noSU3 noSU3_noSU2 novevC notop"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402

_ZM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in sys.path:  # [v12-package]
    sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
HX = os.path.dirname(os.path.abspath(__file__))  # [v12-package] C149_direct5.py of this step
OUT = _zp.w("w/claims_run05/run01/cross_decomp")  # [v12-package] work tree
SH = os.path.join(OUT, "shards")
RUN05 = _zp.w("r3/holonomy_cross/run05")  # [v12-package] work tree
sys.path.insert(0, HX)
os.environ.setdefault("C149_V", "A")
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import C149_direct5 as D5  # noqa: E402

D, W, K, X, V, D4, CO = D5.D, D5.W, D5.K, D5.X, D5.V, D5.D4, D5.CO
for _m, _a in ((W, ("RES", "PROFDIR", "SH1", "SH2")), (D, ("SH", "RES2", "MYPROF")), (X, ("RES",)), (D.T.B, ("RES",))):
    for _x in _a:
        if hasattr(_m, _x):
            setattr(_m, _x, OUT)
PI = math.pi
VARS = {"noY": {"Y"}, "noY_noSU2": {"Y", "SU2"}, "noSU3": {"SU3"}, "noSU3_noSU2": {"SU3", "SU2"}, "novevC": {"vevC"}, "notop": {"topch"}}
_SEEDS = {}


def seeds():
    if not _SEEDS:
        for f in glob.glob(os.path.join(RUN05, "shards", "X_main_*.json")):
            for x in W.jload(f)["cols"]:
                _SEEDS[(round(x["mt"], 2), round(x["u"], 4))] = x
    return _SEEDS


def ext_grid_var(c, off):
    """D5.ext_grid with the vev+Coulomb and top-chiral groups switchable."""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    gZ2 = g2 * g2 + gY * gY
    RC = D5.RC
    out = {k: np.zeros(len(RC)) for k in ("H11", "H33", "Hww", "H3w")}
    fv = 0.0 if "vevC" in off else 1.0
    ft = 0.0 if "topch" in off else 1.0
    for i, r in enumerate(RC):
        dWW, dZZ, _, _ = CO.vev_new(r, g2, gY, lam)
        cz = D5.coul_Z(r, g2, gY, lam) if fv else 0.0
        cw = dWW * (cz / dZZ) if (fv and abs(dZZ) > 1e-14) else 0.0
        mt2 = yt * yt / 2 * r * r
        HZ = dZZ + cz
        out["H11"][i] = fv * (dWW + cw) + ft * 3 * g2 * g2 * D4.top_G("T1", mt2)
        out["H33"][i] = fv * g2 * g2 / gZ2 * HZ + ft * 3 * g2 * g2 * D4.top_G("T3", mt2)
        out["Hww"][i] = fv * gY * gY / gZ2 * HZ + ft * 3 * gY * gY * D4.top_G("Y", mt2)
        out["H3w"][i] = fv * g2 * gY / gZ2 * HZ + ft * 3 * g2 * gY * D4.top_G("H3Y", mt2)
    return out


def cross_increments_var(c, g3, off):
    """[copied from C149_direct5.cross_increments5; switches 'off' added]"""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = D.PV11["rf"]
    rg = D.T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    eg = ext_grid_var(c, off)
    ext = {k: CubicSpline(D5.RC, eg[k])(np.clip(rg, D5.RC[0], D5.RC[-1])) * np.where(rg < D5.RC[0], (rg / D5.RC[0]) ** 2, 1.0)
           for k in eg}
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, np.array([sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows]) + ext[kk])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, np.array([sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows]) + ext[kk])(rr)
    if "SU2" in off:
        for kk in ("H11", "H33", "H11_nl", "H33_nl"):
            Hs[kk] = 0.0 * Hs[kk]
    if "Y" in off:
        for kk in ("Hww", "H3w", "Hww_nl", "H3w_nl"):
            Hs[kk] = 0.0 * Hs[kk]
    which = "SU2" if "Y" in off else "SU2+Y"
    P0, _ = V.newton(lambda p: D.etot_g(p, rf, g2, gY), D.PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    if "Y" in off:
        omx = 0.0 * omx
    Pst = P0.copy()
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    P1, gres = V.newton(lambda p: D.etot_g(p, rf, g2, gY) + V.fcross(p, Hs, which), Pst, iters=12)
    q0 = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    q1 = W.derived(rf, P1, np.gradient(P1, rf, axis=1, edge_order=2), g2, gY)
    fsu2 = V.fcross(P1, Hs, "SU2")
    beta = q1["beta"]
    omx_nl = Hs["Hww_nl"] - 2 * beta * Hs["H3w_nl"] + beta ** 2 * Hs["H33_nl"]
    if "Y" in off:
        omx_nl = 0.0 * omx_nl
    kap = np.array([x["kappa3"] for x in rows])
    k0 = (6 * X.curv(0.0, "P")[0] - 12 * X.curv(0.0, "A")[0]) / (4 * PI ** 2)
    E0, s0 = 12.980 * g3 ** (2 / 3), 3.66 * (g3 / 0.5396) ** (2 / 3)
    Ev, _ = V.su3_var(kap, E0, s0)
    Ev0, _ = V.su3_var(np.array([k0]), E0, s0)
    dU3 = (0.0 if "SU3" in off else 1.0) * (Ev - Ev0[0])
    r2 = rf * rf
    out = dict(dC0=CubicSpline(rf, r2 * (q1["C0"] + fsu2 - q0["C0"])), dBx=CubicSpline(rf, r2 * (q1["Bx"] - q0["Bx"])),
               dOm2e=CubicSpline(rf, q1["Om2e"] - q0["Om2e"]), omx_nl=CubicSpline(rf, omx_nl),
               beta=CubicSpline(rf, beta), dU3=CubicSpline(rg, dU3))
    return out, dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])))


def job(spec):
    fn = os.path.join(SH, spec["tag"] + ".json")
    if os.path.exists(fn):
        return W.jload(fn)
    t0 = time.time()
    u, off = float(spec["u"]), set(spec["off"])
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
        mu = W.mu_node(u)
        c = CS(mt, als, mu)
        g3 = D.g3_at(mt, als, mu)
        Pp, pinfo = D.base_parts(mt, als, u)            # T14b profile file (exists for all columns; read only)
        U0, OM0 = W.build2d(Pp, dmc, r2, c, "2D_ex")
        inc, info = cross_increments_var(c, g3, off)
        rr2 = r2 * r2
        U1[j] = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        Wl.append(None if "Y" in off else D.V_light_c(r2, sg.s, inc["beta"](r2), c["g2"], c["gY"]))
        ph0[j], ph1[j], _, _ = W.phases(U1[j], r2)
        s5 = S5[(round(mt, 2), round(u, 4))]
        if spec.get("wide"):     # large Q shifts (U(1)_Y / SU(2) removed; the +-0.02 window found no minimum): +-0.10 window
            cen[j] = s5["Q"]
            gim[j] = math.sqrt(s5["gamma"] * s5["gamma_T14b"])
        else:
            cen[j], gim[j] = s5["Q"], s5["gamma"]
        rows.append(dict(mt=mt, als=als, u=u, off=sorted(off), info=info, Q_run05=s5["Q"], gamma_run05=s5["gamma"],
                         gamma_T14b=s5["gamma_T14b"], Q_T14b=s5["E_T14b"][0]))
    fun, cnt = D.make_funW(U1, OM, Wl, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    win = D.WINDOW
    if spec.get("wide"):
        win = (0.10, 0.0025)
    E, wd = W.window_roots(fun, cen, gim, win)
    for j in range(P):
        rows[j].update(E=[float(E[j].real), float(E[j].imag)], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                       nc=bool(wd["act"][j]), noloc=bool(wd["noloc"][j]), edge=bool(abs(wd["Emin"][j] - cen[j]) >= win[0] - 1e-9),
                       window=list(win), center=float(cen[j]), dQ_vs_run05=float(E[j].real - rows[j]["Q_run05"]),
                       ratio_vs_run05=float(-2 * E[j].imag / rows[j]["gamma_run05"]))
    out = dict(tag=spec["tag"], var=spec["var"], u=u, off=sorted(off), cols=rows, time_s=time.time() - t0, n_eval=cnt["n"])
    W.wjson(fn, out)
    print("[%s] %.0fs %s" % (spec["tag"], time.time() - t0, "; ".join("%.2f r=%.4f%s%s" % (
        x["mt"], x["ratio_vs_run05"], " NC" if x["nc"] else "", " EDGE" if x["edge"] else "") for x in rows)), flush=True)
    return out


def run(vs, nw=15):
    t0 = time.time()
    os.makedirs(SH, exist_ok=True)
    sp = [dict(tag="%s_u%.4f" % (v, u), var=v, off=sorted(VARS[v]), u=float(u), cols=[dict(mt=mt, als=0.1180) for mt in W.MT_MAIN],
               wide=bool(v in ("noY", "noY_noSU2", "noSU3_noSU2"))) for v in vs for u in W.UN11]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(job, sp))
    print("run %s done: %d jobs %.0fs" % (vs, len(sp), time.time() - t0), flush=True)


def crit_ext(mts, I):
    """W.crit (cubic spline of ln I in m_t) with LINEAR extrapolation of ln I (end slopes) when a level lies outside the
    172.0-173.8 grid (flagged '<level>_extrapolated')."""
    from scipy.optimize import brentq
    x = np.asarray(mts, float)
    y = np.log(np.asarray(I, float))
    sp = CubicSpline(x, y)
    s0, s1 = float(sp(x[0], 1)), float(sp(x[-1], 1))

    def f(z):
        if z < x[0]:
            return float(y[0] + s0 * (z - x[0]))
        if z > x[-1]:
            return float(y[-1] + s1 * (z - x[-1]))
        return float(sp(z))
    out = {"P1_grid": {"%.1f" % m: float(math.exp(-math.exp(v))) for m, v in zip(x, y)}}
    for k, t in W.TGT.items():
        try:
            out[k] = float(brentq(lambda z: f(z) - math.log(t), x[0] - 1.5, x[-1] + 1.5, xtol=1e-10))
            out[k + "_extrapolated"] = bool(not (x[0] <= out[k] <= x[-1]))
        except ValueError:
            out[k] = None
    print("crit_ext", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in out.items() if kk != "P1_grid"},
          {kk: round(vv, 4) for kk, vv in out["P1_grid"].items()}, flush=True)
    return out


def post():
    S5 = W.jload(os.path.join(RUN05, "C149_run05_summary.json"))
    P5 = W.jload(os.path.join(RUN05, "post", "C149_run02_summary.json"))
    R = {}
    for f in glob.glob(os.path.join(SH, "*.json")):
        d = W.jload(f)
        for x in d["cols"]:
            R[(d["var"], round(x["mt"], 2), round(x["u"], 4))] = x
    crit = {"full": S5["crit"]["0.1180"], "none": P5["crit"]["0.1180"]["T14b_recomputed"]}
    flags = [list(k) for k, x in R.items() if x["nc"] or x["edge"] or not x["gamma"] > 0]
    for v in VARS:
        I, ok = [], []
        for mt in W.MT_MAIN:
            g = [R.get((v, round(mt, 2), round(float(u), 4))) for u in W.UN11]
            if any(x is None for x in g):
                continue
            I.append(W.integ(W.UN11, np.array([x["gamma"] for x in g])))
            ok.append(mt)
        if len(ok) >= 4:
            crit[v] = crit_ext(ok, I)
    L = ("P1=0.9", "P1=0.5", "P1=0.1")
    dd = lambda a, b: ({k: (crit[a][k] - crit[b][k]) if (crit[a].get(k) is not None and crit[b].get(k) is not None) else None  # noqa: E731
                        for k in L} if a in crit and b in crit else None)
    out = dict(task="C161 step 6: run05 cross-term decomposition by zero-mode field", crit=crit, flags=flags,
               total_full_minus_none=dd("full", "none"),
               order_a_Y_SU2_SU3=dict(U1Y=dd("full", "noY"), SU2_given_noY=dd("noY", "noY_noSU2"), SU3top_given_noY_noSU2=dd("noY_noSU2", "none")),
               order_b_SU3_SU2_Y=dict(SU3top=dd("full", "noSU3"), SU2_given_noSU3=dd("noSU3", "noSU3_noSU2"), U1Y_given_noSU3_noSU2=dd("noSU3_noSU2", "none")),
               one_loop_groups=dict(vev_mixing_plus_Coulomb=dd("full", "novevC"), top_chiral=dd("full", "notop")),
               definitions=__doc__)
    out["code_sha256"] = {"cross_decomp.py": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    W.wjson(os.path.join(OUT, "cross_decomp_summary.json"), out)
    print(json.dumps(W.clean({k: v for k, v in out.items() if k not in ("definitions", "code_sha256")}), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "run":
        run(sys.argv[2:] or list(VARS))
    elif md == "post":
        post()
    else:
        raise SystemExit(__doc__)
