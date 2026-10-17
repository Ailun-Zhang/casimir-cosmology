# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/higgs_robustness/C152_core.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent higgs-robustness (C152/D124, v12 round 3) on 2026-10-15.  It IMPORTS UNCHANGED (read only,
#   run with python -B so nothing is written there) 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct.py
#   and, through it, C150_twod.py / c150_nnlo_central.py / sm_provider.py / cross_*.py of the same directory.  jobS() is
#   C149_direct.jobX line by line; only (a) the node scale mu = mu_b e^l (instead of mu_b/sqrt(cosh 2u)), (b) the gauge
#   profile (freshly computed with the node couplings when the scale is not a baseline node) and (c) the root seed
#   (the C149 run02 with-cross root interpolated in l) differ.  All output goes to results/.../higgs_robustness/run01/.
r"""[v12 package, 2026-10-15] 2D_ex + NNLO + cross-term Siegert roots at an arbitrary RG scale l = ln(mu/mu_b): the background enters the Higgs chain only through delta_mu = dln mu_b and delta_h = dln hL_b (exact bounce scaling), so background variants need gamma(m_t, l_k + delta_mu).  jobS = C149_direct.jobX with a shifted node scale (zmlib).  Paper: PRD App. H3 (W4/W9 robustness), Table 1 rows '(vi) inflation scale'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C152 core: 2D_ex + NNLO + cross-term Siegert roots at arbitrary RG scale l = ln(mu/mu_b) (mu_b = MPbar/722).

gamma at a hazard node depends on the node only through the couplings at mu (xi_H = 1/6: no explicit hL_b), so a
background change (delta_mu = dln mu_b, delta_h = dln hL_b) needs gamma(m_t, l) at l_k + delta_mu, l_k = -1/2 ln cosh 2u_k.
Usage: python -B C152_core.py test | lgrid | direct <delta> [<delta> ...] | all
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

ROOT = _zp.WORK
HX = os.path.join(ROOT, _zp.ZMLIB)
sys.path.insert(0, HX)
if len(sys.argv) > 1 and sys.argv[1] == "setup":      # never trigger C149_direct.setup()
    raise SystemExit("no setup mode here")
import C149_direct as CD  # noqa: E402

W = CD.W
RES = os.path.join(ROOT, _zp.w("r3/higgs_robustness/run01"))
SH = os.path.join(RES, "shards")
PROF = os.path.join(RES, "profiles")
# redirect every writable path of the imported modules into my run directory
W.RES, W.PROFDIR, W.SH1, W.SH2 = RES, PROF, os.path.join(RES, "sh1"), os.path.join(RES, "sh2")
CD.RES2, CD.SH, CD.MYPROF = RES, SH, PROF
BASESH = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/shards"))
MU_B = float(W.RGE.MU_B)
UN11 = np.asarray(W.UN11, float)
L11 = -0.5 * np.log(np.cosh(2 * UN11))
MT_MAIN = list(W.MT_MAIN)
ALS0 = 0.1180
_BASE = {}


def base_table():
    """C149 run02 with-cross roots: {mt: (l_k, Q_k, gamma_k)} at alpha_s = 0.1180 (read only)."""
    if _BASE:
        return _BASE
    rows = {}
    for f in glob.glob(os.path.join(BASESH, "X_main_u*.json")):
        d = W.jload(f)
        for x in d["cols"]:
            if abs(x["als"] - ALS0) < 1e-9:
                rows.setdefault(round(x["mt"], 2), {})[round(x["u"], 4)] = x
    for mt, g in rows.items():
        us = sorted(g)
        l = np.array([-0.5 * math.log(math.cosh(2 * u)) for u in us])
        o = np.argsort(l)
        _BASE[mt] = (l[o], np.array([g[u]["Q"] for u in us])[o], np.array([g[u]["gamma"] for u in us])[o])
    return _BASE


def seed(mt, l):
    L, Q, G = base_table()[round(mt, 2)]
    lg = np.log(G)
    if L[0] <= l <= L[-1]:
        return float(CubicSpline(L, Q)(l)), float(math.exp(CubicSpline(L, lg)(l)))
    i0, i1 = (0, 1) if l < L[0] else (-2, -1)
    sQ = (Q[i1] - Q[i0]) / (L[i1] - L[i0])
    sG = (lg[i1] - lg[i0]) / (L[i1] - L[i0])
    ie = i0 if l < L[0] else i1
    return float(Q[ie] + sQ * (l - L[ie])), float(math.exp(lg[ie] + sG * (l - L[ie])))


def parts(mt, als, l, c, fresh):
    """gauge-profile parts: T14b run02 file at a baseline node (fresh=False) or recomputed with the node (g2, gY)."""
    if not fresh:
        k = int(np.argmin(np.abs(L11 - l)))
        assert abs(L11[k] - l) < 1e-9, "not a baseline node"
        return CD.base_parts(mt, als, float(UN11[k]))
    fn = os.path.join(PROF, "prof_mt%.2f_as%.4f_l%+.6f.npz" % (mt, als, l))
    if not os.path.exists(fn):
        d0 = W.compute_profile(c["g2"], c["gY"])
        os.makedirs(PROF, exist_ok=True)
        np.savez(fn, **{k: np.asarray(d0[k]) for k in ("rf", "C0", "Bx", "Om2e", "EtBr", "g2", "gY", "chk_Et")})
    d = dict(np.load(fn))
    return W.Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]), dict(g2=float(d["g2"]), gY=float(d["gY"]), file=fn)


def jobS(spec):
    fn = os.path.join(SH, spec["tag"] + ".json")
    if os.path.exists(fn):
        return W.jload(fn)
    t0 = time.time()
    l = float(spec["l"])
    mu = MU_B * math.exp(l)
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    cols = spec["cols"]
    P = len(cols)
    U1, OM = np.empty((P, len(r2))), np.empty((P, len(r2)))
    ph0, ph1 = np.empty(P), np.empty(P)
    Wl, rows, cen, gim = [], [], np.empty(P), np.empty(P)
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    tprof = 0.0
    for j, cc in enumerate(cols):
        mt, als = cc["mt"], cc["als"]
        c = CS(mt, als, mu)
        g3 = CD.g3_at(mt, als, mu)
        tp = time.time()
        Pp, pinfo = parts(mt, als, l, c, bool(spec.get("fresh", True)))
        tprof += time.time() - tp
        U0, OM0 = W.build2d(Pp, dmc, r2, c, "2D_ex")
        inc, info = CD.cross_increments(c, g3)
        rr2 = r2 * r2
        U1[j] = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM[j] = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        Wl.append(CD.V_light_c(r2, sg.s, inc["beta"](r2), c["g2"], c["gY"]))
        ph0[j], ph1[j], _, _ = W.phases(U1[j], r2)
        q0, g0 = seed(mt, l)
        cen[j], gim[j] = q0, g0
        rows.append(dict(mt=mt, als=als, l=l, mu=mu, g2=c["g2"], gY=c["gY"], yt=c["yt"], lam=c["lam"], g3=g3,
                         prof=pinfo["file"], prof_g2=pinfo["g2"], info=info, seed=[q0, g0]))
    fun, cnt = CD.make_funW(U1, OM, Wl, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    E, wd = W.window_roots(fun, cen, gim, CD.WINDOW)
    for j in range(P):
        rows[j].update(E=[float(E[j].real), float(E[j].imag)], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                       Emin=float(wd["Emin"][j]), noloc=bool(wd["noloc"][j]), nc=bool(wd["act"][j]),
                       edge=bool(abs(wd["Emin"][j] - cen[j]) >= CD.WINDOW[0] - 1e-9),
                       ratio_vs_seed=float(-2 * E[j].imag / gim[j]))
    out = dict(tag=spec["tag"], l=l, mu=mu, fresh=bool(spec.get("fresh", True)), cols=rows, cfg=cfg, window=CD.WINDOW,
               time_s=time.time() - t0, time_profiles_s=tprof, n_eval=cnt["n"])
    W.wjson(fn, out)
    print("[S %s] %.0fs (prof %.0fs) %s" % (spec["tag"], out["time_s"], tprof, "; ".join(
        "%.1f g=%.4e%s%s" % (x["mt"], x["gamma"], " NC" if x["nc"] else "", " EDGE" if x["edge"] else "")
        for x in rows)), flush=True)
    return out


def cols_main():
    return [dict(mt=mt, als=ALS0) for mt in MT_MAIN]


LGRID = (+0.28, -5.15, -5.60)


def specs_lgrid():
    return [dict(tag="L_l%+.4f" % l, l=float(l), cols=cols_main(), fresh=True) for l in LGRID]


def specs_direct(delta):
    return [dict(tag="D%+.4f_u%.4f" % (delta, u), l=float(lk + delta), cols=cols_main(), fresh=True)
            for u, lk in zip(UN11, L11)]


def run(sp, nw=int(os.environ.get("DNEC_WORKERS", "3"))):
    t0 = time.time()
    os.makedirs(SH, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(sp)))) as ex:  # [v12-package] was min(nw, 3)
        outs = list(ex.map(jobS, sp))
    print("run done: %d jobs %.0fs" % (len(outs), time.time() - t0), flush=True)


def test():
    out = {}
    B = base_table()
    for fresh in (False, True):
        o = jobS(dict(tag="TEST_l0_fresh%d" % int(fresh), l=0.0, cols=[dict(mt=172.6, als=ALS0), dict(mt=173.0, als=ALS0)],
                      fresh=fresh))
        cmp = []
        for x in o["cols"]:
            Lb, Qb, Gb = B[round(x["mt"], 2)]
            k = int(np.argmin(np.abs(Lb)))
            cmp.append(dict(mt=x["mt"], dQ=x["Q"] - float(Qb[k]), rel_gamma=x["gamma"] / float(Gb[k]) - 1, nc=x["nc"],
                            edge=x["edge"], prof_g2=x["prof_g2"]))
        out["fresh%d" % int(fresh)] = dict(cmp=cmp, time_s=o["time_s"], time_profiles_s=o["time_profiles_s"])
    W.wjson(os.path.join(RES, "C152_core_test.json"), out)
    print(json.dumps(W.clean(out), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "test":
        test()
    elif md == "lgrid":
        run(specs_lgrid())
    elif md == "direct":
        sp = []
        for a in sys.argv[2:]:
            sp += specs_direct(float(a))
        run(sp)
    elif md == "all":
        sp = specs_lgrid()
        for a in sys.argv[2:]:
            sp += specs_direct(float(a))
        run(sp)
    else:
        raise SystemExit(__doc__)
