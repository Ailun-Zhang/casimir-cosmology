# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_eta.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128, v12 round 3) on 2026-10-14. integ/crit re-implemented verbatim
# from 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C150_twod.py ([C144] hazard post); inputs read only.
r"""[v12 package, 2026-10-15] |lam_c| scaling with the transit duration eta (D128 Sec. 1): b_eff, dlam_c/dln eta, dm_c/dln eta from the run02 node widths (WKB structure ln gamma = a - b |lam|^-1/2).  Paper: PRD Sec. IV D.

C156 T1a: duration (eta_inf) scaling of the hazard-equivalent lambda_c and the WKB structure of gamma(lambda).
Inputs: holonomy_cross/run02/shards/X_main_u*.json (11 nodes x 10 m_t, alpha_s = 0.1180; central round-3 model).
  * I(m_t) = int_0^5 gamma(u) du/(hL_b sqrt(cosh 2u)); a pure duration change eta_inf -> f eta_inf (h changed at fixed L_b,
    so mu_b and every node coupling are unchanged) is I -> f I exactly.  m_c(f) by spline root, lambda_c(f) = lambda(mu_b, m_c).
  * WKB: per node fit ln gamma = a - b |lambda|^(-1/2) (and a - b|lambda|^-p); closed form for the duration scaling
    |lambda_c(f)|^(-1/2) = |lambda_c(1)|^(-1/2) + ln f / b_eff, i.e. dlambda_c/dln eta = 2 |lambda_c|^(3/2) / b_eff.
Usage: python -B U156_eta.py
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
import glob
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq, curve_fit

ROOT = _zp.WORK
SH = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/shards"))
OUT = os.path.join(ROOT, _zp.w("r3/universality/run01"))
HLB = 0.00437243
ETA = 1.31102877714606 / HLB
TGT = {"P1=0.9": -math.log(0.9), "P1=0.5": math.log(2.0), "P1=0.1": math.log(10.0)}
FAC = [0.25, 0.5, 1 / math.sqrt(2), 1.0, math.sqrt(2), 2.0, 4.0]


def integ(u, g, umax=5.0, hlb=HLB):
    uf = np.linspace(0.0, umax, 20001)
    wf = 1.0 / (hlb * np.sqrt(np.cosh(2 * uf)))
    spl = CubicSpline(u, np.log(np.maximum(g, 1e-300)))
    return float(simpson(np.exp(spl(uf)) * wf, x=uf))


def crit(x, I, tg):
    sp = CubicSpline(x, np.log(I))
    try:
        return float(brentq(lambda z: sp(z) - math.log(tg), x[0], x[-1]))
    except ValueError:                       # target outside the m_t grid: linear extrapolation of ln I from the end
        e = 0 if math.log(tg) < sp(x[0]) else -1
        s = float(sp(x[e], 1))
        return float(x[e] + (math.log(tg) - sp(x[e])) / s)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    R = {}
    for f in sorted(glob.glob(os.path.join(SH, "X_main_u*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for c in d["cols"]:
            R[(round(c["mt"], 2), round(c["u"], 4))] = c
    us = sorted(set(k[1] for k in R))
    mts = sorted(set(k[0] for k in R))
    G = np.array([[R[(m, u)]["gamma"] for u in us] for m in mts])        # (mt, u)
    LAM = np.array([[R[(m, u)]["lam"] for u in us] for m in mts])
    I = np.array([integ(np.array(us), G[i]) for i in range(len(mts))])
    lam_b = CubicSpline(mts, LAM[:, 0])
    out = dict(inputs=SH, nodes=us, mt=mts, I=I.tolist(), eta_inf=ETA, hLb=HLB)
    # ---- duration scaling (exact, I -> f I)
    tab = {}
    for f in FAC:
        row = {}
        for k, t in TGT.items():
            m = crit(np.array(mts), f * I, t)
            row[k] = dict(mc=m, lam_c=float(lam_b(m)))
        tab["%.4f" % f] = row
    out["eta_scaling"] = tab
    sl = {}
    for k in TGT:
        x = np.log(FAC)
        y = np.array([tab["%.4f" % f][k]["lam_c"] for f in FAC])
        ym = np.array([tab["%.4f" % f][k]["mc"] for f in FAC])
        i0 = FAC.index(1.0)
        sl[k] = dict(dlamc_dlneta_central=float((y[i0 + 1] - y[i0 - 1]) / (x[i0 + 1] - x[i0 - 1])),
                     dmc_dlneta_central=float((ym[i0 + 1] - ym[i0 - 1]) / (x[i0 + 1] - x[i0 - 1])),
                     dmc_per_factor2=[float(ym[FAC.index(0.5)] - ym[i0]), float(ym[FAC.index(2.0)] - ym[i0])])
    out["eta_slopes"] = sl
    # ---- hazard slope dlnI/dm_t at m_c and dlambda/dm_t
    spI = CubicSpline(mts, np.log(I))
    out["dlnI_dmt_at_mc"] = {k: float(spI(tab["1.0000"][k]["mc"], 1)) for k in TGT}
    out["dlam_dmt_at_mc"] = {k: float(lam_b(tab["1.0000"][k]["mc"], 1)) for k in TGT}
    # ---- WKB fits per node
    fits = {}
    for j, u in enumerate(us):
        lam = LAM[:, j]
        lg = np.log(G[:, j])
        x = np.abs(lam) ** -0.5
        A = np.vstack([np.ones_like(x), -x]).T
        (a, b), res, _, _ = np.linalg.lstsq(A, lg, rcond=None)
        rms2 = float(np.sqrt(np.mean((A @ np.array([a, b]) - lg) ** 2)))

        def f3(l, a_, b_, p_):
            return a_ - b_ * np.abs(l) ** (-p_)
        try:
            p3, _ = curve_fit(f3, lam, lg, p0=[a, b, 0.5], maxfev=20000)
            rms3 = float(np.sqrt(np.mean((f3(lam, *p3) - lg) ** 2)))
        except Exception as e:  # noqa: BLE001
            p3, rms3 = [None] * 3, repr(e)
        # local b from the derivative: dln gamma/d|lambda| = (b/2)|lambda|^(-3/2)
        o = np.argsort(np.abs(lam))
        sp = CubicSpline(np.abs(lam)[o], lg[o])
        bloc = {"%.4f" % l: float(2 * sp(l, 1) * l ** 1.5) for l in (0.009, 0.0121, 0.0151)}
        fits["%.2f" % u] = dict(a=float(a), b=float(b), rms_2p=rms2, p3=[float(z) if z is not None else None for z in p3],
                                rms_3p=rms3, b_local=bloc, lam_range=[float(lam.min()), float(lam.max())])
    out["wkb_fits"] = fits
    # ---- effective b for the hazard: dlnI/d|lambda_b| = (b_eff/2)|lambda_c|^(-3/2)
    beff = {}
    for k in TGT:
        m = tab["1.0000"][k]["mc"]
        dlnI_dlam = float(spI(m, 1) / lam_b(m, 1))            # d ln I / d lambda_b (negative: lambda more negative -> I up)
        lc = abs(float(lam_b(m)))
        beff[k] = dict(dlnI_dabslam=-dlnI_dlam, b_eff=float(-2 * dlnI_dlam * lc ** 1.5), lam_c=-lc)
    out["b_eff_hazard"] = beff
    # ---- closed-form duration law vs exact
    cmp_ = {}
    for k in TGT:
        lc1 = abs(tab["1.0000"][k]["lam_c"])
        be = beff[k]["b_eff"]
        cmp_[k] = {"%.4f" % f: dict(exact=tab["%.4f" % f][k]["lam_c"],
                                    wkb=float(-(lc1 ** -0.5 + math.log(f) / be) ** -2)) for f in FAC}
        cmp_[k]["dlamc_dlneta_wkb"] = float(2 * lc1 ** 1.5 / be)
    out["wkb_vs_exact_duration"] = cmp_
    # ---- static criterion at the u = 0 node: gamma(lambda_c) eta_inf = target  (no running)
    st = {}
    sp0 = CubicSpline(mts, np.log(G[:, 0]))
    for k, t in TGT.items():
        m = float(brentq(lambda z: sp0(z) - math.log(t / ETA), mts[0], mts[-1]))
        st[k] = dict(mc_static=m, lam_c_static=float(lam_b(m)), eta_eff_over_eta=float(I[np.argmin(np.abs(np.array(mts) - m))] /
                                                                                       (G[np.argmin(np.abs(np.array(mts) - m)), 0] * ETA)))
    out["static_u0"] = st
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "C156_eta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("eta_slopes", "dlnI_dmt_at_mc", "dlam_dmt_at_mc", "b_eff_hazard", "static_u0")},
                     indent=1))
    print(json.dumps({u: {k: fits[u][k] for k in ("a", "b", "rms_2p", "p3", "rms_3p", "b_local")} for u in ("0.00", "2.50", "5.00")},
                     indent=1))
    print(json.dumps(cmp_, indent=1))


if __name__ == "__main__":
    main()
