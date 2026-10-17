# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/higgs_robustness/C152_post.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent higgs-robustness (C152/D124, v12 round 3) on 2026-10-15; uses C152_core (same dir) and,
#   read only, the C149 run02 shards (holonomy_cross) and C157_xi_quick.json (realtime_2d).
r"""[v12 package, 2026-10-15] Background-input sensitivities from the C152 shards: dlnI/dlam_b, transit-duration slopes (e-fold finite differences -0.170/-0.270/-0.394 GeV), (vi) inflation scale r in [1e-3, 1e-2] (+-0.035/+-0.067/+-0.112 GeV), content inputs.  Paper: PRD App. H3/H4; PRL Table 1.

C152 post: input -> Delta m_c table (full hazard + root finding for every input), m_c(w0) table, kappa_E backreaction
bound, slope update for D115/(vi).  Background map (exact for the bounce 3 M_P^2 h^2 = C_eff/L_b^4):
    mu_b = 1/L_b ∝ C_eff^(-1/4) h^(1/2),   hL_b ∝ C_eff^(1/4) h^(1/2),   h ∝ (A_s r)^(1/2) / (H_*/h).
A variant is (dmu, dh) = (dln mu_b, dln hL_b); I(m_t) = int_0^5 gamma(m_t, l_k + dmu) du/(hL_b e^dh sqrt(cosh 2u)).
gamma(m_t, l): C152_core node roots; 'spline' = cubic spline of ln gamma in l through the 11 baseline nodes + 3 new
scale points; 'direct' = all 11 nodes recomputed at the shifted scale (validation).
Usage: python -B C152_post.py
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
import glob
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.integrate import solve_ivp, trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C152_core as S  # noqa: E402

W = S.W
RES = S.RES
TGT = {"P1=0.9": 0.9, "P1=0.5": 0.5, "P1=0.1": 0.1}
HLB = float(W.HLB)
XI0, NC0 = 0.2999, 25.515
CEFF0 = (1 - XI0) * NC0
HOM0 = HLB / (math.sqrt(CEFF0 * 1.6751 / 3.0) / HLB)          # h/M_P implied by (722.27, 0.00437243), C = N_C*1.6751


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    B = S.base_table()
    pts = {mt: {round(float(l), 6): float(g) for l, g in zip(B[mt][0], B[mt][2])} for mt in B}
    lg = {}
    for f in glob.glob(os.path.join(S.SH, "L_l*.json")):
        d = W.jload(f)
        for x in d["cols"]:
            pts[round(x["mt"], 2)][round(x["l"], 6)] = x["gamma"]
            lg.setdefault("flags", []).extend([(x["mt"], x["l"])] if (x["nc"] or x["edge"] or not x["gamma"] > 0) else [])
    D = {}
    for f in glob.glob(os.path.join(S.SH, "D*_u*.json")):
        d = W.jload(f)
        u = float(os.path.basename(f).split("_u")[1][:-5])
        dl = float(d["l"]) + 0.5 * math.log(math.cosh(2 * u))      # exact shift (the tag is rounded to 4 decimals)
        for x in d["cols"]:
            D.setdefault(round(dl, 6), {}).setdefault(round(x["mt"], 2), {})[round(u, 4)] = x
    return B, pts, D, lg


def splines(pts, drop=None):
    out = {}
    for mt, g in pts.items():
        ls = sorted(k for k in g if drop is None or abs(k - drop) > 1e-9)
        out[mt] = CubicSpline(ls, np.log([g[k] for k in ls]))
    return out


def crit_t(mts, I, targets):
    o = np.argsort(mts)
    x = np.asarray(mts, float)[o]
    sp = CubicSpline(x, np.log(np.asarray(I, float)[o]))
    res = {}
    for k, P in targets.items():
        if not P < 1.0:
            res[k] = None
            continue
        t = -math.log(P)
        try:
            res[k] = float(brentq(lambda z: sp(z) - math.log(t), x[0], x[-1]))
        except ValueError:
            res[k] = None
    return res, sp


def hazard_spline(spl, dmu, dh, lnode=None, wnode=None):
    mts = sorted(spl)
    I = []
    for mt in mts:
        if lnode is None:
            g = np.exp(spl[mt](S.L11 + dmu))
            I.append(W.integ(S.UN11, g) / math.exp(dh))
        else:                                 # general background: node scales l(u) and weight factor w(u) on a fine grid
            I.append(float(trapezoid(np.exp(spl[mt](lnode + dmu)) * wnode, UF)) / (HLB * math.exp(dh)))
    return mts, I


def hazard_direct(D, dl, dh):
    mts, I = [], []
    for mt, g in sorted(D[round(dl, 6)].items()):
        if len(g) < 11:
            continue
        gam = np.array([g[round(float(u), 4)]["gamma"] for u in S.UN11])
        mts.append(mt)
        I.append(W.integ(S.UN11, gam) / math.exp(dh))
    return mts, I


def variant(spl, dmu, dh, targets=TGT):
    mts, I = hazard_spline(spl, dmu, dh)
    c, sp = crit_t(mts, I, targets)
    return c, dict(zip(["%.1f" % m for m in mts], [math.exp(-i) for i in I])), sp


# ------------------------------------------------------------------ kappa_E: shape backreaction on the background
UF = np.linspace(0.0, 5.0, 5001)


def shape_background(modes, beta_rms):
    """modes: list of kappa_A (one entry per mode); incoming constant mode of rms beta_rms (zero canonical momentum).
    Solves beta'' + 3(a'/a)beta' - (kappa/a^4) beta = 0 on the unperturbed bounce for u < 0, then the coupled system
    a''/a = 1 + (1+eps_C)/a^4 - 2 eps_K, eps_C = sum kappa beta^2/6, eps_K = sum beta'^2/6 (mean field) for u >= 0,
    with a(0)^4 = (1+eps_C)/(1+eps_K) (turning point at u = 0).  Returns a(u)/a0(u) on UF and diagnostics."""
    K = np.asarray(modes, float)

    def f0(u, y):
        b, bp = y[:len(K)], y[len(K):]
        th = math.tanh(2 * u)
        return np.concatenate([bp, -3 * th * bp + K / math.cosh(2 * u) ** 2 * b])
    y0 = np.concatenate([np.ones(len(K)), np.zeros(len(K))])
    s0 = solve_ivp(f0, (-9.0, 0.0), y0, rtol=1e-11, atol=1e-13)
    b0, bp0 = s0.y[:len(K), -1] * beta_rms, s0.y[len(K):, -1] * beta_rms
    eC0, eK0 = float(np.sum(K * b0 ** 2) / 6), float(np.sum(bp0 ** 2) / 6)
    ab = ((1 + eC0) / (1 + eK0)) ** 0.25

    def f1(u, y):
        a, ap = y[0], y[1]
        b, bp = y[2:2 + len(K)], y[2 + len(K):]
        eC, eK = np.sum(K * b * b) / 6, np.sum(bp * bp) / 6
        return np.concatenate([[ap, a * (1 + (1 + eC) / a ** 4 - 2 * eK)], bp, -3 * (ap / a) * bp + K / a ** 4 * b])
    s1 = solve_ivp(f1, (0.0, 5.0), np.concatenate([[ab, 0.0], b0, bp0]), t_eval=UF, rtol=1e-11, atol=1e-14)
    a0 = np.sqrt(np.cosh(2 * UF))
    gain = s1.y[2:2 + len(K), -1] / beta_rms
    epsC = np.sum(K[:, None] * s1.y[2:2 + len(K)] ** 2, axis=0) / 6
    epsK = np.sum(s1.y[2 + len(K):] ** 2, axis=0) / 6
    S_ = epsC / a0 ** 4 - epsK
    return s1.y[0] / a0, dict(gain_u5=gain.tolist(), max_abs_S=float(np.max(np.abs(S_))), eps_C0=eC0, eps_K0=eK0,
                              max_abs_dlna=float(np.max(np.abs(np.log(s1.y[0] / a0)))))


def main():
    out = dict(code_sha256={f: sha(os.path.join(HERE, f)) for f in ("C152_core.py", "C152_post.py")},
               base_shards_sha256_sample=sha(os.path.join(S.BASESH, "X_main_u0.0000.json")))
    B, pts, D, lg = load()
    spl = splines(pts)
    # ---- baseline reproduction
    mts = sorted(B)
    I0 = [W.integ(S.UN11, np.array([pts[mt][round(float(l), 6)] for l in S.L11])) for mt in mts]
    c0, sp0 = crit_t(mts, I0, TGT)
    out["baseline"] = dict(crit=c0, P1={"%.2f" % m: math.exp(-math.exp(sp0(m))) for m in (172.2, 172.52, 172.6, 173.0)},
                           mts=mts, I=I0, new_scale_points=sorted({round(l, 4) for g in pts.values() for l in g}))
    cs, _, _ = variant(spl, 0.0, 0.0)
    out["baseline"]["spline_path_crit"] = cs
    # ---- l-spline leave-one-out (interior baseline nodes) -> effect on m_c
    loo = {}
    for k in range(1, 10):
        lk = round(float(S.L11[k]), 6)
        sp_d = splines(pts, drop=lk)
        dev = max(abs(float(sp_d[mt](lk)) - math.log(pts[mt][lk])) for mt in mts)
        Ik = []
        for mt in mts:
            g = np.array([pts[mt][round(float(l), 6)] for l in S.L11])
            g[k] = math.exp(float(sp_d[mt](lk)))
            Ik.append(W.integ(S.UN11, g))
        ck, _ = crit_t(mts, Ik, TGT)
        loo["u=%.1f" % S.UN11[k]] = dict(max_abs_dlngamma=dev, dmc={t: ck[t] - c0[t] for t in TGT})
    out["spline_leave_one_out"] = loo
    # ---- direct validations
    val = {}
    DIRECT_DH = {-0.014531: +0.014531}      # C_eff (xi = 0.258) run: dh = -dmu; the other direct runs: dmu = dh
    for dl in sorted(D):
        dh = DIRECT_DH.get(round(dl, 6), dl)
        md, Id = hazard_direct(D, dl, dh)
        if len(md) < 5:
            continue
        cd, _ = crit_t(md, Id, TGT)
        cspl, _, _ = variant(spl, dl, dh)
        val["%+.6f" % dl] = dict(dmu=dl, dh=dh, direct=cd, spline=cspl, direct_minus_spline={t: (cd[t] - cspl[t]) if cd[t] and cspl[t]
                                                                             else None for t in TGT},
                                 dmc_direct={t: (cd[t] - c0[t]) if cd[t] else None for t in TGT},
                                 flags=[(mt, u) for mt, g in D[round(dl, 6)].items() for u, x in g.items()
                                        if x["nc"] or x["edge"] or not x["gamma"] > 0])
    out["direct_validation"] = val
    out["lgrid_flags"] = lg.get("flags", [])

    # ---- the variants
    def ceff_shift(ratio):
        return -0.25 * math.log(ratio), 0.25 * math.log(ratio)

    yt_v11, yt_now = 0.4585, float(np.mean([x["yt"] for x in W.jload(os.path.join(S.BASESH, "X_main_u0.0000.json"))["cols"]
                                            if abs(x["mt"] - 172.6) < 1e-6]))
    m2_v11 = -0.1740 - 0.0135 + 0.0805
    m2_now = -0.1740 - 0.0135 + 0.0805 * (yt_now / yt_v11) ** 2
    dxi_top = -0.003 * (m2_now / m2_v11 - 1.0)
    V = {}
    for xi in (0.28, 0.283, 0.258, 0.308, 0.2979, XI0 + dxi_top):
        V["xi=%.4f" % xi] = dict(group="xi (background only)", dmu_dh=ceff_shift((1 - xi) / (1 - XI0)))
    for dn, nm in ((0.5, "inflaton +1/2"), (0.701, "3 nu_R +0.70"), (1.201, "inflaton + 3 nu_R")):
        V["N_C %s (Q fixed)" % nm] = dict(group="N_C^eff", dmu_dh=ceff_shift(1 + dn / CEFF0))
        V["N_C %s (xi fixed)" % nm] = dict(group="N_C^eff", dmu_dh=ceff_shift(1 + dn / NC0))
    hinst, hweak = 6.116e-6, 6.7467e-6
    d_inst = 0.5 * math.log(hinst / HOM0)

    def infl(d):
        return (d, d)
    V["h/M_P = 6.116e-6 (v11 instant reheating, consistency of (722, hL_b))"] = dict(group="A_s/plateau", dmu_dh=infl(d_inst))
    V["weak visible decay h/M_P = 6.7467e-6 (N*=50.33, r=0.0042)"] = dict(group="A_s/plateau", dmu_dh=infl(0.5 * math.log(hweak / HOM0)))
    for sgn in (+1, -1):
        V["A_s %+d sigma (dln A_s = %+.3f)" % (sgn, 0.014 * sgn)] = dict(group="A_s/plateau", dmu_dh=infl(0.25 * 0.014 * sgn))
    V["plateau shape H_*/h: 0.986 -> 1 (alpha->0 attractor)"] = dict(group="A_s/plateau", dmu_dh=infl(0.5 * math.log(0.98636)))
    for d in (-0.8803, -0.3147, d_inst + 0.25 * math.log(1e-3 / 0.00346), d_inst + 0.25 * math.log(0.004 / 0.00346),
              d_inst + 0.25 * math.log(0.0042 / 0.00346), +0.2705):
        r = 0.00346 * math.exp(4 * (d - d_inst))
        V["plateau r = %.3g (A_s fixed, H_*/h = 0.986)" % r] = dict(group="A_s/plateau", dmu_dh=infl(d), r=r)
    for de in (+1.0, -1.0, +0.5, -0.5):
        V["hL_b x e^%+.1f at fixed mu_b (eta_inf / e^%+.1f)" % (de, de)] = dict(group="hL_b (turn duration)", dmu_dh=(0.0, de))
    for de in (+0.25, -0.25):
        V["mu_b x e^%+.2f at fixed hL_b" % de] = dict(group="mu_b only", dmu_dh=(de, 0.0))
        V["r x e^%+.0f (dmu = dh = %+.2f)" % (4 * de, de)] = dict(group="per e-fold of r", dmu_dh=(de, de))
        V["C_eff x e^%+.0f" % (4 * de)] = dict(group="per e-fold of C_eff", dmu_dh=(-de, de))
    for k, v in V.items():
        dmu, dh = v["dmu_dh"]
        c, P1, spv = variant(spl, dmu, dh)
        v.update(crit=c, dmc={t: (c[t] - c0[t]) if c[t] else None for t in TGT},
                 P1_172p6=float(math.exp(-math.exp(spv(172.6)))))
        # direct value when available
        for dl, vv in val.items():
            if abs(vv["dmu"] - dmu) < 2e-6 and abs(vv["dh"] - dh) < 2e-6:
                v["dmc_direct"] = vv["dmc_direct"]
    out["variants"] = V
    out["inputs"] = dict(yt_v11=yt_v11, yt_NNLO_mu_b_172p6=yt_now, dxi_top_term=dxi_top, h_over_MP_pipeline=HOM0,
                         C_eff0=CEFF0, d_instant=d_inst)
    # ---- per-e-fold slopes (finite differences of the full chain) and the hazard-equivalent dlnI/dlambda
    lam_b = {mt: float(np.mean([x["lam"] for x in W.jload(os.path.join(S.BASESH, "X_main_u0.0000.json"))["cols"]
                                if abs(x["mt"] - mt) < 1e-6])) for mt in mts}
    lsp = CubicSpline(mts, [lam_b[m] for m in mts])
    sl = {}
    for t in TGT:
        m = c0[t]
        sl[t] = dict(dlnI_dmt=float(sp0(m, 1)), dlam_b_dmt=float(lsp(m, 1)), dlnI_dlam_b=float(sp0(m, 1) / lsp(m, 1)),
                     lam_b=float(lsp(m)))
    out["slopes"] = sl
    # ---- kappa_E backreaction bound (mean field, <beta^2> = delta^2 per mode, E doublet + T2 triplet)
    kb = {}
    for tag, kE, kT in (("kappa_E=19.32 (LO)", 19.324, -9.074), ("kappa_E=18.0 (NLO Hessian)", 18.0, -9.074 * 18.0 / 19.324),
                        ("free fields 13.53", 13.53, -6.35)):
        for brms in (2.04e-4, 3 * 2.04e-4):
            ratio, diag = shape_background([kE, kE, kT, kT, kT], brms)
            lnode = -np.log(np.sqrt(np.cosh(2 * UF)) * ratio)
            wnode = 1.0 / (np.sqrt(np.cosh(2 * UF)) * ratio)
            mts_, Ib = hazard_spline(spl, 0.0, 0.0, lnode, wnode)
            lref = -np.log(np.sqrt(np.cosh(2 * UF)))
            _, Ir = hazard_spline(spl, 0.0, 0.0, lref, 1.0 / np.sqrt(np.cosh(2 * UF)))
            cb, _ = crit_t(mts_, Ib, TGT)
            cr, _ = crit_t(mts_, Ir, TGT)
            kb["%s, beta_rms=%.2e" % (tag, brms)] = dict(diag, dmc={t: cb[t] - cr[t] for t in TGT})
    out["kappaE_backreaction"] = kb
    # ---- m_c(w0)
    mw = {}
    for w0 in (1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.55):
        tg = {k: v / w0 for k, v in TGT.items()}
        c, _ = crit_t(mts, I0, tg)
        mw["%.2f" % w0] = dict(crit=c, dmc={t: (c[t] - c0[t]) if c[t] else None for t in TGT},
                               note={t: ("no critical value: w0 P1 <= w0 < %.1f for every m_t" % TGT[t]) for t in TGT
                                     if TGT[t] / w0 >= 1.0},
                               P_172p6=w0 * math.exp(-math.exp(sp0(172.6))))
    out["mc_w0"] = mw
    W.wjson(os.path.join(RES, "C152_post.json"), out)
    print(json.dumps(W.clean({k: out[k] for k in ("baseline", "direct_validation", "slopes")}), indent=1))
    for k, v in V.items():
        print("%-75s dmu=%+.5f dh=%+.5f  dmc=%s  direct=%s" % (k, v["dmu_dh"][0], v["dmu_dh"][1],
              " ".join("%+.4f" % v["dmc"][t] if v["dmc"][t] is not None else "None" for t in TGT),
              v.get("dmc_direct")))
    for k, v in kb.items():
        print(k, "max|S|=%.2e max|dlna|=%.2e" % (v["max_abs_S"], v["max_abs_dlna"]),
              " ".join("%+.2e" % v["dmc"][t] for t in TGT))
    for k, v in mw.items():
        print("w0=%s" % k, v["crit"], v["note"])
    for k, v in loo.items():
        print("LOO", k, "%.2e" % v["max_abs_dlngamma"], v["dmc"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
