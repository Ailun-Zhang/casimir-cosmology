# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct4.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] D121 run04: run02 node-direct hazard + gauge-Higgs vev mixing + exact top chiral bubble added to the one-loop Hessian before the variational re-minimisation (variants A/B/C).  Imported by run05.  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 run04 (holonomy-cross, D121 appendix run04): run02 node-direct 2D hazard + gauge-Higgs vev mixing + exact top
chiral (axial) bubble, all with the node couplings, added to the one-loop Hessian BEFORE the variational re-minimisation.

vev mixing (Coulomb gauge: transverse fluctuations only, c_pol = 2/3; per channel dH = -(kappa^2/3) S(m1^2, m2^2),
S = MS-bar (mu = 1/L) FV bubble sum' 1/(w1 w2 (w1+w2)) = -4 d2E_std/d(m^2)^2|_M + convergent difference):
   W1 background:  W1_fl x h   (kappa = g2^2 r/2),   B_fl x pi1 (kappa = g2 gY r/2; B = (gY Z - g2 A)/gZ)
   Z background:   Z_fl x h    (kappa = gZ^2 r/2),   W1,2_fl x pi1,2 (kappa = g2 gY^2 r/(2 gZ))
   (anticommutator v^dag {G_bg, T_B} phi; photon background: none, exact via the twists)
   -> dH11 = dH_WW, (dH33, dHww, dH3w) = (g2^2, gY^2, g2 gY) dH_ZZ / gZ^2.
top chiral: dH33 = 3 g2^2 G_T3(m_t^2), dHww = 3 gY^2 G_Y, dH3w = 3 g2 gY H3Y, dH11 = 3 g2^2 G_T1 (C149_top.py, per colour,
   axial log B = q_Phi^2 m_t^2 verified = top part of 2 gamma_phi^C x tree).
Variant (env C149_V): A = vev + top (central), B = vev only, C = top only.
Usage: python -B C149_direct4.py test | run main|als | post | final
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C149_direct as D  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

W, K, X, V = D.W, D.K, D.X, D.V
C = D.T.B.core()
PI = math.pi
VAR = os.environ.get("C149_V", "A")
RES4 = os.path.join(D.ROOT, _zp.w("r3/holonomy_cross/run04"))
D.SH = os.path.join(RES4, "shards_" + VAR)
D.RES2 = os.path.join(RES4, "post_" + VAR)
TOPJ = json.load(open(os.path.join(RES4, "C149_top_chiral.json"), encoding="utf-8"))
k2 = 4 * PI ** 2 * X.QP
NQ = X.NP_
_ORIG_INC = D.cross_increments


def S_ren(m1, m2):
    M = 0.5 * (m1 + m2)
    h = 1e-3 * max(1.0, abs(M))
    d2 = (C.estd(M + h)[0] - 2 * C.estd(M)[0] + C.estd(M - h)[0]) / h ** 2
    w1, w2, wM = np.sqrt(k2 + m1), np.sqrt(k2 + m2), np.sqrt(k2 + M)
    return float(-4 * d2 + np.sum(NQ * (1 / (w1 * w2 * (w1 + w2)) - 1 / (2 * wM ** 3))))


def top_G(name, m2):
    g = np.array(TOPJ[name]["G"] if name != "H3Y" else TOPJ["H3Y"], float)
    mg = np.array(TOPJ["m2"], float)
    if m2 <= mg[-1]:
        return float(CubicSpline(mg, g)(m2))
    return float(g[-1] * m2 / mg[-1])          # G ~ q_Phi^2 m^2 S_A(m^2), S_A slowly varying


def vev_top_dH(r, c):
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    gZ2 = g2 * g2 + gY * gY
    aW, aZ, b = g2 * g2 / 4 * r * r, gZ2 / 4 * r * r, lam * r * r
    mh, mpi, mt2 = 3 * lam * r * r, aW + b, yt * yt / 2 * r * r
    out = dict(H11=0.0, H33=0.0, Hww=0.0, H3w=0.0)
    if VAR in ("A", "B") and r > 0:
        dWW = -((g2 * g2 * r / 2) ** 2 * S_ren(mh, aW)
                + (g2 * gY * r / 2) ** 2 * (g2 * g2 / gZ2 * S_ren(mpi, 0.0) + gY * gY / gZ2 * S_ren(mpi, aZ))) / 3
        dZZ = -((gZ2 * r / 2) ** 2 * S_ren(mh, aZ) + 2 * (g2 * gY * gY * r / 2) ** 2 / gZ2 * S_ren(mpi, aW)) / 3
        out["H11"] += dWW
        out["H33"] += g2 * g2 / gZ2 * dZZ
        out["Hww"] += gY * gY / gZ2 * dZZ
        out["H3w"] += g2 * gY / gZ2 * dZZ
    if VAR in ("A", "C"):
        out["H33"] += 3 * g2 * g2 * top_G("T3", mt2)
        out["Hww"] += 3 * gY * gY * top_G("Y", mt2)
        out["H3w"] += 3 * g2 * gY * top_G("H3Y", mt2)
        out["H11"] += 3 * g2 * g2 * top_G("T1", mt2)
    return out


def cross_increments4(c, g3):
    """C149_direct.cross_increments with the vev-mixing / top-chiral Hessian added (copied logic, extended)."""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = D.PV11["rf"]
    rg = D.T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    ext = [vev_top_dH(float(x), c) for x in rg]
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        e = np.array([x[kk] for x in ext])
        Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows] + e)(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows] + e)(rr)
    P0, _ = V.newton(lambda p: D.etot_g(p, rf, g2, gY), D.PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    Pst = P0.copy()
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    P1, gres = V.newton(lambda p: D.etot_g(p, rf, g2, gY) + V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
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
    sel = [i for i, x in enumerate(rg) if abs(x - 4.0) < 1e-9 or abs(x - 1.2) < 1e-9]
    info = dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])), E0_su3=E0, s0_su3=s0,
                ext_at={f"{rg[i]:.2f}": ext[i] for i in sel})
    return out, info


D.cross_increments = cross_increments4


def final():
    """combine the variants with run02 -> run04/C149_run04_summary.json."""
    S2 = W.jload(os.path.join(D.ROOT, _zp.w("r3/holonomy_cross/run02/C149_run02_summary.json")))
    res = {"run02": S2["crit"], "P1_run02": S2["P1"]}
    for v in ("A", "B", "C"):
        p = os.path.join(RES4, "post_" + v, "C149_run02_summary.json")
        if os.path.exists(p):
            d = W.jload(p)
            res[v] = dict(crit={a: d["crit"][a]["with_cross"] for a in d["crit"]}, P1=d.get("P1"), P1_grid=d.get("P1_grid"),
                          lam_c=d.get("lam_c_hazard_equiv"),
                          shift_vs_run02={a: {k: (d["crit"][a]["with_cross"][k] - S2["crit"][a]["with_cross"][k])
                                              if (d["crit"][a]["with_cross"][k] is not None
                                                  and S2["crit"][a]["with_cross"][k] is not None) else None
                                              for k in d["crit"][a]["with_cross"]} for a in d["crit"] if a in S2["crit"]},
                          shift_vs_T14b={a: d["crit"][a]["shift"] for a in d["crit"]}, flags=d.get("flags"))
    # one-loop log bookkeeping (x-units, W direction, per r^2): vev channels vs required 2 gamma_phi^C(gauge) x tree
    g2, gY = 0.52772, 0.44240                       # node u = 0, (172.6, 0.1180) NNLO (T14b D122 Sec. 1.1)
    gZ2 = g2 * g2 + gY * gY
    tree = g2 * g2 / 4
    vev_log = -(g2 ** 4 + g2 * g2 * gY * gY) / (4 * 3 * 4 * PI ** 2)
    req_gauge = 2 * (-(1.5 * g2 * g2 + 0.5 * gY * gY) / (16 * PI ** 2)) * tree
    res["log_check_W_direction_per_r2"] = dict(vev_mixing=vev_log, required_gauge_2gammaphi_tree=req_gauge,
                                               fraction=vev_log / req_gauge,
                                               top_axial="B = q_Phi^2 m_t^2 per colour (C149_top.json) = top part of "
                                                         "2 gamma_phi^C x tree exactly")
    W.wjson(os.path.join(RES4, "C149_run04_summary.json"), res)
    print(json.dumps(W.clean(res), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else ""
    if md == "test":
        o = D.jobX(dict(tag="TEST_u0.0000", u=0.0, cols=[dict(mt=172.6, als=0.1180), dict(mt=173.0, als=0.1180)]))
        print(json.dumps(W.clean([{k: x[k] for k in ("mt", "gamma_T14b", "gamma", "ratio_vs_T14b", "Q", "center", "edge",
                                                     "nc", "info")} for x in o["cols"]]), indent=1))
    elif md == "run":
        D.run(sys.argv[2])
    elif md == "post":
        os.makedirs(D.RES2, exist_ok=True)
        D.post()
    elif md == "final":
        final()
