# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct5.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] D121 run05 = PRL centre: node-direct 2D hazard with the complete one-loop r^2 c^2 set (CCA cross terms + covariant vev mixing + Coulomb mismatch + exact top chiral).  'run main|als' -> 22 node shards; 'post' -> critical m_t, P1, lam_c; 'final' -> C149_run05_summary.json.  Paper: PRL Table 1, Eq. for lam_c; PRD App. H2', H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 run05 (holonomy-cross, D121 appendix run05): node-direct 2D hazard with the complete one-loop r^2 c^2 set:
CCA cross terms + transverse vev mixing with the background-covariant Coulomb-gauge vertex 2 (c Phi0)^dag V phi
(4 W-channels, 3 Z-channels, c_pol = 2/3) + exact Coulomb-mode (Gauss-law) mismatch (charged w_C, neutral h-pi3-Z0;
Z direction exact, W direction by the custodial ratio dH_WW^coul = dH_WW^vev x dH_ZZ^coul/dH_ZZ^vev) + exact top
chiral bubble (run04).  RG closure of the logs vs 2 gamma_phi^C x tree: 99% (C149_coulomb_check.json).
Large-k coefficients used in the MS-bar renormalisation: charged C = 2 a_W (gZ/2)^2/3 (fit/theory 1.003 at r = 5),
neutral C = 0.353 a_Z (gZ/2)^2 (fitted at r = 5; C ~ r^2 removes the small-r fit noise).
Pieces are evaluated on a coarse r grid and splined.  Usage: python -B C149_direct5.py test | run main|als | post | final
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
os.environ.setdefault("C149_V", "A")
import C149_direct4 as D4  # noqa: E402  (imports C149_direct as D, patches D.cross_increments)
import C149_coulomb as CO  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

D, W, K, X, V = D4.D, D4.W, D4.K, D4.X, D4.V
PI = math.pi
RES5 = os.path.join(D.ROOT, _zp.w("r3/holonomy_cross/run05"))
D.SH = os.path.join(RES5, "shards")
D.RES2 = os.path.join(RES5, "post")
RC = np.concatenate([np.arange(0.05, 2.0, 0.05), np.arange(2.0, 16.7, 0.2)])
CN_COEF = 0.353


def renorm_C(dk, M2, Cc):
    wM = np.sqrt(CO.KN ** 2 + M2)
    return float(np.sum(dk - Cc / (2 * wM ** 3)) + Cc * CO.S0(M2))


def coul_Z(r, g2, gY, lam, hc=1e-3):
    gZ2 = g2 * g2 + gY * gY
    gZ = math.sqrt(gZ2)
    aW, aZ, b, mh = g2 * g2 / 4 * r * r, gZ2 / 4 * r * r, lam * r * r, 3 * lam * r * r
    KV, KN = CO.KV, CO.KN
    qp, qa = (gY * gY - g2 * g2) / (2 * gZ), -g2 * g2 / gZ
    sh = lambda q, c: ((KV - np.outer(np.ones(len(KV)), [q * c, 0, 0])) ** 2).sum(1)  # noqa: E731

    def Ec(c):
        return 0.5 * (np.sqrt((sh(qp, c) + b) * (sh(qa, c) + aW) / sh(qa, c))
                      + np.sqrt((sh(-qp, c) + b) * (sh(-qa, c) + aW) / sh(-qa, c)))
    w = np.sqrt(KN ** 2 + aW + b)
    dkc = (Ec(hc) - 2 * Ec(0.0) + Ec(-hc)) / hc ** 2 - qp * qp * (1 / w - KV[:, 0] ** 2 / w ** 3)
    Fc = renorm_C(dkc, aW, 2 * aW * (gZ / 2) ** 2 / 3)
    qn, k2 = gZ / 2, KN ** 2
    Zk = k2 / (k2 + aZ)

    def En(c):
        t = qn * c
        A, Cc = k2 + mh + t * t, (k2 + b + t * t) / Zk
        disc = np.sqrt(0.25 * (A - Cc) ** 2 + (2 * t * KV[:, 0]) ** 2 / Zk)
        return 0.5 * (np.sqrt(0.5 * (A + Cc) + disc) + np.sqrt(0.5 * (A + Cc) - disc))
    wh, wp = np.sqrt(k2 + mh), np.sqrt(k2 + aZ + b)
    dkn = (En(hc) - 2 * En(0.0) + En(-hc)) / hc ** 2 - qn * qn * 0.5 * ((1 / wh - KV[:, 0] ** 2 / wh ** 3)
                                                                        + (1 / wp - KV[:, 0] ** 2 / wp ** 3))
    Fn = renorm_C(dkn, aZ, CN_COEF * aZ * qn * qn)
    return Fc + Fn


def ext_grid(c):
    """coarse-grid vev + Coulomb + top pieces (x-units) -> dict of arrays on RC."""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    gZ2 = g2 * g2 + gY * gY
    out = {k: np.zeros(len(RC)) for k in ("H11", "H33", "Hww", "H3w")}
    for i, r in enumerate(RC):
        dWW, dZZ, _, _ = CO.vev_new(r, g2, gY, lam)
        cz = coul_Z(r, g2, gY, lam)
        cw = dWW * (cz / dZZ) if abs(dZZ) > 1e-14 else 0.0
        mt2 = yt * yt / 2 * r * r
        HZ = dZZ + cz
        out["H11"][i] = dWW + cw + 3 * g2 * g2 * D4.top_G("T1", mt2)
        out["H33"][i] = g2 * g2 / gZ2 * HZ + 3 * g2 * g2 * D4.top_G("T3", mt2)
        out["Hww"][i] = gY * gY / gZ2 * HZ + 3 * gY * gY * D4.top_G("Y", mt2)
        out["H3w"][i] = g2 * gY / gZ2 * HZ + 3 * g2 * gY * D4.top_G("H3Y", mt2)
    return out


def cross_increments5(c, g3):
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = D.PV11["rf"]
    rg = D.T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    eg = ext_grid(c)
    ext = {k: CubicSpline(RC, eg[k])(np.clip(rg, RC[0], RC[-1])) * np.where(rg < RC[0], (rg / RC[0]) ** 2, 1.0)
           for k in eg}
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, np.array([sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])
                             + ext[kk])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, np.array([sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows])
                                     + ext[kk])(rr)
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
    info = dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])),
                ext_r4={k: float(np.interp(4.0, RC, eg[k])) for k in eg})
    return out, info


D.cross_increments = cross_increments5


def final():
    S4 = W.jload(os.path.join(D4.RES4, "C149_run04_summary.json"))
    d = W.jload(os.path.join(D.RES2, "C149_run02_summary.json"))
    c4 = S4["A"]["crit"]
    res = dict(crit={a: d["crit"][a]["with_cross"] for a in d["crit"]}, P1=d.get("P1"), P1_grid=d.get("P1_grid"),
               lam_c=d.get("lam_c_hazard_equiv"), flags=d.get("flags"),
               shift_vs_run04={a: {k: (d["crit"][a]["with_cross"][k] - c4[a][k]) if (d["crit"][a]["with_cross"][k] is not None
                                                                                    and c4.get(a, {}).get(k) is not None) else None
                                   for k in d["crit"][a]["with_cross"]} for a in d["crit"]},
               shift_vs_T14b={a: d["crit"][a]["shift"] for a in d["crit"]},
               rg_closure=W.jload(os.path.join(RES5, "C149_coulomb_check.json")))
    W.wjson(os.path.join(RES5, "C149_run05_summary.json"), res)
    print(json.dumps(W.clean({k: v for k, v in res.items() if k != "rg_closure"}), indent=1))


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
