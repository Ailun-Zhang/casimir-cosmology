# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/R14_rge.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/R14_rge.py on 2026-10-14 by agent holonomy-cross (C149/D121).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_matching/R14_rge.py on 2026-10-13 by agent higgs-beyond-bo (C140/D112).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12/higgs_survival/xi_scan/R14_rge.py on 2026-10-13 by agent higgs-matching (C138/D110).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Two-loop SM RGE with a three-loop calibration constant (rounds 1-2 SM mapping; superseded for the PRL centre by sm_provider.py, kept for the convergence table and regressions).  mu_b = Mbar_P/722.0 = 3.3726e15 GeV.  Units: GeV.

R14 independent check of D108 -- check 1: SM mapping lambda(1/L_b).

Own implementation (no producer code opened).  Sources verified against the local copy of
arXiv:1307.3536v4 (02_literature/20261007_074655UTC/model_completion/C52_Buttazzo_etal_2013.pdf,
embedded text):
  * MS-bar couplings at mu = M_t: Eq.(55) lambda [PDF p.13], Eq.(57) y_t, Eq.(58) g2, Eq.(59) gY [PDF p.14],
    Eq.(60) g3 [PDF p.15].
  * Planck-scale fit Eqs.(61a-e) [PDF p.16] (their 3-loop RGE + NNLO thresholds) -> used only to calibrate
    the 2-loop truncation, exactly as D108 says it did.
  * RGEs: App. B (written as d/dln mu^2 with GUT g1).  The 1- and 2-loop terms below are converted to
    d/dln mu and gY (g1^2 = 5 gY^2/3); every 2-loop coefficient of beta_lambda and beta_{gY,g2} was checked
    term by term against the App. B text (lambda^2 block -156,-72,54,54/5; lambda*yt^2 block; lambda*g^4 block
    -73/16,1887/400,117/40; yt^4 block 15,-16,-4/5; yt^2 g^4 block -9/8,-171/200,63/20; g^6 block
    305/32,-3411/4000,-289/160,-1677/800).  y_b, y_tau neglected (effect < 1e-4 on lambda).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
PI = math.pi
K1 = 1.0 / (16.0 * PI ** 2)
MPBAR = 2.435e18
MU_B = MPBAR / 722.0
MW_PDG = 80.3692
ZETA3 = 1.2020569031595942
GAM2_NF6 = (1249.0 + (-2216.0 / 27 - 160.0 * ZETA3 / 3) * 6 - 140.0 * 36 / 81) / 64.0
C3_YT_QCD = -128.0 * GAM2_NF6   # (16pi^2)^3 beta_yt  >  C3 * yt * g3^6  (pure QCD 3-loop mass anom. dim.)


def merge_json(path, key, obj):
    d = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
    d[key] = obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)


def fit_Mt(Mt, als, Mh, MW=MW_PDG):
    dt = Mt - 173.34
    da = (als - 0.1184) / 0.0007
    dh = Mh - 125.15
    dw = (MW - 80.384) / 0.014
    lam = 0.12604 + 0.00206 * dh - 0.00004 * dt
    yt = 0.93690 + 0.00556 * dt - 0.00042 * da
    g2 = 0.64779 + 0.00004 * dt + 0.00011 * dw
    gY = 0.35830 + 0.00011 * dt - 0.00020 * dw
    g3 = 1.1666 + 0.00314 * da - 0.00046 * dt
    return np.array([gY, g2, g3, yt, lam])


def planck_fit(Mt, als, Mh, MW=80.384):
    dt = Mt - 173.34
    da = (als - 0.1184) / 0.0007
    dh = Mh - 125.15
    dw = (MW - 80.384) / 0.014
    return dict(g1=0.6154 + 0.0003 * dt - 0.0006 * dw, g2=0.5055, g3=0.4873 + 0.0002 * da,
                yt=0.3825 + 0.0051 * dt - 0.0021 * da,
                lam=-0.0143 - 0.0066 * dt + 0.0018 * da + 0.0029 * dh)


def beta(y, nl=2, qcd3=False):
    gY, g2, g3, yt, lam = y
    gY2, g22, g32, yt2 = gY * gY, g2 * g2, g3 * g3, yt * yt
    bgY = K1 * 41.0 / 6 * gY ** 3
    bg2 = -K1 * 19.0 / 6 * g2 ** 3
    bg3 = -K1 * 7.0 * g3 ** 3
    byt = K1 * yt * (4.5 * yt2 - 17.0 / 12 * gY2 - 2.25 * g22 - 8.0 * g32)
    bl = K1 * (24 * lam * lam - 6 * yt2 * yt2 + 3.0 / 8 * (2 * g22 * g22 + (g22 + gY2) ** 2)
               + lam * (12 * yt2 - 9 * g22 - 3 * gY2))
    if nl >= 2:
        K2 = K1 * K1
        bgY += K2 * gY ** 3 * (199.0 / 18 * gY2 + 4.5 * g22 + 44.0 / 3 * g32 - 17.0 / 6 * yt2)
        bg2 += K2 * g2 ** 3 * (1.5 * gY2 + 35.0 / 6 * g22 + 12 * g32 - 1.5 * yt2)
        bg3 += K2 * g3 ** 3 * (11.0 / 6 * gY2 + 4.5 * g22 - 26 * g32 - 2 * yt2)
        byt += K2 * yt * (-12 * yt2 ** 2 + yt2 * (131.0 / 16 * gY2 + 225.0 / 16 * g22 + 36 * g32 - 12 * lam)
                          + 1187.0 / 216 * gY2 ** 2 - 0.75 * g22 * gY2 + 19.0 / 9 * g32 * gY2
                          - 23.0 / 4 * g22 ** 2 + 9 * g22 * g32 - 108 * g32 ** 2 + 6 * lam ** 2)
        bl += K2 * (-312 * lam ** 3 + 36 * lam ** 2 * (3 * g22 + gY2) - 144 * lam ** 2 * yt2
                    + lam * (-73.0 / 8 * g22 ** 2 + 39.0 / 4 * g22 * gY2 + 629.0 / 24 * gY2 ** 2)
                    + lam * yt2 * (22.5 * g22 + 85.0 / 6 * gY2 + 80 * g32) - 3 * lam * yt2 ** 2
                    + 305.0 / 16 * g22 ** 3 - 289.0 / 48 * g22 ** 2 * gY2 - 559.0 / 48 * g22 * gY2 ** 2
                    - 379.0 / 48 * gY2 ** 3
                    + yt2 * (-9.0 / 4 * g22 ** 2 + 21.0 / 2 * g22 * gY2 - 19.0 / 4 * gY2 ** 2)
                    + yt2 ** 2 * (-8.0 / 3 * gY2 - 32 * g32) + 30 * yt2 ** 3)
    if qcd3:
        K3 = K1 ** 3
        bg3 += K3 * 65.0 / 2 * g3 ** 7
        byt += K3 * C3_YT_QCD * yt * g32 ** 3
    return np.array([bgY, bg2, bg3, byt, bl])


def solve(Mt, als, Mh, MW=MW_PDG, nl=2, qcd3=False, mu_hi=3e19):
    y0 = fit_Mt(Mt, als, Mh, MW)
    return solve_ivp(lambda t, y: beta(y, nl, qcd3), (math.log(Mt), math.log(mu_hi)), y0,
                     method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)


def at(sol, mu):
    return sol.sol(np.log(mu))


# calibration offset 3-loop(61e) - 2-loop at MPbar, filled in main (also importable)
CAL_SETS = [(173.34, 0.1184, 125.15), (172.60, 0.1180, 125.13), (172.10, 0.1180, 125.13),
            (173.34, 0.1191, 125.15), (172.60, 0.1171, 125.13)]


MPL_NONRED = 1.2209e19   # Eq.(61) is at the NON-reduced Planck mass: our 2-loop g1,g2,g3 hit (61a-c) there


def calibration(nl=2, qcd3=False, mu_cal=MPL_NONRED):
    offs = []
    for (Mt, a, Mh) in CAL_SETS:
        s = solve(Mt, a, Mh, MW=80.384, nl=nl, qcd3=qcd3)
        offs.append(planck_fit(Mt, a, Mh)["lam"] - at(s, mu_cal)[4])
    return float(np.mean(offs)), [float(o) for o in offs]


def couplings(Mt, als, Mh=125.13, mu=MU_B, cal=0.0, nl=2, qcd3=False):
    s = solve(Mt, als, Mh, nl=nl, qcd3=qcd3)
    y = at(s, mu)
    return dict(gY=float(y[0]), g2=float(y[1]), g3=float(y[2]), yt=float(y[3]), lam=float(y[4] + cal),
                lam_raw=float(y[4]))


def main():
    out = {}
    out["mu_b_GeV"] = MU_B
    out["C3_yt_QCD"] = C3_YT_QCD
    # --- validation against Buttazzo (61a-e) at their central inputs ---
    s = solve(173.34, 0.1184, 125.15, MW=80.384)
    y = at(s, MPBAR)
    pf = planck_fit(173.34, 0.1184, 125.15)
    out["validation_MPl_central_2L"] = dict(g1=float(math.sqrt(5 / 3) * y[0]), g2=float(y[1]), g3=float(y[2]),
                                           yt=float(y[3]), lam=float(y[4]), buttazzo_61=pf)
    s3 = solve(173.34, 0.1184, 125.15, MW=80.384, qcd3=True)
    y3 = at(s3, MPBAR)
    out["validation_MPl_central_2L+QCD3"] = dict(g3=float(y3[2]), yt=float(y3[3]), lam=float(y3[4]))
    yN = at(s, MPL_NONRED)
    out["validation_at_1.2209e19_central_2L"] = dict(g1=float(math.sqrt(5 / 3) * yN[0]), g2=float(yN[1]),
                                                     g3=float(yN[2]), yt=float(yN[3]), lam=float(yN[4]))
    calR, offsR = calibration(mu_cal=MPBAR)
    out["calibration_if_61_were_at_reduced_MPl"] = dict(mean=calR, offsets=offsR)
    cal, offs = calibration()
    cal3, offs3 = calibration(qcd3=True)
    out["calibration_61e_minus_2L"] = dict(mean=cal, sets=CAL_SETS, offsets=offs, mu_cal=MPL_NONRED)
    out["calibration_61e_minus_2L+QCD3"] = dict(mean=cal3, offsets=offs3)
    # offset growth: compare 2L+QCD3 minus 2L at mu_b and at MPl (how the 3-loop shift scales with scale)
    s2c = solve(172.60, 0.1180, 125.13)
    s3c = solve(172.60, 0.1180, 125.13, qcd3=True)
    out["qcd3_shift_at_mub_vs_MPl"] = dict(mu_b=float(at(s3c, MU_B)[4] - at(s2c, MU_B)[4]),
                                            MPl=float(at(s3c, MPBAR)[4] - at(s2c, MPBAR)[4]))
    # --- requested points ---
    pts = {}
    for Mt in (172.60, 172.10, 172.33, 172.87, 172.22, 172.59, 171.0):
        for a in (0.1180, 0.1171, 0.1189):
            c = couplings(Mt, a, cal=cal)
            pts[f"Mt={Mt:.2f},as={a:.4f}"] = c
    out["points_mu_b"] = pts
    # derivatives (2L, calibrated identical)
    def lam_b(Mt, a, Mh=125.13):
        return couplings(Mt, a, Mh)["lam_raw"]
    d = {}
    d["dlam_dMt"] = (lam_b(172.70, 0.1180) - lam_b(172.50, 0.1180)) / 0.2
    d["dlam_das"] = (lam_b(172.60, 0.1181) - lam_b(172.60, 0.1179)) / 0.0002
    d["dlam_dMh"] = (lam_b(172.60, 0.1180, 125.23) - lam_b(172.60, 0.1180, 125.03)) / 0.2
    d["err_Mt_0.27"] = abs(d["dlam_dMt"]) * 0.27
    d["err_as_0.0009"] = abs(d["dlam_das"]) * 0.0009
    d["err_Mh_0.11"] = abs(d["dlam_dMh"]) * 0.11
    d["err_exp_quad"] = math.sqrt(d["err_Mt_0.27"] ** 2 + d["err_as_0.0009"] ** 2 + d["err_Mh_0.11"] ** 2)
    out["derivatives_mu_b"] = d
    # beta_lambda at mu_b, instability scale, minimum
    yb = at(s2c, MU_B)
    out["beta_lambda_mu_b_2L"] = float(beta(yb)[4])
    f = lambda lm: at(s2c, 10 ** lm)[4]
    out["log10_mu_lambda0_central_2L"] = float(brentq(f, 6, 14))
    lm = np.linspace(12, 19, 701)
    lams = np.array([f(v) for v in lm])
    out["lambda_min_central_2L"] = dict(log10mu=float(lm[np.argmin(lams)]), lam=float(lams.min()),
                                        lam_min_cal=float(lams.min() + cal))
    out["lambda_at_MPl_central_2L"] = float(at(s2c, MPBAR)[4])
    out["lambda_at_MPl_central_cal"] = float(at(s2c, MPBAR)[4] + cal)
    # critical m_t where calibrated lambda(mu_b) = 0
    out["Mt_lambda_b_zero_cal"] = float(brentq(lambda m: couplings(m, 0.1180)["lam_raw"] + cal, 169.0, 173.0))
    merge_json(_zp.w("r1/R14_checks/results.json"), "check1_rge", out)
    st = {"check1": {"done": True, "lam_mu_b_172.60_raw": pts["Mt=172.60,as=0.1180"]["lam_raw"],
                     "lam_mu_b_172.60_cal": pts["Mt=172.60,as=0.1180"]["lam"], "cal": cal,
                     "yt_mu_b": pts["Mt=172.60,as=0.1180"]["yt"]}}
    merge_json(_zp.w("r1/R14_checks/STATUS.json"), "check1", st["check1"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
