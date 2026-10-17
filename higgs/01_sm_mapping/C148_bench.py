# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/C148_bench.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Benchmarks of the SM mapping: Buttazzo 2013 Eq. (61), BKPV critical M_t, the r2 -> r3 mapping chain, linear budget items at M_t = 173.0; reads the SMDR 1.3 benchmark/figure data (third_party/smdr).  Paper: PRD App. H5.

C148 benchmarks + SM-side error budget at the lambda(mu_b) level (hazard propagation is separate)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import re
import sys
import glob
import math
import json
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import r3util as U  # noqa: E402
import qcd as Q  # noqa: E402
import sm_rge as RG  # noqa: E402
import sm_provider as SP  # noqa: E402

SMDR = _zp.smdr()  # [v12-package] SMDR 1.3 tree (third_party/smdr)
MH0, AS0 = 125.13, 0.1180
OUT = {}


def rows(fn):
    d = []
    for ln in open(fn, encoding="utf-8", errors="replace"):
        if ln.startswith("#") or not ln.strip():
            continue
        d.append([float(x) for x in ln.split()])
    return np.array(d)


def refmodel(fn):
    d = {}
    for m in re.finditer(r"SMDR_(\w+)\s*=\s*([0-9eE\.\+\-/ ]+);", open(fn, encoding="utf-8").read()):
        e = m.group(2).strip()
        d[m.group(1)] = (float(e.split("/")[0]) / float(e.split("/")[1])) if "/" in e else float(e)
    return d


def ref_y(d):
    return np.array([d["g3_in"], d["g_in"], d["gp_in"], d["yt_in"], d["yb_in"], d["ytau_in"], d["lambda_in"], d["v_in"]])


# ------------------------------------------------------------------------------------------------ A: RGE vs SMDR data
def sec_A():
    F = rows(os.path.join(SMDR, "fig_data/FIG_RGrun_vs_Q.dat"))
    F = F[F[:, 0] >= 173.1]
    y0 = ref_y(refmodel(os.path.join(SMDR, "benchmark_data/ReferenceModel_2019.dat")))
    res = {}
    for loops in (5, 3):
        s = RG.solve(y0, 173.1, 1.2e19, loops=loops)
        Y = s.sol(np.log(F[:, 0]))
        cols = dict(g3=(0, 2), g=(1, 3), gp=(2, 4), yt=(3, 11), lam=(6, 20), v=(7, 22))
        res["loops%d" % loops] = {k: dict(maxabs=float(np.max(np.abs(Y[i] - F[:, c]))),
                                          at_1e19=[float(Y[i][-1]), float(F[-1, c])])
                                  for k, (i, c) in cols.items()}
        j = int(np.argmin(np.abs(np.log10(F[:, 0]) - 15.5)))
        res["loops%d" % loops]["lam_at_10^15.5"] = [float(Y[6][j]), float(F[j, 20])]
    H = rows(os.path.join(SMDR, "fig_data/FIG_lambda_m2_highQ.dat"))
    s = RG.solve(y0, 173.1, 1.0e20, loops=5)
    sel = H[:, 0] >= 200
    res["lambda_highQ_central_vs_own"] = float(np.max(np.abs(s.sol(np.log(H[sel, 0]))[6] - H[sel, 4])))
    OUT["A_rge_vs_SMDR_FIG_RGrun"] = res


# ------------------------------------------------------------------------------------------------ B: Buttazzo (61)
def sec_B():
    out = {}
    x = SP.buttazzo(173.34, 0.1184, 125.15, 80.384)
    x.update(yb=SP.SMDR25["yb"], ytau=SP.SMDR25["ytau"], v=246.6)
    g3qcd = math.sqrt(4 * math.pi * Q.as6_at(173.34, 0.1184, nrun=4, ndec=3))
    eq61 = dict(g1=0.6154, g2=0.5055, g3=0.4873, yt=0.3825, lam=-0.0143)
    for tag, g3 in (("eq60_g3", x["g3"]), ("ownQCD_g3", g3qcd)):
        for impl, loops in (("b13", 4), ("smdr", 3), ("smdr", 5)):
            y = SP.vec(dict(x, g3=g3))
            s = RG.solve(y, 173.34, 1.3e19, loops=loops, impl=impl)
            for mu in (SP.MPL, SP.MPBAR):
                Y = s.sol(math.log(mu))
                out["%s|%s%d|mu=%.4g" % (tag, impl, loops, mu)] = dict(
                    g1=float(math.sqrt(5 / 3) * Y[2]), g2=float(Y[1]), g3=float(Y[0]), yt=float(Y[3]), lam=float(Y[6]))
    out["eq61"] = eq61
    out["g3_Mt_eq60_vs_ownQCD"] = [x["g3"], g3qcd]
    OUT["B_buttazzo_eq61"] = out


# ------------------------------------------------------------------------------------------------ C: SMDR ref models
def sec_C():
    out = {}
    for fn in sorted(glob.glob(os.path.join(SMDR, "benchmark_data/ReferenceModel*.dat"))):
        d = refmodel(fn)
        Mt, Mh, als, Qin = d["Mt_pole"], d["Mh_pole"], d["alphaS_5_MZ"], d["Q_in"]
        y = ref_y(d)
        if abs(Qin - Mt) > 1e-6:
            y = RG.solve(y, Qin, Mt, loops=5).y[:, -1]
        r = dict(Mt=Mt, Mh=Mh, als=als, MW_PDG_pred=d.get("MW_PDG"), lam=y[6], yt=y[3], g3=y[0], g=y[1], gp=y[2])
        for sch in ("B13", "B13c", "BKPV15", "SMDR25"):
            x = SP.match_Mt(Mt, als, Mh, SP.MW_PDG, sch)
            r["d_" + sch] = {k: x[k] - r[k] for k in ("lam", "yt", "g3", "g", "gp")}
        sol = RG.solve(y, Mt, SP.MU_B * 1.01, loops=5)
        r["lam_b_direct"] = float(sol.sol(math.log(SP.MU_B))[6])
        r["lam_b_central_provider"] = SP.lam_b(Mt, als, Mh)
        out[os.path.basename(fn)] = r
    OUT["C_SMDR_reference_models"] = out


# ------------------------------------------------------------------------------------------------ D: BKPV
def mt_crit_stability(variant, als, Mh, lo=169.0, hi=174.0):
    def lmin(Mt):
        sol, _, _ = SP._solution(Mt, als, Mh, SP.MW_PDG, variant)
        f = lambda lg: float(sol.sol(lg * math.log(10))[6])
        r = minimize_scalar(f, bounds=(14.0, 19.05), method="bounded", options=dict(xatol=1e-6))
        return r.fun, r.x
    m = brentq(lambda M: lmin(M)[0], lo, hi, xtol=1e-6)
    return m, lmin(m)[1]


def sec_D():
    out = {}
    B = SP.BKPV
    for sch in ("SMDR25", "B13", "B13c"):
        x = SP.match_Mt(B["Mt"], B["als"], B["Mh"], B["MW"], sch)
        out["x_minus_BKPV_x0_" + sch] = {k: x[k] - B["x0"][k] for k in B["x0"] if k != "yb"}
    for v in ("BKPV15", "central", "loops3", "B13", "B13c"):
        m, lg = mt_crit_stability(v, B["als"], B["Mh"])
        out["Mt_crit_lam=beta=0|" + v] = dict(Mt=m, log10_mu=lg)
    out["BKPV_quoted"] = dict(Mt_cri=171.44, log10mu=17.752, dmu_plus=-0.36, dmu_minus=0.17, dtru=-0.02,
                              note="BKPV PRL Table II, inputs Mh 125.7, as 0.1185, criterion lam=beta_lam=0")
    OUT["D_BKPV"] = out


# ------------------------------------------------------------------------------------------------ E: R14 + chain
def sec_E():
    import R14_rge as R14
    cal, _ = R14.calibration()
    Mt = 172.6
    out = {}
    c = R14.couplings(Mt, AS0, MH0, cal=cal)
    out["R14_r2"] = dict(lam_cal=c["lam"], lam_raw=c["lam_raw"], yt=c["yt"], g2=c["g2"], gY=c["gY"], g3=c["g3"], cal=cal)
    chain = [("B13_2L_noyb", dict(match="B13", loops=2, qcdN=3, decN=3)),
             ("B13_3L", dict(match="B13", loops=3, qcdN=3, decN=3)),
             ("B13_5L", dict(match="B13", loops=5, qcdN=3, decN=3)),
             ("B13_g3QCD", "B13_g3fix"), ("B13c", "B13c"), ("central_SMDR25", "central")]
    prev = out["R14_r2"]["lam_cal"]
    for nm, v in chain:
        if v == "B13_g3fix":
            x = SP.match_Mt(Mt, AS0, MH0, SP.MW_PDG, "B13")
            x["g3"] = SP.match_Mt(Mt, AS0, MH0, SP.MW_PDG, "B13c")["g3"]
            s = RG.solve(SP.vec(x), Mt, SP.MU_B * 1.01, loops=5)
            l, yt = float(s.sol(math.log(SP.MU_B))[6]), float(s.sol(math.log(SP.MU_B))[3])
        else:
            cc = SP.coup_point(Mt, AS0, MH0, SP.MU_B, v)
            l, yt = cc["lam"], cc["yt"]
        out[nm] = dict(lam_b=l, yt_b=yt, dlam_vs_prev=l - prev)
        prev = l
    nodes = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
    mus = SP.MU_B / np.sqrt(np.cosh(2 * nodes))
    cc = SP.couplings(Mt, AS0, MH0, mus, "central")
    r14 = R14.solve(Mt, AS0, MH0)
    Y14 = r14.sol(np.log(mus))
    out["nodes_172.6"] = dict(u=nodes.tolist(), mu=mus.tolist(), **{k: cc[k].tolist() for k in ("lam", "yt", "g2", "gY", "g3")},
                              R14_lam_cal=(Y14[4] + cal).tolist(), R14_yt=Y14[3].tolist(), R14_g2=Y14[1].tolist(),
                              R14_gY=Y14[0].tolist())
    OUT["E_R14_chain"] = out


# ------------------------------------------------------------------------------------------------ F: error budget
def sec_F(Mt=173.0):
    out = {}
    base = SP.lam_b(Mt, AS0, MH0)
    d = SP.rge_estimates("central", Mt, AS0, MH0)
    S = d["dlam_dMt"]
    out["base"] = dict(Mt=Mt, **d)

    def dm(dl):
        return -dl / S
    items = {}
    for v in ("xi0.5", "xi2", "xi0.316", "xi3.16", "loops2", "loops3", "loops4", "rge_b13", "SMDR19", "BKPV15",
              "B13", "B13c", "B13c_l3", "MWinput", "z4hi", "z4lo", "BKPV15_xi0.5", "BKPV15_xi2",
              "xi0.5b", "xi2b", "xi0.316b", "xi3.16b", "qcd5", "Mh125.20"):
        l = SP.lam_b(Mt, AS0, MH0, v)
        ref = SP.lam_b(Mt, AS0, MH0, "BKPV15") if v.startswith("BKPV15_") else base
        items[v] = dict(dlam_b=l - ref, dmc=dm(l - ref))
    for nm, kw in (("Mh+0.11", dict(Mh=MH0 + 0.11)), ("Mh-0.11", dict(Mh=MH0 - 0.11)), ("Mh=125.20", dict(Mh=125.20)),
                   ("as+0.0009", dict(als=AS0 + 0.0009)), ("as-0.0009", dict(als=AS0 - 0.0009))):
        l = SP.lam_b(Mt, kw.get("als", AS0), kw.get("Mh", MH0))
        items[nm] = dict(dlam_b=l - base, dmc=dm(l - base))
    # threshold truncation by known partial higher orders (sizes)
    x = SP.match_Mt(Mt, AS0, MH0, SP.MW_PDG, "SMDR25")
    a = x["g3"] ** 2 / (4 * math.pi ** 2)
    zt = Q.zm_terms(a, 0.0, 4)
    l3 = SP.lam3_qcd(x["g3"], Mt)
    sens = {}
    for k, h in (("lam", 1e-4), ("yt", 1e-4), ("g3", 1e-4), ("g", 1e-4), ("gp", 1e-4)):
        xx = dict(x)
        xx[k] += h
        s = RG.solve(SP.vec(xx), Mt, SP.MU_B * 1.01, loops=5)
        sens[k] = (float(s.sol(math.log(SP.MU_B))[6]) - base) / h
    out["dlam_b_dx_at_Mt"] = sens
    items["thr_yt_4loopQCD_term_size"] = dict(dlam_b=sens["yt"] * x["yt"] * zt[4], dmc=dm(sens["yt"] * x["yt"] * zt[4]))
    items["thr_lam_3loopQCD_term_size"] = dict(dlam_b=sens["lam"] * l3, dmc=dm(sens["lam"] * l3), lam3=l3)
    g3d3 = math.sqrt(4 * math.pi * Q.as6_at(Mt, AS0, nrun=5, ndec=3))
    g3d4 = math.sqrt(4 * math.pi * Q.as6_at(Mt, AS0, nrun=5, ndec=4))
    g3r4 = math.sqrt(4 * math.pi * Q.as6_at(Mt, AS0, nrun=4, ndec=4))
    items["thr_g3_dec4_minus_dec3"] = dict(dlam_b=sens["g3"] * (g3d4 - g3d3), dmc=dm(sens["g3"] * (g3d4 - g3d3)))
    items["thr_g3_run5_minus_run4"] = dict(dlam_b=sens["g3"] * (g3d4 - g3r4), dmc=dm(sens["g3"] * (g3d4 - g3r4)))
    items["g3_SMDR_minus_ownQCD"] = dict(dlam_b=sens["g3"] * (x["g3"] - g3d4), dmc=dm(sens["g3"] * (x["g3"] - g3d4)),
                                         dg3=x["g3"] - g3d4)
    out["items"] = items
    out["zm_terms"] = zt
    # matching-scale decomposition
    dec = {}
    y = SP.vec(x)
    for xi in (0.5, 2.0):
        yN, det = SP.mu0_shift(y, Mt, AS0, xi, detail=True)
        sol = RG.solve(y, Mt, xi * Mt, loops=5)
        yR = sol.y[:, -1]
        comp = {}
        for nm, idx in (("rest_g3", 0), ("rest_g", 1), ("rest_gp", 2), ("rest_yt", 3), ("rest_lam", 4),
                        ("qcd_yt", 10), ("qcd_g3", 11)):
            yy = yR.copy()
            if idx == 4:
                yy[6] += det["Drest"][4]
            elif idx < 4:
                yy[idx] *= math.exp(det["Drest"][idx])
            elif idx == 10:
                yy[3] *= math.exp(det["dq_yt"])
            else:
                yy[0] *= math.exp(det["dq_g3"])
            s = RG.solve(yy, xi * Mt, SP.MU_B * 1.01, loops=5)
            comp[nm] = float(s.sol(math.log(SP.MU_B))[6]) - base
        s = RG.solve(yN, xi * Mt, SP.MU_B * 1.01, loops=5)
        dec["xi=%g" % xi] = dict(detail=det, dlam_b_components=comp, dlam_b_total=float(s.sol(math.log(SP.MU_B))[6]) - base,
                                 dmc_components={k: dm(v) for k, v in comp.items()})
    out["mu0_decomposition"] = dec
    # SMDR-data matching-scale estimate
    F = rows(os.path.join(SMDR, "fig_data/FIG_Mt_vs_Q_1.dat"))
    sel = (F[:, 0] >= 173.1 / 2 - 1e-9) & (F[:, 0] <= 2 * 173.1 + 1e-9)
    dMt = F[sel, 10] - 173.1
    L = rows(os.path.join(SMDR, "fig_data/FIG_lambda_vs_Q.dat"))
    sl = (L[:, 0] >= 173.1 / 2 - 1e-9) & (L[:, 0] <= 2 * 173.1 + 1e-9)
    rl = L[sl, 12] - 1
    out["SMDR_data_scale"] = dict(Q_range=[float(F[sel, 0].min()), float(F[sel, 0].max())],
                                  dMt_max=float(dMt.max()), dMt_min=float(dMt.min()),
                                  dMt_at_Qmin=float(dMt[0]), dMt_at_Qmax=float(dMt[-1]),
                                  lam_ratio_minus1_max=float(rl.max()), lam_ratio_minus1_min=float(rl.min()),
                                  dmc_from_lam=[dm(sens["lam"] * 0.126 * float(rl.max())),
                                                dm(sens["lam"] * 0.126 * float(rl.min()))],
                                  note="dm_c = Mt(Q)-Mt(Q0) at fixed MSbar parameters (FIG_Mt_vs_Q_1 col 11)")
    OUT["F_budget_Mt%.1f" % Mt] = out


def main():
    t0 = time.time()
    for f in (sec_A, sec_B, sec_C, sec_D, sec_E, sec_F):
        t1 = time.time()
        f()
        print(f.__name__, "%.1fs" % (time.time() - t1), flush=True)
        U.wjson("C148_bench.json", OUT)
    OUT["time_s"] = time.time() - t0
    U.wjson("C148_bench.json", OUT)
    U.status({"stage": "bench done", "next_command": "python -B run_hazards.py (hazard variants) ; python -B C148_post.py"},
             done=["selftest", "bench (A-F)"], files=["C148_selftest.json", "C148_bench.json"])


if __name__ == "__main__":
    main()
