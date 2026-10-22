# -*- coding: utf-8 -*-
"""C170 S0 regression (mandatory first step).

(a) lambda-override path with the couplings of the archived run05 node job at u = 0 (columns m_t = 172.6 and 173.0,
    alpha_s = 0.1180) and the production seeds of C149_direct.jobX (T14b root + run01 seed shift, window (0.02, 0.0015))
    -> widths vs the archived shard (target: relative difference <= 1e-8);
(b) the same with the generic seed of this work (1D adiabatic root, no production seed);
(c) the same at node u = 5.0 (m_t = 172.6) -> override path at mu_b/105 (box-C-like scale);
(d) sm_provider couplings at mu_b for (172.6/173.0, 0.1180, 125.13) vs the couplings recorded in the archived shard;
(e) static threshold of the archived u = 0 widths for N = eta_inf = 299.84 vs the hazard threshold of run05.
Usage: python -B c170_s0.py
"""
import os
import sys
import math
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

W, D = CC.W, CC.D
ARCH = os.path.join(CC.WORK, "archive", "r3", "holonomy_cross", "run05")
TGT = {"P1=0.9": 0.9, "P1=0.5": 0.5, "P1=0.1": 0.1}


def main():
    t0 = time.time()
    stage = CC.stage_inputs()
    CC.status_update({"stage": "S0 running", "S0": {"state": "running"}})
    sh0 = CC.jload(os.path.join(ARCH, "shards", "X_main_u0.0000.json"))
    sh5 = CC.jload(os.path.join(ARCH, "shards", "X_main_u5.0000.json"))
    summ = CC.jload(os.path.join(ARCH, "C149_run05_summary.json"))
    out = dict(stage_record=stage, archived_cfg=sh0["cfg"], archived_window=sh0["window"])
    mu_b = CC.mu_node(0.0)
    out["mu_b_GeV"] = mu_b
    # (d) couplings
    dc = {}
    for x in sh0["cols"]:
        if round(x["mt"], 2) in (172.6, 173.0):
            c = CC.couplings_at(x["mt"], x["als"], mu_b)
            dc["%.1f" % x["mt"]] = {k: dict(sm_provider=c[k], archived=x[k], diff=c[k] - x[k]) for k in
                                    ("g2", "gY", "yt", "lam", "g3")}
    out["couplings_vs_archive_u0"] = dc
    # (a), (b), (c)
    reg = {}
    for tag, sh, u, mts in (("u0", sh0, 0.0, (172.6, 173.0)), ("u5", sh5, 5.0, (172.6,))):
        for x in sh["cols"]:
            if round(x["mt"], 2) not in mts:
                continue
            pt = dict(model="final", g2=x["g2"], gY=x["gY"], yt=x["yt"], g3=x["g3"], lam=x["lam"])
            e14, _ = D.t14_root(x["mt"], x["als"], u)
            dq, rat = D.seed_shift(x["lam"])
            cen, gim = e14.real + dq, -2 * e14.imag * rat
            ra = CC.solve_point(pt, center=cen, gim=gim)
            rows = dict(archived=dict(eps=x["Q"], gamma=x["gamma"], center=x["center"], Emin=x["Emin"]),
                        prod_seed=dict(center=cen, gim=gim, center_minus_archived=cen - x["center"],
                                       eps=ra["eps"], gamma=ra["gamma"], Emin=ra["Emin"], flags=ra["flags"],
                                       rel_gamma=ra["gamma"] / x["gamma"] - 1, d_eps=ra["eps"] - x["Q"],
                                       box_dE=ra.get("box_dE"), barrier=ra["barrier"], root1d=ra.get("root1d"),
                                       status=ra["status"], time_s=ra["time_s"], info=ra["info"]))
            if tag == "u0":
                rb = CC.solve_point(pt)
                rows["generic_seed"] = dict(seed=rb.get("seed"), eps=rb.get("eps"), gamma=rb.get("gamma"),
                                            flags=rb.get("flags"), status=rb.get("status"),
                                            rel_gamma=(rb["gamma"] / x["gamma"] - 1) if rb.get("gamma") else None,
                                            d_eps=(rb["eps"] - x["Q"]) if rb.get("eps") else None,
                                            time_s=rb.get("time_s"))
            reg["%s_mt%.1f" % (tag, x["mt"])] = rows
            print(tag, x["mt"], json.dumps(CC.clean({k: v for k, v in rows["prod_seed"].items()
                                                     if k in ("rel_gamma", "d_eps", "flags", "status", "time_s")})),
                  flush=True)
    out["regression"] = reg
    # (e) static threshold of the archived u = 0 widths, N = eta_inf
    mts = np.array([x["mt"] for x in sh0["cols"]])
    lam = np.array([x["lam"] for x in sh0["cols"]])
    gam = np.array([x["gamma"] for x in sh0["cols"]])
    xx = np.abs(lam) ** -0.5
    o = np.argsort(xx)
    spx = CubicSpline(xx[o], np.log(gam[o]))
    spm = CubicSpline(mts, np.log(gam))
    spl = CubicSpline(mts, lam)
    st = {}
    for k, P in TGT.items():
        tg = math.log(-math.log(P) / CC.ETA)
        xc = brentq(lambda z: spx(z) - tg, xx.min(), xx.max())
        mc = brentq(lambda z: spm(z) - tg, mts.min(), mts.max())
        lam_h = summ["lam_c"][k]
        st[k] = dict(lam_c_static_x_spline=-1 / xc ** 2, lam_c_static_mt_spline=float(spl(mc)), mc_static=mc,
                     lam_c_hazard=lam_h, mc_hazard=summ["crit"]["0.1180"][k],
                     shallower_by=(-1 / xc ** 2) - lam_h, shallower_by_mt_spline=float(spl(mc)) - lam_h,
                     I_over_Gamma0_eta=-math.log(P) / (math.exp(float(spx(abs(lam_h) ** -0.5))) * CC.ETA))
    out["static_threshold_archived_u0"] = dict(N=CC.ETA, rows=st,
                                               note="ln gamma(u=0) interpolated (cubic spline) over the 10 archived "
                                                    "columns m_t = 172.0..173.8 against |lambda|^(-1/2) and against m_t")
    out["time_s"] = time.time() - t0
    CC.wjson(os.path.join(CC.OUT, "S0", "C170_S0.json"), out)
    worst = max(abs(v["prod_seed"]["rel_gamma"]) for v in reg.values())
    worstb = max(abs(v["generic_seed"]["rel_gamma"]) for v in reg.values() if "generic_seed" in v)
    CC.status_update({"stage": "S0 done", "S0": {"state": "done", "max_rel_gamma_prod_seed": worst,
                                                  "max_rel_gamma_generic_seed": worstb,
                                                  "file": "S0/C170_S0.json", "time_s": out["time_s"]}})
    print(json.dumps(CC.clean(dict(worst_prod=worst, worst_generic=worstb, static=st, couplings=dc)), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
