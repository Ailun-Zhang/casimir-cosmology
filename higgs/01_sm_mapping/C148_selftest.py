# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/C148_selftest.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Self-tests of the SM mapping: RGE implementations (SMDR port vs B13), QCD conversions, regressions.  Paper: PRD App. H5.

C148 self-tests: QCD module, SMDR-port vs Buttazzo App.B beta functions, first provider numbers, R14 reference."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import time
import math
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import r3util as U  # noqa: E402
import qcd as Q  # noqa: E402
import sm_rge as RG  # noqa: E402
import sm_provider as SP  # noqa: E402


def beta_compare():
    rng = np.random.default_rng(148)
    pts = [np.array([1.1635, 0.6476, 0.3586, 0.931, 0.0, 0.0, 0.1263, 246.6]),
           np.array([0.488, 0.506, 0.476, 0.383, 0.0, 0.0, -0.0133, 237.0])]
    for _ in range(6):
        pts.append(np.array([rng.uniform(0.4, 1.3), rng.uniform(0.45, 0.7), rng.uniform(0.3, 0.5),
                             rng.uniform(0.3, 1.0), 0.0, 0.0, rng.uniform(-0.03, 0.15), 246.0]))
    names = ["g3", "g", "gp", "yt", "yb", "ytau", "lam"]
    worst = {}
    for p in pts:
        A = RG.smdr_orders(p, 4)
        B = RG.b13_orders(p, 4)
        for n in range(3):
            for i in (0, 1, 2, 3, 6):
                a, b = A[n][i], B[n][i]
                rel = abs(a - b) / max(abs(a), 1e-30)
                k = "L%d_%s" % (n + 1, names[i])
                worst[k] = max(worst.get(k, 0.0), rel)
        q = p.copy()
        q[1:4] = 1e-9
        q[6] = 0.0
        a4 = RG.smdr_orders(q, 4)[3][0]
        b4 = RG.b13_orders(q, 4)[3][0]
        worst["L4_g3_pureQCD"] = max(worst.get("L4_g3_pureQCD", 0.0), abs(a4 - b4) / abs(a4))
    return worst


def main():
    t0 = time.time()
    out = dict(qcd=Q.selfcheck(), smdr_betas_sha256=RG.SRC_SHA, beta_rel_diff_SMDRport_vs_B13=beta_compare())
    # QCD numbers at the reference point
    as6 = Q.as6_at(172.4, 0.1180)
    out["as6_Mt_172.4_0.1180"] = dict(nrun5_ndec4=as6, nrun4_ndec3=Q.as6_at(172.4, 0.1180, nrun=4, ndec=3),
                                      g3=math.sqrt(4 * math.pi * as6), SMDR25_g3=SP.SMDR25["g3"],
                                      B13_g3=SP.buttazzo(172.4, 0.1180, 125.2, 80.3692)["g3"])
    out["as6_Mt_173.34_0.1184_B13point"] = dict(as6=Q.as6_at(173.34, 0.1184),
                                                g3=math.sqrt(4 * math.pi * Q.as6_at(173.34, 0.1184)), B13=1.1666)
    a = as6 / math.pi
    out["zm_terms_at_mu=Mt"] = Q.zm_terms(a, 0.0, 4)
    out["mbar_mbar(172.4)"] = Q.mbar_mbar(172.4, as6)
    # provider
    res = {}
    for v in ["central", "loops3", "loops2", "rge_b13", "SMDR19", "BKPV15", "B13", "B13c", "MWinput", "xi0.5", "xi2"]:
        t1 = time.time()
        c = SP.coup_point(172.6, 0.1180, 125.13, SP.MU_B, v)
        res[v] = dict(lam=c["lam"], yt=c["yt"], g2=c["g2"], gY=c["gY"], g3=c["g3"], s=time.time() - t1)
    out["mu_b_172.6_0.1180_125.13"] = res
    # match values at Mt for the comparison
    out["match_at_Mt_172.6"] = {s: SP.match_Mt(172.6, 0.1180, 125.13, 80.3692, s)
                                for s in ["SMDR25", "SMDR19", "BKPV15", "B13", "B13c", "SMDR25_MW"]}
    # R14 (r2 provider) via unchanged copy
    import R14_rge as R14
    cal, _ = R14.calibration()
    c14 = R14.couplings(172.60, 0.1180, 125.13, cal=cal)
    out["R14_r2"] = dict(cal=cal, **c14)
    out["time_s"] = time.time() - t0
    U.wjson("C148_selftest.json", out)
    print(json.dumps(U.clean(out), indent=1))


if __name__ == "__main__":
    main()
