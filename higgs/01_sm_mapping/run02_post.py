# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/run02_post.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] run02 analysis: compiled SMDR vs Python port, true matching-scale variation, rewritten SM-side budget (C148r2_summary.json).  Paper: PRD App. H5.

C148 run02 post.  'hazards': full C144 hazard (copied) for sfit_* variants into run02/hazard_runs (<=4 parallel);
'analyze': compiled-SMDR vs Python port, true matching-scale spread, Delta m_c, rewritten budget -> run02 files."""
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
import time
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import r02  # noqa: E402
import r3util as U  # noqa: E402

HZ2 = os.path.join(r02.RES2, "hazard_runs")
HVARS = ["sfit_xi1", "sfit_xi0.5", "sfit_xi2"]   # xi = 1/4, 4: linear reference only
KEYS = ["P1=0.9", "P1=0.5", "P1=0.1"]


def env_for(v):
    e = dict(os.environ)
    e.update(C148_VARIANT=v, C148_ALS="0.1180", C148_HAZRES=HZ2, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    return e


def hazards():
    os.makedirs(os.path.join(HZ2, "logs"), exist_ok=True)
    r02.status2({"stage": "hazards running", "next_command": "python -B run02_post.py hazards (resumable) ; "
                                                             "python -B run02_post.py analyze"})
    ps = []
    for v in HVARS:
        fo = open(os.path.join(HZ2, "logs", "haz_%s.log" % v), "w", encoding="utf-8")
        ps.append((subprocess.Popen([sys.executable, "-B", "C144_hazard.py", "haz", "1"], cwd=HERE, env=env_for(v),
                                    stdout=fo, stderr=subprocess.STDOUT), fo, v))
    for p, fo, v in ps:
        p.wait()
        fo.close()
    for v in HVARS:
        with open(os.path.join(HZ2, "logs", "post_%s.log" % v), "w", encoding="utf-8") as fo:
            subprocess.run([sys.executable, "-B", "C144_hazard.py", "post"], cwd=HERE, env=env_for(v), stdout=fo,
                           stderr=subprocess.STDOUT)
    r02.status2({"stage": "hazards done", "next_command": "python -B run02_post.py analyze"},
                done=["run02 hazards " + ",".join(HVARS)], files=["hazard_runs/"])


def analyze():
    import sm_provider as SP
    import sm_rge as RG
    F = U.jload(os.path.join(r02.RES2, "fits", "fits.json"))["fits"]
    b1 = U.jload(os.path.join(U.RES, "C148_bench.json"))
    S = b1["F_budget_Mt173.0"]["base"]["dlam_dMt"]
    sm1 = U.jload(os.path.join(U.RES, "C148_summary.json"))
    out = dict(meta=dict(S_dlam_dMt=S, note="compiled SMDR 1.3 (MinGW gcc 5.3), c148_fit.exe; Mh 125.13, as 0.1180"))
    # A: compiled xi=1 vs Python central at mu = Mt; B: lam_b, yt_b three ways
    A, B = {}, {}
    for mt, d in F["xi1"].items():
        Mt = float(mt)
        x = SP.match_Mt(Mt, 0.1180, 125.13)
        A[mt] = {k: d[kk] - x[k] for k, kk in (("lam", "lambda"), ("yt", "yt"), ("g3", "g3"), ("g", "g"), ("gp", "gp"))}
        y = np.array([d["g3"], d["g"], d["gp"], d["yt"], d["yb"], d["ytau"], d["lambda"], d["v"]])
        s = RG.solve(y, Mt, SP.MU_B * 1.01, loops=5)
        own = s.sol(math.log(SP.MU_B))
        pc = SP.coup_point(Mt, 0.1180, 125.13, SP.MU_B)
        B[mt] = dict(lam_SMDR_RG=d["RG_mub"]["lambda"], lam_ownRG=float(own[6]), lam_py_central=pc["lam"],
                     yt_SMDR_RG=d["RG_mub"]["yt"], yt_ownRG=float(own[3]), yt_py_central=pc["yt"],
                     d_lam_ownRG_minus_SMDRRG=float(own[6]) - d["RG_mub"]["lambda"],
                     d_lam_compiled_minus_python=float(own[6]) - pc["lam"],
                     dmc_lin_compiled_minus_python=-(float(own[6]) - pc["lam"]) / S)
    out["A_match_at_Mt_compiled_minus_python"] = A
    out["B_mu_b"] = B
    # C: matching-scale configs (own RGE from each fitted point), relative to xi1
    C = {}
    for cfg in F:
        C[cfg] = {}
        for mt in ("172.60", "173.10"):
            d, d1 = F[cfg].get(mt), F["xi1"][mt]
            if not d:
                continue
            Mt = float(mt)
            res = []
            for dd in (d, d1):
                y = np.array([dd["g3"], dd["g"], dd["gp"], dd["yt"], dd["yb"], dd["ytau"], dd["lambda"], dd["v"]])
                res.append(RG.solve(y, Mt, SP.MU_B * 1.01, loops=5).sol(math.log(SP.MU_B)))
            dl, dy = float(res[0][6] - res[1][6]), float(res[0][3] - res[1][3])
            C[cfg][mt] = dict(dlam_b=dl, dyt_b=dy, dmc_lin=-dl / S, dlam_SMDR_RG=d["RG_mub"]["lambda"] - d1["RG_mub"]["lambda"],
                              dx_Mt={k: d[k] - d1[k] for k in ("lambda", "yt", "g3", "g", "gp")}, niter=d.get("niter"),
                              time=d.get("time"))
    out["C_configs_vs_xi1"] = C
    # D: full hazard
    H = {}
    for v in HVARS:
        p = os.path.join(HZ2, "C144_hazard_%s.json" % v)
        if os.path.exists(p):
            H[v] = json.load(open(p, encoding="utf-8"))["critical_mt"].get("0.1180")
    cen = sm1["critical_mt_pole"]["0.1180"]
    D = {}
    for v, cm in H.items():
        if cm:
            D[v] = dict(critical_mt=cm, dmc_vs_run01_central={k: cm[k] - cen[k] for k in KEYS},
                        dmc_vs_sfit_xi1=({k: cm[k] - H["sfit_xi1"][k] for k in KEYS} if H.get("sfit_xi1") else None))
    out["D_hazard"] = D
    # E: budget
    def lin(cfg, mt="173.10"):
        return C.get(cfg, {}).get(mt, {}).get("dmc_lin")

    def full(cfg, k="P1=0.5"):
        v = "sfit_" + cfg
        if v in D and D[v].get("dmc_vs_run01_central"):   # delta-mode variant = run01 central + compiled xi-shift
            return D[v]["dmc_vs_run01_central"][k]
        return lin(cfg)
    ms = dict(xi0_5=full("xi0.5"), xi2=full("xi2"), xi0_25=lin("xi0.25"), xi4=lin("xi4"),
              lin_xi0_5=lin("xi0.5"), lin_xi2=lin("xi2"))
    a_pole = max(abs(ms["xi0_5"]), abs(ms["xi2"]))
    q = b1["F_budget_Mt173.0"]["mu0_decomposition"]
    qh, q2 = q["xi=0.5"]["dmc_components"]["qcd_yt"], q["xi=2"]["dmc_components"]["qcd_yt"]
    a_mbar = max(abs(ms["xi0_5"] - qh), abs(ms["xi2"] - q2))
    bud = sm1["budget_P1_0p5"]
    b_pole, b_mbar, c = bud["components"]["b_pole"], bud["components"]["b_mbar"], bud["components"]["c"]
    tot_pole = math.sqrt(a_pole ** 2 + b_pole ** 2 + c ** 2)
    tot_mbar = math.sqrt(a_mbar ** 2 + b_mbar ** 2 + c ** 2)
    out["E_budget_P1=0.5"] = dict(
        a_matching_scale_compiledSMDR=dict(common_xi=ms, half_width_pole=a_pole, components_xi0p5_xi2={
            c_: [lin(c_ + "0.5"), lin(c_ + "2")] for c_ in ("t", "h", "Z", "G", "d")},
            mbar_form=dict(pure_QCD_zm_part=[qh, q2], half_width=a_mbar,
                           note="approx: subtract own pure-QCD z_m tower (run01) from the compiled common-xi shift")),
        run01_RG_reconstruction_envelope=bud["a_matching_scale"]["envelope"],
        b_truncation_pole=b_pole, b_truncation_mbar=b_mbar, c_scheme=c,
        total_pole=tot_pole, total_mbar=tot_mbar, target_met_pole=tot_pole <= 0.1, target_met_mbar=tot_mbar <= 0.1)
    U.wjson(os.path.join(r02.RES2, "C148r2_summary.json"), out)
    r02.status2({"stage": "analyze done", "key_numbers": dict(matching_common_xi=ms, total_pole=tot_pole,
                                                              total_mbar=tot_mbar,
                                                              compiled_minus_python_lam_b=B.get("172.60", {}).get("d_lam_compiled_minus_python"),
                                                              hazard=D),
                 "next_command": "append 'run02 附录' to D120 (manual), then STATUS complete"},
                done=["run02 analyze"], files=["C148r2_summary.json"])
    print(json.dumps(U.clean(dict(A=A, B=B, ms=ms, comps=out["E_budget_P1=0.5"]["a_matching_scale_compiledSMDR"],
                                  D=D, tot_pole=tot_pole, tot_mbar=tot_mbar,
                                  C_small={k: {m: (round(v["dmc_lin"], 4), v["niter"]) for m, v in C[k].items()} for k in C})),
                     indent=1))


if __name__ == "__main__":
    {"hazards": hazards, "analyze": analyze}[sys.argv[1]]()
