# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/C148_post.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Post-processing of the round-3 SM mapping (C148_summary.json: lam(mu_b), y_t(mu_b) at (172.60, 0.1180, 125.13), budget, m_t definitions).  Needs the archived C144-pipeline hazard runs (historical) in the work tree.

C148 post: critical m_t per provider variant (full hazard), linear consistency, SM-side error budget,
m_t definitions, node tables -> C148_summary.json, C148_tables.md, STATUS.json."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import math
import json
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import r3util as U  # noqa: E402
import qcd as Q  # noqa: E402
import sm_provider as SP  # noqa: E402

HZ = os.path.join(U.RES, "hazard_runs")
KEYS = ["P1=0.9", "P1=0.5", "P1=0.1"]
R2 = {"P1=0.9": 172.52, "P1=0.5": 173.02, "P1=0.1": 173.51}
R2_LAMC = {"P1=0.9": -0.00965, "P1=0.5": -0.01291, "P1=0.1": -0.01624}
MH0 = 125.13


def load(v):
    p = os.path.join(HZ, "C144_hazard_%s.json" % v)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def lamc_from_table(tab, mc):
    mts = sorted(float(k) for k in tab)
    lb = [tab["%.2f" % m]["lam_b"] for m in mts]
    return float(CubicSpline(mts, lb)(mc))


def main():
    bench = U.jload("C148_bench.json")
    lin = bench["F_budget_Mt173.0"]["items"]
    S = bench["F_budget_Mt173.0"]["base"]["dlam_dMt"]
    r2 = json.load(open(os.path.join(U.R2RES, "C144_hazard.json"), encoding="utf-8"))
    out = dict(r2_critical=r2["critical_mt"], r2_D116_rounded=dict(R2), r2_lamc=R2_LAMC)
    R2.update(r2["critical_mt"]["0.1180"])   # exact r2 hazard values (D116 table: 172.52/173.02/173.51)
    C = load("central")
    crit = {a: C["critical_mt"][a] for a in C["critical_mt"]}
    out["central"] = dict(critical_mt=crit, P1_at=C["P1_at"], dmc_dalpha_s=C["dmc_dalpha_s"],
                          meta=dict(n_homotopy={k: v["n_homotopy"] for k, v in C["meta"]["tasks"].items()},
                                    flags={k: len(v["flags"]) for k, v in C["meta"]["tasks"].items()},
                                    max_resid={k: v["max_resid"] for k, v in C["meta"]["tasks"].items()}),
                          rge=C["rge"])
    tab = C["P1_table"]["0.1180"]
    lamc = {k: lamc_from_table(tab, crit["0.1180"][k]) for k in KEYS}
    out["central"]["lamc_mu_b"] = lamc
    # couplings at mu_b at the critical masses; m_t definitions
    defs = {}
    for k in KEYS:
        mc = crit["0.1180"][k]
        c = SP.coup_point(mc, 0.1180, MH0, SP.MU_B)
        x = SP.match_Mt(mc, 0.1180, MH0)
        as6 = x["g3"] ** 2 / (4 * math.pi)
        mb = Q.mbar_mbar(mc, as6)
        defs[k] = dict(pole=mc, mbar_mbar_QCD4L=mb, yt_MSbar_at_Mt=x["yt"], lam_mu_b=c["lam"], yt_mu_b=c["yt"],
                       g2_mu_b=c["g2"], gY_mu_b=c["gY"], g3_mu_b=c["g3"],
                       MC_if_pole_identification=mc, MC_if_MSR1p3_identification=mc - 0.36)
    out["mt_definitions"] = defs
    # variants
    var = {}
    for v in ["xi0.5", "xi2", "xi0.5b", "xi2b", "loops3", "BKPV15", "B13c", "B13", "SMDR19", "qcd5", "Mh125.20"]:
        d = load(v)
        if d is None or "0.1180" not in d["critical_mt"]:
            var[v] = dict(missing=True, dmc_linear=lin.get(v, {}).get("dmc"))
            continue
        cm = d["critical_mt"]["0.1180"]
        var[v] = dict(critical_mt=cm, dmc_full={k: (cm[k] - crit["0.1180"][k]) if cm.get(k) else None for k in KEYS},
                      dmc_linear=lin.get(v, {}).get("dmc"),
                      lamc_mu_b={k: lamc_from_table(d["P1_table"]["0.1180"], cm[k]) for k in KEYS if cm.get(k)},
                      n_homotopy=d["meta"]["tasks"]["as0.1180"]["n_homotopy"],
                      flags=len(d["meta"]["tasks"]["as0.1180"]["flags"]))
    out["variants"] = var

    def full(v, k="P1=0.5"):
        x = var.get(v, {})
        if x.get("dmc_full") and x["dmc_full"].get(k) is not None:
            return x["dmc_full"][k]
        return lin[v]["dmc"]
    # budget (P1=0.5 shifts; P1=0.9/0.1 shifts are listed in variants)
    ms = [full("xi0.5"), full("xi2"), full("xi0.5b"), full("xi2b")]
    bud = dict(
        a_matching_scale=dict(envelope=[min(ms), max(ms)], half_width=max(abs(min(ms)), abs(max(ms))),
                              fwd=[full("xi0.5"), full("xi2")], bwd=[full("xi0.5b"), full("xi2b")],
                              SMDR_data=[-0.009, 0.043], BKPV_quoted_FJ=[-0.36, 0.17]),
        b_truncation=dict(rge_5L_vs_3L=abs(full("loops3")), z4_3pct=abs(lin["z4hi"]["dmc"]),
                          pole_series_5loop_est=abs(full("qcd5")), g3_dec_run=abs(lin["thr_g3_dec4_minus_dec3"]["dmc"]) +
                          abs(lin["thr_g3_run5_minus_run4"]["dmc"])),
        c_scheme=dict(tadpole_FJ_vs_free_BKPV=abs(full("BKPV15")), input_set_MW=abs(lin["MWinput"]["dmc"]),
                      SMDR_version_slopes=abs(full("SMDR19")), own_QCD_B13c=abs(full("B13c"))),
        d_parametric=dict(Mh_0p11=abs(lin["Mh+0.11"]["dmc"]), as_0p0009=0.5 * abs(lin["as+0.0009"]["dmc"] - lin["as-0.0009"]["dmc"]),
                          Mh_125p20_shift=full("Mh125.20")))
    a = bud["a_matching_scale"]["half_width"]
    b_mbar = math.sqrt(bud["b_truncation"]["rge_5L_vs_3L"] ** 2 + bud["b_truncation"]["z4_3pct"] ** 2
                       + bud["b_truncation"]["g3_dec_run"] ** 2)
    b_pole = math.sqrt(b_mbar ** 2 + bud["b_truncation"]["pole_series_5loop_est"] ** 2)
    c = math.sqrt(sum(v ** 2 for v in bud["c_scheme"].values()))
    bud["total_SM_side_pole"] = math.sqrt(a * a + b_pole ** 2 + c * c)
    bud["total_SM_side_mbar"] = math.sqrt(a * a + b_mbar ** 2 + c * c)
    bud["components"] = dict(a=a, b_pole=b_pole, b_mbar=b_mbar, c=c)
    bud["target_0p1_met_pole"] = bud["total_SM_side_pole"] <= 0.1
    bud["target_0p1_met_mbar"] = bud["total_SM_side_mbar"] <= 0.1
    out["budget_P1=0.5"] = bud
    # linear consistency  m_c <-> lambda_c
    lc = {}
    for k in KEYS:
        lam_r2_at = None
        lc[k] = dict(mc_new=crit["0.1180"][k], mc_r2=R2[k], dmc=crit["0.1180"][k] - R2[k],
                     lamc_new=lamc[k], lamc_r2=R2_LAMC[k], dlamc=lamc[k] - R2_LAMC[k])
    import R14_rge as R14
    cal, _ = R14.calibration()
    for k in KEYS:
        m = R2[k]
        lam_r2 = R14.couplings(m, 0.1180, MH0, cal=cal)["lam"]
        lam_new = SP.lam_b(m, 0.1180, MH0)
        lc[k]["lam_b_at_mc_r2"] = dict(r2_provider=lam_r2, new_provider=lam_new, diff=lam_new - lam_r2)
        lc[k]["dmc_linear_pred"] = -(lam_new - lam_r2) / S + (lamc[k] - R2_LAMC[k]) / S
        lc[k]["dmc_linear_pred_lamonly"] = -(lam_new - lam_r2) / S
    out["linear_consistency"] = lc
    # nodes at m_c(P1=0.5)
    nodes = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
    mus = SP.MU_B / np.sqrt(np.cosh(2 * nodes))
    cc = SP.couplings(crit["0.1180"]["P1=0.5"], 0.1180, MH0, mus)
    out["nodes_at_mc_P1=0.5"] = dict(Mt=crit["0.1180"]["P1=0.5"], u=nodes.tolist(), mu=mus.tolist(),
                                     **{kk: cc[kk].tolist() for kk in ("lam", "yt", "g2", "gY", "g3")})
    U.wjson("C148_summary_raw.json", out)
    write_tables(out, bench)
    E = bench["E_R14_chain"]
    c0 = SP.coup_point(172.6, 0.1180, MH0, SP.MU_B)
    files = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE)) if f.endswith(".py")]
    inputs = [_zp.smdr("src/betas.c"),
              _zp.smdr("benchmark_data/ReferenceModel.dat"),
              _zp.smdr("benchmark_data/ReferenceModel_2019.dat"),
              _zp.smdr("src/QCDQED_match.c"),
              "C52_Buttazzo_etal_2013.pdf",
              os.path.join(U.LIT, "BKPV_PRL_1507.08833.pdf"), os.path.join(U.LIT, "Marquard_4loop_1502.01030.pdf"),
              os.path.join(U.R2RES, "C144_ueff.json"), os.path.join(U.R2RES, "C144_hazard.json")]
    summ = dict(
        task="C148/D120 sm_mapping", agent="sm-mapping", date="2026-10-14",
        provider="sm_provider.couplings(Mt, alphas, Mh, mu_array, variant); central = SMDR25 matching at mu0=Mt + "
                 "SMDR-port RGE (3L full + 4L gauge + 4L QCD-leading yt, lam + 5L QCD)",
        point_172p60_0p1180_125p13=dict(lam_mu_b=c0["lam"], yt_mu_b=c0["yt"], g2_mu_b=c0["g2"], gY_mu_b=c0["gY"],
                                        g3_mu_b=c0["g3"], r2_R14_lam_cal=E["R14_r2"]["lam_cal"], r2_R14_yt=E["R14_r2"]["yt"],
                                        dlam_vs_r2=c0["lam"] - E["R14_r2"]["lam_cal"]),
        critical_mt_pole=out["central"]["critical_mt"], lamc_mu_b_0p1180=lamc, r2_D116=R2, r2_lamc=R2_LAMC,
        dmc_vs_r2_0p1180={k: crit["0.1180"][k] - R2[k] for k in KEYS},
        dmc_dalpha_s=out["central"]["dmc_dalpha_s"], P1_at_central=out["central"]["P1_at"],
        r2_to_r3_chain_at_172p6={k: dict(lam_b=E[k]["lam_b"], dlam=E[k]["dlam_vs_prev"],
                                         dmc_lin=-E[k]["dlam_vs_prev"] / S)
                                 for k in ("B13_2L_noyb", "B13_3L", "B13_5L", "B13_g3QCD", "B13c", "central_SMDR25")},
        budget_P1_0p5=bud, variants_full_hazard={k: v.get("dmc_full") for k, v in var.items()},
        linear_consistency=lc, mt_definitions=defs,
        target_pm0p1=dict(pole=bud["target_0p1_met_pole"], mbar=bud["target_0p1_met_mbar"],
                          total_pole=bud["total_SM_side_pole"], total_mbar=bud["total_SM_side_mbar"]),
        code_sha256={os.path.basename(p): U.sha256(p) for p in files},
        inputs_sha256={os.path.relpath(p, U.ROOT): U.sha256(p) for p in inputs if os.path.exists(p)},
        files=dict(results=os.path.relpath(U.RES, U.ROOT), code=os.path.relpath(HERE, U.ROOT),
                   derivation=os.path.relpath(os.path.join(U.DER, "D120_SM_MAPPING_NNLO_CN.md"), U.ROOT)))
    U.wjson("C148_summary.json", summ)
    U.status({"stage": "post done", "key_numbers": dict(critical_mt=out["central"]["critical_mt"]["0.1180"],
                                                         lam_mu_b_172p6=c0["lam"], yt_mu_b_172p6=c0["yt"],
                                                         total_SM_pole=bud["total_SM_side_pole"],
                                                         total_SM_mbar=bud["total_SM_side_mbar"]),
              "next_command": "python -B mk_wolfram.py ; write D120_SM_MAPPING_NNLO_CN.md"},
             done=["post (C148_summary.json, C148_tables.md)"], files=["C148_summary.json", "C148_summary_raw.json",
                                                                       "C148_tables.md"])
    print(json.dumps(U.clean(dict(central=out["central"]["critical_mt"], lamc=lamc, budget=bud,
                                  variants={k: v.get("dmc_full") for k, v in var.items()},
                                  linear=lc, defs=defs)), indent=1))


def f(x, n=3):
    return "—" if x is None else ("%+.*f" % (n, x))


def write_tables(o, bench):
    L = ["# C148 tables (sm-mapping, 2026-10-14)", ""]
    cm = o["central"]["critical_mt"]
    L += ["## T1 central critical m_t (pole, GeV) — new provider (SMDR25 + 3L/4L/5L RGE) vs r2 (D116)", "",
          "| alpha_s | P1=0.9 | P1=0.5 | P1=0.1 |", "|---|---|---|---|"]
    for a in sorted(cm):
        L.append("| %s | %s |" % (a, " | ".join("%.3f" % cm[a][k] if cm[a].get(k) else "—" for k in KEYS)))
    L.append("| r2 0.1180 | 172.52 | 173.02 | 173.51 |")
    L += ["", "## T2 variants (alpha_s = 0.1180): Delta m_c full hazard [linear lambda_b estimate at 173.0]", "",
          "| variant | dm_c(0.9) | dm_c(0.5) | dm_c(0.1) | linear |", "|---|---|---|---|---|"]
    for v, d in o["variants"].items():
        if d.get("missing"):
            L.append("| %s | — | — | — | %s |" % (v, f(d.get("dmc_linear"))))
        else:
            L.append("| %s | %s | %s | %s | %s |" % (v, f(d["dmc_full"]["P1=0.9"]), f(d["dmc_full"]["P1=0.5"]),
                                                    f(d["dmc_full"]["P1=0.1"]), f(d["dmc_linear"])))
    b = o["budget_P1=0.5"]
    L += ["", "## T3 SM-side budget for m_c(P1=0.5) (GeV)", "", "```", json.dumps(U.clean(b), indent=1), "```"]
    L += ["", "## T4 m_t definitions at the critical points (alpha_s = 0.1180, Mh = 125.13)", "",
          "| P1 | pole | mbar(mbar) QCD 4L | yt(Mt) MSbar | MC (=pole) | MC (=MSR 1.3 GeV) | lam(mu_b) | yt(mu_b) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, d in o["mt_definitions"].items():
        L.append("| %s | %.3f | %.3f | %.5f | %.3f | %.3f | %.5f | %.5f |" % (
            k, d["pole"], d["mbar_mbar_QCD4L"], d["yt_MSbar_at_Mt"], d["MC_if_pole_identification"],
            d["MC_if_MSR1p3_identification"], d["lam_mu_b"], d["yt_mu_b"]))
    E = bench["E_R14_chain"]
    L += ["", "## T5 r2 -> r3 chain at (172.6, 0.1180, 125.13): lambda(mu_b)", "", "| step | lam_b | d lam_b | d m_c (lin) |",
          "|---|---|---|---|", "| R14 r2 (2L+cal) | %.6f | | |" % E["R14_r2"]["lam_cal"]]
    S = bench["F_budget_Mt173.0"]["base"]["dlam_dMt"]
    for k in ("B13_2L_noyb", "B13_3L", "B13_5L", "B13_g3QCD", "B13c", "central_SMDR25"):
        L.append("| %s | %.6f | %+.2e | %+.3f |" % (k, E[k]["lam_b"], E[k]["dlam_vs_prev"], -E[k]["dlam_vs_prev"] / S))
    L += ["", "## T6 nodes at m_c(P1=0.5) (central)", "", "| u | mu [GeV] | lam | yt | g2 | gY | g3 |", "|---|---|---|---|---|---|---|"]
    N = o["nodes_at_mc_P1=0.5"]
    for i in range(len(N["u"])):
        L.append("| %.3f | %.4e | %.6f | %.5f | %.5f | %.5f | %.5f |" % (N["u"][i], N["mu"][i], N["lam"][i], N["yt"][i],
                                                                     N["g2"][i], N["gY"][i], N["g3"][i]))
    with open(os.path.join(U.RES, "C148_tables.md"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
