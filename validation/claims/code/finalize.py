# claims-blind (V3, R16): assemble BLIND_RESULTS.json (key numbers, extra variants, w0 -> Delta m_c, SHA-256 of code and inputs)
import os, sys, json, glob, hashlib, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stats as S
ROOT = S.ROOT; KB = ROOT + r"\07_reviews\R16_v12_round3\keystone_blind"; OUTD = S.OUTD
ST = json.load(open(os.path.join(OUTD, "stats_results.json"), encoding="utf-8"))
SE = json.load(open(os.path.join(OUTD, "sens_results.json"), encoding="utf-8"))
W0 = json.load(open(os.path.join(OUTD, "w0_results.json"), encoding="utf-8"))
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()

def main():
    R = dict(agent="claims-blind", task="V3 R16 blind claims/robustness check", timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # ---------- 1. marginalisation ----------
    M = ST["marginalisation"]; m1 = {}
    for dn in S.DEFS:
        d = M[dn]; mc = d["pole_0.10|model_A"]["MC"]
        m1[dn] = dict(mu_pole=S.DEFS[dn]["mu"], sigma=S.DEFS[dn]["sig"], Pbar=d["pole_0.10|model_A"]["Pbar_grid"], Pbar_MC=mc["mean"], MC_se=mc["mc_se"],
                      median=mc["median"], CI68=mc["CI68"], CI95=mc["CI95"], prob_P1_gt_0p5=mc["prob_P1_gt_0p5"], prob_P1_lt_0p1=mc["prob_P1_lt_0p1"],
                      variants={k: v["Pbar_grid"] for k, v in d.items()},
                      BKPV_FJ_MC=dict(mean=d["BKPV_FJ_-0.36/+0.17|model_A"]["MC"]["mean"], CI68=d["BKPV_FJ_-0.36/+0.17|model_A"]["MC"]["CI68"], CI95=d["BKPV_FJ_-0.36/+0.17|model_A"]["MC"]["CI95"]),
                      mbar_MC=dict(mean=d["mbar_0.07|model_A"]["MC"]["mean"], CI68=d["mbar_0.07|model_A"]["MC"]["CI68"], CI95=d["mbar_0.07|model_A"]["MC"]["CI95"]))
    # extra variant: Hoang 2020 p.15 calibration m_MC = m_pole + (0.57 +- 0.29) GeV (Ref. 154 there)
    PT = S.Pbar_true("A", True, S.SAS, ("normal", 0.10)); mu, sg = 172.60 - 0.57, float(np.hypot(0.27, 0.29))
    Ba = S.Pbar_meas(PT, mu, sg)
    pm = {}
    for (a, b) in [(170.0, 176.0), (150.0, 200.0)]:
        inr = (S.MG >= a) & (S.MG <= b); lo = max(0.0, min(S.MG[0], b) - a); pm[f"[{a:g},{b:g}]"] = float((np.sum(PT[inr])*S.H_ + lo)/(b - a))
    mcv = S.mc_run(mu, sg, ("normal", 0.10), seed=2026)
    m1["variant_MC_calibration_Hoang_p15"] = dict(mu_pole=mu, sigma=sg, Pbar=Ba, Pbar_MC=mcv["mean"], CI68=mcv["CI68"], CI95=mcv["CI95"],
        B_existence_as_data=Ba, B_conditioned_170_176=Ba/pm["[170,176]"], B_conditioned_150_200=Ba/pm["[150,200]"],
        source="Hoang, arXiv:2004.12915, printed p.15: m_MC = m_pole + (0.57+-0.29) GeV (from its Ref. 154); same page fn.12: m_pole - m_MSR(1.3 GeV) = 360+-250 MeV (its Ref. 40)")
    R["1_marginalisation"] = m1
    # ---------- 2. Bayes ----------
    R["2_bayes"] = {k: v for k, v in ST["bayes"].items()}
    # ---------- 3. exclusion & future ----------
    EX = ST["exclusion"]
    R["3_exclusion"] = dict(mc_P1_0p1={a: EX[a]["P1=0.1"]["mc"] for a in ["0.1180", "0.1171", "0.1189"]},
                            mc_all_levels={a: {k: v["mc"] for k, v in EX[a].items()} for a in ["0.1180", "0.1171", "0.1189"]},
                            model_band_P1_0p1_at_0p1180=EX["0.1180"]["P1=0.1"]["model_band"],
                            marginalised_true_mass_thresholds=EX["marginalised_true_mass_thresholds"],
                            required_central_value_95=EX["required_central_value_95"], threshold_T01_sigma_eff=EX["threshold_T01_sigma_eff"],
                            definitions=dict(defA="posterior P(m_true > m_c(P1=0.1; alpha_s, SM, model)) = 0.95 (one-sided 95% exclusion of the P1>=0.1 region)",
                                             defB="marginalised survival Pbar1(m_meas) = 0.05 (existence-as-data p = 0.05)"))
    R["3_future"] = ST["future"]
    # ---------- 4. sensitivity ----------
    A = SE["a_prefactor"]; RG = SE["rg_response"]
    R["4a_hLb"] = dict(prefactor_exact={k: {"x0.9": A[k]["hLb_x0.9"], "x1.1": A[k]["hLb_x1.1"], "mc": A[k]["mc"]} for k in A},
                       h_driven_mu_b_covarying={k: {t: RG[k][f"V1D|hLb_x{f}_h-driven(mu_b x{f})"]["dmc"] for t, f in [("x0.9", "0.9"), ("x1.1", "1.1")]} for k in RG},
                       note="prefactor: hazard exponent I -> I/f exactly (gamma(u) unchanged at fixed mu_b); h-driven: hL_b ~ h^(1/2) and mu_b ~ h^(1/2) at fixed turning condition, so mu_b x f too (RG shifts of lam, y_t, g2, gY with node-0 1D sensitivities)")
    R["4b_xi"] = dict(turning=SE["turning_condition"],
                      dmc_total={k: RG[k]["V1D|xi_0.28_total"]["dmc"] for k in RG}, dmc_prefactor={k: RG[k]["V1D|xi_0.28_prefactor_only"]["dmc"] for k in RG},
                      dmc_mu_only={k: RG[k]["V1D|xi_0.28_mu_only"]["dmc"] for k in RG}, dmc_C149_prefactor=SE["b_xi_C149_prefactor_only"],
                      dmc_C149_total_estimate={P: SE["b_xi_C149_prefactor_only"][P] + RG["KB_v2_all_2D"]["V1D|xi_0.28_mu_only"]["dmc"][P] for P in ["P0.9", "P0.5", "P0.1"]},
                      kappas_node0=SE["kappas_node0"])
    # ---------- 5. w0 ----------
    w = {}
    for k, v in W0.items():
        if not isinstance(v, dict) or "varmin" not in v: continue
        w[k] = dict(Mt=v.get("Mt"), Q=v["Q"], gamma=v["gamma"], Q_sieg=v.get("Q_sieg"), gamma_sieg=v.get("gamma_sieg"), rtop=v["rtop"], vtop=v["vtop"], rout=v["rout"],
                    s_varmin=v["varmin"]["s"], E_varmin=v["varmin"]["E"], dE_varmin=v["varmin"]["dE"], w0_wall=v["varmin"]["w0_wall"], w0_cprod=v["varmin"]["w0_cprod"],
                    w0_trunc_rtop=v["varmin"]["w0_trunc"]["rtop"], w0_trunc_rout=v["varmin"]["w0_trunc"]["rout"], s_Emin_local=v["Emin"]["s"], w0_wall_Emin=v["Emin"]["w0_wall"],
                    cap_state_imag_frac_inner=v.get("cap_state_imag_frac_inner"), cap_state_L2_beyond_rtop=v.get("cap_state_L2_beyond_rtop"), n_local_minima_var=v.get("n_local_minima_var"))
    R["5_w0"] = w
    # w0 -> Delta m_c on the C149 curve, if survival = w0 * exp(-I) (non-resonant remainder escapes promptly)
    wv = [w[k][q] for k in w if k.startswith("v2_all") for q in ("w0_wall", "w0_cprod", "w0_trunc_rtop", "w0_trunc_rout")]
    eff = {}
    for tag, w0 in [("v2_all_min", min(wv)), ("v2_all_max", max(wv)), ("v2_all_mid", 0.5*(min(wv) + max(wv)))]:
        row = {}
        for P, xl in zip(S.LEV, S.XL):
            Hc = -np.log(P) + np.log(w0)
            row[f"P{P:g}"] = float(S.CV.m_of_level(np.log(Hc)) - S.CV.m_of_level(xl)) if Hc > 0 else None
        eff[tag] = dict(w0=w0, dmc=row, Pbar_direct_scaled=w0*m1["direct_MC"]["Pbar"])
    R["5_w0_effect_on_mc_C149"] = eff
    # ---------- checks ----------
    R["checks"] = dict(
        grid_vs_MC_Pbar={dn: [m1[dn]["Pbar"], m1[dn]["Pbar_MC"], m1[dn]["MC_se"]] for dn in S.DEFS},
        curve_reproduces_table_mc=ST["curve"]["mc_from_curve_without_crit_knots"], table_mc=ST["curve"]["mc_table"],
        level_dependence_alpha_s_effect=float(M["direct_MC"]["pole_0.10|model_A"]["MC"]["mean"] - M["direct_MC"]["pole_0.10|model_A"]["MC_levindep"]["mean"]),
        model_shape_A_vs_B=[M["direct_MC"]["pole_0.10|model_A"]["Pbar_grid"], M["direct_MC"]["pole_0.10|model_B"]["Pbar_grid"]],
        truncated_Ba_sigma_pole_grid_vs_MC=[ST["bayes"]["[170,176]|sigma_pole"]["B_existence_as_data"], ST["bayes"]["MC_check_[170,176]|sigma_pole_Ba"]],
        hLb_prefactor_analytic_P0p5=dict(C149=float(np.log(0.9)/S.CV.slope(S.mc_tab("0.1180", 0.5))), exact=A["C149_with_cross"]["hLb_x0.9"]["P0.5"]),
        w0_gridcheck_dr0p0025=[W0["gridcheck_v2_all_173.0_dr0.0025"]["varmin"]["w0_wall"], W0["v2_all_V1D|173.0"]["varmin"]["w0_wall"]],
        turning_condition_selfconsistency=SE["turning_condition"]["consistency"])
    # ---------- hashes ----------
    code = [os.path.join(HERE, f) for f in ["stats.py", "sens.py", "w0.py", "finalize.py", "fetch_lit.py", "lit_search.py", "inspect_inputs.py"]]
    kbcode = [os.path.join(KB, "work", f) for f in ["production.py", "model.py", "solvers.py", "oneloop.py", "collect.py"]] + [os.path.join(KB, "v2", f) for f in ["xmodel.py", "twisted.py", "run_v2.py"]]
    R["code_sha256"] = {os.path.basename(f): sha(f) for f in code}
    R["reused_keystone_blind_code_sha256"] = {os.path.relpath(f, KB): sha(f) for f in kbcode}
    shards = sorted(glob.glob(os.path.join(KB, "v2", "hazard_shards_all", "*.json")))
    hh = hashlib.sha256(); [hh.update(open(f, "rb").read()) for f in shards]
    LIT = ROOT + r"\02_literature\batches\20261014_claude_v12_r3_L37\claims_blind"
    R["input_sha256"] = {"C149_run02_summary.json": sha(S.C149F), "keystone_blind/results/couplings.json": sha(os.path.join(KB, "results", "couplings.json")),
                         "keystone_blind/results/hazard_summary.json": sha(os.path.join(KB, "results", "hazard_summary.json")),
                         "keystone_blind/v2/hazard_shards_all/*.json (concatenated, sorted)": hh.hexdigest(), "n_shards": len(shards),
                         "keystone_blind/v2/cache/xg_all_node00.pkl": sha(os.path.join(KB, "v2", "cache", "xg_all_node00.pkl")),
                         "keystone_blind/work/cache/gauss_node00.npz": sha(os.path.join(KB, "work", "cache", "gauss_node00.npz")),
                         "HLLHC_YR_WG1_1902.04070.pdf": sha(os.path.join(LIT, "HLLHC_YR_WG1_1902.04070.pdf")), "FCC_FSR_Vol1_2505.00272.pdf": sha(os.path.join(LIT, "FCC_FSR_Vol1_2505.00272.pdf")),
                         "Hoang_topmass_review_2004.12915.txt": sha(ROOT + r"\02_literature\batches\20261014_claude_v12_r3_L37\sm_mapping\Hoang_topmass_review_2004.12915.txt")}
    R["result_files"] = ["results/stats_results.json", "results/sens_results.json", "results/w0_results.json"]
    json.dump(R, open(os.path.join(OUTD, "BLIND_RESULTS.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(dict(m1_calib=m1["variant_MC_calibration_Hoang_p15"], w0eff=eff, checks=R["checks"]), indent=1, default=float))

if __name__ == "__main__":
    main()
