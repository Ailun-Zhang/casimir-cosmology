# claims-blind (V3, R16): comparison-phase diagnostics against production C154 run02 / D124 / D128 numbers.
import os, sys, json, time, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stats as S
H_ = S.H_; AG = H_*np.arange(-2000, 2001)
G = S.G_of_m("A", True)                                     # E_z[P1(m; z)] (model error, shape A)
SIG_MH = 0.11*0.0031655014860756166/0.006566065702598486    # production: M_h +-0.11 mapped to m_c (C154_inputs_v2 slopes) = 0.0530

def d_equal_mass(a, sm, sp):      # my convention: z ~ N(0,1) scaled by sm (z<0) / sp (z>0): median 0, 16/84% points at -sm/+sp, masses 1/2:1/2
    return np.where(a < 0, norm.pdf(a/sm)/sm, norm.pdf(a/sp)/sp)
def d_two_piece(a, sm, sp):       # production convention: continuous two-piece normal, mode 0, masses sm:sp, [-sm,+sp] holds 68.3%
    c = 2/(np.sqrt(2*np.pi)*(sm + sp)); return c*np.exp(-0.5*(a/np.where(a < 0, sm, sp))**2)

def Pbar_split(dfun, sm, sp, s_mh, mu=172.6, sig=0.27, das=S.SAS):
    ga = norm.pdf(AG, 0, np.hypot(S.SBAR*das, max(s_mh, 1e-9)))
    p = np.convolve(ga*H_, dfun(AG, sm, sp), mode="same"); p /= p.sum()*H_
    PT = S.conv_shift(G, p)
    d = dfun(AG, sm, sp); d /= d.sum()*H_
    return S.Pbar_meas(PT, mu, sig), float(np.sum(AG*d)*H_), float(np.sum(d[AG < 0])*H_)

def Pbar_sym(s_sm, s_mh, mu=172.6, sig=0.27, das=S.SAS):
    PT = S.Pbar_true("A", True, das, ("normal", float(np.hypot(s_sm, s_mh)))); return S.Pbar_meas(PT, mu, sig)

def main():
    R = dict(agent="claims-blind", task="V3 comparison-phase diagnosis", timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # ---------------- (1) BKPV/FJ ----------------
    B = {}
    for tag, f in [("equal_mass(blind)", d_equal_mass), ("two_piece(production)", d_two_piece)]:
        for mh in [0.0, SIG_MH]:
            for orient, (sm, sp) in [("down0.36_up0.17", (0.36, 0.17)), ("mirror_down0.17_up0.36", (0.17, 0.36))]:
                P, mean, mleft = Pbar_split(f, sm, sp, mh)
                B[f"{tag}|Mh{mh:.3f}|{orient}"] = dict(Pbar_direct=P, mean_shift_mc=mean, mass_on_down_side=mleft)
    B["symmetric_pole_0.10_noMh"] = Pbar_sym(0.10, 0.0); B["symmetric_pole_0.102_Mh0.053(production main)"] = Pbar_sym(0.102, SIG_MH)
    B["production"] = dict(Pbar_BKPV_FJ_SM_direct=0.625839, Pbar_main_direct=0.727106, src="run02/marginal_P1.csv; kernel(s_Mh, mu, s_up=0.17, s_dn=0.36) with sn_pdf two-piece normal (C154_core_v2.py l.90-100, 133)")
    B["V1_source"] = "07_reviews/R16_v12_round3/sm_mapping_blind/results/DIAGNOSIS.json item1: BKPV mu-variation reproduced as dmc = -0.358 / +0.173 (shift of the critical m_t; mu0=0.5 gives +0.173)"
    R["1_BKPV_FJ"] = B; print("BKPV", json.dumps(B, indent=0)[:1800], flush=True)
    # ---------------- (2) HL-LHC / FCC-ee inputs ----------------
    mod_sym = 0.5*(S.MODEL_SP[1] + S.MODEL_SM[1]); F = {}
    scen = {"prod_HLLHC_direct(0.20,as0.0009,SM0.102,Mh0.03)": (0.20, 0.0009, 0.102, 0.03),
            "HLLHC_0.20_as0.00045(lattice x2)": (0.20, 0.00045, 0.102, 0.03), "HLLHC_0.20_as0.0002(YR lattice assumption)": (0.20, 0.0002, 0.102, 0.03),
            "prod_HLLHC_as_permille(0.20,0.00012)": (0.20, 0.00012, 0.102, 0.03),
            "blind_HLLHC_0.17_as0.0002": (0.17, 0.0002, 0.10, 0.0), "blind_HLLHC_0.17_as0.00045": (0.17, 0.00045, 0.10, 0.0), "blind_HLLHC_0.17_as0.0009": (0.17, 0.0009, 0.10, 0.0),
            "HLLHC_0.17+interp0.5_as0.00045": (float(np.hypot(0.17, 0.5)), 0.00045, 0.10, 0.0),
            "prod_now_direct(0.27,0.0009,0.102,0.11)": (0.27, 0.0009, 0.102, SIG_MH),
            "prod_FCCee_base(0.0525,0.0002,0.070,0.01)": (0.0525, 0.0002, 0.070, 0.01*0.0031655/0.0065661), "blind_FCCee_50MeV(0.0502,0.0001,0.07)": (0.0502, 0.0001, 0.07, 0.0)}
    for k, (dm, da, ssm, smh) in scen.items():
        st = float(np.sqrt(dm**2 + (S.SBAR*da)**2 + ssm**2 + smh**2 + mod_sym**2))
        PT = S.Pbar_true("A", True, da, ("normal", float(np.hypot(ssm, smh))))
        F[k] = dict(sigma_tot=st, D_verdict=0.7/st, D_meas=0.7/dm, alpha_s_term=S.SBAR*da, Asimov_172p6=S.Pbar_meas(PT, 172.6, dm), Asimov_173p3=S.Pbar_meas(PT, 173.3, dm))
    F["production_csv"] = dict(now_direct_sigma_tot=0.462403, HLLHC_direct_sigma_tot=0.422272, HLLHC_permille_sigma_tot=0.242963, FCCee_base_sigma_tot=0.131275,
                               D_from_csv={"now": 0.7/0.462403, "HLLHC": 0.7/0.422272, "HLLHC_permille": 0.7/0.242963, "FCCee_base": 0.7/0.131275})
    F["sources_checked"] = dict(
        YR_1902_04070=dict(Table47_pdf_p149="CMS 3 ab^-1: ttbar lepton+jets total 0.17 GeV (MC-mass template, experimental)", pdf_p87="EW-fit keeps mt 0.4 GeV: 'difficult to further reduce ... at the HL-LHC'",
                           pdf_p88="EW-fit assumption: 'future lattice QCD measurements will provide ... delta alpha_S(MZ) = +-0.0002' (not an HL-LHC measurement)"),
        Corcella_1903_06574=dict(pdf_p13_sec4_2_10="CMS (Ref.[100] ECFA-2016, summarised in YR [98]): 'from about 0.2 GeV (0.1%) for direct reconstruction in the lepton+jets channel to 1.2 GeV (0.7%) from the total ttbar NNLO cross section' -> rounded secondary quote of the same 0.17 GeV"),
        FCC_FSR_2505_00272=dict(pdf_p97="world average 0.1180+-0.0009 'mainly constrained by the precision of lattice calculations, expected to be improved by a factor of two in the next decade' -> ~0.00045 by the HL-LHC era",
                                Table2_pdf_p37="FCC-ee alpha_S(mZ^2): stat 0.1e-4, syst 1e-4"))
    R["2_future_inputs"] = F; print("future", json.dumps({k: (round(v["D_verdict"], 3), round(v["sigma_tot"], 4)) for k, v in F.items() if isinstance(v, dict) and "D_verdict" in v}), flush=True)
    # ---------------- (3) hL_b: secant vs tangent ----------------
    T = {}
    for P, xl in zip(S.LEV, S.XL):
        mc = float(S.CV.m_of_level(xl)); slope = float(S.CV.slope(mc)); row = dict(tangent_dmc_dlneta=-1/slope)
        for lab, dln in [("eta_x_e(+1)", 1.0), ("eta_x_e^-1(-1)", -1.0), ("eta_x_e^0.5", 0.5), ("eta_x_e^-0.5", -0.5), ("hLb_x0.9", -np.log(0.9)), ("hLb_x1.1", -np.log(1.1))]:
            m2 = float(S.CV.m_of_level(xl - dln)); row[lab] = m2 - mc; row[lab + "_per_unit_lneta"] = (m2 - mc)/dln
        row["symmetric_secant_pm1"] = 0.5*(row["eta_x_e(+1)"] - row["eta_x_e^-1(-1)"])
        T[f"P{P:g}"] = row
    T["D124_table_row4"] = {"hLb_x_e^-1 (eta x e)": [-0.170, -0.270, -0.394], "hLb_x_e^+1 (eta / e)": [0.214, 0.367, ">0.39 (off grid)"], "hLb_x_e^-0.5": [-0.090, -0.144, -0.214], "hLb_x_e^+0.5": [0.100, 0.168, 0.262],
                            "definition": "D124 row (4): hL_b (eta_inf ~ 1/hL_b), mu_b fixed; 'per e-fold' = one-sided finite difference for hL_b x e^-1"}
    T["T17_D128"] = dict(dmc_dlneta=[-0.190, -0.311, -0.473], definition="D128 1.1: eta_inf -> f eta_inf exactly equivalent to I -> fI at fixed L_b (h changed); derivative at f=1")
    R["3_hLb_slope"] = T; print("hLb", json.dumps(T)[:1500], flush=True)
    # ---------------- (4) Bayes prior-mean on [150,200] ----------------
    PT = S.Pbar_true("A", True, S.SAS, ("normal", 0.10))
    in1 = (S.MG >= 160) & (S.MG <= 186)
    R["4_bayes_prior_150_200"] = dict(blind_prior_mean=(np.sum(PT[(S.MG >= 150) & (S.MG <= 200)])*H_ + 8.0)/50.0, prior_mean_if_restricted_to_160_186=float(np.sum(PT[in1])*H_/26.0),
        production_prior_mean=0.498279, production_B_selection_direct=1.45923,
        cause="C154_core_v2.py l.343-348: pi = indicator(lo<=mg<=hi) on the theory grid mg = [160,186] (Theory2 lo/hi), renormalised -> the 'flat_150_200' prior is effectively flat on [160,186]",
        corrected_B_selection_direct=0.727106/((np.sum(PT[(S.MG >= 150) & (S.MG <= 200)])*H_ + 8.0)/50.0))
    print("bayes150", R["4_bayes_prior_150_200"], flush=True)
    # ---------------- (5) like-for-like table ----------------
    BL = json.load(open(os.path.join(S.OUTD, "BLIND_RESULTS.json"), encoding="utf-8"))
    m1 = BL["1_marginalisation"]
    R["5_like_for_like"] = {
        "Pbar_direct": [m1["direct_MC"]["Pbar"], 0.727106], "CI68_direct": [m1["direct_MC"]["CI68"], [0.383764, 0.98936]], "CI95_direct": [m1["direct_MC"]["CI95"], [0.0545388, 0.999667]],
        "Pbar_direct+0.5": [m1["direct_MC+0.5"]["Pbar"], 0.67727], "Pbar_sigma_pole": [m1["sigma_pole"]["Pbar"], 0.859556],
        "Pbar_MSR": [m1["MSR_identified"]["Pbar"], 0.496826, "inputs differ: blind m_pole = 172.96 +- 0.368 (brief: 0.36+-0.25, Hoang fn.12); production 172.96 +- 0.568 (MC-MSR identification +-0.5), m-bar scheme"],
        "Pbar_mbar_formulation_direct": [m1["direct_MC"]["variants"]["mbar_0.07|model_A"], 0.726408],
        "Pbar_BKPV_FJ_direct": [m1["direct_MC"]["variants"]["BKPV_FJ_-0.36/+0.17|model_A"], 0.625839],
        "B_exist_direct_170_176": [BL["2_bayes"]["[170,176]|direct_MC"]["B_existence_as_data"], 0.727106], "B_cond_direct_170_176": [BL["2_bayes"]["[170,176]|direct_MC"]["B_conditioned_on_existence"], 1.47685],
        "B_cond_direct_150_200": [BL["2_bayes"]["[150,200]|direct_MC"]["B_conditioned_on_existence"], 1.45923, "production prior effectively [160,186]"],
        "mc_P0.1": [BL["3_exclusion"]["mc_P1_0p1"], "173.4 (+1sigma 173.8)"], "need_central_95_direct_defA": [BL["3_exclusion"]["required_central_value_95"]["direct_MC"]["defA_post_prob_mtrue_above_mc01_is_0p95"], 174.184],
        "D_verdict_now": [BL["3_future"]["scenarios"]["current_direct"]["D_verdict_sigma"], 0.7/0.462403],
        "D_verdict_FCCee": [BL["3_future"]["scenarios"]["FCC-ee_50MeV"]["D_verdict_sigma"], 0.7/0.131275, "alpha_s input 0.0001 (FSR 2025 Table 2) vs 0.0002 (Freitas 2019)"],
        "w0": [[0.983, 0.996], [0.980, 0.995]], "Pbar_w0_multiplicative_direct": [[BL["5_w0_effect_on_mc_C149"]["v2_all_min"]["Pbar_direct_scaled"], BL["5_w0_effect_on_mc_C149"]["v2_all_max"]["Pbar_direct_scaled"]], 0.720919]}
    code = [os.path.join(HERE, f) for f in ["diag.py", "stats.py", "fetch_one.py"]]
    R["code_sha256"] = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in code}
    json.dump(R, open(os.path.join(S.OUTD, "DIAGNOSIS.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("done", flush=True)

if __name__ == "__main__":
    main()
