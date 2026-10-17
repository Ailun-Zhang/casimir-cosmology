# keystone-blind (T3): assemble results/BLIND_RESULTS.json (numbers + numerical errors + code SHA-256 + timestamp)
import os, sys, json, glob, hashlib, datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
J = lambda f: json.load(open(os.path.join(RES, f))) if os.path.exists(os.path.join(RES, f)) else None

def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()

def fp_odd_check():
    """analytic continuation of F_P to m^2<0: F(m2)-F(-m2) = 1/2 sum'[sqrt(k2+m2)-sqrt(k2-m2)-m2/k] + m2 Z(1/2)/(4 pi)."""
    import oneloop
    out = {}
    x = np.arange(-40, 41); X, Y, Z = np.meshgrid(x, x, x, indexing="ij"); n2 = (X**2 + Y**2 + Z**2).ravel(); n2 = n2[n2 > 0]
    k2 = 4*np.pi**2*n2.astype(float); k = np.sqrt(k2)
    for m2 in [2.0, 5.0, 12.0, 20.0]:
        s = 0.5*np.sum(np.sqrt(k2 + m2) - np.sqrt(k2 - m2) - m2/k)
        tail = 0.5*(m2**3/8)*4*np.pi*(2*np.pi)**-5/(2*40.0**2)       # leading m^6/(8k^5) tail beyond |n|=40
        rhs = s + tail + m2*(-2.8372974794806)/(4*np.pi)
        lhs = oneloop.FP(np.array([m2]))[0] - oneloop.FP(np.array([-m2]))[0]
        out[str(m2)] = dict(lhs=float(lhs), rhs=float(rhs), diff=float(lhs - rhs))
    return out

def main():
    C = J("couplings.json"); S = J("static_raw.json"); H = J("hazard_summary.json")
    Sx = J("static_ex_raw.json"); Hx = J("hazard_summary_ex.json"); dE = J("dE_fit.json")
    ref = C["ref_172.6_0.1180"][0]
    R = dict(agent="keystone-blind", task="T3 R16 keystone blind", timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z",
             units="L=1 (energies, widths in 1/L); gamma = -2 Im E (full width); lambda MS-bar (V=lambda|Phi|^4); mu=1/L",
             conventions=dict(hLb=0.00437243, mu_b_GeV=C["mu_b"], eta_tot=S["eta_tot"], P_static="exp(-gamma*eta_tot)",
                              hazard="P1=exp(-int_0^5 gamma(u) du/(hLb sqrt(cosh 2u))), gamma(u) at mu_b/sqrt(cosh 2u); gauge-profile and one-loop g2,gY from (172.6,0.1180); lambda,yt from each (Mt,alpha_s); M_h=125.13, M_W=80.3692"))
    R["couplings"] = dict(Delta_lambda=C["Delta_lambda"], calibration=C["calibration"],
                          at_mu_b_172p6_0p1180=dict(lam=ref["lam"], yt=ref["yt"], g2=ref["g2"], gY=ref["gY"], g3=ref["g3"]),
                          lam_mub_2loop_without_Delta=C["lam2loop_mub_nodelta_172.6"],
                          lam_mub={k: C["grid"][f"0.1180_{k}"][0]["lam"] for k in ["172.20", "172.60", "173.00"]},
                          yt_mub={k: C["grid"][f"0.1180_{k}"][0]["yt"] for k in ["172.20", "172.60", "173.00"]},
                          wolfram_crosscheck="Delta_lambda and couplings at mu_b agree to 3e-15 (results/wolfram_receipts.json)")
    loc = S["lambda0_local"]
    st = dict(couplings_v11=dict(g2=0.5268, gY=0.4415, yt=0.4585),
              lambda0_local=dict(E0_2D_G=loc["2D"]["E"], err_2D=loc["2D"]["err"], E0_1D_BO=loc["V1D"]["E"], err_1D=loc["V1D"]["err"],
                                 E0_BO_lower=loc["Vlow"]["E"], DMC_anchor_v11_exact_local="~4.449 (from v11 xi=0.2993 at lambda=0)"))
    for lam in ["-0.008", "-0.009", "-0.01", "-0.011", "-0.012", "-0.0143"]:
        r = S[f"lam_{lam}"]
        e = dict(Q_2D=r["2D"]["Q"], gamma_2D=r["2D"]["gamma"], Q_1D=r["V1D"]["Q"], gamma_1D=r["V1D"]["gamma"],
                 Q_BOlow=r["Vlow"]["Q"], gamma_BOlow=r["Vlow"]["gamma"], bracket_ok=r["2D"].get("bracket_ok"),
                 gamma_1D_cap=r["V1D"]["gamma_cap"], gamma_1D_siegert=r["V1D"]["gamma_sieg"])
        if "2D_dr0.014" in r:
            e["2D_dr0.014"] = r["2D_dr0.014"]; e["2D_Ns32_Om0.25"] = r["2D_Ns32"]
        st[f"lambda_{lam}"] = e
    st["lambda_c_static"] = S["lambda_c"]; st["gamma_targets"] = S["gamma_targets"]
    st["V1_const_r0_subtracted"] = -10.723891084257204
    R["static"] = st
    def hz(Hs):
        out = {}
        for k, v in Hs["critical"].items():
            out[k] = {kk: (dict(mc=vv["mc"], lam_mub=vv["lam_mub"], d_quad_u=vv["d_quad_u"], d_nodes=vv["d_nodes"], d_Mt_linear=vv["d_Mt_interp"], d_fallback=vv.get("d_fallback"))
                           if isinstance(vv, dict) else vv) for kk, vv in v.items()}
        tabs = {}
        for mdl in ["2D", "V1D", "Vlow"]:
            tabs[mdl] = {k: dict(P1=v["P1"], I=v["I"], gamma_u0=v["gamma_u0"], Q_u0=v["Q_u0"], lam_mub=v["lam_mub"], n_fallback=v.get("n_fallback")) for k, v in Hs["models"][mdl].items()}
        dm = {}
        for mdl in ["2D", "V1D", "Vlow"]:
            try:
                a, b = out[f"{mdl}_0.1171"], out[f"{mdl}_0.1189"]
                dm[mdl] = {P: (b[P]["mc"] - a[P]["mc"])/(0.1189 - 0.1171) for P in ["P0.9", "P0.5", "P0.1"]}
            except KeyError:
                pass
        return dict(critical=out, P1_tables=tabs, dmc_dalphas_GeV_per_unit=dm, ratio_2D_over_low=Hs.get("ratio_2D_over_low"), flags=Hs.get("flags"))
    R["hazard_2D_G_and_1D"] = hz(H)
    dEa = J("dE_fit_avg.json"); Sxa = J("static_ex_raw_avg.json"); Hxa = J("hazard_summary_ex_avg.json")
    def slim(S):
        if not S: return None
        o = dict(lambda0_local_2Dex=S.get("lambda0_local_2Dex"), lambda0_local_1Dex=S.get("lambda0_local_1Dex"), lambda_c=S.get("lambda_c"))
        for k in ["lam_-0.008", "lam_-0.01", "lam_-0.0143"]:
            if k in S: o[k] = {m: dict(Q=S[k][m]["Q"], gamma=S[k][m]["gamma"]) for m in ["2D", "V1D", "Vlow"] if m in S[k]}
        return o
    R["nonGaussian_dE"] = dict(nominal_tau_average=dEa, variant_tau_extrapolated=dict(r=dE["r"], dE=dE["dE"], dE_err=dE["dE_err"]) if dE else None,
                               dmc_pure_SU2=dE["pure"] if dE else None, eps2_g23=dE["eps2_g23"] if dE else None)
    R["static_2Dex_nominal"] = slim(Sxa)
    R["static_2Dex_nominal_scale0.8"] = slim(J("static_ex_raw_avg_scale0.8.json"))
    R["static_2Dex_nominal_scale1.2"] = slim(J("static_ex_raw_avg_scale1.2.json"))
    R["static_2Dex_variant_tauextrap"] = slim(Sx)
    R["static_2Dex_variant_scale0.85"] = slim(J("static_ex_raw_scale0.85.json"))
    R["static_2Dex_variant_scale1.15"] = slim(J("static_ex_raw_scale1.15.json"))
    if Hxa: R["hazard_2Dex_nominal"] = hz(Hxa)
    if Hx: R["hazard_2Dex_variant_tauextrap"] = hz(Hx)
    R["checks"] = dict(oneloop_selftest=J("oneloop_selftest.json"), FP_negative_m2_odd_part=fp_odd_check(),
                       validate_1D="results/validate_res_m0.010.log", validate_2D="results/validate_2d_m0.010.log")
    files = sorted(glob.glob(os.path.join(HERE, "*.py")))
    R["code_sha256"] = {os.path.basename(f): sha(f) for f in files}
    R["input_sha256"] = {"Buttazzo2013_1307.3536v4.pdf": sha(os.path.join(HERE, "..", "inputs", "lit", "Buttazzo2013_1307.3536v4.pdf"))}
    json.dump(R, open(os.path.join(RES, "BLIND_RESULTS.json"), "w"), indent=1, default=float)
    print(json.dumps(R["checks"]["FP_negative_m2_odd_part"], indent=0))
    print("written BLIND_RESULTS.json")

if __name__ == "__main__":
    main()
