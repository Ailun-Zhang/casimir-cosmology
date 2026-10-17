# finalize.py -- ui-blind (V5): collect part 1/2 shards -> results/BLIND_RESULTS.json
import os, sys, json, glob, hashlib, datetime
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
RES = os.path.join(TOP, "results")
ROOT = os.path.abspath(os.path.join(TOP, "..", "..", ".."))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load(name, part="part1"):
    fn = os.path.join(RES, part, name + ".json")
    return json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else None


out = dict(task="V5 ui-blind (R16 v12 round 3)", agent="ui-blind",
           timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))

# ---------------- part 1 ----------------
P1 = {}
chk = load("checks")
P1["checks"] = chk
mmr = {}
for nm in ["mmr_star_172.76", "mmr_quad_172.76", "mmrB_star_172.16", "mmrB_star_173.36", "mmrB_star_171.5"]:
    d = load(nm)
    if d: mmr[nm] = dict(xi_EW=d["xi_EW"], xi_mu_b=d["xi_mu_b"], N_peak=d["info"]["N_peak"], H_peak=d["info"]["H_peak"],
                         B_peak=d["info"]["B_peak"], xi_mustar=d["peak"]["xi_mustar"], lam_mustar=d["peak"]["lam_mustar"],
                         mu_star=d["peak"]["mu_star"])
mmr["literature"] = dict(source="MMR 2021 arXiv:2011.03763v3, Fig. 4 caption p.14 and eqs. (4.1)-(4.3) p.12",
                         xi_EW_star_172p76=0.05938, xi_EW_quad_172p76=0.05998, xi_EW_quart_172p76=0.05875,
                         star_band_mt_pm2sigma="0.059 +0.007 -0.009 (m_t = 172.76 +- 0.60)",
                         H_star_peak=9.96e12, H_quad_peak=1.83e13,
                         inputs="m_h=125.10, m_t=172.76, alpha_s=0.1179, Table-1 couplings at mu_EW; N=60, P_zeta=2.1e-9")
diag = {}
base_mmr = load("mmr_star_172.76")
for f in sorted(glob.glob(os.path.join(RES, "part1", "mmrD_star_*.json"))):
    d = json.load(open(f, encoding="utf-8")); tag = os.path.basename(f)[10:-5]
    if tag == "muEWMZ":
        continue   # invalid: reference scale below the tabulated range (clipped to M_t)
    diag[tag] = dict(xi_EW=d["xi_EW"], shift_vs_base=d["xi_EW"] - base_mmr["xi_EW"], xi_mu_b=d["xi_mu_b"])
mmr["residual_diagnostics"] = diag
P1["MMR_reproduction"] = mmr

v11 = {}
for Mt in ("172.2", "172.6", "173.0"):
    row = {}
    for var in ("base", "xi3", "rge2", "notorus", "zeta0", "noalpha", "muEW246", "nodes60", "L0_2dLSS"):
        d = load(f"v11_{Mt}_{var}")
        if d:
            row[var] = dict(xi_EW=d["xi_EW"], xi_mu_b=d["xi_mu_b"], lnN_at_root=d["lnN_at_root"],
                            N_before_end_peak=d["info"]["N_before_end_peak"], H_peak_GeV=d["info"]["H_peak_GeV"],
                            B_peak=d["info"]["B_peak"], frac_first10efolds_after_turn=d["info"]["frac_first10"],
                            xi_mustar=d["peak"]["xi_mustar"], lam_mustar=d["peak"]["lam_mustar"],
                            mu_star=d["peak"]["mu_star"], hbar_over_H=d["peak"]["h_bar"] / d["info"]["H_peak_GeV"],
                            lam_mu_b=d["lam_mu_b"], K_mu_b=d["K_mu_b"], Npost_raw=d.get("Npost_raw"),
                            frac_ball_cap_torus_end=d.get("frac_end"))
    v11[Mt] = row
P1["v11"] = v11
mts = np.array([172.2, 172.6, 173.0])
xe = np.array([v11[m]["base"]["xi_EW"] for m in ("172.2", "172.6", "173.0")])
xb = np.array([v11[m]["base"]["xi_mu_b"] for m in ("172.2", "172.6", "173.0")])
P1["xi_inf_summary"] = dict(
    definition="<N> = 1 (past light cone, finite T^3 with L0=0.95 d_LSS), Hawking-Moss rate (R/12)^2 exp(-384 pi^2 DV/R^2), "
               "RGI V_eff of Markkanen et al. 2018 with mu=mu*(h,R), 3-loop SM RGE (Buttazzo App. B), 1-loop beta_xi (MMR eq. 2.15), zeta=1",
    M_t=mts.tolist(), xi_EW_inf=xe.tolist(), xi_mu_b_inf=xb.tolist(),
    dxiEW_dMt_per_GeV=[(xe[1] - xe[0]) / 0.4, (xe[2] - xe[1]) / 0.4],
    dximub_dMt_per_GeV=[(xb[1] - xb[0]) / 0.4, (xb[2] - xb[1]) / 0.4],
    systematics_172p6=dict((k, v11["172.6"][k]["xi_EW"] - v11["172.6"]["base"]["xi_EW"]) for k in v11["172.6"] if k != "base"),
    systematics_172p6_mu_b=dict((k, v11["172.6"][k]["xi_mu_b"] - v11["172.6"]["base"]["xi_mu_b"]) for k in v11["172.6"] if k != "base"),
    MMR_offset_mine_minus_lit=mmr["mmr_star_172.76"]["xi_EW"] - 0.05938,
    xi_EW_inf_MMRcalibrated=(xe - (mmr["mmr_star_172.76"]["xi_EW"] - 0.05938)).tolist())
dec = load("v11geom_mmrcoup_172.76")
if dec:
    P1["decomposition_172.76_MMRcouplings"] = dict(v11_geometry=dec["xi_EW"], MMR_geometry=mmr["mmr_star_172.76"]["xi_EW"],
                                                   geometry_shift=dec["xi_EW"] - mmr["mmr_star_172.76"]["xi_EW"])
wr = os.path.join(RES, "wolfram_receipts.json")
if os.path.exists(wr):
    P1["wolfram_check"] = dict(result=json.load(open(wr, encoding="utf-8-sig")).get("result"),
                               note="B_HM at (v11, M_t=172.6, xi_EW root, R_peak) from the same coupling table; attempt 1 failed (absolute AccuracyGoal), kept in receipt")
out["part1"] = P1

# ---------------- part 2 ----------------
P2 = {}
L = {}
for f in sorted(glob.glob(os.path.join(RES, "part2", "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    L[d["name"]] = d
if L:
    def lc(n, m="2D"):
        return L[n][m]["lambda_c"] if n in L else None
    rows = {n: dict(eta_inf=L[n]["case"]["eta"], g2=L[n]["case"]["g2"], yt=L[n]["case"]["yt"],
                    lambda_c_2D=lc(n), lambda_c_1DBO=lc(n, "1D_BO"),
                    dlngamma_dlam_2D=L[n]["2D"]["dlngamma_dlam"], dlngamma_dlam_1D=L[n]["1D_BO"]["dlngamma_dlam"],
                    bracket_ok_last=L[n]["2D"]["iterates"][-1].get("bracket_ok"),
                    flags=sorted(set(it.get("flag", "") for it in L[n]["2D"]["iterates"] + L[n]["1D_BO"]["iterates"])))
            for n in L}
    P2["cases"] = rows
    der = {}
    for m in ("2D", "1D_BO"):
        D = {}
        if all(k in L for k in ("base", "hLb_x0.5", "hLb_x2")):
            a, b, c0 = lc("hLb_x0.5", m), lc("hLb_x2", m), lc("base", m)
            D["dlamc_dlneta_central"] = (a - b) / (2 * np.log(2))
            D["dlamc_dlneta_up(eta x2)"] = (a - c0) / np.log(2)
            D["dlamc_dlneta_down(eta x0.5)"] = (c0 - b) / np.log(2)
            sl = L["base"]["2D" if m == "2D" else "1D_BO"]["dlngamma_dlam"]
            D["dlamc_dlneta_local(-1/dlngamma/dlam)"] = -1.0 / sl
        if all(k in L for k in ("g2_x1.1", "g2_x0.9")):
            D["dlamc_dlng2"] = (lc("g2_x1.1", m) - lc("g2_x0.9", m)) / np.log(1.1 / 0.9)
            D["dlamc_dg2"] = D["dlamc_dlng2"] / 0.5268
            D["g2_asym(up-down)"] = (lc("g2_x1.1", m) - lc("base", m)) - (lc("base", m) - lc("g2_x0.9", m))
        if all(k in L for k in ("yt_x1.1", "yt_x0.9")):
            D["dlamc_dlnyt"] = (lc("yt_x1.1", m) - lc("yt_x0.9", m)) / np.log(1.1 / 0.9)
            D["dlamc_dyt"] = D["dlamc_dlnyt"] / 0.4585
            D["yt_asym(up-down)"] = (lc("yt_x1.1", m) - lc("base", m)) - (lc("base", m) - lc("yt_x0.9", m))
        der[m] = D
    P2["derivatives"] = der
    P2["conversion_note"] = ("delta m_c ~ -delta lambda_c / (d lambda(mu_b)/dM_t), with d lambda(mu_b)/dM_t = -0.00656/GeV "
                             "(keystone-blind; own 3-loop run gives similar), i.e. dm_c/dln eta ~ -dlamc_dlneta/0.00656 GeV")
    if "dlamc_dlneta_central" in der.get("2D", {}):
        P2["dmc_dlneta_GeV_2D"] = -der["2D"]["dlamc_dlneta_central"] / 0.00656
    P2["model"] = ("keystone-blind 2D_G static model (round-2: no cross term, two-loop map), v11 couplings g2=0.5268, gY=0.4415, "
                   "yt=0.4585 at 1/L_b, P1 = exp(-gamma eta_inf), eta_inf = 1.31103/(hL_b), hL_b = 0.00437243")
out["part2"] = P2

# ---------------- provenance ----------------
code = [os.path.join(HERE, f) for f in ("smhiggs.py", "inflation_bound.py", "run_part1.py", "summarize_part1.py",
                                        "part2_lamc.py", "finalize.py", "extract_pdf.py", "export_for_wolfram.py", "check_B.wl")]
code += sorted(glob.glob(os.path.join(HERE, "kb_copy", "*.py")))
out["code_sha256"] = {os.path.relpath(p, TOP): sha(p) for p in code if os.path.exists(p)}
lit = os.path.join(ROOT, "02_literature", "batches", "20261014_claude_v12_r3_L37", "universality_inflation_blind")
out["input_sha256"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(lit, "*.pdf")))}
v11tex = os.path.join(ROOT, "06_paper_draft", "versions", "v11", "main_v11.tex")
if os.path.exists(v11tex):
    out["input_sha256"]["main_v11.tex"] = sha(v11tex)
bt = os.path.join(ROOT, "07_reviews", "R16_v12_round3", "keystone_blind", "work", "buttazzo_text.txt")
if os.path.exists(bt):
    out["input_sha256"]["keystone_blind/work/buttazzo_text.txt (Buttazzo 2013 App. B source)"] = sha(bt)
json.dump(out, open(os.path.join(RES, "BLIND_RESULTS.json"), "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(dict(xi=P1["xi_inf_summary"], part2=P2.get("derivatives"), cases=P2.get("cases")), indent=1, default=float))
