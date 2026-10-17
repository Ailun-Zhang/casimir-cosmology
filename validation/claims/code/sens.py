# claims-blind (V3, R16): input-sensitivity spot checks.
# (a) hL_b x 0.9 / x 1.1 (pure prefactor of the conformal-time measure; exact: I -> I/f), plus an h-driven variant (mu_b x f too)
# (b) zero-mode fraction xi 0.2999 -> 0.28: (L_b, h) from the v11 turning condition L_b^4 = (1-xi) N_C gamma / (3 h^2), and Delta m_c
# Sources of P1(M_t): C149 run02 table (input data), keystone-blind base 2D_G node data, keystone-blind V2 'all' node data (independent).
# Coupling (RG) shifts from keystone-blind couplings.json nodes; gamma sensitivities to (lam, y_t, g2, gY) by finite differences
# with the keystone-blind 1D solvers at node 0 (read-only reuse of their code; nothing written to their directories).
import os, sys, json, glob, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
KB = ROOT + r"\07_reviews\R16_v12_round3\keystone_blind"
OUTD = ROOT + r"\07_reviews\R16_v12_round3\claims_robustness_blind\results"
sys.path.insert(0, KB + r"\work")
C149 = json.load(open(ROOT + r"\04_calculations\results\20261014_claude_v12_r3\holonomy_cross\run02\C149_run02_summary.json"))
CPL = json.load(open(KB + r"\results\couplings.json"))
HLB = 0.00437243
LEV = [0.9, 0.5, 0.1]

def hazard_integral(us, g, umax=5.0):
    lg = np.log(np.maximum(np.asarray(g, float), 1e-300)); f = CubicSpline(us, lg)
    val, _ = quad(lambda u: np.exp(f(u))/(HLB*np.sqrt(np.cosh(2*u))), 0.0, umax, limit=400, epsabs=0, epsrel=1e-11, points=list(us))
    return val

def crit_from(M, lnI, scale=1.0):
    """critical masses with I -> I*scale (scale = 1/f for hL_b -> f hL_b)."""
    sp = CubicSpline(M, lnI); fine = np.linspace(M[0], M[-1], 400001); v = sp(fine) + np.log(scale); out = {}
    for P in LEV:
        t = np.log(-np.log(P)); j = np.where(np.diff(np.sign(v - t)) != 0)[0]
        out[f"P{P:g}"] = float(fine[j[0]] - (v[j[0]] - t)*(fine[j[0]+1] - fine[j[0]])/(v[j[0]+1] - v[j[0]])) if len(j) else None
    return out

def delta(c0, c1): return {k: (c1[k] - c0[k]) if (c0[k] is not None and c1[k] is not None) else None for k in c0}

# ---------------- node data ----------------
ref = CPL["ref_172.6_0.1180"]; US = np.array([n["u"] for n in ref]); LMU = np.log([n["mu"] for n in ref])
def grid_couplings(als="0.1180"):
    keys = sorted([k for k in CPL["grid"] if k.startswith(als + "_")], key=lambda k: float(k.split("_")[1]))
    M = np.array([float(k.split("_")[1]) for k in keys])
    lam = np.array([[n["lam"] for n in CPL["grid"][k]] for k in keys]); yt = np.array([[n["yt"] for n in CPL["grid"][k]] for k in keys])
    return M, lam, yt
MG, LAM, YT = grid_couplings()
G2R = np.array([n["g2"] for n in ref]); GYR = np.array([n["gY"] for n in ref])

def rg_shift(arr_nodes, lnf):
    """value at ln mu_i + ln f minus value at ln mu_i, from a spline in ln mu over the 13 nodes."""
    o = np.argsort(LMU); sp = CubicSpline(LMU[o], np.asarray(arr_nodes)[o]); return sp(LMU + lnf) - sp(LMU)

def load_base_nodes():
    H = json.load(open(KB + r"\results\hazard_summary.json"))["models"]["2D"]
    rows = sorted([v for v in H.values() if abs(v["als"] - 0.1180) < 1e-9], key=lambda v: v["Mt"])
    return np.array([v["Mt"] for v in rows]), np.array([v["gammas"] for v in rows]), np.array([v["I"] for v in rows])

def pick_1d(d):
    if abs(d["Q_sieg"] - d["Q_cap"]) < 2e-4 and d["gamma_sieg"] > 0: return d["gamma_sieg"], d["Q_sieg"]
    return d["gamma_cap"], d["Q_cap"]

def load_v2_nodes(cfg="all", ratio=1.065):
    D = {}
    for f in glob.glob(KB + rf"\v2\hazard_shards_{cfg}\h_0.1180_*.json"):
        x = json.load(open(f)); D.setdefault(round(x["Mt"], 2), {})[x["inode"]] = x
    Ms, Gs = [], []
    for Mt in sorted(D):
        nd = D[Mt]
        if len(nd) < 13: continue
        g = []
        for i in range(13):
            gl, ql = pick_1d(nd[i]["Vlow"]); g1, q1 = pick_1d(nd[i]["V1D"]); d = nd[i]["2D"]
            ok = d["gamma"] > 1e-6 and gl <= d["gamma"] <= g1 and ql <= d["Q"] <= q1
            g.append(d["gamma"] if ok else gl*ratio)
        Ms.append(Mt); Gs.append(g)
    Ms = np.array(Ms); Gs = np.array(Gs); I = np.array([hazard_integral(US, g) for g in Gs])
    return Ms, Gs, I

# ---------------- gamma sensitivities at node 0 (keystone-blind 1D solvers) ----------------
def kappas(Mt_ref=172.9):
    import production as PR, model as MD
    k = f"0.1180_{Mt_ref:.2f}"; own = CPL["grid"][k][0]; g2, gY = ref[0]["g2"], ref[0]["gY"]
    f0 = os.path.join(KB, "work", "cache", "gauss_node00.npz")
    G = PR.GaussCached(g2, gY, "node00") if os.path.exists(f0) else MD.Gauss(g2, gY)
    def lng(Gx, lam, yt, g2l, gYl):
        r = PR.solve_case(Gx, lam, yt, g2l, gYl, True, False); return {m: float(np.log(r[m]["gamma"])) for m in ("V1D", "Vlow")}, r
    t0 = time.time(); base, rb = lng(G, own["lam"], own["yt"], g2, gY)
    dl, dy, dg = 2e-4, 2e-3, 2e-3
    lp, _ = lng(G, own["lam"] + dl, own["yt"], g2, gY); lm, _ = lng(G, own["lam"] - dl, own["yt"], g2, gY)
    yp, _ = lng(G, own["lam"], own["yt"] + dy, g2, gY); ym, _ = lng(G, own["lam"], own["yt"] - dy, g2, gY)
    Gg2 = MD.Gauss(g2 + dg, gY); g2p, _ = lng(Gg2, own["lam"], own["yt"], g2 + dg, gY)
    GgY = MD.Gauss(g2, gY + dg); gYp, _ = lng(GgY, own["lam"], own["yt"], g2, gY + dg)
    out = dict(node=0, Mt_ref=Mt_ref, lam=own["lam"], yt=own["yt"], g2=g2, gY=gY, base=base, runtime_s=time.time() - t0,
               Q_gamma_base={m: [rb[m]["Q"], rb[m]["gamma"]] for m in ("V1D", "Vlow")})
    for m in ("V1D", "Vlow"):
        dL = (lp[m] - lm[m])/(2*dl); dY = (yp[m] - ym[m])/(2*dy); dG2 = (g2p[m] - base[m])/dg; dGY = (gYp[m] - base[m])/dg
        out[m] = dict(dlng_dlam=dL, dlng_dyt=dY, dlng_dg2=dG2, dlng_dgY=dGY, kappa_y=dY/dL, kappa_g2=dG2/dL, kappa_gY=dGY/dL)
    return out

def rg_response(Ms, Gs, lnf_mu, kap, scale_h=1.0):
    """new hazard integrals when every node scale mu_i -> f mu_i (couplings shifted along their RG flow), and I -> I*scale_h."""
    # coupling derivatives along M_t at each node (own lam, yt), and RG shifts
    dlam_dM = np.gradient(LAM, MG, axis=0); dyt_dM = np.gradient(YT, MG, axis=0)
    dG2 = rg_shift(G2R, lnf_mu); dGY = rg_shift(GYR, lnf_mu)
    Inew = []
    for Mt, g in zip(Ms, Gs):
        j = int(np.argmin(np.abs(MG - Mt))); assert abs(MG[j] - Mt) < 1e-6
        dl = rg_shift(LAM[j], lnf_mu); dy = rg_shift(YT[j], lnf_mu)
        # d ln gamma_i / d M_t from the node data (spline over the M_t grid of the source)
        dlg = np.array([CubicSpline(Ms, np.log(Gs[:, i]))(Mt, 1) for i in range(13)])
        dlam_eff = dlam_dM[j] + kap["kappa_y"]*dyt_dM[j]                      # gauge couplings fixed (reference) along M_t
        dlng_dlam = dlg/dlam_eff
        dlng = dlng_dlam*(dl + kap["kappa_y"]*dy + kap["kappa_g2"]*dG2 + kap["kappa_gY"]*dGY)
        Inew.append(scale_h*hazard_integral(US, np.asarray(g)*np.exp(dlng)))
    return np.array(Inew), dict(dlam_u0_at_172p9=float(rg_shift(LAM[int(np.argmin(np.abs(MG - 172.9)))], lnf_mu)[0]),
                                 dyt_u0=float(rg_shift(YT[int(np.argmin(np.abs(MG - 172.9)))], lnf_mu)[0]), dg2_u0=float(dG2[0]), dgY_u0=float(dGY[0]))

def main():
    t0 = time.time(); R = dict(agent="claims-blind", task="V3 sensitivity (a) hL_b, (b) xi")
    # ---- turning condition (v11 Eq. cosh): L_b^4 = (1-xi) N_C gamma/(3 h^2), N_C = 25.515, gamma = 1.6750738 (v11 Sec. III)
    NC, GAM, XI0, XI1 = 25.515, 1.6750738, 0.2999, 0.28
    MPbar = 2.435e18; Lb0 = MPbar/CPL["mu_b"]; h0 = HLB/Lb0
    K0 = (1 - XI0)*NC*GAM/3; Lb_tc = (K0/h0**2)**0.25
    rL = ((1 - XI1)/(1 - XI0))**0.25
    R["turning_condition"] = dict(L_b_MP_input=Lb0, h_over_MP=h0, L_b_from_turning_condition=Lb_tc, consistency=Lb_tc/Lb0 - 1,
        xi0=XI0, xi1=XI1, ratio_Lb=rL, L_b_MP_new=Lb0*rL, hLb_new=HLB*rL, mu_b_new_GeV=CPL["mu_b"]/rL, dlnLb=float(np.log(rL)),
        h_change_bound=dict(rel=2.2e-4/np.log(2/0.95)*np.log(rL), note="v11 Sec. III C: V(phi_b), hL_b change by <=2.2e-4 for ln L0 range ln(2/0.95); the xi change acts like d ln L0 = -dlnLb -> |dh/h| <= this/2"))
    print("turning", R["turning_condition"], flush=True)
    # ---- sources
    src = {}
    m = np.array(sorted(float(k) for k in C149["P1_grid"])); P = np.array([C149["P1_grid"][f"{x:.1f}"] if f"{x:.1f}" in C149["P1_grid"] else None for x in m])
    src["C149_with_cross"] = dict(M=m, lnI=np.log(-np.log(P)))
    Mb, Gb, Ib = load_base_nodes(); src["KB_base_2D_G"] = dict(M=Mb, lnI=np.log(Ib), G=Gb)
    Mv, Gv, Iv = load_v2_nodes("all"); src["KB_v2_all_2D"] = dict(M=Mv, lnI=np.log(Iv), G=Gv)
    # monotone trimming (as keystone-blind collect)
    for k, s in src.items():
        while len(s["M"]) > 4 and s["lnI"][0] >= s["lnI"][1]:
            s["M"], s["lnI"] = s["M"][1:], s["lnI"][1:]
            if "G" in s: s["G"] = s["G"][1:]
    # ---- (a) prefactor
    A = {}
    for k, s in src.items():
        c0 = crit_from(s["M"], s["lnI"]); A[k] = dict(mc=c0, M_range=[float(s["M"][0]), float(s["M"][-1])])
        for f in [0.9, 1.1]:
            A[k][f"hLb_x{f}"] = delta(c0, crit_from(s["M"], s["lnI"], 1/f))
        A[k]["slope_dlnI_dM_at_mc"] = {kk: float(CubicSpline(s["M"], s["lnI"])(v, 1)) for kk, v in c0.items() if v}
    R["a_prefactor"] = A; print("a", json.dumps(A)[:1500], flush=True)
    # ---- gamma sensitivities
    kap = kappas(); R["kappas_node0"] = kap; print("kappas", kap, "%.0fs" % (time.time() - t0), flush=True)
    json.dump(R, open(os.path.join(OUTD, "sens_results.json"), "w"), indent=1, default=float)
    # ---- RG responses
    RG = {}
    for k in ["KB_base_2D_G", "KB_v2_all_2D"]:
        s = src[k]; c0 = crit_from(s["M"], s["lnI"]); RG[k] = {}
        for mdl in ["V1D", "Vlow"]:
            kp = kap[mdl]; kz = dict(kappa_y=0.0, kappa_g2=0.0, kappa_gY=0.0)
            for tag, lnf_mu, scale_h in [("xi_0.28_total", -np.log(rL), 1/rL), ("xi_0.28_mu_only", -np.log(rL), 1.0), ("xi_0.28_prefactor_only", 0.0, 1/rL),
                                         ("hLb_x0.9_h-driven(mu_b x0.9)", np.log(0.9), 1/0.9), ("hLb_x1.1_h-driven(mu_b x1.1)", np.log(1.1), 1/1.1),
                                         ("mu_b_x0.9_only", np.log(0.9), 1.0), ("mu_b_x1.1_only", np.log(1.1), 1.0)]:
                In, info = rg_response(s["M"], s["G"], lnf_mu, kp, scale_h)
                RG[k][f"{mdl}|{tag}"] = dict(dmc=delta(c0, crit_from(s["M"], np.log(In))), info=info)
                if mdl == "V1D":
                    In0, _ = rg_response(s["M"], s["G"], lnf_mu, kz, scale_h)
                    RG[k][f"lam_only|{tag}"] = dict(dmc=delta(c0, crit_from(s["M"], np.log(In0))))
        print(k, json.dumps(RG[k])[:2000], flush=True)
    R["rg_response"] = RG
    # C149 analogue for xi (prefactor exact + mu part from the KB V2 response, which is model-insensitive)
    R["b_xi_C149_prefactor_only"] = delta(crit_from(src["C149_with_cross"]["M"], src["C149_with_cross"]["lnI"]), crit_from(src["C149_with_cross"]["M"], src["C149_with_cross"]["lnI"], 1/rL))
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUTD, "sens_results.json"), "w"), indent=1, default=float)
    print("done %.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
