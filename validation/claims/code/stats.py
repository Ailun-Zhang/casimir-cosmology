# claims-blind (V3, R16): blind statistics layer on top of the C149 run02 P1 table.
# marginalisation over m_t (three definitions), alpha_s, SM-mapping theory error, model error;
# Bayes factors (two conventions, two priors); exclusion thresholds; future discriminating power.
# Two independent integrators: deterministic grid convolution (main) and Monte Carlo (cross-check, quantiles).
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from scipy.optimize import brentq
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
C149F = ROOT + r"\04_calculations\results\20261014_claude_v12_r3\holonomy_cross\run02\C149_run02_summary.json"
OUTD = ROOT + r"\07_reviews\R16_v12_round3\claims_robustness_blind\results"
C = json.load(open(C149F))
LEV = np.array([0.9, 0.5, 0.1]); XL = np.log(-np.log(LEV))
AS0, SAS = 0.1180, 0.0009
MODEL_SP = np.array([0.040, 0.073, 0.118]); MODEL_SM = np.array([0.054, 0.085, 0.129])   # +sigma / -sigma of m_c (GeV)

# ---------------- base curve ln H(m), H = -ln P1, alpha_s = 0.1180 ----------------
class Curve:
    def __init__(self, include_crit=True):
        pts = {float(k): v for k, v in C["P1_grid"].items()}
        pts.update({float(k): v for k, v in C["P1"].items()})
        if include_crit:
            for lev in LEV: pts[float(C["crit"]["0.1180"]["with_cross"][f"P1={lev:g}"])] = float(lev)
        self.m = np.array(sorted(pts)); P = np.array([pts[k] for k in self.m]); self.y = np.log(-np.log(P))
        assert np.all(np.diff(self.y) > 0)
        self.sp = CubicSpline(self.m, self.y)
        self.m0, self.m1 = self.m[0], self.m[-1]
        self.y0, self.y1 = float(self.sp(self.m0)), float(self.sp(self.m1))
        self.d0, self.d1 = float(self.sp(self.m0, 1)), float(self.sp(self.m1, 1))
        mt = np.linspace(150, 200, 50001); self._xt = self.lnH(mt); self._st = self.slope(mt); self._mt = mt
    def lnH(self, x):
        x = np.asarray(x, float); xc = np.clip(x, self.m0, self.m1); out = self.sp(xc)
        out = np.where(x < self.m0, self.y0 + self.d0*(x - self.m0), out)
        return np.where(x > self.m1, self.y1 + self.d1*(x - self.m1), out)
    def slope(self, x):
        x = np.asarray(x, float)
        return np.where(x < self.m0, self.d0, np.where(x > self.m1, self.d1, self.sp(np.clip(x, self.m0, self.m1), 1)))
    def slope_of_level(self, xlev): return np.interp(xlev, self._xt, self._st)
    def m_of_level(self, xlev): return np.interp(xlev, self._xt, self._mt)

CV = Curve(True); CV_nc = Curve(False)
mc_tab = lambda als, lev: float(C["crit"][als]["with_cross"][f"P1={lev:g}"])
S_LEV = np.array([(mc_tab("0.1189", l) - mc_tab("0.1171", l))/0.0018 for l in LEV])            # GeV per unit alpha_s
C_LEV = np.array([(mc_tab("0.1189", l) + mc_tab("0.1171", l) - 2*mc_tab("0.1180", l))/0.0009**2 for l in LEV])
SBAR = float(np.interp(XL[1], XL, S_LEV))
SLOPE_LEV = np.array([float(CV.slope(mc_tab("0.1180", l))) for l in LEV])
ELL_P, ELL_M = MODEL_SP*SLOPE_LEV, MODEL_SM*SLOPE_LEV          # multiplicative-equivalent ln f of the model error

def s_of(x): return np.interp(x, XL, S_LEV)
def c_of(x): return np.interp(x, XL, C_LEV)
def sig_model(x, shape, sp=MODEL_SP, sm=MODEL_SM, curve=CV):
    if shape == "A":      # multiplicative-equivalent ln f interpolated in level, held beyond the ends
        lp, lm = np.interp(x, XL, sp*SLOPE_LEV), np.interp(x, XL, sm*SLOPE_LEV); s = curve.slope_of_level(x)
        return lp/s, lm/s
    return np.interp(x, XL, sp), np.interp(x, XL, sm)   # shape B: shift interpolated, held beyond the ends

def P1_eval(m, dal, dsm, z, shape="A", model_on=True, levdep=True, curve=CV, niter=5):
    """P1 for true pole mass m, alpha_s offset dal, SM-mapping shift dsm of m_c, model nuisance z (split-normal)."""
    m = np.asarray(m, float)
    x = curve.lnH(m - SBAR*dal - dsm)
    for _ in range(niter):
        if levdep: sh = s_of(x)*dal + 0.5*c_of(x)*dal**2 + dsm
        else: sh = SBAR*dal + dsm
        if model_on:
            sp_, sm_ = sig_model(x, shape, curve=curve); sh = sh + np.where(z > 0, sp_, sm_)*z
        x = curve.lnH(m - sh)
    return np.exp(-np.exp(x))

# ---------------- deterministic grid machinery ----------------
H_ = 0.002
MG = np.arange(158.0, 188.0 + H_/2, H_)
ZG = np.arange(-8, 8 + 1e-9, 0.02); WZ = norm.pdf(ZG)*0.02; WZ /= WZ.sum()

def G_of_m(shape="A", model_on=True, curve=CV):
    """E_z[P1(m; z)] on MG (alpha_s/SM shifts excluded)."""
    if not model_on: return P1_eval(MG, 0.0, 0.0, 0.0, model_on=False, curve=curve)
    out = np.zeros_like(MG)
    for z, w in zip(ZG, WZ): out += w*P1_eval(MG, 0.0, 0.0, np.full(MG.shape, z), shape=shape, curve=curve)
    return out

def shift_density(sig_as, sm_spec):
    """density of A = s*dalpha + delta_SM on a symmetric grid (odd length) with spacing H_."""
    ag = H_*np.arange(-2000, 2001)
    ga = norm.pdf(ag, 0, max(SBAR*sig_as, 1e-9))
    if sm_spec[0] == "normal":
        s = sm_spec[1]
        if s <= 0: p = ga
        else: p = norm.pdf(ag, 0, np.hypot(SBAR*sig_as, s))
    else:
        _, smn, spl = sm_spec
        d = np.where(ag < 0, norm.pdf(ag/smn)/smn, norm.pdf(ag/spl)/spl)        # each half integrates to 1/2
        p = np.convolve(ga*H_, d, mode="same")
    p = p/(p.sum()*H_)
    return ag, p

def conv_shift(F, p, left=1.0, right=0.0):
    """(F*p)(m) = int F(m - a) p(a) da on MG, with F padded by its asymptotic values."""
    n = len(p)//2
    Fp = np.concatenate([np.full(n, left), F, np.full(n, right)])
    return np.convolve(Fp, p*H_, mode="valid")

def gauss_kernel(sig):
    n = int(np.ceil(8*sig/H_)) if sig > 0 else 0
    ag = H_*np.arange(-n, n + 1)
    k = norm.pdf(ag, 0, sig) if sig > 0 else np.array([1.0/H_]); return k/(k.sum()*H_)

def Pbar_true(shape="A", model_on=True, sig_as=SAS, sm_spec=("normal", 0.10), curve=CV, _cache={}):
    key = (shape, model_on, curve is CV)
    if key not in _cache: _cache[key] = G_of_m(shape, model_on, curve)
    ag, p = shift_density(sig_as, sm_spec)
    return conv_shift(_cache[key], p)

def interp_root(xg, fg, target):
    """first crossing of decreasing fg through target"""
    i = np.where((fg[:-1] - target)*(fg[1:] - target) <= 0)[0]
    if len(i) == 0: return None
    i = i[0]; return float(xg[i] + (target - fg[i])*(xg[i+1] - xg[i])/(fg[i+1] - fg[i]))

def Pbar_meas(PT, mu, sig):
    return float(np.sum(PT*norm.pdf(MG, mu, sig))*H_)

def Pbar_meas_curve(PT, sig):
    return conv_shift(PT, gauss_kernel(sig))

# ---------------- definitions ----------------
DEFS = {
    "direct_MC":        dict(mu=172.60, sig=0.27, note="m_MC=172.60+-0.27 interpreted as pole"),
    "direct_MC+0.5":    dict(mu=172.60, sig=float(np.hypot(0.27, 0.5)), note="plus MC-pole interpretation 0.5 GeV in quadrature"),
    "sigma_pole":       dict(mu=172.10, sig=0.60, note="cross-section pole mass 172.1+-0.6"),
    "MSR_identified":   dict(mu=172.60 + 0.36, sig=float(np.hypot(0.27, 0.25)), note="m_MC = m_MSR(1.3 GeV); m_pole - m_MSR = 0.36+-0.25 (Hoang 2020)"),
}
SM_VARIANTS = {"pole_0.10": ("normal", 0.10), "mbar_0.07": ("normal", 0.07), "brief_0.086": ("normal", 0.086),
               "BKPV_FJ_-0.36/+0.17": ("split", 0.36, 0.17), "BKPV_FJ_mirror_-0.17/+0.36": ("split", 0.17, 0.36), "none": ("normal", 0.0)}

def mc_run(mu, sig, sm_spec, shape="A", model_on=True, levdep=True, sig_as=SAS, N=2_000_000, seed=12345, trunc=None, curve=CV):
    rng = np.random.default_rng(seed); out = []; kept = 0
    for ch in range(N//500_000):
        n = 500_000
        m = rng.normal(mu, sig, n); dal = rng.normal(0, sig_as, n)
        if sm_spec[0] == "normal": dsm = rng.normal(0, sm_spec[1], n) if sm_spec[1] > 0 else np.zeros(n)
        else:
            t = rng.normal(0, 1, n); dsm = np.where(t < 0, sm_spec[1]*t, sm_spec[2]*t)
        z = rng.normal(0, 1, n)
        if trunc is not None:
            k = (m > trunc[0]) & (m < trunc[1]); m, dal, dsm, z = m[k], dal[k], dsm[k], z[k]
        out.append(P1_eval(m, dal, dsm, z, shape=shape, model_on=model_on, levdep=levdep, curve=curve))
    P = np.concatenate(out); q = np.quantile(P, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(mean=float(P.mean()), mc_se=float(P.std()/np.sqrt(len(P))), median=float(q[2]),
                CI68=[float(q[1]), float(q[3])], CI95=[float(q[0]), float(q[4])], N=int(len(P)),
                prob_P1_gt_0p5=float(np.mean(P > 0.5)), prob_P1_lt_0p1=float(np.mean(P < 0.1)))

def main():
    t0 = time.time(); R = dict(agent="claims-blind", inputs=dict(C149=C149F, alpha_s=[AS0, SAS], M_h=125.13,
            model_error=dict(levels=LEV.tolist(), plus=MODEL_SP.tolist(), minus=MODEL_SM.tolist()), SM_variants={k: list(v) for k, v in SM_VARIANTS.items()},
            defs=DEFS))
    # ---- curve diagnostics ----
    diag = dict(knots_m=CV.m.tolist(), end_slopes=[CV.d0, CV.d1], s_lev=S_LEV.tolist(), c_lev=C_LEV.tolist(), SBAR=SBAR,
                slope_at_mc=SLOPE_LEV.tolist(), ell_plus=ELL_P.tolist(), ell_minus=ELL_M.tolist())
    diag["mc_from_curve_without_crit_knots"] = {f"{l:g}": brentq(lambda x: float(CV_nc.lnH(x)) - xl, 165, 180) for l, xl in zip(LEV, XL)}
    diag["mc_table"] = {f"{l:g}": mc_tab("0.1180", l) for l in LEV}
    diag["P1_curve_check"] = {m: float(np.exp(-np.exp(CV.lnH(m)))) for m in [171.8, 172.0, 172.3, 172.5, 172.7, 172.9, 173.1, 173.5, 174.0]}
    R["curve"] = diag; print("curve", json.dumps(diag)[:900], flush=True)

    # ---- 1. marginalisation ----
    M1 = {}
    for dn, d in DEFS.items():
        M1[dn] = {}
        for sn, sm in SM_VARIANTS.items():
            for shape in (["A", "B"] if sn == "pole_0.10" else ["A"]):
                PT = Pbar_true(shape, True, SAS, sm)
                M1[dn][f"{sn}|model_{shape}"] = dict(Pbar_grid=Pbar_meas(PT, d["mu"], d["sig"]))
            PT0 = Pbar_true("A", False, SAS, sm)
            M1[dn][f"{sn}|no_model"] = dict(Pbar_grid=Pbar_meas(PT0, d["mu"], d["sig"]))
        # decomposition
        M1[dn]["only_mt"] = dict(Pbar_grid=Pbar_meas(Pbar_true("A", False, 0.0, ("normal", 0.0)), d["mu"], d["sig"]))
        M1[dn]["mt+alpha_s"] = dict(Pbar_grid=Pbar_meas(Pbar_true("A", False, SAS, ("normal", 0.0)), d["mu"], d["sig"]))
        # MC (exact level-dependent alpha_s shift + model) for the main and the asymmetric variants
        for key, sm in [("pole_0.10|model_A", SM_VARIANTS["pole_0.10"]), ("mbar_0.07|model_A", SM_VARIANTS["mbar_0.07"]),
                        ("BKPV_FJ_-0.36/+0.17|model_A", SM_VARIANTS["BKPV_FJ_-0.36/+0.17"]), ("brief_0.086|model_A", SM_VARIANTS["brief_0.086"])]:
            M1[dn][key]["MC"] = mc_run(d["mu"], d["sig"], sm)
        M1[dn]["pole_0.10|model_A"]["MC_levindep"] = mc_run(d["mu"], d["sig"], SM_VARIANTS["pole_0.10"], levdep=False, seed=777)
        M1[dn]["pole_0.10|model_B"]["MC"] = mc_run(d["mu"], d["sig"], SM_VARIANTS["pole_0.10"], shape="B", seed=999)
        M1[dn]["pole_0.10|model_A"]["Pbar_grid_curve_without_crit_knots"] = Pbar_meas(Pbar_true("A", True, SAS, ("normal", 0.10), curve=CV_nc), d["mu"], d["sig"])
        print(dn, {k: round(v["Pbar_grid"], 4) for k, v in M1[dn].items()}, "MC main", M1[dn]["pole_0.10|model_A"]["MC"], "%.0fs" % (time.time() - t0), flush=True)
    R["marginalisation"] = M1

    # ---- 2. Bayes factors ----
    PT = Pbar_true("A", True, SAS, ("normal", 0.10))
    assert abs(PT[0] - 1) < 1e-9 and abs(PT[-1]) < 1e-9
    BF = {}
    for (a, b) in [(170.0, 176.0), (150.0, 200.0)]:
        inr = (MG >= a) & (MG <= b)
        lo_ext = max(0.0, min(MG[0], b) - a)          # region a..158 has P1 = 1
        prior_mean = (np.sum(PT[inr])*H_ + lo_ext*1.0)/(b - a)
        for dn, d in DEFS.items():
            Lk = norm.pdf(MG, d["mu"], d["sig"]); num = np.sum((PT*Lk)[inr])*H_; den = np.sum(Lk[inr])*H_
            Ba = num/den; Bb = Ba/prior_mean
            BF[f"[{a:g},{b:g}]|{dn}"] = dict(B_existence_as_data=Ba, B_conditioned_on_existence=Bb, prior_mean_P1=prior_mean,
                                               lnB_a=float(np.log(Ba)), lnB_b=float(np.log(Bb)))
        for sn in ["BKPV_FJ_-0.36/+0.17", "mbar_0.07"]:
            PTv = Pbar_true("A", True, SAS, SM_VARIANTS[sn]); pm = (np.sum(PTv[inr])*H_ + lo_ext)/(b - a)
            d = DEFS["direct_MC"]; Lk = norm.pdf(MG, d["mu"], d["sig"]); Ba = np.sum((PTv*Lk)[inr])/np.sum(Lk[inr])
            BF[f"[{a:g},{b:g}]|direct_MC|{sn}"] = dict(B_existence_as_data=float(Ba), B_conditioned_on_existence=float(Ba/pm), prior_mean_P1=float(pm))
    # MC cross-check of truncated B_a for sigma_pole on [170,176]
    BF["MC_check_[170,176]|sigma_pole_Ba"] = mc_run(172.1, 0.6, ("normal", 0.10), trunc=(170, 176), seed=4242)["mean"]
    R["bayes"] = BF; print("bayes", json.dumps(BF)[:1500], flush=True)

    # ---- 3. exclusion thresholds ----
    EX = {}
    for als, dal in [("0.1180", 0.0), ("0.1171", -0.0009), ("0.1189", +0.0009)]:
        row = {}
        for l, xl in zip(LEV, XL):
            mc0 = float(CV.m_of_level(xl)); mc = mc0 + float(s_of(xl))*dal + 0.5*float(c_of(xl))*dal**2
            sp_, sm_ = sig_model(np.array([xl]), "A"); row[f"P1={l:g}"] = dict(mc=mc, table=mc_tab(als, l), diff=mc - mc_tab(als, l),
                                        model_band=[mc - float(sm_[0]), mc + float(sp_[0])])
        EX[als] = row
    # marginalised thresholds (true mass known exactly; alpha_s, SM, model marginalised)
    EX["marginalised_true_mass_thresholds"] = {f"Pbar={l:g}": interp_root(MG, PT, l) for l in [0.9, 0.5, 0.1, 0.05]}
    EX["marginalised_true_mass_thresholds_noModel"] = {f"Pbar={l:g}": interp_root(MG, Pbar_true("A", False, SAS, ("normal", 0.10)), l) for l in [0.9, 0.5, 0.1]}
    # 95% exclusion: central measured value needed (current precision)
    ag, pA = shift_density(SAS, ("normal", 0.10))
    x01 = XL[2]; mc01 = float(CV.m_of_level(x01)); spm, smm = [float(v[0]) for v in sig_model(np.array([x01]), "A")]
    dT = np.where(ag < 0, norm.pdf(ag/smm)/smm, norm.pdf(ag/spm)/spm)*0.5
    pT = np.convolve(pA*H_, dT, mode="same"); pT /= pT.sum()*H_            # density of T - mc01
    need = {}
    for dn, d in DEFS.items():
        fA = lambda mu: float(np.sum(pT*norm.cdf((mc01 + ag - mu)/d["sig"]))*H_) - 0.05
        muA = brentq(fA, 170, 180)
        PM = Pbar_meas_curve(PT, d["sig"]); muB = interp_root(MG, PM, 0.05); muB10 = interp_root(MG, PM, 0.10)
        off = 0.36 if dn == "MSR_identified" else 0.0
        need[dn] = dict(sigma=d["sig"], defA_post_prob_mtrue_above_mc01_is_0p95=muA - off, defB_Pbar_eq_0p05=muB - off, defB_Pbar_eq_0p10=muB10 - off,
                    note=("value of m_MC (MSR route: m_pole - 0.36)" if off else "central value of this measurement"))
    EX["required_central_value_95"] = need
    EX["threshold_T01_sigma_eff"] = float(np.sqrt(np.sum(pT*ag**2)*H_ - (np.sum(pT*ag)*H_)**2)); EX["threshold_T01_mean_shift"] = float(np.sum(pT*ag)*H_)
    R["exclusion"] = EX; print("exclusion", json.dumps(EX)[:2500], flush=True)

    # ---- 3b. future discriminating power ----
    LIT = dict(
        HL_LHC=dict(src="arXiv:1902.04070v3 (CERN Yellow Rep. Monogr. 7 (2019) 1, WG1 'Standard Model Physics at the HL-LHC and HE-LHC')",
                    dmt=dict(value=0.17, where="Table 47, pdf p.149: CMS projection 3 ab^-1, ttbar lepton+jets total 0.17 GeV (MC mass)"),
                    dmt_EWfit=dict(value=0.40, where="pdf p.87: EW-fit assumption, 'uncertainty on mt at the level of 0.4 GeV ... difficult to further reduce at the HL-LHC'"),
                    das=dict(value=0.0002, where="pdf p.88: assumed future lattice QCD, delta alpha_S(MZ) = +-0.0002 (HL-LHC EW-fit projection)")),
        FCC_ee=dict(src="arXiv:2505.00272v1 (FCC Feasibility Study Report Vol. 1)",
                    dmt_exp=dict(value=float(np.hypot(0.0042, 0.0049)), where="Table 2, pdf p.37: mtop stat 4.2 MeV, syst 4.9 MeV (tt threshold scan; 'QCD uncert. dominate')"),
                    dmt_theory=dict(value=0.050, where="pdf p.99: expected theoretical accuracy O(50) MeV (PS scheme), O(30) MeV scale uncertainty; pdf p.94: O(30 MeV) scale uncert."),
                    das=dict(value=0.00010, where="Table 2, pdf p.37: alpha_S(mZ^2) (x10^4) stat 0.1, syst 1; pdf p.97: 'a precision of 0.1% on alpha_S(mZ^2)'")))
    SCEN = {
        "current_direct": dict(dmt=0.27, das=SAS, sm=("normal", 0.10)),
        "HL-LHC_0.17": dict(dmt=0.17, das=0.0002, sm=("normal", 0.10)),
        "HL-LHC_0.17_as0.00045": dict(dmt=0.17, das=0.00045, sm=("normal", 0.10)),
        "HL-LHC_0.40(EWfit)": dict(dmt=0.40, das=0.0002, sm=("normal", 0.10)),
        "HL-LHC_0.17+interp0.5": dict(dmt=float(np.hypot(0.17, 0.5)), das=0.0002, sm=("normal", 0.10)),
        "FCC-ee_50MeV": dict(dmt=float(np.hypot(LIT["FCC_ee"]["dmt_exp"]["value"], 0.050)), das=0.00010, sm=("normal", 0.07)),
        "FCC-ee_30MeV": dict(dmt=float(np.hypot(LIT["FCC_ee"]["dmt_exp"]["value"], 0.030)), das=0.00010, sm=("normal", 0.07)),
    }
    FU = {}
    mod_sym = 0.5*(MODEL_SP[1] + MODEL_SM[1])
    for sn, s in SCEN.items():
        PTs = Pbar_true("A", True, s["das"], s["sm"]); PTs0 = Pbar_true("A", False, s["das"], s["sm"])
        sig_tot = float(np.sqrt(s["dmt"]**2 + (SBAR*s["das"])**2 + s["sm"][1]**2 + mod_sym**2)); sig_tot0 = float(np.sqrt(s["dmt"]**2 + (SBAR*s["das"])**2 + s["sm"][1]**2))
        row = dict(inputs=s, sigma_tot_GeV=sig_tot, sigma_tot_noModel=sig_tot0, D_meas_sigma=0.7/s["dmt"], D_verdict_sigma=0.7/sig_tot, D_verdict_noModel=0.7/sig_tot0)
        for mt in [172.6, 173.3]:
            row[f"Asimov_Pbar_{mt}"] = Pbar_meas(PTs, mt, s["dmt"]); row[f"Asimov_Pbar_noModel_{mt}"] = Pbar_meas(PTs0, mt, s["dmt"])
            row[f"z_to_mc0.5_{mt}"] = (mt - mc_tab("0.1180", 0.5))/sig_tot; row[f"z_to_mc0.1_{mt}"] = (mt - mc_tab("0.1180", 0.1))/sig_tot
        FU[sn] = row
        print("future", sn, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "inputs"}, flush=True)
    R["future"] = dict(literature=LIT, scenarios=FU, note="D_meas = 0.7/dmt (theory/model errors common to both true values cancel); D_verdict = 0.7/sigma_tot with sigma_tot including alpha_s, SM-mapping and (symmetrised P1=0.5) model errors; Asimov_Pbar = marginalised P1 for a measurement equal to the true value")
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUTD, "stats_results.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("done %.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
