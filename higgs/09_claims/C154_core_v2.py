# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/C154_core_v2.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations\code\20261014_claude_v12_r3\claim_positioning\C154_core_v2.py on 2026-10-15 by claims-run05 (C161); edits: 'CFG_PATH = os.path.join(HERE, "C154_inputs_v2.json")' -> 'CFG_PATH = os.environ.get("C154_CFG", os.path.join(HERE, "C1'; '"task": "C154 run02 / D126 v2 claim positioning", "agent": "' -> '"task": "C161 claims-run05: C154 v2 core rerun (central P1 f'
# C154_core_v2.py -- T13 claim positioning, run02 (round-3 final inputs), agent claim-positioning, 2026-10-15
# PROVENANCE: derived from C154_core.py (same directory, run01 version) on 2026-10-15 by claim-positioning;
#             C154_core.py / C154_inputs.json / run01 outputs are left untouched.
# Changes vs run01: central P1 = C149 run02 (2D direct + NNLO + cross terms) with level-dependent alpha_s map;
#   M_h = 125.13; SM-side errors from R16 V1 diagnosis (pole 0.102 / m-bar 0.070 + variants);
#   level-dependent model error (two-piece in one coherent nuisance Z, median-preserving);
#   (m_t, xi_H) plane from T6 (C157) and T21 (C155).
# Run: python -B C154_core_v2.py   (inputs: C154_inputs_v2.json)
"""[v12 package, 2026-10-15] Claims layer (D126 v2; C161 run05 centre): marginalised survival P1bar over m_t (several mass definitions), alpha_s, M_h, SM-mapping and model errors (asymmetric errors split with equal mass), Bayes factors, exclusion thresholds, FCC-ee / HL-LHC discriminating power, (m_t, alpha_s) contours and figure data.  Inputs: env C154_CFG (default C154_inputs_v2_run05.json; *_t1new = rebuilt Table 1 composite).  Paper: PRL claim sentence, Fig. 2, Table 1; PRD App. H4.  Units: GeV."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os, sys, json, math, hashlib, csv, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.special import ndtr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = _zp.WORK
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.environ.get("C154_CFG", os.path.join(HERE, "C154_inputs_v2_run05.json"))
CFG = json.load(open(CFG_PATH, encoding="utf-8"))
OUT = os.path.join(ROOT, CFG["out_dir"]); os.makedirs(OUT, exist_ok=True)
H = 0.0025
LNLN2 = math.log(math.log(2.0))
LEV = (0.9, 0.5, 0.1); LK = ("0.9", "0.5", "0.1")
A0 = CFG["alpha_s"]["central"]; SA = CFG["alpha_s"]["sigma"]
MH = CFG["Mh"]; DMC_DMH = MH["dlam_dMh"] / abs(MH["dlam_dMt"])
MU_MH = (MH["measured"] - MH["curve_baseline"]) * DMC_DMH; SIG_MH = MH["sigma"] * DMC_DMH
SMC = CFG["sm_mapping"]
TOUCH = []


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wcsv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header)
        for r in rows:
            w.writerow([("%.6g" % x) if isinstance(x, float) else x for x in r])
    TOUCH.append(name)


class LinExt:
    def __init__(self, x, y):
        self.cs = CubicSpline(x, y); self.x0, self.x1 = float(x[0]), float(x[-1])
        k = min(4, len(x))
        self.s0 = np.polyfit(x[:k], y[:k], 1)[0]; self.s1 = np.polyfit(x[-k:], y[-k:], 1)[0]
        self.y0 = float(self.cs(self.x0)); self.y1 = float(self.cs(self.x1))

    def __call__(self, x):
        x = np.asarray(x, float); y = self.cs(np.clip(x, self.x0, self.x1))
        y = np.where(x < self.x0, self.y0 + self.s0 * (x - self.x0), y)
        return np.where(x > self.x1, self.y1 + self.s1 * (x - self.x1), y)


class Surface:
    def __init__(self, tab, a_ref=0.1180):
        self.a = np.array(sorted(tab)); self.a_ref = a_ref; self.spl = []; mc = []
        for a in self.a:
            m, y = tab[a]["m"], tab[a]["lnI"]
            cs = CubicSpline(m, y); c = brentq(lambda x: float(cs(x)) - LNLN2, m[0], m[-1]); mc.append(c)
            self.spl.append(LinExt(m - c, y))
        self.mc05 = np.array(mc); self.mc_coef = np.polyfit(self.a - a_ref, self.mc05, min(2, len(self.a) - 1))

    def mc(self, a):
        return np.polyval(self.mc_coef, np.asarray(a, float) - self.a_ref)

    def lnI(self, m, a):
        m = np.asarray(m, float); a = np.asarray(a, float) * np.ones_like(m)
        x = m - self.mc(a); ac = np.clip(a, self.a[0], self.a[-1]); tot = 0.0
        for i in range(len(self.a)):
            w = np.ones_like(ac)
            for j in range(len(self.a)):
                if j != i:
                    w = w * (ac - self.a[j]) / (self.a[i] - self.a[j])
            tot = tot + w * self.spl[i](x)
        return tot

    def P1(self, m, a):
        return np.exp(-np.exp(np.minimum(self.lnI(m, a), 60.0)))

    def mcrit(self, p, a):
        t = math.log(-math.log(p))
        return brentq(lambda x: float(self.lnI(np.array([x]), np.array([a]))[0]) - t, 150.0, 200.0, xtol=1e-9)


def sn_pdf(x, s_left, s_right):
    c = 2.0 / (math.sqrt(2 * math.pi) * (s_left + s_right)); s = np.where(x < 0, s_left, s_right)
    return c * np.exp(-0.5 * (x / s) ** 2)


def kernel(sig_g, mu_g, s_up, s_dn):
    sig_g = max(sig_g, H); s_up = max(s_up, H); s_dn = max(s_dn, H)
    L = abs(mu_g) + 10 * max(sig_g, s_up, s_dn) + 10 * H; n = int(math.ceil(L / H)); d = np.arange(-n, n + 1) * H
    k = fftconvolve(np.exp(-0.5 * ((d - mu_g) / sig_g) ** 2), sn_pdf(d, s_dn, s_up), mode="same")
    k = np.clip(k, 0, None); k /= k.sum() * H
    return d, k


def conv(f, k):
    n = (len(k) - 1) // 2
    return fftconvolve(np.pad(f, n, mode="edge"), k, mode="valid") * H


def gh(n=40):
    t, w = np.polynomial.hermite.hermgauss(n)
    return math.sqrt(2.0) * t, w / math.sqrt(math.pi)


def composite(comps, drop=()):
    sp = np.zeros(3); sm = np.zeros(3)
    for c in comps:
        if c["id"] in drop:
            continue
        for i, L in enumerate(LK):
            p, m = c["levels"][L]; sp[i] += p * p; sm[i] += m * m
    return np.sqrt(sp), np.sqrt(sm)


class Theory2:
    """G(m) = E_{alpha, delta_g, Z}[ w0(m) P1(m + kappa(alpha-a_c) - delta_g - d(x, Z), alpha) ];
    delta_g ~ N(mu, s_SM (+) s_Mh) [or split SM], d = Z sigma_+(x) (Z>0) / Z sigma_-(x) (Z<0) + det(x), Z ~ N(0,1)."""
    def __init__(self, S, xl, a_c=A0, s_a=SA, s_SM=SMC["pole"], sm_split=None, s_Mh=SIG_MH, mu=MU_MH,
                 msig=None, det=None, kappa=0.0, w0=None, lo=160.0, hi=186.0, nz=40):
        self.S, self.xl, self.a_c, self.s_a, self.kappa = S, np.asarray(xl), a_c, s_a, kappa
        self.mg = np.arange(lo, hi + H / 2, H)
        if sm_split is None:
            self.d, self.k = kernel(math.hypot(s_SM, s_Mh), mu, 0.0, 0.0)
        else:
            self.d, self.k = kernel(s_Mh, mu, sm_split[0], sm_split[1])       # (up, down)
        self.cdf = np.clip(np.cumsum(self.k) * H, 0, 1)
        self.sp = np.asarray(msig[0]); self.sm = np.asarray(msig[1]); self.det = None if det is None else np.asarray(det)
        self.ta, self.wa = (gh(40) if s_a > 0 else (np.array([0.0]), np.array([1.0])))
        self.tz, self.wz = gh(nz); self.w0 = w0
        G = np.zeros_like(self.mg)
        for ti, wi in zip(self.ta, self.wa):
            a = a_c + s_a * ti
            Fa = S.P1(self.mg + kappa * (a - a_c), a)
            if w0 is not None:
                Fa = Fa * w0(self.mg + kappa * (a - a_c))
            Ga = conv(Fa, self.k)
            xs = self.mg + kappa * (a - a_c) - float(S.mc(a))
            spx = np.interp(xs, self.xl, self.sp); smx = np.interp(xs, self.xl, self.sm)
            dtx = np.interp(xs, self.xl, self.det) if self.det is not None else 0.0
            for zj, vj in zip(self.tz, self.wz):
                dd = zj * (spx if zj > 0 else smx) + dtx
                G += wi * vj * np.interp(self.mg - dd, self.mg, Ga)
        self.G = G; self._H = {}

    def lev_sig(self, p):
        xp = self.S.mcrit(p, self.S.a_ref) - self.S.mcrit(0.5, self.S.a_ref)
        return (float(np.interp(xp, self.xl, self.sp)), float(np.interp(xp, self.xl, self.sm)),
                float(np.interp(xp, self.xl, self.det)) if self.det is not None else 0.0)

    def Hp(self, p):
        """P(P1 >= p | true measured-scale m)"""
        if p in self._H:
            return self._H[p]
        sp_, sm_, dt_ = self.lev_sig(p); Hs = np.zeros_like(self.mg)
        for ti, wi in zip(self.ta, self.wa):
            a = self.a_c + self.s_a * ti; mcp = self.S.mcrit(p, a)
            for zj, vj in zip(self.tz, self.wz):
                dd = zj * (sp_ if zj > 0 else sm_) + dt_
                y = self.mg + self.kappa * (a - self.a_c) - dd - mcp
                Hs += wi * vj * (1.0 - np.interp(y, self.d, self.cdf, left=0.0, right=1.0))
        self._H[p] = Hs
        return Hs

    def mc(self, c, sl, sr, n=1_000_000, seed=20261015):
        rng = np.random.default_rng(seed)
        z = np.abs(rng.standard_normal(n)); side = rng.random(n) < sl / (sl + sr)
        m = np.where(side, c - sl * z, c + sr * z)
        a = self.a_c + self.s_a * rng.standard_normal(n)
        dg = np.interp(rng.random(n), self.cdf, self.d)
        Z = rng.standard_normal(n)
        m0 = m + self.kappa * (a - self.a_c) - dg
        xs = m0 - self.S.mc(a)
        sig = np.where(Z > 0, np.interp(xs, self.xl, self.sp), np.interp(xs, self.xl, self.sm))
        dd = Z * sig + (np.interp(xs, self.xl, self.det) if self.det is not None else 0.0)
        P = self.S.P1(m0 - dd, a)
        if self.w0 is not None:
            P = P * self.w0(m0 - dd)
        return P


def meas_params(ms):
    if ms.get("msbar_map"):
        P = np.array(CFG["msbar_points"]["pole"]); Mb = np.array(CFG["msbar_points"]["mbar"]); b, a0 = np.polyfit(Mb, P, 1)
        return a0 + b * ms["m_msbar"], b * ms["s_minus"], b * ms["s_plus"]
    s = math.sqrt(ms["s"] ** 2 + sum(e * e for e in ms.get("extra", [])))
    return ms["m"] + ms.get("offset", 0.0), s, s


def integ(T, arr, c, sl, sr):
    w = sn_pdf(T.mg - c, sl, sr)
    return float(np.trapezoid(arr * w, T.mg) / np.trapezoid(w, T.mg))


def stats(P):
    q = np.quantile(P, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(mean=float(P.mean()), q025=float(q[0]), q16=float(q[1]), median=float(q[2]), q84=float(q[3]), q975=float(q[4]),
                prob_P_ge_0p5=float((P >= 0.5).mean()), prob_P_lt_0p1=float((P < 0.1).mean()))


def main():
    t0 = time.time()
    summ = {"task": "C161 claims-run05: C154 v2 core rerun (central P1 from CFG)", "agent": "claims-run05", "date": "2026-10-15", "inputs_cfg": CFG_PATH}
    # ---------------- central P1 tables (C149 run02) with level-dependent alpha_s map
    pc = os.path.join(ROOT, CFG["central_P1"]["file"]); dc = json.load(open(pc, encoding="utf-8"))
    g = dc["P1_grid"]; ks = sorted(g, key=float)
    mref = np.array([float(k) for k in ks]); Pref = np.array([g[k] for k in ks]); yref = np.log(-np.log(Pref))
    cr = {a: np.array([dc["crit"][a]["with_cross"][f"P1={p}"] for p in LEV]) for a in dc["crit"]}
    cref = cr["0.1180"]; tabs = {}
    for a, ck in cr.items():
        xr = cref - cref[1]; xk = ck - ck[1]; cf = np.polyfit(xr, xk, 2); dcf = np.polyder(cf)
        x = mref - cref[1]
        q = np.polyval(cf, x)
        q = np.where(x < xr[0], np.polyval(cf, xr[0]) + np.polyval(dcf, xr[0]) * (x - xr[0]), q)
        q = np.where(x > xr[2], np.polyval(cf, xr[2]) + np.polyval(dcf, xr[2]) * (x - xr[2]), q)
        tabs[float(a)] = dict(m=ck[1] + q, lnI=yref)
    S = Surface(tabs)
    xl = [S.mcrit(p, 0.118) - S.mcrit(0.5, 0.118) for p in LEV]
    crit = {f"{a:.4f}": {f"P1={p}": S.mcrit(p, a) for p in (0.9, 0.5, 0.1, 0.05)} for a in (0.1171, 0.1180, 0.1189)}
    chk = {a: {f"P1={p}": crit[a][f"P1={p}"] - dc["crit"][a]["with_cross"][f"P1={p}"] for p in LEV} for a in dc["crit"]}
    summ["critical_mt"] = crit; summ["critical_vs_C149"] = chk
    summ["dmc_dalpha_s"] = {f"P1={p}": (crit["0.1189"][f"P1={p}"] - crit["0.1171"][f"P1={p}"]) / 0.0018 for p in LEV}
    # interpolation checks: leave-one-out on the 10-point grid; T1 + level shift reconstruction
    loo = []
    for i in range(1, len(mref) - 1):
        keep = np.arange(len(mref)) != i
        cs = CubicSpline(mref[keep], yref[keep]); loo.append(float(np.exp(-np.exp(cs(mref[i]))) - Pref[i]))
    t1 = json.load(open(os.path.join(ROOT, CFG["t1_tables"]), encoding="utf-8"))["P1_table"]["0.1180"]
    k1 = sorted(t1, key=float); m1 = np.array([float(k) for k in k1]); y1 = np.log(np.array([t1[k]["I"] for k in k1]))
    lam1 = np.array([t1[k]["lam_b"] for k in k1]); cs1 = CubicSpline(m1, y1)
    t1c = np.array([brentq(lambda x: float(cs1(x)) - math.log(-math.log(p)), m1[0], m1[-1]) for p in LEV])
    shift = cref - t1c
    PA = np.exp(-np.exp(cs1(mref - np.interp(mref, cref, shift))))
    summ["interpolation_check"] = {"loo_10pt_grid_max_abs_dP": float(np.max(np.abs(loo))), "loo_values": loo,
                                   "T1_plus_level_shift_vs_C149_grid_max_abs_dP": float(np.max(np.abs(PA - Pref))),
                                   "level_shift_vs_T1": shift.tolist()}
    print("crit", json.dumps(crit)); print("chk", chk); print("interp", summ["interpolation_check"]["loo_10pt_grid_max_abs_dP"],
                                                          summ["interpolation_check"]["T1_plus_level_shift_vs_C149_grid_max_abs_dP"])
    # relative form
    lf = np.polyfit(m1 - 172.6, lam1, 2); m0 = brentq(lambda x: np.polyval(lf, x - 172.6), 165.0, 172.5)
    lam_sm = float(np.polyval(lf, 0.0)) + MH["dlam_dMh"] * (MH["measured"] - MH["curve_baseline"])
    summ["relative_form"] = {"m0": m0, "mc_minus_m0": {f"P1={p}": crit["0.1180"][f"P1={p}"] - m0 for p in LEV},
                             "lambda_c_C149": dc.get("lam_c_hazard_equiv"),
                             "lambda_c_from_T1_lam_fit": {f"P1={p}": float(np.polyval(lf, crit["0.1180"][f"P1={p}"] - 172.6)) for p in LEV},
                             "lambda_SM_mu_b_172p60": lam_sm,
                             "margin_lambda": lam_sm - dc["lam_c_hazard_equiv"]["P1=0.5"],
                             "margin_mc05_minus_172p60": crit["0.1180"]["P1=0.5"] + MU_MH - 172.60}

    # ---------------- model error composite (Table 1)
    comps = CFG["model_components"]; SP, SM = composite(comps)
    SP_novi, SM_novi = composite(comps, drop=("inflation_scale_vi",))
    SP_now0, SM_now0 = composite(comps, drop=("w0_T15",))
    summ["model_composite"] = {"plus": SP.tolist(), "minus": SM.tolist(), "plus_no_vi": SP_novi.tolist(), "minus_no_vi": SM_novi.tolist(), "levels": LK}
    trow = []
    for c in comps:
        trow.append([c["id"], c["type"]] + [v for L in LK for v in c["levels"][L]] + [c["src"]])
    trow.append(["MODEL TOTAL (quadrature per side)", "composite"] + [v for i in range(3) for v in (float(SP[i]), float(SM[i]))] + ["this work"])
    trow.append(["model total without (vi)", "composite"] + [v for i in range(3) for v in (float(SP_novi[i]), float(SM_novi[i]))] + ["this work"])
    sm_rows = [["SM: matching-scale envelope (m-bar)", "SM", 0.0544, 0.0415, 0.0544, 0.0415, 0.0544, 0.0415, "R16 V1 DIAGNOSIS recommended.mbar"],
               ["SM: scheme spread mr vs SMDR (incl. 0.04 implementation)", "SM", 0.0419, 0.0419, 0.0419, 0.0419, 0.0419, 0.0419, "same"],
               ["SM: RGE / alpha_s order / W,Z pole convention", "SM", 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, "same (0.005, 0.003, 0.011 in quadrature)"],
               ["SM TOTAL m-bar formulation", "SM", SMC["msbar"], SMC["msbar"], SMC["msbar"], SMC["msbar"], SMC["msbar"], SMC["msbar"], "DIAGNOSIS total 0.0698"],
               ["SM: pole series O(alpha_s^5) (pole formulation only)", "SM", 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, "DIAGNOSIS"],
               ["SM TOTAL pole formulation (4-loop truncated)", "SM", SMC["pole"], SMC["pole"], SMC["pole"], SMC["pole"], SMC["pole"], SMC["pole"], "DIAGNOSIS 0.1024"],
               ["SM: all-orders pole offset (one-sided, criterion up)", "SM-variant", SMC["pole_allorders_offset"], 0.0, SMC["pole_allorders_offset"], 0.0, SMC["pole_allorders_offset"], 0.0, "Beneke 2016 Eq.(4.7) via DIAGNOSIS"],
               ["SM: renormalon ambiguity", "SM-variant", 0.11, 0.07, 0.11, 0.07, 0.11, 0.07, "range 0.07-0.11 (not additive to 4-loop convention)"],
               ["SM: BKPV/FJ scale variation (robustness)", "SM-variant", SMC["BKPV_FJ"]["plus"], SMC["BKPV_FJ"]["minus"], SMC["BKPV_FJ"]["plus"], SMC["BKPV_FJ"]["minus"], SMC["BKPV_FJ"]["plus"], SMC["BKPV_FJ"]["minus"], "DIAGNOSIS item1 (-0.358/+0.173)"],
               ["parametric alpha_s +-0.0009", "param", 0.0009 * summ["dmc_dalpha_s"]["P1=0.9"], 0.0009 * summ["dmc_dalpha_s"]["P1=0.9"], 0.0009 * summ["dmc_dalpha_s"]["P1=0.5"], 0.0009 * summ["dmc_dalpha_s"]["P1=0.5"], 0.0009 * summ["dmc_dalpha_s"]["P1=0.1"], 0.0009 * summ["dmc_dalpha_s"]["P1=0.1"], "C149 crit"],
               ["parametric M_h +-0.11 (PDG 2026)", "param", SIG_MH, SIG_MH, SIG_MH, SIG_MH, SIG_MH, SIG_MH, "T1 slopes"],
               ["measurement m_t direct (PDG 2026)", "meas", 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, "PDG 2026"],
               ["MC-pole interpretation", "meas-variant", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, "Hoang 2020"]]
    wcsv("table1_error_budget_final.csv", ["source", "type", "P0.9_plus", "P0.9_minus", "P0.5_plus", "P0.5_minus", "P0.1_plus", "P0.1_minus", "ref"],
         trow + sm_rows)
    print("model composite +", SP, "-", SM)

    # ---------------- theories
    MS = (SP, SM)
    th = {"pole": Theory2(S, xl, s_SM=SMC["pole"], msig=MS), "msbar": Theory2(S, xl, s_SM=SMC["msbar"], msig=MS)}
    s_all = math.hypot(SMC["msbar"], SMC["renormalon"][1])
    var_th = {
        "BKPV_FJ_SM": Theory2(S, xl, sm_split=(SMC["BKPV_FJ"]["plus"], SMC["BKPV_FJ"]["minus"]), msig=MS),
        "allorders_pole": Theory2(S, xl, s_SM=s_all, mu=MU_MH + SMC["pole_allorders_offset"], msig=MS),
        "T14a_central_correction": Theory2(S, xl, s_SM=SMC["pole"], msig=MS, det=next(c for c in comps if c["id"] == "fulldim_T14a")["central_correction"]),
        "vi_r1e-4": Theory2(S, xl, s_SM=SMC["pole"], msig=(SP_novi, SM_novi), det=next(c for c in comps if c["id"] == "inflation_scale_vi")["r1e-4_one_sided"]),
        "model_without_vi": Theory2(S, xl, s_SM=SMC["pole"], msig=(SP_novi, SM_novi)),
        "w0_multiplicative": Theory2(S, xl, s_SM=SMC["pole"], msig=(SP_now0, SM_now0),
                                     w0=lambda m: np.interp(m, CFG["w0_B3_wall1p2"]["m"], CFG["w0_B3_wall1p2"]["w0"])),
    }
    marg = {}; rows = []
    for ms in CFG["measurements"]:
        c, sl, sr = meas_params(ms); T = th[ms["scheme"]]
        Pq = integ(T, T.G, c, sl, sr); st = stats(T.mc(c, sl, sr))
        e = dict(label=ms["label"], center_pole4L_equiv=c, s_left=sl, s_right=sr, scheme=ms["scheme"],
                 P1_curve_at_center=float(S.P1(np.array([c - MU_MH]), A0)[0]), Pbar=Pq, mc=st,
                 prob_P_ge_0p5=integ(T, T.Hp(0.5), c, sl, sr))
        if ms["scheme"] == "pole":
            for vn, VT in var_th.items():
                e[f"Pbar_{vn}"] = integ(VT, VT.G, c, sl, sr)
        marg[ms["id"]] = e
        rows.append([ms["id"], c, sl, ms["scheme"], e["P1_curve_at_center"], Pq, st["mean"], st["median"], st["q16"], st["q84"], st["q025"], st["q975"],
                     e["prob_P_ge_0p5"]] + [e.get(f"Pbar_{vn}", "") for vn in var_th])
        print(f"{ms['id']:28s} Pq={Pq:.4f} MC={st['mean']:.4f} med={st['median']:.3f} 68=[{st['q16']:.3f},{st['q84']:.3f}] 95=[{st['q025']:.3f},{st['q975']:.3f}] "
              f"Pge05={e['prob_P_ge_0p5']:.3f} " + " ".join(f"{vn}={e.get('Pbar_' + vn, float('nan')):.3f}" for vn in var_th))
    summ["marginal"] = marg
    wcsv("marginal_P1.csv", ["id", "center_pole4L_equiv", "sigma_meas", "scheme", "P1_curve_center", "Pbar_quad", "Pbar_mc", "median", "q16", "q84",
                             "q025", "q975", "prob_P_ge_0p5"] + [f"Pbar_{vn}" for vn in var_th], rows)
    # progression (direct)
    prog = []
    steps = [("curve only", dict(s_a=0.0, s_SM=0.0, s_Mh=0.0, msig=(np.zeros(3), np.zeros(3))), 0.0),
             ("+ m_t exp 0.27", dict(s_a=0.0, s_SM=0.0, s_Mh=0.0, msig=(np.zeros(3), np.zeros(3))), 0.27),
             ("+ alpha_s 0.0009", dict(s_SM=0.0, s_Mh=0.0, msig=(np.zeros(3), np.zeros(3))), 0.27),
             ("+ M_h 0.11", dict(s_SM=0.0, msig=(np.zeros(3), np.zeros(3))), 0.27),
             ("+ SM mapping pole 0.102", dict(s_SM=SMC["pole"], msig=(np.zeros(3), np.zeros(3))), 0.27),
             ("+ model without (vi)", dict(s_SM=SMC["pole"], msig=(SP_novi, SM_novi)), 0.27),
             ("+ model incl. (vi) [primary]", dict(s_SM=SMC["pole"], msig=MS), 0.27),
             ("+ MC-pole interpretation 0.5", dict(s_SM=SMC["pole"], msig=MS), math.hypot(0.27, 0.5))]
    for name, kw, sm in steps:
        T = Theory2(S, xl, **kw)
        Pv = integ(T, T.G, 172.60, sm, sm) if sm > 0 else float(np.interp(172.60, T.mg, T.G))
        prog.append([name, sm, Pv]); print(f"prog {name:34s} {Pv:.4f}")
    wcsv("table1_Pbar_progression_direct.csv", ["step", "sigma_meas", "Pbar"], prog)
    summ["Pbar_progression_direct"] = {r[0]: r[2] for r in prog}
    # probit check
    w_ = (crit["0.1180"]["P1=0.1"] - crit["0.1180"]["P1=0.9"]) / (2 * 1.2815516)
    vm = 0.5 * (SP[1] ** 2 + SM[1] ** 2)
    sig_t = math.sqrt(0.27 ** 2 + (summ["dmc_dalpha_s"]["P1=0.5"] * SA) ** 2 + SMC["pole"] ** 2 + SIG_MH ** 2 + vm)
    summ["probit_check_D1"] = {"Pbar_probit": float(ndtr((crit["0.1180"]["P1=0.5"] + MU_MH - 172.60) / math.hypot(w_, sig_t))),
                               "Pbar_quad": marg["D1_direct_pole"]["Pbar"], "w": w_, "sigma_tot": sig_t}
    print("probit", summ["probit_check_D1"])

    # ---------------- Bayes
    lam_fit = lf; brows = []; bay = {}
    for ms in CFG["measurements"]:
        c, sl, sr = meas_params(ms); T = th[ms["scheme"]]; L = sn_pdf(T.mg - c, sl, sr); bay[ms["id"]] = {}
        for pr in CFG["priors"]:
            if pr["type"] == "flat_m":
                pi = ((T.mg >= pr["lo"]) & (T.mg <= pr["hi"])).astype(float)
            else:
                lam = np.polyval(lam_fit, T.mg - 172.6); dl = np.abs(np.polyval(np.polyder(lam_fit), T.mg - 172.6))
                pi = dl * ((lam >= pr["lo"]) & (lam <= pr["hi"]))
            pi = pi / np.trapezoid(pi, T.mg)
            Bex = float(np.trapezoid(T.G * L * pi, T.mg) / np.trapezoid(L * pi, T.mg)); pm = float(np.trapezoid(T.G * pi, T.mg))
            bay[ms["id"]][pr["id"]] = dict(B_existence=Bex, prior_mean_P1=pm, B_selection=Bex / pm, B_selection_max=1 / pm)
            brows.append([ms["id"], pr["id"], Bex, pm, Bex / pm, 1 / pm])
    summ["bayes"] = bay
    wcsv("bayes_factors.csv", ["measurement", "prior", "B_existence", "prior_mean_P1", "B_selection", "B_selection_max"], brows)
    print("bayes D1", bay["D1_direct_pole"])

    # ---------------- exclusion thresholds
    xrows = []; excl = {}
    for a in (0.1171, 0.1180, 0.1189):
        Tt = Theory2(S, xl, a_c=a, s_a=0.0, s_SM=SMC["pole"], msig=MS)
        Tm = Theory2(S, xl, a_c=a, s_a=SA, s_SM=SMC["pole"], msig=MS)
        r = {}
        for p in (0.1, 0.05):
            r[f"curve_P{p}"] = S.mcrit(p, a) + MU_MH
            Hp = Tt.Hp(p); r[f"theory95_P{p}"] = brentq(lambda m: 1 - float(np.interp(m, Tt.mg, Hp)) - 0.95, 168.0, 182.0)
            Hm = Tm.Hp(p)
            r[f"meas0p27_95_P{p}"] = brentq(lambda cc: 1 - integ(Tm, Hm, cc, 0.27, 0.27) - 0.95, 168.0, 182.0)
            r[f"meas_interp_95_P{p}"] = brentq(lambda cc: 1 - integ(Tm, Hm, cc, math.hypot(0.27, 0.5), math.hypot(0.27, 0.5)) - 0.95, 168.0, 184.0)
        excl[f"{a:.4f}"] = r; xrows.append([a] + [r[k] for k in sorted(r)])
    wcsv("exclusion_mt.csv", ["alpha_s_central"] + sorted(excl["0.1180"]), xrows); summ["exclusion_mt"] = excl
    print("excl", json.dumps(excl))

    # ---------------- future reach
    b_ms = np.polyfit(CFG["msbar_points"]["mbar"], CFG["msbar_points"]["pole"], 1)[0]; frows = []; fut = {}
    for sc in CFG["future"]:
        sm = math.sqrt(sc["s_m"] ** 2 + sum(e * e for e in sc["extra"])) * (b_ms if sc["scheme"] == "msbar" else 1.0)
        T = Theory2(S, xl, s_a=sc["s_as"], s_SM=sc["s_SM"], s_Mh=sc["s_Mh"] * DMC_DMH, msig=MS, kappa=sc["kappa"])
        H5, H1, H05 = T.Hp(0.5), T.Hp(0.1), T.Hp(0.05)
        ev = lambda arr, cc: integ(T, arr, cc, sm, sm)
        thr_s = brentq(lambda cc: ev(H5, cc) - 0.95, 168.0, 178.0)
        thr_1 = brentq(lambda cc: 1 - ev(H1, cc) - 0.95, 168.0, 182.0); thr_05 = brentq(lambda cc: 1 - ev(H05, cc) - 0.95, 168.0, 182.0)
        stot = math.sqrt(sm ** 2 + ((summ["dmc_dalpha_s"]["P1=0.5"] - sc["kappa"]) * sc["s_as"]) ** 2 + sc["s_SM"] ** 2 + (sc["s_Mh"] * DMC_DMH) ** 2 + vm)
        e = dict(label=sc["label"], src=sc["src"], sigma_meas=sm, sigma_tot=stot, mhat_surv95=thr_s, mhat_excl_P0p1=thr_1, mhat_excl_P0p05=thr_05,
                 separation_172p6_173p3=0.7 / stot, truth={})
        for mt in CFG["future_true_m"]:
            st = stats(T.mc(mt, sm, sm, n=400_000, seed=3))
            e["truth"][str(mt)] = dict(Pbar=ev(T.G, mt), q025=st["q025"], q975=st["q975"], prob_P_ge_0p5=ev(H5, mt), prob_P_le_0p1=1 - ev(H1, mt),
                                       power_surv=float(ndtr((thr_s - mt) / sm)), power_excl=float(1 - ndtr((thr_1 - mt) / sm)))
            tr = e["truth"][str(mt)]
            frows.append([sc["id"], mt, sm, stot, tr["Pbar"], tr["q025"], tr["q975"], tr["prob_P_ge_0p5"], tr["prob_P_le_0p1"], tr["power_surv"],
                          tr["power_excl"], thr_s, thr_1, thr_05])
        fut[sc["id"]] = e
        print(f"future {sc['id']:24s} s_tot={stot:.3f} surv<{thr_s:.3f} excl0.1>{thr_1:.3f} excl0.05>{thr_05:.3f} "
              f"P(>=.5|172.6)={e['truth']['172.6']['prob_P_ge_0p5']:.4f} P(>=.5|173.3)={e['truth']['173.3']['prob_P_ge_0p5']:.4f}")
    summ["future"] = fut
    wcsv("future_reach.csv", ["scenario", "m_true", "sigma_meas", "sigma_tot", "Pbar", "q025", "q975", "prob_P_ge_0p5", "prob_P_le_0p1",
                              "power_establish_survival", "power_exclude_P0p1", "mhat_surv95", "mhat_excl_P0p1_95", "mhat_excl_P0p05_95"], frows)

    # ---------------- (m_t, alpha_s) plane
    th_up = np.sqrt(SMC["pole"] ** 2 + SIG_MH ** 2 + SP ** 2); th_dn = np.sqrt(SMC["pole"] ** 2 + SIG_MH ** 2 + SM ** 2)
    crow = []
    for a in np.round(np.arange(0.1150, 0.12101, 0.0001), 4):
        r = [float(a)]
        for i, p in enumerate((0.9, 0.5, 0.1, 0.05)):
            mc_ = S.mcrit(p, a) + MU_MH; j = min(i, 2); r += [mc_, mc_ - float(th_dn[j]), mc_ + float(th_up[j])]
        crow.append(r)
    hdr = ["alpha_s"] + [f"mc_P{p}{s}" for p in ("0.9", "0.5", "0.1", "0.05") for s in ("", "_th_lo1sig", "_th_hi1sig")]
    wcsv("fig2a_contours_mt_alphas.csv", hdr, crow)
    mgr = np.round(np.arange(171.0, 175.0001, 0.02), 3); grow = []
    for a in np.round(np.arange(0.1150, 0.12101, 0.0002), 4):
        P = S.P1(mgr - MU_MH, a); Tg = Theory2(S, xl, a_c=float(a), s_a=0.0, s_SM=SMC["pole"], msig=MS, lo=165.0, hi=181.0, nz=24)
        Gs = np.interp(mgr, Tg.mg, Tg.G)
        grow += [[float(mgr[i]), float(a), float(P[i]), float(Gs[i])] for i in range(len(mgr))]
    wcsv("fig2a_P1_grid_mt_alphas.csv", ["m_t", "alpha_s", "P1_central", "P1_theory_smeared"], grow)
    erow = []; tt = np.linspace(0, 2 * math.pi, 181)
    for ms in CFG["measurements"]:
        if ms.get("msbar_map"):
            continue
        c, sl, sr = meas_params(ms)
        for cl, k2 in (("68", 2.2789), ("95", 5.9915)):
            for t_ in tt:
                erow.append([ms["id"], cl, c + sl * math.sqrt(k2) * math.cos(t_), A0 + SA * math.sqrt(k2) * math.sin(t_)])
    wcsv("fig2a_ellipses.csv", ["measurement", "CL2D", "m_t_pole4L_equiv", "alpha_s"], erow)

    # ---------------- (m_t, xi_H) plane: T6 zero-mode xi_c and T21 inflation bound
    d6 = json.load(open(os.path.join(ROOT, CFG["xi"]["T6"]), encoding="utf-8"))
    d21 = json.load(open(os.path.join(ROOT, CFG["xi"]["T21"]), encoding="utf-8"))
    s1 = d6["step1_1D_xi_c"]; kk = sorted(s1, key=float); mx = np.array([float(k) for k in kk])
    xc = np.array([s1[k]["xi_c"] for k in kk]); xlo = np.array([s1[k]["spread_all"][0] for k in kk]); xhi = np.array([s1[k]["spread_all"][1] for k in kk])
    t21 = d6["step3_R15_items"]["T21"]; xew = np.array([t21[k]["xi_c_EW_image"] for k in kk])
    rows21 = [r for r in d21["xi_inf_rows"] if abs(r.get("als", 0) - 0.118) < 1e-6 and "xi_inf_ew" in r]
    rows21.sort(key=lambda r: r["mt"])
    m21 = np.array([r["mt"] for r in rows21]); iew = np.array([r["xi_inf_ew"] for r in rows21]); imb = np.array([r["xi_inf_mub"] for r in rows21])
    fc, flo, fhi, few = (PchipInterpolator(mx, v) for v in (xc, xlo, xhi, xew))
    xrow = []
    for m in np.round(np.arange(171.4, 173.6001, 0.01), 3):
        inr = mx[0] <= m <= mx[-1]
        v = [float(fc(m)), float(flo(m)), float(fhi(m)), float(few(m))] if inr else ["", "", "", ""]
        im = float(np.interp(m, m21, imb)) if m21[0] <= m <= m21[-1] else ""; ie = float(np.interp(m, m21, iew)) if m21[0] <= m <= m21[-1] else ""
        strong = ("zero-mode" if (inr and im != "" and v[0] > im) else ("inflation" if (inr and im != "") else ""))
        xrow.append([float(m), float(S.P1(np.array([m - MU_MH]), A0)[0])] + v + [im, ie, strong])
    wcsv("fig2b_xi_curves.csv", ["m_t", "P1_central", "xi_c_zero_mode_mu_b", "xi_c_lo", "xi_c_hi", "xi_c_EW_image", "xi_inf_mu_b", "xi_inf_EW",
                                  "stronger_at_mu_b"], xrow)
    keyp = {"172.60 (measured)": 172.60, "P1=0.9 point": crit["0.1180"]["P1=0.9"], "P1=0.5 point": crit["0.1180"]["P1=0.5"], "173.00": 173.0}
    summ["xi_points"] = {k: {"m_t": v, "P1": float(S.P1(np.array([v]), A0)[0]), "xi_c_mu_b": float(fc(v)), "xi_c_lo": float(flo(v)), "xi_c_hi": float(fhi(v)),
                             "xi_c_EW_image": float(few(v)), "xi_inf_mu_b": float(np.interp(v, m21, imb)), "xi_inf_EW": float(np.interp(v, m21, iew))}
                         for k, v in keyp.items()}
    summ["xi_T6_nodes"] = {k: {"xi_c": s1[k]["xi_c"], "err": s1[k]["err"], "spread": s1[k]["spread_all"], "EW_image": t21[k]["xi_c_EW_image"],
                               "xi_inf_mub": t21[k]["xi_inf_mub"], "xi_inf_ew": t21[k]["xi_inf_ew"], "stronger_same_scale": t21[k]["stronger_same_scale"]} for k in kk}
    json.dump({"T6_nodes": summ["xi_T6_nodes"], "T21_rows_alpha0p118": [[float(a), float(b), float(c)] for a, b, c in zip(m21, imb, iew)],
               "P1_verticals": {f"P1={p}": crit["0.1180"][f"P1={p}"] for p in (0.9, 0.5, 0.1)}, "lit": CFG["xi"]["lit"],
               "reading": "xi_c is xi_H(mu_b = MPbar/722) of the zero-mode Hamiltonian; EW image via Herranen Eq.8 running (T6/T21); "
                          "xi_inf from T21 (MMR2021 field-theory criterion on the project plateau), given at mu = m_t (EW) and at mu_b"},
              open(os.path.join(OUT, "fig2b_points.json"), "w", encoding="utf-8"), indent=1)
    TOUCH.append("fig2b_points.json")
    print("xi", json.dumps(summ["xi_points"]))

    # ---------------- small shift scan (robustness of the claim)
    srows = []
    for D in CFG["shift_scan"]:
        Tp = Theory2(S, xl, s_SM=SMC["pole"], mu=MU_MH + D, msig=MS); Tm = Theory2(S, xl, s_SM=SMC["msbar"], mu=MU_MH + D, msig=MS)
        r = [D]
        for mid in ("D1_direct_pole", "D1b_direct_pole_interp", "D2_sigma_pole", "D3_MSR"):
            ms = next(x for x in CFG["measurements"] if x["id"] == mid); c, sl, sr = meas_params(ms)
            T = Tp if ms["scheme"] == "pole" else Tm; r.append(integ(T, T.G, c, sl, sr))
        c, sl, sr = meas_params(CFG["measurements"][0]); L = sn_pdf(Tp.mg - c, sl, sr)
        pi = ((Tp.mg >= 170) & (Tp.mg <= 176)).astype(float); pi /= np.trapezoid(pi, Tp.mg)
        Bex = float(np.trapezoid(Tp.G * L * pi, Tp.mg) / np.trapezoid(L * pi, Tp.mg)); r += [Bex, Bex / float(np.trapezoid(Tp.G * pi, Tp.mg))]
        r += [crit["0.1180"]["P1=0.5"] + MU_MH + D, dc["lam_c_hazard_equiv"]["P1=0.5"] - 0.00656 * D, float(fc(172.60 - D)) if mx[0] <= 172.60 - D <= mx[-1] else ""]
        srows.append(r)
    wcsv("shift_scan_mc.csv", ["shift", "Pbar_D1", "Pbar_D1b", "Pbar_D2", "Pbar_D3", "B_ex_D1", "B_sel_D1", "mc05", "lambda_c05", "xi_c_172p60"], srows)

    summ["sha256"] = {"C154_core_v2.py": sha(os.path.abspath(__file__)), "C154_inputs_v2.json": sha(CFG_PATH), CFG["central_P1"]["file"]: sha(pc),
                      CFG["t1_tables"]: sha(os.path.join(ROOT, CFG["t1_tables"])), CFG["xi"]["T6"]: sha(os.path.join(ROOT, CFG["xi"]["T6"])),
                      CFG["xi"]["T21"]: sha(os.path.join(ROOT, CFG["xi"]["T21"]))}
    summ["files"] = TOUCH; summ["runtime_s"] = time.time() - t0
    json.dump(summ, open(os.path.join(OUT, "C154_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
