"""C171 analysis: collect raw results -> C171_results.json, CSV tables, checks.json, diagnostic PNG."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import math
import glob
import csv
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c171_core as C
from c171_scan import lam_list, E_SM, X_MAX, RES, RAW

B = C.B_ASYM                      # (2 pi sqrt2 / 3) * 3/2 = pi sqrt2
LN2 = math.log(2.0)


def wkb_TS(Lam, E, c2L=0.0):
    """Langer-corrected BO potential U = 1.5 r + c2L/r^2 + Lam r^4/4: classical period T(E) and barrier action S(E)."""
    U = lambda r: 1.5 * r + c2L / r ** 2 + Lam / 4.0 * r ** 4
    lm = C.bo_landmarks(Lam, E, 1.5, c2=max(c2L, 1e-12))
    if lm["above"]:
        return float("nan"), float("nan")
    rb = lm["rb"]
    r2 = brentq(lambda r: U(r) - E, 1e-12 if c2L <= 0 else lm["rmin"], rb, xtol=1e-15)
    r1 = 0.0 if c2L <= 0 else brentq(lambda r: U(r) - E, 1e-10, lm["rmin"], xtol=1e-15)
    invp = lambda r: 1.0 / math.sqrt(max(2.0 * (E - U(r)), 1e-300)) if r > 0 else 1.0 / math.sqrt(2.0 * E)
    T = 2.0 * C._int_between(invp, r1, r2)
    kap = lambda r: math.sqrt(max(2.0 * (U(r) - E), 0.0))
    S = 2.0 * C._int_between(kap, r2, lm["r3"])
    return T, S


def load(name):
    p = os.path.join(RAW, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def gam_of(run):
    return run["Gamma_flux"] if run.get("Gamma_flux") else run["Gamma_rq"]


def main():
    lams = lam_list()
    oned = load("oneD_all")
    ext = load("oneD_ext") or {}
    jnames = sorted([k for k in lams if k.startswith("j")])
    rows = []
    # ------------------------------------------------------------------ convergence study
    conv = {}
    for jn in ["j00", "j08", "j12", "j16", "j21", "j23"]:
        base = load(f"conv_{jn}_base")
        if base is None:
            continue
        lg0 = base["rich_lnG"]["R1"]
        E0 = base["rich_E"]["R1"]
        var = {}
        for f in glob.glob(os.path.join(RAW, f"conv_{jn}_*.json")):
            d = json.load(open(f, encoding="utf-8"))
            vn = os.path.basename(f)[len(f"conv_{jn}_"):-5]
            var[vn] = dict(dlnG=d["rich_lnG"]["R1"] - lg0, dE=d["rich_E"]["R1"] - E0,
                           lnG_R1=d["rich_lnG"]["R1"], E_R1=d["rich_E"]["R1"],
                           G_h=[gam_of(r) for r in d["runs"]], E_h=[r["E_re"] for r in d["runs"]],
                           grid=d["grid"])
        spread_lnG = max(abs(v["dlnG"]) for v in var.values())
        spread_E = max(abs(v["dE"]) for v in var.values())
        conv[jn] = dict(Lam=lams[jn], x=abs(lams[jn]) ** -0.5, variants=var, max_abs_dlnG=spread_lnG,
                        max_abs_dE=spread_E)
    conv_x = np.array([conv[k]["x"] for k in conv])
    conv_s = np.array([conv[k]["max_abs_dlnG"] for k in conv])
    conv_e = np.array([conv[k]["max_abs_dE"] for k in conv])

    def var_err(x):  # parameter-variation error interpolated in x (log), conservative: max of neighbours
        if len(conv_x) == 0:
            return 0.0, 0.0
        o = np.argsort(conv_x)
        xs, ss, ee = conv_x[o], conv_s[o], conv_e[o]
        i = np.searchsorted(xs, x)
        lo, hi = max(i - 1, 0), min(i, len(xs) - 1)
        return float(max(ss[lo], ss[hi])), float(max(ee[lo], ee[hi]))

    fine = {}
    for jn in ["j08", "j16"]:
        d = load(f"fine_{jn}")
        if d:
            fine[jn] = dict(hs=d["hs"], G_h=[gam_of(r) for r in d["runs"]], E_h=[r["E_re"] for r in d["runs"]],
                            rich_lnG=d["rich_lnG"], rich_E=d["rich_E"])
    # ------------------------------------------------------------------ main table
    for name in jnames + ["SM_lam-0.010", "SM_lam-0.013"]:
        d = load("main_" + name)
        if d is None:
            continue
        Lam = lams[name]
        x = abs(Lam) ** -0.5
        G_h = [gam_of(r) for r in d["runs"]]
        Grq_h = [r["Gamma_rq"] for r in d["runs"]]
        E_h = [r["E_re"] for r in d["runs"]]
        rl, re_ = d["rich_lnG"], d["rich_E"]
        lnG = rl.get("R2", rl["R1"])
        E = re_.get("R2", re_["R1"])
        verr_lnG, verr_E = var_err(x)
        err_lnG = abs(rl["R1"] - lnG) + verr_lnG + 1e-12
        err_E = abs(re_["R1"] - E) + verr_E + 1e-14
        o = oned[name]
        rec = dict(name=name, Lam=Lam, x=x, E_exact=E, E_err=err_E, Gamma_exact=math.exp(lnG),
                   Gamma_rel_err=err_lnG, lnG_exact=lnG, h=d["hs"], Gamma_h=G_h, Gamma_rq_h=Grq_h, E_h=E_h,
                   h_ratio_lnG=(math.log(G_h[0]) - math.log(G_h[1])) / (math.log(G_h[1]) - math.log(G_h[2])),
                   rq_vs_flux_rel=abs(Grq_h[-1] / G_h[-1] - 1.0), resid=d["runs"][-1]["resid"],
                   grid=d["grid"], landmarks=d["landmarks"],
                   holonomy=d["runs"][-1]["holonomy_tail"], P_well=d["runs"][-1]["P_well_over_unscaled"],
                   E_BO=o["res_BO"]["E_re"], Gamma_BO=o["res_BO"]["Gamma"],
                   E_BOD=o["res_BOD"]["E_re"], Gamma_BOD=o["res_BOD"]["Gamma"],
                   BO_shoot_reldiff=o["res_BO"]["rel_diff_alt"], BOD_shoot_reldiff=o["res_BOD"]["rel_diff_alt"],
                   WKB_BO=o["wkb_BO"], WKB_BOD=o["wkb_BOD"],
                   S_BO_at_Eexact=C.barrier_action(Lam, E, 0.0), S_asym=B * x)
        T_e, S_e_ = wkb_TS(Lam, E, 0.0)
        rec["T_BO_at_Eexact"] = T_e
        rec["lnGamma_WKB_at_Eexact"] = -S_e_ - math.log(T_e) if T_e == T_e else float("nan")
        rec["ratio_exact_WKB_at_Eexact"] = math.exp(lnG - rec["lnGamma_WKB_at_Eexact"]) if T_e == T_e else float("nan")
        rec["A_eff"] = rec["Gamma_exact"] * math.exp(B * x)
        rec["ratio_exact_BO"] = rec["Gamma_exact"] / rec["Gamma_BO"]
        rec["ratio_exact_BOD"] = rec["Gamma_exact"] / rec["Gamma_BOD"]
        rows.append(rec)
    R = {r["name"]: r for r in rows}
    J = [R[j] for j in jnames if j in R]
    xs = np.array([r["x"] for r in J])
    lnG = np.array([r["lnG_exact"] for r in J])
    lnBO = np.log([r["Gamma_BO"] for r in J])
    lnBOD = np.log([r["Gamma_BOD"] for r in J])
    b_ex = -np.gradient(lnG, xs)
    b_bo = -np.gradient(lnBO, xs)
    b_bod = -np.gradient(lnBOD, xs)
    S_e = np.array([r["S_BO_at_Eexact"] for r in J])
    dS = np.gradient(S_e, xs)
    for r, a, b_, c, s in zip(J, b_ex, b_bo, b_bod, dS):
        r["b_exact"], r["b_BO"], r["b_BOD"], r["dS_BO_at_Eexact_dx"] = float(a), float(b_), float(c), float(s)
        r["b_pred_energy_shift"] = B - 1.2601 * r["E_exact"] * r["x"] ** (-2.0 / 3.0)
    # extended 1D slopes
    ext_rows = []
    if ext:
        ex = sorted(ext.values(), key=lambda v: v["x"])
        # join with main 1D at x<=10 for derivative continuity
        xx = np.array([v["x"] for v in ex])
        for lab in ("BO", "BOD"):
            lg = np.log([v["res_" + lab]["Gamma"] for v in ex])
            bb = -np.gradient(lg, xx)
            for v, bv in zip(ex, bb):
                v["b_" + lab] = float(bv)
        for v in ex:
            ext_rows.append(dict(x=v["x"], Lam=v["Lam"], E_BO=v["res_BO"]["E_re"], Gamma_BO=v["res_BO"]["Gamma"],
                                 b_BO=v["b_BO"], E_BOD=v["res_BOD"]["E_re"], Gamma_BOD=v["res_BOD"]["Gamma"],
                                 b_BOD=v["b_BOD"],
                                 b_pred_BO=B - 1.2601 * v["res_BO"]["E_re"] * v["x"] ** (-2.0 / 3.0)))
    # ------------------------------------------------------------------ thresholds
    cs = CubicSpline(xs, lnG)
    cs_bo = CubicSpline(xs, lnBO)
    cs_bod = CubicSpline(xs, lnBOD)
    lin = lambda x: np.interp(x, xs, lnG)
    thr = []
    Ns = [10, 300, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16]
    errs = np.array([r["Gamma_rel_err"] for r in J])
    for N in Ns:
        target = math.log(LN2 / N)
        rec = dict(N=N, lnN=math.log(N), Gamma_target=LN2 / N)
        for lab, f in (("exact", cs), ("BO", cs_bo), ("BOD", cs_bod)):
            fx = lambda x: float(f(x)) - target
            if fx(xs[0]) * fx(xs[-1]) > 0:
                rec["x_c_" + lab] = None
                continue
            xc = brentq(fx, xs[0], xs[-1], xtol=1e-12)
            rec["x_c_" + lab] = xc
            rec["Lam_c_" + lab] = -1.0 / xc ** 2
            rec["sqrtLam_c_" + lab] = 1.0 / xc
        if rec.get("x_c_exact"):
            xc = rec["x_c_exact"]
            bloc = -float(cs(xc, 1))
            elnG = float(np.interp(xc, xs, errs))
            xlin = brentq(lambda x: lin(x) - target, xs[0], xs[-1])
            dx = elnG / bloc + abs(xlin - xc) * 0.1
            rec["b_local"] = bloc
            rec["x_c_err"] = dx
            rec["Lam_c_err"] = 2.0 * dx / xc ** 3
            rec["sqrtLam_c_lnN"] = (1.0 / xc) * math.log(N)
            rec["dsqrtLam_d_invlnN"] = math.log(N) ** 2 / (xc ** 2 * bloc)
        thr.append(rec)
    # thresholds for general e: e^{2/3} F(Lambda_c) N = ln 2  ->  lambda_c = e^2 Lambda_c
    thr_e = []
    for ee in (E_SM, 0.5):
        for N in (10, 300, 1e4, 1e6):
            target = math.log(LN2 / (N * ee ** (2.0 / 3.0)))
            fx = lambda x: float(cs(x)) - target
            if fx(xs[0]) * fx(xs[-1]) > 0:
                continue
            xc = brentq(fx, xs[0], xs[-1], xtol=1e-12)
            thr_e.append(dict(e=ee, N=N, Lam_c=-1.0 / xc ** 2, lam_c=-ee ** 2 / xc ** 2,
                              sigma0=1.5 * ee))
    # linear fit of |Lam_c|^{1/2} vs 1/ln N (exact) over the large-N points
    fitpts = [t for t in thr if t.get("x_c_exact") and t["N"] >= 1e4]
    u = np.array([1.0 / t["lnN"] for t in fitpts])
    v = np.array([t["sqrtLam_c_exact"] for t in fitpts])
    p1 = np.polyfit(u, v, 1)
    p2 = np.polyfit(u, v, 2)
    fit = dict(points_N=[t["N"] for t in fitpts], linear=dict(slope=p1[0], intercept=p1[1],
               max_resid=float(np.max(abs(np.polyval(p1, u) - v)))),
               quadratic=dict(coef=p2.tolist()))
    # asymptotic regime: WKB (Langer BO) at the exact energy; eps(x) extrapolated with eps0 + c x^-2 fit
    xfit = xs[-6:]
    efit = np.array([r["E_exact"] for r in J])[-6:]
    pe = np.polyfit(xfit ** -2.0, efit, 1)
    eps0 = float(pe[1])
    eps_of = lambda x: float(np.interp(x, xs, [r["E_exact"] for r in J])) if x <= xs[-1] else float(np.polyval(pe, x ** -2.0))
    xa = np.geomspace(2.0, 2000.0, 61)
    lnGw = []
    for x in xa:
        T_, S_ = wkb_TS(-1.0 / x ** 2, eps_of(x), 0.0)
        lnGw.append(-S_ - math.log(T_))
    lnGw = np.array(lnGw)
    b_w = -np.gradient(lnGw, xa)
    wkb_asym = dict(eps0=eps0, eps_fit=pe.tolist(), x=xa.tolist(), lnGamma_WKB=lnGw.tolist(), b_WKB=b_w.tolist(),
                    b_pred=(B - 1.2601 * np.array([eps_of(x) for x in xa]) * xa ** (-2.0 / 3.0)).tolist())
    ratio_tail = J[-1]["ratio_exact_WKB_at_Eexact"]
    # large-N thresholds from Gamma_asym(x) = ratio_tail * Gamma_WKB(eps(x); x) (anchored at the smallest |Lambda|)
    csw = CubicSpline(xa, lnGw + math.log(ratio_tail))
    asym_thr = []
    for N in [1e6, 1e10, 1e16, 1e20, 1e30, 1e50, 1e100, 1e200, 1e300]:
        target = math.log(LN2 / N)
        fx = lambda x: float(csw(x)) - target
        if fx(xa[0]) * fx(xa[-1]) > 0:
            continue
        xc = brentq(fx, xa[0], xa[-1])
        bl = -float(csw(xc, 1))       # d|Lam_c|^{1/2}/d(1/ln N) = (ln N)^2 / (x_c^2 b_local)
        asym_thr.append(dict(N=N, x_c=xc, sqrtLam_c=1.0 / xc, sqrtLam_c_lnN=math.log(N) / xc,
                             b_local=bl, dsqrtLam_d_invlnN=math.log(N) ** 2 / (xc ** 2 * bl)))
    # ------------------------------------------------------------------ SM-like and direct scaling checks
    sm = {}
    for lab, lam_phys in (("lam-0.010", -0.010), ("lam-0.013", -0.013)):
        r = R.get("SM_" + lab)
        dd = load(f"direct_e{E_SM:.4f}_{lab}")
        if r is None:
            continue
        rec = dict(e=E_SM, sigma0=1.5 * E_SM, lam=lam_phys, Lam=r["Lam"], x=r["x"],
                   F=r["Gamma_exact"], F_rel_err=r["Gamma_rel_err"],
                   Gamma_phys=E_SM ** (2.0 / 3.0) * r["Gamma_exact"], E_phys=E_SM ** (2.0 / 3.0) * r["E_exact"],
                   Gamma_BO_phys=E_SM ** (2.0 / 3.0) * r["Gamma_BO"], Gamma_BOD_phys=E_SM ** (2.0 / 3.0) * r["Gamma_BOD"],
                   WKB_asym_exponent=B * r["x"], holonomy_scaled=r["holonomy"])
        if dd:
            Gd = math.exp(dd["rich_lnG"].get("R2", dd["rich_lnG"]["R1"]))
            rec["Gamma_direct"] = Gd
            rec["E_direct"] = dd["rich_E"].get("R2", dd["rich_E"]["R1"])
            rec["scaling_rel_diff"] = abs(Gd / rec["Gamma_phys"] - 1.0)
            rec["holonomy_direct"] = dd["runs"][-1]["holonomy_tail"]
            for N in (10, 300, 1e4, 1e6):
                rec[f"survival_N{N:g}"] = math.exp(-Gd * N)
        sm[lab] = rec
    d05 = load("direct_e0.5_j12")
    scal05 = None
    if d05 and "j12" in R:
        Gd = math.exp(d05["rich_lnG"].get("R2", d05["rich_lnG"]["R1"]))
        scal05 = dict(e=0.5, lam=d05["Lam"], Lam=R["j12"]["Lam"], Gamma_direct=Gd,
                      Gamma_scaled=0.5 ** (2.0 / 3.0) * R["j12"]["Gamma_exact"],
                      rel_diff=abs(Gd / (0.5 ** (2.0 / 3.0) * R["j12"]["Gamma_exact"]) - 1.0),
                      E_direct=d05["rich_E"].get("R2", d05["rich_E"]["R1"]),
                      E_scaled=0.5 ** (2.0 / 3.0) * R["j12"]["E_exact"],
                      holonomy_direct=d05["runs"][-1]["holonomy_tail"])
    # ------------------------------------------------------------------ CAP cross-check
    cap = []
    for f in sorted(glob.glob(os.path.join(RAW, "cap_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        jn = d["name"].split("_")[1]
        r = R.get(jn)
        if r is None:
            continue
        cap.append(dict(name=d["name"], Lam=d["Lam"], x=r["x"], h=d["h"], Nj=d["Nj"], Rc=d["Rc"], rmax=d["rmax"],
                        Gamma_cap0=d["Gamma0"], Gamma_cap1=d["Gamma1"], Gamma_cap_err1=d["Gamma_err1"],
                        E_cap1=d["E1"][0], E_cap0=d["E0"][0], eta_opt1=d["eta_opt1"],
                        Gamma_ecs=r["Gamma_exact"], E_ecs=r["E_exact"],
                        rel_diff_G1=abs(d["Gamma1"] / r["Gamma_exact"] - 1.0),
                        rel_diff_G0=abs(d["Gamma0"] / r["Gamma_exact"] - 1.0),
                        dE1=d["E1"][0] - r["E_exact"]))
    # ------------------------------------------------------------------ e -> 0: inverted quartic alone
    nogauge = None
    try:
        Lam = -0.03
        g = abs(Lam) / 4.0
        E0 = complex(1.0, -1.7) * g ** (1.0 / 3.0)
        tmax = C.ray_tmax(Lam, 1.0, 1.0, 0.5, 30.0, sigma0=0.0)
        f = lambda E: C.shoot_1d(E, Lam, -0.125, 1.0, 0.5, tmax, 0.05, 0.0, 1e-11)[0]
        E = E0
        for _ in range(40):
            fE = f(E)
            dE = -fE / ((f(E + 1e-6) - fE) / 1e-6)
            E += dE
            if abs(dE) < 1e-12:
                break
        nogauge = dict(Lam=Lam, E_re=E.real, E_im=E.imag, arg_deg=math.degrees(math.atan2(E.imag, E.real)),
                       Gamma_over_ReE=-2 * E.imag / E.real)
    except Exception as exc:
        nogauge = dict(error=str(exc))
    # ------------------------------------------------------------------ checks
    checks = []

    def chk(name, ok, **kw):
        checks.append(dict(check=name, passed=bool(ok), **kw))

    nar = [r for r in J if r["Gamma_exact"] > 1e-11]
    chk("flux_vs_rayleigh_gamma_agree", max(r["rq_vs_flux_rel"] for r in nar) < 1e-4,
        max_rel=max(r["rq_vs_flux_rel"] for r in nar), note="Gamma > 1e-11; below that Im E(RQ) hits the 1e-15 floor")
    chk("rayleigh_floor_below_1e-11", True,
        values={r["name"]: r["rq_vs_flux_rel"] for r in J if r["Gamma_exact"] <= 1e-11})
    hr = [r["h_ratio_lnG"] for r in J]
    chk("h2_convergence_ratio_near_4", all(3.3 < q < 4.7 for q in hr), ratios=hr)
    chk("parameter_variation_dlnG_small", max(conv_s) < 2e-4 if len(conv_s) else False,
        max_abs_dlnG={k: conv[k]["max_abs_dlnG"] for k in conv}, max_abs_dE={k: conv[k]["max_abs_dE"] for k in conv})
    chk("residuals_small", max(r["resid"] for r in J) < 1e-10, max_resid=max(r["resid"] for r in J))
    chk("BO_bounds_E_BO<E_exact<E_BOD", all(r["E_BO"] < r["E_exact"] < r["E_BOD"] for r in J if not r["landmarks"]["above"]),
        detail=[[r["x"], r["E_BO"], r["E_exact"], r["E_BOD"]] for r in J])
    sel1d = [(r["name"], r["Gamma_BO"], r["BO_shoot_reldiff"], r["Gamma_BOD"], r["BOD_shoot_reldiff"]) for r in J]
    big = [max(q[2] if q[1] > 1e-14 else 0.0, q[4] if q[3] > 1e-14 else 0.0) for q in sel1d]
    chk("1D_shooting_alt_params_agree_Gamma>1e-14", max(big) < 1e-7, max_rel=max(big),
        below_1em14={q[0]: [q[1], q[2], q[3], q[4]] for q in sel1d if q[1] <= 1e-14 or q[3] <= 1e-14},
        note="1D Newton/shooting floor ~1e-21 absolute in Gamma (real-part rounding x derivative error)")
    if fine:
        fr = {}
        for jn, fd in fine.items():
            fr[jn] = dict(R1_fine=fd["rich_lnG"]["R1"], R2_main=R[jn]["lnG_exact"],
                          dlnG=fd["rich_lnG"]["R1"] - R[jn]["lnG_exact"], dE=fd["rich_E"]["R1"] - R[jn]["E_exact"])
        chk("finest_h_0.0025_consistent", all(abs(q["dlnG"]) < 1e-5 for q in fr.values()), detail=fr)
    if cap:
        chk("CAP_vs_ECS_gamma", all(c["rel_diff_G1"] < max(3e-3, 3 * c["Gamma_cap_err1"] / c["Gamma_ecs"]) for c in cap),
            detail=[[c["name"], c["Gamma_ecs"], c["Gamma_cap1"], c["rel_diff_G1"], c["Gamma_cap_err1"], c["dE1"]] for c in cap])
    for lab, rec in sm.items():
        if "scaling_rel_diff" in rec:
            chk(f"e_scaling_direct_e{E_SM:.4f}_{lab}", rec["scaling_rel_diff"] < 1e-4, rel_diff=rec["scaling_rel_diff"])
    if scal05:
        chk("e_scaling_direct_e0.5", scal05["rel_diff"] < 1e-4, rel_diff=scal05["rel_diff"])
    if nogauge and "arg_deg" in nogauge:
        chk("no_gauge_modes_no_narrow_resonance", abs(nogauge["arg_deg"] + 60.0) < 1e-6 and nogauge["Gamma_over_ReE"] > 3,
            **nogauge)
    # ------------------------------------------------------------------ write outputs
    out = dict(
        task="abelian Higgs zero-mode resonance (exact 2D)", run="C171",
        units="e = 1 scaled units (lengths e^{-1/3}/L, energies e^{2/3}/L, L = 1); Lambda = lambda/e^2; "
              "general e: Gamma = e^{2/3} F(Lambda), E = e^{2/3} eps(Lambda)",
        model="H = -1/2 Lap_2D(r) - 1/2 Lap_3D(w) + 1/2 e^2 r^2 s^2 + (lambda/4) r^4, m=0, s-wave",
        B_asym=B,
        method_A="psi-form 2D-radial FD (2nd order, h = 0.02/0.01/0.005 x e^{-1/3}, Richardson on ln Gamma), smooth "
                 "exterior complex scaling in r (theta=0.45, transition 0.6), sine DVR in s (S=10, ds=0.14), "
                 "ARPACK shift-invert with exact block-tridiagonal LU; Gamma from the discrete flux identity",
        method_B="CAP -i eta ((r-Rc)/(rmax-Rc))^2, 4th-order FD in r, odd-HO basis in s (w0=2.5), eta-trajectory "
                 "(Riss-Meyer first-order corrected)",
        table=rows, convergence=conv, finest_h=fine, thresholds=thr, thresholds_general_e=thr_e, threshold_fit=fit,
        thresholds_asymptotic_BO_scaled=asym_thr, SM_like=sm, scaling_e05=scal05, cap=cap,
        oneD_extended=ext_rows, wkb_asymptotic=wkb_asym, no_gauge_modes=nogauge)
    with open(os.path.join(RES, "C171_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    with open(os.path.join(RES, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    # CSV: Gamma table
    with open(os.path.join(RES, "C171_gamma_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "Lambda", "x=|Lambda|^-1/2", "eps_exact", "eps_err", "Gamma_exact", "Gamma_rel_err",
                    "Gamma_h0.02", "Gamma_h0.01", "Gamma_h0.005", "eps_BO", "Gamma_BO", "eps_BOD", "Gamma_BOD",
                    "Gamma_exact/Gamma_BO", "Gamma_exact/Gamma_BOD", "E_WKB_BOD(BS)", "Gamma_WKB_BOD_Kemble",
                    "E_WKB_BO(BS)", "Gamma_WKB_BO_Kemble", "S_asym=4.443x", "exp(-S_asym)", "A_eff=Gamma*exp(S_asym)",
                    "S_BO(eps_exact)", "b_exact", "b_BO", "b_BOD", "dS_BO(eps_exact)/dx", "barrier_top_U_BOD",
                    "above_barrier", "Gamma_WKB_BO_at_eps_exact=exp(-S)/T", "Gamma_exact/Gamma_WKB_BO_at_eps_exact"])
        for r in rows:
            w.writerow([r["name"], f"{r['Lam']:.8f}", f"{r['x']:.6f}", f"{r['E_exact']:.10f}", f"{r['E_err']:.1e}",
                        f"{r['Gamma_exact']:.8e}", f"{r['Gamma_rel_err']:.1e}"] +
                       [f"{g:.8e}" for g in r["Gamma_h"]] +
                       [f"{r['E_BO']:.10f}", f"{r['Gamma_BO']:.8e}", f"{r['E_BOD']:.10f}", f"{r['Gamma_BOD']:.8e}",
                        f"{r['ratio_exact_BO']:.6f}", f"{r['ratio_exact_BOD']:.6f}",
                        f"{r['WKB_BOD']['E_BS']:.6f}", f"{r['WKB_BOD']['Gamma_Kemble']:.6e}",
                        f"{r['WKB_BO']['E_BS']:.6f}", f"{r['WKB_BO']['Gamma_Kemble']:.6e}",
                        f"{r['S_asym']:.6f}", f"{math.exp(-r['S_asym']):.6e}", f"{r['A_eff']:.6e}",
                        f"{r['S_BO_at_Eexact']:.6f}", f"{r.get('b_exact', float('nan')):.5f}",
                        f"{r.get('b_BO', float('nan')):.5f}", f"{r.get('b_BOD', float('nan')):.5f}",
                        f"{r.get('dS_BO_at_Eexact_dx', float('nan')):.5f}", f"{r['landmarks']['Umax']:.6f}",
                        int(r["landmarks"]["above"]), f"{math.exp(r['lnGamma_WKB_at_Eexact']):.6e}",
                        f"{r['ratio_exact_WKB_at_Eexact']:.5f}"])
    with open(os.path.join(RES, "C171_thresholds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "Gamma_target=ln2/N", "Lambda_c_exact", "Lambda_c_err", "|Lambda_c|^1/2", "1/lnN",
                    "|Lambda_c|^1/2*lnN", "b_local", "Lambda_c_BO", "Lambda_c_BOD"])
        for t in thr:
            if not t.get("x_c_exact"):
                continue
            w.writerow([f"{t['N']:g}", f"{t['Gamma_target']:.6e}", f"{t['Lam_c_exact']:.7f}", f"{t['Lam_c_err']:.1e}",
                        f"{t['sqrtLam_c_exact']:.7f}", f"{1/t['lnN']:.6f}", f"{t['sqrtLam_c_lnN']:.5f}",
                        f"{t['b_local']:.4f}",
                        f"{t.get('Lam_c_BO', float('nan')):.7f}", f"{t.get('Lam_c_BOD', float('nan')):.7f}"])
    with open(os.path.join(RES, "C171_holonomy_tails.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = list(J[0]["holonomy"].keys())
        w.writerow(["name", "Lambda", "x"] + keys)
        for r in rows:
            w.writerow([r["name"], f"{r['Lam']:.8f}", f"{r['x']:.5f}"] + [f"{r['holonomy'][k]:.4e}" for k in keys])
    if ext_rows:
        with open(os.path.join(RES, "C171_oneD_extended.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(ext_rows[0].keys()))
            for r in ext_rows:
                w.writerow([f"{v:.10g}" for v in r.values()])
    if cap:
        with open(os.path.join(RES, "C171_cap_crosscheck.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(cap[0].keys()))
            for c in cap:
                w.writerow([c[k] if isinstance(c[k], str) else f"{c[k]:.10g}" for k in c])
    # ------------------------------------------------------------------ plot
    make_plot(J, wkb_asym, os.path.join(RES, "C171_lnGamma_vs_x.png"))
    print(json.dumps([[c["check"], c["passed"]] for c in checks], indent=0))
    for r in J:
        print(f"{r['name']} x={r['x']:.3f} eps={r['E_exact']:.7f} G={r['Gamma_exact']:.6e} (+-{r['Gamma_rel_err']:.1e}) "
              f"BO={r['Gamma_BO']:.3e} BOD={r['Gamma_BOD']:.3e} b={r['b_exact']:.3f} bBO={r['b_BO']:.3f}")
    for t in thr:
        if t.get("x_c_exact"):
            print(f"N={t['N']:g} Lam_c={t['Lam_c_exact']:.6f}+-{t['Lam_c_err']:.1e} sqrt*lnN={t['sqrtLam_c_lnN']:.4f} "
                  f"BO={t.get('Lam_c_BO')} BOD={t.get('Lam_c_BOD')}")


def make_plot(J, wa, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.array([r["x"] for r in J])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    # resonance energy above the (Langer) BO barrier top 1.125 (1.5 x^2)^{1/3} for x < x_top
    eps = np.array([r["E_exact"] for r in J])
    gap = eps - 1.125 * (1.5 * xs ** 2) ** (1.0 / 3.0)
    x_top = float(np.interp(0.0, -gap, xs))
    for a in ax:
        a.axvspan(0.9 if a is ax[0] else 1.0, x_top, color="0.92", lw=0)
    ax[1].text(1.24, 2.15, "above barrier top", fontsize=7, color="0.35", rotation=90, ha="center", va="bottom")
    a = ax[0]
    a.plot(xs, np.log([r["Gamma_BO"] for r in J]), "-", color="#1f77b4", lw=1.4, label="1D BO")
    a.plot(xs, np.log([r["Gamma_BOD"] for r in J]), "--", color="#d62728", lw=1.4, label="1D BO + diag. corr.")
    a.plot(xs, [r["lnGamma_WKB_at_Eexact"] for r in J], ":", color="#2ca02c", lw=1.8,
           label=r"WKB (BO barrier, exact $\varepsilon$)")
    a.plot(xs, [r["lnG_exact"] for r in J], "o", ms=4, color="k", label="exact 2D")
    x0, y0 = xs[-1], J[-1]["lnG_exact"]
    a.plot(xs, y0 - B * (xs - x0), "-", color="0.6", lw=0.9, label=r"slope $-\pi\sqrt{2}$")
    a.set_xlabel(r"$x=|\Lambda|^{-1/2}$")
    a.set_ylabel(r"$\ln\,(\Gamma\, e^{-2/3} L)$")
    a.set_ylim(y0 - 4, 3.0)
    a.set_xlim(0.9, xs[-1] + 0.4)
    a.legend(fontsize=7.5, loc="lower left", frameon=False)
    a = ax[1]
    a.plot(xs, [r["b_exact"] for r in J], "o", ms=4, color="k", label="exact 2D")
    a.plot(xs, [r["b_BO"] for r in J], "-", color="#1f77b4", lw=1.4, label="1D BO")
    a.plot(xs, [r["b_BOD"] for r in J], "--", color="#d62728", lw=1.4, label="1D BO + diag. corr.")
    a.plot(wa["x"], wa["b_WKB"], ":", color="#2ca02c", lw=1.8, label=r"WKB, exact $\varepsilon$")
    a.axhline(B, color="0.5", lw=0.9)
    a.text(1.1, B + 0.07, r"$\pi\sqrt{2}=4.443$", fontsize=8, color="0.3")
    a.set_xscale("log")
    a.set_xlim(1.0, 2000.0)
    a.set_ylim(0.0, 4.9)
    a.set_xlabel(r"$x=|\Lambda|^{-1/2}$")
    a.set_ylabel(r"local slope $b=-d\ln\Gamma/dx$")
    a.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
