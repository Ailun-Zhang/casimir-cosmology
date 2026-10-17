# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/C154_correction_v2c.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations\code\20261014_claude_v12_r3\claim_positioning\C154_correction_v2c.py on 2026-10-15 by claims-run05 (C161); edits: '"task": "D126 v2 correction addendum (C154 run02 *_corr)", "' -> '"task": "C161 claims-run05: D126 v2 correction addendum reru'; '"C154_inputs_v2.json": V.sha(V.CFG_PATH)' -> 'os.path.basename(V.CFG_PATH): V.sha(V.CFG_PATH)'
# C154_correction_v2c.py -- correction addendum to D126 v2 (C154 run02), agent claim-positioning, 2026-10-15
# PROVENANCE: imports C154_core_v2.py (unmodified) and C154_inputs_v2.json; writes NEW *_corr files into run02
#             (run02 v2 files are kept). Responds to V3 blind diagnosis (R16 claims_robustness_blind DIAGNOSIS.json):
# (1) one asymmetric-error convention everywhere: equal-mass split (median = central value, quoted +-sigma = 16/84 %);
#     continuous two-piece normal reported as convention sensitivity;
# (2) Bayes priors on a wide grid [140, 210] GeV (v2 truncated flat_150_200 to the theory grid [160, 186]);
# (3) HL-LHC reach with dm_t = 0.17 GeV (YR 1902.04070 Table 47, pdf p.149, per V3) and alpha_s premises
#     0.0009 (unchanged) / 0.00045 (lattice x2, FCC FSR pdf p.97) / 0.0002 (YR pdf p.88 lattice assumption).
"""[v12 package, 2026-10-15] D126 v2 correction addendum (equal-mass split, prior-support Bayes, HL-LHC corrected inputs) -> *_corr files and C154_correction_summary.json.  Paper: PRL P1bar = 0.73; PRD H4."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os, sys, json, math, time, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import C154_core_v2 as V
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

CFG, ROOT, OUT, H, LEV = V.CFG, V.ROOT, V.OUT, V.H, V.LEV
CONT = V.sn_pdf


def eq_pdf(x, s_left, s_right):
    s = np.where(x < 0, s_left, s_right)
    return np.exp(-0.5 * (x / s) ** 2) / (math.sqrt(2 * math.pi) * s)


def set_conv(c):
    V.sn_pdf = eq_pdf if c == "equal_mass" else CONT


def wcsv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header)
        for r in rows:
            w.writerow([("%.6g" % x) if isinstance(x, float) else x for x in r])


def build_surface():
    dc = json.load(open(os.path.join(ROOT, CFG["central_P1"]["file"]), encoding="utf-8"))
    g = dc["P1_grid"]; ks = sorted(g, key=float)
    mref = np.array([float(k) for k in ks]); yref = np.log(-np.log(np.array([g[k] for k in ks])))
    cr = {a: np.array([dc["crit"][a]["with_cross"][f"P1={p}"] for p in LEV]) for a in dc["crit"]}
    cref = cr["0.1180"]; tabs = {}
    for a, ck in cr.items():
        xr = cref - cref[1]; xk = ck - ck[1]; cf = np.polyfit(xr, xk, 2); dcf = np.polyder(cf)
        x = mref - cref[1]; q = np.polyval(cf, x)
        q = np.where(x < xr[0], np.polyval(cf, xr[0]) + np.polyval(dcf, xr[0]) * (x - xr[0]), q)
        q = np.where(x > xr[2], np.polyval(cf, xr[2]) + np.polyval(dcf, xr[2]) * (x - xr[2]), q)
        tabs[float(a)] = dict(m=ck[1] + q, lnI=yref)
    S = V.Surface(tabs); xl = [S.mcrit(p, 0.118) - S.mcrit(0.5, 0.118) for p in LEV]
    t1 = json.load(open(os.path.join(ROOT, CFG["t1_tables"]), encoding="utf-8"))["P1_table"]["0.1180"]
    k1 = sorted(t1, key=float)
    lf = np.polyfit(np.array([float(k) for k in k1]) - 172.6, np.array([t1[k]["lam_b"] for k in k1]), 2)
    return S, xl, lf, dc


def make_T(S, xl, conv, **kw):
    set_conv(conv)
    T = V.Theory2(S, xl, **kw)
    if conv == "continuous":                       # model nuisance: side probability from P1 = 0.5 widths
        pp = T.sp[1] / (T.sp[1] + T.sm[1])
        T.wz = np.where(T.tz > 0, 2 * pp * T.wz, 2 * (1 - pp) * T.wz)
        G = np.zeros_like(T.mg)
        for ti, wi in zip(T.ta, T.wa):
            a = T.a_c + T.s_a * ti
            Fa = T.S.P1(T.mg + T.kappa * (a - T.a_c), a)
            Ga = V.conv(Fa, T.k); xs = T.mg + T.kappa * (a - T.a_c) - float(T.S.mc(a))
            spx = np.interp(xs, T.xl, T.sp); smx = np.interp(xs, T.xl, T.sm)
            dtx = np.interp(xs, T.xl, T.det) if T.det is not None else 0.0
            for zj, vj in zip(T.tz, T.wz):
                G += wi * vj * np.interp(T.mg - (zj * (spx if zj > 0 else smx) + dtx), T.mg, Ga)
        T.G = G; T._H = {}
    return T


def mc_eq(T, c, sl, sr, n=1_000_000, seed=20261015):
    """equal-mass measurement sampling + kernel sampling (same as Theory2.mc otherwise)"""
    rng = np.random.default_rng(seed)
    z = np.abs(rng.standard_normal(n)); side = rng.random(n) < 0.5
    m = np.where(side, c - sl * z, c + sr * z)
    a = T.a_c + T.s_a * rng.standard_normal(n); dg = np.interp(rng.random(n), T.cdf, T.d); Z = rng.standard_normal(n)
    m0 = m + T.kappa * (a - T.a_c) - dg; xs = m0 - T.S.mc(a)
    sig = np.where(Z > 0, np.interp(xs, T.xl, T.sp), np.interp(xs, T.xl, T.sm))
    return T.S.P1(m0 - Z * sig, a)


def main():
    t0 = time.time(); S, xl, lf, dc = build_surface()
    SP, SM = V.composite(CFG["model_components"]); MS = (SP, SM); SMC = CFG["sm_mapping"]
    dmc_da = (S.mcrit(0.5, 0.1189) - S.mcrit(0.5, 0.1171)) / 0.0018
    out = {"task": "C161 claims-run05: D126 v2 correction addendum rerun (*_corr)", "agent": "claims-run05", "date": "2026-10-15",
           "convention_chosen": "equal_mass", "dmc_dalpha_s": dmc_da}
    # ---- (1) main table + BKPV under both conventions
    res = {}
    for conv in ("equal_mass", "continuous"):
        th = {"pole": make_T(S, xl, conv, s_SM=SMC["pole"], msig=MS), "msbar": make_T(S, xl, conv, s_SM=SMC["msbar"], msig=MS)}
        bk = make_T(S, xl, conv, sm_split=(SMC["BKPV_FJ"]["plus"], SMC["BKPV_FJ"]["minus"]), msig=MS)
        set_conv(conv); r = {}
        for ms in CFG["measurements"]:
            c, sl, sr = V.meas_params(ms); T = th[ms["scheme"]]
            e = {"Pbar": V.integ(T, T.G, c, sl, sr), "prob_P_ge_0p5": V.integ(T, T.Hp(0.5), c, sl, sr)}
            if ms["scheme"] == "pole":
                e["Pbar_BKPV_FJ"] = V.integ(bk, bk.G, c, sl, sr)
            if conv == "equal_mass":
                q = np.quantile(mc_eq(T, c, sl, sr), [0.025, 0.16, 0.5, 0.84, 0.975]); e["quantiles_025_16_50_84_975"] = q.tolist()
            r[ms["id"]] = e
        res[conv] = r
        print(conv, {k: (round(v["Pbar"], 4), round(v.get("Pbar_BKPV_FJ", float("nan")), 4)) for k, v in r.items()})
    out["marginal"] = res
    rows = []
    for ms in CFG["measurements"]:
        a, b = res["equal_mass"][ms["id"]], res["continuous"][ms["id"]]; q = a["quantiles_025_16_50_84_975"]
        rows.append([ms["id"], a["Pbar"], q[2], q[1], q[3], q[0], q[4], a["prob_P_ge_0p5"], a.get("Pbar_BKPV_FJ", ""), b["Pbar"], b.get("Pbar_BKPV_FJ", "")])
    wcsv("marginal_P1_corr.csv", ["id", "Pbar_equal_mass", "median", "q16", "q84", "q025", "q975", "prob_P_ge_0p5", "Pbar_BKPV_FJ_equal_mass",
                                  "Pbar_continuous_all", "Pbar_BKPV_FJ_continuous_all"], rows)
    # ---- (2) Bayes on a wide grid
    set_conv("equal_mass")
    thw = {"pole": make_T(S, xl, "equal_mass", s_SM=SMC["pole"], msig=MS, lo=140.0, hi=210.0),
           "msbar": make_T(S, xl, "equal_mass", s_SM=SMC["msbar"], msig=MS, lo=140.0, hi=210.0)}
    set_conv("equal_mass"); brows = []; bay = {}
    for ms in CFG["measurements"]:
        c, sl, sr = V.meas_params(ms); T = thw[ms["scheme"]]; L = V.sn_pdf(T.mg - c, sl, sr); bay[ms["id"]] = {}
        for pr in CFG["priors"]:
            if pr["type"] == "flat_m":
                pi = ((T.mg >= pr["lo"]) & (T.mg <= pr["hi"])).astype(float)
            else:
                lam = np.polyval(lf, T.mg - 172.6); pi = np.abs(np.polyval(np.polyder(lf), T.mg - 172.6)) * ((lam >= pr["lo"]) & (lam <= pr["hi"]))
            pi = pi / np.trapezoid(pi, T.mg)
            Bex = float(np.trapezoid(T.G * L * pi, T.mg) / np.trapezoid(L * pi, T.mg)); pm = float(np.trapezoid(T.G * pi, T.mg))
            sup = [float(T.mg[pi > 0][0]), float(T.mg[pi > 0][-1])]
            bay[ms["id"]][pr["id"]] = dict(B_existence=Bex, prior_mean_P1=pm, B_selection=Bex / pm, B_selection_max=1 / pm, prior_support=sup)
            brows.append([ms["id"], pr["id"], Bex, pm, Bex / pm, 1 / pm, sup[0], sup[1]])
    out["bayes"] = bay
    wcsv("bayes_factors_corr.csv", ["measurement", "prior", "B_existence", "prior_mean_P1", "B_selection", "B_selection_max", "support_lo", "support_hi"], brows)
    print("bayes D1", {k: (round(v["B_existence"], 4), round(v["B_selection"], 4), round(v["B_selection_max"], 3)) for k, v in bay["D1_direct_pole"].items()})
    # ---- (3) HL-LHC
    vm = 0.5 * (SP[1] ** 2 + SM[1] ** 2); frows = []; fut = {}
    for s_as, tag in ((0.0009, "alpha_s unchanged"), (0.00045, "lattice x2 (FCC FSR pdf p.97)"), (0.0002, "YR pdf p.88 lattice assumption")):
        for extra in ([], [0.5]):
            sm = math.sqrt(0.17 ** 2 + sum(e * e for e in extra))
            T = make_T(S, xl, "equal_mass", s_a=s_as, s_SM=SMC["pole"], s_Mh=0.03 * V.DMC_DMH, msig=MS); set_conv("equal_mass")
            H5, H1 = T.Hp(0.5), T.Hp(0.1)
            thr_s = brentq(lambda cc: V.integ(T, H5, cc, sm, sm) - 0.95, 168.0, 178.0)
            thr_1 = brentq(lambda cc: 1 - V.integ(T, H1, cc, sm, sm) - 0.95, 168.0, 182.0)
            stot = math.sqrt(sm ** 2 + (dmc_da * s_as) ** 2 + SMC["pole"] ** 2 + (0.03 * V.DMC_DMH) ** 2 + vm)
            key = f"HLLHC_0.17_as{s_as}" + ("_interp0.5" if extra else "")
            e = dict(premise=tag, sigma_meas=sm, sigma_tot=stot, separation_172p6_173p3=0.7 / stot, mhat_surv95=thr_s, mhat_excl_P0p1=thr_1,
                     Pbar_172p6=V.integ(T, T.G, 172.6, sm, sm), Pbar_173p3=V.integ(T, T.G, 173.3, sm, sm),
                     Pge05_172p6=V.integ(T, H5, 172.6, sm, sm), Pge05_173p3=V.integ(T, H5, 173.3, sm, sm))
            fut[key] = e
            frows.append([key, tag, sm, s_as, stot, e["separation_172p6_173p3"], thr_s, thr_1, e["Pbar_172p6"], e["Pbar_173p3"], e["Pge05_172p6"], e["Pge05_173p3"]])
            print(f"{key:28s} s_tot={stot:.4f} D={0.7 / stot:.3f} surv<{thr_s:.3f} excl>{thr_1:.3f} P(>=.5|172.6)={e['Pge05_172p6']:.3f} |173.3={e['Pge05_173p3']:.3f}")
    out["HLLHC"] = fut
    wcsv("future_HLLHC_corr.csv", ["scenario", "alpha_s_premise", "sigma_meas", "delta_alpha_s", "sigma_tot", "separation_sigma", "mhat_surv95",
                                   "mhat_excl_P0p1_95", "Pbar_true172p6", "Pbar_true173p3", "P_ge_0p5_true172p6", "P_ge_0p5_true173p3"], frows)
    out["sha256"] = {"C154_correction_v2c.py": V.sha(os.path.abspath(__file__)), "C154_core_v2.py": V.sha(os.path.join(HERE, "C154_core_v2.py")),
                     os.path.basename(V.CFG_PATH): V.sha(V.CFG_PATH)}
    out["files"] = ["marginal_P1_corr.csv", "bayes_factors_corr.csv", "future_HLLHC_corr.csv", "C154_correction_summary.json"]
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(OUT, "C154_correction_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("done", out["runtime_s"])


if __name__ == "__main__":
    main()
