# -*- coding: utf-8 -*-
"""C170 post-processing: S1 tables, thresholds lambda_c(N; P), fits, N_eff, lambda_SM / N* per box, lambda_SM(mu) grid;
S2 lambda_c(kappa); S3 coupling-scan thresholds and logarithmic derivatives.  Writes the CSV tables and the per-step JSON
summaries (S1/C170_S1_summary.json, S2/..., S3/...); c170_final.py assembles C170_results.json and checks.json.
Usage: python -B c170_post.py s1 | lamsm | s2 | s3 | all
"""
import os
import sys
import math
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq, curve_fit  # noqa: E402

NLIST = [10, 30, 100, 300, 1e3, 1e4, 1e5, 1e6]
PLIST = {"P=0.9": 0.9, "P=0.5": 0.5, "P=0.1": 0.1}
HAZ = {"P=0.9": -0.009133908342503662, "P=0.5": -0.012141512452939195, "P=0.1": -0.015184854005126681}  # run05 lam_c
HAZ_I_OVER = {"P=0.9": 0.77, "P=0.5": 0.82, "P=0.1": 0.87}                                             # report c07
GOOD = CC.GOOD


def load_chains(pattern):
    pts = {}
    dup = []
    for fn in sorted(glob.glob(pattern)):
        if fn.endswith(".pruned.json"):
            continue
        d = CC.jload(fn)
        for p in d["points"]:
            p["status"] = CC.classify(p)
            k = round(p["pt"]["lam"], 12)
            if k in pts:
                q = pts[k]
                if p.get("gamma") and q.get("gamma"):
                    dup.append(dict(lam=p["pt"]["lam"], rel_gamma=p["gamma"] / q["gamma"] - 1, d_eps=p["eps"] - q["eps"]))
                continue
            p["_chain"] = os.path.basename(fn)
            pts[k] = p
    return [pts[k] for k in sorted(pts)], dup


def interp_table(points):
    """GOOD points sorted by x = |lambda|^(-1/2) -> arrays and spline of ln Gamma(x)."""
    g = [p for p in points if p.get("status") in GOOD and p["pt"]["lam"] < 0]
    x = np.array([CC.x_of_lam(p["pt"]["lam"]) for p in g])
    o = np.argsort(x)
    x = x[o]
    lg = np.log(np.array([g[i]["gamma"] for i in o]))
    keep = np.concatenate([[True], np.diff(x) > 1e-9])
    x, lg = x[keep], lg[keep]
    return x, lg, (CubicSpline(x, lg) if len(x) >= 4 else None)


def threshold(x, lg, sp, target_lng, cease_x=None):
    """x_c with ln Gamma(x_c) = target (ln Gamma decreasing in x); linear extrapolation outside, flagged.
    cease_x: x of the first computed point below the narrow range without a narrow resonance; a target width larger
    than the largest narrow width is then 'beyond_narrow' (extrapolated into the regime where the level reaches the
    barrier top: no valid threshold of a narrow resonance)."""
    if sp is None:
        return None
    if lg.max() >= target_lng >= lg.min():
        xs = brentq(lambda z: float(sp(z)) - target_lng, x[0], x[-1])
        return dict(x_c=xs, lam_c=CC.lam_of_x(xs), status="interpolated", extrap_dx=0.0)
    if target_lng > lg.max():                      # needs a larger width: more negative lambda than computed
        e, sl = 0, float(np.polyfit(x[:3], lg[:3], 1)[0])
    else:                                          # needs a smaller width: lambda closer to 0 than computed
        e, sl = -1, float(np.polyfit(x[-3:], lg[-3:], 1)[0])
    xs = x[e] + (target_lng - lg[e]) / sl
    st = "extrapolated"
    if target_lng > lg.max() and cease_x is not None:
        st = "beyond_narrow"
    return dict(x_c=xs, lam_c=CC.lam_of_x(xs), status=st, extrap_dx=float(xs - x[e]))


def fits(x, lg, window=None):
    sel = np.ones(len(x), bool) if window is None else (x >= window[0]) & (x <= window[1])
    xs, ys = x[sel], lg[sel]
    out = {}
    if len(xs) >= 3:
        b1, a1 = np.polyfit(xs, ys, 1)
        out["two_param"] = dict(a=float(a1), b=float(-b1), rms=float(np.sqrt(np.mean((a1 + b1 * xs - ys) ** 2))),
                                n=int(len(xs)), x_range=[float(xs.min()), float(xs.max())])
    if len(xs) >= 5:
        def f(xx, a, b, p):
            return a - b * xx ** (2 * p)                        # |lambda|^-p = x^(2p)
        try:
            popt, pcov = curve_fit(f, xs, ys, p0=[out["two_param"]["a"], out["two_param"]["b"], 0.5], maxfev=20000)
            out["three_param"] = dict(a=float(popt[0]), b=float(popt[1]), p=float(popt[2]),
                                      p_err=float(math.sqrt(max(pcov[2, 2], 0.0))),
                                      rms=float(np.sqrt(np.mean((f(xs, *popt) - ys) ** 2))))
        except Exception as e:  # noqa: BLE001
            out["three_param"] = repr(e)
    return out


def s1():
    B = CC.jload(os.path.join(CC.OUT, "S1", "boxes.json"))
    summ = dict(definition=dict(
        P="survival probability exp(-Gamma N); Gamma = full width in units 1/L; N = T/L (static box held for T)",
        lambda_c="lambda at mu = 1/L with Gamma(lambda_c) N = -ln P at fixed (g2, gY, y_t, g3) of the box",
        interpolation="cubic spline of ln Gamma against x = |lambda|^(-1/2) through the narrow-resonance points "
                      "(status ok / ok_broad); outside the computed range: linear extrapolation in x from the last "
                      "three points (flag 'extrapolated', extrap_dx = distance in x)",
        status="see c170_core.classify"), boxes={})
    rows_g, rows_c, rows_sm = [], [], []
    for b in ("D", "A", "B", "C", "E"):
        pts, dup = load_chains(os.path.join(CC.OUT, "S1", "chains", "S1_%s_*.json" % b))
        if not pts:
            continue
        bx = B[b]
        x, lg, sp = interp_table(pts)
        tab = []
        for p in pts:
            bar = p.get("barrier") or {}
            r1 = p.get("root1d") or {}
            tab.append(dict(lam=p["pt"]["lam"], x=CC.x_of_lam(p["pt"]["lam"]), eps=p.get("eps"), Gamma=p.get("gamma"),
                            status=p.get("status"), flags=p.get("flags"), U_top=bar.get("U_top"), r_top=bar.get("r_top"),
                            U_well=bar.get("U_well"), r_well=bar.get("r_well"), eps_minus_Utop=bar.get("E_minus_Utop"),
                            S_wkb=bar.get("S_wkb"), eps_1D=r1.get("eps"), Gamma_1D=r1.get("gamma"),
                            box_dE=p.get("box_dE"), box_rel_gamma=p.get("box_rel_gamma"), window=p.get("window"),
                            retried=p.get("retried"), time_s=p.get("time_s")))
            rows_g.append([b, bx["mu_GeV"], p["pt"]["lam"], CC.x_of_lam(p["pt"]["lam"]), p.get("eps"), p.get("gamma"),
                           p.get("status"), bar.get("U_top"), bar.get("r_top"), bar.get("E_minus_Utop"), bar.get("S_wkb"),
                           r1.get("gamma"), p.get("box_dE")])
        good = [t for t in tab if t["status"] in GOOD]
        bad = [t for t in tab if t["status"] not in GOOD]
        cease = None
        if good:
            xg = min(t["x"] for t in good)
            below = [t for t in bad if t["x"] < xg and t["status"] in ("above_top", "no_barrier")]
            if below:
                t2 = max(below, key=lambda t: t["x"])
                cease = dict(last_narrow=dict(lam=CC.lam_of_x(xg), x=xg),
                             first_not_narrow=dict(lam=t2["lam"], x=t2["x"], status=t2["status"],
                                                   eps_minus_Utop=t2["eps_minus_Utop"], Gamma=t2["Gamma"]))
            else:
                cease = dict(last_narrow=dict(lam=CC.lam_of_x(xg), x=xg), first_not_narrow=None,
                             note="no non-narrow point computed below the last narrow one (chain end or flags)")
        thr = {}
        cx = (cease or {}).get("first_not_narrow") or {}
        for kP, P in PLIST.items():
            thr[kP] = {}
            for N in NLIST:
                r = threshold(x, lg, sp, math.log(-math.log(P) / N), cease_x=cx.get("x"))
                thr[kP]["%g" % N] = r
                if r:
                    rows_c.append([b, bx["mu_GeV"], P, N, r["lam_c"], r["x_c"], r["status"], r["extrap_dx"]])
        ft = dict(all=fits(x, lg), window_lam_m0p018_m0p007=fits(x, lg, (abs(-0.018) ** -0.5, abs(-0.007) ** -0.5)))
        smt = {}
        for mt, lsm in bx["lam_SM"].items():
            ent = dict(lam_SM=lsm, x=CC.x_of_lam(lsm) if lsm < 0 else None)
            if lsm < 0:
                direct = [p for p in pts if abs(p["pt"]["lam"] - lsm) < 1e-14]
                if direct and direct[0].get("status") in GOOD:
                    ent["Gamma_direct"] = direct[0]["gamma"]
                    ent["eps_direct"] = direct[0]["eps"]
                xo = [i for i in range(len(x)) if abs(x[i] - ent["x"]) > 1e-9]
                if len(xo) >= 4 and x[xo].min() < ent["x"] < x[xo].max():
                    ent["Gamma_interp_without_point"] = float(math.exp(CubicSpline(x[xo], lg[xo])(ent["x"])))
                gsel = ent.get("Gamma_direct") or ent.get("Gamma_interp_without_point")
                if gsel:
                    ent["Nstar_ln2_over_Gamma"] = math.log(2) / gsel
                    ent["N_P0.9"] = -math.log(0.9) / gsel
            else:
                ent["Nstar_ln2_over_Gamma"] = None
                ent["note"] = "lambda_SM >= 0: no instability of the zero mode"
            smt[mt] = ent
            rows_sm.append([b, bx["mu_GeV"], float(mt), lsm, ent.get("Gamma_direct"), ent.get("Gamma_interp_without_point"),
                            ent.get("Nstar_ln2_over_Gamma")])
        summ["boxes"][b] = dict(mu_GeV=bx["mu_GeV"], mu_over_mu_b=bx["mu_over_mu_b"], couplings=bx["couplings"],
                                n_points=len(tab), n_good=len(good), table=tab, duplicate_start_point=dup,
                                narrow_range=dict(x=[float(x.min()), float(x.max())] if len(x) else None,
                                                  lam=[CC.lam_of_x(float(x.min())), CC.lam_of_x(float(x.max()))]
                                                  if len(x) else None),
                                cease=cease, thresholds=thr, fits=ft, lam_SM=smt,
                                max_box_rel_gamma=max([abs(t["box_rel_gamma"]) for t in good
                                                       if t.get("box_rel_gamma") is not None] or [None]),
                                max_abs_ln_ratio_2D_1D=max([abs(math.log(t["Gamma"] / t["Gamma_1D"])) for t in good
                                                            if t.get("Gamma_1D")] or [None]))
        if b == "A":
            ne = {}
            for kP, P in PLIST.items():
                lh = HAZ[kP]
                if sp is not None and x.min() <= CC.x_of_lam(lh) <= x.max():
                    g = float(math.exp(sp(CC.x_of_lam(lh))))
                    ne[kP] = dict(lam_c_hazard=lh, Gamma_A=g, N_eff=-math.log(P) / g, N_eff_over_eta=-math.log(P) / g / CC.ETA,
                                  report_I_over_Gamma0_eta=HAZ_I_OVER[kP])
            summ["boxes"][b]["N_eff"] = ne
    # CSV tables
    od = os.path.join(CC.OUT, "S1")
    with open(os.path.join(od, "C170_S1_gamma_lambda.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S1 static boxes, run05 final model at fixed (g2, gY, y_t, g3) of (172.60, 0.1180, 125.13) at mu; "
                "Gamma, eps in 1/L; status ok/ok_broad = narrow resonance (see README)\n")
        f.write("box,mu_GeV,lambda,x_abs_lambda_m1_2,eps,Gamma,status,U_top,r_top,eps_minus_Utop,S_wkb,Gamma_1D_adiabatic,"
                "box_check_dE\n")
        for r in rows_g:
            f.write(",".join("" if v is None else (("%.10g" % v) if isinstance(v, float) else str(v)) for v in r) + "\n")
    with open(os.path.join(od, "C170_S1_lambda_c_N.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S1 static thresholds: Gamma(lambda_c) N = -ln P; status interpolated/extrapolated (extrap_dx in x)\n")
        f.write("box,mu_GeV,P,N,lambda_c,x_c,status,extrap_dx\n")
        for r in rows_c:
            f.write(",".join("" if v is None else (("%.10g" % v) if isinstance(v, float) else str(v)) for v in r) + "\n")
    with open(os.path.join(od, "C170_S1_lambdaSM_Nstar.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S1: SM quartic lambda_SM(mu) (sm_provider central, alpha_s 0.1180, M_h 125.13) in each box, "
                "width of the box model at lambda_SM (box couplings of m_t = 172.6), N* = ln2/Gamma\n")
        f.write("box,mu_GeV,m_t,lambda_SM,Gamma_direct,Gamma_interp_without_point,Nstar\n")
        for r in rows_sm:
            f.write(",".join("" if v is None else (("%.10g" % v) if isinstance(v, float) else str(v)) for v in r) + "\n")
    CC.wjson(os.path.join(od, "C170_S1_summary.json"), summ)
    short = {b: dict(n=v["n_points"], good=v["n_good"], cease=v["cease"], narrow=v["narrow_range"],
                     lamc_P05={N: (t["lam_c"], t["status"]) if t else None for N, t in v["thresholds"]["P=0.5"].items()},
                     fit2=v["fits"]["all"].get("two_param"), fit3=v["fits"]["all"].get("three_param"),
                     fitwin=v["fits"]["window_lam_m0p018_m0p007"].get("two_param"),
                     Neff=v.get("N_eff"), lamSM=v["lam_SM"], box=v["max_box_rel_gamma"], r21=v["max_abs_ln_ratio_2D_1D"],
                     dup=v["duplicate_start_point"])
             for b, v in summ["boxes"].items()}
    print(json.dumps(CC.clean(short), indent=1))
    return summ


def lamsm():
    """lambda_SM(mu) on a log grid 1e10 GeV ... M-bar_P for m_t = 172.2/172.6/173.0 (+ g2, gY, y_t, g3 at 172.6)."""
    mus = np.logspace(10.0, math.log10(CC.SP.MPBAR), 400)
    cols = {}
    for mt in (172.2, 172.6, 173.0):
        d = CC.SP.couplings(mt, 0.1180, 125.13, mus, "central")
        cols["lam_%.1f" % mt] = np.asarray(d["lam"], float)
        if mt == 172.6:
            for k in ("g2", "gY", "yt", "g3"):
                cols["%s_172.6" % k] = np.asarray(d[k], float)
    # consistency with coup_point at the box scales
    chk = {}
    B = CC.jload(os.path.join(CC.OUT, "S1", "boxes.json"))
    for b, bx in B.items():
        lp = float(np.interp(math.log(bx["mu_GeV"]), np.log(mus), cols["lam_172.6"]))
        chk[b] = dict(lam_grid_interp=lp, lam_box=bx["lam_SM"]["172.6"], diff=lp - bx["lam_SM"]["172.6"])
    p = os.path.join(CC.OUT, "S1", "C170_lambdaSM_running.csv")
    names = ["lam_172.2", "lam_172.6", "lam_173.0", "g2_172.6", "gY_172.6", "yt_172.6", "g3_172.6"]
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("# SM MS-bar couplings (sm_provider 'central' = production NNLO mapping; alpha_s(M_Z) = 0.1180, M_h = 125.13;"
                " m_t = pole mass in GeV), mu from 1e10 GeV to M-bar_P = %.4g GeV; mu_b = M-bar_P/722 = %.6g GeV\n"
                % (CC.SP.MPBAR, CC.SP.MU_B))
        f.write("mu_GeV,log10_mu," + ",".join(names) + "\n")
        for i, mu in enumerate(mus):
            f.write("%.8e,%.6f," % (mu, math.log10(mu)) + ",".join("%.10e" % cols[n][i] for n in names) + "\n")
    zero = {}
    for mt in (172.2, 172.6, 173.0):
        lv = cols["lam_%.1f" % mt]
        idx = np.where(np.diff(np.sign(lv)) != 0)[0]
        zero["%.1f" % mt] = [float(np.exp(np.interp(0.0, [lv[i], lv[i + 1]], [math.log(mus[i]), math.log(mus[i + 1])])))
                             if lv[i] > lv[i + 1] else float(np.exp(np.interp(0.0, [lv[i + 1], lv[i]],
                                                                              [math.log(mus[i + 1]), math.log(mus[i])])))
                             for i in idx]
        zero["min_%.1f" % mt] = dict(lam_min=float(lv.min()), mu_at_min=float(mus[int(np.argmin(lv))]))
    out = dict(file=os.path.relpath(p, CC.OUT), n=len(mus), check_vs_boxes=chk, lambda_zero_crossing_GeV=zero)
    CC.wjson(os.path.join(CC.OUT, "S1", "C170_lambdaSM_running.json"), out)
    print(json.dumps(CC.clean(out), indent=1))
    return out


def lam_threshold(points, target_g):
    """lambda with Gamma = target from a cubic spline of ln Gamma(lambda) over the narrow-resonance points (lambda <= 0);
    returns dict or None; flags extrapolation (linear in lambda from the end points)."""
    g = sorted([p for p in points if p.get("status") in GOOD], key=lambda p: p["pt"]["lam"])
    if len(g) < 2:
        return None
    lam = np.array([p["pt"]["lam"] for p in g])
    lg = np.log(np.array([p["gamma"] for p in g]))
    tl = math.log(target_g)
    sp = CubicSpline(lam, lg) if len(lam) >= 4 else None
    for i in range(len(lam) - 1):                       # ln Gamma decreases with lambda (towards 0-)
        if (lg[i] - tl) * (lg[i + 1] - tl) <= 0:
            if sp is not None:
                lc = brentq(lambda z: float(sp(z)) - tl, lam[i], lam[i + 1])
            else:
                lc = lam[i] + (tl - lg[i]) * (lam[i + 1] - lam[i]) / (lg[i + 1] - lg[i])
            return dict(lam_c=float(lc), status="interpolated", bracket=[float(lam[i]), float(lam[i + 1])])
    if tl > lg.max():
        i0, i1 = 0, 1
    else:
        i0, i1 = -2, -1
    sl = (lg[i1] - lg[i0]) / (lam[i1] - lam[i0])
    lc = lam[i1 if tl < lg.max() else i0] + (tl - lg[i1 if tl < lg.max() else i0]) / sl
    if lc > 0 or (tl < lg.min() and lam.max() >= 0):
        return dict(lam_c=None, status="none_at_lambda_le_0", range=[float(lam.min()), float(lam.max())],
                    note="the width stays above the target up to lambda = 0 (Gamma(0) = %.4g)" % math.exp(lg[-1]))
    return dict(lam_c=float(lc), status="extrapolated", range=[float(lam.min()), float(lam.max())])


def s2():
    tgt = math.log(2.0) / 300.0
    files = sorted(glob.glob(os.path.join(CC.OUT, "S2", "chains", "S2_*_k*_*.json")))
    keys = sorted(set(os.path.basename(f).rsplit("_", 1)[0] for f in files if not f.endswith(".pruned.json")))
    out, rows = {}, []
    for k in keys:
        pts, dup = load_chains(os.path.join(CC.OUT, "S2", "chains", k + "_*.json"))
        pts = [p for p in pts if not p.get("_chain", "").endswith(".pruned.json")]
        model = k.split("_k")[0].replace("S2_", "")
        kap = float(k.split("_k")[1].split("_")[0])
        at = {}
        for lam0 in (-0.012, 0.0):
            q = [p for p in pts if abs(p["pt"]["lam"] - lam0) < 1e-12]
            if q:
                q = q[0]
                bar = q.get("barrier") or {}
                at["%g" % lam0] = dict(status=q.get("status"), Gamma=q.get("gamma") if q.get("status") in GOOD else None,
                                       Gamma_raw=q.get("gamma"), eps=q.get("eps"), U_top=bar.get("U_top"),
                                       U_well=bar.get("U_well"), r_top=bar.get("r_top"), S_wkb=bar.get("S_wkb"),
                                       has_barrier=bar.get("has_barrier"), flags=q.get("flags"),
                                       box_rel_gamma=q.get("box_rel_gamma"),
                                       eps_1D=(q.get("root1d") or {}).get("eps"), Gamma_1D=(q.get("root1d") or {}).get("gamma"),
                                       eps1D_minus_Utop=((q.get("root1d") or {}).get("eps") - bar["U_top"])
                                       if (q.get("root1d") and bar.get("U_top") is not None) else None)
        thr = lam_threshold(pts, tgt)
        good = [p for p in pts if p.get("status") in GOOD]
        tab = [dict(lam=p["pt"]["lam"], status=p.get("status"), eps=p.get("eps"), Gamma=p.get("gamma"),
                    U_top=(p.get("barrier") or {}).get("U_top"), U_well=(p.get("barrier") or {}).get("U_well"),
                    has_barrier=(p.get("barrier") or {}).get("has_barrier"), S_wkb=(p.get("barrier") or {}).get("S_wkb"),
                    Gamma_1D=(p.get("root1d") or {}).get("gamma"), box_rel_gamma=p.get("box_rel_gamma"),
                    flags=p.get("flags")) for p in pts]
        out[k] = dict(model=model, kappa=kap, threshold_N300_P05=thr, at=at, n_points=len(pts), n_good=len(good),
                      narrow_lam_range=[min(p["pt"]["lam"] for p in good), max(p["pt"]["lam"] for p in good)] if good else None,
                      table=tab)
        if thr:
            st_thr = thr["status"]
        elif good:
            st_thr = "none_at_lambda_le_0"
        else:
            st_thr = "no narrow resonance"
        rows.append([model, kap, thr["lam_c"] if thr else None, st_thr,
                     (at.get("-0.012") or {}).get("Gamma"), (at.get("-0.012") or {}).get("status"),
                     (at.get("0") or {}).get("Gamma"), (at.get("0") or {}).get("status"),
                     out[k]["narrow_lam_range"][0] if good else None, out[k]["narrow_lam_range"][1] if good else None])
    od = os.path.join(CC.OUT, "S2")
    with open(os.path.join(od, "C170_S2_lambda_c_kappa.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S2 kappa switch (box A couplings, 2D model without cross terms): static lambda_c(N = 300, P = 1/2); "
                "Gamma at lambda = -0.012 and at lambda = 0 (units 1/L); 2D_Gdmcs = DMC shape transported by the exact "
                "zero-mode scaling (main), 2D_G = DMC shape dropped\n")
        f.write("model,kappa,lambda_c_N300_P05,status,Gamma_lam-0.012,status_lam-0.012,Gamma_lam0,status_lam0,"
                "narrow_lam_min,narrow_lam_max\n")
        for r in sorted(rows, key=lambda z: (z[0], z[1])):
            f.write(",".join("" if v is None else (("%.10g" % v) if isinstance(v, float) else str(v)) for v in r) + "\n")
    CC.wjson(os.path.join(od, "C170_S2_summary.json"), out)
    print(json.dumps(CC.clean({k: dict(thr=v["threshold_N300_P05"], at=v["at"], n=v["n_points"], good=v["n_good"],
                                       rng=v["narrow_lam_range"]) for k, v in out.items()}), indent=1))
    return out


def s3():
    """S3: lambda_c(N = 300; P) per (coupling, s) and logarithmic derivatives; s = 1 from S1 box A."""
    base_pts, _ = load_chains(os.path.join(CC.OUT, "S1", "chains", "S1_A_*.json"))
    xb, lgb, spb = interp_table(base_pts)
    res = {"1.00": {}}
    for kP, P in PLIST.items():
        res["1.00"][kP] = threshold(xb, lgb, spb, math.log(-math.log(P) / 300.0))
    scan = {}
    rows = []
    for g in ("g2", "gY", "yt"):
        scan[g] = {"1.00": res["1.00"]}
        for s in (0.7, 0.85, 1.15, 1.3):
            pts, _ = load_chains(os.path.join(CC.OUT, "S3", "chains", "S3_%s_s%.2f_*.json" % (g, s)))
            pts = [p for p in pts if not p.get("_chain", "").endswith(".pruned.json")]
            if not pts:
                continue
            x, lg, sp = interp_table(pts)
            ent = {}
            for kP, P in PLIST.items():
                ent[kP] = threshold(x, lg, sp, math.log(-math.log(P) / 300.0))
            ent["n_good"] = int(len(x))
            ent["newton_grad_max"] = max([(p.get("info") or {}).get("newton_grad_max_r_gt_0p05") or 0.0 for p in pts])
            ent["newton_safe_used"] = any((p.get("info") or {}).get("newton_safe_used") for p in pts)
            ent["Gamma_at_lam_m0p012"] = float(math.exp(sp(CC.x_of_lam(-0.012)))) if sp is not None and \
                x.min() <= CC.x_of_lam(-0.012) <= x.max() else None
            scan[g]["%.2f" % s] = ent
            for kP in PLIST:
                t = ent[kP]
                rows.append([g, s, kP, t["lam_c"] if t else None, t["status"] if t else None])
        for kP in PLIST:
            rows.append([g, 1.0, kP, res["1.00"][kP]["lam_c"], res["1.00"][kP]["status"]])
    der = {}
    for g in scan:
        der[g] = {}
        for kP in PLIST:
            d = {}
            for a, b in (("0.85", "1.15"), ("0.70", "1.30")):
                if a in scan[g] and b in scan[g] and scan[g][a][kP] and scan[g][b][kP]:
                    la, lb = abs(scan[g][a][kP]["lam_c"]), abs(scan[g][b][kP]["lam_c"])
                    d["sym_%s_%s" % (a, b)] = (math.log(lb) - math.log(la)) / (math.log(float(b)) - math.log(float(a)))
            l1 = abs(res["1.00"][kP]["lam_c"])
            for a in ("0.70", "0.85", "1.15", "1.30"):
                if a in scan[g] and scan[g][a][kP]:
                    d["one_sided_%s" % a] = (math.log(abs(scan[g][a][kP]["lam_c"])) - math.log(l1)) / math.log(float(a))
            sv = [float(s) for s in ("0.70", "0.85", "1.00", "1.15", "1.30") if s in scan[g] and scan[g][s][kP]]
            if len(sv) >= 3:
                yv = [math.log(abs(scan[g]["%.2f" % s][kP]["lam_c"])) for s in sv]
                cf = np.polyfit(np.log(sv), yv, 2)
                d["quadratic_fit_slope_at_1"] = float(cf[1])
                d["quadratic_fit_curvature"] = float(2 * cf[0])
            der[g][kP] = d
    d128 = {"g2": {"P=0.5": 0.66}, "gY": {"P=0.5": 0.16}, "yt": {"P=0.9": 0.575, "P=0.5": 0.627, "P=0.1": 0.649}}
    out = dict(definition="lambda_c(N = 300; P) of box A (run05 final model) with one coupling multiplied by s in all "
                          "sectors; derivatives d ln|lambda_c| / d ln g", thresholds=scan, derivatives=der,
               D128_hazard_derivatives=d128,
               D128_note="D128: hazard (whole turn), earlier model (run02), +-10%; g2 and gY nearly equal at P1 = 0.9 and 0.1")
    od = os.path.join(CC.OUT, "S3")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "C170_S3_coupling_scan.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S3: static threshold lambda_c(N = 300; P) of box A (mu = mu_b) with one coupling scaled by s in all "
                "sectors (run05 final model)\n")
        f.write("coupling,s,P,lambda_c,status\n")
        for r in sorted(rows, key=lambda z: (z[0], z[2], z[1])):
            f.write(",".join("" if v is None else (("%.10g" % v) if isinstance(v, float) else str(v)) for v in r) + "\n")
    CC.wjson(os.path.join(od, "C170_S3_summary.json"), out)
    print(json.dumps(CC.clean(dict(der=der, thr={g: {s: {k: (v[k]["lam_c"] if v.get(k) else None) for k in PLIST}
                                                       for s, v in d.items()} for g, d in scan.items()})), indent=1))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "all"
    if md in ("s1", "all"):
        s1()
    if md in ("lamsm", "all"):
        lamsm()
    if md in ("s2", "all"):
        s2()
    if md in ("s3", "all"):
        s3()
