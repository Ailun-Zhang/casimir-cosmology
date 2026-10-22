"""Derived numbers quoted in the report v1.1 static-box subsections (Secs. 7, 9, 20) that are not stored as single
keys in the C170 results: closed-form accuracy, local exponents, WKB exponents and outer turning points at the
thresholds, the Fubini-type bounce action, sigma_0 and the periodic-top slope, the Planck-operator estimate, and
unit conversions.  Reads only data/higgs/static_box_c170; writes out/derived_rep_v11_static_box.json.
Usage: python -I -B derive_rep_v11_static_box.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "higgs" / "static_box_c170"
OUT = HERE / "out"
DLAM_DMT = 0.00659          # |d lambda(mu_b)/d m_t| in 1/GeV (INV_HIGGS a.2; PRL text 0.0066; factor of lcAthreeErr*)
MPBAR_OVER_MUB = 722.0      # M-bar_P / mu_b = L_b M-bar_P (H022)


def rcsv(name):
    with open(D / name, encoding="utf-8") as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def main():
    res = json.loads((D / "C170_results.json").read_text(encoding="utf-8"))
    out = {}
    A = res["S1"]["boxes"]["A"]
    cA = A["couplings"]
    g2, gY, yt = cA["g2"], cA["gY"], cA["yt"]
    gZ = math.hypot(g2, gY)
    out["sigma0_boxA"] = 1.5 * g2 + 0.75 * gZ
    out["six_yt_over_sqrt2"] = {"y_t=0.430": 6 * 0.430 / math.sqrt(2), "boxA y_t": 6 * yt / math.sqrt(2), "y_t_boxA": yt}
    out["wkb_coefficient_2pi_sqrt2_over_3"] = 2 * math.pi * math.sqrt(2) / 3

    # ---- window fit and closed form
    fw = A["fits"]["window_lam_m0p018_m0p007"]["two_param"]
    a, b = fw["a"], fw["b"]
    out["window_fit_boxA"] = {"a": a, "b": b, "rms": fw["rms"], "n": fw["n"], "x_range": fw["x_range"],
                              "sigma_eff": b / out["wkb_coefficient_2pi_sqrt2_over_3"]}
    lc = rcsv("lambda_c_N_boxes.csv")
    cf = {}
    for N in (10, 30, 100, 300, 1000, 10000, 100000, 1000000):
        direct = [float(r["lambda_c"]) for r in lc if r["box"] == "A" and r["P"] == "0.5" and int(float(r["N"])) == N][0]
        closed = -(b / (a + math.log(N / math.log(2)))) ** 2
        closed_r = -(1.306 / (5.915 + math.log(N / math.log(2)))) ** 2
        cf[str(N)] = {"direct": direct, "closed": closed, "rel": closed / direct - 1, "closed_rounded_ab": closed_r,
                      "rel_rounded_ab": closed_r / direct - 1}
    out["closed_form_boxA_P05"] = cf
    out["closed_form_max_rel_1e2_1e4"] = max(abs(cf[k]["rel"]) for k in ("100", "300", "1000", "10000"))
    out["dln_lamc_dlnN_at_300_closed"] = -2 / (a + math.log(300 / math.log(2)))
    # slope of x_c = |lambda_c|^(-1/2) in ln N (box A, P = 1/2) against 1/b
    xs = {N: float([r["x_c"] for r in lc if r["box"] == "A" and r["P"] == "0.5" and int(float(r["N"])) == N][0])
          for N in (10, 1000000)}
    out["xc_slope_lnN_boxA"] = {"x_c(10)": xs[10], "x_c(1e6)": xs[1000000],
                                "slope": (xs[1000000] - xs[10]) / math.log(1e5), "one_over_b": 1 / b}

    # ---- local exponent deep in the well (box A, lambda in [-0.0072, -0.003])
    g = [r for r in rcsv("gamma_lambda_boxes.csv") if r["box"] == "A" and r["status"] in ("ok", "ok_broad")]
    lam = np.array([float(r["lambda"]) for r in g])
    gam = np.array([float(r["Gamma"]) for r in g])
    sel = (lam >= -0.0072) & (lam <= -0.003)
    f3 = lambda x, a0, b0, p: a0 - b0 * np.abs(x) ** (-p)
    popt, pcov = curve_fit(f3, lam[sel], np.log(gam[sel]), p0=(2.0, 0.5, 0.6))
    out["deep_well_boxA"] = {"lam_range": [float(lam[sel].min()), float(lam[sel].max())], "n": int(sel.sum()),
                             "p": float(popt[2]), "p_err": float(math.sqrt(pcov[2, 2]))}
    x = np.abs(lam[sel]) ** -0.5
    bb, aa = np.polyfit(x, np.log(gam[sel]), 1)
    out["deep_well_boxA"]["two_param"] = {"a": float(aa), "b": float(-bb)}

    # ---- WKB exponent, level gap and outer turning point along box A (chains), and at the thresholds
    pts = []
    for f in ("S1_A_up.json", "S1_A_down.json"):
        for p in json.loads((D / f).read_text(encoding="utf-8"))["points"]:
            br = p.get("barrier", {})
            if p.get("status") in ("ok", "ok_broad") and br.get("r_out") is not None:
                pts.append((p["pt"]["lam"], br["S_wkb"], br["r_out"], br["E_minus_Utop"], p["gamma"]))
    pts = sorted(set(pts))
    lam_p = np.array([q[0] for q in pts])
    xq = np.abs(lam_p) ** -0.5
    o = np.argsort(xq)
    xq, Sw, ro, gap = xq[o], np.array([q[1] for q in pts])[o], np.array([q[2] for q in pts])[o], \
        np.array([q[3] for q in pts])[o]
    keep = np.concatenate(([True], np.diff(xq) > 1e-9))
    xq, Sw, ro, gap = xq[keep], Sw[keep], ro[keep], gap[keep]
    sS, sR, sG = CubicSpline(xq, Sw), CubicSpline(xq, ro), CubicSpline(xq, gap)
    at = {}
    for N in (10, 100, 300, 10000, 1000000):
        lcN = [float(r["lambda_c"]) for r in lc if r["box"] == "A" and r["P"] == "0.5" and int(float(r["N"])) == N][0]
        xc = abs(lcN) ** -0.5
        at[str(N)] = {"lambda_c": lcN, "S_wkb": float(sS(xc)), "r_out": float(sR(xc)), "eps_minus_Utop": float(sG(xc)),
                      "fubini_action": 8 * math.pi ** 2 / (3 * abs(lcN))}
    lam_sm = A["lam_SM"]["172.6"]["lam_SM"]
    xsm = abs(lam_sm) ** -0.5
    at["lambda_SM_172.6"] = {"lambda": lam_sm, "S_wkb": float(sS(xsm)), "r_out": float(sR(xsm))}
    win = (np.abs(xq ** -2) <= 0.018 + 1e-12) & (np.abs(xq ** -2) >= 0.007 - 1e-12)
    out["boxA_along_lambda"] = {"at_thresholds_P05": at,
                                "window_S_wkb": [float(Sw[win].min()), float(Sw[win].max())],
                                "window_gap": [float(-gap[win].max()), float(-gap[win].min())],
                                "window_r_out": [float(ro[win].min()), float(ro[win].max())]}
    out["fubini_action_lam0.01"] = 8 * math.pi ** 2 / (3 * 0.01)
    zc = res["lambda_SM_running"]["lambda_zero_crossing_GeV"]
    mins = {mt_: zc[f"min_{mt_}"] for mt_ in ("172.2", "172.6", "173.0")}
    out["lambda_SM_min"] = mins
    out["fubini_action_at_lambda_min"] = {k: 8 * math.pi ** 2 / (3 * abs(v["lam_min"])) for k, v in mins.items()}
    out["bounce_over_zero_mode_min_ratio"] = min(v["fubini_action"] / v["S_wkb"]
                                                 for k, v in at.items() if k != "lambda_SM_172.6")

    # ---- Planck-suppressed operator c r^6 / M-bar_P^2 in units 1/L at mu = mu_b: Delta lambda_eff = 4 c (r mu_b/M_P)^2
    pl = {}
    for r in (5.3, 5.7, 6.0):
        dl = 4 * (r / MPBAR_OVER_MUB) ** 2
        pl[str(r)] = {"phi_over_MPbar": r / MPBAR_OVER_MUB, "dlambda_eff_per_c": dl, "dmc_GeV_per_c": dl / DLAM_DMT}
    out["planck_operator"] = pl

    # ---- relation to the expanding torus
    s0 = res["S0"]["static_threshold_archived_u0"]["rows"]
    out["static_minus_hazard"] = {k: v["shallower_by"] for k, v in s0.items()}
    out["static_threshold_N_eta"] = {k: v["lam_c_static_x_spline"] for k, v in s0.items()}
    out["N_eff_boxA"] = {k: {"N_eff": v["N_eff"], "ratio": v["N_eff_over_eta"]} for k, v in res["S1"]["N_eff_boxA"].items()}
    k1 = res["S2"]["kappa1_vs_final_model"]
    out["cross_term_shift"] = {"static_lamc300": -k1["2D_Gdmcs_kappa1"]["dlam_c_vs_final"],
                               "hazard_0.118GeV_x_dlam_dmt": 0.118 * DLAM_DMT,
                               "DMC_shape_static": k1["2D_G_kappa1"]["lam_c_N300_P05"] - k1["2D_Gdmcs_kappa1"]["lam_c_N300_P05"]}
    out["model_error_lamc300"] = {"up": 0.0509 * DLAM_DMT, "down": 0.0267 * DLAM_DMT,
                                  "note": "zero-mode-model row of Table I at P1 = 1/2 (+0.0267/-0.0509 GeV, R5-039 no_vi) x 0.00659; m_c down = lambda_c up"}

    # ---- boxes: spread of lambda_c(300) and of N*
    l300 = {r["box"]: float(r["lambda_c"]) for r in lc if r["P"] == "0.5" and int(float(r["N"])) == 300}
    out["lambda_c_300_boxes"] = l300
    out["lambda_c_300_E_over_D"] = l300["E"] / l300["D"]
    # box dependence predicted by the S3 coupling derivatives (sym 0.85/1.15, P = 1/2), g3 not included
    der = res["S3"]["derivatives"]
    cD, cE = res["S1"]["boxes"]["D"]["couplings"], res["S1"]["boxes"]["E"]["couplings"]
    pred = sum(der[k]["P=0.5"]["sym_0.85_1.15"] * math.log(cE[k] / cD[k]) for k in ("g2", "gY", "yt"))
    out["box_ratio_from_coupling_derivatives"] = {"ln_ratio_pred": pred, "ratio_pred": math.exp(pred),
                                                  "ratio_computed": l300["E"] / l300["D"],
                                                  "terms": {k: der[k]["P=0.5"]["sym_0.85_1.15"] * math.log(cE[k] / cD[k])
                                                            for k in ("g2", "gY", "yt")}}
    # halving / doubling the duration at N = 300 with the closed form (box A window fit)
    lcf = lambda N: -(b / (a + math.log(N / math.log(2)))) ** 2
    out["closed_form_half_double_N300"] = {"half": lcf(150) / lcf(300) - 1, "double": lcf(600) / lcf(300) - 1}
    # pure zero-mode scaling comparison for the kappa scan (2D_Gdmcs): lambda_c(kappa)/lambda_c(1) vs kappa^2
    kap = {float(r["kappa"]): float(r["lambda_c_N300_P05"]) for r in rcsv("lambda_c_kappa.csv")
           if r["model"] == "2D_Gdmcs" and r["lambda_c_N300_P05"]}
    out["kappa_ratio_vs_kappa2"] = {str(k): {"ratio": v / kap[1.0], "kappa2": k * k} for k, v in kap.items()}
    # ---- kappa = 0 monotonicity bound
    k0 = res["S2"]["kappa0"]
    out["kappa0_max_dUdr"] = {k: v["max_dU_dr"] for k, v in k0.items() if isinstance(v, dict) and "max_dU_dr" in v}
    out["kappa0_max_dUdr_overall"] = max(out["kappa0_max_dUdr"].values())
    # ---- checks
    ch = json.loads((D / "checks.json").read_text(encoding="utf-8"))
    out["checks"] = {"n": ch["n_checks"], "pass": ch["n_pass"]}

    OUT.mkdir(exist_ok=True)
    (OUT / "derived_rep_v11_static_box.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
