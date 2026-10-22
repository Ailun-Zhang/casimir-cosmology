"""Origin of the small effective string tension of the static-box widths (report v1.1, Sec. 9.7; review R23 FQ-01).
(1) Local exponent b(x) = -d ln Gamma/dx, x = |lambda|^(-1/2), of the computed box-A widths (cubic spline).
(2) One-dimensional Langer-WKB model of box A: U = 1/(2 r^2) + zp + dmc + dV_MC(r; lambda) + lambda r^4/4 with the
    kappa = 1 components of the S2 potential file and the production matching dV_MC evaluated at each lambda
    (higgs/zmlib/C138_core.dV); Bohr-Sommerfeld ground level, barrier action S, and the slope dS/dx over the fit
    window lambda in [-0.018, -0.007] for four variants: full; without dV; zero-point energy zp + dmc replaced by its
    harmonic form sigma_0 r (matched at r = 1); both.  Barrier heights with and without dV at lambda = -0.012.
Reads data/higgs/static_box_c170; writes out/derived_rep_v11_wkb_split.json.
Usage: python -I -B derive_rep_v11_wkb_split.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
D = HERE / "data" / "higgs" / "static_box_c170"
OUT = HERE / "out"
sys.path.insert(0, str(PKG / "higgs" / "zmlib"))
import C138_core as C  # noqa: E402

LAM0 = -0.012                       # lambda of the archived potential components
WINDOW = (-0.018, -0.007)           # fit window of the box-A width law (Sec. 9.7)


def rcsv(name):
    with open(D / name, encoding="utf-8") as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def wells(U):
    """first local minimum from small r and the first local maximum after it (indices)."""
    d = np.diff(U)
    iw = np.where((d[:-1] < 0) & (d[1:] >= 0))[0][0] + 1
    it = iw + np.where((d[iw:-1] > 0) & (d[iw + 1:] <= 0))[0][0] + 1
    return iw, it


def classically_allowed(U, i0, E, sign):
    """index range around i0 where sign*(E - U) > 0."""
    lo = hi = i0
    while lo > 0 and sign * (E - U[lo - 1]) > 0:
        lo -= 1
    while hi < len(U) - 1 and sign * (E - U[hi + 1]) > 0:
        hi += 1
    return slice(lo, hi + 1)


def level(r, U, iw, it):
    """Bohr-Sommerfeld ground level: int sqrt(2(E - U)) dr = pi/2 (Langer-corrected centrifugal term)."""
    def f(E):
        s = classically_allowed(U, iw, E, +1)
        return np.trapezoid(np.sqrt(np.maximum(2 * (E - U[s]), 0)), r[s]) - math.pi / 2
    return brentq(f, U[iw] + 1e-9, U[it] - 1e-9)


def action(r, U, E, it):
    s = classically_allowed(U, it, E, -1)
    return 2 * np.trapezoid(np.sqrt(np.maximum(2 * (U[s] - E), 0)), r[s])


def main():
    res = json.loads((D / "C170_results.json").read_text(encoding="utf-8"))
    cA = res["S1"]["boxes"]["A"]["couplings"]
    g2, gY, yt = cA["g2"], cA["gY"], cA["yt"]
    sigma0 = 1.5 * g2 + 0.75 * math.hypot(g2, gY)
    rows = rcsv("potential_kappa_lam-0.012.csv")
    r = np.array([float(q["r"]) for q in rows])
    col = lambda name: np.array([float(q[name]) for q in rows])
    zp, dmc, dV0 = col("zp_k1.00"), col("dmc_k1.00"), col("dV_k1.00")
    out = {"couplings_boxA": cA, "sigma0_boxA": sigma0, "window_lambda": list(WINDOW)}

    # dV_MC at each lambda from the production matching; check against the archived column at lambda = -0.012
    dV = lambda lam: C.dV("MC", r, lam, g2, gY, yt)
    out["dV_check_max_abs_diff_lam-0.012"] = float(np.max(np.abs(dV(LAM0) - dV0)))

    base = zp + dmc
    lin = (np.interp(1.0, r, base) - sigma0) + sigma0 * r
    lang = 1.0 / (2 * r ** 2)
    variants = {"full": lambda lam, v: lang + base + v,
                "no_dV": lambda lam, v: lang + base,
                "zp_harmonic": lambda lam, v: lang + lin + v,
                "zp_harmonic_no_dV": lambda lam, v: lang + lin}
    lo, hi = WINDOW
    xs = np.linspace(abs(lo) ** -0.5, abs(hi) ** -0.5, 19)
    dv_at = {x: dV(-x ** -2) for x in xs}
    wkb = {}
    for name, f in variants.items():
        S, S_well, E, gap = [], [], [], []
        for x in xs:
            lam = -x ** -2
            U = f(lam, dv_at[x]) + lam * r ** 4 / 4
            iw, it = wells(U)
            e = level(r, U, iw, it)
            E.append(e)
            gap.append(U[it] - e)
            S.append(action(r, U, e, it))
            S_well.append(action(r, U, U[iw], it))
        wkb[name] = {"window_slope_dS_dx": float(np.polyfit(xs, S, 1)[0]),
                     "window_slope_at_well_bottom": float(np.polyfit(xs, S_well, 1)[0]),
                     "S_range": [float(min(S)), float(max(S))],
                     "Utop_minus_E_range": [float(min(gap)), float(max(gap))]}
    out["wkb_1d_boxA"] = wkb
    out["deep_well_b_2pi_sqrt2_over_3_sigma0"] = 2 * math.pi * math.sqrt(2) / 3 * sigma0
    fit = res["S1"]["boxes"]["A"]["fits"]["window_lam_m0p018_m0p007"]["two_param"]
    out["window_fit_b_boxA"] = fit["b"]
    out["wkb_full_over_fit"] = wkb["full"]["window_slope_dS_dx"] / fit["b"]

    # barrier above the well bottom and above the level, with and without dV, at lambda = -0.012
    bar = {}
    for name, U in (("with_dV", lang + base + dV0 + LAM0 * r ** 4 / 4), ("without_dV", lang + base + LAM0 * r ** 4 / 4)):
        iw, it = wells(U)
        e = level(r, U, iw, it)
        bar[name] = {"r_well": float(r[iw]), "r_top": float(r[it]), "Utop_minus_Uwell": float(U[it] - U[iw]),
                     "Utop_minus_E": float(U[it] - e)}
    out["barrier_lam-0.012_wkb_1d"] = bar

    # local exponent of the computed widths (box A)
    g = [q for q in rcsv("gamma_lambda_boxes.csv") if q["box"] == "A" and q["status"] in ("ok", "ok_broad")]
    xg = np.array([abs(float(q["lambda"])) ** -0.5 for q in g])
    lg = np.array([math.log(float(q["Gamma"])) for q in g])
    o = np.argsort(xg)
    xg, lg = xg[o], lg[o]
    keep = np.concatenate(([True], np.diff(xg) > 1e-9))
    cs = CubicSpline(xg[keep], lg[keep])
    out["local_exponent_boxA"] = {f"{x:g}": {"b": float(-cs(x, 1)), "sigma": float(-cs(x, 1) / (2 * math.pi * math.sqrt(2) / 3))}
                                  for x in (8, 10, 12, 13, 14, 15, 16, 17, 18) if x <= xg[keep].max()}
    out["x_range_boxA"] = [float(xg.min()), float(xg.max())]

    OUT.mkdir(exist_ok=True)
    (OUT / "derived_rep_v11_wkb_split.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
