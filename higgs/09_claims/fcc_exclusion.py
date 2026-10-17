# PROVENANCE: new script written 2026-10-16 by rev-code (R18 revision, DA-01/EIC-01) for the v12 code/data package.
"""[v12 package, R18] Strength of the FCC-ee exclusion statement.  At the measured top-quark pole mass above which
FCC-ee would establish P1 < 0.1 at 95% confidence (C154 future.FCCee_base.mhat_excl_P0p1 = 173.68 GeV, alpha_s =
0.1180): the central survival probability P1(mhat) and the posterior mean of P1, i.e. the exclusion factor
B_ex = Pbar of PRD Eq. (h-bayes) for that measurement.

Inputs (work tree; written by C154_core_v2.py on the run05 centre, run.py stages/creates them):
  w/claims_run05/run01/C154_summary.json            future.FCCee_base: mhat_excl_P0p1, sigma_tot, sigma_meas
  w/claims_run05/run01/fig2a_P1_grid_mt_alphas.csv  P1_central on m_t = 171-175 GeV (step 0.02) x alpha_s grid
Headline:
  P1_central(mhat)  ln(-ln P1) interpolated linearly in m_t on the alpha_s = 0.1180 column of the grid;
  B_ex              Pbar of the C154 machinery for an FCC-ee measurement centred at mhat (imported from
                    C154_core_v2.py with the run05 inputs): the theory-smeared P1 (Theory2 with the FCC-ee alpha_s,
                    SM and M_h errors and the asymmetric, level-dependent model error) averaged over
                    N(mhat, sigma_meas), the construction of future.*.truth.*.Pbar and of every Pbar of the papers.
Cross-checks:
  B_ex_gauss        posterior mean of P1_central over N(mhat, sigma_tot), sigma_tot = 0.131 GeV being the FCC-ee
                    total error of C154 (measurement, alpha_s, SM mapping, M_h and model error in quadrature; the
                    model error symmetrised) -- the one-Gaussian estimate of the R18 referee (0.030);
  the exact central curve S.P1(m - mu_Mh, 0.1180) instead of the grid interpolation;
  replication test: the same Theory2 construction reproduces the archived future.FCCee_base values (truth 172.6
                    Pbar, mhat_excl_P0p1) to machine precision.
Output: w/claims_run05/run01/fcc_exclusion.json (no time stamp).  Paper: PRL End Matter B and PRD Sec. V F
(exclusion sentence; macros \\PoneAtExclFCC, \\BexFCC in numbers/numbers_v12.json).  Run: python -B fcc_exclusion.py
"""
import os as _os  # path bootstrap: zmlib + package root (as in the other packaged scripts)
import sys as _sys
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")
if _ZM not in _sys.path:
    _sys.path.insert(1, _ZM)
import _zpaths as _zp  # noqa: E402
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("C154_CFG", os.path.join(HERE, "C154_inputs_v2_run05.json"))
sys.path.insert(0, HERE)
import C154_core_v2 as C  # noqa: E402  (module level only reads its config; main() is not run)

NEW = _zp.w("w/claims_run05/run01")
A_REF = "0.1180"


def central_column():
    m, y = [], []
    with open(os.path.join(NEW, "fig2a_P1_grid_mt_alphas.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if abs(float(r["alpha_s"]) - float(A_REF)) < 1e-9:
                m.append(float(r["m_t"]))
                y.append(math.log(-math.log(float(r["P1_central"]))))
    o = np.argsort(m)
    return np.array(m)[o], np.array(y)[o]


def surface():
    """The run05 survival surface of C154_core_v2.main() (same construction, lines 'central P1 tables')."""
    dc = json.load(open(os.path.join(C.ROOT, C.CFG["central_P1"]["file"]), encoding="utf-8"))
    g = dc["P1_grid"]
    ks = sorted(g, key=float)
    mref = np.array([float(k) for k in ks])
    yref = np.log(-np.log(np.array([g[k] for k in ks])))
    cr = {a: np.array([dc["crit"][a]["with_cross"][f"P1={p}"] for p in C.LEV]) for a in dc["crit"]}
    cref, tabs = cr["0.1180"], {}
    for a, ck in cr.items():
        xr, xk = cref - cref[1], ck - ck[1]
        cf = np.polyfit(xr, xk, 2)
        dcf = np.polyder(cf)
        x = mref - cref[1]
        q = np.polyval(cf, x)
        q = np.where(x < xr[0], np.polyval(cf, xr[0]) + np.polyval(dcf, xr[0]) * (x - xr[0]), q)
        q = np.where(x > xr[2], np.polyval(cf, xr[2]) + np.polyval(dcf, xr[2]) * (x - xr[2]), q)
        tabs[float(a)] = dict(m=ck[1] + q, lnI=yref)
    return C.Surface(tabs)


def gauss_mean(f, c, s, n=4001):
    """Mean of f(m) over N(c, s), trapezoid on c +- 6 s."""
    m = np.linspace(c - 6 * s, c + 6 * s, n)
    w = np.exp(-0.5 * ((m - c) / s) ** 2)
    return float(np.trapezoid(f(m) * w, m) / np.trapezoid(w, m))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summ = json.load(open(os.path.join(NEW, "C154_summary.json"), encoding="utf-8"))
    fb = summ["future"]["FCCee_base"]
    mhat, s_tot, s_meas = fb["mhat_excl_P0p1"], fb["sigma_tot"], fb["sigma_meas"]
    mg, yg = central_column()
    if not (mg[0] <= mhat - 6 * s_tot and mhat + 6 * s_tot <= mg[-1]):
        raise SystemExit("grid does not cover mhat +- 6 sigma_tot")

    def p1_grid(m):
        return np.exp(-np.exp(np.interp(m, mg, yg)))
    out = {"task": "R18 DA-01: central P1 and exclusion factor B_ex at the FCC-ee exclusion threshold (run05 centre)",
           "alpha_s": float(A_REF), "mhat_excl_P0p1": mhat, "sigma_tot": s_tot, "sigma_meas": s_meas,
           "inputs": {"C154_summary": "w/claims_run05/run01/C154_summary.json#/future/FCCee_base",
                      "grid": "w/claims_run05/run01/fig2a_P1_grid_mt_alphas.csv (alpha_s = 0.1180, P1_central)"}}
    # ---- the C154 machinery for the FCC-ee scenario
    S = surface()
    a0 = float(A_REF)

    def p1_exact(m):
        return S.P1(np.asarray(m, float) - C.MU_MH, a0)
    xl = [S.mcrit(p, a0) - S.mcrit(0.5, a0) for p in C.LEV]
    SP, SM = C.composite(C.CFG["model_components"])
    sc = next(s for s in C.CFG["future"] if s["id"] == "FCCee_base")
    b_ms = np.polyfit(C.CFG["msbar_points"]["mbar"], C.CFG["msbar_points"]["pole"], 1)[0]
    sm = math.sqrt(sc["s_m"] ** 2 + sum(e * e for e in sc["extra"])) * (b_ms if sc["scheme"] == "msbar" else 1.0)
    T = C.Theory2(S, xl, s_a=sc["s_as"], s_SM=sc["s_SM"], s_Mh=sc["s_Mh"] * C.DMC_DMH, msig=(SP, SM), kappa=sc["kappa"])
    H1 = T.Hp(0.1)
    thr_1 = brentq(lambda cc: 1 - C.integ(T, H1, cc, sm, sm) - 0.95, 168.0, 182.0)
    out["headline"] = {
        "P1_central_at_mhat": float(p1_grid(np.array([mhat]))[0]),
        "Bex": C.integ(T, T.G, mhat, sm, sm),
        "prob_P1_le_0p1": 1 - C.integ(T, H1, mhat, sm, sm),
        "definition": "P1 on the central curve at mhat (grid); B_ex = Pbar of PRD Eq. (h-bayes) for an FCC-ee "
                      "measurement centred at mhat: C154 Theory2 (FCC-ee alpha_s, SM, M_h errors, asymmetric "
                      "level-dependent model error) averaged over N(mhat, sigma_meas), as future.*.truth.*.Pbar"}
    out["cross_check"] = {
        "Bex_gauss_sigma_tot": gauss_mean(p1_grid, mhat, s_tot),
        "P1_central_exact_at_mhat": float(p1_exact([mhat])[0]),
        "Bex_gauss_exact_curve": gauss_mean(p1_exact, mhat, s_tot),
        "replication": {"sigma_meas": sm, "sigma_meas_archived": s_meas,
                        "Pbar_truth_172p6": C.integ(T, T.G, 172.6, sm, sm),
                        "Pbar_truth_172p6_archived": fb["truth"]["172.6"]["Pbar"],
                        "mhat_excl_P0p1": thr_1, "mhat_excl_P0p1_archived": mhat},
        "definition": "Bex_gauss = posterior mean of P1_central over N(mhat, sigma_tot) (all FCC-ee errors in one "
                      "Gaussian, model error symmetrised; the referee's estimate); exact = central curve "
                      "S.P1(m - mu_Mh, 0.1180) instead of the grid"}
    with open(os.path.join(NEW, "fcc_exclusion.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"headline": out["headline"], "cross_check": out["cross_check"]}, indent=1))


if __name__ == "__main__":
    main()
