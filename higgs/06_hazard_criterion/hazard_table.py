# PROVENANCE: new script written 2026-10-15 by code-higgs (v12 package).  It reads the node widths produced by the
# production code (04_cross_terms: C149_direct5 run05 shards) and calls the production integration routines
# C150_twod.integ / C150_twod.crit; the 'independent' columns use a separate quadrature as a numerical cross-check.
"""Hazard criterion along the turn (D122 Sec. 1.3; D121 run05): single-branch survival of the Higgs zero mode.

    I(m_t, alpha_s) = int_0^5 gamma(m_t, alpha_s; u) du / (hL_b sqrt(cosh 2u)),     P1 = exp(-I),
    m_c(P1) = root of ln I(m_t) = ln(-ln P1)      (P1 = 0.9 / 0.5 / 0.1),

with gamma the 2D resonance width (in 1/L) at the 11 hazard nodes u = 0, 0.5, ..., 5 where the SM couplings are taken at
mu(u) = mu_b / sqrt(cosh 2u), hL_b = 0.00437243 (eta_inf = 1.31103/hL_b = 299.84).
Production recipe (C150_twod.integ/crit): cubic spline of ln gamma in u + Simpson on 20001 points; cubic spline of
ln I in m_t + root.  Independent cross-check here: adaptive quadrature (scipy.integrate.quad) of the same spline and
of a PCHIP interpolant of ln gamma; Brent root of a PCHIP interpolant of ln I.

Also assembles the convergence table of the critical masses (PRD H4: MU -> ... -> 2D direct -> +NNLO -> +cross terms
-> run04 -> run05) from the summaries in the work tree (recomputed or staged).
Output: <work>/r3/hazard_criterion/hazard_table.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "zmlib"))
import _zpaths as _zp  # noqa: E402
import C150_twod as W  # noqa: E402

LEVELS = {"P1=0.9": 0.9, "P1=0.5": 0.5, "P1=0.1": 0.1}


def load_widths(shard_dir):
    R = {}
    for fn in sorted(os.listdir(shard_dir)):
        if not fn.startswith("X_") or not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(shard_dir, fn), encoding="utf-8"))
        for x in d["cols"]:
            R[(round(x["als"], 4), round(x["mt"], 2), round(x["u"], 4))] = x["gamma"]
    return R


def I_alt(u, g, kind):
    lg = np.log(g)
    f = CubicSpline(u, lg) if kind == "spline" else PchipInterpolator(u, lg)
    val, err = quad(lambda x: math.exp(float(f(x))) / (W.HLB * math.sqrt(math.cosh(2 * x))), 0.0, 5.0,
                    limit=400, epsabs=0.0, epsrel=1e-11)
    return val


def crit_alt(mts, I, kind="pchip"):
    f = (PchipInterpolator if kind == "pchip" else CubicSpline)(np.array(mts), np.log(np.array(I)))
    out = {}
    for k, p in LEVELS.items():
        t = math.log(-math.log(p))
        try:
            out[k] = brentq(lambda m: float(f(m)) - t, mts[0], mts[-1], xtol=1e-12)
        except ValueError:
            out[k] = None
    return out


def get(path, *keys):
    try:
        o = json.load(open(path, encoding="utf-8"))
        for k in keys:
            o = o[k]
        return o
    except Exception:  # noqa: BLE001
        return None


def main():
    sh = _zp.w("r3/holonomy_cross/run05/shards")
    R = load_widths(sh)
    u = np.array(W.UN11, float)
    out = dict(meta=dict(hLb=W.HLB, eta_inf=W.ETA, nodes=u.tolist(), source="r3/holonomy_cross/run05/shards"),
               I={}, P1={}, crit_production={}, crit_independent_spline_quad={}, crit_independent_pchip={})
    for als in sorted({k[0] for k in R}):
        mts = sorted({k[1] for k in R if k[0] == als})
        mts = [m for m in mts if all((als, m, round(float(x), 4)) in R for x in u)]
        G = [np.array([R[(als, m, round(float(x), 4))] for x in u]) for m in mts]
        Ip = [W.integ(u, g) for g in G]
        Is = [I_alt(u, g, "spline") for g in G]
        Ih = [I_alt(u, g, "pchip") for g in G]
        a = "%.4f" % als
        out["I"][a] = {"%.2f" % m: dict(production=i0, quad_spline=i1, quad_pchip=i2) for m, i0, i1, i2 in zip(mts, Ip, Is, Ih)}
        out["P1"][a] = {"%.2f" % m: math.exp(-i0) for m, i0 in zip(mts, Ip)}
        cp, _ = W.crit(mts, Ip)
        out["crit_production"][a] = cp
        out["crit_independent_spline_quad"][a] = crit_alt(mts, Is, "spline")
        out.setdefault("crit_pchip_in_u", {})[a] = crit_alt(mts, Ih, "spline")
        out.setdefault("crit_pchip_in_mt", {})[a] = crit_alt(mts, Ip, "pchip")
        out["crit_independent_pchip"][a] = crit_alt(mts, Ih, "pchip")
    for k, nm in (("crit_independent_spline_quad", "max_abs_diff_spline_quad_vs_production_GeV"),
                  ("crit_pchip_in_u", "max_abs_diff_pchip_in_u_GeV"),
                  ("crit_pchip_in_mt", "max_abs_diff_pchip_in_mt_GeV"),
                  ("crit_independent_pchip", "max_abs_diff_pchip_vs_production_GeV")):
        # spline_quad: same cubic splines (ln gamma in u, ln I in m_t), adaptive quadrature instead of Simpson -> tests
        # the quadrature; pchip_*: monotone C1 interpolants (less accurate than cubic splines; the production 11- vs
        # 29-node difference is <= 7e-5 GeV) -> upper bounds on interpolation sensitivity, not production errors
        out[nm] = max(abs(out[k][a][lv] - out["crit_production"][a][lv]) for a in out["crit_production"] for lv in LEVELS
                      if out[k][a].get(lv) and out["crit_production"][a].get(lv))
    out["max_abs_diff_independent_vs_production_GeV"] = out["max_abs_diff_spline_quad_vs_production_GeV"]
    # ---- convergence table (alpha_s = 0.1180)
    w = _zp.w
    r1 = w("r3/twod_direct_sm/run01/C150_summary.json")
    rows = []
    for row in (get(r1, "convergence_table", "rows") or []):
        val = row.get("recomputed_this_pipeline") or row.get("published")
        if val:
            rows.append(dict(stage=row["stage"], mapping="R14 (2-loop + 3-loop calibration)", m_c=val))
    rows.append(dict(stage="+ NNLO SM mapping (D120), 2D direct (T14b run02)", mapping="NNLO",
                     m_c=get(w("r3/twod_direct_sm/run02/C150_summary.json"), "main_2D_direct_as0.1180", "critical_mt")))
    rows.append(dict(stage="+ cross terms, CCA, node-direct (D121 run02)", mapping="NNLO",
                     m_c=get(w("r3/holonomy_cross/run02/C149_run02_summary.json"), "crit", "0.1180", "with_cross")))
    rows.append(dict(stage="+ vev mixing (old vertex) + exact top chiral (D121 run04)", mapping="NNLO",
                     m_c=get(w("r3/holonomy_cross/run04/C149_run04_summary.json"), "A", "crit", "0.1180")))
    rows.append(dict(stage="+ covariant Coulomb-gauge vertex + Coulomb mismatch (D121 run05) = PRL centre", mapping="NNLO",
                     m_c=out["crit_production"].get("0.1180")))
    out["convergence_table_0p1180"] = rows
    p = _zp.w("r3/hazard_criterion/hazard_table.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(W.clean(out), f, indent=1)
    print(json.dumps(W.clean(dict(crit=out["crit_production"], independent=out["crit_independent_spline_quad"],
                                  maxdiff=out["max_abs_diff_independent_vs_production_GeV"],
                                  table=[(r["stage"], r["m_c"]) for r in rows])), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
