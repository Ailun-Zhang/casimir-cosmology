# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/xi_compare.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.
r"""[v12 package, 2026-10-15] xi_c at the run05 centre vs run02 (D129), EW image, and the universal xi_c(P1) reading (PCHIP through the run02 central points; residuals of the 16 run02 variants and of the run05 points) -> xi_run05_summary.json, xi_run05_compare.csv.  Paper: PRL xi_H(mu_b) >~ 0.08; PRD H3.

Step 4 post: xi_c at the run05 centre (copied D129 pipeline, xi_run05/C157_xi_quick.json) vs run02 (D129: realtime_2d
run01 C157_xi_quick.json central + run02 C157_run02_summary.json variants), EW image, and the 'xi_c(P1) universal' reading:
xi_c^univ(x), x = ln(-ln P1), = PCHIP through the 5 run02 central points; residuals of the 16 run02 variants and of the
run05 direct points.  Run: python -B xi_compare.py"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import csv

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.stdout.reconfigure(encoding="utf-8")
ROOT = _zp.WORK
RT1 = os.path.join(ROOT, _zp.w("r3/realtime_2d/run01"))
RT2 = os.path.join(ROOT, _zp.w("r3/realtime_2d/run02"))
X5 = os.path.join(ROOT, _zp.w("w/claims_run05/run01/xi_run05"))
NEW = os.path.join(ROOT, _zp.w("w/claims_run05/run01"))
P1C05 = 172.95945013527066


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    q2, s2 = jl(os.path.join(RT1, "C157_xi_quick.json")), jl(os.path.join(RT2, "C157_run02_summary.json"))
    q5, cal5 = jl(os.path.join(X5, "C157_xi_quick.json")), jl(os.path.join(X5, "C157_calib.json"))
    Xi = {float(k): v["central"]["Xi"] for k, v in s2["bands"].items()}
    xs_ = sorted(Xi)

    def Xi_at(m):
        return float(np.interp(m, xs_, [Xi[x] for x in xs_]))
    # run02 universal curve
    pts = sorted((math.log(-math.log(c["P1_hazard_2D_C149run02"])), c["merged"], float(k)) for k, c in q2["central"].items()
                 if c.get("merged") is not None and c.get("P1_hazard_2D_C149run02"))
    f = PchipInterpolator([p[0] for p in pts], [p[1] for p in pts], extrapolate=True)
    var_res = {k: v["merged"] - float(f(math.log(-math.log(v["P1_1D"])))) for k, v in s2["variants"].items() if v.get("merged") is not None}
    out = dict(task="C161 step 4: xi_c on the run05 centre", definition=q5.get("definition"), run05={}, run02={}, universality={})
    rows = []
    for k, c in sorted(q5["central"].items(), key=lambda t: float(t[0])):
        m = float(k)
        P1_2D = cal5["P1_check"][k]["P1_2D"]
        e = dict(m_t=m, P1_2D_run05=P1_2D, P1_1D_calibrated=cal5["P1_check"][k]["P1_1D_calibrated"], xi_c=c.get("merged"),
                 half_diff=c.get("half_diff"), lam_phi=c.get("lam_phi"), lam_1L=c.get("lam_1L"), spread_all=c.get("spread_all"),
                 P_surv_16_u6=c.get("P_surv_16_u6"), Xi_mu_b=Xi_at(m))
        if e["xi_c"] is not None:
            e["xi_c_EW"] = 1 / 6 + (e["xi_c"] - 1 / 6) / e["Xi_mu_b"]
            e["xi_univ_pred"] = float(f(math.log(-math.log(P1_2D))))
            e["dev_vs_universal"] = e["xi_c"] - e["xi_univ_pred"]
        o = q2["central"].get("%.2f" % m)
        if o and o.get("merged") is not None:
            e["run02_xi_c"], e["run02_half_diff"], e["run02_spread_all"] = o["merged"], o["half_diff"], o["spread_all"]
            e["run02_P1"] = o["P1_hazard_2D_C149run02"]
            e["run05_minus_run02"] = e["xi_c"] - o["merged"] if e["xi_c"] is not None else None
            e["run02_xi_c_EW"] = 1 / 6 + (o["merged"] - 1 / 6) / e["Xi_mu_b"]
        out["run05"][k] = e
        rows.append(e)
    # P1 = 0.5 point of run05 (172.9595) by linear interpolation 172.95 / 172.96
    a, b = out["run05"].get("172.95"), out["run05"].get("172.96")
    if a and b and a.get("xi_c") is not None and b.get("xi_c") is not None:
        w = (P1C05 - 172.95) / 0.01
        out["P1_half_point_run05"] = dict(m_t=P1C05, xi_c=(1 - w) * a["xi_c"] + w * b["xi_c"],
                                          half_diff=(1 - w) * a["half_diff"] + w * b["half_diff"],
                                          xi_c_EW=(1 - w) * a["xi_c_EW"] + w * b["xi_c_EW"],
                                          run02_P1_half_point=dict(m_t=172.9537, xi_c=s2["bands"]["172.95"]["central"]["xi_c"]))
    out["universality"] = dict(curve_points_run02=pts, variant_residuals_run02=var_res,
                               max_abs_variant_residual=max(abs(v) for v in var_res.values()),
                               max_abs_run05_dev=max(abs(e["dev_vs_universal"]) for e in rows if e.get("dev_vs_universal") is not None))
    # [R18 rev-code 2026-10-16, MH-13] the universality test proper: only points whose x = ln(-ln P1) lies inside the
    # span of the central curve are compared by interpolation; outside it the PCHIP curve is extrapolated
    x_lo, x_hi = min(p[0] for p in pts), max(p[0] for p in pts)
    var_x = {k: math.log(-math.log(v["P1_1D"])) for k, v in s2["variants"].items() if v.get("merged") is not None}
    ins = {k: r for k, r in var_res.items() if x_lo <= var_x[k] <= x_hi}
    r05 = {k: e["dev_vs_universal"] for k, e in out["run05"].items() if e.get("dev_vs_universal") is not None
           and x_lo <= math.log(-math.log(e["P1_2D_run05"])) <= x_hi}
    out["universality"]["interpolation_range"] = dict(
        definition="universality residuals restricted to points inside the P1 span of the run02 central curve "
                   "(interpolation); points outside are extrapolations of the curve and are listed separately",
        curve_P1_span=[math.exp(-math.exp(x_hi)), math.exp(-math.exp(x_lo))], curve_m_t=[p[2] for p in pts],
        variant_P1={k: math.exp(-math.exp(x)) for k, x in var_x.items()},
        variants_inside=sorted(ins), n_inside=len(ins), n_variants=len(var_res),
        tested_P1_range_variants=[min(math.exp(-math.exp(var_x[k])) for k in ins), max(math.exp(-math.exp(var_x[k])) for k in ins)],
        max_abs_variant_residual_inside=max(abs(v) for v in ins.values()),
        variants_outside_extrapolated={k: dict(P1=math.exp(-math.exp(var_x[k])), residual=r) for k, r in var_res.items() if k not in ins},
        max_abs_run05_dev_inside=max(abs(v) for v in r05.values()), run05_points_inside=sorted(r05))
    out["run02"] = {k: dict(xi_c=v["merged"], half_diff=v["half_diff"], spread_all=v["spread_all"], P1=v["P1_hazard_2D_C149run02"])
                    for k, v in q2["central"].items() if v.get("merged") is not None}
    json.dump(out, open(os.path.join(NEW, "xi_run05_summary.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(NEW, "xi_run05_compare.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["m_t", "P1_run05", "xi_c_run05", "half_diff_run05", "spread_lo", "spread_hi", "xi_c_EW_run05", "P1_run02", "xi_c_run02",
                     "run05_minus_run02", "xi_univ_pred_from_run02_curve", "dev_vs_universal"])
        for e in rows:
            sp = e.get("spread_all") or [None, None]
            wr.writerow(["%.2f" % e["m_t"]] + [("%.5f" % v) if isinstance(v, float) else "" for v in
                                               (e["P1_2D_run05"], e.get("xi_c"), e.get("half_diff"), sp[0], sp[1], e.get("xi_c_EW"), e.get("run02_P1"),
                                                e.get("run02_xi_c"), e.get("run05_minus_run02"), e.get("xi_univ_pred"), e.get("dev_vs_universal"))])
    print(json.dumps({k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("spread_all", "run02_spread_all")}
                      for k, v in out["run05"].items()}, indent=1))
    print("P1=0.5 point", out.get("P1_half_point_run05"))
    print("universality: max variant residual %.4f ; max run05 dev %.4f" % (out["universality"]["max_abs_variant_residual"],
                                                                           out["universality"]["max_abs_run05_dev"]))
    ir = out["universality"]["interpolation_range"]
    print("interpolation range: curve P1 span %.4f-%.4f; %d/%d variants inside (P1 %.4f-%.4f), max |residual| %.4f; outside: %s"
          % (ir["curve_P1_span"][0], ir["curve_P1_span"][1], ir["n_inside"], ir["n_variants"], ir["tested_P1_range_variants"][0],
             ir["tested_P1_range_variants"][1], ir["max_abs_variant_residual_inside"],
             {k: (round(v["P1"], 4), round(v["residual"], 4)) for k, v in ir["variants_outside_extrapolated"].items()}))


if __name__ == "__main__":
    main()
