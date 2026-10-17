# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/compare_claims.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.
r"""[v12 package, 2026-10-15] run02 vs run05 claims comparison table (claims_compare.json, claims_headline.csv, claims_run02_vs_run05_full.csv).

Step-1 checks and the run02 -> run05 comparison of the claim layer (C154 v2 core + correction addendum).
(a) regression: the PROVENANCE copies with the run02 inputs reproduce run02 C154_summary.json and
    C154_correction_summary.json (max |diff| over all numeric leaves);
(b) alpha_s-interpolation check of the C154 surface built from run05 (crit at 0.1171/0.1180/0.1189 + 10-point grid at
    0.1180) against DIRECT node hazards at alpha_s = 0.1171 / 0.1189 (run05 X_als shards, 5 masses each), and the
    0.1180 grid against the X_main shards;
(c) side-by-side table run02 vs run05 -> claims_run02_vs_run05_full.csv, claims_headline.csv, claims_compare.json.
Run: python -B compare_claims.py   (env C154_CFG unset -> run05 inputs)"""
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
import glob
import csv
import hashlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.pop("C154_CFG", None)
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.integrate import simpson  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = _zp.WORK
OLD = os.path.join(ROOT, _zp.w("r3/claim_positioning/run02"))
NEW = os.path.join(ROOT, _zp.w("w/claims_run05/run01"))
REG = os.path.join(NEW, "regress_run02")
SH5 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run05/shards"))
SKIP = {"sha256", "runtime_s", "files", "inputs_cfg", "task", "agent", "date"}
HLB = 0.00437243                      # hL_b (C150_twod.HLB)
UN11 = np.round(np.arange(0.0, 5.0001, 0.5), 6)


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k in SKIP:
                continue
            out.update(flat(v, pre + "." + str(k) if pre else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, "%s[%d]" % (pre, i)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool) and d is not None and math.isfinite(d):
        out[pre] = float(d)
    return out


def hazard_I(gam):
    uf = np.linspace(0.0, 5.0, 20001)
    return float(simpson(np.exp(CubicSpline(UN11, np.log(np.maximum(gam, 1e-300)))(uf)) / (HLB * np.sqrt(np.cosh(2 * uf))), x=uf))


def direct_P1():
    R = {}
    for f in glob.glob(os.path.join(SH5, "X_*.json")):
        for x in jl(f)["cols"]:
            R.setdefault((round(x["als"], 4), round(x["mt"], 2)), {})[round(x["u"], 4)] = x["gamma"]
    return {k: math.exp(-hazard_I(np.array([g[round(float(u), 4)] for u in UN11]))) for k, g in R.items() if len(g) == 11}


def main():
    res = {}
    # (a) regression
    reg = {}
    for fn in ("C154_summary.json", "C154_correction_summary.json"):
        a, b = flat(jl(os.path.join(OLD, fn))), flat(jl(os.path.join(REG, fn)))
        com = sorted(set(a) & set(b))
        dif = {k: abs(a[k] - b[k]) for k in com}
        reg[fn] = dict(n_common=len(com), only_old=sorted(set(a) - set(b))[:20], only_new=sorted(set(b) - set(a))[:20],
                       max_abs_diff=max(dif.values()), worst=sorted(dif, key=dif.get)[-3:])
    res["regression_copies_vs_run02"] = reg
    print(json.dumps(reg, indent=1))
    # (b) interpolation check
    import C154_correction_v2c as CORR
    S, xl, lf, dc = CORR.build_surface()
    dP = direct_P1()
    chk = []
    for (a, m), p in sorted(dP.items()):
        ps = float(S.P1(np.array([m]), a)[0])
        chk.append(dict(alpha_s=a, mt=m, P1_direct=p, P1_C154_surface=ps, diff=ps - p))
    res["alpha_s_interpolation_check"] = dict(points=chk, max_abs_diff_0p1180=max(abs(x["diff"]) for x in chk if abs(x["alpha_s"] - 0.118) < 1e-9),
                                              max_abs_diff_other=max(abs(x["diff"]) for x in chk if abs(x["alpha_s"] - 0.118) > 1e-9))
    print("interp check", res["alpha_s_interpolation_check"]["max_abs_diff_0p1180"], res["alpha_s_interpolation_check"]["max_abs_diff_other"])
    # (c) comparison
    full = []
    for fn in ("C154_summary.json", "C154_correction_summary.json"):
        a, b = flat(jl(os.path.join(OLD, fn))), flat(jl(os.path.join(NEW, fn)))
        for k in sorted(set(a) & set(b)):
            full.append([fn, k, a[k], b[k], b[k] - a[k]])
    with open(os.path.join(NEW, "claims_run02_vs_run05_full.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "key", "run02", "run05", "run05_minus_run02"])
        for r in full:
            w.writerow([r[0], r[1]] + ["%.8g" % x for x in r[2:]])
    F = {(r[0], r[1]): r for r in full}

    def g(fn, k):
        r = F[(fn, k)]
        return r[2], r[3]
    cs, cr = "C154_summary.json", "C154_correction_summary.json"
    H = []

    def add(label, fn, k):
        o, n = g(fn, k)
        H.append([label, fn + ":" + k, o, n, n - o])
    for a in ("0.1171", "0.1180", "0.1189"):
        for p in ("0.9", "0.5", "0.1", "0.05"):
            add("crit m_t alpha_s=%s P1=%s" % (a, p), cs, "critical_mt.%s.P1=%s" % (a, p))
    for p in ("0.9", "0.5", "0.1"):
        add("dm_c/dalpha_s P1=%s" % p, cs, "dmc_dalpha_s.P1=%s" % p)
    add("P1 curve at 172.60", cs, "marginal.D1_direct_pole.P1_curve_at_center")
    ids = ["D1_direct_pole", "D1b_direct_pole_interp", "D1c_direct_interp_renorm", "D2_sigma_pole", "D2b_sigma_pole_PDG2025", "D3_MSR",
           "D4_direct_as_mbar", "D3b_msbar_sigma", "C1_ATLAS_CMS", "C2_CMS"]
    for i in ids:
        add("Pbar equal-mass %s" % i, cr, "marginal.equal_mass.%s.Pbar" % i)
        add("P(P1>=0.5) %s" % i, cr, "marginal.equal_mass.%s.prob_P_ge_0p5" % i)
        for j, q in enumerate(("q025", "q16", "median", "q84", "q975")):
            add("%s %s" % (q, i), cr, "marginal.equal_mass.%s.quantiles_025_16_50_84_975[%d]" % (i, j))
        add("Pbar continuous %s" % i, cr, "marginal.continuous.%s.Pbar" % i)
        if (cr, "marginal.equal_mass.%s.Pbar_BKPV_FJ" % i) in F:
            add("Pbar BKPV/FJ equal-mass %s" % i, cr, "marginal.equal_mass.%s.Pbar_BKPV_FJ" % i)
            add("Pbar BKPV/FJ continuous %s" % i, cr, "marginal.continuous.%s.Pbar_BKPV_FJ" % i)
        for vn in ("allorders_pole", "T14a_central_correction", "vi_r1e-4", "model_without_vi", "w0_multiplicative"):
            if (cs, "marginal.%s.Pbar_%s" % (i, vn)) in F:
                add("Pbar variant %s %s (v2 mixed conv.)" % (vn, i), cs, "marginal.%s.Pbar_%s" % (i, vn))
    for k in [x[1] for x in full if x[0] == cs and x[1].startswith("Pbar_progression_direct.")]:
        add("progression " + k.split(".", 1)[1], cs, k)
    for i in ids:
        for pr in ("flat_170_176", "flat_150_200", "flat_lambda_pm0p05"):
            for q in ("B_existence", "B_selection", "B_selection_max", "prior_mean_P1"):
                add("Bayes %s %s %s" % (i, pr, q), cr, "bayes.%s.%s.%s" % (i, pr, q))
    for a in ("0.1171", "0.1180", "0.1189"):
        for k in ("curve_P0.1", "theory95_P0.1", "meas0p27_95_P0.1", "meas_interp_95_P0.1", "curve_P0.05", "theory95_P0.05",
                  "meas0p27_95_P0.05", "meas_interp_95_P0.05"):
            add("exclusion %s %s" % (a, k), cs, "exclusion_mt.%s.%s" % (a, k))
    for sc in ("now_direct", "now_direct_interp", "HLLHC_direct", "HLLHC_direct_interp", "HLLHC_direct_as_permille", "FCCee_base",
               "FCCee_ambitious", "FCCee_ambitious_SM0p04"):
        for k in ("sigma_tot", "separation_172p6_173p3", "mhat_surv95", "mhat_excl_P0p1", "mhat_excl_P0p05"):
            add("future %s %s" % (sc, k), cs, "future.%s.%s" % (sc, k))
        for mt in ("172.6", "173.3"):
            for k in ("Pbar", "prob_P_ge_0p5", "prob_P_le_0p1", "power_surv", "power_excl"):
                add("future %s truth %s %s" % (sc, mt, k), cs, "future.%s.truth.%s.%s" % (sc, mt, k))
    for k in [x[1] for x in full if x[0] == cr and x[1].startswith("HLLHC.")]:
        add("HL-LHC corr " + k.split(".", 1)[1], cr, k)
    for k in [x[1] for x in full if x[0] == cs and x[1].startswith("relative_form.")]:
        add(k, cs, k)
    for k in [x[1] for x in full if x[0] == cs and x[1].startswith("probit_check_D1.")]:
        add(k, cs, k)
    # Bayes ranges over the corrected table (all measurements x priors)
    rng = {}
    for tag, d in (("run02", jl(os.path.join(OLD, cr))["bayes"]), ("run05", jl(os.path.join(NEW, cr))["bayes"])):
        bex = [v["B_existence"] for m in d.values() for v in m.values()]
        bse = [v["B_selection"] for m in d.values() for v in m.values()]
        bmx = [v["B_selection_max"] for m in d.values() for v in m.values()]
        rng[tag] = dict(B_ex_min=min(bex), B_ex_max=max(bex), B_sel_min=min(bse), B_sel_max=max(bse),
                        abs_lnB_max=max(abs(math.log(x)) for x in bex + bse), B_sel_upper_direct_150_200=d["D1_direct_pole"]["flat_150_200"]["B_selection_max"],
                        B_sel_max_over_all_bounds=max(bmx))
        H.append(["Bayes ranges " + tag, "derived", rng[tag]["B_ex_min"], rng[tag]["B_ex_max"], rng[tag]["abs_lnB_max"]])
    res["bayes_ranges"] = rng
    with open(os.path.join(NEW, "claims_headline.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "source_key", "run02", "run05", "run05_minus_run02"])
        for r in H:
            w.writerow([r[0], r[1]] + ["%.6g" % x for x in r[2:]])
    res["headline"] = {r[0]: dict(run02=r[2], run05=r[3], diff=r[4], key=r[1]) for r in H}
    res["sha256"] = {f: hashlib.sha256(open(os.path.join(NEW, f), "rb").read()).hexdigest()
                     for f in ("C154_summary.json", "C154_correction_summary.json")}
    json.dump(res, open(os.path.join(NEW, "claims_compare.json"), "w", encoding="utf-8"), indent=1)
    for lab in ("crit m_t alpha_s=0.1180 P1=0.5", "P1 curve at 172.60", "Pbar equal-mass D1_direct_pole", "P(P1>=0.5) D1_direct_pole",
                "q16 D1_direct_pole", "q84 D1_direct_pole", "Pbar equal-mass D2_sigma_pole", "Pbar equal-mass D3_MSR",
                "Pbar BKPV/FJ equal-mass D1_direct_pole", "Bayes D1_direct_pole flat_150_200 B_selection",
                "exclusion 0.1180 theory95_P0.1", "exclusion 0.1180 meas0p27_95_P0.1", "future FCCee_base separation_172p6_173p3",
                "future FCCee_base mhat_excl_P0p1", "relative_form.margin_mc05_minus_172p60", "relative_form.margin_lambda"):
        x = res["headline"][lab]
        print("%-48s %.5f -> %.5f (%+.5f)" % (lab, x["run02"], x["run05"], x["diff"]))
    print(json.dumps(rng, indent=1))


if __name__ == "__main__":
    main()
