# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/higgs_robustness/C152_account.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent higgs-robustness (C152/D124, v12 round 3) on 2026-10-15.
r"""[v12 package, 2026-10-15] Bookkeeping of the background constants (h/M_P implied by the pipeline vs v11, L_b Mbar_P) and their m_c effect.  Paper: PRD App. H3 (footnote).

C152 account: compact deliverables from C152_post.json / C152_w0_B3.json (+ read-only C149 run02 summary, C157):
C152_input_dmc.json (input -> Delta m_c), C152_mc_w0.json, C152_accounting.json, C152_summary.json."""
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
import hashlib

ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r3/higgs_robustness/run01"))
C149 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/C149_run02_summary.json"))
C157 = os.path.join(ROOT, _zp.w("r3/realtime_2d/run01/C157_xi_quick.json"))
T = ("P1=0.9", "P1=0.5", "P1=0.1")


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def wj(name, obj):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    P = jl(os.path.join(RES, "C152_post.json"))
    w0f = os.path.join(RES, "C152_w0_B3.json")
    W0 = jl(w0f) if os.path.exists(w0f) else None
    S149, X157 = jl(C149), jl(C157)
    V = P["variants"]
    base = P["baseline"]["crit"]

    def row(item, key, note=""):
        v = V[key]
        d = dict(item=item, variant=key, dmu=round(v["dmu_dh"][0], 6), dh=round(v["dmu_dh"][1], 6),
                 L_bM_P=round(722.27 * math.exp(-v["dmu_dh"][0]), 2), hL_b=round(0.00437243 * math.exp(v["dmu_dh"][1]), 7),
                 dmc={t: r4(v["dmc"][t]) for t in T}, P1_172p6=r4(v["P1_172p6"]), note=note)
        if "dmc_direct" in v:
            d["dmc_direct_all_nodes"] = {t: r4(v["dmc_direct"][t]) for t in T}
        return d
    rows = [
        row("(1) xi", "xi=0.2800", "next-order xi (v11 first-order estimate), background only"),
        row("(1) xi", "xi=0.2580", "lower bound"), row("(1) xi", "xi=0.3080", "upper (Gaussian variational) bound"),
        row("(1) xi", "xi=0.2979", "SM lambda instead of lambda=+0.01 in the DMC (audit: dxi = -0.002)"),
        row("(2) N_C^eff", "N_C inflaton +1/2 (Q fixed)"), row("(2) N_C^eff", "N_C 3 nu_R +0.70 (Q fixed)"),
        row("(2) N_C^eff", "N_C inflaton + 3 nu_R (Q fixed)"), row("(2) N_C^eff", "N_C inflaton + 3 nu_R (xi fixed)"),
        row("(3) A_s / plateau", "A_s +1 sigma (dln A_s = +0.014)"), row("(3) A_s / plateau", "A_s -1 sigma (dln A_s = -0.014)"),
        row("(3) A_s / plateau", "h/M_P = 6.116e-6 (v11 instant reheating, consistency of (722, hL_b))",
            "pipeline (722, 0.00437243) implies h/M_P = 6.054e-6"),
        row("(3) A_s / plateau", "weak visible decay h/M_P = 6.7467e-6 (N*=50.33, r=0.0042)"),
        row("(3) A_s / plateau", "plateau shape H_*/h: 0.986 -> 1 (alpha->0 attractor)"),
    ]
    for k in V:
        if k.startswith("plateau r ="):
            rows.append(row("(3) A_s / plateau", k))
    for k in ("hL_b x e^+0.5 at fixed mu_b (eta_inf / e^+0.5)", "hL_b x e^-0.5 at fixed mu_b (eta_inf / e^-0.5)",
              "hL_b x e^+1.0 at fixed mu_b (eta_inf / e^+1.0)", "hL_b x e^-1.0 at fixed mu_b (eta_inf / e^-1.0)"):
        rows.append(row("(4) hL_b (turn duration)", k, "pure hazard rescaling I -> I e^-dh (exact)"))
    rows.append(row("(6) zero-mode next order via xi", "xi=0.2830", "dxi = -0.017 (curvature -0.014, Higgs mass -0.003)"))
    k_top = [k for k in V if k.startswith("xi=") and k not in ("xi=0.2800", "xi=0.2830", "xi=0.2580", "xi=0.3080", "xi=0.2979")]
    for k in k_top:
        rows.append(row("(6) zero-mode next order via xi", k, "top term of m_Phi^2 with NNLO y_t(mu_b) (%.4f) instead of 0.4585"
                        % P["inputs"]["yt_NNLO_mu_b_172p6"]))
    kb = P["kappaE_backreaction"]
    kE = dict(item="(5) kappa_E", statement="Q is shape independent at leading order (unit-Jacobian canonical map), so kappa_E "
              "does not enter the zero-mode Hamiltonian; it enters the background only through the mean-field shape "
              "backreaction (Casimir Hessian + shape kinetic energy) during the turn",
              rows={k: dict(dmc={t: float("%.2e" % v["dmc"][t]) for t in T}, max_abs_dlna=float("%.2e" % v["max_abs_dlna"]))
                    for k, v in kb.items()},
              bound="|dm_c| <= 2.2e-4 GeV at <beta^2> = delta^2 (kappa_E = 19.32); 19.32 -> 18.0 changes m_c by < 1e-4 GeV")
    sl = P["slopes"]
    per = {k: {t: r4(V[k]["dmc"][t]) for t in T} for k in V if k.startswith(("r x e", "C_eff x e", "mu_b x e"))}
    per.update({k: {t: r4(V[k]["dmc"][t]) for t in T} for k in V if k.startswith("hL_b x e^")})
    plate = [k for k in V if k.startswith("plateau r =")]
    rr = {k: V[k]["r"] for k in plate}
    band = [k for k in plate if 0.00099 <= rr[k] <= 0.01001]
    vi = {t: [r4(min(V[k]["dmc"][t] for k in band)), r4(max(V[k]["dmc"][t] for k in band))] for t in T}
    low = [k for k in plate if rr[k] < 2e-4]
    tab = dict(baseline_C149_run02=base, items=rows, kappa_E=kE,
               vi_inflation_scale={"range_r_1e-3_to_1e-2": vi,
                                   "r_1e-4": {t: r4(V[low[0]]["dmc"][t]) for t in T} if low else None,
                                   "old_D116": "+-0.05 (MU slope -790)",
                                   "recommended": {t: "+-%.3f" % max(abs(vi[t][0]), abs(vi[t][1])) for t in T}},
               slope_update=dict(dlnI_dlam_b={t: round(sl[t]["dlnI_dlam_b"], 1) for t in T}, dlnI_dmt={t: round(sl[t]["dlnI_dmt"], 3)
                                                                                                      for t in T},
                                 old="dln gamma/dlambda = -790 (MU, D115)", per_efold=per),
               spline_validation=dict(leave_one_out_max_dmc=max(abs(x) for v in P["spline_leave_one_out"].values()
                                                                for x in v["dmc"].values()),
                                      direct=P["direct_validation"]))
    wj("C152_input_dmc.json", tab)
    mw = {k: dict(crit={t: r4(v["crit"][t]) for t in T}, dmc={t: r4(v["dmc"][t]) for t in T}, note=v["note"],
                  P_172p6=r4(v["P_172p6"])) for k, v in P["mc_w0"].items()}
    b3 = None
    if W0 is not None:                       # m_c with the B3 w0(m_t) (walls 1.1/1.2/1.4; failed wall selections dropped)
        import numpy as np
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq
        mts, I = P["baseline"]["mts"], P["baseline"]["I"]
        spI = CubicSpline(mts, np.log(I))
        cm = sorted(float(m) for m in W0["cols"])
        wv = {}
        for tag in ("wall1.1", "wall1.2", "wall1.4", "CAPlike_mean"):
            vals = []
            for m in cm:
                c_ = W0["cols"]["%.2f" % m]
                ws = [c_["B3_minvar"]["w0"][k] for k in ("wall1.1", "wall1.2", "wall1.4")
                      if abs(c_[k]["E_minus_Q2D"]) < 0.01]
                vals.append(float(np.mean(ws)) if tag == "CAPlike_mean" else
                            (c_["B3_minvar"]["w0"][tag] if abs(c_[tag]["E_minus_Q2D"]) < 0.01 else float(np.mean(ws))))
            wv[tag] = vals
        b3 = dict(mt_nodes=cm, w0=wv, crit={})
        for tag, vals in wv.items():
            wf = np.poly1d(np.polyfit(cm, vals, 1))
            cr = {}
            for t, Pt in (("P1=0.9", 0.9), ("P1=0.5", 0.5), ("P1=0.1", 0.1)):
                f = lambda z: math.log(wf(z)) - math.exp(spI(z)) - math.log(Pt)  # noqa: E731
                try:
                    cr[t] = round(brentq(f, mts[0], mts[-1]), 4)
                except ValueError:
                    cr[t] = None
            b3["crit"][tag] = dict(crit=cr, dmc={t: r4(cr[t] - base[t]) if cr[t] else None for t in T})
    wj("C152_mc_w0.json", dict(rule="P = w0 * exp(-I): critical m_t solves exp(-I) = P_target/w0; none if P_target >= w0",
                               table=mw, with_B3_w0_of_mt=b3))
    c = X157["central"]
    acc = dict(
        P1_0p5_critical=dict(round2_C144_D116=173.012, round2_R15_selfconsistent=173.016, audit_proposal="173.02",
                             round3_central_C149run02=round(base["P1=0.5"], 3),
                             cite="round 3: 172.95 (P1=0.5), 172.50 (0.9), 173.41 (0.1); round-2 173.01/173.02 superseded"),
        P1_172p60=dict(round2=0.856, round3=round(S149["P1"]["172.6"], 3)),
        error_composition=dict(
            rule="state explicitly whether (vi) is included; (vi) now P1-level dependent",
            round2_model_excl_vi={"P1=0.9": "+0.014/-0.038", "P1=0.5": "+0.018/-0.048"},
            round2_model_incl_old_vi={"P1=0.9": "+0.052/-0.063", "P1=0.5": "+0.053/-0.070"},
            round2_model_incl_new_vi={t: "+%.3f/-%.3f" % (math.hypot(a, float(tab["vi_inflation_scale"]["recommended"][t][2:])),
                                                          math.hypot(b, float(tab["vi_inflation_scale"]["recommended"][t][2:])))
                                      for t, a, b in (("P1=0.9", 0.014, 0.038), ("P1=0.5", 0.018, 0.048))},
            note="round-3 model components (T1 SM side +-0.086 pole, T2/T14b) are owned by other agents; only (vi) is updated here"),
        xi_c=dict(round2_D116={"172.2": "0.036+-0.003", "172.6": "0.072+-0.003", "173.0": "0.120+-0.001"},
                  round2_prescription_band_audit={"172.2": "0.023-0.041", "172.6": "0.066-0.075"},
                  round3_T6_C157={m: dict(merged=round(c[m]["merged"], 4), half_diff=round(c[m]["half_diff"], 4),
                                          spread_all=[round(x, 4) for x in c[m]["spread_all"]],
                                          u_sw5_lam_phi=round(c[m]["u_sw5_lam_phi"], 4)) for m in c},
                  quote="0.035/0.074/0.120/0.125 at 172.2/172.6/172.95/173.0 (not recomputed here)"),
        vi_inflation_scale=tab["vi_inflation_scale"],
        h_over_MP_consistency=row("(3)", "h/M_P = 6.116e-6 (v11 instant reheating, consistency of (722, hL_b))"),
        w0=dict(table="C152_mc_w0.json", B3=None if W0 is None else {m: dict(B3=v["B3_minvar"], Emin=v["Emin"],
                                                                              D111_type=v["D111_type_radial_gaussian"])
                                                                     for m, v in W0["cols"].items()}))
    wj("C152_accounting.json", acc)
    summ = dict(task="T4+T15+T18(Higgs) C152/D124", baseline=base, input_dmc="C152_input_dmc.json", mc_w0="C152_mc_w0.json",
                w0_B3="C152_w0_B3.json", accounting="C152_accounting.json", post="C152_post.json",
                vi=tab["vi_inflation_scale"], slope=tab["slope_update"]["dlnI_dlam_b"],
                sha256={f: hashlib.sha256(open(os.path.join(RES, f), "rb").read()).hexdigest()
                        for f in ("C152_post.json", "C152_input_dmc.json", "C152_mc_w0.json", "C152_accounting.json")})
    if W0 is not None:
        summ["w0_B3_values"] = {m: dict(w0_wall=v["B3_minvar"]["w0"], sigma=[v["B3_minvar"]["sigma_r"], v["B3_minvar"]["sigma_s"]],
                                        w0_Emin=v["Emin"]["w0"]) for m, v in W0["cols"].items()}
    wj("C152_summary.json", summ)
    print(json.dumps(dict(vi=tab["vi_inflation_scale"], slope=tab["slope_update"]["dlnI_dlam_b"],
                          err=acc["error_composition"]["round2_model_incl_new_vi"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
