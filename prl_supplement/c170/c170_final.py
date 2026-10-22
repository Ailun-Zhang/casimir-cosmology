# -*- coding: utf-8 -*-
"""C170 final assembly: C170_results.json, checks.json, top-level CSV copies, timing summary.
Inputs: the per-step summaries written by c170_s0.py, c170_post.py (s1, lamsm, s2, s3), c170_s2.py (curves, kzero,
scalecheck, reg2dex) and c170_s1.py conv.  Usage: python -B c170_final.py
"""
import os
import sys
import math
import json
import glob
import shutil
import time
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

O = CC.OUT


def J(rel):
    p = os.path.join(O, rel)
    return CC.jload(p) if os.path.exists(p) else None


def chain_times(pattern):
    tot, n, mx = 0.0, 0, 0.0
    for fn in glob.glob(os.path.join(O, pattern)):
        if fn.endswith(".pruned.json"):
            continue
        for p in CC.jload(fn)["points"]:
            t = p.get("time_s") or 0.0
            tot += t
            n += 1
            mx = max(mx, t)
    return dict(n_points=n, cpu_s_sum_of_points=round(tot, 1), max_point_s=round(mx, 1),
                mean_point_s=round(tot / n, 1) if n else None)


def kappa_boundaries(s2):
    """kappa_thr: where lambda_c(300, 1/2) -> 0- (Gamma(kappa, lambda = 0) = ln2/300, interpolated in kappa, ln Gamma);
    kappa_min: smallest kappa with a narrow resonance at lambda = 0 (between the largest kappa without and the smallest
    kappa with a narrow resonance there)."""
    out = {}
    tgt = math.log(2) / 300
    for model in ("2D_Gdmcs", "2D_G"):
        rows = sorted([(v["kappa"], v) for k, v in s2.items() if v["model"] == model], key=lambda z: z[0])
        at0 = [(kap, (v["at"].get("0") or {})) for kap, v in rows]
        narrow0 = [(kap, a.get("Gamma")) for kap, a in at0 if a.get("status") in CC.GOOD]
        none0 = [(kap, a.get("status")) for kap, a in at0 if a and a.get("status") not in CC.GOOD]
        ent = dict(lambda0_status={("%.4f" % kap): dict(status=a.get("status"), Gamma=a.get("Gamma"),
                                                       eps_2D=a.get("eps") if a.get("status") in CC.GOOD else None,
                                                       U_top=a.get("U_top"), U_well=a.get("U_well"), S_wkb=a.get("S_wkb"),
                                                       eps_1D=a.get("eps_1D"), Gamma_1D=a.get("Gamma_1D"),
                                                       eps1D_minus_Utop=a.get("eps1D_minus_Utop"))
                                   for kap, a in at0 if a},
                   note="status 'flagged' at lambda = 0 (kappa = 0.35-0.425): no 2D root consistent with the adiabatic "
                        "level, and that level lies above the barrier top (eps1D_minus_Utop > 0): no narrow resonance")
        if narrow0 and none0:
            kmin_hi = min(k for k, _ in narrow0)
            below = [k for k, _ in none0 if k < kmin_hi]
            ent["kappa_min_bracket"] = [max(below) if below else None, kmin_hi]
        if len(narrow0) >= 2:
            ks = np.array([k for k, _ in narrow0])
            lg = np.log(np.array([g for _, g in narrow0]))
            for i in range(len(ks) - 1):
                if (lg[i] - math.log(tgt)) * (lg[i + 1] - math.log(tgt)) <= 0:
                    kt = ks[i] + (math.log(tgt) - lg[i]) * (ks[i + 1] - ks[i]) / (lg[i + 1] - lg[i])
                    ent["kappa_thr_N300_P05"] = dict(kappa=float(kt), bracket=[float(ks[i]), float(ks[i + 1])],
                                                     method="linear in ln Gamma(kappa, lambda = 0)")
            if "kappa_thr_N300_P05" not in ent and lg.min() < math.log(tgt):
                ent["kappa_thr_N300_P05"] = dict(kappa=None, note="Gamma(lambda=0) below ln2/300 at every kappa with a "
                                                                  "narrow resonance at lambda = 0: kappa_thr <= kappa_min")
        out[model] = ent
    return out


def main():
    t0 = time.time()
    s0 = J("S0/C170_S0.json")
    s1 = J("S1/C170_S1_summary.json")
    lsm = J("S1/C170_lambdaSM_running.json")
    conv = J("S1/convergence_boxA.json")
    s2 = J("S2/C170_S2_summary.json")
    s2c = J("S2/C170_S2_curves.json")
    s2k0 = J("S2/C170_S2_kappa0.json")
    s2sc = J("S2/C170_S2_scalecheck.json")
    s2reg = J("S2/C170_S2_reg2dex.json")
    s3 = J("S3/C170_S3_summary.json")
    boxes = J("S1/boxes.json")
    R = dict(meta=dict(
        task="static boxes A-E (mu_b, mu_b/10, mu_b/100, 10 mu_b, mu_b/1000); lambda_SM and N* per box; "
             "lambda_SM(mu) grid",
        run="C170", date="2026-10-19",
        code="c170/ (production modules copied unchanged from the code release v1.0.0: higgs/zmlib, "
             "higgs/04_cross_terms C149_direct4/5, C149_coulomb)",
        work_tree="<C170_OUT>/work",
        environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                         processor=platform.processor(), cpu_count=os.cpu_count()),
        units="L = 1: energies eps, widths Gamma in 1/L; lambda, g2, gY, y_t, g3 MS-bar at mu = 1/L; mu in GeV",
        setting="static flat three-torus of fixed edge L (Minkowski time, no gravity, no turning point), AAA fermions, "
                "SM couplings at mu = 1/L, held for T = N L: P = exp(-Gamma N)",
        model="run05 production model (PRL centre): 2D (r, |w|) model with conditional-Gaussian SU(2)xU(1) zero modes, "
              "DMC shape correction, one-loop MC finite-volume matching (mu = 1/L), complete cross-term set; "
              "numerics FAST (N, Smax, h, Rout) = (24, 12, 0.004, 16), window (0.02, 0.0015), box check (28, 14)",
        definitions=dict(
            lambda_c="Gamma(lambda_c) N = -ln P at fixed (g2, gY, y_t, g3) of the box",
            x="|lambda|^(-1/2)",
            status_point="ok / ok_broad = narrow resonance (see README); above_top, no_level, no_barrier = none",
            status_threshold="interpolated (inside the computed narrow range), extrapolated (linear in x beyond the "
                             "range, extrap_dx given), beyond_narrow (target width larger than every narrow width: "
                             "the level reaches the barrier top first; not a valid threshold)",
            N_eff="static duration of box A that reproduces the run05 hazard threshold: N_eff = -ln P / Gamma_A("
                  "lambda_c^hazard)",
            Nstar="ln 2 / Gamma(lambda_SM): duration after which the SM zero mode of that box survives with P = 1/2")))
    R["S0"] = s0
    # ---------------------------------------------------------------- S1
    if s1:
        S1 = dict(definition=s1["definition"], boxes={})
        lamc_mu = []
        for b, v in s1["boxes"].items():
            S1["boxes"][b] = {k: v[k] for k in ("mu_GeV", "mu_over_mu_b", "couplings", "n_points", "n_good",
                                                "narrow_range", "cease", "thresholds", "fits", "lam_SM",
                                                "max_box_rel_gamma", "max_abs_ln_ratio_2D_1D", "duplicate_start_point")}
            S1["boxes"][b]["table"] = [{k: t[k] for k in ("lam", "x", "eps", "Gamma", "status", "flags", "U_top", "r_top",
                                                          "U_well", "r_well", "eps_minus_Utop", "S_wkb", "eps_1D",
                                                          "Gamma_1D", "box_rel_gamma", "window", "retried", "time_s")}
                                       for t in v["table"]]
            if "N_eff" in v:
                S1["N_eff_boxA"] = v["N_eff"]
            for kP, d in v["thresholds"].items():
                for N, t in d.items():
                    if t:
                        lamc_mu.append(dict(box=b, mu_GeV=v["mu_GeV"], P=kP, N=float(N), lam_c=t["lam_c"],
                                            status=t["status"]))
        S1["lambda_c_vs_mu"] = lamc_mu
        # coupling-mismatch check: box A (172.6 couplings) at lambda_SM(m_t) vs the archived u = 0 widths at their own
        # couplings (m_t = 172.2, 173.0)
        arch = CC.jload(os.path.join(CC.WORK, "archive", "r3", "holonomy_cross", "run05", "shards", "X_main_u0.0000.json"))
        mm = {}
        for mt in ("172.2", "173.0"):
            x = [c for c in arch["cols"] if abs(c["mt"] - float(mt)) < 1e-6][0]
            gA = s1["boxes"]["A"]["lam_SM"][mt].get("Gamma_direct")
            if gA:
                mm[mt] = dict(lam=x["lam"], Gamma_archived_own_couplings=x["gamma"], Gamma_boxA_couplings=gA,
                              ratio_boxA_over_archived=gA / x["gamma"],
                              d_yt=s1["boxes"]["A"]["couplings"]["yt"] - x["yt"])
        S1["coupling_mismatch_check_boxA"] = mm
        S1["convergence_boxA"] = conv
        R["S1"] = S1
    R["lambda_SM_running"] = lsm
    R["boxes_input"] = boxes
    # ---------------------------------------------------------------- S2
    if s2:
        S2 = dict(decision_DMC=(
            "The DMC shape correction is KEPT in transported form: the zero-mode Hamiltonian (gauge kinetic + magnetic + "
            "mass terms + body-frame angular term + hypercharge oscillator) with couplings kappa*(g2, gY) is unitarily "
            "equivalent to kappa^(2/3) times the kappa = 1 Hamiltonian at r' = kappa^(1/3) r (x -> kappa^(-1/3) x), so "
            "the exact-minus-Gaussian energy obeys dE_DMC(kappa; r) = kappa^(2/3) dE_DMC(1; kappa^(1/3) r) exactly; "
            "no new DMC run is needed and the model reduces to the production 2D_ex at kappa = 1 ('2D_Gdmcs', main). "
            "Keeping dE_DMC unscaled would attach the kappa = 1 correction to a different Hamiltonian (it would even "
            "survive at kappa = 0). Dropping it ('2D_G') gives the strict variational bracket. The exact scaling was "
            "verified numerically without the matching (scalecheck)."),
            models={"2D_Gdmcs": "main (transported DMC shape; = 2D_ex at kappa = 1)", "2D_G": "DMC shape dropped"},
            kappa_scan={k: {kk: v[kk] for kk in ("model", "kappa", "threshold_N300_P05", "at", "n_points", "n_good",
                                                  "narrow_lam_range", "table")} for k, v in s2.items()},
            kappa0=s2k0, curves=s2c, scalecheck=s2sc, regression_2D_ex=s2reg, boundaries=kappa_boundaries(s2))
        # kappa = 1 comparison with the full model (S1 box A)
        if s1:
            pts = s1["boxes"]["A"]["table"]
            g = sorted([t for t in pts if t["status"] in CC.GOOD], key=lambda t: t["x"])
            xs = np.array([t["x"] for t in g])
            lg = np.log([t["Gamma"] for t in g])
            sp = CubicSpline(xs, lg)
            full_g012 = float(math.exp(sp(CC.x_of_lam(-0.012))))
            full_thr = s1["boxes"]["A"]["thresholds"]["P=0.5"]["300"]["lam_c"]
            cmp = dict(final_model_boxA=dict(lam_c_N300_P05=full_thr, Gamma_lam_m0p012=full_g012))
            for m in ("2D_Gdmcs", "2D_G"):
                kk = "S2_%s_k1.0000" % m
                if kk in s2:
                    t = s2[kk]["threshold_N300_P05"]
                    gg = (s2[kk]["at"].get("-0.012") or {}).get("Gamma")
                    cmp[m + "_kappa1"] = dict(lam_c_N300_P05=t["lam_c"] if t else None, Gamma_lam_m0p012=gg,
                                              dlam_c_vs_final=(t["lam_c"] - full_thr) if t else None,
                                              Gamma_ratio_vs_final=(gg / full_g012) if gg else None)
            S2["kappa1_vs_final_model"] = cmp
        R["S2"] = S2
    # ---------------------------------------------------------------- S3
    R["S3"] = s3
    # ---------------------------------------------------------------- timings
    R["timings"] = dict(S1=chain_times("S1/chains/*.json"), S2=chain_times("S2/chains/*.json"),
                        S3=chain_times("S3/chains/*.json"), S0_s=(s0 or {}).get("time_s"),
                        note="cpu_s_sum_of_points = sum of single-process wall times per point (chains ran in parallel, "
                             "6-10 workers); per-point log lines with times are in logs/*.log")
    CC.wjson(os.path.join(O, "C170_results.json"), R)
    # ---------------------------------------------------------------- checks
    chk = []

    def add(name, what, value, reference, tol, passed, src):
        chk.append(dict(name=name, what=what, value=value, reference=reference, tolerance=tol, passed=bool(passed),
                        computed_from=src))
    if s0:
        for k, v in s0["regression"].items():
            p = v["prod_seed"]
            add("S0_%s_gamma_prod_seed" % k, "Gamma of the lambda-override path (archived node couplings, production "
                "seed) vs archived run05 shard", abs(p["rel_gamma"]), 0.0, "rel <= 1e-8", abs(p["rel_gamma"]) <= 1e-8,
                "S0/C170_S0.json#regression/%s/prod_seed/rel_gamma" % k)
            add("S0_%s_eps_prod_seed" % k, "Re E vs archived", abs(p["d_eps"]), 0.0, "abs <= 1e-10", abs(p["d_eps"]) <= 1e-10,
                "S0/C170_S0.json#regression/%s/prod_seed/d_eps" % k)
            if "generic_seed" in v:
                gs = v["generic_seed"]
                add("S0_%s_gamma_generic_seed" % k, "Gamma with this work's generic seed (1D adiabatic root) vs archived",
                    abs(gs["rel_gamma"]), 0.0, "rel <= 1e-8", abs(gs["rel_gamma"]) <= 1e-8,
                    "S0/C170_S0.json#regression/%s/generic_seed/rel_gamma" % k)
        mx = max(abs(d["diff"]) for m in s0["couplings_vs_archive_u0"].values() for d in m.values())
        add("S0_couplings_sm_provider", "sm_provider couplings at mu_b vs archived node couplings (g2, gY, y_t, lam, g3)",
            mx, 0.0, "abs <= 1e-15", mx <= 1e-15, "S0/C170_S0.json#couplings_vs_archive_u0")
        ref02 = {"P1=0.9": 0.00030, "P1=0.5": 0.00036, "P1=0.1": 0.00042}
        for kP, r in s0["static_threshold_archived_u0"]["rows"].items():
            add("S0_static_minus_hazard_%s" % kP, "static threshold of the archived u = 0 widths (N = eta_inf) minus "
                "the run05 hazard threshold (PRL: shallower by 0.0003-0.0004; run02 values 0.00030/0.00036/0.00042)",
                r["shallower_by"], ref02[kP], "abs <= 3e-5", abs(r["shallower_by"] - ref02[kP]) <= 3e-5,
                "S0/C170_S0.json#static_threshold_archived_u0/rows/%s/shallower_by" % kP)
    if s1:
        A = s1["boxes"]["A"]
        g = A["lam_SM"]["172.6"]["Gamma_direct"]
        add("S1_boxA_Gamma_lamSM_vs_archive", "box A at lambda_SM(172.6) (continuation seed) vs archived run05 u = 0 width",
            abs(g / 7.147016111909779e-4 - 1), 0.0, "rel <= 1e-8", abs(g / 7.147016111909779e-4 - 1) <= 1e-8,
            "S1/C170_S1_summary.json#boxes/A/lam_SM/172.6/Gamma_direct")
        mb = max(v["max_box_rel_gamma"] for v in s1["boxes"].values())
        add("S1_box_check", "max relative change of Gamma under the (N, Smax) = (28, 14) box check, all narrow points",
            mb, 0.0, "<= 1e-5", mb <= 1e-5, "S1/C170_S1_summary.json#boxes/*/max_box_rel_gamma")
        mr = max(v["max_abs_ln_ratio_2D_1D"] for v in s1["boxes"].values())
        add("S1_2D_vs_1D_adiabatic", "max |ln(Gamma_2D/Gamma_1D adiabatic)| over narrow points (report: 2D exceeds the "
            "adiabatic 1D width by 1.0-1.8% in the final model)", mr, 0.018, "<= 0.05", mr <= 0.05,
            "S1/C170_S1_summary.json#boxes/*/max_abs_ln_ratio_2D_1D")
        dups = [abs(d["rel_gamma"]) for v in s1["boxes"].values() for d in v["duplicate_start_point"]]
        add("S1_start_point_reproducible", "start point computed twice (up and down chains)", max(dups) if dups else None,
            0.0, "rel <= 1e-10", (max(dups) if dups else 0) <= 1e-10, "S1/C170_S1_summary.json#boxes/*/duplicate_start_point")
        ref = {"P=0.9": 0.77, "P=0.5": 0.82, "P=0.1": 0.87}
        for kP, r in (s1["boxes"]["A"].get("N_eff") or {}).items():
            add("S1_N_eff_over_eta_%s" % kP, "N_eff/eta_inf of box A at the run05 hazard threshold vs report "
                "I/(Gamma(0) eta_inf) (D128, earlier model)", r["N_eff_over_eta"], ref[kP], "abs <= 0.03",
                abs(r["N_eff_over_eta"] - ref[kP]) <= 0.03, "S1/C170_S1_summary.json#boxes/A/N_eff/%s" % kP)
        if s0 and R["S1"].get("coupling_mismatch_check_boxA"):
            mmc = R["S1"]["coupling_mismatch_check_boxA"]
            sl_hi = math.log(mmc["173.0"]["ratio_boxA_over_archived"]) / 0.4      # d ln Gamma / d m_t at fixed lambda
            sl_lo = math.log(mmc["172.2"]["ratio_boxA_over_archived"]) / -0.4     # (box couplings minus own couplings)
            for kP in ("P=0.9", "P=0.5", "P=0.1"):
                k0 = kP.replace("P=", "P1=")
                rr = s0["static_threshold_archived_u0"]["rows"][k0]
                a = rr["I_over_Gamma0_eta"]
                dm = rr["mc_hazard"] - 172.6
                pred = a / math.exp((sl_hi if dm > 0 else sl_lo) * dm)
                b = s1["boxes"]["A"]["N_eff"][kP]["N_eff_over_eta"]
                add("S1_N_eff_vs_archived_u0_%s" % kP, "N_eff/eta from box A (fixed 172.6 couplings) vs the archived u = 0 "
                    "value I/(Gamma0 eta) at the critical m_t (own couplings) corrected by the measured coupling effect "
                    "Gamma_A/Gamma_own = exp(s (m_c - 172.6)), s from the 172.2/173.0 columns", b,
                    dict(archived=a, corrected=pred, m_c=rr["mc_hazard"]), "abs <= 0.005", abs(pred - b) <= 0.005,
                    "C170_results.json#S1/N_eff_boxA, #S1/coupling_mismatch_check_boxA; S0 static_threshold_archived_u0")
        if conv:
            for name in ("BASE", "Rout14", "N32_S16"):
                rr = [abs(r["rel_gamma"]) for r in conv["rows"] if r["cfg"] == name and r.get("rel_gamma") is not None]
                if rr:
                    add("S1_convergence_%s" % name, "max relative change of Gamma, box A points x = 6..18, FAST -> %s"
                        % name, max(rr), 0.0, "<= 1e-3", max(rr) <= 1e-3, "S1/convergence_boxA.json")
    if lsm:
        mx = max(abs(v["diff"]) for v in lsm["check_vs_boxes"].values())
        add("lambdaSM_grid_vs_coup_point", "lambda_SM(mu) log grid (linear interpolation in ln mu) vs coup_point at the "
            "box scales", mx, 0.0, "abs <= 1e-7", mx <= 1e-7, "S1/C170_lambdaSM_running.json#check_vs_boxes")
    if s2reg:
        add("S2_2D_ex_path_vs_archive", "no-cross-term path (2D_ex, box A couplings, lambda_SM) vs archived T14b run02 root",
            abs(s2reg["rel_gamma"]), 0.0, "rel <= 1e-8", abs(s2reg["rel_gamma"]) <= 1e-8, "S2/C170_S2_reg2dex.json")
    if s2sc:
        mx = max(abs(r["rel_gamma"]) for r in s2sc["rows"])
        add("S2_exact_zero_mode_scaling", "without the matching, Gamma(kappa, kappa^2 lam) = kappa^(2/3) Gamma(1, lam) "
            "(kappa = 0.5, 1.5; grids not scaled)", mx, 0.0, "rel <= 1e-4", mx <= 1e-4, "S2/C170_S2_scalecheck.json")
    if s2k0:
        npos = sum(v["n_positive_slope"] for k, v in s2k0.items() if isinstance(v, dict) and "n_positive_slope" in v)
        add("S2_kappa0_no_barrier", "kappa = 0: number of grid points with dU/dr > 0 for lambda in [-0.04, -1e-4], "
            "r in (0.02, min(16, r_series)]", npos, 0, "== 0", npos == 0, "S2/C170_S2_kappa0.json")
    if s2c:
        red = [s2c["slopes"]["1.00"][r]["reduction"] for r in ("1.0", "2.0", "3.0", "4.0", "5.0")]
        add("S2_kappa1_slope_reduction", "kappa = 1 (box A couplings, lambda = -0.012): local slope reduction of the zero-"
            "point energy by dV_MC at r = 1-5 (report: 10-33%)", [min(red), max(red)], [0.10, 0.33], "inside +-0.03",
            min(red) >= 0.07 and max(red) <= 0.36, "S2/C170_S2_curves.json#slopes/1.00")
    CC.wjson(os.path.join(O, "checks.json"), dict(package="C170", n_checks=len(chk),
                                                 n_pass=sum(1 for c in chk if c["passed"]), checks=chk))
    # ---------------------------------------------------------------- top-level CSV copies
    for src, dst in (("S1/C170_S1_gamma_lambda.csv", "C170_gamma_lambda_boxes.csv"),
                     ("S1/C170_S1_lambda_c_N.csv", "C170_lambda_c_N_boxes.csv"),
                     ("S1/C170_S1_lambdaSM_Nstar.csv", "C170_lambdaSM_Nstar_boxes.csv"),
                     ("S1/C170_lambdaSM_running.csv", "C170_lambdaSM_running.csv"),
                     ("S2/C170_S2_lambda_c_kappa.csv", "C170_lambda_c_kappa.csv"),
                     ("S2/C170_S2_potential_lam-0.012.csv", "C170_potential_kappa_lam-0.012.csv"),
                     ("S3/C170_S3_coupling_scan.csv", "C170_coupling_scan_lambda_c.csv")):
        if os.path.exists(os.path.join(O, src)):
            shutil.copy2(os.path.join(O, src), os.path.join(O, dst))
    print("checks: %d/%d passed" % (sum(1 for c in chk if c["passed"]), len(chk)))
    for c in chk:
        print("  %-5s %-40s %s" % ("PASS" if c["passed"] else "FAIL", c["name"], c["value"]))
    print("time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
