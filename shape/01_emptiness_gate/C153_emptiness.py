# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_emptiness.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Emptiness-gate numbers with v12 inputs (D125 part C): quanta budget sum|n| < 0.26660 (1-xi) N_C (4.76 at xi = 0.30, 5.05 at 0.258, 4.90 at 0.28, 6.80 free; budget = 5 at xi = 0.2649), reheating excess 10^94.04 (j = 1) ... 10^96.48 (j = 85), T_max = 7.264/7.177, 3-dof budget crossing 6.91/6.82; --check: direct test of the de Sitter scaling with C56 (4 modes). Paper: PRD Sec. VI A-C, App. app:gate (tab:s-budget).

C153_emptiness (D125 part C, T9): emptiness-gate numbers with v12 inputs.
1. quanta budget  Sum|n - theta| < N_C^eff gamma/(2 pi) = 0.26660 (1 - xi) N_C,  N_C = 25.515 (SM, AAA), gamma = 16.5323/pi^2;
2. reheating excess R = rho_gamma/|rho_C| of the time-symmetric past (stored C55 slices, free Casimir) x 1/(1 - xi);
   h-calibration sensitivity R ~ (h/M_P)^2;
3. latest natural preparation T: ln(2 pi/hL_b) and the 3-dof budget crossing from the stored C56 SM_AAA occupations,
   using the exact de Sitter scaling N_n(T, hL_b) = N_n(T + ln(hL_b0/hL_b), hL_b0) (modes are deep inside the Hubble
   radius at the turn, k/(a_b h) >= 1300); a direct check of the scaling is run with --check (C56 copy, 4 modes).
    python C153_emptiness.py [--check]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, math, hashlib, argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_hardening/run01/emptiness")
C56R = WP.wfile("r5/C56_preparation_gate/C56_SM_AAA.json")
C55R = WP.wdir("r5/C55_window_gate")
GAMMA = 16.532315959761668 / math.pi ** 2
NC_SM = 25.515
XIS = {"0.299897 (v11 DMC, lambda=+0.01)": 0.299897, "0.298 (SM lambda, audit est.)": 0.298,
       "0.2878 (g2=g3=0.5,gY=0.46)": 0.2878, "0.2829 (v11 next order, y_t=0.4585)": 0.2999 - 0.014 - 0.003,
       "0.2826 (next order, y_t=0.428)": None, "0.28 (v11 'xi~0.28')": 0.28, "0.3083 (Gaussian upper bound)": 0.3083,
       "0.258 (pure-gauge lower bound)": 0.258, "0 (free fields)": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "gamma_over_2pi": GAMMA / (2 * math.pi), "N_C_SM": NC_SM}
    # y_t = 0.428 in the Higgs zero-mode mass (v11 App. zero modes): top term 0.0805 at y_t = 0.4585
    ctop = 0.0805 / 0.4585 ** 2
    m2_old, m2_new = -0.1740 - 0.0135 + ctop * 0.4585 ** 2, -0.1740 - 0.0135 + ctop * 0.428 ** 2
    dxi_h_new = -0.003 * m2_new / m2_old
    XIS["0.2826 (next order, y_t=0.428)"] = 0.2999 - 0.014 + dxi_h_new
    res["higgs_mass_term"] = {"m2L2_y0.4585": m2_old, "m2L2_y0.428": m2_new, "dxi_higgs_y0.428": dxi_h_new}
    # 1. quanta budget
    B0 = GAMMA / (2 * math.pi) * NC_SM
    rows = {}
    for lab, xi in XIS.items():
        B = B0 * (1 - xi)
        rows[lab] = {"xi": xi, "N_C_eff": NC_SM * (1 - xi), "budget_sum_n": B, "max_bosonic_|n|=1_quanta": math.floor(B - 1e-12),
                     "max_AAA_fermion_quanta": math.floor(B / (math.sqrt(3) / 2) - 1e-12), "fermion_budget": B / (math.sqrt(3) / 2),
                     "single_|n|=5_quantum_forbids": bool(B < 5.0), "single_|n|=sqrt26_forbids": bool(B < math.sqrt(26))}
    res["quanta_budget"] = rows
    res["xi_where_budget_equals"] = {"5": 1 - 5 / B0, "sqrt26=5.099": 1 - math.sqrt(26) / B0,
                                     "fermion_6": 1 - 6 * math.sqrt(3) / 2 / B0, "fermion_5": 1 - 5 * math.sqrt(3) / 2 / B0}
    # 2. reheating excess (stored C55, free Casimir)
    c55 = {}
    for fn in ("C55_window_gate.json", "C55_window_gate_refined.json"):
        p = C55R / fn
        if p.exists():
            j = json.loads(p.read_text())
            c55[fn] = {"h_over_Mp": j["meta"]["h_over_Mp"], "nq": j["meta"].get("nq"),
                       "log10R_free": {str(s["turning_point_index_before_climb"]): s["log10_R"] for s in j["slices"]}}
    res["C55_stored"] = c55
    ref = c55.get("C55_window_gate_refined.json") or c55.get("C55_window_gate.json")
    rx = {}
    for lab, xi in (("xi0.30", 0.299897), ("xi0.28", 0.28), ("xi0.258", 0.258)):
        sh = -math.log10(1 - xi)
        rx[lab] = {"shift_log10": sh, "log10R": {j: v + sh for j, v in ref["log10R_free"].items()}}
    rx["h_sensitivity"] = {"dlog10R_per_1pct_h": 2 * math.log10(1.01),
                           "H_eff_instead_of_h(0.98949)": 2 * math.log10(0.98949),
                           "instant_reheating_h_6.116e-6_vs_C55_6.7467e-6": 2 * math.log10(6.116e-6 / ref["h_over_Mp"])}
    rx["L0_sensitivity"] = {"note": "fixed hL_b; R ~ e^{4 n_end}: 0.95->2 d_LSS raises log10R by 4 log10(2/0.95)",
                            "dlog10R_L0_0.95_to_2": 4 * math.log10(2 / 0.95)}
    res["reheating_excess"] = rx
    # 3. preparation window
    c56 = json.loads(C56R.read_text())
    hLb0, Bst = c56["hLb"], c56["budget_sum_n_quanta"]
    Ts = np.array([r["T"] for r in c56["rows"]])
    tw = {}
    for lab, hl in (("hLb0.0044", 0.0044), ("hLb0.0048", 0.0048), ("hLb0.00483(v11 free)", hLb0), ("H_eff_Lb0.0043724", 0.00437243)):
        tw[lab] = {"T_max_ln(2pi/hLb)": math.log(2 * math.pi / hl), "budget_crossing_3dof": {}}
        for st in ("ground", "conformal"):
            S = np.array([r["ratio_total_%s_to_budget_per_dof" % st] for r in c56["rows"]]) * Bst
            m = (Ts >= 5.0)
            f = PchipInterpolator(Ts[m], np.log(S[m]))
            for xl, xi in (("xi0.30", 0.299897), ("xi0.28", 0.28), ("xi0.258", 0.258), ("free", 0.0)):
                target = B0 * (1 - xi) / 3.0
                Tc = brentq(lambda t: float(f(t)) - math.log(target), 5.5, 8.0)
                tw[lab]["budget_crossing_3dof"]["%s_%s" % (st, xl)] = Tc + math.log(hLb0 / hl)
    tw["growth_rate_dlnS_dT_7.0_7.25"] = math.log(c56["rows"][5]["ratio_total_ground_to_budget_per_dof"] /
                                                  c56["rows"][4]["ratio_total_ground_to_budget_per_dof"]) / 0.25
    res["preparation_window"] = tw
    if a.check:
        sys.path.insert(0, str(HERE))
        import C56_preparation_time_gate as C56                 # copy (PROVENANCE line), occupation() only
        chk = []
        for n_ in (1.0, 2.0):
            for T in (7.0,):
                k_new = 2 * math.pi / 0.0044 * n_
                T0 = T + math.log(0.0044 / hLb0)                 # same a_i/k = e^T hL_b/(2 pi n) at hL_b0
                k_old = 2 * math.pi / hLb0 * n_
                o_new = C56.occupation((k_new, T, "ground"))
                o_old = C56.occupation((k_old, T0, "ground"))
                chk.append({"n": n_, "T_new": T, "T_old": T0, "N_new(hLb0.0044)": o_new[0], "N_old(hLb0)": o_old[0],
                            "rel_diff": o_new[0] / o_old[0] - 1, "werr": [o_new[1], o_old[1]]})
                print("CHECK", chk[-1], flush=True)
        res["scaling_check"] = chk
    (OUT / "C153_emptiness.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for lab, r in rows.items():
        print("BUDGET %-36s N_C^eff=%.3f  sum|n|<%.3f  fermion<%.3f  |n|=5 forbids:%s" % (lab, r["N_C_eff"], r["budget_sum_n"],
                                                                                    r["fermion_budget"], r["single_|n|=5_quantum_forbids"]))
    print("XI_AT", res["xi_where_budget_equals"], "HIGGS", res["higgs_mass_term"])
    print("C55", json.dumps(c55))
    print("REHEAT", json.dumps(rx))
    print("PREP", json.dumps(tw, indent=0))


if __name__ == "__main__":
    main()
