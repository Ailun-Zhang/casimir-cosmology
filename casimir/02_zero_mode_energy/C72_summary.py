# PROVENANCE: 04_calculations/code/20261011_claude_followup/C72/C72_summary.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Collects C72 analysis + DMC into C72_summary.json (r band, coupled vs decoupled Higgs cost, fossil table). Paper: PRD Sec. IV C (fzm values), App. app:zeromodes.

C72 summary: collects C72_analysis_results.json and C72_dmc_results.json into C72_summary.json
(claim-by-claim comparison with Codex, the r band, coupled vs decoupled Higgs cost, fossil table).

    python C72_summary.py
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C72_common import RES, C_FREE, GAMMA  # noqa: E402

SIGMA_E_PLATEAU = 0.0587   # paper, A_s-calibrated plateau background (free field)


def main():
    A = json.loads((RES / "C72_analysis_results.json").read_text(encoding="utf-8"))
    cx = A["codex_claims"]
    B, D, M, F = A["B_numeric"], A["D_couplings_bounds"], A["DMC_summary"], A["E_fossil"]
    scan = {row["lam"]: row for row in B["EW_scan_g05_gY046"]}
    bm, rn = D["benchmark_g05"], D["running_mu_1_over_Lb"]
    upl = lambda blk, lam: [u for u in blk["upper"] if u["lam"] == lam][0]
    cmp = {
        "a_eps_SU2_dmc_vs_vanBaal": (M["eps_SU2_dmc"], M["se_SU2"], cx["eps_SU2"]),
        "a_eps_SU3_dmc_vs_Pavel": (M["eps_SU3_dmc"], M["se_SU3"], cx["eps_SU3"]),
        "validation_quartic4_dmc_vs_certified": (M["quartic4_dmc"], M["se_quartic4"], 2.1407),
        "b_gauss_SU2": (B["SU2_gauss"]["E_closed"], cx["gauss_SU2"]),
        "b_gauss_SU3": (B["SU3_gauss"]["E_closed"], cx["gauss_SU3"]),
        "c_EW_gauss_min_lam0.01": (scan[0.01]["E"], cx["ew_E"]),
        "c_abz_lam0.01": ((scan[0.01]["a"], scan[0.01]["b"], scan[0.01]["z"]), cx["ew_abz"]),
        "c_hess_log_lam0.01": (scan[0.01]["hess_log_eigs"], cx["ew_hess_log"]),
        "c_SU3_gauss_g05": (scan[0.01]["SU3_gauss"], cx["su3_gauss_g05"]),
        "c_Q_upper_lam0.01": (scan[0.01]["Q_upper"], cx["Q_upper"]),
        "c_upper_pct_lams": ([round(100 * scan[l]["r_upper"], 4) for l in (0.0, 0.001, 0.01, 0.05, 0.1)], cx["frac_pct"]),
        "d_D0_g05": (bm["D0"], cx["D0"]), "d_D0_pct": (100 * bm["r_D0"], cx["D0_pct"]),
        "d_SU3_pct": (100 * bm["r_SU3_ref"], cx["SU3_pct"]),
        "running_g2_g3": ((rn["g2"], rn["g3"]), (cx["run_g2"], cx["run_g3"])),
        "running_D0": (rn["D0"], cx["run_D0"]), "running_D0_pct": (100 * rn["r_D0"], cx["run_D0_pct"]),
        "kappaE_at_rD0": (F["ref_D0_g05_literature"]["kappa_E"], cx["kappaE_ref"]),
        "sumkappa_at_rD0": (F["ref_D0_g05_literature"]["sum_kappa"], cx["sumkappa_ref"]),
    }
    band = {}
    for lab, blk, key in (("g2=g3=0.5,gY=0.46", bm, "g05"), ("one-loop mu=1/L_b", rn, "running")):
        band[lab] = dict(
            couplings=(blk["g2"], blk["gY"], blk["g3"]),
            lower_operator_bound_r=blk["r_D0"],
            lower_with_dmc_eps_r=M["r_D0_g05_dmc"] if key == "g05" else M["r_D0_running_dmc"],
            exact_local_dmc_r={k.split("_", 1)[1]: v for k, v in M["r_exact_local"].items() if k.startswith(key)},
            upper_gauss_r={u["lam"]: u["r_upper"] for u in blk["upper"]},
            upper_gauss_EW_plus_PavelSU3_r={u["lam"]: u["r_upper_PavelSU3"] for u in blk["upper"]})
    higgs = {"decoupled_pure_quartic_lam0.01_energy": M["decoupled_quartic_higgs_lam0.01"],
             "decoupled_pure_quartic_NC_units": M["decoupled_quartic_higgs_lam0.01"] / GAMMA,
             "decoupled_pure_quartic_frac_C": M["decoupled_quartic_higgs_lam0.01"] / C_FREE,
             "coupled_U1_Higgs_increment_energy": M["EW_increment_g05_lam0.01"],
             "coupled_increment_NC_units": M["EW_increment_g05_lam0.01"] / GAMMA,
             "coupled_increment_frac_C": M["EW_increment_g05_lam0.01"] / C_FREE,
             "ratio_coupled_to_decoupled": M["EW_increment_g05_lam0.01"] / M["decoupled_quartic_higgs_lam0.01"],
             "decoupled_total_Q_D0_plus_quartic_r": (bm["D0"] + M["decoupled_quartic_higgs_lam0.01"]) / C_FREE}
    fossil = {}
    for k, row in F.items():
        if "sigma_E_natural" in row and "sigma_T2_natural" in row:
            fossil[k] = dict(r=row["r"], kappa_E=row["kappa_E"], kappa_T2=row["kappa_T2"], sum_kappa=row["sum_kappa"],
                             sigma_E_natural=row["sigma_E_natural"], sigma_E_floor=row["sigma_E_floor"],
                             sigma_T2_natural=row["sigma_T2_natural"], sigma_T2_floor=row["sigma_T2_floor"],
                             sigma_E_ratio=row["sigma_E_ratio_to_free"], sigma_T2_ratio=row["sigma_T2_ratio_to_free"],
                             sigma_E_plateau_scaled=SIGMA_E_PLATEAU * row["sigma_E_ratio_to_free"],
                             budget_NCgamma_1mr_over_2pi=row["NC_gamma_1mr_over_2pi"])
    scale = {"mu_1_over_Lb_GeV": D["mu_1_over_Lb_GeV"],
             "r_D0_mu_1_over_Lb": rn["r_D0"], "r_D0_mu_2pi_over_Lb": D["running_mu_2pi_over_Lb"]["r_D0"],
             "r_upper_lam0.01_mu_1_over_Lb": upl(rn, 0.01)["r_upper"],
             "r_upper_lam0.01_mu_2pi_over_Lb": D["running_mu_2pi_over_Lb"]["upper"][0]["r_upper"],
             "r_D0_self_consistent_C_eff": D["self_consistent_reference"]["r_D0"],
             "self_consistent_mu_GeV": D["self_consistent_reference"]["mu_GeV"]}
    out = dict(comparisons_ours_vs_codex=cmp, r_band=band, higgs_coupled_vs_decoupled=higgs, fossil=fossil,
               scale_sensitivity=scale, isospectrality=dict(numeric=A["C_isospectrality"],
                                                             exact=A["C_isospectrality_exact"]),
               symbolic=dict(A=A["A_symbolic"], B=A["B_symbolic"]))
    (RES / "C72_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in cmp.items():
        print(k, v)
    print(json.dumps(band, indent=0)); print(json.dumps(higgs, indent=0)); print(json.dumps(scale, indent=0))
    for k, v in fossil.items():
        print(f"{k:28s} r={v['r']:.5f} sE={v['sigma_E_natural']:.5f} (x{v['sigma_E_ratio']:.4f}, plateau {v['sigma_E_plateau_scaled']:.5f}) "
              f"sT2={v['sigma_T2_natural']:.6f} (x{v['sigma_T2_ratio']:.4f}) floorT2={v['sigma_T2_floor']:.6f} budget={v['budget_NCgamma_1mr_over_2pi']:.4f}")
    print("wrote", RES / "C72_summary.json")


if __name__ == "__main__":
    main()
